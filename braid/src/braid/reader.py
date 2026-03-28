"""
BRAID Shard Reader class.

Provides chunk-wise access to sharded Arrow files with CSV coordinate indices.
Supports local files and GCS URIs (gs://bucket/path) transparently.

GCS access uses google-cloud-storage (install with ``pip install braid[gcs]``)
rather than PyArrow's native ``pyarrow.fs.GcsFileSystem``, which suffers from
CURL error 81 ("Socket not ready for send/recv") on Google Cloud Run.
See docs/GoogleCloudRunIssues.md for details.
"""

from collections import namedtuple
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import csv as csv_mod
import io
import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.ipc as ipc
from .decompressor import DVIDDecompressor
from .exceptions import BraidError, ChunkNotFoundError, DecompressionError, InvalidShardFormatError, InvalidCoordinateError

# Per-chunk location within an Arrow IPC shard file.
ChunkLocation = namedtuple("ChunkLocation", ["offset", "size", "batch_idx"])


def _parse_gs_uri(uri: str):
    """Split gs://bucket/path into (bucket, path)."""
    rest = uri[len("gs://"):]
    bucket, _, path = rest.partition("/")
    return bucket, path.rstrip("/")


def _read_gcs_range(uri: str, start: int, size: int) -> bytes:
    """Read a byte range from a GCS URI using google-cloud-storage."""
    try:
        from google.cloud import storage
    except ImportError:
        raise BraidError(
            "google-cloud-storage is required for GCS access. "
            "Install with: pip install braid[gcs]"
        )
    bucket_name, blob_path = _parse_gs_uri(uri)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    return blob.download_as_bytes(start=start, end=start + size - 1)


def _read_bytes(path: str) -> bytes:
    """Read all bytes from a local path or GCS URI.

    For local paths, uses standard file I/O.
    For gs:// URIs, downloads via google-cloud-storage.
    """
    if path.startswith("gs://"):
        return _read_gcs_bytes(path)
    local = Path(path).resolve()
    if not local.exists():
        raise BraidError(f"File not found: {path}")
    return local.read_bytes()


def _read_gcs_bytes(uri: str) -> bytes:
    """Download bytes from a gs:// URI using google-cloud-storage."""
    try:
        from google.cloud import storage
    except ImportError:
        raise BraidError(
            "google-cloud-storage is required for GCS access. "
            "Install with: pip install braid[gcs]"
        )
    rest = uri[len("gs://"):]
    bucket_name, _, blob_path = rest.partition("/")
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    if not blob.exists():
        raise BraidError(f"GCS file not found: {uri}")
    return blob.download_as_bytes()


def _parse_csv_index(data: bytes):
    """Parse a CSV chunk index, detecting old or new format.

    New format (from DVID export-shards with batch_size support):
        # schema_size=688
        x,y,z,offset,size,batch_idx
        1,1,0,688,792,0

    Old format (legacy):
        x,y,z,rec
        1,1,0,0

    Returns:
        (chunk_index, schema_size) where chunk_index maps
        (x, y, z) -> ChunkLocation(offset, size, batch_idx).
        For old-format CSVs, offset and size are -1 (unknown)
        and batch_idx stores the record index; schema_size is 0.
    """
    text = data.decode("utf-8")

    # Parse optional schema_size comment header
    schema_size = 0
    for line in text.splitlines():
        if line.startswith("# schema_size="):
            schema_size = int(line.split("=", 1)[1])
            break

    # Parse CSV rows (skip comment lines)
    clean_lines = (l for l in io.StringIO(text) if not l.startswith("#"))
    reader = csv_mod.DictReader(clean_lines)
    columns = set(reader.fieldnames or [])

    new_cols = {"x", "y", "z", "offset", "size", "batch_idx"}
    old_cols = {"x", "y", "z", "rec"}
    new_format = new_cols.issubset(columns)
    old_format = old_cols.issubset(columns)

    if not new_format and not old_format:
        raise InvalidShardFormatError(
            f"CSV missing required columns. "
            f"Expected {new_cols} (new format) or {old_cols} (old format), "
            f"found {columns}"
        )

    chunk_index = {}
    for row in reader:
        coord = (int(row["x"]), int(row["y"]), int(row["z"]))
        if new_format:
            chunk_index[coord] = ChunkLocation(
                offset=int(row["offset"]),
                size=int(row["size"]),
                batch_idx=int(row["batch_idx"]),
            )
        else:
            chunk_index[coord] = ChunkLocation(
                offset=-1, size=-1, batch_idx=int(row["rec"]),
            )
    return chunk_index, schema_size


class LabelType(Enum):
    """Enumeration for label types available in shard files."""
    LABELS = "labels"           # Agglomerated label IDs
    SUPERVOXELS = "supervoxels" # Original supervoxel IDs


class ShardReader:
    """
    BRAID shard reader for Arrow files with CSV coordinate indices.

    Reads Arrow IPC shard files and CSV chunk indices from local disk or
    GCS (gs:// URIs).  GCS access uses google-cloud-storage for reliability
    on Cloud Run (see docs/GoogleCloudRunIssues.md).

    Example:
        >>> # Local files
        >>> reader = ShardReader("shard_0_0_0.arrow", "shard_0_0_0.csv")

        >>> # GCS files (downloaded via google-cloud-storage)
        >>> reader = ShardReader(
        ...     "gs://bucket/shards/s0/0_0_0.arrow",
        ...     "gs://bucket/shards/s0/0_0_0.csv",
        ... )

        >>> chunk_data = reader.read_chunk(64, 128, 0, label_type=LabelType.LABELS)
        >>> print(chunk_data.shape)  # (64, 64, 64)
    """

    # Expected Arrow schema for shard files (compatible with DVID export format)
    EXPECTED_SCHEMA = pa.schema([
        pa.field('chunk_x', pa.int32(), nullable=False),
        pa.field('chunk_y', pa.int32(), nullable=False),
        pa.field('chunk_z', pa.int32(), nullable=False),
        pa.field('labels', pa.list_(pa.uint64()), nullable=False),
        pa.field('supervoxels', pa.list_(pa.uint64()), nullable=False),
        pa.field('dvid_compressed_block', pa.binary(), nullable=False),
        pa.field('uncompressed_size', pa.uint32(), nullable=False)
    ])

    def __init__(self, arrow_path: Union[str, Path], csv_path: Union[str, Path]):
        """
        Initialize BRAID shard reader.

        Args:
            arrow_path: Path or URI to the Arrow IPC shard file (local, gs://, s3://)
            csv_path: Path or URI to the CSV chunk index file

        Raises:
            BraidError: If files cannot be loaded or have invalid format
        """
        self.arrow_path = str(arrow_path)
        self.csv_path = str(csv_path)

        # Load data during initialization (existence checked during read)
        self._table = self._load_arrow_data()
        self._chunk_index, self._schema_size = self._load_csv_index()
        self._decompressor = DVIDDecompressor()

        # Build row index from the table's own coordinate columns.
        # This is robust regardless of CSV format or batch_size since
        # read_all() concatenates all record batches into a flat table.
        self._row_index = self._build_row_index()

        # Validate loaded data
        self._validate_data()

    def _load_arrow_data(self) -> pa.Table:
        """Load Arrow IPC table from bytes.

        Downloads the file (local or GCS), then parses via PyArrow IPC.
        Tries File format first (random access) then falls back to
        Streaming format (sequential) which DVID's export-shards writes.
        """
        try:
            data = _read_bytes(self.arrow_path)
            buf = pa.BufferReader(data)
            try:
                reader = ipc.open_file(buf)
                return reader.read_all()
            except pa.ArrowInvalid:
                buf = pa.BufferReader(data)
                reader = ipc.open_stream(buf)
                return reader.read_all()
        except Exception as e:
            if isinstance(e, BraidError):
                raise
            raise BraidError(f"Failed to load Arrow file {self.arrow_path}: {e}")

    def _load_csv_index(self):
        """Load CSV chunk index (new or old format).

        Returns:
            (chunk_index, schema_size) — see ``_parse_csv_index``.
        """
        try:
            data = _read_bytes(self.csv_path)
            return _parse_csv_index(data)
        except Exception as e:
            if isinstance(e, BraidError):
                raise
            raise BraidError(f"Failed to load CSV index {self.csv_path}: {e}")

    def _build_row_index(self) -> Dict[Tuple[int, int, int], int]:
        """Build (x,y,z) -> global row index from the Arrow table."""
        xs = self._table.column("chunk_x").to_pylist()
        ys = self._table.column("chunk_y").to_pylist()
        zs = self._table.column("chunk_z").to_pylist()
        return {(x, y, z): i for i, (x, y, z) in enumerate(zip(xs, ys, zs))}

    def _validate_data(self):
        """Validate that loaded data is consistent and valid."""
        table_fields = {field.name for field in self._table.schema}
        expected_fields = {field.name for field in self.EXPECTED_SCHEMA}

        if not expected_fields.issubset(table_fields):
            missing = expected_fields - table_fields
            raise InvalidShardFormatError(
                f"Arrow table missing required fields: {missing}"
            )

        # Ensure every CSV coordinate has a matching row in the table
        missing_coords = set(self._chunk_index.keys()) - set(self._row_index.keys())
        if missing_coords:
            sample = list(missing_coords)[:3]
            raise InvalidShardFormatError(
                f"CSV references {len(missing_coords)} coordinates not found "
                f"in Arrow table, e.g. {sample}"
            )

    @property
    def chunk_count(self) -> int:
        """Number of chunks available in this shard."""
        return len(self._chunk_index)

    @property
    def is_empty(self) -> bool:
        """True if no chunk contains non-zero labels or supervoxels.

        Checks the Arrow column data directly (no decompression needed).
        A shard is empty when every chunk's label and supervoxel lists
        contain only zero (background) or are empty.  Such shards produce
        all-zero voxel data when decompressed, and TensorStore will not
        write a .shard file for all-zero (fill-value) data.
        """
        labels_col = self._table.column("labels")
        supervoxels_col = self._table.column("supervoxels")
        for i in range(len(self._table)):
            for val in labels_col[i].as_py():
                if val != 0:
                    return False
            for val in supervoxels_col[i].as_py():
                if val != 0:
                    return False
        return True

    @property
    def available_chunks(self) -> List[Tuple[int, int, int]]:
        """List of all available chunk coordinates (x, y, z)."""
        return list(self._chunk_index.keys())

    def has_chunk(self, x: int, y: int, z: int) -> bool:
        """Check if a chunk exists at the given coordinates."""
        return (x, y, z) in self._chunk_index

    def get_chunk_info(self, x: int, y: int, z: int) -> Dict[str, any]:
        """Get metadata about a chunk without decompressing it."""
        coord_key = (x, y, z)
        if coord_key not in self._chunk_index:
            raise ChunkNotFoundError(f"Chunk not found at coordinates {coord_key}")

        row_idx = self._row_index[coord_key]

        return {
            'coordinates': coord_key,
            'record_index': row_idx,
            'chunk_x': self._table['chunk_x'][row_idx].as_py(),
            'chunk_y': self._table['chunk_y'][row_idx].as_py(),
            'chunk_z': self._table['chunk_z'][row_idx].as_py(),
            'labels_count': len(self._table['labels'][row_idx].as_py()),
            'supervoxels_count': len(self._table['supervoxels'][row_idx].as_py()),
            'compressed_size': len(self._table['dvid_compressed_block'][row_idx].as_py()),
            'uncompressed_size': self._table['uncompressed_size'][row_idx].as_py()
        }

    def read_chunk(self, x: int, y: int, z: int,
                   label_type: LabelType = LabelType.LABELS,
                   chunk_shape: Tuple[int, int, int] = (64, 64, 64)) -> np.ndarray:
        """Read and decompress a chunk at the given coordinates."""
        if not all(isinstance(coord, int) and coord >= 0 for coord in [x, y, z]):
            raise InvalidCoordinateError(f"Invalid coordinates: ({x}, {y}, {z})")

        coord_key = (x, y, z)
        if coord_key not in self._chunk_index:
            raise ChunkNotFoundError(f"Chunk not found at coordinates {coord_key}")

        row_idx = self._row_index[coord_key]

        try:
            labels = self._table['labels'][row_idx].as_py()
            supervoxels = self._table['supervoxels'][row_idx].as_py()
            compressed_data = self._table['dvid_compressed_block'][row_idx].as_py()

            if label_type == LabelType.LABELS:
                return self._decompressor.decompress_block(
                    compressed_data=compressed_data,
                    agglo_labels=labels,
                    supervoxels=supervoxels,
                    block_shape=chunk_shape
                )
            elif label_type == LabelType.SUPERVOXELS:
                return self._decompressor.decompress_block(
                    compressed_data=compressed_data,
                    agglo_labels=supervoxels,
                    supervoxels=None,
                    block_shape=chunk_shape
                )
            else:
                raise ValueError(f"Invalid label type: {label_type}")

        except Exception as e:
            if isinstance(e, (ChunkNotFoundError, InvalidCoordinateError)):
                raise
            raise DecompressionError(f"Failed to read chunk at {coord_key}: {e}")

    def read_chunk_raw(self, x: int, y: int, z: int) -> Dict[str, any]:
        """Read raw chunk data without decompression."""
        coord_key = (x, y, z)
        if coord_key not in self._chunk_index:
            raise ChunkNotFoundError(f"Chunk not found at coordinates {coord_key}")

        row_idx = self._row_index[coord_key]

        return {
            'coordinates': coord_key,
            'chunk_x': self._table['chunk_x'][row_idx].as_py(),
            'chunk_y': self._table['chunk_y'][row_idx].as_py(),
            'chunk_z': self._table['chunk_z'][row_idx].as_py(),
            'labels': self._table['labels'][row_idx].as_py(),
            'supervoxels': self._table['supervoxels'][row_idx].as_py(),
            'compressed_data': self._table['dvid_compressed_block'][row_idx].as_py(),
            'uncompressed_size': self._table['uncompressed_size'][row_idx].as_py()
        }

    def __repr__(self) -> str:
        return (f"ShardReader(arrow_path='{self.arrow_path}', "
                f"csv_path='{self.csv_path}', chunks={self.chunk_count})")

    def __len__(self) -> int:
        """Return number of chunks in this shard."""
        return self.chunk_count


class ShardRangeReader:
    """
    Memory-efficient shard reader using byte offsets from the CSV index.

    Instead of downloading the entire Arrow file into memory, this reader
    uses byte offsets from the CSV (produced by DVID ``export-shards``) to
    fetch individual record batches via GCS byte-range reads.

    When ``batch_size > 1``, multiple chunks share a record batch.  The
    reader caches the most recently fetched batch so that consecutive
    chunks within the same batch are served from cache without an
    additional GCS round-trip.

    Memory per access: ~schema bytes + one record batch vs the full Arrow
    file (up to ~6.6 GB for the largest shards).

    The public interface matches ``ShardReader`` so the two are
    interchangeable in the worker:

        >>> reader = ShardRangeReader(
        ...     "gs://bucket/shards/s0/0_0_0.arrow",
        ...     "gs://bucket/shards/s0/0_0_0.csv",
        ... )
        >>> for cx, cy, cz in reader.available_chunks:
        ...     data = reader.read_chunk(cx, cy, cz)
    """

    def __init__(self, arrow_path: Union[str, Path],
                 csv_path: Union[str, Path]):
        """
        Initialize range-read shard reader.

        Args:
            arrow_path: Path or GCS URI to the Arrow IPC shard file.
            csv_path: Path or GCS URI to the CSV chunk index file.
                Must contain columns ``x, y, z, offset, size, batch_idx``
                and a ``# schema_size=N`` comment header.
        """
        self.arrow_path = str(arrow_path)
        self.csv_path = str(csv_path)

        # Load the small CSV index (contains offsets + batch positions)
        self._chunk_index, self._schema_size = self._load_csv_index()

        # Lazily downloaded Arrow schema bytes (first schema_size bytes)
        self._schema_bytes: bytes = b""
        self._schema: Optional[pa.Schema] = None

        # Batch cache: avoid re-downloading when consecutive chunks share
        # the same record batch (batch_size > 1).
        self._cached_batch_key: Optional[Tuple[int, int]] = None
        self._cached_batch: Optional[pa.RecordBatch] = None
        self._batch_fetches: int = 0  # for testing / monitoring

        self._decompressor = DVIDDecompressor()

    def _load_csv_index(self):
        """Load the CSV chunk index (new or old format).

        Returns:
            (chunk_index, schema_size) — see ``_parse_csv_index``.
        """
        data = _read_bytes(self.csv_path)
        return _parse_csv_index(data)

    def _ensure_schema(self):
        """Download the Arrow schema bytes on first access."""
        if self._schema is not None:
            return
        if self._schema_size == 0:
            raise BraidError("schema_size is 0 in CSV; cannot read schema")
        if self.arrow_path.startswith("gs://"):
            self._schema_bytes = _read_gcs_range(
                self.arrow_path, 0, self._schema_size)
        else:
            with open(self.arrow_path, "rb") as f:
                self._schema_bytes = f.read(self._schema_size)

        msg = ipc.read_message(pa.BufferReader(self._schema_bytes))
        self._schema = ipc.read_schema(msg)

    def _get_batch(self, offset: int, size: int) -> pa.RecordBatch:
        """Fetch a record batch, returning cached batch on cache hit."""
        key = (offset, size)
        if key == self._cached_batch_key:
            return self._cached_batch

        self._ensure_schema()

        if self.arrow_path.startswith("gs://"):
            raw = _read_gcs_range(self.arrow_path, offset, size)
        else:
            with open(self.arrow_path, "rb") as f:
                f.seek(offset)
                raw = f.read(size)

        msg = ipc.read_message(pa.BufferReader(raw))
        batch = ipc.read_record_batch(msg, self._schema)
        self._cached_batch_key = key
        self._cached_batch = batch
        self._batch_fetches += 1
        return batch

    @property
    def chunk_count(self) -> int:
        """Number of chunks available in this shard."""
        return len(self._chunk_index)

    @property
    def available_chunks(self) -> List[Tuple[int, int, int]]:
        """List of all available chunk coordinates (x, y, z)."""
        return list(self._chunk_index.keys())

    def has_chunk(self, x: int, y: int, z: int) -> bool:
        """Check if a chunk exists at the given coordinates."""
        return (x, y, z) in self._chunk_index

    def read_chunk(self, x: int, y: int, z: int,
                   label_type: LabelType = LabelType.LABELS,
                   chunk_shape: Tuple[int, int, int] = (64, 64, 64)) -> np.ndarray:
        """Read and decompress a single chunk via byte-range read.

        Only downloads the Arrow schema (once) plus the specific record
        batch for this chunk.  If another chunk in the same batch was
        read recently, the cached batch is reused.
        """
        if not all(isinstance(c, int) and c >= 0 for c in [x, y, z]):
            raise InvalidCoordinateError(f"Invalid coordinates: ({x}, {y}, {z})")

        coord_key = (x, y, z)
        if coord_key not in self._chunk_index:
            raise ChunkNotFoundError(
                f"Chunk not found at coordinates {coord_key}")

        loc = self._chunk_index[coord_key]

        try:
            batch = self._get_batch(loc.offset, loc.size)
            idx = loc.batch_idx

            labels = batch.column("labels")[idx].as_py()
            supervoxels = batch.column("supervoxels")[idx].as_py()
            compressed_data = batch.column(
                "dvid_compressed_block")[idx].as_py()

            if label_type == LabelType.LABELS:
                return self._decompressor.decompress_block(
                    compressed_data=compressed_data,
                    agglo_labels=labels,
                    supervoxels=supervoxels,
                    block_shape=chunk_shape,
                )
            elif label_type == LabelType.SUPERVOXELS:
                return self._decompressor.decompress_block(
                    compressed_data=compressed_data,
                    agglo_labels=supervoxels,
                    supervoxels=None,
                    block_shape=chunk_shape,
                )
            else:
                raise ValueError(f"Invalid label type: {label_type}")

        except Exception as e:
            if isinstance(e, (ChunkNotFoundError, InvalidCoordinateError)):
                raise
            raise DecompressionError(
                f"Failed to read chunk at {coord_key}: {e}")

    def read_chunk_raw(self, x: int, y: int, z: int) -> Dict[str, any]:
        """Read raw chunk data without decompression via byte-range read."""
        coord_key = (x, y, z)
        if coord_key not in self._chunk_index:
            raise ChunkNotFoundError(
                f"Chunk not found at coordinates {coord_key}")

        loc = self._chunk_index[coord_key]
        batch = self._get_batch(loc.offset, loc.size)
        idx = loc.batch_idx

        return {
            "coordinates": coord_key,
            "chunk_x": batch.column("chunk_x")[idx].as_py(),
            "chunk_y": batch.column("chunk_y")[idx].as_py(),
            "chunk_z": batch.column("chunk_z")[idx].as_py(),
            "labels": batch.column("labels")[idx].as_py(),
            "supervoxels": batch.column("supervoxels")[idx].as_py(),
            "compressed_data": batch.column(
                "dvid_compressed_block")[idx].as_py(),
            "uncompressed_size": batch.column(
                "uncompressed_size")[idx].as_py(),
        }

    def __repr__(self) -> str:
        return (f"ShardRangeReader(arrow_path='{self.arrow_path}', "
                f"chunks={self.chunk_count})")

    def __len__(self) -> int:
        """Return number of chunks in this shard."""
        return self.chunk_count
