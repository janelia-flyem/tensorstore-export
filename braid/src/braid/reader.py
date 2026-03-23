"""
BRAID Shard Reader class.

Provides chunk-wise access to sharded Arrow files with CSV coordinate indices.
Supports local files and GCS URIs (gs://bucket/path) transparently.

GCS access uses google-cloud-storage (install with ``pip install braid[gcs]``)
rather than PyArrow's native ``pyarrow.fs.GcsFileSystem``, which suffers from
CURL error 81 ("Socket not ready for send/recv") on Google Cloud Run.
See docs/GoogleCloudRunIssues.md for details.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.ipc as ipc
from .decompressor import DVIDDecompressor
from .exceptions import BraidError, ChunkNotFoundError, DecompressionError, InvalidShardFormatError, InvalidCoordinateError


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
        self._chunk_index = self._load_csv_index()
        self._decompressor = DVIDDecompressor()

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

    def _load_csv_index(self) -> Dict[Tuple[int, int, int], int]:
        """Load CSV chunk index using PyArrow's CSV reader.

        Downloads the file (local or GCS), parses via pyarrow.csv in C++,
        then builds the Python lookup dict from Arrow columns.
        """
        try:
            data = _read_bytes(self.csv_path)
            table = pcsv.read_csv(pa.BufferReader(data))

            # Validate columns
            expected = {"x", "y", "z", "rec"}
            actual = set(table.column_names)
            if not expected.issubset(actual):
                raise InvalidShardFormatError(
                    f"CSV missing required columns. Expected: {expected}, Found: {actual}"
                )

            # Build lookup dict from Arrow columns
            xs = table.column("x").to_pylist()
            ys = table.column("y").to_pylist()
            zs = table.column("z").to_pylist()
            recs = table.column("rec").to_pylist()
            return {(x, y, z): rec for x, y, z, rec in zip(xs, ys, zs, recs)}

        except Exception as e:
            if isinstance(e, BraidError):
                raise
            raise BraidError(f"Failed to load CSV index {self.csv_path}: {e}")

    def _validate_data(self):
        """Validate that loaded data is consistent and valid."""
        table_fields = {field.name for field in self._table.schema}
        expected_fields = {field.name for field in self.EXPECTED_SCHEMA}

        if not expected_fields.issubset(table_fields):
            missing = expected_fields - table_fields
            raise InvalidShardFormatError(
                f"Arrow table missing required fields: {missing}"
            )

        max_record_idx = max(self._chunk_index.values()) if self._chunk_index else -1
        if max_record_idx >= self._table.num_rows:
            raise InvalidShardFormatError(
                f"CSV references record index {max_record_idx} but table only has "
                f"{self._table.num_rows} rows"
            )

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

    def get_chunk_info(self, x: int, y: int, z: int) -> Dict[str, any]:
        """Get metadata about a chunk without decompressing it."""
        coord_key = (x, y, z)
        if coord_key not in self._chunk_index:
            raise ChunkNotFoundError(f"Chunk not found at coordinates {coord_key}")

        record_idx = self._chunk_index[coord_key]

        return {
            'coordinates': coord_key,
            'record_index': record_idx,
            'chunk_x': self._table['chunk_x'][record_idx].as_py(),
            'chunk_y': self._table['chunk_y'][record_idx].as_py(),
            'chunk_z': self._table['chunk_z'][record_idx].as_py(),
            'labels_count': len(self._table['labels'][record_idx].as_py()),
            'supervoxels_count': len(self._table['supervoxels'][record_idx].as_py()),
            'compressed_size': len(self._table['dvid_compressed_block'][record_idx].as_py()),
            'uncompressed_size': self._table['uncompressed_size'][record_idx].as_py()
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

        record_idx = self._chunk_index[coord_key]

        try:
            labels = self._table['labels'][record_idx].as_py()
            supervoxels = self._table['supervoxels'][record_idx].as_py()
            compressed_data = self._table['dvid_compressed_block'][record_idx].as_py()

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

        record_idx = self._chunk_index[coord_key]

        return {
            'coordinates': coord_key,
            'chunk_x': self._table['chunk_x'][record_idx].as_py(),
            'chunk_y': self._table['chunk_y'][record_idx].as_py(),
            'chunk_z': self._table['chunk_z'][record_idx].as_py(),
            'labels': self._table['labels'][record_idx].as_py(),
            'supervoxels': self._table['supervoxels'][record_idx].as_py(),
            'compressed_data': self._table['dvid_compressed_block'][record_idx].as_py(),
            'uncompressed_size': self._table['uncompressed_size'][record_idx].as_py()
        }

    def __repr__(self) -> str:
        return (f"ShardReader(arrow_path='{self.arrow_path}', "
                f"csv_path='{self.csv_path}', chunks={self.chunk_count})")

    def __len__(self) -> int:
        """Return number of chunks in this shard."""
        return self.chunk_count


class ShardRangeReader:
    """
    Memory-efficient shard reader using pre-computed byte offsets.

    Instead of downloading the entire Arrow file into memory, this reader
    uses a companion offset CSV (produced by ``pixi run compute-offsets``)
    to fetch individual record batches via GCS byte-range reads.

    Memory per chunk: ~schema bytes + one record batch (~5-50 KB) vs
    the full Arrow file (up to ~6.6 GB for the largest shards).

    The public interface matches ``ShardReader`` so the two are
    interchangeable in the worker:

        >>> reader = ShardRangeReader(
        ...     "gs://bucket/shards/s0/0_0_0.arrow",
        ...     "gs://bucket/shards/s0/0_0_0.csv",
        ...     "gs://bucket/shards/s0/0_0_0-offsets.csv",
        ... )
        >>> for cx, cy, cz in reader.available_chunks:
        ...     data = reader.read_chunk(cx, cy, cz)
    """

    def __init__(self, arrow_path: Union[str, Path],
                 csv_path: Union[str, Path],
                 offsets_csv_path: Union[str, Path]):
        """
        Initialize range-read shard reader.

        Args:
            arrow_path: Path or GCS URI to the Arrow IPC shard file.
            csv_path: Path or GCS URI to the CSV chunk index file.
            offsets_csv_path: Path or GCS URI to the byte-offset CSV
                produced by ``compute_offsets.py``.
        """
        self.arrow_path = str(arrow_path)
        self.csv_path = str(csv_path)
        self.offsets_csv_path = str(offsets_csv_path)

        # Load the small metadata files (CSV index + offsets)
        self._chunk_index = self._load_csv_index()
        self._record_offsets, self._schema_size = self._load_offsets()

        # Lazily downloaded Arrow schema bytes (first schema_size bytes)
        self._schema_bytes: bytes = b""
        self._schema: pa.Schema = None

        self._decompressor = DVIDDecompressor()

    def _load_csv_index(self) -> Dict[Tuple[int, int, int], int]:
        """Load the (x,y,z) -> record_index mapping from the CSV index."""
        data = _read_bytes(self.csv_path)
        table = pcsv.read_csv(pa.BufferReader(data))
        xs = table.column("x").to_pylist()
        ys = table.column("y").to_pylist()
        zs = table.column("z").to_pylist()
        recs = table.column("rec").to_pylist()
        return {(x, y, z): rec for x, y, z, rec in zip(xs, ys, zs, recs)}

    def _load_offsets(self) -> tuple:
        """Load byte offsets from the offset CSV.

        Returns:
            (record_offsets, schema_size) where record_offsets is a dict
            mapping record_index -> (byte_offset, byte_size).
        """
        import csv as csv_mod
        import io

        data = _read_bytes(self.offsets_csv_path)
        text = data.decode("utf-8")

        # Parse schema_size from comment header
        schema_size = 0
        for line in text.splitlines():
            if line.startswith("# schema_size="):
                schema_size = int(line.split("=", 1)[1])
                break

        reader = csv_mod.DictReader(
            (l for l in io.StringIO(text) if not l.startswith("#")))
        offsets = {}
        for row in reader:
            rec = int(row["rec"])
            offsets[rec] = (int(row["offset"]), int(row["size"]))
        return offsets, schema_size

    def _ensure_schema(self):
        """Download the Arrow schema bytes on first access."""
        if self._schema is not None:
            return
        if self._schema_size == 0:
            raise BraidError("schema_size is 0 in offset CSV; "
                             "cannot read schema")
        if self.arrow_path.startswith("gs://"):
            self._schema_bytes = _read_gcs_range(
                self.arrow_path, 0, self._schema_size)
        else:
            with open(self.arrow_path, "rb") as f:
                self._schema_bytes = f.read(self._schema_size)

        msg = ipc.read_message(pa.BufferReader(self._schema_bytes))
        self._schema = ipc.read_schema(msg)

    def _read_record(self, record_idx: int) -> pa.RecordBatch:
        """Fetch and parse a single record batch by byte range."""
        if record_idx not in self._record_offsets:
            raise BraidError(f"Record {record_idx} not in offset CSV")

        offset, size = self._record_offsets[record_idx]
        self._ensure_schema()

        if self.arrow_path.startswith("gs://"):
            raw = _read_gcs_range(self.arrow_path, offset, size)
        else:
            with open(self.arrow_path, "rb") as f:
                f.seek(offset)
                raw = f.read(size)

        msg = ipc.read_message(pa.BufferReader(raw))
        return ipc.read_record_batch(msg, self._schema)

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
        batch for this chunk, instead of the entire Arrow file.
        """
        if not all(isinstance(c, int) and c >= 0 for c in [x, y, z]):
            raise InvalidCoordinateError(f"Invalid coordinates: ({x}, {y}, {z})")

        coord_key = (x, y, z)
        if coord_key not in self._chunk_index:
            raise ChunkNotFoundError(
                f"Chunk not found at coordinates {coord_key}")

        record_idx = self._chunk_index[coord_key]

        try:
            batch = self._read_record(record_idx)

            labels = batch.column("labels")[0].as_py()
            supervoxels = batch.column("supervoxels")[0].as_py()
            compressed_data = batch.column(
                "dvid_compressed_block")[0].as_py()

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

        record_idx = self._chunk_index[coord_key]
        batch = self._read_record(record_idx)

        return {
            "coordinates": coord_key,
            "chunk_x": batch.column("chunk_x")[0].as_py(),
            "chunk_y": batch.column("chunk_y")[0].as_py(),
            "chunk_z": batch.column("chunk_z")[0].as_py(),
            "labels": batch.column("labels")[0].as_py(),
            "supervoxels": batch.column("supervoxels")[0].as_py(),
            "compressed_data": batch.column(
                "dvid_compressed_block")[0].as_py(),
            "uncompressed_size": batch.column(
                "uncompressed_size")[0].as_py(),
        }

    def __repr__(self) -> str:
        return (f"ShardRangeReader(arrow_path='{self.arrow_path}', "
                f"chunks={self.chunk_count})")

    def __len__(self) -> int:
        """Return number of chunks in this shard."""
        return self.chunk_count
