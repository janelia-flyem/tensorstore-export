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
