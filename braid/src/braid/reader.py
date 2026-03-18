"""
BRAID Shard Reader class.

Provides chunk-wise access to sharded Arrow files with CSV coordinate indices.
Supports multiple compression backends and label types.
"""

import csv
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from .decompressor import DVIDDecompressor
from .exceptions import BraidError, ChunkNotFoundError, DecompressionError, InvalidShardFormatError, InvalidCoordinateError


class LabelType(Enum):
    """Enumeration for label types available in shard files."""
    LABELS = "labels"           # Agglomerated label IDs
    SUPERVOXELS = "supervoxels" # Original supervoxel IDs


class ShardReader:
    """
    BRAID shard reader for Arrow files with CSV coordinate indices.
    
    This class provides efficient chunk-wise access to sharded segmentation data
    with support for multiple label types and compression backends.
    
    Example:
        >>> reader = ShardReader("shard_0_0_0.arrow", "shard_0_0_0.csv")
        >>> chunk_data = reader.read_chunk(64, 128, 0, label_type=LabelType.LABELS)
        >>> print(chunk_data.shape)  # (64, 64, 64)
        >>> print(chunk_data.dtype)  # uint64
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
            arrow_path: Path to the Arrow IPC shard file
            csv_path: Path to the CSV chunk index file
            
        Raises:
            BraidError: If files cannot be loaded or have invalid format
        """
        self.arrow_path = Path(arrow_path)
        self.csv_path = Path(csv_path)
        
        if not self.arrow_path.exists():
            raise BraidError(f"Arrow file not found: {arrow_path}")
        if not self.csv_path.exists():
            raise BraidError(f"CSV file not found: {csv_path}")
        
        # Load data during initialization
        self._table = self._load_arrow_data()
        self._chunk_index = self._load_csv_index()
        self._decompressor = DVIDDecompressor()
        
        # Validate loaded data
        self._validate_data()
        
    def _load_arrow_data(self) -> pa.Table:
        """Load Arrow IPC table from file."""
        try:
            with open(self.arrow_path, 'rb') as f:
                with ipc.open_file(f) as reader:
                    return reader.read_all()
        except Exception as e:
            raise BraidError(f"Failed to load Arrow file {self.arrow_path}: {e}")
    
    def _load_csv_index(self) -> Dict[Tuple[int, int, int], int]:
        """Load CSV chunk index into memory for fast lookup."""
        chunk_index = {}
        
        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                
                # Validate CSV headers
                expected_headers = {'x', 'y', 'z', 'rec'}
                if not expected_headers.issubset(reader.fieldnames or []):
                    raise InvalidShardFormatError(
                        f"CSV file missing required headers. Expected: {expected_headers}, "
                        f"Found: {reader.fieldnames}"
                    )
                
                for row_num, row in enumerate(reader, 1):
                    try:
                        coord_key = (int(row['x']), int(row['y']), int(row['z']))
                        record_idx = int(row['rec'])
                        
                        if record_idx < 0:
                            raise ValueError(f"Negative record index: {record_idx}")
                        
                        chunk_index[coord_key] = record_idx
                        
                    except (ValueError, KeyError) as e:
                        raise InvalidShardFormatError(
                            f"Invalid CSV format at row {row_num}: {e}"
                        )
                        
        except Exception as e:
            if isinstance(e, BraidError):
                raise
            raise BraidError(f"Failed to load CSV index {self.csv_path}: {e}")
        
        return chunk_index
    
    def _validate_data(self):
        """Validate that loaded data is consistent and valid."""
        # Check Arrow table has expected schema structure
        table_fields = {field.name for field in self._table.schema}
        expected_fields = {field.name for field in self.EXPECTED_SCHEMA}
        
        if not expected_fields.issubset(table_fields):
            missing = expected_fields - table_fields
            raise InvalidShardFormatError(
                f"Arrow table missing required fields: {missing}"
            )
        
        # Check that all CSV record indices are valid
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
        """
        Check if a chunk exists at the given coordinates.
        
        Args:
            x, y, z: Chunk coordinates
            
        Returns:
            True if chunk exists, False otherwise
        """
        return (x, y, z) in self._chunk_index
    
    def get_chunk_info(self, x: int, y: int, z: int) -> Dict[str, any]:
        """
        Get metadata about a chunk without decompressing it.
        
        Args:
            x, y, z: Chunk coordinates
            
        Returns:
            Dictionary with chunk metadata
            
        Raises:
            ChunkNotFoundError: If chunk doesn't exist
        """
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
        """
        Read and decompress a chunk at the given coordinates.
        
        Args:
            x, y, z: Chunk coordinates
            label_type: Which label type to use (LABELS or SUPERVOXELS)
            chunk_shape: Expected shape of the output chunk
            
        Returns:
            Decompressed uint64 array of specified shape
            
        Raises:
            ChunkNotFoundError: If chunk doesn't exist
            InvalidCoordinateError: If coordinates are invalid
            DecompressionError: If decompression fails
        """
        # Validate coordinates
        if not all(isinstance(coord, int) and coord >= 0 for coord in [x, y, z]):
            raise InvalidCoordinateError(f"Invalid coordinates: ({x}, {y}, {z})")
        
        coord_key = (x, y, z)
        if coord_key not in self._chunk_index:
            raise ChunkNotFoundError(f"Chunk not found at coordinates {coord_key}")
        
        record_idx = self._chunk_index[coord_key]
        
        try:
            # Extract data from Arrow record
            labels = self._table['labels'][record_idx].as_py()
            supervoxels = self._table['supervoxels'][record_idx].as_py()
            compressed_data = self._table['dvid_compressed_block'][record_idx].as_py()
            # Note: uncompressed_size not needed since zstd decompressor handles this automatically
            
            # Decompress using DVID decompressor with label mapping
            if label_type == LabelType.LABELS:
                # Use agglomerated labels with supervoxel mapping
                decompressed_chunk = self._decompressor.decompress_block(
                    compressed_data=compressed_data,
                    agglo_labels=labels,
                    supervoxels=supervoxels,
                    block_shape=chunk_shape
                )
            elif label_type == LabelType.SUPERVOXELS:
                # Use supervoxels directly (no mapping)
                decompressed_chunk = self._decompressor.decompress_block(
                    compressed_data=compressed_data,
                    agglo_labels=supervoxels,
                    supervoxels=None,
                    block_shape=chunk_shape
                )
            else:
                raise ValueError(f"Invalid label type: {label_type}")
            
            return decompressed_chunk
            
        except Exception as e:
            if isinstance(e, (ChunkNotFoundError, InvalidCoordinateError)):
                raise
            raise DecompressionError(f"Failed to read chunk at {coord_key}: {e}")
    
    def read_chunk_raw(self, x: int, y: int, z: int) -> Dict[str, any]:
        """
        Read raw chunk data without decompression.
        
        Args:
            x, y, z: Chunk coordinates
            
        Returns:
            Dictionary with all raw chunk data
            
        Raises:
            ChunkNotFoundError: If chunk doesn't exist
        """
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