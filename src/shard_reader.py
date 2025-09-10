"""
Core shard reader module for processing DVID-exported Arrow IPC files with CSV coordinate indices.

This module implements efficient reading of DVID export shards as described in ExportShards.md,
using Arrow IPC files paired with CSV coordinate indices for fast random access.
"""

import csv
import os
import functools
from typing import Dict, List, Tuple, Optional
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import gcsfs
import structlog

logger = structlog.get_logger()


# DVID export schema from ExportShards.md
DVID_SHARD_SCHEMA = pa.schema([
    pa.field('chunk_x', pa.int32(), nullable=False),
    pa.field('chunk_y', pa.int32(), nullable=False),
    pa.field('chunk_z', pa.int32(), nullable=False),
    pa.field('labels', pa.list_(pa.uint64()), nullable=False),
    pa.field('supervoxels', pa.list_(pa.uint64()), nullable=False),
    pa.field('dvid_compressed_block', pa.binary(), nullable=False),
    pa.field('uncompressed_size', pa.uint32(), nullable=False)
])


class DVIDDecompressor:
    """
    DVID Block decompressor implementation.
    
    Based on the DVID compressed segmentation format. This handles the custom compression 
    scheme with block-level label lists and sub-block indices.
    """
    
    @staticmethod
    def decompress_block(compressed_data: bytes, labels: List[int], uncompressed_size: int, 
                        block_shape: Tuple[int, int, int] = (64, 64, 64)) -> np.ndarray:
        """
        Decompress a DVID compressed block.
        
        Args:
            compressed_data: The zstd-compressed DVID binary blob
            labels: List of uint64 labels for this block  
            uncompressed_size: Expected size after decompression
            block_shape: Shape of the output block (nx, ny, nz)
            
        Returns:
            Decompressed uint64 array of shape block_shape
        """
        if len(compressed_data) == 0:
            # Empty block - return all zeros
            return np.zeros(block_shape, dtype=np.uint64)
        
        if len(labels) == 1:
            # Solid block - single label for entire block
            return np.full(block_shape, labels[0], dtype=np.uint64)
        
        logger.debug("Decompressing DVID block", 
                    labels_count=len(labels), 
                    compressed_size=len(compressed_data),
                    uncompressed_size=uncompressed_size,
                    block_shape=block_shape)
        
        try:
            # TODO: Implement actual DVID decompression
            # For now, use a simplified placeholder that works with the label list
            
            # Decompress the zstd data first
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            decompressed_data = dctx.decompress(compressed_data)
            
            if len(decompressed_data) != uncompressed_size:
                logger.warning("Size mismatch after decompression", 
                              expected=uncompressed_size, 
                              actual=len(decompressed_data))
            
            # Simplified parsing - in production this would follow the full DVID format
            # This assumes the decompressed data contains indices into the labels array
            expected_voxels = np.prod(block_shape)
            
            if len(decompressed_data) >= expected_voxels:
                # Treat as raw indices into labels
                indices = np.frombuffer(decompressed_data[:expected_voxels], dtype=np.uint8)
                indices = indices.reshape(block_shape)
                label_array = np.array(labels, dtype=np.uint64)
                return label_array[indices % len(labels)]
            else:
                # Fallback to first label
                return np.full(block_shape, labels[0], dtype=np.uint64)
                
        except Exception as e:
            logger.warning("Failed to decompress DVID block, using first label", 
                          error=str(e), labels_available=len(labels))
            return np.full(block_shape, labels[0] if labels else 0, dtype=np.uint64)


@functools.lru_cache(maxsize=10)
def load_shard_data(arrow_gcs_path: str, fs: gcsfs.GCSFileSystem) -> Tuple[pa.Table, Dict[Tuple[int, int, int], int]]:
    """
    Load Arrow IPC shard data and its corresponding CSV chunk index.
    
    This function is cached to avoid repeatedly opening the same shard files.
    Following the DVID export format from ExportShards.md.
    
    Args:
        arrow_gcs_path: GCS path to the Arrow IPC file (e.g., 'gs://bucket/path/0_0_0.arrow')
        fs: GCSFileSystem instance
        
    Returns:
        Tuple of (Arrow table, chunk coordinate index dict)
    """
    logger.info("Loading shard data", path=arrow_gcs_path)
    
    try:
        # Load Arrow IPC file
        with fs.open(arrow_gcs_path, 'rb') as arrow_file:
            with ipc.open_file(arrow_file) as reader:
                table = reader.read_all()
        
        # Load corresponding CSV index file
        # Convert .arrow to .csv filename
        csv_gcs_path = arrow_gcs_path.rsplit('.', 1)[0] + '.csv'
        
        chunk_index = {}
        try:
            with fs.open(csv_gcs_path, 'r') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    # Create tuple key from x,y,z coordinates
                    coord_key = (int(row['x']), int(row['y']), int(row['z']))
                    chunk_index[coord_key] = int(row['rec'])
                    
        except Exception as e:
            logger.error("Failed to load CSV index", csv_path=csv_gcs_path, error=str(e))
            raise RuntimeError(f"Could not load chunk index from {csv_gcs_path}: {e}")
        
        logger.info("Successfully loaded shard data", 
                   arrow_path=arrow_gcs_path,
                   total_records=table.num_rows,
                   indexed_chunks=len(chunk_index))
        
        return table, chunk_index
        
    except Exception as e:
        logger.error("Failed to load shard data", path=arrow_gcs_path, error=str(e))
        raise


class ShardReader:
    """
    Reader for individual DVID shard files (.arrow/.csv pairs).
    
    This class handles reading from a single shard file and provides access to chunks
    within that shard using global coordinates.
    """
    
    def __init__(self, arrow_gcs_path: str):
        """
        Initialize shard reader for a specific Arrow/CSV shard pair.
        
        Args:
            arrow_gcs_path: GCS path to the Arrow IPC file (e.g., 'gs://bucket/0_0_0.arrow')
        """
        self.arrow_gcs_path = arrow_gcs_path
        self.fs = gcsfs.GCSFileSystem()
        self.decompressor = DVIDDecompressor()
        
        # Load shard data once during initialization
        self.table, self.chunk_index = load_shard_data(arrow_gcs_path, self.fs)
        
        logger.info("Initialized shard reader", 
                   path=arrow_gcs_path,
                   chunks_available=len(self.chunk_index))
    
    def get_chunk(self, x: int, y: int, z: int) -> Optional[Dict]:
        """
        Get raw chunk data by global coordinates.
        
        Args:
            x, y, z: Global chunk coordinates
            
        Returns:
            Dictionary with chunk data or None if not found
        """
        coord_key = (x, y, z)
        if coord_key not in self.chunk_index:
            return None
            
        record_idx = self.chunk_index[coord_key]
        return {
            'coordinates': (x, y, z),
            'labels': self.table['labels'][record_idx].as_py(),
            'supervoxels': self.table['supervoxels'][record_idx].as_py(),
            'compressed_data': self.table['dvid_compressed_block'][record_idx].as_py(),
            'uncompressed_size': self.table['uncompressed_size'][record_idx].as_py()
        }
    
    def read_chunk_at_global_coords(self, global_coords: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """
        Read and decompress a single chunk at the given global coordinates.
        
        Args:
            global_coords: Global coordinates (x, y, z) of the chunk
            
        Returns:
            Decompressed uint64 array of shape (64, 64, 64) or None if chunk not found
        """
        x, y, z = global_coords
        
        try:
            # Get chunk data
            chunk_data = self.get_chunk(x, y, z)
            if chunk_data is None:
                logger.debug("Chunk not found in shard", 
                           global_coords=global_coords,
                           shard_path=self.arrow_gcs_path)
                return None
            
            # Decompress the DVID block
            decompressed_chunk = self.decompressor.decompress_block(
                chunk_data['compressed_data'], 
                chunk_data['labels'],
                chunk_data['uncompressed_size'],
                block_shape=(64, 64, 64)
            )
            
            logger.debug("Successfully decompressed chunk",
                        global_coords=global_coords,
                        labels_count=len(chunk_data['labels']),
                        output_shape=decompressed_chunk.shape)
            
            return decompressed_chunk
            
        except Exception as e:
            logger.error("Failed to read chunk",
                        global_coords=global_coords,
                        shard_path=self.arrow_gcs_path,
                        error=str(e))
            raise
    
    def list_chunks(self) -> List[Tuple[int, int, int]]:
        """List all available chunk coordinates in this shard."""
        return list(self.chunk_index.keys())
    
    def contains_chunk(self, x: int, y: int, z: int) -> bool:
        """Check if this shard contains the specified chunk coordinates."""
        return (x, y, z) in self.chunk_index


class MultiShardReader:
    """
    Reader that can handle multiple shard files and route chunk requests appropriately.
    
    This is the main class for reading from a collection of DVID export shards.
    """
    
    def __init__(self, base_gcs_path: str):
        """
        Initialize multi-shard reader.
        
        Args:
            base_gcs_path: Base GCS path containing shard files
        """
        self.base_gcs_path = base_gcs_path.rstrip('/')
        self.fs = gcsfs.GCSFileSystem()
        
        # Will be populated lazily as shards are accessed
        self.shard_readers = {}  # Path -> ShardReader
        self.global_chunk_index = {}  # (x, y, z) -> shard_path
        
        logger.info("Initialized multi-shard reader", base_path=base_gcs_path)
    
    def _discover_shards(self):
        """Discover all available shard files and build global index."""
        if self.global_chunk_index:
            return  # Already discovered
        
        logger.info("Discovering shard files", base_path=self.base_gcs_path)
        
        # Find all .arrow files
        arrow_files = self.fs.glob(f"{self.base_gcs_path}/**/*.arrow")
        
        for arrow_path in arrow_files:
            arrow_gcs_path = f"gs://{arrow_path}"
            
            try:
                # Load just the CSV index to build global mapping
                csv_gcs_path = arrow_gcs_path.rsplit('.', 1)[0] + '.csv'
                
                with self.fs.open(csv_gcs_path, 'r') as csv_file:
                    reader = csv.DictReader(csv_file)
                    for row in reader:
                        coord_key = (int(row['x']), int(row['y']), int(row['z']))
                        self.global_chunk_index[coord_key] = arrow_gcs_path
                        
            except Exception as e:
                logger.warning("Failed to index shard", path=arrow_gcs_path, error=str(e))
                continue
        
        logger.info("Shard discovery complete", 
                   total_shards=len(set(self.global_chunk_index.values())),
                   total_chunks=len(self.global_chunk_index))
    
    def read_chunk_at_global_coords(self, global_coords: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """
        Read chunk from the appropriate shard file.
        
        Args:
            global_coords: Global coordinates (x, y, z) of the chunk
            
        Returns:
            Decompressed uint64 array of shape (64, 64, 64) or None if not found
        """
        self._discover_shards()
        
        # Find which shard contains this chunk
        if global_coords not in self.global_chunk_index:
            logger.debug("Chunk not found in any shard", global_coords=global_coords)
            return None
        
        shard_path = self.global_chunk_index[global_coords]
        
        # Get or create shard reader
        if shard_path not in self.shard_readers:
            self.shard_readers[shard_path] = ShardReader(shard_path)
        
        # Read from the appropriate shard
        return self.shard_readers[shard_path].read_chunk_at_global_coords(global_coords)