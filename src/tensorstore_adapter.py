"""
TensorStore virtual chunked adapter for DVID shard data.

This module provides the bridge between DVID shard readers and TensorStore's
virtual_chunked driver, enabling seamless integration with TensorStore operations.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import tensorstore as ts
import structlog
from .shard_reader import MultiShardReader

logger = structlog.get_logger()


class DVIDVirtualChunkedAdapter:
    """
    Adapter that creates TensorStore virtual chunked volumes from DVID shard data.
    
    This class handles the creation of virtual TensorStore volumes that can read
    from DVID export shards on-demand using TensorStore's virtual_chunked driver.
    """
    
    def __init__(self, base_gcs_path: str, volume_shape: Tuple[int, int, int], 
                 chunk_shape: Tuple[int, int, int] = (64, 64, 64)):
        """
        Initialize the adapter.
        
        Args:
            base_gcs_path: Base GCS path containing DVID shard files
            volume_shape: Total shape of the volume (z, y, x)  
            chunk_shape: Shape of individual chunks (typically 64x64x64)
        """
        self.base_gcs_path = base_gcs_path
        self.volume_shape = volume_shape
        self.chunk_shape = chunk_shape
        
        # Initialize the multi-shard reader
        self.shard_reader = MultiShardReader(base_gcs_path)
        
        logger.info("Initialized DVID virtual adapter",
                   base_path=base_gcs_path,
                   volume_shape=volume_shape,
                   chunk_shape=chunk_shape)
    
    def _read_chunk_function(self, output_array: ts.Array, read_params: ts.VirtualChunkedReadParameters) -> ts.Future[ts.TimestampedStorageGeneration]:
        """
        Read function called by TensorStore for each chunk request.
        
        This function is called by TensorStore's virtual_chunked driver whenever
        it needs to read a specific chunk of data.
        
        Args:
            output_array: TensorStore array to fill with data
            read_params: Read parameters from TensorStore
            
        Returns:
            Future containing timestamped storage generation
        """
        try:
            # Get the domain (coordinate range) for this chunk request
            domain = output_array.domain()
            
            # TensorStore gives us the chunk origin coordinates
            # domain.origin() returns (z, y, x) coordinates of the chunk
            chunk_origin = tuple(domain.origin())
            
            logger.debug("TensorStore chunk read request", 
                        chunk_origin=chunk_origin,
                        chunk_shape=tuple(domain.shape()))
            
            # Convert chunk origin to chunk coordinates (divide by chunk size)
            # Note: DVID uses (x, y, z) ordering, TensorStore uses (z, y, x)
            z_coord, y_coord, x_coord = chunk_origin
            chunk_x = x_coord // self.chunk_shape[2]
            chunk_y = y_coord // self.chunk_shape[1] 
            chunk_z = z_coord // self.chunk_shape[0]
            
            # Read the chunk data using global coordinates
            chunk_data = self.shard_reader.read_chunk_at_global_coords((chunk_x, chunk_y, chunk_z))
            
            if chunk_data is not None:
                # Fill the output array with the decompressed data
                # Note: Need to handle coordinate system conversion (ZYX vs XYZ)
                output_array[...] = chunk_data
                
                logger.debug("Successfully filled chunk", 
                           chunk_coords=(chunk_x, chunk_y, chunk_z),
                           data_shape=chunk_data.shape)
            else:
                # Chunk not found - fill with zeros
                output_array[...] = 0
                
                logger.debug("Chunk not found, filled with zeros", 
                           chunk_coords=(chunk_x, chunk_y, chunk_z))
            
            # Return success with a simple generation stamp
            return ts.Future[ts.TimestampedStorageGeneration].result(
                ts.TimestampedStorageGeneration(
                    generation=ts.StorageGeneration.from_string("dvid_chunk"),
                    time=ts.time.now()
                )
            )
            
        except Exception as e:
            logger.error("Failed to read chunk", 
                        chunk_origin=chunk_origin if 'chunk_origin' in locals() else "unknown",
                        error=str(e))
            
            # Return error future
            return ts.Future[ts.TimestampedStorageGeneration].exception(e)
    
    def create_virtual_tensorstore(self, grid_origin: Optional[Tuple[int, int, int]] = None) -> ts.TensorStore:
        """
        Create a TensorStore virtual_chunked volume.
        
        Args:
            grid_origin: Origin of the grid in global coordinates (z, y, x). 
                        If None, defaults to (0, 0, 0).
                        
        Returns:
            TensorStore virtual chunked volume
        """
        if grid_origin is None:
            grid_origin = (0, 0, 0)
        
        logger.info("Creating virtual TensorStore volume",
                   volume_shape=self.volume_shape,
                   chunk_shape=self.chunk_shape,
                   grid_origin=grid_origin)
        
        # Create the virtual chunked spec
        spec = {
            'driver': 'virtual_chunked',
            'dtype': 'uint64',
            'domain': {
                'shape': list(self.volume_shape),
                'origin': list(grid_origin)
            },
            'chunk_layout': {
                'chunk_shape': list(self.chunk_shape)
            },
            # The read function will be added when opening
        }
        
        # Open with the read function
        store = ts.open(
            spec,
            read_function=self._read_chunk_function,
            create=True
        ).result()
        
        logger.info("Created virtual TensorStore volume", 
                   store_spec=str(store.spec()))
        
        return store
    
    def create_source_for_shard_processing(self, target_shard_path: str, 
                                         shard_shape: Tuple[int, int, int] = (2048, 2048, 2048)) -> ts.TensorStore:
        """
        Create a virtual TensorStore for processing a single shard.
        
        This is used by Cloud Run workers to create a source TensorStore that represents
        a single shard file, which can then be copied to a destination volume.
        
        Args:
            target_shard_path: Path to the specific shard to process
            shard_shape: Shape of the shard volume
            
        Returns:
            TensorStore representing the single shard
        """
        # Create a specialized adapter for a single shard
        single_shard_adapter = SingleShardAdapter(target_shard_path, shard_shape, self.chunk_shape)
        
        spec = {
            'driver': 'virtual_chunked',
            'dtype': 'uint64',
            'domain': {'shape': list(shard_shape)},
            'chunk_layout': {'chunk_shape': list(self.chunk_shape)}
        }
        
        store = ts.open(
            spec,
            read_function=single_shard_adapter._read_chunk_function,
            create=True
        ).result()
        
        logger.info("Created single-shard TensorStore", 
                   shard_path=target_shard_path,
                   shard_shape=shard_shape)
        
        return store


class SingleShardAdapter:
    """
    Adapter for processing a single DVID shard file.
    
    This is used by Cloud Run workers to create TensorStore sources for individual shards.
    """
    
    def __init__(self, shard_gcs_path: str, shard_shape: Tuple[int, int, int], 
                 chunk_shape: Tuple[int, int, int] = (64, 64, 64)):
        """
        Initialize single shard adapter.
        
        Args:
            shard_gcs_path: GCS path to the specific shard Arrow file
            shard_shape: Shape of this shard (typically 2048x2048x2048)
            chunk_shape: Chunk size (typically 64x64x64)
        """
        from .shard_reader import ShardReader
        
        self.shard_gcs_path = shard_gcs_path
        self.shard_shape = shard_shape
        self.chunk_shape = chunk_shape
        
        # Initialize single shard reader
        self.shard_reader = ShardReader(shard_gcs_path)
        
        logger.info("Initialized single shard adapter", path=shard_gcs_path)
    
    def _read_chunk_function(self, output_array: ts.Array, read_params: ts.VirtualChunkedReadParameters) -> ts.Future[ts.TimestampedStorageGeneration]:
        """Read function for a single shard."""
        try:
            domain = output_array.domain()
            chunk_origin = tuple(domain.origin())
            
            # Convert TensorStore coordinates to chunk coordinates
            z_coord, y_coord, x_coord = chunk_origin
            chunk_x = x_coord // self.chunk_shape[2]
            chunk_y = y_coord // self.chunk_shape[1]
            chunk_z = z_coord // self.chunk_shape[0]
            
            # Read chunk - coordinates are relative to the shard origin
            # which will be handled by the shard reader's global coordinate system
            chunk_data = self.shard_reader.read_chunk_at_global_coords((chunk_x, chunk_y, chunk_z))
            
            if chunk_data is not None:
                output_array[...] = chunk_data
            else:
                output_array[...] = 0
            
            return ts.Future[ts.TimestampedStorageGeneration].result(
                ts.TimestampedStorageGeneration(
                    generation=ts.StorageGeneration.from_string("dvid_shard"),
                    time=ts.time.now()
                )
            )
            
        except Exception as e:
            logger.error("Failed to read shard chunk", error=str(e))
            return ts.Future[ts.TimestampedStorageGeneration].exception(e)


def create_neuroglancer_destination(bucket: str, path: str,
                                    ng_spec: Optional[Dict[str, Any]] = None,
                                    volume_shape: Optional[Tuple[int, int, int]] = None,
                                    resolution: Tuple[int, int, int] = (8, 8, 8),
                                    chunk_shape: Tuple[int, int, int] = (64, 64, 64)) -> ts.TensorStore:
    """
    Create a Neuroglancer precomputed destination volume on GCS.

    If ng_spec is provided (the neuroglancer multiscale volume JSON used by
    DVID export-shards), the info file is written with the full sharding
    configuration so the output volume matches the shard partitioning exactly.
    Otherwise falls back to a simple unsharded single-scale spec.

    Args:
        bucket: GCS bucket name
        path: Path within bucket
        ng_spec: Full neuroglancer multiscale volume spec dict (optional)
        volume_shape: Total volume shape (x, y, z) — used only if ng_spec is None
        resolution: Voxel resolution in nm — used only if ng_spec is None
        chunk_shape: Chunk size — used only if ng_spec is None

    Returns:
        TensorStore for Neuroglancer precomputed volume (scale 0)
    """
    import json as _json
    from google.cloud import storage as gcs

    if ng_spec and ng_spec.get("scales"):
        # Write the full info file from the ng spec so sharding params are correct.
        info = dict(ng_spec)  # shallow copy
        # TensorStore expects the info file to live at <path>/info
        client = gcs.Client()
        info_blob = client.bucket(bucket).blob(f"{path}/info")
        if not info_blob.exists():
            info_blob.upload_from_string(
                _json.dumps(info, indent=2),
                content_type="application/json",
            )
            logger.info("Wrote neuroglancer info file",
                        bucket=bucket, path=f"{path}/info",
                        num_scales=len(info["scales"]))

        spec = {
            'driver': 'neuroglancer_precomputed',
            'kvstore': {
                'driver': 'gcs',
                'bucket': bucket,
                'path': path,
            },
            'scale_index': 0,
            'open': True,
        }
    else:
        # Fallback: minimal single-scale spec without sharding
        if volume_shape is None:
            raise ValueError("Either ng_spec or volume_shape must be provided")
        spec = {
            'driver': 'neuroglancer_precomputed',
            'kvstore': {
                'driver': 'gcs',
                'bucket': bucket,
                'path': path,
            },
            'multiscale_metadata': {
                'data_type': 'uint64',
                'num_channels': 1,
                'type': 'segmentation',
            },
            'scale_metadata': {
                'key': f'{volume_shape[0]}_{volume_shape[1]}_{volume_shape[2]}',
                'size': list(volume_shape),
                'resolution': list(resolution),
                'encoding': 'raw',
            },
            'scale_index': 0,
            'create': True,
            'delete_existing': False,
        }

    store = ts.open(spec).result()

    logger.info("Opened Neuroglancer destination",
                bucket=bucket, path=path,
                domain=str(store.domain))

    return store