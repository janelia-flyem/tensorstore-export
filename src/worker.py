"""
Cloud Run worker for distributed DVID shard processing.

This module implements the main worker logic for processing DVID export shards
and writing them to a Neuroglancer precomputed volume, using the transactional
file movement strategy described in ShardExportDesign.md.
"""

import base64
import json
import os
import time
import asyncio
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import structlog
import tensorstore as ts
from google.cloud import storage
from pydantic import BaseModel
from .tensorstore_adapter import SingleShardAdapter, create_neuroglancer_destination
from .shard_reader import ShardReader

logger = structlog.get_logger()


class WorkerConfig(BaseModel):
    """Configuration for the Cloud Run worker."""

    # GCS configuration
    source_bucket: str
    dest_bucket: str
    dest_path: str

    # Neuroglancer multiscale volume spec (the same JSON used for DVID export-shards).
    # This is the single source of truth for volume geometry and sharding parameters.
    ng_spec: Dict[str, Any]

    # Volume configuration derived from ng_spec scale 0
    total_volume_shape: Tuple[int, int, int]  # (x, y, z) from spec
    chunk_shape: Tuple[int, int, int] = (64, 64, 64)
    resolution: Tuple[int, int, int] = (8, 8, 8)  # nm

    # Worker behavior
    max_processing_time_minutes: int = 55  # Cloud Run timeout is 60min
    polling_interval_seconds: int = 10
    max_retries: int = 3
    
    # GCS paths for job management
    unprocessed_prefix: str = "unprocessed/"
    processing_prefix: str = "processing/"
    finished_prefix: str = "finished/"
    failed_prefix: str = "failed/"


class ShardProcessor:
    """
    Main processor class that handles the lifecycle of processing a single shard.
    """
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.storage_client = storage.Client()
        self.source_bucket = self.storage_client.bucket(config.source_bucket)
        
        logger.info("Initialized shard processor", config=config.dict())
    
    def find_available_shard(self) -> Optional[str]:
        """
        Find an available shard to process.
        
        Uses atomic GCS move operations to claim a shard for processing,
        implementing optimistic locking as described in ShardExportDesign.md.
        
        Returns:
            Name of claimed shard file (without .arrow extension) or None
        """
        logger.debug("Looking for available shards", 
                    prefix=self.config.unprocessed_prefix)
        
        # List files in unprocessed directory
        blobs = self.source_bucket.list_blobs(prefix=self.config.unprocessed_prefix)
        
        for blob in blobs:
            if not blob.name.endswith('.arrow'):
                continue
                
            # Extract shard name (e.g., "0_0_0" from "unprocessed/0_0_0.arrow")
            shard_name = Path(blob.name).stem
            arrow_src = blob.name
            csv_src = arrow_src.replace('.arrow', '.csv')
            
            # Target paths in processing directory
            arrow_dest = f"{self.config.processing_prefix}{shard_name}.arrow"
            csv_dest = f"{self.config.processing_prefix}{shard_name}.csv"
            
            try:
                # Atomic move operation - this is our lock
                logger.debug("Attempting to claim shard", shard=shard_name)
                
                # Move both files atomically
                self._move_blob(arrow_src, arrow_dest)
                self._move_blob(csv_src, csv_dest)
                
                logger.info("Successfully claimed shard", shard=shard_name)
                return shard_name
                
            except Exception as e:
                # Another worker claimed it first, or files are missing
                logger.debug("Failed to claim shard", shard=shard_name, error=str(e))
                continue
        
        logger.debug("No available shards found")
        return None
    
    def _move_blob(self, source_name: str, dest_name: str):
        """
        Atomically move a blob within the same bucket.
        
        Args:
            source_name: Source blob name
            dest_name: Destination blob name
        """
        source_blob = self.source_bucket.blob(source_name)
        dest_blob = self.source_bucket.blob(dest_name)
        
        # Copy to destination
        dest_blob.rewrite(source_blob)
        
        # Delete source (making the move atomic)
        source_blob.delete()
    
    def _extract_shard_origin_from_name(self, shard_name: str) -> Tuple[int, int, int]:
        """
        Extract shard origin coordinates from filename.
        
        Args:
            shard_name: Shard name like "0_0_0" or "2048_4096_0"
            
        Returns:
            Origin coordinates (x, y, z)
        """
        parts = shard_name.split('_')
        if len(parts) != 3:
            raise ValueError(f"Invalid shard name format: {shard_name}")
        
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    
    def process_shard(self, shard_name: str) -> bool:
        """
        Process a single shard file.
        
        Args:
            shard_name: Name of the shard to process (e.g., "0_0_0")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting shard processing", shard=shard_name)
            
            # Construct GCS paths for the shard files
            arrow_path = f"gs://{self.config.source_bucket}/{self.config.processing_prefix}{shard_name}.arrow"
            
            # Extract shard origin from filename
            shard_origin = self._extract_shard_origin_from_name(shard_name)
            
            logger.info("Processing shard", 
                       shard=shard_name,
                       arrow_path=arrow_path,
                       shard_origin=shard_origin)
            
            # Create virtual source TensorStore for this shard
            shard_adapter = SingleShardAdapter(arrow_path, self.config.shard_shape, self.config.chunk_shape)
            
            source_spec = {
                'driver': 'virtual_chunked',
                'dtype': 'uint64',
                'domain': {'shape': list(self.config.shard_shape)},
                'chunk_layout': {'chunk_shape': list(self.config.chunk_shape)}
            }
            
            source_store = ts.open(
                source_spec,
                read_function=shard_adapter._read_chunk_function,
                create=True
            ).result()
            
            # Create destination TensorStore (the global Neuroglancer volume)
            dest_store = create_neuroglancer_destination(
                bucket=self.config.dest_bucket,
                path=self.config.dest_path,
                volume_shape=self.config.total_volume_shape,
                resolution=self.config.resolution,
                chunk_shape=self.config.chunk_shape
            )
            
            # Calculate the slice where this shard should be written
            # Shard origin gives us the global coordinates where this shard starts
            x_origin, y_origin, z_origin = shard_origin
            x_end = x_origin + self.config.shard_shape[2]  # shard_shape is (z,y,x)
            y_end = y_origin + self.config.shard_shape[1]
            z_end = z_origin + self.config.shard_shape[0]
            
            # Create the slice for the destination
            dest_slice = dest_store[z_origin:z_end, y_origin:y_end, x_origin:x_end]
            
            logger.info("Starting shard copy",
                       shard=shard_name,
                       source_shape=self.config.shard_shape,
                       dest_slice=f"[{z_origin}:{z_end}, {y_origin}:{y_end}, {x_origin}:{x_end}]")
            
            # Execute the copy operation
            copy_future = ts.copy(source_store, dest_slice)
            copy_result = copy_future.result()
            
            logger.info("Shard copy completed successfully", 
                       shard=shard_name,
                       copy_result=str(copy_result))
            
            return True
            
        except Exception as e:
            logger.error("Failed to process shard", 
                        shard=shard_name,
                        error=str(e))
            return False
    
    def mark_shard_finished(self, shard_name: str):
        """Move shard files to finished directory."""
        try:
            arrow_src = f"{self.config.processing_prefix}{shard_name}.arrow"
            csv_src = f"{self.config.processing_prefix}{shard_name}.csv"
            arrow_dest = f"{self.config.finished_prefix}{shard_name}.arrow"
            csv_dest = f"{self.config.finished_prefix}{shard_name}.csv"
            
            self._move_blob(arrow_src, arrow_dest)
            self._move_blob(csv_src, csv_dest)
            
            logger.info("Marked shard as finished", shard=shard_name)
            
        except Exception as e:
            logger.error("Failed to mark shard as finished", 
                        shard=shard_name, error=str(e))
    
    def mark_shard_failed(self, shard_name: str, error_msg: str):
        """Move shard files to failed directory with error info."""
        try:
            arrow_src = f"{self.config.processing_prefix}{shard_name}.arrow"
            csv_src = f"{self.config.processing_prefix}{shard_name}.csv"
            arrow_dest = f"{self.config.failed_prefix}{shard_name}.arrow"
            csv_dest = f"{self.config.failed_prefix}{shard_name}.csv"
            
            self._move_blob(arrow_src, arrow_dest)
            self._move_blob(csv_src, csv_dest)
            
            # Also create an error log file
            error_blob = self.source_bucket.blob(f"{self.config.failed_prefix}{shard_name}.error")
            error_blob.upload_from_string(error_msg)
            
            logger.info("Marked shard as failed", shard=shard_name)
            
        except Exception as e:
            logger.error("Failed to mark shard as failed", 
                        shard=shard_name, error=str(e))


class CloudRunWorker:
    """
    Main Cloud Run worker that orchestrates the shard processing workflow.
    """
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.processor = ShardProcessor(config)
        self.start_time = time.time()
        
        logger.info("Initialized Cloud Run worker")
    
    def _should_continue(self) -> bool:
        """Check if worker should continue processing (within time limit)."""
        elapsed_minutes = (time.time() - self.start_time) / 60
        return elapsed_minutes < self.config.max_processing_time_minutes
    
    async def run(self):
        """
        Main worker loop - finds and processes shards until time limit.
        """
        logger.info("Starting worker main loop")
        
        processed_count = 0
        failed_count = 0
        
        while self._should_continue():
            try:
                # Find an available shard
                shard_name = self.processor.find_available_shard()
                
                if shard_name is None:
                    # No work available - wait and check again
                    logger.debug("No work available, sleeping")
                    await asyncio.sleep(self.config.polling_interval_seconds)
                    continue
                
                # Process the shard
                success = self.processor.process_shard(shard_name)
                
                if success:
                    self.processor.mark_shard_finished(shard_name)
                    processed_count += 1
                    logger.info("Successfully processed shard", 
                               shard=shard_name,
                               total_processed=processed_count)
                else:
                    self.processor.mark_shard_failed(shard_name, "Processing failed")
                    failed_count += 1
                    logger.warning("Failed to process shard", 
                                  shard=shard_name,
                                  total_failed=failed_count)
                
            except Exception as e:
                logger.error("Unexpected error in worker loop", error=str(e))
                await asyncio.sleep(self.config.polling_interval_seconds)
        
        elapsed_minutes = (time.time() - self.start_time) / 60
        logger.info("Worker loop completed", 
                   elapsed_minutes=elapsed_minutes,
                   processed_count=processed_count,
                   failed_count=failed_count)


# Environment variable configuration
def create_config_from_env() -> WorkerConfig:
    """Create worker configuration from environment variables.

    The NG_SPEC env var is a base64-encoded neuroglancer multiscale volume
    spec JSON — the same file used for DVID's export-shards command.  Volume
    geometry and sharding parameters are derived from it.
    """
    ng_spec_b64 = os.environ.get("NG_SPEC")
    if not ng_spec_b64:
        # Fall back to flat env vars for backwards compatibility
        volume_shape = tuple(map(int, os.environ["VOLUME_SHAPE"].split(",")))
        ng_spec = {}
    else:
        ng_spec = json.loads(base64.b64decode(ng_spec_b64))
        scale0 = ng_spec["scales"][0]
        volume_shape = tuple(int(v) for v in scale0["size"])

    chunk_shape = tuple(map(int, os.environ.get("CHUNK_SHAPE", "64,64,64").split(",")))
    resolution = tuple(map(int, os.environ.get("RESOLUTION", "8,8,8").split(",")))

    # If we have the spec, override chunk_shape and resolution from scale 0
    if ng_spec and "scales" in ng_spec:
        scale0 = ng_spec["scales"][0]
        if scale0.get("chunk_sizes"):
            chunk_shape = tuple(int(v) for v in scale0["chunk_sizes"][0])
        if scale0.get("resolution"):
            resolution = tuple(int(v) for v in scale0["resolution"])

    return WorkerConfig(
        source_bucket=os.environ["SOURCE_BUCKET"],
        dest_bucket=os.environ["DEST_BUCKET"],
        dest_path=os.environ["DEST_PATH"],
        ng_spec=ng_spec,
        total_volume_shape=volume_shape,
        chunk_shape=chunk_shape,
        resolution=resolution,
        max_processing_time_minutes=int(os.environ.get("MAX_PROCESSING_TIME", "55")),
        polling_interval_seconds=int(os.environ.get("POLLING_INTERVAL", "10")),
    )


async def main():
    """Main entry point for the Cloud Run worker."""
    # Setup structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    try:
        config = create_config_from_env()
        worker = CloudRunWorker(config)
        await worker.run()
        
    except Exception as e:
        logger.error("Worker failed to start", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())