"""
Cloud Run worker for distributed DVID shard processing.

Reads DVID Arrow IPC shard files (produced by export-shards) from GCS,
decompresses each chunk using the BRAID library, and writes the raw uint64
label data to a neuroglancer precomputed volume on GCS.

Workers self-coordinate by scanning a source prefix for available shard files.
Each worker processes shards one at a time until the time limit is reached.
"""

import base64
import json
import os
import time
import asyncio
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import structlog
import tensorstore as ts
from google.cloud import storage
from pydantic import BaseModel

from braid import ShardReader, LabelType

logger = structlog.get_logger()

CHUNK_VOXELS = 64  # voxels per chunk dimension


class WorkerConfig(BaseModel):
    """Configuration for the Cloud Run worker."""

    # GCS configuration
    source_bucket: str
    source_prefix: str  # e.g., "dvid-exports/mCNS-98d699/segmentation"
    dest_bucket: str
    dest_path: str  # e.g., "v1.0/segmentation/precomputed"

    # Neuroglancer multiscale volume spec (same JSON used for DVID export-shards)
    ng_spec: Dict[str, Any] = {}

    # Which scales to process (e.g., [0, 1])
    scales: List[int] = [0]

    # Scales to generate by downsampling the previous scale's data from the
    # destination volume rather than reading DVID export shards.  Useful for
    # scales not materialized in DVID (e.g., MaxDownresLevel was set lower
    # than the number of scales in the spec).
    downres_scales: List[int] = []

    # Worker behavior
    max_processing_time_minutes: int = 55
    polling_interval_seconds: int = 10


class ShardProcessor:
    """Processes DVID Arrow shard files into neuroglancer precomputed chunks."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.storage_client = storage.Client()
        self.source_bucket_obj = self.storage_client.bucket(config.source_bucket)
        self._dest_stores: Dict[int, ts.TensorStore] = {}

        logger.info("Initialized shard processor",
                     source=f"gs://{config.source_bucket}/{config.source_prefix}",
                     dest=f"gs://{config.dest_bucket}/{config.dest_path}",
                     scales=config.scales)

    def _open_dest_scale(self, scale: int) -> ts.TensorStore:
        """Open the destination neuroglancer precomputed volume at a given scale.

        The info file must already exist (created by setup_destination.py).
        """
        if scale in self._dest_stores:
            return self._dest_stores[scale]

        spec = {
            "driver": "neuroglancer_precomputed",
            "kvstore": {
                "driver": "gcs",
                "bucket": self.config.dest_bucket,
                "path": self.config.dest_path,
            },
            "scale_index": scale,
            "open": True,
        }
        store = ts.open(spec).result()
        self._dest_stores[scale] = store
        logger.info("Opened destination scale", scale=scale, domain=str(store.domain))
        return store

    def find_available_shard(self) -> Optional[Tuple[int, str]]:
        """Find an available shard to process across configured scales.

        Returns:
            (scale, shard_name) tuple, or None if no work available.
            shard_name is the stem (e.g., "0_0_0").
        """
        for scale in self.config.scales:
            prefix = f"{self.config.source_prefix}/s{scale}/"
            blobs = self.source_bucket_obj.list_blobs(prefix=prefix)

            for blob in blobs:
                if not blob.name.endswith(".arrow"):
                    continue

                shard_name = Path(blob.name).stem
                logger.debug("Found shard", scale=scale, shard=shard_name)
                return (scale, shard_name)

        return None

    def process_shard(self, scale: int, shard_name: str) -> bool:
        """Process a single shard: read from GCS, decompress, write to destination.

        BRAID's ShardReader reads directly from GCS via PyArrow's native
        filesystem — no temp files or intermediate copies.

        Args:
            scale: Scale level (0, 1, 2, ...)
            shard_name: Shard origin name (e.g., "30720_24576_28672")

        Returns:
            True if successful.
        """
        try:
            logger.info("Processing shard", scale=scale, shard=shard_name)

            dest = self._open_dest_scale(scale)

            arrow_uri = f"gs://{self.config.source_bucket}/{self.config.source_prefix}/s{scale}/{shard_name}.arrow"
            csv_uri = f"gs://{self.config.source_bucket}/{self.config.source_prefix}/s{scale}/{shard_name}.csv"
            reader = ShardReader(arrow_uri, csv_uri)

            logger.info("Shard loaded",
                         shard=shard_name,
                         chunks=reader.chunk_count)

            for i, (cx, cy, cz) in enumerate(reader.available_chunks):
                chunk_data = reader.read_chunk(cx, cy, cz, label_type=LabelType.LABELS)

                # BRAID outputs ZYX order; neuroglancer precomputed is XYZ + channel
                transposed = chunk_data.transpose(2, 1, 0)

                # Write to the destination at the chunk's global voxel coordinates
                x0 = cx * CHUNK_VOXELS
                y0 = cy * CHUNK_VOXELS
                z0 = cz * CHUNK_VOXELS
                dest[x0:x0 + CHUNK_VOXELS,
                     y0:y0 + CHUNK_VOXELS,
                     z0:z0 + CHUNK_VOXELS,
                     0].write(transposed).result()

                if (i + 1) % 1000 == 0:
                    logger.info("Progress",
                                 shard=shard_name,
                                 chunks_written=i + 1,
                                 total=reader.chunk_count)

            logger.info("Shard complete",
                         shard=shard_name,
                         chunks_written=reader.chunk_count)
            return True

        except Exception as e:
            logger.error("Failed to process shard",
                          scale=scale, shard=shard_name, error=str(e))
            return False

    def downres_scale(self, scale: int) -> bool:
        """Generate a scale by downsampling the previous scale from the destination.

        Reads scale N-1 from the neuroglancer precomputed volume on GCS,
        downsamples 2× in each dimension using majority vote, and writes
        scale N.  Requires scale N-1 to be fully written.

        Args:
            scale: Scale level to generate (must be >= 1)

        Returns:
            True if successful.
        """
        if scale < 1:
            logger.error("Cannot downres scale 0 — no previous scale exists")
            return False

        try:
            logger.info("Generating scale by downres", scale=scale, source_scale=scale - 1)

            source = self._open_dest_scale(scale - 1)
            dest = self._open_dest_scale(scale)

            # Create a downsampled virtual view of the source scale.
            # 'mode' = majority vote, correct for uint64 segmentation labels.
            # The channel dimension (last) is not downsampled.
            downsampled = ts.downsample(source, [2, 2, 2, 1], "mode")

            # Copy the downsampled view into the destination scale.
            # TensorStore handles chunk-aligned writes internally.
            logger.info("Starting downres copy",
                         source_domain=str(source.domain),
                         dest_domain=str(dest.domain))

            ts.copy(downsampled, dest).result()

            logger.info("Downres complete", scale=scale)
            return True

        except Exception as e:
            logger.error("Failed to downres scale",
                          scale=scale, error=str(e))
            return False


class CloudRunWorker:
    """Main Cloud Run worker that finds and processes shards until time limit."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.processor = ShardProcessor(config)
        self.start_time = time.time()

    def _should_continue(self) -> bool:
        elapsed_minutes = (time.time() - self.start_time) / 60
        return elapsed_minutes < self.config.max_processing_time_minutes

    async def run(self):
        logger.info("Starting worker",
                     scales=self.config.scales,
                     downres_scales=self.config.downres_scales)

        processed_count = 0
        failed_count = 0

        # Phase 1: Process DVID export shards
        while self._should_continue():
            try:
                result = self.processor.find_available_shard()

                if result is None:
                    if self.config.downres_scales:
                        break  # Move on to downres phase
                    logger.debug("No work available, sleeping")
                    await asyncio.sleep(self.config.polling_interval_seconds)
                    continue

                scale, shard_name = result
                success = self.processor.process_shard(scale, shard_name)

                if success:
                    processed_count += 1
                    logger.info("Shard processed successfully",
                                 shard=shard_name, scale=scale,
                                 total_processed=processed_count)
                else:
                    failed_count += 1
                    logger.warning("Shard processing failed",
                                    shard=shard_name, scale=scale,
                                    total_failed=failed_count)

            except Exception as e:
                logger.error("Unexpected error in worker loop", error=str(e))
                await asyncio.sleep(self.config.polling_interval_seconds)

        # Phase 2: Generate downres scales from previous scale data
        for scale in sorted(self.config.downres_scales):
            if not self._should_continue():
                logger.warning("Time limit reached, skipping downres", scale=scale)
                break
            if self.processor.downres_scale(scale):
                processed_count += 1
            else:
                failed_count += 1

        elapsed = (time.time() - self.start_time) / 60
        logger.info("Worker finished",
                     elapsed_minutes=round(elapsed, 1),
                     processed=processed_count,
                     failed=failed_count)


def create_config_from_env() -> WorkerConfig:
    """Create worker configuration from environment variables.

    The NG_SPEC env var is a base64-encoded neuroglancer multiscale volume
    spec JSON.  SCALES is a comma-separated list of scale indices.
    """
    ng_spec_b64 = os.environ.get("NG_SPEC", "")
    ng_spec = json.loads(base64.b64decode(ng_spec_b64)) if ng_spec_b64 else {}

    scales_str = os.environ.get("SCALES", "0")
    scales = [int(s.strip()) for s in scales_str.split(",")]

    downres_str = os.environ.get("DOWNRES_SCALES", "")
    downres_scales = [int(s.strip()) for s in downres_str.split(",") if s.strip()]

    return WorkerConfig(
        source_bucket=os.environ["SOURCE_BUCKET"],
        source_prefix=os.environ["SOURCE_PREFIX"],
        dest_bucket=os.environ["DEST_BUCKET"],
        dest_path=os.environ["DEST_PATH"],
        ng_spec=ng_spec,
        scales=scales,
        downres_scales=downres_scales,
        max_processing_time_minutes=int(os.environ.get("MAX_PROCESSING_TIME", "55")),
        polling_interval_seconds=int(os.environ.get("POLLING_INTERVAL", "10")),
    )


async def main():
    """Main entry point for the Cloud Run worker."""
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
            structlog.processors.JSONRenderer(),
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
