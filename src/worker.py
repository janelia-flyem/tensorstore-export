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
from pathlib import Path
from typing import Tuple, Dict, Any, List

import structlog
import tensorstore as ts
from google.cloud import storage
from pydantic import BaseModel

from braid import ShardReader, LabelType

logger = structlog.get_logger()

CHUNK_VOXELS = 64  # voxels per chunk dimension


def _parse_gs_uri(uri: str) -> Tuple[str, str]:
    """Split gs://bucket/path into (bucket, path)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    rest = uri[len("gs://"):]
    bucket, _, path = rest.partition("/")
    return bucket, path.rstrip("/")


class WorkerConfig(BaseModel):
    """Configuration for the Cloud Run worker."""

    # GCS URIs (e.g., "gs://bucket/path/to/shards", "gs://bucket/path/to/output")
    source_path: str  # contains s0/, s1/, ... with .arrow + .csv pairs
    dest_path: str    # neuroglancer precomputed volume root

    # Neuroglancer multiscale volume spec (same JSON used for DVID export-shards)
    ng_spec: Dict[str, Any] = {}

    # Which scales to process (e.g., [0, 1])
    scales: List[int] = [0]

    # Scales to generate by downsampling the previous scale's data from the
    # destination volume rather than reading DVID export shards.  Useful for
    # scales not materialized in DVID (e.g., MaxDownresLevel was set lower
    # than the number of scales in the spec).
    downres_scales: List[int] = []

    # Label type: "labels" for agglomerated labels (default), "supervoxels"
    # for raw supervoxel IDs.  Agglomerated labels are the standard
    # "segmentation" view where proofreading merges are applied.
    label_type: str = "labels"

    # Worker behavior
    max_processing_time_minutes: int = 55
    polling_interval_seconds: int = 10


class ShardProcessor:
    """Processes DVID Arrow shard files into neuroglancer precomputed chunks."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self._source_bucket, self._source_prefix = _parse_gs_uri(config.source_path)
        self._dest_bucket, self._dest_prefix = _parse_gs_uri(config.dest_path)
        self.storage_client = storage.Client()
        self.source_bucket_obj = self.storage_client.bucket(self._source_bucket)
        self._dest_stores: Dict[int, ts.TensorStore] = {}

        # Cloud Run task index for work partitioning.
        # Each task processes shards where index % task_count == task_index.
        self._task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
        self._task_count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1"))

        logger.info("Initialized shard processor",
                     source=config.source_path,
                     dest=config.dest_path,
                     scales=config.scales,
                     task_index=self._task_index,
                     task_count=self._task_count)

    def _open_dest_scale(self, scale: int) -> ts.TensorStore:
        """Open the destination neuroglancer precomputed volume at a given scale.

        The info file must already exist (created by setup_destination.py).
        """
        if scale in self._dest_stores:
            return self._dest_stores[scale]

        spec = {
            "driver": "neuroglancer_precomputed",
            "kvstore": self.config.dest_path,
            "scale_index": scale,
            "open": True,
        }
        store = ts.open(spec).result()
        self._dest_stores[scale] = store
        logger.info("Opened destination scale", scale=scale, domain=str(store.domain))
        return store

    def list_my_shards(self) -> List[Tuple[int, str]]:
        """List all shards assigned to this task.

        Each Cloud Run task gets a unique CLOUD_RUN_TASK_INDEX (0 to N-1).
        Shards are assigned round-robin: task i processes shards where
        shard_index % task_count == task_index.  This ensures no two tasks
        process the same shard.

        Returns:
            List of (scale, shard_name) tuples assigned to this task.
        """
        all_shards = []
        for scale in self.config.scales:
            prefix = f"{self._source_prefix}/s{scale}/"
            blobs = self.source_bucket_obj.list_blobs(prefix=prefix)
            for blob in blobs:
                if blob.name.endswith(".arrow"):
                    shard_name = Path(blob.name).stem
                    all_shards.append((scale, shard_name))

        # Partition by task index
        my_shards = [s for i, s in enumerate(all_shards)
                     if i % self._task_count == self._task_index]

        logger.info("Shard assignment",
                     total_shards=len(all_shards),
                     my_shards=len(my_shards),
                     task_index=self._task_index,
                     task_count=self._task_count)
        return my_shards

    def _open_shard_with_retry(self, arrow_uri: str, csv_uri: str,
                               max_retries: int = 5) -> ShardReader:
        """Open a shard with exponential backoff for transient GCS errors."""
        for attempt in range(max_retries):
            try:
                return ShardReader(arrow_uri, csv_uri)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                logger.warning("Shard read failed, retrying",
                                arrow=arrow_uri, attempt=attempt + 1,
                                wait_seconds=wait, error=str(e)[:200])
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def process_shard(self, scale: int, shard_name: str) -> bool:
        """Process a single shard: read from GCS, decompress, write to destination.

        Downloads Arrow+CSV via google-cloud-storage, decompresses each chunk
        using BRAID, and writes to the neuroglancer precomputed volume.
        Individual chunk failures are logged and skipped — the shard is still
        considered successful if at least some chunks were written.

        Args:
            scale: Scale level (0, 1, 2, ...)
            shard_name: Shard origin name (e.g., "30720_24576_28672")

        Returns:
            True if the shard was processed (even with some chunk errors).
            False only if the shard could not be loaded at all.
        """
        try:
            logger.info("Processing shard", scale=scale, shard=shard_name)

            dest = self._open_dest_scale(scale)

            arrow_uri = f"{self.config.source_path}/s{scale}/{shard_name}.arrow"
            csv_uri = f"{self.config.source_path}/s{scale}/{shard_name}.csv"
            reader = self._open_shard_with_retry(arrow_uri, csv_uri)

            logger.info("Shard loaded",
                         shard=shard_name,
                         chunks=reader.chunk_count)

            lt = LabelType(self.config.label_type)
            chunks_written = 0
            chunks_failed = 0

            for i, (cx, cy, cz) in enumerate(reader.available_chunks):
                try:
                    chunk_data = reader.read_chunk(cx, cy, cz, label_type=lt)

                    # BRAID outputs ZYX order; neuroglancer precomputed is XYZ + channel
                    transposed = chunk_data.transpose(2, 1, 0)

                    # Write to the destination at the chunk's global voxel coordinates.
                    # Clip to volume bounds for boundary chunks where the volume
                    # size isn't a multiple of the chunk size.
                    x0 = cx * CHUNK_VOXELS
                    y0 = cy * CHUNK_VOXELS
                    z0 = cz * CHUNK_VOXELS
                    x1 = min(x0 + CHUNK_VOXELS, dest.shape[0])
                    y1 = min(y0 + CHUNK_VOXELS, dest.shape[1])
                    z1 = min(z0 + CHUNK_VOXELS, dest.shape[2])
                    dest[x0:x1, y0:y1, z0:z1, 0].write(
                        transposed[:x1 - x0, :y1 - y0, :z1 - z0]
                    ).result()
                    chunks_written += 1

                except Exception as e:
                    chunks_failed += 1
                    # Searchable event name for pixi run export-errors
                    logger.error("Chunk failed",
                                  scale=scale, shard=shard_name,
                                  chunk_x=cx, chunk_y=cy, chunk_z=cz,
                                  error=str(e)[:500])

                if (i + 1) % 1000 == 0:
                    logger.info("Progress",
                                 shard=shard_name,
                                 chunks_written=chunks_written,
                                 chunks_failed=chunks_failed,
                                 total=reader.chunk_count)

            logger.info("Shard complete",
                         shard=shard_name,
                         chunks_written=chunks_written,
                         chunks_failed=chunks_failed,
                         total=reader.chunk_count)
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

        # Phase 1: Process assigned DVID export shards
        my_shards = self.processor.list_my_shards()

        for scale, shard_name in my_shards:
            if not self._should_continue():
                logger.warning("Time limit reached, stopping",
                                processed=processed_count, remaining=len(my_shards) - processed_count - failed_count)
                break

            try:
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
                failed_count += 1
                logger.error("Unexpected error processing shard",
                              shard=shard_name, scale=scale, error=str(e))

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
        source_path=os.environ["SOURCE_PATH"],
        dest_path=os.environ["DEST_PATH"],
        ng_spec=ng_spec,
        scales=scales,
        downres_scales=downres_scales,
        label_type=os.environ.get("LABEL_TYPE", "labels"),
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
