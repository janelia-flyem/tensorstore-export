"""
Cloud Run worker for distributed DVID shard processing.

Reads DVID Arrow IPC shard files (produced by export-shards) from GCS,
decompresses each chunk using the BRAID library, and writes the raw uint64
label data to a neuroglancer precomputed volume on GCS.

Workers self-coordinate by scanning a source prefix for available shard files.
Each worker processes shards one at a time until the time limit is reached.
"""

import base64
import csv
import io
import json
import os
import shutil
import time
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import structlog
import tensorstore as ts
from google.cloud import storage
from pydantic import BaseModel

from braid import ShardReader, LabelType
from src.ng_sharding import shard_chunk_coords, load_ng_spec_from_dict

logger = structlog.get_logger()

CHUNK_VOXELS = 64  # voxels per chunk dimension

# Number of chunks to write per transaction batch.  Each batch accumulates
# ~2MB/chunk in TensorStore's transaction buffer; on commit the buffer is
# flushed to the local staging disk via read-modify-write and memory is freed.
BATCH_SIZE = 100

# Default local staging directory.  On Cloud Run Gen 2 this is part of the
# in-memory container filesystem (tmpfs) — writes consume the memory budget.
# Falls back to a temp dir if unavailable.
DEFAULT_STAGING_PATH = "/mnt/staging"


def _reset_cgroup_peak() -> bool:
    """Try to reset cgroups v2 memory.peak to current usage.

    On cgroups v2 (Linux 5.19+), writing "0" to memory.peak resets the
    high-water mark to the current usage.  This enables true per-shard peak
    tracking without sampling gaps.

    Returns True if the reset succeeded, False otherwise.
    """
    try:
        with open("/sys/fs/cgroup/memory.peak", "w") as f:
            f.write("0\n")
        return True
    except (FileNotFoundError, PermissionError, OSError):
        return False


def _read_cgroup_memory() -> tuple:
    """Read current and peak memory usage from the container's cgroup (bytes).

    Returns (current_bytes, limit_bytes, peak_bytes).  Returns (0, 0, 0) if
    cgroup memory accounting is unavailable (e.g., macOS local development).

    Peak is read from memory.peak (cgroups v2) or
    memory.max_usage_in_bytes (cgroups v1).  If memory.peak was reset via
    _reset_cgroup_peak(), the value reflects the peak since last reset.
    """
    current = 0
    limit = 0
    peak = 0
    # cgroups v2 (Cloud Run Gen2), then v1 fallback
    for path in [
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ]:
        try:
            with open(path) as f:
                current = int(f.read().strip())
            break
        except (FileNotFoundError, ValueError):
            pass
    for path in [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]:
        try:
            with open(path) as f:
                content = f.read().strip()
                if content != "max":
                    limit = int(content)
            break
        except (FileNotFoundError, ValueError):
            pass
    # High-water mark since cgroup creation
    for path in [
        "/sys/fs/cgroup/memory.peak",
        "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
    ]:
        try:
            with open(path) as f:
                peak = int(f.read().strip())
            break
        except (FileNotFoundError, ValueError):
            pass
    return current, limit, peak


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

    # Local staging directory for TensorStore writes.  On Cloud Run Gen 2,
    # this is part of the in-memory container filesystem (tmpfs) — writes
    # consume the memory budget.  If the path doesn't exist, falls back to
    # a temp dir.
    staging_path: str = DEFAULT_STAGING_PATH

    # Cloud Run worker memory allocation in GiB.  Must match the --memory flag
    # passed to `gcloud run jobs create`.  Used as a fallback for the
    # transaction memory budget when cgroup memory accounting is unavailable.
    worker_memory_gib: float = 4.0

    # GCS URI to a manifest JSON file that maps CLOUD_RUN_TASK_INDEX to a
    # list of (scale, shard_name) pairs.  If set, the worker reads its
    # assignment from the manifest instead of self-partitioning via
    # list_my_shards().  Generated by `pixi run precompute-manifest`.
    manifest_uri: str = ""

    # Worker behavior — set high so tasks run to completion.
    # Cloud Run TASK_TIMEOUT is the hard limit.
    max_processing_time_minutes: int = 1440  # 24 hours
    polling_interval_seconds: int = 10


class ShardProcessor:
    """Processes DVID Arrow shard files into neuroglancer precomputed chunks."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self._source_bucket, self._source_prefix = _parse_gs_uri(config.source_path)
        self._dest_bucket, self._dest_prefix = _parse_gs_uri(config.dest_path)
        self.storage_client = storage.Client()
        self.source_bucket_obj = self.storage_client.bucket(self._source_bucket)
        self.dest_bucket_obj = self.storage_client.bucket(self._dest_bucket)
        self._dest_stores: Dict[int, ts.TensorStore] = {}

        # Cloud Run task index for work partitioning.
        # Each task processes shards where index % task_count == task_index.
        self._task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
        self._task_count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1"))

        # Set up local staging directory.  On Cloud Run Gen 2 this is an
        # in-memory filesystem (tmpfs — consumes memory).  For local dev, use a temp dir.
        staging_base = config.staging_path
        if not os.path.isdir(staging_base):
            import tempfile
            staging_base = tempfile.mkdtemp(prefix="ng-staging-")
            logger.info("Staging path not found, using temp dir",
                         configured=config.staging_path, actual=staging_base)
        self._staging_base = staging_base

        # Build the info file from the ng_spec in config.  This avoids
        # depending on the GCS info file which can be stale due to Cloud Run's
        # internal GCS caching.  We add compressed_segmentation_block_size
        # which TensorStore requires but the DVID spec doesn't include.
        ng_spec = config.ng_spec
        if ng_spec:
            for scale in ng_spec.get("scales", []):
                if scale.get("encoding") == "compressed_segmentation":
                    scale.setdefault("compressed_segmentation_block_size", [8, 8, 8])
            self._info_json = json.dumps(ng_spec)
        else:
            # Fallback: download from GCS if ng_spec wasn't provided
            self._info_json = self.dest_bucket_obj.blob(
                f"{self._dest_prefix}/info"
            ).download_as_text()
        logger.info("Info file ready",
                     length=len(self._info_json),
                     source="ng_spec" if ng_spec else "gcs",
                     has_cseg="compressed_segmentation_block_size" in self._info_json)

        logger.info("Initialized shard processor",
                     source=config.source_path,
                     dest=config.dest_path,
                     staging=self._staging_base,
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

    def load_manifest(self) -> List[Tuple[int, str]]:
        """Load this task's shard assignments from a per-task manifest on GCS.

        MANIFEST_URI is the tier directory prefix (e.g.,
        gs://bucket/exports/seg/manifests/tier-4gi).  Each task reads
        its own file: {MANIFEST_URI}/task-{CLOUD_RUN_TASK_INDEX}.json,
        which contains a JSON array of {scale, shard} objects.

        Generated by precompute_manifest.py.

        Returns:
            List of (scale, shard_name) tuples for this task.
        """
        prefix = self.config.manifest_uri.rstrip("/")
        task_uri = f"{prefix}/task-{self._task_index}.json"
        bucket_name, blob_path = _parse_gs_uri(task_uri)
        blob = self.storage_client.bucket(bucket_name).blob(blob_path)
        entries = json.loads(blob.download_as_text())
        my_shards = [(e["scale"], e["shard"]) for e in entries]

        logger.info("Loaded manifest",
                     manifest_uri=task_uri,
                     task_index=self._task_index,
                     assigned_shards=len(my_shards))
        return my_shards

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

    def upload_staging_dir(self, staging_dir: str) -> int:
        """Upload all shard files from a staging directory to GCS.

        Walks the staging directory and uploads every file except 'info',
        preserving the directory structure (e.g., s0/070e7.shard).
        Uses streaming upload with a small chunk size to avoid buffering
        the entire shard file in memory.

        Returns the number of bytes uploaded.
        """
        uploaded_bytes = 0
        file_count = 0
        for root, _dirs, files in os.walk(staging_dir):
            for fn in files:
                if fn == "info":
                    continue
                local_path = os.path.join(root, fn)
                rel_path = os.path.relpath(local_path, staging_dir)
                blob_path = f"{self._dest_prefix}/{rel_path}"
                blob = self.dest_bucket_obj.blob(blob_path)
                blob.chunk_size = 8 * (1 << 20)  # 8 MiB
                blob.upload_from_filename(local_path)
                size = os.path.getsize(local_path)
                uploaded_bytes += size
                file_count += 1

        if file_count == 0:
            logger.warning("No shard files in staging directory",
                           staging_dir=staging_dir)

        return uploaded_bytes

    def process_shard(self, scale: int, shard_name: str,
                      dest: ts.TensorStore) -> bool:
        """Process a single shard: read from GCS, write chunks to local volume.

        The TensorStore handle and staging directory are managed by the caller
        (CloudRunWorker.run).  This method only reads DVID data and writes
        chunks via batched transactions.

        Args:
            scale: Scale level (0, 1, 2, ...)
            shard_name: Shard origin name (e.g., "30720_24576_28672")
            dest: Pre-opened TensorStore handle for the local staging volume

        Returns:
            True if the shard was processed (even with some chunk errors).
            False only if the shard could not be loaded at all.
        """
        try:
            logger.info("Processing shard", scale=scale, shard=shard_name)

            arrow_uri = f"{self.config.source_path}/s{scale}/{shard_name}.arrow"
            csv_uri = f"{self.config.source_path}/s{scale}/{shard_name}.csv"
            reader = self._open_shard_with_retry(arrow_uri, csv_uri)

            logger.info("Shard loaded",
                         shard=shard_name,
                         chunks=reader.chunk_count)

            # Skip shards where all chunks have empty labels/supervoxels.
            # These produce all-zero data and TensorStore correctly writes
            # no .shard file for fill-value-only data.
            if reader.is_empty:
                logger.info("Shard skipped (empty)",
                             scale=scale, shard=shard_name,
                             chunks=reader.chunk_count)
                return True

            # Per-shard bounds check: scan available_chunks once to detect
            # shards that extend beyond the volume.  Diagnostic only (warn).
            vol_shape = tuple(dest.shape[:3])
            max_voxel = [0, 0, 0]
            for (cx, cy, cz) in reader.available_chunks:
                for d, c in enumerate((cx, cy, cz)):
                    v = c * CHUNK_VOXELS + CHUNK_VOXELS
                    if v > max_voxel[d]:
                        max_voxel[d] = v
            if any(mv > vs for mv, vs in zip(max_voxel, vol_shape)):
                logger.info("Shard extends beyond volume",
                               scale=scale, shard=shard_name,
                               shard_max_voxel=tuple(max_voxel),
                               volume_shape=vol_shape)

            lt = LabelType(self.config.label_type)
            chunks_written = 0
            chunks_failed = 0
            chunks_outside = 0
            shard_uncompressed_bytes = 0
            batches_committed = 0
            shard_start_time = time.time()

            # Track per-shard peak memory.  Try to reset cgroups v2
            # memory.peak for a true kernel-level per-shard high-water mark.
            # Fall back to max(memory.current) sampling if reset unavailable.
            peak_reset_ok = _reset_cgroup_peak()
            mem_at_start, _, _ = _read_cgroup_memory()
            shard_peak_mem = mem_at_start
            last_progress_time = shard_start_time
            progress_interval = 60  # seconds between progress logs

            # Batched transactions: write BATCH_SIZE chunks per transaction,
            # commit to local disk (RMW), start a new transaction.  Each
            # commit flushes TensorStore's buffer to disk, keeping RSS flat.
            txn = ts.Transaction()
            batch_chunks = 0

            for i, (cx, cy, cz) in enumerate(reader.available_chunks):
                try:
                    chunk_data = reader.read_chunk(cx, cy, cz, label_type=lt)

                    # BRAID outputs ZYX order; neuroglancer precomputed is XYZ + channel
                    transposed = chunk_data.transpose(2, 1, 0)

                    x0 = cx * CHUNK_VOXELS
                    y0 = cy * CHUNK_VOXELS
                    z0 = cz * CHUNK_VOXELS
                    x1 = min(x0 + CHUNK_VOXELS, dest.shape[0])
                    y1 = min(y0 + CHUNK_VOXELS, dest.shape[1])
                    z1 = min(z0 + CHUNK_VOXELS, dest.shape[2])
                    if x1 <= x0 or y1 <= y0 or z1 <= z0:
                        chunks_outside += 1
                        if chunks_outside == 1:
                            logger.warning("Chunk outside volume bounds",
                                           scale=scale, shard=shard_name,
                                           chunk_x=cx, chunk_y=cy, chunk_z=cz,
                                           voxel_origin=(x0, y0, z0),
                                           volume_shape=tuple(dest.shape[:3]))
                        continue
                    clipped = transposed[:x1 - x0, :y1 - y0, :z1 - z0]
                    dest.with_transaction(txn)[x0:x1, y0:y1, z0:z1, 0].write(
                        clipped
                    ).result()
                    chunks_written += 1
                    batch_chunks += 1
                    shard_uncompressed_bytes += clipped.nbytes

                except Exception as e:
                    chunks_failed += 1
                    logger.error("Chunk failed",
                                  scale=scale, shard=shard_name,
                                  chunk_x=cx, chunk_y=cy, chunk_z=cz,
                                  error=str(e)[:500])

                # Commit batch to local disk — sample memory before and after
                # commit to bracket the RMW peak (old + new shard coexist).
                if batch_chunks >= BATCH_SIZE:
                    mem_current, _, _ = _read_cgroup_memory()
                    shard_peak_mem = max(shard_peak_mem, mem_current)
                    txn.commit_async().result()
                    batches_committed += 1
                    mem_current, _, _ = _read_cgroup_memory()
                    shard_peak_mem = max(shard_peak_mem, mem_current)
                    txn = ts.Transaction()
                    batch_chunks = 0

                # Wall-clock progress logging
                now = time.time()
                if (now - last_progress_time) >= progress_interval:
                    mem_current, _, _ = _read_cgroup_memory()
                    shard_peak_mem = max(shard_peak_mem, mem_current)
                    mem_gib = mem_current / (1 << 30) if mem_current > 0 else 0
                    elapsed = now - shard_start_time
                    logger.info("Shard progress",
                                 shard=shard_name,
                                 scale=scale,
                                 elapsed_s=round(elapsed, 1),
                                 chunks_written=chunks_written,
                                 chunks_failed=chunks_failed,
                                 total=reader.chunk_count,
                                 memory_gib=round(mem_gib, 2),
                                 batches=batches_committed)
                    last_progress_time = now

            # Commit remaining chunks
            if batch_chunks > 0:
                mem_current, _, _ = _read_cgroup_memory()
                shard_peak_mem = max(shard_peak_mem, mem_current)
                txn.commit_async().result()
                batches_committed += 1
                mem_current, _, _ = _read_cgroup_memory()
                shard_peak_mem = max(shard_peak_mem, mem_current)

            # Fail if no chunks were written (all outside bounds or all failed)
            if chunks_written == 0:
                logger.error("Shard produced no output",
                             scale=scale, shard=shard_name,
                             total_chunks=reader.chunk_count,
                             chunks_outside=chunks_outside,
                             chunks_failed=chunks_failed)
                return False

            elapsed = time.time() - shard_start_time
            mem_current, mem_limit, cgroup_peak = _read_cgroup_memory()
            shard_peak_mem = max(shard_peak_mem, mem_current)
            shard_peak_gib = shard_peak_mem / (1 << 30) if shard_peak_mem > 0 else 0
            mem_limit_gib = mem_limit / (1 << 30) if mem_limit > 0 else 0

            # If we reset memory.peak before this shard, cgroup_peak is the
            # true kernel-level per-shard high-water mark (no sampling gaps).
            # Otherwise it's the container-lifetime peak.
            if peak_reset_ok:
                kernel_peak_gib = cgroup_peak / (1 << 30) if cgroup_peak > 0 else 0
            else:
                kernel_peak_gib = 0  # not per-shard, omit to avoid confusion

            # Best available per-shard peak: kernel peak if available,
            # else sampled max(memory.current).
            peak_gib = kernel_peak_gib if kernel_peak_gib > 0 else shard_peak_gib

            # Distinct, easily queryable log for per-shard memory analysis.
            # Query: textPayload=~"Shard memory peak"
            logger.info("Shard memory peak",
                         scale=scale,
                         shard=shard_name,
                         peak_memory_gib=round(peak_gib, 2),
                         sampled_peak_gib=round(shard_peak_gib, 2),
                         kernel_peak_gib=round(kernel_peak_gib, 2),
                         memory_limit_gib=round(mem_limit_gib, 2),
                         chunks=chunks_written)

            logger.info("Shard complete",
                         scale=scale,
                         shard=shard_name,
                         elapsed_s=round(elapsed, 1),
                         uncompressed_gib=round(shard_uncompressed_bytes / (1 << 30), 3),
                         peak_memory_gib=round(peak_gib, 2),
                         memory_limit_gib=round(mem_limit_gib, 2),
                         batches=batches_committed,
                         chunks_written=chunks_written,
                         chunks_failed=chunks_failed,
                         chunks_outside=chunks_outside)
            return True

        except Exception as e:
            logger.error("Failed to process shard",
                          scale=scale, shard=shard_name, error=str(e))
            return False

    def downres_scale(self, scale: int) -> bool:
        """Generate a scale by downsampling the previous scale from the destination.

        Reads scale N-1 from the neuroglancer precomputed volume on GCS,
        downsamples 2x in each dimension using majority vote, and writes
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

            dest.write(downsampled).result()

            logger.info("Downres complete", scale=scale)
            return True

        except Exception as e:
            logger.error("Failed to downres scale",
                          scale=scale, error=str(e))
            return False

    def _upload_label_csvs(self, scale: int, shard_number: int,
                           actual_labels: dict):
        """Upload actual label counts and next-scale predictions to source bucket.

        Args:
            scale: Current scale.
            shard_number: NG shard number.
            actual_labels: {(cx, cy, cz): set_of_labels} from read-back.
        """
        spec_params = load_ng_spec_from_dict(self.config.ng_spec)
        scale_params = spec_params[scale]
        shard_bits = scale_params["shard_bits"]
        hex_digits = -(-shard_bits // 4)
        shard_hex = f"{shard_number:0{hex_digits}x}"

        # Actual labels CSV for this scale
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["x", "y", "z", "num_labels", "num_supervoxels",
                         "unique_labels"])
        for (cx, cy, cz) in sorted(actual_labels.keys()):
            ul = len(actual_labels[(cx, cy, cz)])
            writer.writerow([cx, cy, cz, ul, ul, ul])

        blob_path = (f"{self._source_prefix}/s{scale}/"
                     f"{shard_hex}-labels.csv")
        blob = self.source_bucket_obj.blob(blob_path)
        blob.upload_from_string(out.getvalue(), content_type="text/csv")
        logger.info("Uploaded actual labels CSV",
                     scale=scale, shard_hex=shard_hex,
                     chunks=len(actual_labels))

        # Next-scale predicted labels (unless last scale in spec)
        max_scale = max(spec_params.keys())
        if scale < max_scale:
            child_groups = defaultdict(set)
            for (cx, cy, cz), label_set in actual_labels.items():
                child_coord = (cx // 2, cy // 2, cz // 2)
                child_groups[child_coord] |= label_set

            out = io.StringIO()
            writer = csv.writer(out)
            writer.writerow(["x", "y", "z", "num_labels",
                             "num_supervoxels", "unique_labels"])
            for (cx, cy, cz) in sorted(child_groups.keys()):
                ul = len(child_groups[(cx, cy, cz)])
                writer.writerow([cx, cy, cz, ul, ul, ul])

            next_scale = scale + 1
            blob_path = (f"{self._source_prefix}/s{scale}/"
                         f"{shard_hex}-s{next_scale}-predicted.csv")
            blob = self.source_bucket_obj.blob(blob_path)
            blob.upload_from_string(out.getvalue(), content_type="text/csv")
            logger.info("Uploaded predicted labels CSV",
                         scale=scale, shard_hex=shard_hex,
                         target_scale=next_scale,
                         predicted_chunks=len(child_groups))

    def downres_shard(self, scale: int, shard_bbox: dict) -> Tuple[bool, int]:
        """Generate one output shard at `scale` by downsampling scale-1 from GCS.

        1. Open source scale (N-1) from the dest bucket on GCS (read-only)
        2. Create tmpfs staging dir with info file
        3. Open local staging TensorStore (file driver)
        4. local_dest[bbox].write(downsampled[bbox]).result()
        5. Upload shard file(s) to GCS
        6. Delete staging dir

        Args:
            scale: Scale level to generate (must be >= 1)
            shard_bbox: Dict with shard_number, shard_origin, shard_extent,
                        num_chunks from ng_sharding.shard_bbox()

        Returns:
            (success, uploaded_bytes) tuple
        """
        shard_number = shard_bbox["shard_number"]
        x0, y0, z0 = shard_bbox["shard_origin"]
        sx, sy, sz = shard_bbox["shard_extent"]
        shard_start = time.time()

        staging_dir = os.path.join(
            self._staging_base, f"downres-s{scale}-{shard_number}"
        )

        try:
            estimated_memory_gib = shard_bbox.get("estimated_memory_gib")
            estimated_subtotal_gib = shard_bbox.get("estimated_subtotal_gib")
            estimated_output_gib = shard_bbox.get("estimated_output_gib")
            estimated_tmpfs_gib = shard_bbox.get("estimated_tmpfs_gib")
            estimated_raw_batch_gib = shard_bbox.get("estimated_raw_batch_gib")
            estimated_overhead_gib = shard_bbox.get("estimated_overhead_gib")
            estimated_commit_spike_gib = shard_bbox.get(
                "estimated_commit_spike_gib")
            estimate_model = shard_bbox.get("estimate_model", "")
            estimated_total_unique_labels = shard_bbox.get(
                "estimated_total_unique_labels")

            logger.info("Downres shard start",
                        scale=scale, shard_number=shard_number,
                        origin=(x0, y0, z0), extent=(sx, sy, sz),
                        num_chunks=shard_bbox["num_chunks"],
                        estimate_model=estimate_model,
                        estimated_memory_gib=estimated_memory_gib,
                        estimated_subtotal_gib=estimated_subtotal_gib,
                        estimated_output_gib=estimated_output_gib,
                        estimated_tmpfs_gib=estimated_tmpfs_gib,
                        estimated_raw_batch_gib=estimated_raw_batch_gib,
                        estimated_overhead_gib=estimated_overhead_gib,
                        estimated_commit_spike_gib=estimated_commit_spike_gib,
                        estimated_total_unique_labels=estimated_total_unique_labels)

            # Reset cgroup peak for per-shard memory tracking
            peak_reset_ok = _reset_cgroup_peak()
            mem_at_start, mem_limit, _ = _read_cgroup_memory()
            shard_peak_mem = mem_at_start
            mem_limit_gib = mem_limit / (1 << 30) if mem_limit else 0

            def _mem_gib():
                m, _, _ = _read_cgroup_memory()
                return m / (1 << 30) if m else 0

            logger.info("Downres memory: baseline",
                        scale=scale, shard_number=shard_number,
                        memory_gib=round(_mem_gib(), 2),
                        memory_limit_gib=round(mem_limit_gib, 2))

            # Open source scale (N-1) from GCS with bounded cache
            source_spec = {
                "driver": "neuroglancer_precomputed",
                "kvstore": self.config.dest_path,
                "scale_index": scale - 1,
                "open": True,
                "context": {
                    "cache_pool": {"total_bytes_limit": 256 * (1 << 20)},
                },
            }
            source = ts.open(source_spec).result()
            downsampled = ts.downsample(source, [2, 2, 2, 1], "mode")

            logger.info("Downres memory: after open source",
                        scale=scale, shard_number=shard_number,
                        memory_gib=round(_mem_gib(), 2))

            # Create staging dir and write info file
            os.makedirs(staging_dir, exist_ok=True)
            with open(os.path.join(staging_dir, "info"), "w") as f:
                f.write(self._info_json)

            # Open local staging volume.  Small cache helps RMW performance:
            # each batch commit re-reads the shard file to merge new chunks,
            # and the cache avoids redundant decode of the growing shard.
            local_dest = ts.open({
                "driver": "neuroglancer_precomputed",
                "kvstore": {"driver": "file", "path": staging_dir},
                "scale_index": scale,
                "open": True,
                "context": {
                    "cache_pool": {"total_bytes_limit": 256 * (1 << 20)},
                },
            }).result()

            # Write in batched transactions.  Between write() and commit(),
            # TensorStore holds raw uint64 arrays (2 MiB per 64^3 chunk) in
            # ChunkCache — cache_pool eviction does NOT apply to explicit
            # transactions.  One Z-plane per batch.  After commit, arrays
            # are encoded (compressed_segmentation + gzip) and flushed to
            # the shard file on tmpfs via read-modify-write.
            chunk_z = 64  # chunk size in Z voxels
            batch_z_voxels = chunk_z  # one Z-plane per batch
            num_batches = max(1, (sz + batch_z_voxels - 1) // batch_z_voxels)
            batches_committed = 0

            logger.info("Downres memory: before write",
                        scale=scale, shard_number=shard_number,
                        memory_gib=round(_mem_gib(), 2),
                        num_batches=num_batches)

            write_peak_mem = 0
            for z_off in range(z0, z0 + sz, batch_z_voxels):
                z_end = min(z_off + batch_z_voxels, z0 + sz)
                txn = ts.Transaction()
                local_dest.with_transaction(txn)[
                    x0:x0+sx, y0:y0+sy, z_off:z_end, :
                ].write(
                    downsampled[x0:x0+sx, y0:y0+sy, z_off:z_end, :]
                ).result()
                txn.commit_async().result()
                batches_committed += 1
                mem_current, _, _ = _read_cgroup_memory()
                shard_peak_mem = max(shard_peak_mem, mem_current)
                write_peak_mem = max(write_peak_mem, mem_current)
                # Log every batch so memory data survives OOM.
                # Walk staging dir recursively — shard files are in
                # subdirectories like <scale_key>/<hex>.shard.
                batch_tmpfs = 0
                for dirpath, _, filenames in os.walk(staging_dir):
                    for fname in filenames:
                        fpath = os.path.join(dirpath, fname)
                        try:
                            batch_tmpfs += os.path.getsize(fpath)
                        except OSError:
                            pass
                logger.info("Downres write batch",
                            scale=scale, shard_number=shard_number,
                            batch=f"{batches_committed}/{num_batches}",
                            z_range=f"{z_off}-{z_end}",
                            memory_gib=round(mem_current / (1 << 30), 2),
                            memory_limit_gib=round(mem_limit_gib, 2),
                            tmpfs_mib=round(batch_tmpfs / (1 << 20), 1),
                            num_chunks=shard_bbox["num_chunks"])

            # Get tmpfs usage for the staging dir (recursive)
            tmpfs_bytes = 0
            for dirpath, _, filenames in os.walk(staging_dir):
                for fname in filenames:
                    try:
                        tmpfs_bytes += os.path.getsize(
                            os.path.join(dirpath, fname))
                    except OSError:
                        pass

            logger.info("Downres memory: after write",
                        scale=scale, shard_number=shard_number,
                        memory_gib=round(_mem_gib(), 2),
                        write_peak_gib=round(write_peak_mem / (1 << 30), 2),
                        tmpfs_mib=round(tmpfs_bytes / (1 << 20), 1),
                        batches=batches_committed,
                        estimated_memory_gib=estimated_memory_gib,
                        estimated_output_gib=estimated_output_gib,
                        estimated_raw_batch_gib=estimated_raw_batch_gib,
                        estimated_commit_spike_gib=estimated_commit_spike_gib)

            # Read back chunks from tmpfs to count actual unique labels
            # and compute next-scale predictions.
            label_readback_start = time.time()
            actual_labels = {}  # (cx, cy, cz) -> set of unique label values

            if self.config.ng_spec:
                spec_params = load_ng_spec_from_dict(self.config.ng_spec)
                scale_params = spec_params.get(scale)
                if scale_params:
                    chunk_coords = shard_chunk_coords(shard_number, scale_params)
                    chunk_size = scale_params["chunk_size"]

                    # Open a read-only handle to the staged shard on tmpfs
                    local_reader = ts.open({
                        "driver": "neuroglancer_precomputed",
                        "kvstore": {"driver": "file", "path": staging_dir},
                        "scale_index": scale,
                        "open": True,
                    }).result()

                    total_labels_stored = 0
                    for cx, cy, cz in chunk_coords:
                        vx0 = cx * chunk_size[0]
                        vy0 = cy * chunk_size[1]
                        vz0 = cz * chunk_size[2]
                        vx1 = min(vx0 + chunk_size[0], scale_params["vol_size"][0])
                        vy1 = min(vy0 + chunk_size[1], scale_params["vol_size"][1])
                        vz1 = min(vz0 + chunk_size[2], scale_params["vol_size"][2])
                        try:
                            chunk = local_reader[
                                vx0:vx1, vy0:vy1, vz0:vz1, :
                            ].read().result()
                            labels = set(np.unique(chunk).tolist())
                            # Skip all-zero chunks (background only)
                            if labels != {0}:
                                actual_labels[(cx, cy, cz)] = labels
                                total_labels_stored += len(labels)
                        except Exception:
                            pass  # skip chunks that fail to read

                    label_readback_elapsed = time.time() - label_readback_start
                    logger.info("Downres memory: after label readback",
                                scale=scale, shard_number=shard_number,
                                memory_gib=round(_mem_gib(), 2),
                                chunks_read=len(actual_labels),
                                total_labels_stored=total_labels_stored,
                                elapsed_s=round(label_readback_elapsed, 2))

            # Upload shard file(s) to GCS
            upload_start = time.time()
            uploaded_bytes = self.upload_staging_dir(staging_dir)
            upload_elapsed = time.time() - upload_start

            # Upload label CSVs to source bucket after shard upload
            if actual_labels:
                try:
                    self._upload_label_csvs(
                        scale, shard_number, actual_labels)
                except Exception as e:
                    logger.warning("Failed to upload label CSVs",
                                   scale=scale, shard_number=shard_number,
                                   error=str(e)[:200])

            # Free label data before final memory measurement
            del actual_labels

            elapsed = time.time() - shard_start

            # Memory reporting
            mem_current, mem_limit, cgroup_peak = _read_cgroup_memory()
            shard_peak_mem = max(shard_peak_mem, mem_current)
            peak_gib = (cgroup_peak / (1 << 30) if peak_reset_ok and cgroup_peak
                        else shard_peak_mem / (1 << 30))

            logger.info("Downres shard complete",
                        scale=scale, shard_number=shard_number,
                        elapsed_s=round(elapsed, 1),
                        upload_s=round(upload_elapsed, 1),
                        uploaded_gib=round(uploaded_bytes / (1 << 30), 3),
                        peak_memory_gib=round(peak_gib, 2),
                        num_chunks=shard_bbox["num_chunks"],
                        tmpfs_mib=round(tmpfs_bytes / (1 << 20), 1),
                        batches=batches_committed,
                        estimate_model=estimate_model,
                        estimated_memory_gib=estimated_memory_gib,
                        estimated_subtotal_gib=estimated_subtotal_gib,
                        estimated_output_gib=estimated_output_gib,
                        estimated_tmpfs_gib=estimated_tmpfs_gib,
                        estimated_raw_batch_gib=estimated_raw_batch_gib,
                        estimated_overhead_gib=estimated_overhead_gib,
                        estimated_commit_spike_gib=estimated_commit_spike_gib,
                        estimated_total_unique_labels=estimated_total_unique_labels,
                        prediction_error_gib=(
                            round(peak_gib - estimated_memory_gib, 2)
                            if estimated_memory_gib is not None else None),
                        prediction_ratio=(
                            round(peak_gib / estimated_memory_gib, 3)
                            if estimated_memory_gib else None))

            return True, uploaded_bytes

        except Exception as e:
            logger.error("Failed to downres shard",
                         scale=scale, shard_number=shard_number,
                         error=str(e))
            return False, 0

        finally:
            # Always clean up staging dir to free tmpfs
            if os.path.isdir(staging_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)


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
        mem_current, mem_limit, _ = _read_cgroup_memory()
        mem_limit_gib = mem_limit / (1 << 30) if mem_limit > 0 else 0
        logger.info("Starting worker",
                     scales=self.config.scales,
                     downres_scales=self.config.downres_scales,
                     memory_limit_gib=round(mem_limit_gib, 2),
                     memory_current_gib=round(mem_current / (1 << 30), 2) if mem_current else 0,
                     worker_memory_gib=self.config.worker_memory_gib)

        processed_count = 0
        failed_count = 0

        # Phase 1: Process assigned DVID export shards
        if self.config.manifest_uri:
            my_shards = self.processor.load_manifest()
        else:
            my_shards = self.processor.list_my_shards()

        # Open one staging dir + TensorStore handle per scale.
        scales_needed = sorted(set(scale for scale, _ in my_shards))
        staging_dirs: Dict[int, str] = {}
        ts_handles: Dict[int, ts.TensorStore] = {}

        for scale in scales_needed:
            staging_dir = os.path.join(self.processor._staging_base, f"s{scale}")
            os.makedirs(staging_dir, exist_ok=True)
            with open(os.path.join(staging_dir, "info"), "w") as f:
                f.write(self.processor._info_json)

            ts_handles[scale] = ts.open({
                "driver": "neuroglancer_precomputed",
                "kvstore": {"driver": "file", "path": staging_dir},
                "scale_index": scale,
                "open": True,
            }).result()
            staging_dirs[scale] = staging_dir
            logger.info("Opened local staging volume",
                         scale=scale,
                         staging_dir=staging_dir,
                         shape=list(ts_handles[scale].shape))

        for scale, shard_name in my_shards:
            if not self._should_continue():
                logger.warning("Time limit reached, stopping",
                                processed=processed_count, remaining=len(my_shards) - processed_count - failed_count)
                break

            # Check memory headroom before starting a new shard.
            mem_current, mem_limit, _ = _read_cgroup_memory()
            if mem_limit > 0 and mem_current > 0:
                usage_pct = mem_current / mem_limit
                if usage_pct > 0.90:
                    logger.error("Memory critical before shard",
                                  scale=scale, shard=shard_name,
                                  memory_gib=round(mem_current / (1 << 30), 2),
                                  memory_limit_gib=round(mem_limit / (1 << 30), 2),
                                  usage_pct=round(usage_pct * 100, 1))
                elif usage_pct > 0.75:
                    logger.warning("Memory pressure before shard",
                                    scale=scale, shard=shard_name,
                                    memory_gib=round(mem_current / (1 << 30), 2),
                                    memory_limit_gib=round(mem_limit / (1 << 30), 2),
                                    usage_pct=round(usage_pct * 100, 1))

            try:
                success = self.processor.process_shard(scale, shard_name, ts_handles[scale])

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

        # Upload all accumulated shard files from staging dirs to GCS
        for scale in scales_needed:
            staging_dir = staging_dirs[scale]
            upload_start = time.time()
            uploaded_bytes = self.processor.upload_staging_dir(staging_dir)
            upload_elapsed = time.time() - upload_start
            logger.info("Upload complete",
                         scale=scale,
                         uploaded_bytes=uploaded_bytes,
                         uploaded_gib=round(uploaded_bytes / (1 << 30), 3),
                         elapsed_s=round(upload_elapsed, 1))

        # Clean up staging dirs to free tmpfs memory
        for staging_dir in staging_dirs.values():
            shutil.rmtree(staging_dir, ignore_errors=True)

        # Phase 2: Generate downres scales from previous scale data
        # (legacy path — only used when downres_scales is set directly,
        # not in manifest-driven DOWNRES_MODE)
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

    async def run_downres(self):
        """Run manifest-driven per-shard downres processing.

        Loads a downres manifest (list of shard bboxes), iterates over
        assigned output shards, calls downres_shard() for each.
        """
        mem_current, mem_limit, _ = _read_cgroup_memory()
        mem_limit_gib = mem_limit / (1 << 30) if mem_limit > 0 else 0
        logger.info("Starting downres worker",
                     memory_limit_gib=round(mem_limit_gib, 2),
                     memory_current_gib=round(mem_current / (1 << 30), 2) if mem_current else 0,
                     worker_memory_gib=self.config.worker_memory_gib)

        # Load downres manifest (raw dicts with shard_origin, shard_extent, etc.)
        prefix = self.config.manifest_uri.rstrip("/")
        task_uri = f"{prefix}/task-{self.processor._task_index}.json"
        bucket_name, blob_path = _parse_gs_uri(task_uri)
        blob = self.processor.storage_client.bucket(bucket_name).blob(blob_path)
        entries = json.loads(blob.download_as_text())

        logger.info("Loaded downres manifest",
                     manifest_uri=task_uri,
                     assigned_shards=len(entries))

        processed_count = 0
        failed_count = 0
        total_uploaded = 0

        for entry in entries:
            if not self._should_continue():
                logger.warning("Time limit reached, stopping downres",
                               processed=processed_count,
                               remaining=len(entries) - processed_count - failed_count)
                break

            # Check memory headroom
            mem_current, mem_limit, _ = _read_cgroup_memory()
            if mem_limit > 0 and mem_current > 0:
                usage_pct = mem_current / mem_limit
                if usage_pct > 0.90:
                    logger.error("Memory critical before downres shard",
                                 scale=entry["scale"],
                                 shard_number=entry["shard_number"],
                                 memory_gib=round(mem_current / (1 << 30), 2),
                                 usage_pct=round(usage_pct * 100, 1))
                elif usage_pct > 0.75:
                    logger.warning("Memory pressure before downres shard",
                                   scale=entry["scale"],
                                   shard_number=entry["shard_number"],
                                   memory_gib=round(mem_current / (1 << 30), 2),
                                   usage_pct=round(usage_pct * 100, 1))

            shard_bbox = {
                "shard_number": entry["shard_number"],
                "shard_origin": entry["shard_origin"],
                "shard_extent": entry["shard_extent"],
                "num_chunks": entry["num_chunks"],
                "estimate_model": entry.get("estimate_model"),
                "estimated_memory_gib": entry.get("estimated_memory_gib"),
                "estimated_subtotal_gib": entry.get("estimated_subtotal_gib"),
                "estimated_output_gib": entry.get("estimated_output_gib"),
                "estimated_tmpfs_gib": entry.get("estimated_tmpfs_gib"),
                "estimated_raw_batch_gib": entry.get("estimated_raw_batch_gib"),
                "estimated_overhead_gib": entry.get("estimated_overhead_gib"),
                "estimated_total_unique_labels": entry.get(
                    "estimated_total_unique_labels"),
            }

            success, uploaded_bytes = self.processor.downres_shard(
                entry["scale"], shard_bbox
            )
            if success:
                processed_count += 1
                total_uploaded += uploaded_bytes
            else:
                failed_count += 1

        elapsed = (time.time() - self.start_time) / 60
        logger.info("Downres worker finished",
                     elapsed_minutes=round(elapsed, 1),
                     processed=processed_count,
                     failed=failed_count,
                     total_uploaded_gib=round(total_uploaded / (1 << 30), 3))


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
        staging_path=os.environ.get("STAGING_PATH", DEFAULT_STAGING_PATH),
        worker_memory_gib=float(os.environ.get("WORKER_MEMORY_GIB", "4")),
        manifest_uri=os.environ.get("MANIFEST_URI", ""),
        max_processing_time_minutes=int(os.environ.get("MAX_PROCESSING_TIME", "1440")),
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
        if os.environ.get("DOWNRES_MODE") == "1":
            await worker.run_downres()
        else:
            await worker.run()
    except Exception as e:
        logger.error("Worker failed to start", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())
