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
import shutil
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

    def _create_local_volume(self, scale: int, shard_name: str) -> str:
        """Create a local staging directory with the neuroglancer info file.

        Returns the staging directory path.  The info file is written from
        the cached copy downloaded at startup.
        """
        staging_dir = os.path.join(
            self._staging_base, f"s{scale}_{shard_name}")
        os.makedirs(staging_dir, exist_ok=True)
        info_path = os.path.join(staging_dir, "info")
        with open(info_path, "w") as f:
            f.write(self._info_json)
        # Verify the file was written correctly
        written_size = os.path.getsize(info_path)
        if written_size != len(self._info_json.encode("utf-8")):
            logger.error("Info file size mismatch",
                          expected=len(self._info_json.encode("utf-8")),
                          written=written_size, path=info_path)
        return staging_dir

    def _upload_shard_files(self, staging_dir: str,
                            scale: int = -1, shard_name: str = "") -> int:
        """Upload shard files from local staging to the GCS destination.

        Walks the staging directory and uploads every file except 'info',
        preserving the directory structure (e.g., s0/070e7.shard).
        Uses streaming upload with a small chunk size to avoid buffering
        the entire shard file in memory.

        Returns the number of bytes uploaded.
        """
        uploaded_bytes = 0
        all_files = []
        for root, _dirs, files in os.walk(staging_dir):
            for fn in files:
                local_path = os.path.join(root, fn)
                rel_path = os.path.relpath(local_path, staging_dir)
                size = os.path.getsize(local_path)
                all_files.append((rel_path, size))
                if fn == "info":
                    continue
                blob_path = f"{self._dest_prefix}/{rel_path}"
                blob = self.dest_bucket_obj.blob(blob_path)
                # Use resumable upload with 8MB chunks to avoid loading
                # the entire shard file into memory.  Default upload
                # buffers the whole file, which OOMs for large shards.
                blob.chunk_size = 8 * (1 << 20)  # 8 MiB
                blob.upload_from_filename(local_path)
                uploaded_bytes += size

        # Diagnostic: log staging contents when no shard files were produced.
        shard_files = [(f, s) for f, s in all_files if not f.endswith("/info") and f != "info"]
        if not shard_files:
            logger.warning("No shard files in staging directory",
                           scale=scale, shard=shard_name,
                           staging_dir=staging_dir,
                           all_files=[(f, s) for f, s in all_files])

        return uploaded_bytes

    @staticmethod
    def _list_staging(staging_dir: str, label: str):
        """Log all files in the staging directory for debugging."""
        entries = []
        for root, dirs, files in os.walk(staging_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                rel = os.path.relpath(fp, staging_dir)
                try:
                    sz = os.path.getsize(fp)
                except OSError:
                    sz = -1
                entries.append(f"{rel}:{sz}")
        logger.info("DEBUG staging listing",
                     label=label,
                     staging_dir=staging_dir,
                     files=entries,
                     pid=os.getpid())

    def process_shard(self, scale: int, shard_name: str) -> bool:
        """Process a single shard: read from GCS, write to local disk, upload.

        1. Downloads Arrow+CSV from GCS and decompresses chunks via BRAID.
        2. Writes chunks to a local neuroglancer precomputed volume using
           batched transactions (BATCH_SIZE chunks per batch).  Each batch
           commit does a read-modify-write on local disk, keeping memory flat.
        3. Uploads the finished shard file(s) to GCS.
        4. Cleans up the local staging directory.

        Args:
            scale: Scale level (0, 1, 2, ...)
            shard_name: Shard origin name (e.g., "30720_24576_28672")

        Returns:
            True if the shard was processed (even with some chunk errors).
            False only if the shard could not be loaded at all.
        """
        staging_dir = None
        try:
            logger.info("Processing shard", scale=scale, shard=shard_name)

            arrow_uri = f"{self.config.source_path}/s{scale}/{shard_name}.arrow"
            csv_uri = f"{self.config.source_path}/s{scale}/{shard_name}.csv"
            reader = self._open_shard_with_retry(arrow_uri, csv_uri)

            logger.info("Shard loaded",
                         shard=shard_name,
                         chunks=reader.chunk_count)

            # Create local staging volume with info file
            staging_dir = self._create_local_volume(scale, shard_name)

            # DEBUG: verify info file
            info_path = os.path.join(staging_dir, "info")
            info_size = os.path.getsize(info_path)
            with open(info_path) as _f:
                info_head = _f.read(200)
            import json as _json
            try:
                _json.loads(open(info_path).read())
                info_valid = True
            except Exception:
                info_valid = False
            logger.info("DEBUG info file",
                         scale=scale, shard=shard_name,
                         info_size=info_size,
                         info_valid=info_valid,
                         info_head=info_head[:100],
                         staging_dir=staging_dir,
                         pid=os.getpid())

            self._list_staging(staging_dir, "after info write")

            dest = ts.open({
                "driver": "neuroglancer_precomputed",
                "kvstore": {"driver": "file", "path": staging_dir},
                "scale_index": scale,
                "open": True,
            }).result()

            # DEBUG: log resolved TensorStore spec
            resolved_spec = dest.spec(retain_context=True).to_json()
            logger.info("DEBUG tensorstore opened",
                         scale=scale, shard=shard_name,
                         kvstore_path=resolved_spec.get("kvstore", {}).get("path", "?"),
                         shape=list(dest.shape),
                         dtype=str(dest.dtype),
                         fill_value=str(dest.fill_value))

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
                    import numpy as _np
                    logger.info("DEBUG chunk write",
                                 scale=scale, shard=shard_name,
                                 chunk_x=cx, chunk_y=cy, chunk_z=cz,
                                 voxel_range=f"[{x0}:{x1},{y0}:{y1},{z0}:{z1}]",
                                 shape=list(clipped.shape),
                                 nonzero=int(_np.count_nonzero(clipped)),
                                 dtype=str(clipped.dtype),
                                 min_val=int(clipped.min()),
                                 max_val=int(clipped.max()),
                                 txn_id=id(txn))
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
                    self._list_staging(staging_dir, f"BEFORE batch commit #{batches_committed}")
                    logger.info("DEBUG committing batch",
                                 scale=scale, shard=shard_name,
                                 batch_chunks=batch_chunks,
                                 txn_id=id(txn))
                    mem_current, _, _ = _read_cgroup_memory()
                    shard_peak_mem = max(shard_peak_mem, mem_current)
                    txn.commit_async().result()
                    batches_committed += 1
                    mem_current, _, _ = _read_cgroup_memory()
                    shard_peak_mem = max(shard_peak_mem, mem_current)
                    self._list_staging(staging_dir, f"AFTER batch commit #{batches_committed-1}")
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
                self._list_staging(staging_dir, "BEFORE final commit")
                logger.info("DEBUG committing final batch",
                             scale=scale, shard=shard_name,
                             batch_chunks=batch_chunks,
                             chunks_written=chunks_written,
                             txn_id=id(txn))
                mem_current, _, _ = _read_cgroup_memory()
                shard_peak_mem = max(shard_peak_mem, mem_current)
                txn.commit_async().result()
                batches_committed += 1
                mem_current, _, _ = _read_cgroup_memory()
                shard_peak_mem = max(shard_peak_mem, mem_current)
                self._list_staging(staging_dir, "AFTER final commit")
            else:
                logger.info("DEBUG no final commit needed",
                             scale=scale, shard=shard_name,
                             batch_chunks=batch_chunks,
                             chunks_written=chunks_written)

            # Fail if no chunks were written (all outside bounds or all failed)
            if chunks_written == 0:
                logger.error("Shard produced no output",
                             scale=scale, shard=shard_name,
                             total_chunks=reader.chunk_count,
                             chunks_outside=chunks_outside,
                             chunks_failed=chunks_failed)
                return False

            # Release the TensorStore handle before scanning for output
            # files.  This ensures any deferred writes (cache writeback,
            # sharded format finalization) are flushed to the filesystem.
            self._list_staging(staging_dir, "BEFORE del dest")
            del dest
            import gc as _gc
            _gc.collect()
            self._list_staging(staging_dir, "AFTER del dest + gc")

            # Upload finished shard file(s) to GCS
            upload_start = time.time()
            uploaded_bytes = self._upload_shard_files(staging_dir, scale, shard_name)
            upload_elapsed = time.time() - upload_start

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
                         chunks=chunks_written,
                         uploaded_mib=round(uploaded_bytes / (1 << 20), 1))

            logger.info("Shard complete",
                         scale=scale,
                         shard=shard_name,
                         elapsed_s=round(elapsed, 1),
                         upload_s=round(upload_elapsed, 1),
                         uploaded_mib=round(uploaded_bytes / (1 << 20), 1),
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

        finally:
            # Clean up local staging to free disk for the next shard
            if staging_dir and os.path.isdir(staging_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)

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
        await worker.run()
    except Exception as e:
        logger.error("Worker failed to start", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())
