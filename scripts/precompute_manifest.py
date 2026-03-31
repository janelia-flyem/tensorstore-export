#!/usr/bin/env python3
"""
Precompute task manifests for tier-based Cloud Run job execution.

Scans all Arrow source files across configured scales, estimates memory
requirements for each shard, assigns shards to memory tiers, and distributes
them across tasks with load balancing.  Writes per-tier manifest JSON files
to GCS.

Usage:
    pixi run precompute-manifest
    pixi run precompute-manifest --tiers 4:2600,8:30,16:5
"""

import argparse
import json
import heapq
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE

# Available memory tiers with Cloud Run CPU coupling constraints.
# Cloud Run Gen2: CPU=1→max 4Gi, CPU=2→max 8Gi, CPU=4→max 16Gi, CPU=8→max 32Gi.
TIER_CPU = {4: 2, 8: 2, 16: 4, 24: 6, 32: 8}

# Default max tasks per tier; override with --tiers.
DEFAULT_TIER_MAX_TASKS = {4: 5000, 8: 5000, 16: 2500, 24: 100, 32: 20}


def list_arrow_files(source_path: str, scales: list) -> list:
    """List all Arrow files with sizes and chunk counts across scales.

    Returns list of (scale, shard_name, size_bytes, chunk_count) tuples.
    Chunk count is obtained from the companion CSV file (lines - 1 header).
    """
    all_files = []
    arrow_by_scale = {}  # scale -> {name: size_bytes}

    # Pass 1: list Arrow files and sizes
    for scale in scales:
        prefix = f"{source_path}/s{scale}/"
        result = subprocess.run(
            ["gsutil", "ls", "-l", prefix],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  Warning: could not list {prefix}: {result.stderr.strip()}")
            continue

        arrow_by_scale[scale] = {}
        csv_sizes = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("TOTAL"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                if parts[2].endswith(".arrow"):
                    size_bytes = int(parts[0])
                    name = parts[2].split("/")[-1].replace(".arrow", "")
                    arrow_by_scale[scale][name] = size_bytes
                elif parts[2].endswith(".csv"):
                    size_bytes = int(parts[0])
                    name = parts[2].split("/")[-1].replace(".csv", "")
                    csv_sizes[name] = size_bytes

        # Estimate chunk count from CSV size: each line ~15 bytes (x,y,z,rec)
        # minus 1 header line
        for name, arrow_size in arrow_by_scale[scale].items():
            csv_size = csv_sizes.get(name, 0)
            chunk_count = max(0, csv_size // 15 - 1) if csv_size > 0 else 0
            all_files.append((scale, name, arrow_size, chunk_count))

    return all_files


# ---------------------------------------------------------------------------
# Memory estimation — empirical calibration from production data
# ---------------------------------------------------------------------------
#
# The output .shard file on tmpfs uses compressed_segmentation + gzip encoding,
# byte-identical to the final GCS output.  tmpfs consumption is the dominant
# variable cost; everything else is fixed overhead.
#
# See docs/ExportShardsOptimization.md §12 for derivation and calibration
# instructions.

# Max observed bytes per chunk in output .shard file, by scale.
# Derived as: max(ng_output_bytes / chunk_count) across all shards at each
# scale, taking the maximum across all calibrated datasets (v0.11 and FMC).
#
# IMPORTANT: this is NOT max_shard_size / full_shard_chunks.  The densest
# bytes-per-chunk shard is often a partially-filled shard with high label
# density, not a full 32K-chunk interior shard.
#
# This is a conservative worst-case upper bound.  The brain sparsely fills
# the bounding volume, so most shards (especially boundary shards) have
# much lower bytes-per-chunk.  The label-aware model (BYTES_PER_UNIQUE_LABEL)
# is much tighter when label profiles are available.
#
# Sources: v0.11 + false-merge-corrected (max across both datasets).
# Refresh after each new dataset: see docs/ExportShardsOptimization.md.
BYTES_PER_CHUNK = {
    0:  13_932,   # 13.6 KB — v0.11 = FMC
    1:  27_919,   # 27.3 KB — v0.11 = FMC
    2:  44_482,   # 43.4 KB — v0.11 = FMC
    3:  78_376,   # 76.5 KB — FMC > v0.11 (77,480)
    4: 130_244,   # 127.2 KB — FMC > v0.11 (128,757)
    5: 199_396,   # 194.7 KB — FMC > v0.11 (196,923)
    6: 258_999,   # 252.9 KB — FMC > v0.11 (255,166)
    7: 170_713,   # 166.7 KB — FMC > v0.11 (166,127)
    8: 159_269,   # 155.5 KB — FMC > v0.11 (153,717)
    9: 129_385,   # 126.4 KB — FMC > v0.11 (125,268)
}

# Fixed overhead for shard processing (s0): Python + BRAID + pyarrow +
# TensorStore + GCS client + decompression buffers.  Measured from cgroup
# memory.current at worker startup before any shard processing begins.
SHARD_PROC_OVERHEAD_GIB = 2.0

# Downres memory model constants.
#
# This path writes one output shard at a time from a lazy downsampled source.
# The dominant modeled terms are:
#   - raw uint64 output arrays held in the explicit transaction
#   - 2x output shard bytes during local read-modify-write commit
#   - fixed process/caches/readback headroom
#   - transient write/commit spike during TensorStore encode + RMW
# A safety factor is applied to cover additional source-side working set and
# encode/merge scratch space that are not cleanly observable from static shard
# metadata alone.
DOWNRES_SOURCE_CACHE_GIB = 0.25
DOWNRES_DEST_CACHE_GIB = 0.25
DOWNRES_RUNTIME_GIB = 0.5
DOWNRES_LABEL_READBACK_GIB = 0.5
# For s4+, the dominant missing term is not a write spike; it is the large
# in-memory actual_labels structure after readback.  The label summary stores
# the same per-shard total later logged by the worker as total_labels_stored,
# so use it to scale the readback estimate directly.
DOWNRES_LABEL_READBACK_BYTES_PER_LABEL = 80
DOWNRES_DYNAMIC_LABEL_READBACK_SCALE = 4
# The failed downres tasks were killed after logging RSS well below the tier
# boundary, which implies a short-lived write/commit spike not captured by the
# post-commit memory samples. Keep explicit scale floors so later scales can be
# pushed into larger tiers without forcing an excessively steep linear ramp.
DOWNRES_COMMIT_SPIKE_GIB_BY_SCALE = {
    1: 0.5,
    2: 1.0,
    3: 1.5,
    4: 2.0,
}
DOWNRES_DEFAULT_COMMIT_SPIKE_GIB = 2.0


def downres_commit_spike_gib(scale: int) -> float:
    """Return the scale-aware hidden commit/write spike term in GiB."""
    return DOWNRES_COMMIT_SPIKE_GIB_BY_SCALE.get(
        scale, DOWNRES_DEFAULT_COMMIT_SPIKE_GIB)


def downres_label_readback_gib(scale: int,
                               total_unique_labels: int | None) -> float:
    """Estimate the post-readback label structure footprint in GiB."""
    if (scale < DOWNRES_DYNAMIC_LABEL_READBACK_SCALE or
            total_unique_labels is None):
        return DOWNRES_LABEL_READBACK_GIB

    dynamic_gib = (
        total_unique_labels * DOWNRES_LABEL_READBACK_BYTES_PER_LABEL
    ) / (1 << 30)
    return max(DOWNRES_LABEL_READBACK_GIB, dynamic_gib)


DOWNRES_OVERHEAD_GIB = (
    DOWNRES_SOURCE_CACHE_GIB +
    DOWNRES_DEST_CACHE_GIB +
    DOWNRES_RUNTIME_GIB +
    DOWNRES_LABEL_READBACK_GIB
)
DOWNRES_SAFETY_FACTOR = 1.2

# Bytes per unique label in the output shard file, by scale.
# Derived from linear regression of (total_unique_labels, ng_output_bytes)
# across production shards.  R² > 0.95 for all scales.
# Much tighter than BYTES_PER_CHUNK when label profiles are available.
#
# Sources: v0.11 + false-merge-corrected (max across both datasets).
# Refresh after each new dataset: see docs/ExportShardsOptimization.md.
BYTES_PER_UNIQUE_LABEL = {0: 394, 1: 341, 2: 153, 3: 55, 4: 23, 5: 13}


def estimate_tmpfs_gib(chunk_count: int, scale: int) -> float:
    """Estimate the output .shard file size on tmpfs.

    Uses the worst-case bytes-per-chunk rate observed across production
    datasets.  Conservative for boundary shards and sparse regions.
    """
    bpc = BYTES_PER_CHUNK.get(scale, 80_000)
    return chunk_count * bpc / (1 << 30)


def estimate_memory_gib(arrow_size_bytes: int, chunk_count: int = 0,
                        scale: int = 0) -> float:
    """Estimate total memory to process a DVID Arrow shard.

    Memory = Arrow in RAM + output .shard on tmpfs (×2 for RMW) + fixed overhead.

    The RMW factor: during batched transaction commits, TensorStore holds
    both the old and new shard file on tmpfs simultaneously, so peak tmpfs
    is ~2× the final shard size.
    """
    arrow_gib = arrow_size_bytes / (1 << 30)
    tmpfs_gib = estimate_tmpfs_gib(chunk_count, scale)
    return arrow_gib + 2 * tmpfs_gib + SHARD_PROC_OVERHEAD_GIB


def pick_tier(mem_needed_gib: float, min_tier: int = 4) -> int:
    """Return the smallest tier (GiB) that fits the estimated memory need.

    min_tier enforces a floor: even tiny shards need memory for the Python
    runtime, TensorStore, pyarrow, and the output shard file on tmpfs.
    """
    for gib in sorted(TIER_CPU.keys()):
        if gib >= min_tier and mem_needed_gib <= gib:
            return gib
    return max(TIER_CPU.keys())  # largest available


def assign_tiers(files: list, max_tasks: dict) -> dict:
    """Assign each shard to the smallest tier that fits.

    Args:
        files: list of (scale, shard_name, size_bytes)
        max_tasks: dict mapping tier_gib -> max tasks for that tier

    Returns:
        dict mapping tier_gib -> list of (scale, shard_name, size_bytes)
        Only tiers with at least one shard are included.
    """
    tier_map = {}

    for entry in files:
        scale, name, size_bytes = entry[0], entry[1], entry[2]
        chunk_count = entry[3] if len(entry) > 3 else 0
        mem_needed = estimate_memory_gib(size_bytes, chunk_count, scale=scale)
        gib = pick_tier(mem_needed)
        tier_map.setdefault(gib, []).append((scale, name, size_bytes))

    return tier_map


def distribute_tasks(shards: list, max_tasks: int) -> dict:
    """Distribute shards across tasks using greedy load balancing.

    Each shard goes to the task with the smallest cumulative Arrow bytes.

    Args:
        shards: list of (scale, shard_name, size_bytes)
        max_tasks: maximum number of tasks for this tier

    Returns:
        dict mapping task_index (str) -> list of {scale, shard} dicts
    """
    num_tasks = min(max_tasks, len(shards))
    if num_tasks == 0:
        return {}

    # Min-heap of (cumulative_bytes, task_index)
    heap = [(0, i) for i in range(num_tasks)]
    tasks = {str(i): [] for i in range(num_tasks)}

    # Process largest shards first for better balance
    for scale, name, size_bytes in sorted(shards, key=lambda s: -s[2]):
        cum_bytes, task_idx = heapq.heappop(heap)
        tasks[str(task_idx)].append({"scale": scale, "shard": name})
        heapq.heappush(heap, (cum_bytes + size_bytes, task_idx))

    return tasks


def estimate_downres_raw_batch_gib(chunk_count: int) -> float:
    """Estimate raw uint64 batch memory for one output Z-plane write."""
    chunks_per_z_plane = max(1, int(chunk_count ** (2/3)))
    return chunks_per_z_plane * 2 * (1 << 20) / (1 << 30)


def estimate_downres_components(scale: int, chunk_count: int,
                                total_unique_labels: int | None = None) -> dict:
    """Return a structured downres memory estimate.

    The estimate intentionally separates the modeled subtotal from the applied
    safety factor so Cloud Run logs can later be fit against observed peaks.
    """
    if total_unique_labels is not None:
        bpul = BYTES_PER_UNIQUE_LABEL.get(scale, 13)
        output_gib = total_unique_labels * bpul / (1 << 30)
        model = "label_aware"
    else:
        output_gib = estimate_tmpfs_gib(chunk_count, scale)
        model = "chunk_count"

    raw_batch_gib = estimate_downres_raw_batch_gib(chunk_count)
    label_readback_gib = downres_label_readback_gib(
        scale, total_unique_labels)
    commit_spike_gib = downres_commit_spike_gib(scale)
    subtotal_gib = (
        raw_batch_gib +
        2 * output_gib +
        DOWNRES_SOURCE_CACHE_GIB +
        DOWNRES_DEST_CACHE_GIB +
        DOWNRES_RUNTIME_GIB +
        label_readback_gib +
        commit_spike_gib
    )
    total_gib = subtotal_gib * DOWNRES_SAFETY_FACTOR
    overhead_gib = (
        DOWNRES_SOURCE_CACHE_GIB +
        DOWNRES_DEST_CACHE_GIB +
        DOWNRES_RUNTIME_GIB +
        label_readback_gib
    )
    return {
        "estimate_model": model,
        "chunk_count": chunk_count,
        "scale": scale,
        "total_unique_labels": total_unique_labels,
        "output_gib": output_gib,
        "tmpfs_gib": output_gib,
        "raw_batch_gib": raw_batch_gib,
        "source_cache_gib": DOWNRES_SOURCE_CACHE_GIB,
        "dest_cache_gib": DOWNRES_DEST_CACHE_GIB,
        "runtime_gib": DOWNRES_RUNTIME_GIB,
        "label_readback_gib": label_readback_gib,
        "commit_spike_gib": commit_spike_gib,
        "overhead_gib": overhead_gib,
        "subtotal_gib": subtotal_gib,
        "safety_factor": DOWNRES_SAFETY_FACTOR,
        "total_gib": total_gib,
    }


def estimate_downres_memory_gib(chunk_count: int, scale: int) -> float:
    """Estimate memory for a downres shard using the chunk-count model."""
    return estimate_downres_components(scale, chunk_count)["total_gib"]


def estimate_downres_memory_label_aware(total_unique_labels: int,
                                       scale: int,
                                       chunk_count: int = 0) -> float:
    """Estimate memory for a downres shard using the label-aware model."""
    return estimate_downres_components(
        scale, chunk_count, total_unique_labels)["total_gib"]


def _read_shard_labels(source_path: str, scale: int) -> dict:
    """Read per-shard label totals for a scale.

    Tries labels-summary.json first (single file, written by
    aggregate_predicted_labels.py). Falls back to reading individual
    -labels.csv files if no summary exists.

    Returns:
        Dict mapping shard_hex (str) -> total_unique_labels (int).
        Empty dict if no label data found.
    """
    prefix = f"{source_path}/s{scale}/"
    summary_path = f"{prefix}labels-summary.json"

    # --- Try summary JSON first (fast path) ---
    if prefix.startswith("gs://"):
        result = subprocess.run(
            ["gsutil", "-q", "cat", summary_path],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            shard_labels = json.loads(result.stdout)
            # Convert values to int (JSON may decode as int already)
            shard_labels = {k: int(v) for k, v in shard_labels.items()}
            print(f"  Read label summary: {summary_path}"
                  f" ({len(shard_labels)} shards)")
            return shard_labels
    else:
        from pathlib import Path as P
        local_summary = P(summary_path)
        if local_summary.exists():
            shard_labels = json.loads(local_summary.read_text())
            shard_labels = {k: int(v) for k, v in shard_labels.items()}
            print(f"  Read label summary: {summary_path}"
                  f" ({len(shard_labels)} shards)")
            return shard_labels

    # --- Fallback: read individual CSV files ---
    import csv as csv_mod
    import io
    import time as _time

    shard_labels = {}

    if prefix.startswith("gs://"):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from google.cloud import storage as gcs_storage
        rest = prefix[len("gs://"):]
        bucket_name, _, blob_prefix = rest.partition("/")
        client = gcs_storage.Client()
        bucket = client.bucket(bucket_name)

        label_blobs = []
        for blob in bucket.list_blobs(prefix=blob_prefix):
            if blob.name.endswith("-labels.csv"):
                label_blobs.append(blob.name)

        if not label_blobs:
            return shard_labels

        print(f"  Reading {len(label_blobs)} label files from {prefix} ...")
        t0 = _time.monotonic()

        def _read_one(blob_name):
            text = bucket.blob(blob_name).download_as_text()
            shard_hex = blob_name.split("/")[-1].replace("-labels.csv", "")
            total = 0
            for row in csv_mod.DictReader(io.StringIO(text)):
                total += int(row["unique_labels"])
            return shard_hex, total

        read_count = 0
        with ThreadPoolExecutor(max_workers=64) as pool:
            futures = {pool.submit(_read_one, name): name
                       for name in label_blobs}
            for future in as_completed(futures):
                shard_hex, total = future.result()
                shard_labels[shard_hex] = total
                read_count += 1
                if read_count % 500 == 0:
                    print(f"    {read_count}/{len(label_blobs)} label files"
                          f" ({_time.monotonic() - t0:.0f}s)")
        print(f"    {read_count} label files read ({_time.monotonic() - t0:.1f}s)")
    else:
        from pathlib import Path as P
        p = P(prefix)
        if p.exists():
            for f in p.glob("*-labels.csv"):
                shard_hex = f.stem.replace("-labels", "")
                total = 0
                with open(f) as fh:
                    for row in csv_mod.DictReader(fh):
                        total += int(row["unique_labels"])
                shard_labels[shard_hex] = total

    return shard_labels


def _list_existing_ng_shards(dest_path: str, scale_key: str) -> list[int]:
    """List existing NG shard numbers for a scale from the destination path."""
    prefix = f"{dest_path}/{scale_key}/"
    shard_numbers = set()

    result = subprocess.run(
        ["gsutil", "ls", prefix],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.endswith(".shard"):
            continue
        name = line.split("/")[-1].replace(".shard", "")
        try:
            shard_numbers.add(int(name, 16))
        except ValueError:
            continue

    return sorted(shard_numbers)


def generate_downres_manifests(
    ng_spec_path: str,
    source_path: str,
    dest_path: str,
    scales: list,
    downres_scales: list,
    max_tasks: dict,
    only_missing: bool = False,
    dry_run: bool = False,
) -> dict:
    """Generate manifest chain for downres scales.

    The s0 DVID source files determine s0 shards, which determine s1
    shards, which determine s2 shards, etc.  The full chain is computed
    at manifest-generation time.

    Args:
        ng_spec_path: Path to NG spec JSON file.
        source_path: GCS URI to DVID Arrow shard export root.
        scales: List of DVID source scales available under SOURCE_PATH.
        downres_scales: List of target scales to generate (e.g., [1, 2, 3]).
        max_tasks: Dict mapping tier_gib -> max tasks.
        dry_run: If True, don't write to GCS.

    Returns:
        Dict mapping target_scale -> {tier_gib -> (manifest_uri, num_tasks)}
    """
    from src.ng_sharding import (
        load_ng_spec,
        parent_shards_to_child_shards,
        shard_bbox,
        dvid_to_ng_shard_number,
    )

    spec = load_ng_spec(ng_spec_path)
    downres_scales = sorted(downres_scales)

    # Step 1: Seed the derivation chain.
    # Only scan DVID source shards when s1 is being generated in this run.
    # For each downres scale, derive child shards from parent shards.
    # shard_numbers_by_scale[scale] = sorted shard numbers known to exist.
    shard_numbers_by_scale = {}
    if downres_scales and min(downres_scales) <= 1 and scales:
        print("\nScanning source shards for derivation chain...")
        source_shard_numbers = set()
        for scale in scales:
            params = spec[scale]
            prefix = f"{source_path}/s{scale}/"
            result = subprocess.run(
                ["gsutil", "ls", prefix],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"  Warning: could not list {prefix}: {result.stderr.strip()}")
                continue
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.endswith(".arrow"):
                    name = line.split("/")[-1].replace(".arrow", "")
                    ng_shard = dvid_to_ng_shard_number(name, params)
                    source_shard_numbers.add(ng_shard)

        shard_numbers_by_scale[scales[0]] = sorted(source_shard_numbers)
        print(f"  Found {len(source_shard_numbers)} unique NG shard numbers "
              f"at source scale(s)")
    elif downres_scales:
        start_scale = min(downres_scales)
        print(f"\nSkipping source shard scan; s{start_scale} will seed from "
              f"existing destination s{start_scale - 1} output")

    all_scale_results = {}  # scale -> {tier_gib -> (uri, num_tasks)}

    for target_scale in downres_scales:
        parent_scale = target_scale - 1
        if parent_scale not in shard_numbers_by_scale:
            parent_params = spec[parent_scale]
            existing_parent_shards = _list_existing_ng_shards(
                dest_path, parent_params["key"])
            if existing_parent_shards:
                shard_numbers_by_scale[parent_scale] = existing_parent_shards
                print(f"  Seeded s{parent_scale} from destination: "
                      f"{len(existing_parent_shards)} existing NG shards")
            else:
                print(f"  Warning: no parent shards for scale {target_scale}, skipping")
                continue

        parent_params = spec[parent_scale]
        child_params = spec[target_scale]

        parent_shards = shard_numbers_by_scale[parent_scale]
        child_shards = parent_shards_to_child_shards(
            parent_shards, parent_params, child_params
        )
        shard_numbers_by_scale[target_scale] = child_shards

        print(f"\n  Scale {target_scale}: {len(child_shards)} output shards "
              f"(from {len(parent_shards)} parent shards at s{parent_scale})")

        if only_missing:
            existing_target_shards = set(
                _list_existing_ng_shards(dest_path, child_params["key"]))
            missing_child_shards = [
                sn for sn in child_shards if sn not in existing_target_shards
            ]
            print(f"  Filtering to missing output shards at s{target_scale}: "
                  f"{len(missing_child_shards)} missing / {len(child_shards)} total")
            child_shards = missing_child_shards

        # Step 3: Compute shard bboxes and estimate memory.
        # Try label-aware model first (much tighter); fall back to chunk-count.
        shard_labels = _read_shard_labels(source_path, target_scale)
        if shard_labels:
            hex_digits = -(-child_params["shard_bits"] // 4)
            print(f"  Using label-aware model ({len(shard_labels)} shard label files"
                  f" for tier assignment)")
        else:
            print(f"  No label files for s{target_scale},"
                  f" using chunk-count model for tier assignment")

        shard_entries = []  # (scale, shard_number, estimate_dict, bbox_dict)
        for sn in child_shards:
            bbox = shard_bbox(sn, child_params)
            shard_hex = f"{sn:0{hex_digits}x}" if shard_labels else ""
            if shard_hex in shard_labels:
                estimate = estimate_downres_components(
                    target_scale, bbox["num_chunks"], shard_labels[shard_hex])
            else:
                estimate = estimate_downres_components(
                    target_scale, bbox["num_chunks"])
            shard_entries.append((target_scale, sn, estimate, bbox))

        # Step 4: Assign to tiers.
        tier_map = {}  # tier_gib -> list of (scale, shard_number, mem, bbox)
        for entry in shard_entries:
            mem = entry[2]["total_gib"]
            gib = pick_tier(mem)
            tier_map.setdefault(gib, []).append(entry)

        if not child_shards:
            print("  No output shards selected after filtering; skipping")
            all_scale_results[target_scale] = {}
            continue

        print("  Tier assignments:")
        tier_info = {}
        stale_deleted = False
        for gib in sorted(tier_map.keys()):
            entries = tier_map[gib]
            tier_max = max_tasks.get(gib, 1000)
            num_tasks = min(tier_max, len(entries))
            cpu = TIER_CPU.get(gib, 2)
            total_chunks = sum(e[3]["num_chunks"] for e in entries)
            print(f"    {gib}Gi (cpu={cpu}): {len(entries)} shards, "
                  f"{num_tasks} tasks, {total_chunks:,} chunks")

            if dry_run:
                continue

            # Delete stale manifests for this scale (once per scale)
            if not stale_deleted:
                scale_uri = f"{source_path}/manifests-downres/s{target_scale}/"
                result = subprocess.run(
                    ["gsutil", "-m", "-q", "rm", "-r", scale_uri],
                    capture_output=True, text=True,
                )
                if result.returncode == 0:
                    print(f"    Deleted stale manifests under {scale_uri}")
                stale_deleted = True

            # Distribute shards across tasks using greedy load balancing
            # (balance by num_chunks as a proxy for work)
            tasks_data = _distribute_downres_tasks(entries, num_tasks)

            # Write per-task manifests to GCS
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from google.cloud import storage as gcs_storage
            bucket_name, source_prefix = source_path.replace("gs://", "").split("/", 1)
            client = gcs_storage.Client()
            bucket = client.bucket(bucket_name)

            tier_prefix = f"{source_prefix}/manifests-downres/s{target_scale}/tier-{gib}gi"
            tier_uri = f"{source_path}/manifests-downres/s{target_scale}/tier-{gib}gi"

            def _upload_manifest(item):
                task_idx, shard_list = item
                blob = bucket.blob(f"{tier_prefix}/task-{task_idx}.json")
                blob.upload_from_string(
                    json.dumps(shard_list, separators=(",", ":")),
                    content_type="application/json",
                )

            uploaded = 0
            with ThreadPoolExecutor(max_workers=32) as pool:
                futures = {
                    pool.submit(_upload_manifest, item): item
                    for item in tasks_data.items()
                }
                for future in as_completed(futures):
                    future.result()
                    uploaded += 1
                    if uploaded % 500 == 0 or uploaded == len(tasks_data):
                        print(f"    Writing manifests: {uploaded}/{len(tasks_data)}")

            print(f"    Written {len(tasks_data)} task manifests: {tier_uri}/")
            tier_info[gib] = (tier_uri, len(tasks_data))

        all_scale_results[target_scale] = tier_info

    return all_scale_results


def _distribute_downres_tasks(entries: list, num_tasks: int) -> dict:
    """Distribute downres shard entries across tasks.

    Balance by num_chunks (proxy for work/memory).

    Args:
        entries: list of (scale, shard_number, mem, bbox_dict)
        num_tasks: number of tasks to distribute across

    Returns:
        dict mapping task_index (str) -> list of manifest entry dicts
    """
    if num_tasks == 0:
        return {}

    num_tasks = min(num_tasks, len(entries))
    heap = [(0, i) for i in range(num_tasks)]
    tasks = {str(i): [] for i in range(num_tasks)}

    # Process largest shards first for better balance
    for scale, sn, estimate, bbox in sorted(entries, key=lambda e: -e[3]["num_chunks"]):
        cum_chunks, task_idx = heapq.heappop(heap)
        tasks[str(task_idx)].append({
            "scale": scale,
            "shard_number": sn,
            "shard_origin": bbox["shard_origin"],
            "shard_extent": bbox["shard_extent"],
            "num_chunks": bbox["num_chunks"],
            "estimate_model": estimate["estimate_model"],
            "estimated_memory_gib": round(estimate["total_gib"], 3),
            "estimated_subtotal_gib": round(estimate["subtotal_gib"], 3),
            "estimated_output_gib": round(estimate["output_gib"], 3),
            "estimated_tmpfs_gib": round(estimate["tmpfs_gib"], 3),
            "estimated_raw_batch_gib": round(estimate["raw_batch_gib"], 3),
            "estimated_overhead_gib": round(estimate["overhead_gib"], 3),
            "estimated_source_cache_gib": round(estimate["source_cache_gib"], 3),
            "estimated_dest_cache_gib": round(estimate["dest_cache_gib"], 3),
            "estimated_runtime_gib": round(estimate["runtime_gib"], 3),
            "estimated_label_readback_gib": round(
                estimate["label_readback_gib"], 3),
            "estimated_commit_spike_gib": round(
                estimate["commit_spike_gib"], 3),
            "estimated_safety_factor": estimate["safety_factor"],
            "estimated_total_unique_labels": estimate["total_unique_labels"],
        })
        heapq.heappush(heap, (cum_chunks + bbox["num_chunks"], task_idx))

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Precompute tier-based task manifests for Cloud Run export jobs.",
    )
    parser.add_argument(
        "--tiers",
        help="Override max tasks per tier as GiB:maxTasks pairs, comma-separated. "
             "E.g., 4:3000,8:50,16:10.  Tiers are auto-selected from "
             "1,2,4,8,16,32 GiB based on shard memory estimates.",
    )
    parser.add_argument(
        "--scales",
        help="Comma-separated scales to include (default: from .env SCALES)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print tier assignments without writing manifests to GCS",
    )
    parser.add_argument(
        "--downres-scales",
        help="Comma-separated target scales for downres manifest generation. "
             "E.g., 1,2,3.  Derives output shard lists from parent scale "
             "shards using the manifest chain approach.",
    )
    parser.add_argument(
        "--only-missing", action="store_true",
        help="For downres manifests, include only output shards missing from "
             "DEST_PATH at the target scale.",
    )
    parser.add_argument(
        "--exclude-empty", type=str, default=None,
        help="Path to JSON file listing empty shards to exclude. "
             "Format: [{\"scale\": 0, \"shard\": \"name\"}, ...]. "
             "Generate with: scripts/check_empty_shards.py --report ... --output-empty",
    )
    args = parser.parse_args()

    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)

    source_path = env.get("SOURCE_PATH", "")
    dest_path = env.get("DEST_PATH", "")
    if not source_path or not dest_path:
        print("Error: SOURCE_PATH and DEST_PATH must be configured in .env")
        sys.exit(1)

    scales_str = args.scales or env.get("SCALES", "0")
    scales = [int(s.strip()) for s in scales_str.split(",")]

    # Parse tier overrides
    max_tasks = dict(DEFAULT_TIER_MAX_TASKS)
    if args.tiers:
        for pair in args.tiers.split(","):
            gib_s, _, tasks_s = pair.partition(":")
            max_tasks[int(gib_s)] = int(tasks_s) if tasks_s else 1000

    # --- Downres manifest generation mode ---
    if args.downres_scales:
        ng_spec_path = env.get("NG_SPEC_PATH", "")
        if not ng_spec_path:
            print("Error: NG_SPEC_PATH must be configured in .env")
            sys.exit(1)
        spec_path = Path(ng_spec_path)
        if not spec_path.is_absolute():
            spec_path = Path(__file__).resolve().parent.parent / spec_path

        downres_scales = [int(s.strip()) for s in args.downres_scales.split(",")]

        print(f"Generating downres manifests for scales {downres_scales}")
        print(f"  Source scales: {scales}")
        print(f"  NG spec: {spec_path}")

        results = generate_downres_manifests(
            str(spec_path), source_path, dest_path, scales, downres_scales,
            max_tasks, only_missing=args.only_missing, dry_run=args.dry_run,
        )

        if args.dry_run:
            print("\n(dry run — no manifests written)")
        else:
            print("\nDownres manifest summary:")
            for target_scale in sorted(results.keys()):
                tier_info = results[target_scale]
                for gib in sorted(tier_info.keys()):
                    uri, num_tasks = tier_info[gib]
                    print(f"  s{target_scale} {gib}Gi: {num_tasks} tasks → {uri}/")
        return

    # Scan Arrow files
    print(f"Scanning Arrow files across {len(scales)} scales...")
    all_files = list_arrow_files(source_path, scales)
    print(f"  Found {len(all_files)} Arrow files")
    print("  Memory formula: arrow + 2 * shard_on_tmpfs + 2 GiB")

    if not all_files:
        print("No Arrow files found. Check SOURCE_PATH and SCALES in .env.")
        sys.exit(1)

    # Exclude known-empty shards (all-zero labels/supervoxels in Arrow metadata).
    # These produce no NG output since TensorStore skips fill-value-only writes.
    if args.exclude_empty:
        with open(args.exclude_empty) as f:
            empty_list = json.load(f)
        empty_set = {(e["scale"], e["shard"]) for e in empty_list}
        before = len(all_files)
        all_files = [f for f in all_files if (f[0], f[1]) not in empty_set]
        excluded = before - len(all_files)
        print(f"  Excluded {excluded} empty shards (from {args.exclude_empty})")

    # Assign to tiers
    tier_map = assign_tiers(all_files, max_tasks)

    # Build chunk count lookup: (scale, shard_name) -> chunk_count
    chunk_counts = {(s, n): cc for s, n, _, cc in all_files}
    total_chunks_all = sum(cc for cc in chunk_counts.values())
    print(f"  Total chunks across all shards: {total_chunks_all:,}")

    # Print summary and write per-task manifests
    print("\nTier assignments:")
    manifest_prefixes = {}
    tier_summary = {}  # for summary.json
    storage_client = None
    bucket_name, source_prefix = source_path.replace("gs://", "").split("/", 1)

    for gib in sorted(tier_map.keys()):
        shards = tier_map[gib]
        tier_max = max_tasks.get(gib, 1000)
        tasks = distribute_tasks(shards, tier_max)
        num_tasks = len(tasks)
        total_bytes = sum(s for _, _, s in shards)
        max_arrow = max(s for _, _, s in shards)
        cpu = TIER_CPU.get(gib, 2)

        tier_chunks = sum(chunk_counts.get((s, n), 0) for s, n, _ in shards)
        tier_summary[str(gib)] = {
            "shards": len(shards),
            "tasks": num_tasks,
            "chunks": tier_chunks,
        }

        print(f"  {gib}Gi (cpu={cpu}): {len(shards)} shards, {num_tasks} tasks, "
              f"{tier_chunks:,} chunks, "
              f"total={total_bytes/1e9:.1f}GB, max_arrow={max_arrow/1e6:.0f}MB")

        if args.dry_run:
            continue

        # Write one manifest file per task under source_path/manifests/tier-{N}gi/
        if storage_client is None:
            from google.cloud import storage
            storage_client = storage.Client()

        tier_prefix = f"{source_prefix}/manifests/tier-{gib}gi"
        tier_uri = f"{source_path}/manifests/tier-{gib}gi"
        gcs_bucket = storage_client.bucket(bucket_name)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _upload_manifest(item):
            task_idx, shard_list = item
            blob = gcs_bucket.blob(f"{tier_prefix}/task-{task_idx}.json")
            blob.upload_from_string(
                json.dumps(shard_list, separators=(",", ":")),
                content_type="application/json",
            )

        uploaded = 0
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = {
                pool.submit(_upload_manifest, item): item
                for item in tasks.items()
            }
            for future in as_completed(futures):
                future.result()
                uploaded += 1
                if uploaded % 500 == 0 or uploaded == num_tasks:
                    print(f"    Writing manifests: {uploaded}/{num_tasks}")

        print(f"    Written {num_tasks} task manifests: {tier_uri}/task-*.json")
        manifest_prefixes[gib] = tier_uri

    if args.dry_run:
        print("\n(dry run — no manifests written)")
        return

    # Write summary.json with chunk totals for export-status progress tracking
    summary = {
        "total_shards": len(all_files),
        "total_chunks": total_chunks_all,
        "tiers": tier_summary,
    }
    summary_blob = gcs_bucket.blob(f"{source_prefix}/manifests/summary.json")
    summary_blob.upload_from_string(
        json.dumps(summary, indent=2), content_type="application/json")
    print(f"\n  Written summary: {source_path}/manifests/summary.json")

    # Print next step
    print("\nNext step:")
    print("  pixi run export --wait")


if __name__ == "__main__":
    main()
