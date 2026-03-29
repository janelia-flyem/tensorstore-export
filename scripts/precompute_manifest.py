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
# scale, from analysis/v011_shard_memory.csv (25,541 shards).
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
# Source: v0.11 (gs://flyem-male-cns/v0.11/segmentation).
# Refresh after each new dataset: see docs/ExportShardsOptimization.md §13.
BYTES_PER_CHUNK = {
    0:  13_932,   # 13.6 KB — shard 36864_43008_108544 (32,768 chunks)
    1:  27_919,   # 27.3 KB — shard 26624_34816_49152  (24,603 chunks)
    2:  44_482,   # 43.4 KB — shard 6144_10240_26624   (1,558 chunks)
    3:  77_480,   # 75.7 KB — shard 6144_2048_2048     (32,768 chunks)
    4: 128_757,   # 125.7 KB — shard 2048_0_0          (22,565 chunks)
    5: 196_923,   # 192.3 KB — shard 0_0_2048          (6,138 chunks)
    6: 255_166,   # 249.2 KB — shard 0_0_0             (3,301 chunks)
    7: 166_127,   # 162.2 KB — shard 0_0_0             (828 chunks)
    8: 153_717,   # 150.1 KB — shard 0_0_0             (143 chunks)
    9: 125_268,   # 122.3 KB — shard 0_0_0             (29 chunks)
}

# Fixed overhead for shard processing (s0): Python + BRAID + pyarrow +
# TensorStore + GCS client + decompression buffers.  Measured from cgroup
# memory.current at worker startup before any shard processing begins.
SHARD_PROC_OVERHEAD_GIB = 2.0

# Fixed overhead for downres: Python + TensorStore + GCS client + source
# read cache (bounded by cache_pool at 256 MB).  Lighter than shard
# processing — no pyarrow/BRAID/Arrow overhead.
DOWNRES_OVERHEAD_GIB = 1.5


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


def estimate_downres_memory_gib(chunk_count: int, scale: int) -> float:
    """Estimate memory for a downres shard.

    Memory = output .shard on tmpfs + fixed overhead.

    No RMW factor: downres writes to a fresh staging dir (no pre-existing
    shard), so there's only one copy on tmpfs.  No Arrow/pyarrow/BRAID
    overhead — the source is read from GCS via byte-range requests with a
    bounded cache.
    """
    tmpfs_gib = estimate_tmpfs_gib(chunk_count, scale)
    return tmpfs_gib + DOWNRES_OVERHEAD_GIB


def generate_downres_manifests(
    ng_spec_path: str,
    source_path: str,
    scales: list,
    downres_scales: list,
    max_tasks: dict,
    dry_run: bool = False,
) -> dict:
    """Generate manifest chain for downres scales.

    The s0 DVID source files determine s0 shards, which determine s1
    shards, which determine s2 shards, etc.  The full chain is computed
    at manifest-generation time.

    Args:
        ng_spec_path: Path to NG spec JSON file.
        source_path: GCS URI to DVID Arrow shard export root.
        scales: List of s0 source scales (for deriving the initial shard set).
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

    # Step 1: Build the initial shard set from DVID Arrow source files.
    # Scan the source bucket for .arrow files at the base scales.
    print(f"\nScanning s0 source shards for derivation chain...")
    s0_shard_numbers = set()
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
                s0_shard_numbers.add(ng_shard)

    print(f"  Found {len(s0_shard_numbers)} unique NG shard numbers at source scale(s)")

    # Step 2: Build the derivation chain.
    # For each downres scale, derive child shards from parent shards.
    # parent_shard_numbers[scale] = set of shard numbers that exist at that scale
    shard_numbers_by_scale = {scales[0]: sorted(s0_shard_numbers)}

    all_scale_results = {}  # scale -> {tier_gib -> (uri, num_tasks)}

    for target_scale in downres_scales:
        parent_scale = target_scale - 1
        if parent_scale not in shard_numbers_by_scale:
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

        # Step 3: Compute shard bboxes and estimate memory.
        shard_entries = []  # (scale, shard_number, estimated_mem, bbox_dict)
        for sn in child_shards:
            bbox = shard_bbox(sn, child_params)
            mem = estimate_downres_memory_gib(bbox["num_chunks"], target_scale)
            shard_entries.append((target_scale, sn, mem, bbox))

        # Step 4: Assign to tiers.
        tier_map = {}  # tier_gib -> list of (scale, shard_number, mem, bbox)
        for entry in shard_entries:
            mem = entry[2]
            gib = pick_tier(mem)
            tier_map.setdefault(gib, []).append(entry)

        print("  Tier assignments:")
        tier_info = {}
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

            # Distribute shards across tasks using greedy load balancing
            # (balance by num_chunks as a proxy for work)
            tasks_data = _distribute_downres_tasks(entries, num_tasks)

            # Write per-task manifests to GCS
            from google.cloud import storage as gcs_storage
            bucket_name, source_prefix = source_path.replace("gs://", "").split("/", 1)
            client = gcs_storage.Client()
            bucket = client.bucket(bucket_name)

            tier_prefix = f"{source_prefix}/manifests-downres/s{target_scale}/tier-{gib}gi"
            tier_uri = f"{source_path}/manifests-downres/s{target_scale}/tier-{gib}gi"

            for task_idx, shard_list in tasks_data.items():
                blob = bucket.blob(f"{tier_prefix}/task-{task_idx}.json")
                blob.upload_from_string(
                    json.dumps(shard_list, separators=(",", ":")),
                    content_type="application/json",
                )

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
    for scale, sn, mem, bbox in sorted(entries, key=lambda e: -e[3]["num_chunks"]):
        cum_chunks, task_idx = heapq.heappop(heap)
        tasks[str(task_idx)].append({
            "scale": scale,
            "shard_number": sn,
            "shard_origin": bbox["shard_origin"],
            "shard_extent": bbox["shard_extent"],
            "num_chunks": bbox["num_chunks"],
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
            str(spec_path), source_path, scales, downres_scales,
            max_tasks, dry_run=args.dry_run,
        )

        if args.dry_run:
            print("\n(dry run — no manifests written)")
        else:
            print(f"\nDownres manifest summary:")
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
    print(f"  Memory formula: arrow + 2 * shard_on_tmpfs + 2 GiB")

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
    print(f"\nTier assignments:")
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

        for task_idx, shard_list in tasks.items():
            blob = gcs_bucket.blob(f"{tier_prefix}/task-{task_idx}.json")
            blob.upload_from_string(
                json.dumps(shard_list, separators=(",", ":")),
                content_type="application/json",
            )

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

    # Print execution commands
    scales_arg = ",".join(str(s) for s in scales)
    print(f"\nExecution commands:")
    for gib in sorted(tier_map.keys()):
        shards = tier_map[gib]
        tier_max = max_tasks.get(gib, 1000)
        num_tasks = min(tier_max, len(shards))
        cpu = TIER_CPU.get(gib, 2)
        uri = manifest_prefixes.get(gib, "")
        print(f"  pixi run generate-scale --scales {scales_arg} "
              f"--tasks {num_tasks} --memory {gib}Gi --cpu {cpu} "
              f"--manifest-uri {uri}")


if __name__ == "__main__":
    main()
