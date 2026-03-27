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


def estimate_memory_gib(arrow_size_bytes: int, chunk_count: int = 0,
                        scale: int = 0) -> float:
    """Estimate total memory needed to process a shard.

    Cloud Run Gen 2 uses an in-memory filesystem (tmpfs), NOT disk-backed
    storage.  Writes to /mnt/staging consume the container's memory budget.
    The output neuroglancer shard file lives on tmpfs and grows as chunks are
    committed via TensorStore's batched read-modify-write.

    Memory components:
      - Arrow file loaded into RAM by BRAID (~1× file size)
      - Output shard file on tmpfs (chunks × KB_per_chunk)
      - TensorStore RMW peak: old + new shard in memory (~2× shard size)
      - Additive headroom for Python runtime, libraries, GCS client

    KB/chunk rates are the observed MAX from the mCNS v0.11 export
    (25,541 shards, March 2026).  See analysis/v011_shard_memory.csv.
    """
    # Observed max compressed KB per chunk in output shard, by scale.
    # From: pixi run analyze-memory (v0.11 production data, 25,541 shards).
    KB_PER_CHUNK = {
        0: 14,    # p95=7,   max=14
        1: 27,    # p95=21,  max=27
        2: 43,    # p95=39,  max=43
        3: 76,    # p95=67,  max=76
        4: 126,   # p95=116, max=126
        5: 192,   # p95=192, max=192
        6: 249,   # p95=249, max=249
        7: 162,   # p95=162, max=162
        8: 150,   # p95=150, max=150
        9: 122,   # p95=122, max=122
    }
    kb_per_chunk = KB_PER_CHUNK.get(scale, 150)

    arrow_gib = arrow_size_bytes / (1 << 30)
    shard_gib = chunk_count * kb_per_chunk / (1024 * 1024)

    # During batched RMW commit, TensorStore holds old + new shard in memory,
    # so peak tmpfs is ~2× the final shard size.  Additive 2 GiB covers
    # Python runtime, BRAID, pyarrow, GCS client, and decompression buffers.
    return arrow_gib + 2 * shard_gib + 2.0


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

    # Scan Arrow files
    print(f"Scanning Arrow files across {len(scales)} scales...")
    all_files = list_arrow_files(source_path, scales)
    print(f"  Found {len(all_files)} Arrow files")
    print(f"  Memory formula: arrow + 2 * shard_on_tmpfs + 2 GiB")

    if not all_files:
        print("No Arrow files found. Check SOURCE_PATH and SCALES in .env.")
        sys.exit(1)

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
