#!/usr/bin/env python3
"""
Detect and optionally delete empty DVID Arrow shard files from GCS.

Empty shards have all-zero labels and supervoxels in every chunk. DVID's
export-shards emits them for spatial regions within the bounding box that
contain no segmentation labels. They produce no neuroglancer output and
waste export compute time.

Detection uses two stages:
  1. Fast heuristic: Arrow file size / chunk count < 1000 bytes/chunk.
     Empty shards compress to ~660 bytes/chunk (Arrow overhead + zstd
     zero blocks). Real shards are 5,000+ bytes/chunk.
  2. Arrow metadata verification: for heuristic candidates, check that
     all chunks have empty labels and supervoxels lists.

Stage 1 runs locally (only needs gsutil ls + CSV sizes from
precompute_manifest.list_arrow_files). Stage 2 runs as a Cloud Run job
for fast same-region GCS access.

Usage:
    pixi run remove-empty-shards                    # detect only (dry run)
    pixi run remove-empty-shards -- --delete        # detect + delete + log
    pixi run remove-empty-shards -- --scales 0,1    # specific scales
"""

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE
from scripts.precompute_manifest import list_arrow_files

# Empty shards have ~660 bytes/chunk. Real shards have 5,000+.
# Use a conservative threshold that catches empties without false positives.
EMPTY_BYTES_PER_CHUNK_THRESHOLD = 1000


def find_candidates(all_files: list) -> list:
    """Find shards that are likely empty based on bytes-per-chunk ratio.

    Args:
        all_files: list of (scale, shard_name, size_bytes, chunk_count)

    Returns:
        list of (scale, shard_name, size_bytes, chunk_count, bytes_per_chunk)
        for shards below the threshold.
    """
    candidates = []
    for scale, name, size_bytes, chunk_count in all_files:
        if chunk_count == 0:
            continue
        bpc = size_bytes / chunk_count
        if bpc < EMPTY_BYTES_PER_CHUNK_THRESHOLD:
            candidates.append((scale, name, size_bytes, chunk_count, bpc))
    return candidates


def verify_via_cloud_run(candidates: list, env: dict, source_path: str) -> set:
    """Launch a Cloud Run job to verify candidates are truly empty.

    Returns set of (scale, shard_name) that are confirmed empty.
    """
    from google.cloud import storage

    image = env.get("DOCKER_IMAGE", "")
    if not image:
        print("Error: DOCKER_IMAGE not set in .env. Run `pixi run deploy` first.")
        sys.exit(1)

    project = env.get("PROJECT_ID", "")
    region = env.get("REGION", "us-central1")
    bucket_name, source_prefix = source_path.replace("gs://", "").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Upload per-task manifests
    shards_per_task = 50
    shard_list = [{"scale": s, "shard": n} for s, n, _, _, _ in candidates]
    num_tasks = max(1, math.ceil(len(shard_list) / shards_per_task))

    manifest_prefix = f"{source_prefix}/manifests-check-empty"
    for i in range(num_tasks):
        chunk = shard_list[i * shards_per_task:(i + 1) * shards_per_task]
        blob = bucket.blob(f"{manifest_prefix}/task-{i}.json")
        blob.upload_from_string(
            json.dumps(chunk, separators=(",", ":")),
            content_type="application/json",
        )

    manifest_uri = f"{source_path}/manifests-check-empty"
    job_name = "check-empty-shards"

    # Delete existing job
    subprocess.run(
        ["gcloud", "run", "jobs", "delete", job_name,
         "--project", project, "--region", region, "--quiet"],
        capture_output=True,
    )

    # Create and execute
    print(f"  Launching verification job ({num_tasks} tasks, {len(candidates)} shards)...")
    create_cmd = [
        "gcloud", "run", "jobs", "create", job_name,
        "--project", project, "--region", region,
        "--image", image,
        "--tasks", str(num_tasks),
        "--task-timeout", "600s",
        "--max-retries", "1",
        "--memory", "2Gi", "--cpu", "1",
        "--set-env-vars",
        f"SOURCE_PATH={source_path},MANIFEST_URI={manifest_uri}",
        "--command", "python",
        "--args", "scripts/check_empty_shards.py,--worker",
    ]
    result = subprocess.run(create_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error creating job: {result.stderr[:300]}")
        sys.exit(1)

    result = subprocess.run(
        ["gcloud", "run", "jobs", "execute", job_name,
         "--project", project, "--region", region, "--wait"],
    )
    if result.returncode != 0:
        print("Error: verification job failed.")
        sys.exit(1)

    # Collect confirmed empties from Cloud Logging
    print("  Collecting results from Cloud Logging...")
    log_result = subprocess.run(
        ["gcloud", "logging", "read",
         'resource.type="cloud_run_job" AND '
         'resource.labels.job_name="check-empty-shards" AND '
         'jsonPayload.event="Shard is empty"',
         "--project", project,
         "--limit", "50000",
         "--format", "json"],
        capture_output=True, text=True,
    )
    if log_result.returncode != 0:
        print(f"Error reading logs: {log_result.stderr[:200]}")
        sys.exit(1)

    entries = json.loads(log_result.stdout)
    confirmed = set()
    for entry in entries:
        jp = entry.get("jsonPayload", {})
        if jp.get("shard") and jp.get("scale") is not None:
            confirmed.add((jp["scale"], jp["shard"]))

    return confirmed


def delete_shards(confirmed: set, source_path: str, env: dict):
    """Delete confirmed empty Arrow+CSV files from GCS and write a deletion log.

    The log is written to {source_path}/deleted-empty-shards-{timestamp}.json.
    """
    from google.cloud import storage

    bucket_name, source_prefix = source_path.replace("gs://", "").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    deleted = []
    errors = []

    for scale, shard_name in sorted(confirmed):
        arrow_path = f"{source_prefix}/s{scale}/{shard_name}.arrow"
        csv_path = f"{source_prefix}/s{scale}/{shard_name}.csv"

        # Safety check: verify Arrow file is small before deleting
        arrow_blob = bucket.blob(arrow_path)
        arrow_blob.reload()
        arrow_size = arrow_blob.size or 0

        csv_blob = bucket.blob(csv_path)
        csv_exists = csv_blob.exists()

        # Estimate chunk count from CSV size
        if csv_exists:
            csv_blob.reload()
            csv_size = csv_blob.size or 0
            chunk_count = max(1, csv_size // 15 - 1) if csv_size > 0 else 1
        else:
            chunk_count = 1

        bpc = arrow_size / chunk_count if chunk_count > 0 else arrow_size

        if bpc >= EMPTY_BYTES_PER_CHUNK_THRESHOLD:
            errors.append({
                "scale": scale,
                "shard": shard_name,
                "reason": f"bytes_per_chunk={bpc:.0f} >= {EMPTY_BYTES_PER_CHUNK_THRESHOLD} "
                          f"(arrow={arrow_size}, chunks={chunk_count})",
            })
            print(f"  SKIPPED s{scale}/{shard_name}: "
                  f"bytes/chunk={bpc:.0f} too large, refusing to delete")
            continue

        # Delete
        arrow_blob.delete()
        if csv_exists:
            csv_blob.delete()

        deleted.append({
            "scale": scale,
            "shard": shard_name,
            "arrow_bytes": arrow_size,
            "chunk_count": chunk_count,
            "bytes_per_chunk": round(bpc, 1),
        })

    # Write deletion log to GCS
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_entry = {
        "timestamp": timestamp,
        "source_path": source_path,
        "threshold_bytes_per_chunk": EMPTY_BYTES_PER_CHUNK_THRESHOLD,
        "deleted_count": len(deleted),
        "skipped_count": len(errors),
        "deleted": deleted,
        "skipped": errors,
    }

    log_blob_path = f"{source_prefix}/deleted-empty-shards-{timestamp}.json"
    log_blob = bucket.blob(log_blob_path)
    log_blob.upload_from_string(
        json.dumps(log_entry, indent=2),
        content_type="application/json",
    )

    print(f"\n  Deleted {len(deleted)} empty shards ({len(errors)} skipped)")
    print(f"  Deletion log: gs://{bucket_name}/{log_blob_path}")

    return deleted, errors


def main():
    parser = argparse.ArgumentParser(
        description="Detect and optionally delete empty DVID Arrow shard files.",
    )
    parser.add_argument(
        "--scales",
        help="Comma-separated scales to check (default: from .env SCALES)",
    )
    parser.add_argument(
        "--delete", action="store_true",
        help="Delete confirmed empty shards from GCS (default: detect only)",
    )
    args = parser.parse_args()

    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)
    source_path = env.get("SOURCE_PATH", "").rstrip("/")
    if not source_path:
        print("Error: SOURCE_PATH not set in .env")
        sys.exit(1)

    scales_str = args.scales or env.get("SCALES", "0")
    scales = [int(s.strip()) for s in scales_str.split(",")]

    # Stage 1: Fast heuristic scan (local, uses gsutil ls)
    print(f"Scanning Arrow files across {len(scales)} scales...")
    all_files = list_arrow_files(source_path, scales)
    print(f"  Found {len(all_files)} Arrow files")

    candidates = find_candidates(all_files)
    print(f"  {len(candidates)} candidates below {EMPTY_BYTES_PER_CHUNK_THRESHOLD} "
          f"bytes/chunk threshold")

    if not candidates:
        print("\nNo empty shard candidates found.")
        return

    # Show breakdown
    by_scale = {}
    for s, n, sz, cc, bpc in candidates:
        by_scale.setdefault(s, []).append((n, sz, cc, bpc))
    for s in sorted(by_scale):
        items = by_scale[s]
        total_bytes = sum(sz for _, sz, _, _ in items)
        print(f"  Scale {s}: {len(items)} candidates, "
              f"{total_bytes / 1e6:.1f} MB total")

    # Stage 2: Verify via Cloud Run
    print(f"\nVerifying {len(candidates)} candidates via Cloud Run...")
    confirmed = verify_via_cloud_run(candidates, env, source_path)
    print(f"  Confirmed empty: {len(confirmed)}")

    # Candidates that passed heuristic but are NOT empty
    false_positives = set((s, n) for s, n, _, _, _ in candidates) - confirmed
    if false_positives:
        print(f"  False positives (small but not empty): {len(false_positives)}")
        for s, n in sorted(false_positives)[:5]:
            print(f"    s{s}/{n}")

    if not confirmed:
        print("\nNo confirmed empty shards.")
        return

    if not args.delete:
        print(f"\n{len(confirmed)} empty shards found. "
              f"Run with --delete to remove them from GCS.")
        return

    # Stage 3: Delete with safety checks
    print(f"\nDeleting {len(confirmed)} confirmed empty shards...")
    deleted, errors = delete_shards(confirmed, source_path, env)


if __name__ == "__main__":
    main()
