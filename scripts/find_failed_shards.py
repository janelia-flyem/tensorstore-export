#!/usr/bin/env python3
"""
Identify failed shards from a Cloud Run export and create retry manifests.

Compares completed shards (from Cloud Logging "Shard complete" events) against
the full set of source Arrow files to find shards that weren't processed.
Optionally writes retry manifests at a specified memory tier.

Usage:
    pixi run find-failed                              # show failed shards
    pixi run find-failed -- --retry-tier 16           # create retry manifests
    pixi run find-failed -- --retry-tier 16 --max-tasks 500
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE
from scripts.export_status import discover_jobs, get_latest_execution
from scripts.precompute_manifest import (
    TIER_CPU,
    list_arrow_files,
    distribute_tasks,
)


def query_completed_shards(job_name, project, region, execution, limit=50000):
    """Query 'Shard complete' log events to get completed (scale, shard) set.

    Returns (completed_set, was_truncated).
    """
    parts = [
        'resource.type="cloud_run_job"',
        f'resource.labels.job_name="{job_name}"',
        'textPayload=~"Shard complete"',
    ]
    if execution:
        parts.append(
            f'labels."run.googleapis.com/execution_name"="{execution}"'
        )

    cmd = [
        "gcloud", "logging", "read", " AND ".join(parts),
        f"--project={project}",
        f"--limit={limit}",
        "--format=json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return set(), False

    completed = set()
    try:
        entries = json.loads(result.stdout)
    except json.JSONDecodeError:
        return set(), False

    for entry in entries:
        text = entry.get("textPayload", "")
        idx = text.find("{")
        if idx == -1:
            continue
        try:
            payload = json.loads(text[idx:])
            scale = payload.get("scale")
            shard = payload.get("shard")
            if scale is not None and shard:
                completed.add((int(scale), shard))
        except (json.JSONDecodeError, ValueError):
            pass

    return completed, len(entries) >= limit


def main():
    parser = argparse.ArgumentParser(
        description="Identify failed shards and create retry manifests.",
    )
    parser.add_argument(
        "--retry-tier", type=int,
        help="Memory tier (GiB) for retry manifests. "
             "If not set, just prints the failed shard list.",
    )
    parser.add_argument(
        "--max-tasks", type=int, default=5000,
        help="Max tasks for retry tier (default: 5000)",
    )
    parser.add_argument(
        "--manifest-subdir", default="manifests-retry",
        help="GCS subdirectory for retry manifests (default: manifests-retry)",
    )
    parser.add_argument(
        "--scales",
        help="Comma-separated scales (default: from .env SCALES)",
    )
    args = parser.parse_args()

    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)
    source_path = env.get("SOURCE_PATH", "")
    project = env.get("PROJECT_ID", "")
    region = env.get("REGION", "us-central1")
    base_name = env.get("BASE_JOB_NAME", "")

    if not source_path or not project:
        print("Error: SOURCE_PATH and PROJECT_ID must be configured in .env")
        sys.exit(1)

    scales_str = args.scales or env.get("SCALES", "0")
    scales = [int(s.strip()) for s in scales_str.split(",")]

    # Step 1: List all source Arrow files
    print(f"Scanning source shards across scales {scales}...")
    all_files = list_arrow_files(source_path, scales)
    source_shards = {(scale, name): size for scale, name, size, _ in all_files}
    print(f"  Found {len(source_shards)} source shards")

    # Step 2: Query completed shards from all tier jobs
    jobs = discover_jobs(base_name, project, region)
    if not jobs:
        print(f"No jobs found matching {base_name}-*")
        sys.exit(1)

    print(f"\nQuerying completed shards from {len(jobs)} jobs...")
    completed = set()
    any_truncated = False

    for label, job_name in jobs:
        execution = get_latest_execution(job_name, project, region)
        if not execution:
            print(f"  {label}: no execution found, skipping")
            continue
        job_completed, truncated = query_completed_shards(
            job_name, project, region, execution
        )
        print(f"  {label}: {len(job_completed)} completed shards")
        completed |= job_completed
        if truncated:
            any_truncated = True
            print("    WARNING: log query hit limit, results may be incomplete")

    print(f"\nTotal completed: {len(completed)} shards")

    # Step 3: Compute failed set
    failed_keys = set(source_shards.keys()) - completed
    failed_by_scale = defaultdict(list)
    for scale, name in sorted(failed_keys):
        size = source_shards.get((scale, name), 0)
        failed_by_scale[scale].append((scale, name, size))

    print(f"Failed shards: {len(failed_keys)}")
    for scale in sorted(failed_by_scale.keys()):
        shards = failed_by_scale[scale]
        total_size = sum(s for _, _, s in shards)
        print(f"  s{scale}: {len(shards)} shards ({total_size / 1e9:.1f} GB)")

    if not failed_keys:
        print("\nAll shards completed successfully!")
        return

    if any_truncated:
        print("\nWARNING: One or more log queries hit the 50k entry limit. "
              "Some completed shards may have been missed, inflating the "
              "failed count. Consider verifying with --scales for specific scales.")

    # Step 4: Create retry manifests (if requested)
    if args.retry_tier is None:
        print("\nTo create retry manifests, re-run with --retry-tier <GiB>")
        print("  Example: pixi run find-failed -- --retry-tier 16")
        return

    gib = args.retry_tier
    if gib not in TIER_CPU:
        print(f"Error: tier {gib} not valid. Choose from: {sorted(TIER_CPU.keys())}")
        sys.exit(1)

    cpu = TIER_CPU[gib]
    all_failed = []
    for shards in failed_by_scale.values():
        all_failed.extend(shards)

    tasks = distribute_tasks(all_failed, args.max_tasks)
    num_tasks = len(tasks)

    print("\nCreating retry manifests:")
    print(f"  Tier: {gib}Gi (cpu={cpu})")
    print(f"  Shards: {len(all_failed)}")
    print(f"  Tasks: {num_tasks}")

    # Write to GCS
    from google.cloud import storage
    bucket_name, source_prefix = source_path.replace("gs://", "").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    subdir = args.manifest_subdir.strip("/")
    tier_prefix = f"{source_prefix}/{subdir}/tier-{gib}gi"
    tier_uri = f"{source_path}/{subdir}/tier-{gib}gi"

    # Delete any stale retry manifests for this tier
    old_blobs = list(bucket.list_blobs(prefix=f"{tier_prefix}/"))
    if old_blobs:
        print(f"  Deleting {len(old_blobs)} stale retry manifests...")
        for blob in old_blobs:
            blob.delete()

    for task_idx, shard_list in tasks.items():
        blob = bucket.blob(f"{tier_prefix}/task-{task_idx}.json")
        blob.upload_from_string(
            json.dumps(shard_list, separators=(",", ":")),
            content_type="application/json",
        )

    print(f"  Written {num_tasks} manifests: {tier_uri}/task-*.json")
    print("\nTo launch retry:")
    print(f"  pixi run export -- --manifest-dir {subdir}")


if __name__ == "__main__":
    main()
