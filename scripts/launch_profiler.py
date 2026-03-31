#!/usr/bin/env python3
"""
Launch shard profiling as a Cloud Run job.

Pre-scans source and output paths to find which shards need profiling,
writes per-task manifests to GCS, then launches a Cloud Run job where
each task processes only its assigned shards.

Usage:
    pixi run launch-profiler --source gs://bucket/exports/seg --output gs://bucket2/exports/seg
    pixi run launch-profiler --source gs://... --output gs://... --tasks 100
    pixi run launch-profiler --source gs://... --output gs://... --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE
from scripts.profile_shards import list_shards, _bulk_scan_existing, _get_gcs_client, _parse_gs


def write_profiler_manifests(output_path: str, shards: list,
                             num_tasks: int) -> str:
    """Write per-task manifest files to GCS.

    Each manifest is a JSON array of [scale, shard_name] pairs.
    Returns the manifest URI prefix.
    """
    # Cap tasks to number of shards
    num_tasks = min(num_tasks, len(shards))

    # Round-robin distribute
    tasks = {i: [] for i in range(num_tasks)}
    for idx, (scale, name) in enumerate(shards):
        tasks[idx % num_tasks].append([scale, name])

    manifest_prefix = f"{output_path}/profiler-manifests"
    bucket_name, blob_prefix = _parse_gs(manifest_prefix)
    client = _get_gcs_client()
    bucket = client.bucket(bucket_name)

    def _upload(task_idx):
        blob_path = f"{blob_prefix}/task-{task_idx}.json"
        data = json.dumps(tasks[task_idx], separators=(",", ":"))
        bucket.blob(blob_path).upload_from_string(
            data, content_type="application/json")

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_upload, range(num_tasks)))

    print(f"  Written {num_tasks} manifests to {manifest_prefix}/")
    return manifest_prefix


def main():
    parser = argparse.ArgumentParser(
        description="Launch shard profiling as a Cloud Run job.",
    )
    parser.add_argument(
        "--source", required=True,
        help="GCS path to read Arrow shard files from",
    )
    parser.add_argument(
        "--output", required=True,
        help="GCS path to write -labels.csv files to",
    )
    parser.add_argument(
        "--scales", default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated scales (default: 0-9)",
    )
    parser.add_argument(
        "--tasks", type=int, default=100,
        help="Max number of Cloud Run tasks (default: 100)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Threads per task (default: 4)",
    )
    parser.add_argument(
        "--memory", default="8Gi",
        help="Memory per task (default: 8Gi)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and show what needs profiling without launching",
    )
    parser.add_argument(
        "--wait", action="store_true",
        help="Block until profiler completes, then aggregate s1 predictions",
    )
    parser.add_argument(
        "--ng-spec",
        help="Path to NG spec JSON (for post-profiler aggregation; default: NG_SPEC_PATH from .env)",
    )
    args = parser.parse_args()

    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)
    project = env.get("PROJECT_ID", "")
    region = env.get("REGION", "us-east4")
    image = env.get("DOCKER_IMAGE", "")

    if not project:
        print("Error: PROJECT_ID not set in .env")
        sys.exit(1)
    if not image:
        print("Error: DOCKER_IMAGE not set in .env. Run 'pixi run deploy' first.")
        sys.exit(1)

    scales = [int(s.strip()) for s in args.scales.split(",")]

    # --- Pre-scan: find which shards need profiling ---
    print(f"Scanning source shards in {args.source}...")
    all_shards = list_shards(args.source, scales)
    print(f"  Found {len(all_shards)} Arrow shards")

    print(f"Scanning existing -labels.csv in {args.output}...")
    existing = _bulk_scan_existing(args.output, scales)
    need_profiling = [(s, n) for s, n in all_shards if (s, n) not in existing]
    print(f"  {len(need_profiling)} shards need profiling "
          f"({len(existing)} already done)")

    if not need_profiling:
        print("All shards already profiled.")
        return

    # Cap tasks to actual shard count
    num_tasks = min(args.tasks, len(need_profiling))
    shards_per_task = len(need_profiling) / num_tasks

    # Pick CPU based on memory (Cloud Run coupling constraints)
    mem_gib = int(args.memory.replace("Gi", ""))
    if mem_gib <= 8:
        cpu = 2
    elif mem_gib <= 16:
        cpu = 4
    elif mem_gib <= 24:
        cpu = 6
    else:
        cpu = 8

    # Show per-scale breakdown
    by_scale = {}
    for scale, _ in need_profiling:
        by_scale[scale] = by_scale.get(scale, 0) + 1
    for s in sorted(by_scale):
        print(f"    scale {s}: {by_scale[s]} shards")

    print(f"\n  Tasks: {num_tasks} ({shards_per_task:.1f} shards/task)")
    print(f"  Workers/task: {args.workers}")
    print(f"  Memory: {args.memory}, CPU: {cpu}")

    if args.dry_run:
        print("\n(dry run — not launched)")
        return

    # --- Delete stale manifests from previous runs ---
    manifest_prefix = f"{args.output}/profiler-manifests/"
    print(f"\nDeleting stale manifests under {manifest_prefix} ...")
    result = subprocess.run(
        ["gsutil", "-m", "-q", "rm", "-r", manifest_prefix],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("  Deleted.")
    elif "No URLs matched" in result.stderr or "CommandException" in result.stderr:
        print("  No stale manifests found.")

    # --- Write per-task manifests to GCS ---
    print("Writing manifests...")
    manifest_uri = write_profiler_manifests(
        args.output, need_profiling, num_tasks)

    # --- Create and execute Cloud Run job ---
    job_name = f"{env.get('BASE_JOB_NAME', 'tensorstore-dvid-export')}-profiler"

    env_vars = {
        "PROFILER_SOURCE": args.source,
        "PROFILER_OUTPUT": args.output,
        "PROFILER_MANIFEST_URI": manifest_uri,
        "PROFILER_WORKERS": str(args.workers),
    }
    env_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="profiler-env-")
    for k, v in env_vars.items():
        env_file.write(f"{k}: '{v}'\n")
    env_file.close()

    cmd = [
        "gcloud", "run", "jobs", "create", job_name,
        f"--image={image}",
        f"--region={region}",
        f"--project={project}",
        f"--tasks={num_tasks}",
        f"--parallelism={num_tasks}",
        "--max-retries=1",
        "--task-timeout=3600s",
        f"--memory={args.memory}",
        f"--cpu={cpu}",
        f"--env-vars-file={env_file.name}",
        "--execution-environment=gen2",
        "--command=python",
        "--args=scripts/profile_shards.py",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 and "already exists" in result.stderr:
            cmd[3] = "update"
            result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Failed to create/update job: {result.stderr}")
            sys.exit(1)

        exec_cmd = [
            "gcloud", "run", "jobs", "execute", job_name,
            f"--region={region}",
            f"--project={project}",
            f"--tasks={num_tasks}",
            "--wait" if args.wait else "--async",
        ]
        result = subprocess.run(exec_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Failed to execute job: {result.stderr}")
            sys.exit(1)

        print(f"\nLaunched {job_name} with {num_tasks} tasks "
              f"({len(need_profiling)} shards)")

        if args.wait:
            # Aggregate s1 predictions after profiler completes
            ng_spec = args.ng_spec or env.get("NG_SPEC_PATH", "")
            if ng_spec:
                from scripts.aggregate_predicted_labels import aggregate_labels
                spec_path = Path(ng_spec)
                if not spec_path.is_absolute():
                    spec_path = Path(__file__).resolve().parent.parent / spec_path
                print("\nAggregating s1 label predictions...")
                try:
                    aggregate_labels(args.output, target_scale=1,
                                     ng_spec_path=str(spec_path))
                except Exception as e:
                    print(f"  Warning: aggregation failed: {e}")
            else:
                print("\nSkipping s1 aggregation (no --ng-spec or NG_SPEC_PATH)")
        else:
            print(f"Monitor: gcloud run jobs executions list --job={job_name} "
                  f"--region={region} --project={project}")
    finally:
        os.unlink(env_file.name)


if __name__ == "__main__":
    main()
