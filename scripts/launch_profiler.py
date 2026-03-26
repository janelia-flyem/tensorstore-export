#!/usr/bin/env python3
"""
Launch shard profiling as a Cloud Run job.

Creates a Cloud Run job that runs profile_shards.py across N tasks,
each processing a round-robin partition of shards with multi-threaded
parallelism.  No egress charges when source/output buckets are in the
same region as Cloud Run.

Usage:
    pixi run launch-profiler --source gs://bucket/exports/seg --output gs://bucket2/exports/seg
    pixi run launch-profiler --source gs://bucket/exports/seg --output gs://bucket2/exports/seg --tasks 100
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE


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
        help="Number of Cloud Run tasks (default: 100)",
    )
    parser.add_argument(
        "--workers", type=int, default=32,
        help="Threads per task (default: 32)",
    )
    parser.add_argument(
        "--memory", default="8Gi",
        help="Memory per task (default: 8Gi)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the gcloud command without running it",
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

    job_name = f"{env.get('BASE_JOB_NAME', 'tensorstore-dvid-export')}-profiler"

    # Write env vars to temp YAML
    env_vars = {
        "PROFILER_SOURCE": args.source,
        "PROFILER_OUTPUT": args.output,
        "PROFILER_SCALES": args.scales,
        "PROFILER_WORKERS": str(args.workers),
    }
    env_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="profiler-env-")
    for k, v in env_vars.items():
        env_file.write(f"{k}: '{v}'\n")
    env_file.close()

    # Override the entrypoint to run profile_shards.py instead of main.py
    cmd = [
        "gcloud", "run", "jobs", "create", job_name,
        f"--image={image}",
        f"--region={region}",
        f"--project={project}",
        f"--tasks={args.tasks}",
        f"--parallelism={args.tasks}",
        "--max-retries=1",
        "--task-timeout=3600s",
        f"--memory={args.memory}",
        "--cpu=2",
        f"--env-vars-file={env_file.name}",
        "--execution-environment=gen2",
        "--command=python",
        "--args=scripts/profile_shards.py",
    ]

    print(f"Job: {job_name}")
    print(f"  Image: {image}")
    print(f"  Region: {region}")
    print(f"  Tasks: {args.tasks}, workers/task: {args.workers}")
    print(f"  Memory: {args.memory}, CPU: 2")
    print(f"  Source: {args.source}")
    print(f"  Output: {args.output}")
    print(f"  Scales: {args.scales}")

    if args.dry_run:
        print(f"\nCommand:\n  {' '.join(cmd)}")
        print("\n(dry run — not launched)")
        os.unlink(env_file.name)
        return

    try:
        # Create (or update if exists)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 and "already exists" in result.stderr:
            cmd[3] = "update"
            result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Failed to create/update job: {result.stderr}")
            sys.exit(1)

        # Execute
        exec_cmd = [
            "gcloud", "run", "jobs", "execute", job_name,
            f"--region={region}",
            f"--project={project}",
            f"--tasks={args.tasks}",
            "--async",
        ]
        result = subprocess.run(exec_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Failed to execute job: {result.stderr}")
            sys.exit(1)

        print(f"\nLaunched {job_name} with {args.tasks} tasks")
        print(f"Monitor: gcloud run jobs executions list --job={job_name} "
              f"--region={region} --project={project}")
    finally:
        os.unlink(env_file.name)


if __name__ == "__main__":
    main()
