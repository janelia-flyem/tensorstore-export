#!/usr/bin/env python3
"""
Execute Cloud Run jobs to generate neuroglancer precomputed scales.

Each scale has its own Cloud Run job ({BASE_JOB_NAME}-s{N}), created by
``pixi run deploy``.  This script executes one or more scale jobs,
optionally overriding resource settings per invocation.

Usage:
    pixi run generate-scale
    pixi run generate-scale --scales 0 --tasks 800
    pixi run generate-scale --scales 0,1,2 --tasks 200
    pixi run generate-scale --scales 3 --tasks 50 --memory 16Gi
    pixi run generate-scale --label-type supervoxels
    pixi run generate-scale --downres 10
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE


def main():
    parser = argparse.ArgumentParser(
        description="Execute Cloud Run jobs to generate neuroglancer precomputed scales.",
    )
    parser.add_argument(
        "--scales",
        help="Comma-separated scales to process (default: from .env SCALES)",
    )
    parser.add_argument(
        "--downres",
        help="Comma-separated scales to generate by downsampling previous scale",
    )
    parser.add_argument(
        "--label-type",
        choices=["labels", "supervoxels"],
        help='Label type: "labels" for agglomerated (default), "supervoxels" for raw IDs',
    )
    parser.add_argument(
        "--tasks", type=int,
        help="Number of parallel worker tasks per scale job",
    )
    parser.add_argument(
        "--cpu", type=int,
        help="CPUs per worker",
    )
    parser.add_argument(
        "--memory",
        help="Memory per worker (e.g., 4Gi, 16Gi)",
    )
    parser.add_argument(
        "--wait", action="store_true",
        help="Block until all jobs complete (default: return immediately)",
    )
    args = parser.parse_args()

    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)

    base_name = env.get("BASE_JOB_NAME", env.get("JOB_NAME", ""))
    project = env.get("PROJECT_ID", "")
    region = env.get("REGION", "us-central1")

    if not base_name or not project or project == "your-gcp-project":
        print("Error: BASE_JOB_NAME and PROJECT_ID must be configured.")
        print("Run 'pixi run deploy' first, or edit .env manually.")
        sys.exit(1)

    scales_str = args.scales or env.get("SCALES", "0")
    scales = [int(s.strip()) for s in scales_str.split(",")]

    tasks = args.tasks or int(env.get("TASKS", "200"))
    cpu = args.cpu or int(env.get("CPU", "2"))
    memory = args.memory or env.get("MEMORY", "4Gi")
    label_type = args.label_type or env.get("LABEL_TYPE", "labels")

    downres = args.downres or env.get("DOWNRES_SCALES", "")

    print(f"Executing {len(scales)} scale job(s): {base_name}")
    print(f"  Project:    {project}")
    print(f"  Region:     {region}")
    print(f"  Scales:     {','.join(str(s) for s in scales)}")
    if downres:
        print(f"  Downres:    {downres}")
    print(f"  Label type: {label_type}")
    print(f"  Tasks:      {tasks}")
    print(f"  CPU:        {cpu}")
    print(f"  Memory:     {memory}")
    print()

    # Update job definition: always sync parallelism to match tasks,
    # and update CPU/memory if overridden.
    for scale in scales:
        job_name = f"{base_name}-s{scale}"
        update_cmd = [
            "gcloud", "run", "jobs", "update", job_name,
            f"--region={region}",
            f"--project={project}",
            f"--parallelism={tasks}",
            f"--cpu={cpu}",
            f"--memory={memory}",
        ]
        result = subprocess.run(update_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: failed to update {job_name}: {result.stderr.strip()}")

    # Execute each scale job
    for scale in scales:
        job_name = f"{base_name}-s{scale}"

        # Build env var overrides
        overrides = {"LABEL_TYPE": label_type}
        if downres:
            overrides["DOWNRES_SCALES"] = downres

        cmd = [
            "gcloud", "run", "jobs", "execute", job_name,
            f"--region={region}",
            f"--project={project}",
            f"--tasks={tasks}",
        ]

        if overrides:
            override_str = ";".join(f"{k}={v}" for k, v in overrides.items())
            cmd.append(f"--update-env-vars=^;^{override_str}")

        if args.wait:
            cmd.append("--wait")
        else:
            cmd.append("--async")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  {job_name}: FAILED")
            print(f"    {result.stderr.strip()}")
        else:
            print(f"  {job_name}: executing ({tasks} tasks)")

    print("\nCheck for errors (works during or after execution):")
    print("  pixi run export-errors")
    if len(scales) == 1:
        print(f"  pixi run export-errors -- --scale {scales[0]}")
    print("  pixi run export-errors -- --details")
    print()
    print("Monitor job status:")
    for scale in scales:
        print(f"  gcloud run jobs executions list --job={base_name}-s{scale} --region={region} --project={project}")


if __name__ == "__main__":
    main()
