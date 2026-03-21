#!/usr/bin/env python3
"""
Execute the Cloud Run job to generate neuroglancer precomputed scales.

Reads job name, project, and region from .env.  Command-line options
override the scales and label type for this execution without changing
the .env file.

Usage:
    pixi run generate-scale
    pixi run generate-scale -- --scales 0 --tasks 400
    pixi run generate-scale -- --scales 2,3 --memory 8Gi --cpu 2
    pixi run generate-scale -- --label-type supervoxels
    pixi run generate-scale -- --downres 10
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE


def main():
    parser = argparse.ArgumentParser(
        description="Execute Cloud Run job to generate neuroglancer precomputed scales.",
    )
    parser.add_argument(
        "--scales",
        help="Comma-separated scales to process from DVID shards (overrides .env SCALES)",
    )
    parser.add_argument(
        "--downres",
        help="Comma-separated scales to generate by downsampling previous scale (overrides .env DOWNRES_SCALES)",
    )
    parser.add_argument(
        "--label-type",
        choices=["labels", "supervoxels"],
        help='Label type: "labels" for agglomerated (default), "supervoxels" for raw IDs',
    )
    parser.add_argument(
        "--memory",
        help="Memory per worker (e.g., 4Gi) — overrides .env MEMORY for this execution",
    )
    parser.add_argument(
        "--tasks", type=int,
        help="Number of parallel worker tasks (overrides .env TASKS)",
    )
    parser.add_argument(
        "--cpu", type=int,
        help="CPUs per worker (overrides .env CPU)",
    )
    parser.add_argument(
        "--wait", action="store_true",
        help="Block until the job completes (default: return immediately)",
    )
    args = parser.parse_args()

    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)

    job_name = env.get("JOB_NAME", "")
    project = env.get("PROJECT_ID", "")
    region = env.get("REGION", "us-central1")

    if not job_name or not project or project == "your-gcp-project":
        print("Error: JOB_NAME and PROJECT_ID must be configured.")
        print("Run 'pixi run deploy' first, or edit .env manually.")
        sys.exit(1)

    # Build env var overrides from CLI args
    overrides = {}
    scales = args.scales or env.get("SCALES", "0")
    overrides["SCALES"] = scales

    downres = args.downres or env.get("DOWNRES_SCALES", "")
    if downres:
        overrides["DOWNRES_SCALES"] = downres

    label_type = args.label_type or env.get("LABEL_TYPE", "labels")
    overrides["LABEL_TYPE"] = label_type

    print(f"Executing Cloud Run job: {job_name}")
    print(f"  Project:    {project}")
    print(f"  Region:     {region}")
    print(f"  Scales:     {scales}")
    if downres:
        print(f"  Downres:    {downres}")
    print(f"  Label type: {label_type}")
    tasks = args.tasks or int(env.get("TASKS", "200"))
    cpu = args.cpu or int(env.get("CPU", "2"))
    memory = args.memory or env.get("MEMORY", "4Gi")

    print(f"  Tasks:      {tasks}" + (" (override)" if args.tasks else ""))
    print(f"  CPU:        {cpu}" + (" (override)" if args.cpu else ""))
    print(f"  Memory:     {memory}" + (" (override)" if args.memory else ""))
    print()

    # CPU and memory are job-level settings — update the job definition first
    # if either was overridden.
    if args.cpu or args.memory:
        update_cmd = [
            "gcloud", "run", "jobs", "update", job_name,
            f"--region={region}",
            f"--project={project}",
            f"--cpu={cpu}",
            f"--memory={memory}",
        ]
        print("Updating job resource limits...")
        result = subprocess.run(update_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to update job: {result.stderr}")
            sys.exit(1)

    cmd = [
        "gcloud", "run", "jobs", "execute", job_name,
        f"--region={region}",
        f"--project={project}",
        f"--tasks={tasks}",
    ]

    # Pass env var overrides.  Use ^;^ as delimiter since values contain commas.
    override_str = ";".join(f"{k}={v}" for k, v in overrides.items())
    cmd.append(f"--update-env-vars=^;^{override_str}")

    if args.wait:
        cmd.append("--wait")
    else:
        cmd.append("--async")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nFailed (exit code {result.returncode})")
        sys.exit(1)

    print("\nCheck for errors (works during or after execution):")
    print("  pixi run export-errors")
    print("  pixi run export-errors -- --details")
    print()
    print("Monitor job status:")
    print(f"  gcloud run jobs executions list --job={job_name} --region={region} --project={project}")


if __name__ == "__main__":
    main()
