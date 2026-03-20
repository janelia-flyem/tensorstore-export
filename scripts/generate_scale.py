#!/usr/bin/env python3
"""
Execute the Cloud Run job to generate neuroglancer precomputed scales.

Reads job name, project, and region from .env.  Command-line options
override the scales and label type for this execution without changing
the .env file.

Usage:
    pixi run generate-scale
    pixi run generate-scale -- --scales 0,1
    pixi run generate-scale -- --scales 2,3 --memory 4Gi
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
    if args.memory:
        print(f"  Memory:     {args.memory} (override)")
    print()

    cmd = [
        "gcloud", "run", "jobs", "execute", job_name,
        f"--region={region}",
        f"--project={project}",
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

    print("\nMonitor with:")
    print(f"  gcloud run jobs describe {job_name} --region={region} --project={project}")
    print(f"  gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name={job_name}\" --project={project} --limit=100")


if __name__ == "__main__":
    main()
