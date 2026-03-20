#!/usr/bin/env python3
"""
Execute the Cloud Run job to generate neuroglancer precomputed scales.

Reads job name, project, and region from .env and runs the job.

Usage:
    pixi run generate-scale
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE


def main():
    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)

    job_name = env.get("JOB_NAME", "")
    project = env.get("PROJECT_ID", "")
    region = env.get("REGION", "us-central1")

    if not job_name or not project or project == "your-gcp-project":
        print("Error: JOB_NAME and PROJECT_ID must be configured.")
        print("Run 'pixi run deploy' first, or edit .env manually.")
        sys.exit(1)

    print(f"Executing Cloud Run job: {job_name}")
    print(f"  Project: {project}")
    print(f"  Region:  {region}")
    print(f"  Scales:  {env.get('SCALES', '?')}")
    if env.get("DOWNRES_SCALES"):
        print(f"  Downres: {env['DOWNRES_SCALES']}")
    print()

    result = subprocess.run([
        "gcloud", "run", "jobs", "execute", job_name,
        f"--region={region}",
        f"--project={project}",
    ])

    if result.returncode != 0:
        print(f"\nFailed (exit code {result.returncode})")
        sys.exit(1)

    print(f"\nJob submitted. Monitor with:")
    print(f"  gcloud run jobs describe {job_name} --region={region} --project={project}")
    print(f"  gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name={job_name}\" --project={project} --limit=100")


if __name__ == "__main__":
    main()
