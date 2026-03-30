#!/usr/bin/env python3
"""
Export downres memory telemetry from Cloud Logging to CSV.

Reads "Downres shard complete" events from the latest execution of each
downres Cloud Run job and writes a CSV suitable for fitting a better memory
formula.

Usage:
    pixi run analyze-downres-memory
    pixi run analyze-downres-memory -- --output analysis/downres_s1.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE
from scripts.export_status import (
    discover_jobs,
    get_latest_execution,
    _query_log_events,
    _is_downres_job,
    _normalize_downres_payload,
)


def main():
    parser = argparse.ArgumentParser(
        description="Export downres memory events from Cloud Logging to CSV.",
    )
    parser.add_argument(
        "--output", default="analysis/downres_memory.csv",
        help="Output CSV path (default: analysis/downres_memory.csv)",
    )
    parser.add_argument(
        "--limit", type=int, default=50000,
        help="Per-job Cloud Logging query limit (default: 50000)",
    )
    args = parser.parse_args()

    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)
    base_name = env.get("BASE_JOB_NAME", env.get("JOB_NAME", ""))
    project = env.get("PROJECT_ID", "")
    region = env.get("REGION", "us-central1")

    if not base_name or not project:
        print("Error: BASE_JOB_NAME and PROJECT_ID must be configured in .env")
        sys.exit(1)

    jobs = discover_jobs(base_name, project, region)
    downres_jobs = [(label, job_name) for label, job_name in jobs
                    if _is_downres_job(label)]
    if not downres_jobs:
        print(f"No downres jobs found matching {base_name}-*")
        sys.exit(1)

    rows = []
    for label, job_name in downres_jobs:
        execution = get_latest_execution(job_name, project, region)
        if not execution:
            continue
        events = _query_log_events(
            job_name, project, region,
            "Downres shard complete", execution, limit=args.limit,
        )
        for payload in events:
            payload = _normalize_downres_payload(payload)
            payload["job_label"] = label
            payload["job_name"] = job_name
            payload["execution"] = execution
            rows.append(payload)

    if not rows:
        print("No downres completion events found.")
        sys.exit(1)

    rows.sort(key=lambda r: (r.get("scale", 0), r.get("shard", 0)))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "job_label",
        "job_name",
        "execution",
        "scale",
        "shard",
        "num_chunks",
        "estimate_model",
        "estimated_total_unique_labels",
        "estimated_memory_gib",
        "estimated_subtotal_gib",
        "estimated_output_gib",
        "estimated_tmpfs_gib",
        "estimated_raw_batch_gib",
        "estimated_overhead_gib",
        "estimated_commit_spike_gib",
        "tmpfs_mib",
        "peak_memory_gib",
        "prediction_error_gib",
        "prediction_ratio",
        "batches",
        "elapsed_s",
        "upload_s",
        "uploaded_gib",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"Wrote {len(rows)} downres rows to {out_path}")


if __name__ == "__main__":
    main()
