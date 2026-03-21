#!/usr/bin/env python3
"""
Show status of all per-scale Cloud Run job executions.

Displays elapsed time, task completion counts, and overall progress
for each scale's latest execution.

Usage:
    pixi run export-status
    pixi run export-status --scale 0
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE


def get_execution_info(job_name: str, project: str, region: str) -> dict:
    """Get the latest execution details for a Cloud Run job."""
    # Find latest execution
    cmd = [
        "gcloud", "run", "jobs", "executions", "list",
        f"--job={job_name}",
        f"--region={region}",
        f"--project={project}",
        "--limit=1",
        "--sort-by=~createTime",
        "--format=json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return {}

    try:
        executions = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}

    if not executions:
        return {}

    ex = executions[0]
    metadata = ex.get("metadata", {})
    status = ex.get("status", {})

    name = metadata.get("name", "?")
    if "/" in name:
        name = name.rsplit("/", 1)[-1]

    create_time = metadata.get("creationTimestamp", "")
    completion_time = status.get("completionTime", "")
    start_time = status.get("startTime", create_time)

    # Parse conditions for overall status
    conditions = status.get("conditions", [])
    overall_status = "Unknown"
    for cond in conditions:
        if cond.get("type") == "Completed":
            if cond.get("status") == "True":
                overall_status = "Completed"
            elif cond.get("reason") == "ContainerFailed":
                overall_status = "Failed"
            else:
                overall_status = cond.get("reason", "Unknown")
            break
    else:
        # No Completed condition yet
        running = status.get("runningCount", 0)
        succeeded = status.get("succeededCount", 0)
        if running > 0:
            overall_status = "Running"
        elif succeeded > 0:
            overall_status = "Completing"

    succeeded = status.get("succeededCount", 0)
    failed = status.get("failedCount", 0)
    running = status.get("runningCount", 0)
    task_count = ex.get("spec", {}).get("taskCount", 0)

    # Calculate elapsed time
    elapsed_str = ""
    if start_time:
        from datetime import datetime, timezone
        try:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            if completion_time:
                end_dt = datetime.fromisoformat(completion_time.replace("Z", "+00:00"))
            else:
                end_dt = datetime.now(timezone.utc)
            elapsed = end_dt - start_dt
            total_secs = int(elapsed.total_seconds())
            hours, remainder = divmod(total_secs, 3600)
            minutes, secs = divmod(remainder, 60)
            if hours > 0:
                elapsed_str = f"{hours}h{minutes:02d}m{secs:02d}s"
            elif minutes > 0:
                elapsed_str = f"{minutes}m{secs:02d}s"
            else:
                elapsed_str = f"{secs}s"
        except (ValueError, TypeError):
            pass

    return {
        "execution": name,
        "status": overall_status,
        "tasks": task_count,
        "succeeded": succeeded,
        "failed": failed,
        "running": running,
        "elapsed": elapsed_str,
        "start_time": start_time,
        "completion_time": completion_time,
    }


def job_exists(job_name: str, project: str, region: str) -> bool:
    """Check if a Cloud Run job exists."""
    cmd = [
        "gcloud", "run", "jobs", "describe", job_name,
        f"--region={region}",
        f"--project={project}",
        "--format=value(name)",
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Show status of per-scale Cloud Run job executions.",
    )
    parser.add_argument(
        "--scale", type=int,
        help="Show only this scale (default: all scales)",
    )
    args = parser.parse_args()

    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)
    base_name = env.get("BASE_JOB_NAME", env.get("JOB_NAME", ""))
    project = env.get("PROJECT_ID", "")
    region = env.get("REGION", "us-central1")

    if not base_name or not project:
        print("Error: BASE_JOB_NAME and PROJECT_ID must be configured in .env")
        sys.exit(1)

    # Find scale jobs
    if args.scale is not None:
        job_names = [(args.scale, f"{base_name}-s{args.scale}")]
    else:
        job_names = []
        for s in range(20):
            jn = f"{base_name}-s{s}"
            if job_exists(jn, project, region):
                job_names.append((s, jn))
        if not job_names:
            print(f"No jobs found matching {base_name}-s*")
            sys.exit(1)

    print(f"{'Scale':<7s} {'Status':<12s} {'Tasks':>14s} {'Elapsed':>10s}  Execution")
    print("-" * 75)

    total_succeeded = 0
    total_failed = 0
    total_running = 0
    total_tasks = 0

    for scale, jn in job_names:
        info = get_execution_info(jn, project, region)
        if not info:
            print(f"s{scale:<5d} {'No execution':<12s}")
            continue

        succeeded = info["succeeded"]
        failed = info["failed"]
        running = info["running"]
        tasks = info["tasks"]
        status = info["status"]
        elapsed = info["elapsed"]
        execution = info["execution"]

        total_succeeded += succeeded
        total_failed += failed
        total_running += running
        total_tasks += tasks

        task_str = f"{succeeded}"
        if failed:
            task_str += f"+{failed}err"
        if running:
            task_str += f"+{running}run"
        task_str += f"/{tasks}"

        print(f"s{scale:<5d} {status:<12s} {task_str:>14s} {elapsed:>10s}  {execution}")

    if len(job_names) > 1:
        print("-" * 75)
        summary = f"{total_succeeded}"
        if total_failed:
            summary += f"+{total_failed}err"
        if total_running:
            summary += f"+{total_running}run"
        summary += f"/{total_tasks}"
        print(f"{'Total':<7s} {'':12s} {summary:>14s}")


if __name__ == "__main__":
    main()
