#!/usr/bin/env python3
"""
Show status of Cloud Run export job executions.

Discovers both tier-based jobs ({BASE_JOB_NAME}-tier-{N}gi) and legacy
per-scale jobs ({BASE_JOB_NAME}-s{N}).  Optionally queries memory profile
logs for peak memory, batch counts, and throughput stats.

Usage:
    pixi run export-status
    pixi run export-status --memory
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE


def discover_jobs(base_name: str, project: str, region: str) -> list:
    """Find all Cloud Run jobs matching the base name.

    Returns list of (label, job_name) tuples, preferring tier jobs over
    scale jobs.
    """
    # List all jobs in the project/region
    cmd = [
        "gcloud", "run", "jobs", "list",
        f"--region={region}",
        f"--project={project}",
        f"--filter=metadata.name~^{base_name}",
        "--format=value(metadata.name)",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []

    jobs = []
    for name in result.stdout.strip().splitlines():
        name = name.strip()
        if not name:
            continue
        if name.startswith(f"{base_name}-tier-"):
            tier = name.replace(f"{base_name}-tier-", "")
            jobs.append((f"tier-{tier}", name))
        elif name.startswith(f"{base_name}-s"):
            scale = name.replace(f"{base_name}-s", "")
            jobs.append((f"s{scale}", name))

    return sorted(jobs)


def get_execution_info(job_name: str, project: str, region: str) -> dict:
    """Get the latest execution details for a Cloud Run job."""
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

    conditions = status.get("conditions", [])
    running = status.get("runningCount", 0)
    succeeded = status.get("succeededCount", 0)

    overall_status = "Unknown"
    for cond in conditions:
        if cond.get("type") == "Completed":
            if cond.get("status") == "True":
                overall_status = "Completed"
            elif cond.get("reason") == "ContainerFailed":
                overall_status = "Failed"
            break

    if overall_status == "Unknown":
        if running > 0:
            overall_status = "Running"
        elif succeeded > 0:
            overall_status = "Completing"

    succeeded = status.get("succeededCount", 0)
    failed = status.get("failedCount", 0)
    running = status.get("runningCount", 0)
    task_count = ex.get("spec", {}).get("taskCount", 0)

    elapsed_str = ""
    if start_time:
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
    }


def query_memory_profiles(job_name: str, project: str, region: str,
                          execution: str = "", limit: int = 5000) -> list:
    """Query 'Shard memory profile' log events for a job."""
    parts = [
        'resource.type="cloud_run_job"',
        f'resource.labels.job_name="{job_name}"',
        'textPayload=~"Shard memory profile"',
    ]
    if execution:
        parts.append(f'labels."run.googleapis.com/execution_name"="{execution}"')

    cmd = [
        "gcloud", "logging", "read", " AND ".join(parts),
        f"--project={project}",
        f"--limit={limit}",
        "--format=json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return []

    profiles = []
    try:
        entries = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    for entry in entries:
        text = entry.get("textPayload", "")
        idx = text.find("{")
        if idx == -1:
            continue
        try:
            payload = json.loads(text[idx:])
            profiles.append(payload)
        except json.JSONDecodeError:
            pass

    return profiles


def summarize_memory(profiles: list) -> dict:
    """Summarize memory profile events into a stats dict."""
    if not profiles:
        return {}

    peaks = [p.get("peak_memory_gib", 0) for p in profiles]
    limits = [p.get("memory_limit_gib", 0) for p in profiles]
    batches = [p.get("batches", 1) for p in profiles]
    elapsed = [p.get("elapsed_s", 0) for p in profiles]
    multi_batch = sum(1 for b in batches if b > 1)

    return {
        "shards_profiled": len(profiles),
        "peak_memory_avg": sum(peaks) / len(peaks),
        "peak_memory_max": max(peaks),
        "memory_limit": max(limits) if limits else 0,
        "rewrites": multi_batch,
        "avg_elapsed_s": sum(elapsed) / len(elapsed) if elapsed else 0,
        "max_elapsed_s": max(elapsed) if elapsed else 0,
    }


def get_latest_execution(job_name: str, project: str, region: str) -> str:
    """Get the most recent execution name for a Cloud Run job."""
    cmd = [
        "gcloud", "run", "jobs", "executions", "list",
        f"--job={job_name}",
        f"--region={region}",
        f"--project={project}",
        "--limit=1",
        "--sort-by=~createTime",
        "--format=value(name)",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    name = result.stdout.strip()
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Show status of Cloud Run export job executions.",
    )
    parser.add_argument(
        "--memory", action="store_true",
        help="Query logs for memory profile stats (slower — reads Cloud Logging)",
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
    if not jobs:
        print(f"No jobs found matching {base_name}-*")
        sys.exit(1)

    print(f"{'Job':<16s} {'Status':<12s} {'Tasks':>14s} {'Elapsed':>10s}  Execution")
    print("-" * 78)

    total_succeeded = 0
    total_failed = 0
    total_running = 0
    total_tasks = 0

    job_executions = {}  # job_name -> execution_name for memory queries

    for label, job_name in jobs:
        info = get_execution_info(job_name, project, region)
        if not info:
            print(f"{label:<16s} {'No execution':<12s}")
            continue

        succeeded = info["succeeded"]
        failed = info["failed"]
        running = info["running"]
        tasks = info["tasks"]

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

        print(f"{label:<16s} {info['status']:<12s} {task_str:>14s} "
              f"{info['elapsed']:>10s}  {info['execution']}")

        job_executions[job_name] = info["execution"]

    if len(jobs) > 1:
        print("-" * 78)
        summary = f"{total_succeeded}"
        if total_failed:
            summary += f"+{total_failed}err"
        if total_running:
            summary += f"+{total_running}run"
        summary += f"/{total_tasks}"
        print(f"{'Total':<16s} {'':12s} {summary:>14s}")

    # Memory profile stats
    if args.memory:
        print()
        print(f"{'Job':<16s} {'Shards':>7s} {'PeakAvg':>8s} {'PeakMax':>8s} "
              f"{'Limit':>6s} {'Rewrites':>8s} {'AvgTime':>8s} {'MaxTime':>8s}")
        print("-" * 78)

        for label, job_name in jobs:
            execution = job_executions.get(job_name, "")
            if not execution:
                continue

            profiles = query_memory_profiles(job_name, project, region, execution)
            stats = summarize_memory(profiles)
            if not stats:
                print(f"{label:<16s} {'(no data)':>7s}")
                continue

            print(f"{label:<16s} {stats['shards_profiled']:>7d} "
                  f"{stats['peak_memory_avg']:>7.1f}G {stats['peak_memory_max']:>7.1f}G "
                  f"{stats['memory_limit']:>5.0f}G {stats['rewrites']:>8d} "
                  f"{stats['avg_elapsed_s']:>7.0f}s {stats['max_elapsed_s']:>7.0f}s")

        print()
        print("  PeakAvg/PeakMax: average/maximum peak cgroup memory across shards")
        print("  Rewrites: shards that needed >1 transaction commit (memory pressure)")
        print("  AvgTime/MaxTime: wall-clock seconds per shard")

    print()
    print("Check chunk-level errors: pixi run export-errors")
    if not args.memory:
        print("Memory profile stats:    pixi run export-status --memory")


if __name__ == "__main__":
    main()
