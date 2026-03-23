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
    prefix = f"{base_name}-"
    for name in result.stdout.strip().splitlines():
        name = name.strip()
        if not name or not name.startswith(prefix):
            continue
        # Derive label from everything after the base name prefix.
        # Handles tier jobs ({base_name}-tier-8gi),
        # retry jobs ({base_name}-retry-tier-16gi),
        # and legacy per-scale jobs ({base_name}-s0).
        label = name[len(prefix):]
        jobs.append((label, name))

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


def _query_log_events(job_name: str, project: str, region: str,
                      event_name: str, execution: str = "",
                      limit: int = 50000) -> list:
    """Query structured log events by event name for a job."""
    parts = [
        'resource.type="cloud_run_job"',
        f'resource.labels.job_name="{job_name}"',
        f'textPayload=~"{event_name}"',
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

    events = []
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
            events.append(payload)
        except json.JSONDecodeError:
            pass

    return events


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
    job_failed = {}     # job_name -> failed task count

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
        job_failed[job_name] = failed

    if len(jobs) > 1:
        print("-" * 78)
        summary = f"{total_succeeded}"
        if total_failed:
            summary += f"+{total_failed}err"
        if total_running:
            summary += f"+{total_running}run"
        summary += f"/{total_tasks}"
        print(f"{'Total':<16s} {'':12s} {summary:>14s}")

    # Query Cloud Logging for progress and memory stats per tier
    print()
    for label, job_name in jobs:
        execution = job_executions.get(job_name, "")
        if not execution:
            continue

        # Query in-flight progress and completed shard events
        progress = _query_log_events(
            job_name, project, region, "Shard progress", execution)
        completed = _query_log_events(
            job_name, project, region, "Shard complete", execution)

        # Deduplicate progress: keep latest per (scale, shard)
        latest = {}
        for p in progress:
            key = (p.get("scale"), p.get("shard"))
            latest[key] = p  # entries are newest-first from gcloud

        # Completed shards
        completed_keys = set()
        for c in completed:
            completed_keys.add((c.get("scale"), c.get("shard")))

        in_flight = {k: v for k, v in latest.items()
                     if k not in completed_keys}

        total_chunks_inflight = sum(
            p.get("chunks_written", 0) for p in in_flight.values())
        total_chunks_total = sum(
            p.get("total", 0) for p in in_flight.values())

        print(f"{label}:")
        failed_count = job_failed.get(job_name, 0)
        if failed_count:
            print(f"  Failed tasks: {failed_count}")
        completed_note = "+" if len(completed) >= 50000 else ""
        print(f"  Completed shards: {len(completed_keys)}{completed_note}")

        # Memory stats for completed shards
        stats = summarize_memory(completed)
        if stats:
            print(f"  Memory: avg {stats['peak_memory_avg']:.1f}G, "
                  f"max {stats['peak_memory_max']:.1f}G, "
                  f"rewrites: {stats['rewrites']}")
            print(f"  Timing: avg {stats['avg_elapsed_s']:.0f}s, "
                  f"max {stats['max_elapsed_s']:.0f}s per shard")

        # In-flight shards sorted by % complete
        if in_flight:
            mem_vals = [p.get("memory_gib", 0) for p in in_flight.values() if p.get("memory_gib", 0) > 0]
            mem_summary = ""
            if mem_vals:
                mem_summary = (f", memory: avg {sum(mem_vals)/len(mem_vals):.1f}G"
                               f", max {max(mem_vals):.1f}G")
            print(f"  In-flight shards: {len(in_flight)}, "
                  f"chunks: {total_chunks_inflight:,}/{total_chunks_total:,}"
                  f"{mem_summary}")

            def _pct(item):
                p = item[1]
                t = p.get("total", 0)
                return p.get("chunks_written", 0) / t if t else 0

            by_pct = sorted(in_flight.items(), key=_pct, reverse=True)

            def _print_shard(item):
                (scale, shard), p = item
                written = p.get("chunks_written", 0)
                total_c = p.get("total", 0)
                mem = p.get("memory_gib", 0)
                elapsed = p.get("elapsed_s", 0)
                pct = 100 * written / total_c if total_c else 0
                print(f"    s{scale}/{shard}: "
                      f"{written:,}/{total_c:,} ({pct:.0f}%), "
                      f"{mem:.1f}G, {elapsed:.0f}s")

            top5 = by_pct[:5]
            bottom5 = by_pct[-5:]
            middle = len(by_pct) - 10

            for item in top5:
                _print_shard(item)
            if middle > 0:
                print(f"    ... {middle} more ...")
                for item in bottom5:
                    _print_shard(item)
            elif len(by_pct) > 5:
                # 6-10 shards: just show the rest that weren't in top5
                for item in by_pct[5:]:
                    _print_shard(item)

        if not completed and not in_flight:
            print("  (no log data yet)")

        print()

    print("Check chunk-level errors: pixi run export-errors")


if __name__ == "__main__":
    main()
