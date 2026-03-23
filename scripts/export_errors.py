#!/usr/bin/env python3
"""
Query Cloud Run job logs for export errors and produce a summary.

With per-scale jobs ({BASE_JOB_NAME}-s0, -s1, ...), this script can query
a single scale or aggregate across all scales.

Usage:
    pixi run export-errors                     # all scales, latest execution each
    pixi run export-errors --scale 0        # just s0
    pixi run export-errors --details        # full error details
    pixi run export-errors --all            # all executions, not just latest
"""

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE


def query_logs(job_name: str, project: str, region: str,
               event_filter: str, limit: int,
               execution: str = "") -> list:
    """Run a gcloud logging read query and return parsed entries."""
    parts = [
        'resource.type="cloud_run_job"',
        f'resource.labels.job_name="{job_name}"',
        f'textPayload=~"{event_filter}"',
    ]
    if execution:
        parts.append(f'labels."run.googleapis.com/execution_name"="{execution}"')

    filter_str = " AND ".join(parts)
    cmd = [
        "gcloud", "logging", "read", filter_str,
        f"--project={project}",
        f"--limit={limit}",
        "--format=json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error querying logs for {job_name}: {result.stderr}", file=sys.stderr)
        return []
    try:
        return json.loads(result.stdout) if result.stdout.strip() else []
    except json.JSONDecodeError:
        print("Failed to parse log output", file=sys.stderr)
        return []


def parse_structured_payload(entry: dict) -> dict:
    """Extract structured fields from a textPayload JSON log line."""
    text = entry.get("textPayload", "")
    idx = text.find("{")
    if idx == -1:
        return {"raw": text}
    try:
        return json.loads(text[idx:])
    except json.JSONDecodeError:
        return {"raw": text}


def classify_error(error_str: str) -> str:
    """Classify an error string into a short category."""
    if "zstd" in error_str.lower():
        return "zstd_decompression"
    if "CURL error" in error_str:
        return "curl_gcs"
    if "not found" in error_str.lower():
        return "file_not_found"
    if "timeout" in error_str.lower():
        return "timeout"
    return "other"


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
    if result.returncode != 0:
        return ""
    name = result.stdout.strip()
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return name


def job_exists(job_name: str, project: str, region: str) -> bool:
    """Check if a Cloud Run job exists."""
    cmd = [
        "gcloud", "run", "jobs", "describe", job_name,
        f"--region={region}",
        f"--project={project}",
        "--format=value(name)",
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def query_scale_logs(job_name: str, project: str, region: str,
                     limit: int, use_all: bool, execution: str = ""):
    """Query all log types for a single scale job. Returns (chunks, shards, successes)."""
    # Resolve execution for this specific job
    exec_filter = execution
    if not exec_filter and not use_all:
        exec_filter = get_latest_execution(job_name, project, region)

    chunk_entries = query_logs(job_name, project, region,
                               "Chunk failed", limit, exec_filter)
    shard_entries = query_logs(job_name, project, region,
                               "Failed to process shard", limit, exec_filter)
    success_entries = query_logs(job_name, project, region,
                                 "Shard complete", limit, exec_filter)
    return chunk_entries, shard_entries, success_entries, exec_filter


def main():
    parser = argparse.ArgumentParser(
        description="Query Cloud Run job logs for export errors.",
    )
    parser.add_argument(
        "--scale", type=int,
        help="Query only this scale (default: all scales)",
    )
    parser.add_argument(
        "--limit", type=int, default=50000,
        help="Max log entries to fetch per query (default: 50000)",
    )
    parser.add_argument(
        "--details", action="store_true",
        help="Print each error with full details",
    )
    parser.add_argument(
        "--execution",
        help="Filter to a specific execution name",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Show errors from ALL executions (default: most recent only)",
    )
    args = parser.parse_args()

    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)
    base_name = env.get("BASE_JOB_NAME", env.get("JOB_NAME", ""))
    project = env.get("PROJECT_ID", "")
    region = env.get("REGION", "us-central1")

    if not base_name or not project:
        print("Error: BASE_JOB_NAME and PROJECT_ID must be configured in .env")
        sys.exit(1)

    # Determine which jobs to query
    if args.scale is not None:
        job_names = [f"{base_name}-s{args.scale}"]
    else:
        # Find all existing jobs: tier-based and per-scale
        job_names = []
        cmd = [
            "gcloud", "run", "jobs", "list",
            f"--region={region}",
            f"--project={project}",
            f"--filter=metadata.name~^{base_name}",
            "--format=value(metadata.name)",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            for name in result.stdout.strip().splitlines():
                name = name.strip()
                if name and name != base_name:
                    job_names.append(name)
        if not job_names:
            if job_exists(base_name, project, region):
                job_names = [base_name]
            else:
                print(f"No jobs found matching {base_name}-*")
                sys.exit(1)
        job_names.sort()

    print(f"Querying errors for {len(job_names)} job(s)")
    print(f"  Project: {project}")
    print(f"  Region:  {region}")
    print(f"  Jobs:    {', '.join(job_names)}")
    print()

    # Aggregate results across all scale jobs
    all_chunk_entries = []
    all_shard_entries = []
    all_success_entries = []
    executions = {}

    for jn in job_names:
        chunks, shards, successes, exec_name = query_scale_logs(
            jn, project, region, args.limit, args.all, args.execution
        )
        all_chunk_entries.extend(chunks)
        all_shard_entries.extend(shards)
        all_success_entries.extend(successes)
        if exec_name:
            executions[jn] = exec_name

    if executions:
        print("Executions:")
        for jn, ex in executions.items():
            print(f"  {jn}: {ex}")
        print()

    # Parse chunk errors
    error_types = Counter()
    errors_by_shard = defaultdict(list)
    errors_by_scale = Counter()
    all_errors = []

    for entry in all_chunk_entries:
        payload = parse_structured_payload(entry)
        error_str = payload.get("error", payload.get("raw", "unknown"))
        shard = payload.get("shard", "unknown")
        scale = payload.get("scale", "?")
        cx = payload.get("chunk_x", "?")
        cy = payload.get("chunk_y", "?")
        cz = payload.get("chunk_z", "?")
        task_idx = entry.get("labels", {}).get(
            "run.googleapis.com/task_index", "?"
        )

        category = classify_error(error_str)
        error_types[category] += 1
        errors_by_shard[shard].append((cx, cy, cz, category))
        errors_by_scale[scale] += 1
        all_errors.append({
            "scale": scale, "shard": shard,
            "chunk": (cx, cy, cz), "task": task_idx,
            "category": category, "error": error_str,
        })

    # Parse shard-level failures
    shard_load_failures = []
    for entry in all_shard_entries:
        payload = parse_structured_payload(entry)
        if "chunk_x" not in payload:
            shard_load_failures.append({
                "shard": payload.get("shard", "unknown"),
                "scale": payload.get("scale", "?"),
                "error": payload.get("error", "unknown"),
            })

    # Parse successes
    success_chunks = 0
    success_shards = 0
    failed_chunks_in_success = 0
    for entry in all_success_entries:
        payload = parse_structured_payload(entry)
        success_shards += 1
        success_chunks += payload.get("chunks_written", 0)
        failed_chunks_in_success += payload.get("chunks_failed", 0)

    # Print summary
    print("=" * 60)
    print("EXPORT ERROR SUMMARY")
    print("=" * 60)
    print()

    truncated = len(all_success_entries) >= args.limit
    trunc_note = "+" if truncated else ""
    print(f"Shards completed:       {success_shards}{trunc_note}")
    print(f"  Chunks written:       {success_chunks:,}{trunc_note}")
    print(f"  Chunks failed:        {failed_chunks_in_success}")
    if truncated:
        print(f"  (counts truncated at --limit={args.limit}; "
              f"use --limit=N to increase)")
    print(f"Shard load failures:    {len(shard_load_failures)}")
    print(f"Chunk errors (from logs): {len(all_errors)}")
    print()

    if error_types:
        print("Errors by type:")
        for category, count in error_types.most_common():
            print(f"  {category:25s} {count:>6d}")
        print()

    if errors_by_scale:
        print("Errors by scale:")
        for scale, count in sorted(errors_by_scale.items()):
            print(f"  s{scale}:  {count}")
        print()

    if errors_by_shard:
        print(f"Shards with chunk errors: {len(errors_by_shard)}")
        worst = sorted(errors_by_shard.items(), key=lambda x: -len(x[1]))[:10]
        for shard, errs in worst:
            print(f"  {shard:30s} {len(errs):>4d} failed chunks")
        if len(errors_by_shard) > 10:
            print(f"  ... and {len(errors_by_shard) - 10} more shards")
        print()

    if shard_load_failures:
        print(f"Shard load failures ({len(shard_load_failures)}):")
        for f in shard_load_failures[:20]:
            print(f"  s{f['scale']} {f['shard']}: {f['error'][:100]}")
        print()

    if args.details and all_errors:
        print("=" * 60)
        print("DETAILED CHUNK ERRORS")
        print("=" * 60)
        for err in all_errors:
            cx, cy, cz = err["chunk"]
            print(f"  s{err['scale']} {err['shard']} "
                  f"chunk=({cx},{cy},{cz}) "
                  f"task={err['task']} "
                  f"[{err['category']}]")
            print(f"    {err['error'][:200]}")
        print()

    if not all_errors and not shard_load_failures:
        print("No errors found!")


if __name__ == "__main__":
    main()
