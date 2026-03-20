#!/usr/bin/env python3
"""
Query Cloud Run job logs for export errors and produce a summary.

Searches for "Chunk failed" and "Failed to process shard" events,
aggregates by error type, scale, and shard, and optionally prints
full details.

Usage:
    pixi run export-errors
    pixi run export-errors -- --details
    pixi run export-errors -- --limit 500
    pixi run export-errors -- --execution tensorstore-dvid-export-test1-bqblh
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
        print(f"Error querying logs: {result.stderr}", file=sys.stderr)
        return []
    try:
        return json.loads(result.stdout) if result.stdout.strip() else []
    except json.JSONDecodeError:
        print("Failed to parse log output", file=sys.stderr)
        return []


def parse_structured_payload(entry: dict) -> dict:
    """Extract structured fields from a textPayload JSON log line."""
    text = entry.get("textPayload", "")
    # structlog lines look like: "ERROR:src.worker:{...json...}"
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
    # Output is the full resource name; extract just the execution name
    name = result.stdout.strip()
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Query Cloud Run job logs for export errors.",
    )
    parser.add_argument(
        "--limit", type=int, default=1000,
        help="Max log entries to fetch per query (default: 1000)",
    )
    parser.add_argument(
        "--details", action="store_true",
        help="Print each error with full details",
    )
    parser.add_argument(
        "--execution",
        help="Filter to a specific execution name (e.g., tensorstore-dvid-export-test1-bqblh)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Show errors from ALL executions (default: most recent only)",
    )
    args = parser.parse_args()

    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)
    job_name = env.get("JOB_NAME", "")
    project = env.get("PROJECT_ID", "")
    region = env.get("REGION", "us-central1")

    if not job_name or not project:
        print("Error: JOB_NAME and PROJECT_ID must be configured in .env")
        sys.exit(1)

    # Resolve execution: explicit > latest > all
    execution = args.execution
    if not execution and not args.all:
        execution = get_latest_execution(job_name, project, region)
        if not execution:
            print("Warning: could not determine latest execution, showing all")

    print(f"Querying errors for job: {job_name}")
    print(f"  Project:   {project}")
    print(f"  Region:    {region}")
    if execution:
        print(f"  Execution: {execution}")
    else:
        print("  Execution: (all)")
    print()

    # Query chunk-level failures
    print("Fetching chunk errors...")
    chunk_entries = query_logs(
        job_name, project, region,
        "Chunk failed", args.limit, execution,
    )

    # Query shard-level failures
    print("Fetching shard errors...")
    shard_entries = query_logs(
        job_name, project, region,
        "Failed to process shard", args.limit, execution,
    )

    # Query successes for context
    print("Fetching shard successes...")
    success_entries = query_logs(
        job_name, project, region,
        "Shard complete", args.limit, execution,
    )
    print()

    # Parse chunk errors
    error_types = Counter()
    errors_by_shard = defaultdict(list)
    errors_by_scale = Counter()
    all_errors = []

    for entry in chunk_entries:
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

    # Parse shard-level failures (entire shard couldn't load)
    shard_load_failures = []
    for entry in shard_entries:
        payload = parse_structured_payload(entry)
        # Only count entries that aren't also chunk-level (the old format)
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
    for entry in success_entries:
        payload = parse_structured_payload(entry)
        success_shards += 1
        success_chunks += payload.get("chunks_written", 0)
        failed_chunks_in_success += payload.get("chunks_failed", 0)

    # Print summary
    print("=" * 60)
    print("EXPORT ERROR SUMMARY")
    print("=" * 60)
    print()

    print(f"Shards completed:       {success_shards}")
    print(f"  Chunks written:       {success_chunks}")
    print(f"  Chunks failed:        {failed_chunks_in_success}")
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
        # Show top 10 worst shards
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
