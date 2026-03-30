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


def _is_downres_job(job_name: str) -> bool:
    """Check if a job name indicates a downres job."""
    return "downres" in job_name


def _normalize_downres_payload(payload: dict) -> dict:
    """Map downres log fields to match regular export field names."""
    if "shard_number" in payload and "shard" not in payload:
        payload["shard"] = payload["shard_number"]
    if "num_chunks" in payload and "chunks_written" not in payload:
        payload["chunks_written"] = payload["num_chunks"]
    payload.setdefault("chunks_failed", 0)
    return payload


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
    if "memory" in error_str.lower() or "oom" in error_str.lower():
        return "memory_pressure"
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
    """Query all log types for a single scale job.

    Returns (chunks, shards, successes, progress, memory_warnings, exec_filter).
    """
    # Resolve execution for this specific job
    exec_filter = execution
    if not exec_filter and not use_all:
        exec_filter = get_latest_execution(job_name, project, region)

    is_downres = _is_downres_job(job_name)
    if is_downres:
        chunk_entries = []  # downres doesn't emit per-chunk errors
        shard_entries = query_logs(job_name, project, region,
                                   "Failed to downres shard", limit, exec_filter)
        success_entries = query_logs(job_name, project, region,
                                     "Downres shard complete", limit, exec_filter)
        progress_entries = []  # downres doesn't emit progress events
    else:
        chunk_entries = query_logs(job_name, project, region,
                                   "Chunk failed", limit, exec_filter)
        shard_entries = query_logs(job_name, project, region,
                                   "Failed to process shard", limit, exec_filter)
        success_entries = query_logs(job_name, project, region,
                                     "Shard complete", limit, exec_filter)
        progress_entries = query_logs(job_name, project, region,
                                      "Shard progress", limit, exec_filter)
    memory_entries = query_logs(job_name, project, region,
                                "Memory critical\\|Memory pressure", limit, exec_filter)
    return (chunk_entries, shard_entries, success_entries,
            progress_entries, memory_entries, exec_filter)


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
    all_progress_entries = []
    all_memory_entries = []
    executions = {}

    for jn in job_names:
        (chunks, shards, successes,
         progress, memory_warns, exec_name) = query_scale_logs(
            jn, project, region, args.limit, args.all, args.execution
        )
        all_chunk_entries.extend(chunks)
        all_shard_entries.extend(shards)
        all_success_entries.extend(successes)
        all_progress_entries.extend(progress)
        all_memory_entries.extend(memory_warns)
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
        payload = _normalize_downres_payload(parse_structured_payload(entry))
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
        payload = _normalize_downres_payload(parse_structured_payload(entry))
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

    # Detect incomplete shards: had "Shard progress" but no "Shard complete"
    # or "Failed to process shard".  These are likely OOM-killed tasks.
    completed_shards = set()
    for entry in all_success_entries:
        payload = _normalize_downres_payload(parse_structured_payload(entry))
        completed_shards.add((payload.get("scale"), payload.get("shard")))
    for entry in all_shard_entries:
        payload = _normalize_downres_payload(parse_structured_payload(entry))
        completed_shards.add((payload.get("scale"), payload.get("shard")))

    in_progress_shards = {}
    for entry in all_progress_entries:
        payload = _normalize_downres_payload(parse_structured_payload(entry))
        key = (payload.get("scale"), payload.get("shard"))
        if key not in in_progress_shards:
            in_progress_shards[key] = payload

    incomplete = {k: v for k, v in in_progress_shards.items()
                  if k not in completed_shards}

    if incomplete:
        print(f"Incomplete shards (started but no completion log): {len(incomplete)}")
        print("  These shards had progress logs but no Shard complete or failure.")
        print("  Likely cause: OOM kill, task timeout, or infrastructure error.")
        for (scale, shard), payload in sorted(incomplete.items())[:20]:
            mem = payload.get("memory_gib", 0)
            written = payload.get("chunks_written", 0)
            total = payload.get("total", 0)
            print(f"  s{scale} {shard}: "
                  f"{written}/{total} chunks, {mem:.1f}G memory at last progress")
        if len(incomplete) > 20:
            print(f"  ... and {len(incomplete) - 20} more")
        print()

    # Report memory pressure warnings from worker
    if all_memory_entries:
        mem_warnings = []
        for entry in all_memory_entries:
            payload = parse_structured_payload(entry)
            mem_warnings.append(payload)
        critical = [m for m in mem_warnings
                    if "critical" in entry.get("textPayload", "").lower()]
        print(f"Memory pressure events: {len(mem_warnings)} "
              f"({len(critical)} critical)")
        for m in mem_warnings[:10]:
            print(f"  s{m.get('scale', '?')} {m.get('shard', '?')}: "
                  f"{m.get('memory_gib', 0):.1f}G / "
                  f"{m.get('memory_limit_gib', 0):.0f}G "
                  f"({m.get('usage_pct', 0):.0f}%)")
        if len(mem_warnings) > 10:
            print(f"  ... and {len(mem_warnings) - 10} more")
        print()

    if (not all_errors and not shard_load_failures
            and not incomplete and not all_memory_entries):
        print("No errors found!")


if __name__ == "__main__":
    main()
