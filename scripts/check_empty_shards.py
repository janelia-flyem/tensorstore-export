#!/usr/bin/env python3
"""
Check whether missing NG shards correspond to empty DVID Arrow files.

Can run locally or as a Cloud Run Job.  When running as a Cloud Run Job,
reads a task manifest from GCS (same format as precompute_manifest.py)
and checks the assigned shards.  Results are logged via structlog for
Cloud Logging queries.

Local usage:
    pixi run python scripts/check_empty_shards.py --report /tmp/missing_shards.json
    pixi run python scripts/check_empty_shards.py --report /tmp/missing_shards.json --max-shards 10

Cloud Run Job usage (reads SOURCE_PATH, MANIFEST_URI from env):
    # 1. Upload the missing shards list to GCS:
    pixi run python scripts/check_empty_shards.py --upload-manifest /tmp/missing_shards.json

    # 2. Launch the Cloud Run job:
    pixi run python scripts/check_empty_shards.py --launch

    # Or run the worker directly (called by Cloud Run):
    MANIFEST_URI=gs://... SOURCE_PATH=gs://... python scripts/check_empty_shards.py --worker
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE

# How many shards per Cloud Run task
SHARDS_PER_TASK = 20


def check_shard_empty(source_path: str, scale: int, shard_name: str) -> dict:
    """Check if a DVID Arrow shard has empty labels/supervoxels.

    Reads the Arrow file and checks whether any chunk has non-empty
    labels or supervoxels lists in the Arrow metadata.
    """
    from braid import ShardReader

    arrow_uri = f"{source_path}/s{scale}/{shard_name}.arrow"
    csv_uri = f"{source_path}/s{scale}/{shard_name}.csv"

    try:
        reader = ShardReader(arrow_uri, csv_uri)
    except Exception as e:
        return {
            "shard": shard_name,
            "scale": scale,
            "error": str(e)[:200],
        }

    chunks_with_labels = 0
    chunks_with_supervoxels = 0
    total_chunks = reader.chunk_count

    for cx, cy, cz in reader.available_chunks:
        info = reader.get_chunk_info(cx, cy, cz)
        if info.get("labels"):
            chunks_with_labels += 1
        if info.get("supervoxels"):
            chunks_with_supervoxels += 1

    return {
        "shard": shard_name,
        "scale": scale,
        "total_chunks": total_chunks,
        "chunks_with_labels": chunks_with_labels,
        "chunks_with_supervoxels": chunks_with_supervoxels,
        "is_empty": chunks_with_labels == 0 and chunks_with_supervoxels == 0,
    }


def run_worker():
    """Cloud Run worker: read manifest, check assigned shards, log results."""
    import structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
    )
    logger = structlog.get_logger()

    source_path = os.environ["SOURCE_PATH"]
    manifest_uri = os.environ["MANIFEST_URI"]
    task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))

    # Read this task's manifest
    from google.cloud import storage
    bucket_name, blob_path = manifest_uri.replace("gs://", "").split("/", 1)
    task_blob_path = f"{blob_path}/task-{task_index}.json"
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(task_blob_path)
    shards = json.loads(blob.download_as_text())

    logger.info("Worker starting",
                task_index=task_index,
                assigned_shards=len(shards),
                source_path=source_path)

    empty_count = 0
    nonempty_count = 0
    error_count = 0

    for entry in shards:
        scale = entry["scale"]
        shard_name = entry["shard"]
        result = check_shard_empty(source_path, scale, shard_name)

        if "error" in result:
            error_count += 1
            logger.error("Shard check error", **result)
        elif result["is_empty"]:
            empty_count += 1
            logger.info("Shard is empty", **result)
        else:
            nonempty_count += 1
            logger.warning("Shard has data but no NG output", **result)

    logger.info("Worker finished",
                task_index=task_index,
                checked=len(shards),
                empty=empty_count,
                nonempty=nonempty_count,
                errors=error_count)


def upload_manifest(report_path: str) -> str:
    """Upload per-task manifests to GCS from a verify_export JSON report.

    Returns the GCS manifest URI prefix.
    """
    from google.cloud import storage

    env = load_env(ENV_FILE)
    source_path = env["SOURCE_PATH"].rstrip("/")
    bucket_name, source_prefix = source_path.replace("gs://", "").split("/", 1)

    with open(report_path) as f:
        results = json.load(f)

    # Flatten to list of {scale, shard}
    missing = []
    for r in results:
        for entry in r["missing"]:
            for dvid_name in entry["dvid_shards"]:
                missing.append({"scale": r["scale"], "shard": dvid_name})

    if not missing:
        print("No missing shards in report.")
        sys.exit(0)

    # Distribute across tasks
    num_tasks = math.ceil(len(missing) / SHARDS_PER_TASK)
    tasks = {}
    for i, entry in enumerate(missing):
        task_idx = i % num_tasks
        tasks.setdefault(str(task_idx), []).append(entry)

    # Upload
    manifest_prefix = f"{source_prefix}/manifests-check-empty"
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for task_idx, shard_list in tasks.items():
        blob = bucket.blob(f"{manifest_prefix}/task-{task_idx}.json")
        blob.upload_from_string(
            json.dumps(shard_list, separators=(",", ":")),
            content_type="application/json",
        )

    manifest_uri = f"{source_path}/manifests-check-empty"
    print(f"Uploaded {num_tasks} task manifests ({len(missing)} shards)")
    print(f"  {manifest_uri}/task-*.json")
    return manifest_uri


def launch_job(manifest_uri: str):
    """Launch a Cloud Run Job to check the shards."""
    env = load_env(ENV_FILE)
    project = env.get("PROJECT_ID", "")
    region = env.get("REGION", "us-central1")
    source_path = env["SOURCE_PATH"].rstrip("/")
    docker_image = env.get("DOCKER_IMAGE", "")

    if not docker_image:
        print("Error: DOCKER_IMAGE not set in .env. Run `pixi run deploy` first.")
        sys.exit(1)

    # Count tasks from manifest
    from google.cloud import storage
    bucket_name, prefix = manifest_uri.replace("gs://", "").split("/", 1)
    client = storage.Client()
    blobs = list(client.bucket(bucket_name).list_blobs(prefix=prefix + "/task-"))
    num_tasks = len(blobs)

    job_name = "check-empty-shards"

    # Delete existing job if any
    subprocess.run(
        ["gcloud", "run", "jobs", "delete", job_name,
         "--project", project, "--region", region, "--quiet"],
        capture_output=True,
    )

    # Create and run the job
    cmd = [
        "gcloud", "run", "jobs", "create", job_name,
        "--project", project,
        "--region", region,
        "--image", docker_image,
        "--tasks", str(num_tasks),
        "--task-timeout", "600s",
        "--max-retries", "1",
        "--memory", "2Gi",
        "--cpu", "1",
        "--set-env-vars",
        f"SOURCE_PATH={source_path},"
        f"MANIFEST_URI={manifest_uri},"
        "CHECK_EMPTY=1",
        "--command", "python",
        "--args", "scripts/check_empty_shards.py,--worker",
    ]
    print(f"Creating Cloud Run job: {job_name} ({num_tasks} tasks)")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error creating job: {result.stderr}")
        sys.exit(1)

    # Execute
    print("Executing job...")
    exec_cmd = [
        "gcloud", "run", "jobs", "execute", job_name,
        "--project", project,
        "--region", region,
        "--wait",
    ]
    subprocess.run(exec_cmd)


def collect_empty_from_logs(output_path: str):
    """Collect empty shard list from Cloud Logging after a Cloud Run job.

    Queries the check-empty-shards job logs for "Shard is empty" events
    and writes a JSON file suitable for precompute_manifest.py --exclude-empty.
    """
    import subprocess

    env = load_env(ENV_FILE)
    project = env.get("PROJECT_ID", "")

    print("Querying Cloud Logging for empty shards...")
    result = subprocess.run(
        ["gcloud", "logging", "read",
         'resource.type="cloud_run_job" AND '
         'resource.labels.job_name="check-empty-shards" AND '
         'jsonPayload.event="Shard is empty"',
         "--project", project,
         "--limit", "10000",
         "--format", "json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error querying logs: {result.stderr}")
        sys.exit(1)

    entries = json.loads(result.stdout)
    empty_shards = []
    for entry in entries:
        jp = entry.get("jsonPayload", {})
        if jp.get("shard") and jp.get("scale") is not None:
            empty_shards.append({"scale": jp["scale"], "shard": jp["shard"]})

    # Deduplicate
    seen = set()
    unique = []
    for e in empty_shards:
        key = (e["scale"], e["shard"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    with open(output_path, "w") as f:
        json.dump(unique, f, indent=2)

    print(f"Wrote {len(unique)} empty shards to {output_path}")


def run_local(report_path: str, max_shards: int):
    """Run checks locally (slow — downloads each Arrow file)."""
    with open(report_path) as f:
        results = json.load(f)

    env = load_env(ENV_FILE)
    source_path = env["SOURCE_PATH"].rstrip("/")

    missing = []
    for r in results:
        for entry in r["missing"]:
            for dvid_name in entry["dvid_shards"]:
                missing.append((r["scale"], dvid_name))

    if max_shards > 0:
        missing = missing[:max_shards]

    print(f"Checking {len(missing)} missing DVID shards...\n")

    empty_count = 0
    nonempty_count = 0
    error_count = 0
    nonempty_shards = []

    for i, (scale, shard_name) in enumerate(missing):
        result = check_shard_empty(source_path, scale, shard_name)

        if "error" in result:
            error_count += 1
            print(f"  ERROR s{scale}/{shard_name}: {result['error']}")
        elif result["is_empty"]:
            empty_count += 1
        else:
            nonempty_count += 1
            nonempty_shards.append(result)
            print(f"  NON-EMPTY s{scale}/{shard_name}: "
                  f"{result['chunks_with_labels']} with labels, "
                  f"{result['chunks_with_supervoxels']} with svs "
                  f"(of {result['total_chunks']})")

        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(missing)} "
                  f"(empty={empty_count}, non-empty={nonempty_count})")

    print(f"\n{'=' * 60}")
    print(f"Checked: {len(missing)}  Empty: {empty_count}  "
          f"Non-empty: {nonempty_count}  Errors: {error_count}")

    if nonempty_shards:
        print(f"\n*** {nonempty_count} non-empty shards missing NG output! ***")
        for r in nonempty_shards:
            print(f"  s{r['scale']}/{r['shard']}: "
                  f"labels={r['chunks_with_labels']}, "
                  f"svs={r['chunks_with_supervoxels']}")
    else:
        print(f"\nAll {empty_count} missing shards are empty (all-zero). "
              f"DVID exported empty Arrow files — no data was lost.")


def main():
    parser = argparse.ArgumentParser(
        description="Check whether missing NG shards are empty DVID Arrow files")
    parser.add_argument("--report", type=str,
                        help="Path to JSON report from verify_export.py")
    parser.add_argument("--max-shards", type=int, default=0,
                        help="Max shards to check locally (0 = all)")
    parser.add_argument("--upload-manifest", type=str, metavar="REPORT_PATH",
                        help="Upload per-task manifests to GCS from report")
    parser.add_argument("--launch", action="store_true",
                        help="Upload manifests and launch Cloud Run job")
    parser.add_argument("--worker", action="store_true",
                        help="Run as Cloud Run worker (reads MANIFEST_URI env)")
    parser.add_argument("--output-empty", type=str, metavar="PATH",
                        help="Write empty shard list as JSON (for --exclude-empty "
                             "in precompute_manifest.py). Extracts from Cloud Logging.")
    args = parser.parse_args()

    if args.worker:
        run_worker()
    elif args.output_empty:
        collect_empty_from_logs(args.output_empty)
    elif args.launch:
        if not args.report and not args.upload_manifest:
            print("Error: --launch requires --report or --upload-manifest")
            sys.exit(1)
        report = args.report or args.upload_manifest
        uri = upload_manifest(report)
        launch_job(uri)
    elif args.upload_manifest:
        upload_manifest(args.upload_manifest)
    elif args.report:
        run_local(args.report, args.max_shards)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
