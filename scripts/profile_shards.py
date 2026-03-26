#!/usr/bin/env python3
"""
Generate per-shard label profile CSVs from DVID Arrow IPC shard files.

For each Arrow shard, reads the labels and supervoxels list columns and
writes a companion CSV with per-chunk label counts.  This data is used
for label-aware memory estimation in tier assignment.

Output format (<shard>-labels.csv):
    x,y,z,num_labels,num_supervoxels
    480,384,448,13,14
    480,384,449,25,31

Can run locally or as a Cloud Run job.  In Cloud Run mode, uses
CLOUD_RUN_TASK_INDEX and CLOUD_RUN_TASK_COUNT for work partitioning,
plus PROFILER_SOURCE, PROFILER_OUTPUT, and PROFILER_SCALES env vars.

Usage (local):
    pixi run profile-shards
    pixi run profile-shards --scales 0,1
    pixi run profile-shards --source gs://bucket/exports/seg --output gs://bucket2/exports/seg

Usage (Cloud Run):
    pixi run launch-profiler --source gs://... --output gs://... --tasks 100
"""

import argparse
import csv
import io
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.ipc as ipc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE

# Shared GCS client — creating one per call is extremely expensive.
# Thread-safe: google-cloud-storage Client uses urllib3 connection pooling.
_gcs_client = None


def _get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client


def _parse_gs(path: str):
    rest = path[len("gs://"):]
    bucket_name, _, blob_path = rest.partition("/")
    return bucket_name, blob_path


def _read_bytes(path: str) -> bytes:
    """Read all bytes from a local path or GCS URI."""
    if path.startswith("gs://"):
        bucket_name, blob_path = _parse_gs(path)
        blob = _get_gcs_client().bucket(bucket_name).blob(blob_path)
        return blob.download_as_bytes()
    return Path(path).read_bytes()


def _write_string(path: str, content: str):
    """Write a string to a local path or GCS URI."""
    if path.startswith("gs://"):
        bucket_name, blob_path = _parse_gs(path)
        blob = _get_gcs_client().bucket(bucket_name).blob(blob_path)
        blob.upload_from_string(content, content_type="text/csv")
    else:
        Path(path).write_text(content)


def _exists(path: str) -> bool:
    """Check if a file exists locally or on GCS."""
    if path.startswith("gs://"):
        bucket_name, blob_path = _parse_gs(path)
        return _get_gcs_client().bucket(bucket_name).blob(blob_path).exists()
    return Path(path).exists()


def list_shards(source_path: str, scales: list) -> list:
    """List all Arrow shard files across scales.

    Returns list of (scale, shard_name) tuples.
    """
    all_shards = []
    for scale in scales:
        prefix = f"{source_path}/s{scale}/"
        if prefix.startswith("gs://"):
            bucket_name, blob_prefix = _parse_gs(prefix)
            blobs = _get_gcs_client().bucket(bucket_name).list_blobs(
                prefix=blob_prefix)
            for blob in blobs:
                if blob.name.endswith(".arrow"):
                    name = blob.name.split("/")[-1].replace(".arrow", "")
                    all_shards.append((scale, name))
        else:
            result = subprocess.run(
                ["ls", prefix], capture_output=True, text=True)
            for line in result.stdout.splitlines():
                if line.strip().endswith(".arrow"):
                    name = line.strip().replace(".arrow", "")
                    all_shards.append((scale, name))
    return all_shards


def profile_shard(source_path: str, scale: int, shard_name: str) -> dict:
    """Read an Arrow shard and extract per-chunk label counts.

    Returns a dict with 'rows' (list of (x,y,z,num_labels,num_sv)) and
    summary stats, or a dict with 'error' key on failure.
    """
    arrow_path = f"{source_path}/s{scale}/{shard_name}.arrow"
    try:
        buf = _read_bytes(arrow_path)
        reader = ipc.open_stream(buf)
        table = reader.read_all()
    except Exception as e:
        return {"error": str(e)[:200]}

    arrow_bytes = len(buf)
    xs = table.column("chunk_x").to_pylist()
    ys = table.column("chunk_y").to_pylist()
    zs = table.column("chunk_z").to_pylist()
    label_lengths = table.column("labels").combine_chunks().value_lengths().to_pylist()
    sv_lengths = table.column("supervoxels").combine_chunks().value_lengths().to_pylist()

    rows = list(zip(xs, ys, zs, label_lengths, sv_lengths))
    total_labels = sum(label_lengths)
    total_sv = sum(sv_lengths)
    n = len(rows)

    return {
        "rows": rows,
        "chunk_count": n,
        "arrow_bytes": arrow_bytes,
        "total_labels": total_labels,
        "total_supervoxels": total_sv,
        "mean_labels": total_labels / n if n else 0,
        "max_labels": max(label_lengths) if label_lengths else 0,
        "mean_supervoxels": total_sv / n if n else 0,
        "max_supervoxels": max(sv_lengths) if sv_lengths else 0,
    }


def write_labels_csv(output_path: str, scale: int, shard_name: str,
                     rows: list):
    """Write a -labels.csv file for a shard."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["x", "y", "z", "num_labels", "num_supervoxels"])
    for x, y, z, nl, nsv in rows:
        writer.writerow([x, y, z, nl, nsv])

    csv_path = f"{output_path}/s{scale}/{shard_name}-labels.csv"
    _write_string(csv_path, out.getvalue())
    return csv_path


def process_shards(source_path: str, output_path: str,
                   shards: list, workers: int):
    """Profile a list of shards and write label CSVs.

    Args:
        source_path: GCS or local path to read Arrow files from
        output_path: GCS or local path to write -labels.csv files to
        shards: list of (scale, shard_name) tuples
        workers: number of parallel threads

    Returns:
        (completed, errors, total_arrow_bytes, elapsed_seconds)
    """
    completed = 0
    errors = 0
    total_arrow_bytes = 0
    start_time = time.time()

    def _process_one(scale_and_name):
        scale, name = scale_and_name
        result = profile_shard(source_path, scale, name)
        if "error" not in result:
            write_labels_csv(output_path, scale, name, result["rows"])
        return scale, name, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one, sn): sn
            for sn in shards
        }
        for future in as_completed(futures):
            scale, name, result = future.result()
            if "error" in result:
                errors += 1
                print(f"  Error: s{scale}/{name}: {result['error']}")
            else:
                completed += 1
                total_arrow_bytes += result.get("arrow_bytes", 0)
                if completed % 100 == 0 or completed == len(shards):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"  {completed}/{len(shards)} profiled"
                          f" ({errors} errors)"
                          f" [{elapsed:.0f}s, {rate:.1f}/s]")

    elapsed = time.time() - start_time
    return completed, errors, total_arrow_bytes, elapsed


def run_cloud_run_task(source_path: str, output_path: str,
                       all_shards: list, workers: int):
    """Cloud Run mode: process this task's partition of shards.

    Uses CLOUD_RUN_TASK_INDEX and CLOUD_RUN_TASK_COUNT for partitioning.
    """
    task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
    task_count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1"))

    # Round-robin partition
    my_shards = [s for i, s in enumerate(all_shards)
                 if i % task_count == task_index]

    print(f"Task {task_index}/{task_count}: "
          f"{len(my_shards)} shards assigned (of {len(all_shards)} total)")

    if not my_shards:
        print("No shards to process.")
        return

    completed, errors, arrow_bytes, elapsed = process_shards(
        source_path, output_path, my_shards, workers)

    total_gb = arrow_bytes / 1e9
    rate = completed / elapsed if elapsed > 0 else 0
    bw = total_gb / elapsed if elapsed > 0 else 0
    print(f"\nTask {task_index} done: {completed} profiled, {errors} errors")
    print(f"  Elapsed: {elapsed / 60:.1f} min")
    print(f"  Throughput: {rate:.1f} shards/s, {bw:.2f} GB/s")


def main():
    # Check for Cloud Run mode first
    if os.environ.get("CLOUD_RUN_TASK_INDEX") is not None:
        source_path = os.environ["PROFILER_SOURCE"]
        output_path = os.environ["PROFILER_OUTPUT"]
        scales_str = os.environ.get("PROFILER_SCALES", "0,1,2,3,4,5,6,7,8,9")
        scales = [int(s.strip()) for s in scales_str.split(",")]
        workers = int(os.environ.get("PROFILER_WORKERS", "32"))

        print("Cloud Run profiler task")
        print(f"  Source: {source_path}")
        print(f"  Output: {output_path}")
        print(f"  Scales: {scales_str}")
        print(f"  Workers: {workers}")

        all_shards = list_shards(source_path, scales)
        print(f"  Total shards: {len(all_shards)}")
        run_cloud_run_task(source_path, output_path, all_shards, workers)
        return

    # Local CLI mode
    parser = argparse.ArgumentParser(
        description="Generate per-shard label profile CSVs from Arrow files.",
    )
    parser.add_argument(
        "--source",
        help="GCS or local path to shard export (default: SOURCE_PATH from .env)",
    )
    parser.add_argument(
        "--output",
        help="GCS or local path to write -labels.csv files (default: same as --source)",
    )
    parser.add_argument(
        "--scales",
        help="Comma-separated scales (default: SCALES from .env)",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count shards needing profiling without writing",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate even if -labels.csv already exists",
    )
    parser.add_argument(
        "--skip-check", action="store_true",
        help="Skip checking for existing -labels.csv files (faster first run)",
    )
    args = parser.parse_args()

    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)
    source_path = args.source or env.get("SOURCE_PATH", "")
    output_path = args.output or source_path
    scales_str = args.scales or env.get("SCALES", "0")
    scales = [int(s.strip()) for s in scales_str.split(",")]

    if not source_path:
        print("Error: SOURCE_PATH not configured. Use --source or set in .env.")
        sys.exit(1)

    print(f"Source: {source_path}")
    if output_path != source_path:
        print(f"Output: {output_path}")
    print(f"Scanning shards across {len(scales)} scales...")
    all_shards = list_shards(source_path, scales)
    print(f"  Found {len(all_shards)} Arrow shards")

    # Check which shards already have label CSVs (in output path)
    need_profiling = []
    if args.force or args.skip_check:
        need_profiling = all_shards
        if args.skip_check:
            print("  Skipping existence check (--skip-check)")
    else:
        # List existing label CSVs in bulk (one list_blobs per scale)
        # instead of one exists() call per shard (26K API calls).
        print("Scanning for existing -labels.csv files...")
        existing = set()
        for scale in scales:
            prefix = f"{output_path}/s{scale}/"
            if prefix.startswith("gs://"):
                bucket_name, blob_prefix = _parse_gs(prefix)
                for blob in _get_gcs_client().bucket(bucket_name).list_blobs(
                        prefix=blob_prefix):
                    if blob.name.endswith("-labels.csv"):
                        name = blob.name.split("/")[-1].replace("-labels.csv", "")
                        existing.add((scale, name))
            else:
                for f in Path(prefix).glob("*-labels.csv"):
                    existing.add((scale, f.stem.replace("-labels", "")))
        for scale, name in all_shards:
            if (scale, name) not in existing:
                need_profiling.append((scale, name))
        print(f"  {len(need_profiling)} shards need profiling "
              f"({len(all_shards) - len(need_profiling)} already done)")

    if args.dry_run:
        by_scale = {}
        for scale, _ in need_profiling:
            by_scale[scale] = by_scale.get(scale, 0) + 1
        for s in sorted(by_scale):
            print(f"    scale {s}: {by_scale[s]} shards")
        print("\n(dry run — nothing written)")
        return

    if not need_profiling:
        print("All shards already profiled.")
        return

    print(f"\nProfiling {len(need_profiling)} shards "
          f"with {args.workers} workers...")

    completed, errors, arrow_bytes, elapsed = process_shards(
        source_path, output_path, need_profiling, args.workers)

    total_gb = arrow_bytes / 1e9
    rate = completed / elapsed if elapsed > 0 else 0
    bw = total_gb / elapsed if elapsed > 0 else 0
    print(f"\nDone: {completed} profiled, {errors} errors")
    print(f"  Elapsed: {elapsed / 60:.1f} min ({elapsed:.0f}s)")
    print(f"  Throughput: {rate:.1f} shards/s with {args.workers} workers")
    print(f"  Downloaded: {total_gb:.1f} GB ({bw:.2f} GB/s)")


if __name__ == "__main__":
    main()
