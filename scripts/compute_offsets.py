#!/usr/bin/env python3
"""
Compute byte offsets for Arrow IPC streaming shard files.

Scans each Arrow file to find the byte offset and size of every record batch
message, then writes a companion CSV with columns (x,y,z,rec,offset,size).
The first line of the CSV is a comment with the schema size so that readers
can download the schema bytes separately for decoding.

This enables GCS range reads: instead of downloading an entire multi-GB Arrow
file, a reader can fetch just the schema (first `schema_size` bytes) and then
individual record batches by byte range.

Distributable via Cloud Run Jobs: each task reads its shard list from a
manifest file or from a round-robin partition of all shards.

Usage:
    # Local: process all scales
    pixi run compute-offsets

    # Dry run: show what would be processed
    pixi run compute-offsets --dry-run

    # Specific scales only
    pixi run compute-offsets --scales 0,1

    # As a Cloud Run task (reads CLOUD_RUN_TASK_INDEX)
    python scripts/compute_offsets.py
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE

# Lazy imports for pyarrow (only needed at processing time)
pa = None
ipc = None


def _ensure_pyarrow():
    global pa, ipc
    if pa is None:
        import pyarrow as _pa
        import pyarrow.ipc as _ipc
        pa = _pa
        ipc = _ipc


def _parse_gs_uri(uri: str):
    rest = uri[len("gs://"):]
    bucket, _, path = rest.partition("/")
    return bucket, path.rstrip("/")


def scan_record_offsets(arrow_bytes: bytes) -> tuple:
    """Scan an Arrow IPC stream to find byte offsets of each record batch.

    Args:
        arrow_bytes: Raw bytes of the Arrow IPC streaming file.

    Returns:
        (schema_size, offsets) where:
          - schema_size: byte size of the schema message on wire
          - offsets: list of (record_index, byte_offset, byte_size) tuples
    """
    _ensure_pyarrow()
    buf = pa.BufferReader(arrow_bytes)
    pos = 0
    offsets = []
    schema_size = 0

    while True:
        try:
            msg = ipc.read_message(buf)
            if msg is None:
                break
        except (StopIteration, EOFError):
            break

        meta_len = msg.metadata.size if msg.metadata else 0
        body_len = msg.body.size if msg.body else 0
        padded_meta = (meta_len + 7) & ~7
        padded_body = (body_len + 7) & ~7
        total = 4 + 4 + padded_meta + padded_body  # continuation + len + meta + body

        if msg.type == "schema":
            schema_size = total
        elif msg.type == "record batch":
            offsets.append((len(offsets), pos, total))

        pos += total

    return schema_size, offsets


def process_shard(storage_client, source_path: str, scale: int,
                  shard_name: str) -> dict:
    """Compute offsets for one shard and upload the offset CSV to GCS.

    Returns a dict with processing stats.
    """
    from google.cloud import storage as gcs_storage

    bucket_name, prefix = _parse_gs_uri(source_path)
    bucket = storage_client.bucket(bucket_name)

    arrow_blob_path = f"{prefix}/s{scale}/{shard_name}.arrow"
    csv_blob_path = f"{prefix}/s{scale}/{shard_name}.csv"
    offsets_blob_path = f"{prefix}/s{scale}/{shard_name}-offsets.csv"

    # Check if offset CSV already exists (idempotent)
    offsets_blob = bucket.blob(offsets_blob_path)
    if offsets_blob.exists():
        return {"shard": shard_name, "scale": scale, "status": "exists"}

    t0 = time.time()

    # Download Arrow file
    arrow_blob = bucket.blob(arrow_blob_path)
    arrow_bytes = arrow_blob.download_as_bytes()
    download_s = time.time() - t0

    # Scan for record offsets
    t1 = time.time()
    schema_size, offsets = scan_record_offsets(arrow_bytes)
    scan_s = time.time() - t1

    # Download existing CSV to get (x,y,z) -> rec mapping
    csv_blob = bucket.blob(csv_blob_path)
    csv_bytes = csv_blob.download_as_bytes()
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))

    # Build offset CSV: merge (x,y,z,rec) with (offset,size)
    out = io.StringIO()
    out.write(f"# schema_size={schema_size}\n")
    writer = csv.writer(out)
    writer.writerow(["x", "y", "z", "rec", "offset", "size"])
    for row in reader:
        rec = int(row["rec"])
        if rec < len(offsets):
            _, off, sz = offsets[rec]
            writer.writerow([row["x"], row["y"], row["z"], rec, off, sz])

    # Upload offset CSV
    t2 = time.time()
    offsets_blob.upload_from_string(out.getvalue(), content_type="text/csv")
    upload_s = time.time() - t2

    return {
        "shard": shard_name,
        "scale": scale,
        "status": "ok",
        "records": len(offsets),
        "schema_size": schema_size,
        "arrow_mb": len(arrow_bytes) / 1e6,
        "download_s": round(download_s, 1),
        "scan_s": round(scan_s, 1),
        "upload_s": round(upload_s, 1),
    }


def list_shards(source_path: str, scales: list) -> list:
    """List all (scale, shard_name) pairs across scales."""
    all_shards = []
    for scale in scales:
        prefix = f"{source_path}/s{scale}/"
        result = subprocess.run(
            ["gsutil", "ls", prefix],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  Warning: could not list {prefix}")
            continue
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.endswith(".arrow"):
                name = line.split("/")[-1].replace(".arrow", "")
                all_shards.append((scale, name))
    return all_shards


def main():
    parser = argparse.ArgumentParser(
        description="Compute byte offsets for Arrow shard files.",
    )
    parser.add_argument(
        "--scales",
        help="Comma-separated scales (default: from .env SCALES)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without doing it",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel download/upload threads (default: 8)",
    )
    args = parser.parse_args()

    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)
    source_path = env.get("SOURCE_PATH", "")
    if not source_path:
        print("Error: SOURCE_PATH must be configured in .env")
        sys.exit(1)

    scales_str = args.scales or env.get("SCALES", "0")
    scales = [int(s.strip()) for s in scales_str.split(",")]

    # Cloud Run task partitioning
    task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
    task_count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1"))

    print(f"Scanning shards across {len(scales)} scales...")
    all_shards = list_shards(source_path, scales)
    print(f"  Found {len(all_shards)} shards")

    # Partition for Cloud Run
    my_shards = [s for i, s in enumerate(all_shards)
                 if i % task_count == task_index]
    if task_count > 1:
        print(f"  Task {task_index}/{task_count}: processing {len(my_shards)} shards")

    if args.dry_run:
        from collections import Counter
        by_scale = Counter(s for s, _ in my_shards)
        for s in sorted(by_scale):
            print(f"  Scale {s}: {by_scale[s]} shards")
        print(f"\n(dry run — nothing processed)")
        return

    from google.cloud import storage
    storage_client = storage.Client()

    processed = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_shard, storage_client, source_path, scale, name): (scale, name)
            for scale, name in my_shards
        }
        for future in as_completed(futures):
            scale, name = futures[future]
            try:
                result = future.result()
                if result["status"] == "exists":
                    skipped += 1
                else:
                    processed += 1
                    if processed % 100 == 0 or processed == 1:
                        elapsed = time.time() - t_start
                        rate = processed / elapsed if elapsed > 0 else 0
                        print(f"  [{processed}/{len(my_shards)}] "
                              f"s{scale}/{name}: {result.get('records', 0)} records, "
                              f"{result.get('arrow_mb', 0):.0f}MB "
                              f"({rate:.1f} shards/s)")
            except Exception as e:
                failed += 1
                print(f"  FAILED s{scale}/{name}: {str(e)[:200]}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s: {processed} processed, "
          f"{skipped} skipped (exist), {failed} failed")


if __name__ == "__main__":
    main()
