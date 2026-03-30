#!/usr/bin/env python3
"""
Aggregate partial next-scale label predictions into per-shard label files.

After the profiler (or a downres worker) writes partial predictions
(<shard>-s{N}-predicted.csv) for the next scale, this script merges them
into complete per-target-shard -labels.csv files that the manifest
generator can consume for label-aware tier assignment.

Multiple parent shards may contribute chunks to the same child shard.
Each partial file has the exact label union for the parent's subset of
child chunks.  For chunks that appear in only one partial file (the
common case), the count is exact.  For any overlap, we take the max
(closer to the true union than summing).

Usage:
    pixi run aggregate-labels --source gs://bucket/exports/seg --target-scale 1
    pixi run aggregate-labels --source gs://... --target-scale 2 --ng-spec path/to/spec.json
"""

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE
from scripts.profile_shards import _get_gcs_client, _parse_gs, _write_string

_gcs_client_local = None


def _list_prediction_files(source_path: str, parent_scale: int,
                           target_scale: int) -> list:
    """List all *-s{target_scale}-predicted.csv files at parent_scale.

    Returns list of GCS blob paths (or local paths).
    """
    suffix = f"-s{target_scale}-predicted.csv"
    prefix = f"{source_path}/s{parent_scale}/"

    if prefix.startswith("gs://"):
        bucket_name, blob_prefix = _parse_gs(prefix)
        client = _get_gcs_client()
        paths = []
        for blob in client.bucket(bucket_name).list_blobs(prefix=blob_prefix):
            if blob.name.endswith(suffix):
                paths.append(f"gs://{bucket_name}/{blob.name}")
        return paths
    else:
        p = Path(prefix)
        if not p.exists():
            return []
        return [str(f) for f in p.glob(f"*{suffix}")]


def _read_csv_rows(path: str) -> list:
    """Read a labels CSV and return list of (x, y, z, unique_labels) tuples."""
    if path.startswith("gs://"):
        bucket_name, blob_path = _parse_gs(path)
        text = _get_gcs_client().bucket(bucket_name).blob(blob_path).download_as_text()
    else:
        text = Path(path).read_text()

    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        rows.append((
            int(row["x"]), int(row["y"]), int(row["z"]),
            int(row["unique_labels"]),
        ))
    return rows


def aggregate_labels(source_path: str, target_scale: int,
                     ng_spec_path: str) -> int:
    """Merge partial predictions into per-target-shard -labels.csv files.

    Args:
        source_path: GCS or local path to the export root (contains s0/, s1/, ...).
        target_scale: The scale whose labels are being assembled (e.g., 1).
        ng_spec_path: Path to the NG spec JSON file (for shard assignment).

    Returns:
        Number of target shard label files written.
    """
    from src.ng_sharding import (
        load_ng_spec,
        compressed_z_index,
        chunk_shard_info,
    )

    parent_scale = target_scale - 1
    spec = load_ng_spec(ng_spec_path)
    target_params = spec[target_scale]

    # Step 1: List all prediction files from the parent scale
    pred_files = _list_prediction_files(source_path, parent_scale, target_scale)
    if not pred_files:
        print(f"  No prediction files found at s{parent_scale} for target s{target_scale}")
        return 0

    print(f"  Found {len(pred_files)} prediction files at s{parent_scale}")

    # Step 2: Read all predictions and merge by chunk coordinate.
    # For chunks appearing in multiple files, take max unique_labels
    # (each file has exact union within its parent shard; max is closer
    # to the true union than sum).
    chunk_labels = {}  # (cx, cy, cz) -> max unique_labels
    files_read = 0

    def _read_one(path):
        return _read_csv_rows(path)

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_read_one, p): p for p in pred_files}
        for future in as_completed(futures):
            for cx, cy, cz, ul in future.result():
                key = (cx, cy, cz)
                if key not in chunk_labels or ul > chunk_labels[key]:
                    chunk_labels[key] = ul
            files_read += 1
            if files_read % 1000 == 0 or files_read == len(pred_files):
                print(f"  Reading predictions: {files_read}/{len(pred_files)} files, "
                      f"{len(chunk_labels):,} chunks so far")

    print(f"  Merged {len(chunk_labels):,} unique chunk predictions")

    # Step 3: Group chunks by target shard number
    coord_bits = target_params["coord_bits"]
    preshift = target_params["preshift_bits"]
    minishard_bits = target_params["minishard_bits"]
    shard_bits = target_params["shard_bits"]

    shard_chunks = defaultdict(list)  # shard_number -> [(cx, cy, cz, ul)]
    grouped = 0
    total_chunks = len(chunk_labels)
    log_interval = max(1, total_chunks // 10)
    for (cx, cy, cz), ul in chunk_labels.items():
        morton = compressed_z_index((cx, cy, cz), coord_bits)
        sn, _ = chunk_shard_info(morton, preshift, minishard_bits, shard_bits)
        shard_chunks[sn].append((cx, cy, cz, ul))
        grouped += 1
        if grouped % log_interval == 0:
            print(f"  Grouping: {grouped:,}/{total_chunks:,} chunks")

    print(f"  Grouped into {len(shard_chunks)} target shards")

    # Step 4: Write one -labels.csv per target shard
    hex_digits = -(-shard_bits // 4)
    written = 0

    def _write_shard_csv(shard_number, chunks):
        shard_hex = f"{shard_number:0{hex_digits}x}"
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["x", "y", "z", "num_labels", "num_supervoxels",
                         "unique_labels"])
        for cx, cy, cz, ul in sorted(chunks):
            writer.writerow([cx, cy, cz, ul, ul, ul])
        csv_path = f"{source_path}/s{target_scale}/{shard_hex}-labels.csv"
        _write_string(csv_path, out.getvalue())

    total_to_write = len(shard_chunks)
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {
            pool.submit(_write_shard_csv, sn, chunks): sn
            for sn, chunks in shard_chunks.items()
        }
        for future in as_completed(futures):
            future.result()  # propagate exceptions
            written += 1
            if written % 500 == 0 or written == total_to_write:
                print(f"  Writing label CSVs: {written}/{total_to_write}")

    print(f"  Written {written} label files to {source_path}/s{target_scale}/")

    # Write a summary JSON so downstream tools can read one file instead of thousands
    summary = {
        f"{sn:0{hex_digits}x}": sum(ul for _, _, _, ul in chunks)
        for sn, chunks in shard_chunks.items()
    }
    summary_path = f"{source_path}/s{target_scale}/labels-summary.json"
    _write_string(summary_path, json.dumps(summary, separators=(",", ":")))
    print(f"  Written label summary: {summary_path}")

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate partial label predictions into per-shard label files.",
    )
    parser.add_argument(
        "--source",
        help="GCS or local path to export root (default: SOURCE_PATH from .env)",
    )
    parser.add_argument(
        "--target-scale", type=int, required=True,
        help="Target scale to assemble labels for (e.g., 1)",
    )
    parser.add_argument(
        "--ng-spec",
        help="Path to NG spec JSON file (default: NG_SPEC_PATH from .env)",
    )
    args = parser.parse_args()

    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)
    source_path = args.source or env.get("SOURCE_PATH", "")
    ng_spec_path = args.ng_spec or env.get("NG_SPEC_PATH", "")

    if not source_path:
        print("Error: SOURCE_PATH not configured. Use --source or set in .env.")
        sys.exit(1)
    if not ng_spec_path:
        print("Error: NG_SPEC_PATH not configured. Use --ng-spec or set in .env.")
        sys.exit(1)

    spec_path = Path(ng_spec_path)
    if not spec_path.is_absolute():
        spec_path = Path(__file__).resolve().parent.parent / spec_path

    print(f"Aggregating s{args.target_scale} labels from "
          f"s{args.target_scale - 1} predictions...")
    written = aggregate_labels(source_path, args.target_scale, str(spec_path))
    if written == 0:
        print("No label files written.")
    else:
        print(f"Done: {written} shard label files.")


if __name__ == "__main__":
    main()
