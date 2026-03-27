#!/usr/bin/env python3
"""
Validate DVID export shard-to-chunk assignment before running the pipeline.

For each DVID Arrow shard, downloads the CSV index and verifies that all
chunks within the shard map to the same neuroglancer shard number via the
compressed Z-index.  Catches upstream DVID bugs like incorrect Morton code
implementations.

Usage:
    pixi run validate-dvid
    pixi run validate-dvid -- --scales 0,1,2
    pixi run validate-dvid -- --scales 0 --sample 100
"""

import argparse
import csv
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE
from src.ng_sharding import (
    compressed_z_index,
    chunk_shard_info,
    load_ng_spec,
)

_gcs_client = None


def _get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client


def _parse_gs(path):
    rest = path[len("gs://"):]
    bucket_name, _, blob_path = rest.partition("/")
    return bucket_name, blob_path


def download_csv_text(source_path, scale, shard_name):
    """Download CSV index for a DVID shard. Returns text content."""
    uri = f"{source_path}/s{scale}/{shard_name}.csv"
    bucket_name, blob_path = _parse_gs(uri)
    client = _get_gcs_client()
    blob = client.bucket(bucket_name).blob(blob_path)
    return blob.download_as_text()


def parse_csv_chunks(csv_text):
    """Parse chunk coordinates from a DVID shard CSV index.

    Returns list of (cx, cy, cz) tuples.
    """
    chunks = []
    lines = csv_text.strip().splitlines()
    # Skip comment lines
    data_lines = [l for l in lines if not l.startswith("#")]
    reader = csv.DictReader(data_lines)
    for row in reader:
        chunks.append((int(row["x"]), int(row["y"]), int(row["z"])))
    return chunks


def validate_shard(source_path, scale, shard_name, scale_params):
    """Validate a single DVID shard.

    Returns dict with validation results.
    """
    csv_text = download_csv_text(source_path, scale, shard_name)
    chunks = parse_csv_chunks(csv_text)

    coord_bits = scale_params["coord_bits"]
    preshift = scale_params["preshift_bits"]
    minishard_bits = scale_params["minishard_bits"]
    shard_bits = scale_params["shard_bits"]
    grid_shape = scale_params["grid_shape"]

    shard_numbers = set()
    out_of_bounds = []

    for cx, cy, cz in chunks:
        # Check bounds against grid shape
        if cx >= grid_shape[0] or cy >= grid_shape[1] or cz >= grid_shape[2]:
            out_of_bounds.append((cx, cy, cz))

        z_idx = compressed_z_index((cx, cy, cz), coord_bits)
        shard_num, _ = chunk_shard_info(z_idx, preshift, minishard_bits, shard_bits)
        shard_numbers.add(shard_num)

    consistent = len(shard_numbers) <= 1

    # Also check origin-based prediction
    origin_shard = None
    parts = shard_name.split("_")
    if len(parts) == 3:
        ox, oy, oz = int(parts[0]), int(parts[1]), int(parts[2])
        cs = scale_params["chunk_size"]
        origin_cx, origin_cy, origin_cz = ox // cs[0], oy // cs[1], oz // cs[2]
        z_idx = compressed_z_index((origin_cx, origin_cy, origin_cz), coord_bits)
        origin_shard, _ = chunk_shard_info(z_idx, preshift, minishard_bits, shard_bits)

    return {
        "shard_name": shard_name,
        "chunk_count": len(chunks),
        "ng_shard_numbers": sorted(shard_numbers),
        "consistent": consistent,
        "origin_shard": origin_shard,
        "origin_matches": (origin_shard in shard_numbers) if origin_shard is not None else None,
        "out_of_bounds_count": len(out_of_bounds),
        "out_of_bounds_samples": out_of_bounds[:5],
    }


def list_dvid_shard_names(source_path, scale):
    """List DVID shard names for a scale."""
    prefix = f"{source_path}/s{scale}/"
    bucket_name, blob_prefix = _parse_gs(prefix)
    client = _get_gcs_client()
    names = []
    for blob in client.bucket(bucket_name).list_blobs(prefix=blob_prefix):
        if blob.name.endswith(".csv"):
            name = blob.name.split("/")[-1].replace(".csv", "")
            names.append(name)
    return names


def validate_scale(source_path, scale, scale_params, sample_size=None,
                   max_workers=16):
    """Validate all (or a sample of) DVID shards for a scale.

    Returns (total_checked, errors, total_oob_chunks, shard_results).
    """
    shard_names = list_dvid_shard_names(source_path, scale)
    if sample_size and sample_size < len(shard_names):
        shard_names = random.sample(shard_names, sample_size)

    errors = []
    total_chunks = 0
    total_oob = 0

    def _validate_one(name):
        return validate_shard(source_path, scale, name, scale_params)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_validate_one, n): n for n in shard_names}
        for future in as_completed(futures):
            result = future.result()
            total_chunks += result["chunk_count"]
            total_oob += result["out_of_bounds_count"]

            if not result["consistent"]:
                errors.append(result)
            elif result["origin_matches"] is False:
                errors.append(result)

    return len(shard_names), errors, total_chunks, total_oob


def main():
    parser = argparse.ArgumentParser(
        description="Validate DVID export chunk-to-shard assignment")
    parser.add_argument("--scales", type=str, default=None,
                        help="Comma-separated scale indices (default: all)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Random sample N shards per scale")
    parser.add_argument("--workers", type=int, default=16,
                        help="Max parallel CSV downloads (default: 16)")
    args = parser.parse_args()

    env = load_env(ENV_FILE)
    if not env:
        print(f"Error: {ENV_FILE} not found. Copy from {ENV_EXAMPLE}.")
        sys.exit(1)

    source_path = env.get("SOURCE_PATH", "").rstrip("/")
    ng_spec_path = env.get("NG_SPEC_PATH", "")

    if not all([source_path, ng_spec_path]):
        print("Error: SOURCE_PATH and NG_SPEC_PATH must be set in .env")
        sys.exit(1)

    scale_info = load_ng_spec(ng_spec_path)
    if args.scales:
        scales = [int(s) for s in args.scales.split(",")]
    else:
        scales = sorted(scale_info.keys())

    sample_note = f" (sampling {args.sample} per scale)" if args.sample else ""
    print(f"Validating {len(scales)} scales{sample_note}")
    print(f"  Source: {source_path}")

    any_errors = False
    for s in scales:
        if s not in scale_info:
            print(f"\n  Warning: scale {s} not in NG spec, skipping")
            continue

        params = scale_info[s]
        checked, errors, total_chunks, total_oob = validate_scale(
            source_path, s, params,
            sample_size=args.sample, max_workers=args.workers)

        grid = params["grid_shape"]
        print(f"\nScale {s} ({params['key']}): "
              f"{checked} shards, {total_chunks} chunks checked")

        if not errors:
            print("  All chunks consistent "
                  "(every DVID shard maps to exactly one NG shard)")
        else:
            any_errors = True
            for err in errors[:10]:
                if not err["consistent"]:
                    print(f"  *** Shard {err['shard_name']}: chunks map to "
                          f"MULTIPLE NG shards: {err['ng_shard_numbers']} ***")
                elif err["origin_matches"] is False:
                    print(f"  *** Shard {err['shard_name']}: origin maps to "
                          f"NG shard {err['origin_shard']} but chunks map to "
                          f"{err['ng_shard_numbers']} ***")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")

        if total_oob > 0:
            print(f"  {total_oob} chunks outside grid bounds {grid}")

    if any_errors:
        print("\n*** VALIDATION FAILED ***")
        sys.exit(1)
    else:
        print("\nAll scales validated successfully.")


if __name__ == "__main__":
    main()
