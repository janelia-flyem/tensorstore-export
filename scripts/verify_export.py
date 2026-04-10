#!/usr/bin/env python3
"""
Verify export completeness by comparing DVID source shards against
neuroglancer output shard files on GCS.

For each scale, lists all DVID Arrow source files, computes the expected
neuroglancer shard filename for each, and checks whether that file exists
on GCS.  Reports missing shards (DVID sources with no NG output) and
orphaned shards (NG files with no corresponding DVID source).

No dependency on Cloud Logging -- works entirely from GCS listings.

Usage:
    pixi run verify-export
    pixi run verify-export -- --scales 0,1,2
    pixi run verify-export -- --json-report report.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE
from src.ng_sharding import (
    dvid_to_ng_shard_number,
    load_ng_spec,
    ng_shard_filename,
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


def list_dvid_shards(source_path, scale):
    """List DVID Arrow shard names for a scale.

    Returns list of shard names (e.g. "10240_40960_43008").
    """
    prefix = f"{source_path}/s{scale}/"
    bucket_name, blob_prefix = _parse_gs(prefix)
    client = _get_gcs_client()
    names = []
    for blob in client.bucket(bucket_name).list_blobs(prefix=blob_prefix):
        if blob.name.endswith(".arrow"):
            name = blob.name.split("/")[-1].replace(".arrow", "")
            names.append(name)
    return names


def list_ng_shard_files(dest_path, scale_key):
    """List .shard files on GCS for a given scale.

    Returns set of filenames (e.g. {"061c5.shard", "00000.shard"}).
    """
    prefix = f"{dest_path}/{scale_key}/"
    bucket_name, blob_prefix = _parse_gs(prefix)
    client = _get_gcs_client()
    files = set()
    for blob in client.bucket(bucket_name).list_blobs(prefix=blob_prefix):
        fname = blob.name.split("/")[-1]
        if fname.endswith(".shard"):
            files.add(fname)
    return files


def verify_scale(source_path, dest_path, scale_idx, scale_params,
                  z_compress=0):
    """Verify a single scale.

    Returns dict with keys: scale, dvid_shards, expected_ng_shards,
    actual_ng_shards, missing, orphaned, dvid_to_ng_map.
    """
    dvid_names = list_dvid_shards(source_path, scale_idx)

    # Map each DVID shard to its expected NG shard filename
    dvid_to_ng = {}
    ng_to_dvid = defaultdict(list)
    for name in dvid_names:
        shard_num = dvid_to_ng_shard_number(name, scale_params,
                                             z_compress=z_compress)
        fname = ng_shard_filename(shard_num, scale_params["shard_bits"])
        dvid_to_ng[name] = fname
        ng_to_dvid[fname].append(name)

    expected_ng = set(ng_to_dvid.keys())
    actual_ng = list_ng_shard_files(dest_path, scale_params["key"])

    missing = expected_ng - actual_ng
    orphaned = actual_ng - expected_ng

    # Map missing NG shards back to their DVID source shards
    missing_dvid = []
    for ng_file in sorted(missing):
        missing_dvid.append({
            "ng_shard": ng_file,
            "dvid_shards": ng_to_dvid[ng_file],
        })

    return {
        "scale": scale_idx,
        "scale_key": scale_params["key"],
        "dvid_shards": len(dvid_names),
        "expected_ng_shards": len(expected_ng),
        "actual_ng_shards": len(actual_ng),
        "missing_count": len(missing),
        "orphaned_count": len(orphaned),
        "missing": missing_dvid,
        "orphaned": sorted(orphaned),
    }


def verify_all_scales(source_path, dest_path, ng_spec_path, scales,
                      z_compress=0):
    """Verify all requested scales.

    Returns (total_missing_dvid_shards, list_of_scale_reports).
    """
    scale_info = load_ng_spec(ng_spec_path)
    results = []
    total_missing_dvid = 0

    for s in scales:
        if s not in scale_info:
            print(f"  Warning: scale {s} not in NG spec, skipping")
            continue
        result = verify_scale(source_path, dest_path, s, scale_info[s],
                              z_compress=z_compress)
        results.append(result)
        for entry in result["missing"]:
            total_missing_dvid += len(entry["dvid_shards"])

    return total_missing_dvid, results


def print_report(results):
    """Print human-readable verification report."""
    all_ok = True
    for r in results:
        if r["missing_count"] > 0:
            all_ok = False

        dvid_affected = sum(len(m["dvid_shards"]) for m in r["missing"])
        print(f"\nScale {r['scale']} ({r['scale_key']}): "
              f"{r['dvid_shards']} DVID shards -> "
              f"{r['expected_ng_shards']} expected NG shards")

        if r["missing_count"] == 0:
            print(f"  All {r['actual_ng_shards']} NG shards present on GCS")
        else:
            print(f"  *** {r['missing_count']} NG shards MISSING "
                  f"({dvid_affected} DVID source shards affected) ***")
            for entry in r["missing"][:10]:
                dvid_list = ", ".join(entry["dvid_shards"][:5])
                if len(entry["dvid_shards"]) > 5:
                    dvid_list += f" ... (+{len(entry['dvid_shards']) - 5} more)"
                print(f"    {entry['ng_shard']}: {dvid_list}")
            if len(r["missing"]) > 10:
                print(f"    ... and {len(r['missing']) - 10} more missing NG shards")

        if r["orphaned_count"] > 0:
            print(f"  {r['orphaned_count']} orphaned NG shards "
                  f"(no DVID source)")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Verify export completeness against GCS")
    parser.add_argument("--scales", type=str, default=None,
                        help="Comma-separated scale indices (default: all in spec)")
    parser.add_argument("--json-report", type=str, default=None,
                        help="Write JSON report to file")
    parser.add_argument("--z-compress", type=int, default=0, metavar="N",
                        help="Z compression factor used during export. "
                             "Adjusts DVID Z coordinates by 1/(N+1) before "
                             "mapping to NG shard numbers. Default: 0.")
    args = parser.parse_args()

    env = load_env(ENV_FILE)
    if not env:
        print(f"Error: {ENV_FILE} not found. Copy from {ENV_EXAMPLE}.")
        sys.exit(1)

    source_path = env.get("SOURCE_PATH", "").rstrip("/")
    dest_path = env.get("DEST_PATH", "").rstrip("/")
    ng_spec_path = env.get("NG_SPEC_PATH", "")

    if not all([source_path, dest_path, ng_spec_path]):
        print("Error: SOURCE_PATH, DEST_PATH, and NG_SPEC_PATH must be set in .env")
        sys.exit(1)

    scale_info = load_ng_spec(ng_spec_path)
    if args.scales:
        scales = [int(s) for s in args.scales.split(",")]
    else:
        scales = sorted(scale_info.keys())

    print(f"Verifying {len(scales)} scales: {scales}")
    print(f"  Source: {source_path}")
    print(f"  Dest:   {dest_path}")

    total_missing, results = verify_all_scales(
        source_path, dest_path, ng_spec_path, scales,
        z_compress=args.z_compress)

    all_ok = print_report(results)

    if args.json_report:
        with open(args.json_report, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nJSON report written to {args.json_report}")

    if not all_ok:
        total_dvid_affected = sum(
            len(m["dvid_shards"])
            for r in results
            for m in r["missing"]
        )
        print(f"\n*** VERIFICATION FAILED: {total_dvid_affected} DVID shards "
              f"have no NG output ***")
        sys.exit(1)
    else:
        print("\nAll scales verified successfully.")


if __name__ == "__main__":
    main()
