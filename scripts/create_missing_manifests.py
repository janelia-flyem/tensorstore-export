#!/usr/bin/env python3
"""
Identify missing NG output shards and create cleanup manifests.

Compares DVID source shards against the destination NG shard files on GCS,
then writes tiered manifests for the source shards whose expected NG output
is missing. The manifests can be launched with:

    pixi run export -- --manifest-dir manifests-missing --job-suffix missing

Usage:
    pixi run create-missing-manifests
    pixi run create-missing-manifests -- --scales 0,1
    pixi run create-missing-manifests -- --manifest-subdir manifests-missing-s0
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE
from scripts.precompute_manifest import (
    DEFAULT_TIER_MAX_TASKS,
    TIER_CPU,
    assign_tiers,
    distribute_tasks,
    list_arrow_files,
)
from scripts.verify_export import print_report, verify_all_scales


def main():
    parser = argparse.ArgumentParser(
        description="Create cleanup manifests for DVID shards missing NG output.",
    )
    parser.add_argument(
        "--scales",
        help="Comma-separated scales to verify (default: from .env SCALES)",
    )
    parser.add_argument(
        "--tiers",
        help="Override max tasks per tier as GiB:maxTasks pairs. "
             "E.g., 4:3000,8:50.",
    )
    parser.add_argument(
        "--manifest-subdir",
        default="manifests-missing",
        help="GCS subdirectory for cleanup manifests "
             "(default: manifests-missing)",
    )
    parser.add_argument(
        "--json-report",
        help="Optional path to write the verify-export style JSON report.",
    )
    args = parser.parse_args()

    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)
    source_path = env.get("SOURCE_PATH", "").rstrip("/")
    dest_path = env.get("DEST_PATH", "").rstrip("/")
    ng_spec_path = env.get("NG_SPEC_PATH", "")

    if not all([source_path, dest_path, ng_spec_path]):
        print("Error: SOURCE_PATH, DEST_PATH, and NG_SPEC_PATH must be set in .env")
        sys.exit(1)

    scales_str = args.scales or env.get("SCALES", "0")
    scales = [int(s.strip()) for s in scales_str.split(",") if s.strip()]

    max_tasks = dict(DEFAULT_TIER_MAX_TASKS)
    if args.tiers:
        for pair in args.tiers.split(","):
            gib_s, _, tasks_s = pair.partition(":")
            max_tasks[int(gib_s)] = int(tasks_s) if tasks_s else 1000

    print(f"Verifying destination completeness for scales {scales}...")
    total_missing, results = verify_all_scales(
        source_path, dest_path, ng_spec_path, scales)
    print_report(results)

    if args.json_report:
        with open(args.json_report, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nJSON report written to {args.json_report}")

    if total_missing == 0:
        print("\nNo missing NG output detected. No manifests written.")
        return

    missing_by_scale = defaultdict(set)
    for scale_report in results:
        scale = scale_report["scale"]
        for entry in scale_report["missing"]:
            for dvid_shard in entry["dvid_shards"]:
                missing_by_scale[scale].add(dvid_shard)

    print(f"\nMissing source shards to re-export: {total_missing}")
    for scale in sorted(missing_by_scale):
        print(f"  s{scale}: {len(missing_by_scale[scale])} shards")

    print("\nScanning source shard metadata for tier assignment...")
    all_files = list_arrow_files(source_path, scales)
    source_entries = {(scale, name): (scale, name, size, chunk_count)
                      for scale, name, size, chunk_count in all_files}

    missing_files = []
    for scale in sorted(missing_by_scale):
        for name in sorted(missing_by_scale[scale]):
            entry = source_entries.get((scale, name))
            if entry is None:
                print(f"  Warning: source shard metadata not found for s{scale}/{name}")
                continue
            missing_files.append(entry)

    if not missing_files:
        print("No source metadata found for missing shards. No manifests written.")
        sys.exit(1)

    tier_map = assign_tiers(missing_files, max_tasks)

    from google.cloud import storage
    bucket_name, source_prefix = source_path.replace("gs://", "").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    subdir = args.manifest_subdir.strip("/")
    manifest_prefix = f"{source_prefix}/{subdir}"
    old_blobs = list(bucket.list_blobs(prefix=f"{manifest_prefix}/"))
    if old_blobs:
        print(f"\nDeleting {len(old_blobs)} stale manifests from {subdir}/ ...")
        for blob in old_blobs:
            blob.delete()

    print("\nWriting cleanup manifests:")
    for gib in sorted(tier_map):
        shards = tier_map[gib]
        tier_max = max_tasks.get(gib, 1000)
        tasks = distribute_tasks(shards, tier_max)
        tier_prefix = f"{manifest_prefix}/tier-{gib}gi"
        tier_uri = f"{source_path}/{subdir}/tier-{gib}gi"
        cpu = TIER_CPU.get(gib, 2)

        for task_idx, shard_list in tasks.items():
            blob = bucket.blob(f"{tier_prefix}/task-{task_idx}.json")
            blob.upload_from_string(
                json.dumps(shard_list, separators=(",", ":")),
                content_type="application/json",
            )

        total_bytes = sum(size for _, _, size in shards)
        print(f"  {gib}Gi (cpu={cpu}): {len(shards)} shards, {len(tasks)} tasks, "
              f"total={total_bytes / 1e9:.1f}GB")
        print(f"    {tier_uri}/task-*.json")

    print("\nTo launch cleanup export:")
    print(f"  pixi run export -- --manifest-dir {subdir} --job-suffix missing")


if __name__ == "__main__":
    main()
