#!/usr/bin/env python3
"""
Create a single-shard debug manifest for targeted Cloud Run testing.

Uploads a manifest with one shard to GCS under manifests-debug/tier-4gi/,
then use: pixi run export --manifest-dir manifests-debug --job-suffix debug --wait

Usage:
    pixi run create-debug-manifest
    pixi run create-debug-manifest -- --shard 67584_47104_20480 --scale 0
    pixi run create-debug-manifest -- --shard 14336_12288_45056 --scale 0 --tier 4
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE


def main():
    parser = argparse.ArgumentParser(
        description="Create a single-shard debug manifest on GCS.")
    parser.add_argument("--shard", default="67584_47104_20480",
                        help="DVID shard name (default: 67584_47104_20480)")
    parser.add_argument("--scale", type=int, default=0,
                        help="Scale index (default: 0)")
    parser.add_argument("--tier", type=int, default=4,
                        help="Memory tier in GiB (default: 4)")
    args = parser.parse_args()

    env = load_env(ENV_FILE)
    source_path = env.get("SOURCE_PATH", "").rstrip("/")
    if not source_path:
        print("Error: SOURCE_PATH not set in .env")
        sys.exit(1)

    from google.cloud import storage
    project = env.get("PROJECT_ID", "")
    client = storage.Client(project=project or None)

    bucket_name, source_prefix = source_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)

    manifest = [{"scale": args.scale, "shard": args.shard}]
    manifest_json = json.dumps(manifest, separators=(",", ":"))

    blob_path = f"{source_prefix}/manifests-debug/tier-{args.tier}gi/task-0.json"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(manifest_json, content_type="application/json")

    print(f"Uploaded debug manifest:")
    print(f"  gs://{bucket_name}/{blob_path}")
    print(f"  Content: {manifest_json}")
    print()
    print(f"Run the debug export:")
    print(f"  pixi run export --manifest-dir manifests-debug --job-suffix debug --wait")


if __name__ == "__main__":
    main()
