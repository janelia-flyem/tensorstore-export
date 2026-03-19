#!/usr/bin/env python3
"""
Write the neuroglancer precomputed info file to GCS.

This is a one-time setup step before Cloud Run workers start.  It reads the
neuroglancer multiscale volume spec JSON (the same file used for DVID's
export-shards) and writes it as the `info` file at the destination path.

The encoding is changed from whatever DVID used (e.g., compressed_segmentation)
to 'raw', since workers decompress DVID blocks to raw uint64 before writing.

Usage:
    pixi run setup-destination
"""

import json
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deploy import load_env, load_ng_spec, ENV_FILE, ENV_EXAMPLE


def adapt_spec_for_precomputed(ng_spec: dict) -> dict:
    """Adapt DVID ng spec for neuroglancer precomputed output.

    Changes encoding to 'raw' since workers write decompressed uint64 data.
    Removes DVID-specific fields that don't apply to the output volume.
    """
    spec = copy.deepcopy(ng_spec)

    for scale in spec.get("scales", []):
        # Workers write raw decompressed uint64 chunks
        scale["encoding"] = "raw"
        # Remove DVID-specific fields
        scale.pop("compressed_segmentation_block_size", None)

    return spec


def main():
    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)

    ng_spec_path = env.get("NG_SPEC_PATH", "ng-specs.json")
    dest_bucket = env.get("DEST_BUCKET")
    dest_path = env.get("DEST_PATH")

    if not dest_bucket or dest_bucket == "your-dest-bucket":
        print("Error: DEST_BUCKET not configured. Run 'pixi run deploy' or edit .env first.")
        sys.exit(1)
    if not dest_path or dest_path == "path/to/precomputed/output":
        print("Error: DEST_PATH not configured. Run 'pixi run deploy' or edit .env first.")
        sys.exit(1)

    print(f"Reading ng spec from {ng_spec_path}...")
    ng_spec = load_ng_spec(ng_spec_path)

    info = adapt_spec_for_precomputed(ng_spec)
    info_json = json.dumps(info, indent=2)

    gcs_path = f"gs://{dest_bucket}/{dest_path}/info"
    print(f"Writing info file to {gcs_path}...")

    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(dest_bucket)
    blob = bucket.blob(f"{dest_path}/info")

    if blob.exists():
        print(f"  Info file already exists at {gcs_path}")
        overwrite = input("  Overwrite? [y/N]: ").strip().lower()
        if overwrite != "y":
            print("  Skipping.")
            return

    blob.upload_from_string(info_json, content_type="application/json")
    print(f"  Written ({len(info_json)} bytes, {len(info['scales'])} scales)")


if __name__ == "__main__":
    main()
