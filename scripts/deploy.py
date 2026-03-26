#!/usr/bin/env python3
"""
Interactive deployment script for tensorstore-export Cloud Run jobs.

Reads the neuroglancer multiscale volume spec JSON (the same file used for
DVID's export-shards command) as the single source of truth for volume
geometry and sharding parameters. GCP-specific settings come from .env.

Usage:
    pixi run deploy
"""

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

# Fields grouped by section for interactive prompting.
# Each field is (key, default, description).
SECTIONS = [
    (
        "GCP Settings",
        [
            ("PROJECT_ID", "your-gcp-project", "GCP project ID"),
            ("REGION", "us-central1", "GCP region"),
        ],
    ),
    (
        "Data Settings",
        [
            ("SOURCE_PATH", "gs://your-bucket/path/to/shard/export", "GCS path to DVID export shards"),
            ("DEST_PATH", "gs://your-bucket/path/to/precomputed/output", "GCS path for neuroglancer output"),
        ],
    ),
    (
        "Neuroglancer Volume Spec",
        [
            ("NG_SPEC_PATH", "ng-specs.json", "path to neuroglancer spec JSON"),
        ],
    ),
    (
        "Deployment",
        [
            ("SCALES", "0,1", "scales to export (comma-separated)"),
            ("BASE_JOB_NAME", "tensorstore-dvid-export", "Cloud Run job name prefix"),
            ("MAX_RETRIES", "3", "max retries per failed worker"),
            ("TASK_TIMEOUT", "86400s", "timeout per worker (max 24h)"),
        ],
    ),
]


def load_env(path: Path) -> dict:
    """Load KEY=VALUE pairs from a file, ignoring comments and blank lines."""
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


def save_env(path: Path, env: dict):
    """Save env dict back to .env, preserving comments from .env.example."""
    lines = []
    if ENV_EXAMPLE.exists():
        for line in ENV_EXAMPLE.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                lines.append(line)
            elif "=" in stripped:
                key = stripped.partition("=")[0].strip()
                if key in env:
                    lines.append(f"{key}={env[key]}")
                else:
                    lines.append(line)
    else:
        for key, value in env.items():
            lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")


def prompt_value(key: str, default: str, description: str = "") -> str:
    """Prompt user for a value, showing description and default."""
    desc = f" ({description})" if description else ""
    try:
        raw = input(f"  {key}{desc} [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    return raw if raw else default


def load_ng_spec(spec_path: str) -> dict:
    """Load and validate a neuroglancer multiscale volume spec JSON."""
    path = Path(spec_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        print(f"\n  Error: spec file not found: {path}")
        sys.exit(1)

    spec = json.loads(path.read_text())

    expected_type = "neuroglancer_multiscale_volume"
    if spec.get("@type") != expected_type:
        print(f"\n  Error: spec @type is {spec.get('@type')!r}, expected {expected_type!r}")
        sys.exit(1)

    if not spec.get("scales"):
        print("\n  Error: spec has no scales defined")
        sys.exit(1)

    return spec


def display_spec_summary(spec: dict):
    """Print a summary of the neuroglancer spec with memory estimates."""
    data_type = spec.get("data_type", "?")
    vol_type = spec.get("type", "?")
    num_scales = len(spec["scales"])
    print(f"\n  Type: {vol_type}, data_type: {data_type}, {num_scales} scale(s)")

    for i, scale in enumerate(spec["scales"]):
        size = scale.get("size", [0, 0, 0])
        res = scale.get("resolution", [0, 0, 0])
        sharding = scale.get("sharding", {})
        shard_bits = sharding.get("shard_bits", "?")
        minishard_bits = sharding.get("minishard_bits", "?")
        preshift_bits = sharding.get("preshift_bits", "?")
        res_str = f"{int(res[0])}nm" if res[0] == res[1] == res[2] else f"{res}"
        print(
            f"  Scale {i}: {size[0]}x{size[1]}x{size[2]} @ {res_str}, "
            f"shard_bits={shard_bits} minishard_bits={minishard_bits} preshift_bits={preshift_bits}"
        )

    # Estimate shard sizes per scale from sharding params
    print("\n  Estimated shard sizes (from sharding params):")
    for i, scale in enumerate(spec["scales"]):
        sharding = scale.get("sharding", {})
        preshift = sharding.get("preshift_bits", 0)
        minishard = sharding.get("minishard_bits", 0)
        # shardSideBits = (18 + preshift + minishard) / 3
        shard_side_bits = (18 + preshift + minishard) // 3
        shard_dim_voxels = 1 << shard_side_bits
        max_chunks = (shard_dim_voxels // 64) ** 3
        # Rough estimate: compressed block ~5-50 KB depending on scale
        est_kb = 5 if i == 0 else min(50, 5 * (2 ** i))
        est_max_mb = max_chunks * est_kb / 1024
        print(f"    Scale {i}: shard={shard_dim_voxels}^3 voxels, up to {max_chunks} chunks, ~{est_max_mb:.0f} MB max")

    print()


def run_cmd(args: list, description: str) -> bool:
    """Run a shell command, printing description and streaming output."""
    print(f"\n{description}...")
    result = subprocess.run(args)
    if result.returncode != 0:
        print(f"  Failed (exit code {result.returncode})")
        return False
    return True


def _parse_gs_uri(uri: str):
    """Split gs://bucket/path into (bucket, path)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    rest = uri[len("gs://"):]
    bucket, _, path = rest.partition("/")
    return bucket, path.rstrip("/")


def validate_and_configure_buckets(env: dict):
    """Validate and configure GCS buckets for export.

    Checks destination (and optionally source) bucket settings:
    - Creates the destination bucket if it doesn't exist (single-region,
      matching REGION, soft delete disabled)
    - Warns if bucket is multi-region or in a mismatched region
    - Auto-disables soft delete if enabled

    Called after the user confirms settings, before writing the info file.
    """
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.bucket_utils import (
        get_bucket_info, create_bucket, disable_soft_delete,
        validate_bucket_region, check_write_permission, check_read_permission,
    )

    region = env.get("REGION", "us-central1")
    dest_uri = env.get("DEST_PATH", "")
    source_uri = env.get("SOURCE_PATH", "")

    if not dest_uri:
        return

    dest_bucket, _ = _parse_gs_uri(dest_uri)
    source_bucket = None
    if source_uri:
        source_bucket, _ = _parse_gs_uri(source_uri)

    print("\n--- Bucket Validation ---")

    # --- Source bucket (check first to inform dest bucket creation) ---
    src_info = None
    src_region = None  # resolved single-region location, if available
    if source_bucket and source_bucket != dest_bucket:
        src_info = get_bucket_info(source_bucket)
        if src_info and "error" not in src_info:
            src_loc = src_info["location"]
            src_type = src_info.get("location_type", "region")
            if src_type.lower() in ("multi-region", "dual-region"):
                print(f"  Source bucket '{source_bucket}': {src_type} "
                      f"({src_loc.lower()}) — read-only, no action needed")
            else:
                src_region = src_loc.upper()
                if src_region != region.upper():
                    print(f"  Warning: source bucket '{source_bucket}' is in "
                          f"{src_loc.lower()}, but REGION is {region}. "
                          f"Cross-region reads will incur egress charges. "
                          f"Consider setting REGION={src_loc.lower()}.")
                else:
                    print(f"  Source bucket '{source_bucket}': "
                          f"{src_loc.lower()} (matches REGION)")

    # --- Destination bucket ---
    dest_info = get_bucket_info(dest_bucket)

    if dest_info is None:
        # Bucket doesn't exist — create it in the same region as source
        # (if available and single-region), otherwise use REGION.
        create_region = region
        if src_region and src_region != region.upper():
            print(f"  Note: creating dest bucket in {src_region.lower()} "
                  f"to match source bucket (overrides REGION={region}).")
            create_region = src_region.lower()
        elif src_region:
            create_region = src_region.lower()

        print(f"  Destination bucket '{dest_bucket}' does not exist.")
        print(f"  Creating single-region bucket in {create_region}...")
        if create_bucket(dest_bucket, create_region):
            print(f"  Created '{dest_bucket}' (single-region {create_region}, "
                  f"HNS enabled, soft delete disabled)")
        else:
            print("  Could not create bucket. Create it manually:")
            print(f"    gcloud storage buckets create gs://{dest_bucket} "
                  f"--location={create_region} --no-soft-delete "
                  f"--enable-hierarchical-namespace")
        return

    if "error" in dest_info:
        print(f"  Warning: could not read metadata for '{dest_bucket}' "
              f"(permission denied). Skipping bucket validation.")
        return

    # Region check — dest vs Cloud Run REGION
    warnings = validate_bucket_region(dest_info, region, dest_bucket)
    if warnings:
        for w in warnings:
            print(f"  Warning: {w}")
    else:
        loc = dest_info['location']
        loc_type = dest_info.get('location_type', 'region')
        print(f"  Dest bucket '{dest_bucket}': {loc_type} {loc.lower()} "
              f"(matches Cloud Run region {region})")

    # Region check — dest vs source
    dest_loc = dest_info["location"].upper()
    dest_type = dest_info.get("location_type", "region").lower()
    if (src_region and dest_type == "region"
            and dest_loc != src_region):
        print(f"  Warning: dest bucket '{dest_bucket}' is in "
              f"{dest_loc.lower()} but source bucket '{source_bucket}' "
              f"is in {src_region.lower()}. Cross-region transfer between "
              f"source and dest will incur egress charges.")

    # Soft delete
    retention = dest_info["soft_delete_retention_seconds"]
    if retention > 0:
        try:
            old = disable_soft_delete(dest_bucket)
            days = old // 86400
            print(f"  \033[1mSoft delete disabled\033[0m on '{dest_bucket}' "
                  f"(was {days}-day retention). "
                  f"Export buckets should not use soft delete.")
        except PermissionError as e:
            print(f"  Warning: {e}")
    else:
        print("  Soft delete: disabled")

    # --- Permission checks ---
    _, dest_prefix = _parse_gs_uri(dest_uri)
    if not check_write_permission(dest_bucket, dest_prefix):
        print("\n  \033[1mDeploy aborted\033[0m: cannot write to destination bucket.")
        print("  Fix permissions before running export — otherwise Cloud Run tasks")
        print("  will burn CPU/memory for hours before failing at upload.")
        sys.exit(1)
    else:
        print("  Dest write permission: ok")

    if source_bucket and source_bucket != dest_bucket:
        _, src_prefix = _parse_gs_uri(source_uri)
        if not check_read_permission(source_bucket, src_prefix):
            print("\n  \033[1mWarning\033[0m: cannot read from source bucket.")
            print("  Cloud Run tasks will fail to download Arrow shard files.")
        else:
            print("  Source read permission: ok")


def setup_destination_info(dest_uri: str, ng_spec: dict):
    """Write the neuroglancer info file to GCS if it doesn't already exist."""
    import copy
    from google.cloud import storage

    bucket_name, prefix = _parse_gs_uri(dest_uri)

    info = copy.deepcopy(ng_spec)
    for scale in info.get("scales", []):
        scale["encoding"] = "compressed_segmentation"
        scale.setdefault("compressed_segmentation_block_size", [8, 8, 8])

    info_json = json.dumps(info, indent=2)
    info_uri = f"{dest_uri}/info"

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(f"{prefix}/info")

    if blob.exists():
        print(f"\n  Info file already exists at {info_uri}")
    else:
        print(f"\nWriting neuroglancer info file to {info_uri}...")
        blob.upload_from_string(info_json, content_type="application/json")
        print(f"  Written ({len(info_json)} bytes, {len(info['scales'])} scales)")



# Placeholder values from .env.example that should be treated as "not configured"
PLACEHOLDERS = {"your-gcp-project", "gs://your-bucket/path/to/shard/export",
                "gs://your-bucket/path/to/precomputed/output", "ng-specs.json"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deploy tensorstore-export to Cloud Run.")
    parser.add_argument(
        "--use-env", action="store_true",
        help="Use all values from .env without prompting (only prompt for missing values)",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip Docker image build (reuse existing image in GCR)",
    )
    args = parser.parse_args()

    print("\n=== TensorStore Export — Cloud Run Deployment ===\n")

    # Check for GCP credentials early, before interactive prompting
    try:
        import google.auth
        google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        print("Error: GCP credentials not found.")
        print("Run 'gcloud auth application-default login' first.")
        sys.exit(1)

    # Load existing .env as defaults
    env = {}
    if ENV_FILE.exists():
        env = load_env(ENV_FILE)
        print(f"Loading defaults from {ENV_FILE.name}...")
    elif ENV_EXAMPLE.exists():
        env = load_env(ENV_EXAMPLE)
        print(f"No .env found, using defaults from {ENV_EXAMPLE.name}...")
    else:
        print("No .env or .env.example found, using built-in defaults...")

    # Interactive prompting by section
    final = {}
    ng_spec = None

    for section_name, fields in SECTIONS:
        print(f"\n--- {section_name} ---")

        # Default SCALES to all scales defined in the ng spec.
        if section_name == "Deployment" and ng_spec:
            all_scales = ",".join(str(i) for i in range(len(ng_spec["scales"])))
            fields = [(k, all_scales if k == "SCALES" else d, desc)
                      for k, d, desc in fields]

        for key, builtin_default, description in fields:
            default = env.get(key, builtin_default)
            if args.use_env and key in env and env[key] not in PLACEHOLDERS:
                # --use-env: accept .env value silently
                final[key] = env[key]
                print(f"  {key} ({description}): {env[key]}")
            else:
                final[key] = prompt_value(key, default, description)

        # After the NG spec section, load and display the spec
        if section_name == "Neuroglancer Volume Spec":
            spec_path = final["NG_SPEC_PATH"]
            print(f"\n  Reading {spec_path}...")
            ng_spec = load_ng_spec(spec_path)
            display_spec_summary(ng_spec)

    # Offer to save — skip if --use-env and nothing changed
    changed = any(final.get(k) != env.get(k) for k in final)
    if changed:
        try:
            save_answer = input("Save updated values to .env? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            save_answer = "n"

        if save_answer != "n":
            save_env(ENV_FILE, final)
            print(f"  Saved to {ENV_FILE}")

    # Validate and configure GCS buckets (creates dest bucket if needed,
    # disables soft delete, warns about region mismatches).
    validate_and_configure_buckets(final)

    # Setup destination info file
    setup_destination_info(final["DEST_PATH"], ng_spec)

    # Build and push Docker image.
    # Tag with a hash of the files that go into the Docker image so we
    # skip rebuilding when only deploy scripts or docs change.
    import hashlib
    content_hash_input = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--",
         "Dockerfile", "requirements.txt", "main.py", "src/", "scripts/", "braid/src/", "braid/csrc/", "braid/pyproject.toml"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    ).stdout.strip()
    content_tag = hashlib.sha256(content_hash_input.encode()).hexdigest()[:12] if content_hash_input else "latest"

    image_base = f"gcr.io/{final['PROJECT_ID']}/{final['BASE_JOB_NAME']}"
    image_tagged = f"{image_base}:{content_tag}"
    image = image_tagged

    # Check if an image with this content hash already exists in GCR
    image_exists = subprocess.run(
        ["gcloud", "container", "images", "describe", image_tagged,
         f"--project={final['PROJECT_ID']}"],
        capture_output=True,
    ).returncode == 0

    if args.skip_build or image_exists:
        if image_exists:
            print(f"\n  Image already exists ({content_tag}) — skipping build.")
        else:
            print(f"\n  Skipping build (--skip-build) — reusing image: {image_base}:latest")
            image = f"{image_base}:latest"
    else:
        # Use Cloud Build if Docker is not available locally
        try:
            docker_ok = subprocess.run(
                ["docker", "version"], capture_output=True, timeout=5
            ).returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            docker_ok = False

        if docker_ok:
            if not run_cmd(
                ["docker", "build", "-t", image_tagged, "-t", f"{image_base}:latest",
                 str(PROJECT_ROOT)],
                f"Building Docker image locally: {image_tagged}",
            ):
                sys.exit(1)
            if not run_cmd(
                ["docker", "push", "--all-tags", image_base],
                f"Pushing image to {image_base}",
            ):
                sys.exit(1)
        else:
            print("\n  Docker not found locally — using Google Cloud Build instead.")
            if not run_cmd(
                [
                    "gcloud", "builds", "submit",
                    f"--tag={image_tagged}",
                    f"--project={final['PROJECT_ID']}",
                    str(PROJECT_ROOT),
                ],
                f"Building and pushing image via Cloud Build: {image_tagged}",
            ):
                sys.exit(1)

    # Save image URI to .env so export.py can find it.
    final["DOCKER_IMAGE"] = image
    save_env(ENV_FILE, final)

    print(f"\nDone. Image: {image}")
    print("\nNext steps:")
    print("  pixi run export            # scan shards, create tier jobs, launch")
    print("  pixi run export --dry-run  # preview tier assignments without launching")


if __name__ == "__main__":
    main()
