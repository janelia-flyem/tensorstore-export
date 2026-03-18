#!/usr/bin/env python3
"""
Interactive deployment script for tensorstore-export Cloud Run jobs.

Reads the neuroglancer multiscale volume spec JSON (the same file used for
DVID's export-shards command) as the single source of truth for volume
geometry and sharding parameters. GCP-specific settings come from .env.

Usage:
    pixi run deploy
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

# Fields grouped by section for interactive prompting
SECTIONS = [
    (
        "GCP Settings",
        [
            ("PROJECT_ID", "your-gcp-project"),
            ("REGION", "us-central1"),
            ("JOB_NAME", "tensorstore-dvid-export"),
        ],
    ),
    (
        "Data Settings",
        [
            ("SOURCE_BUCKET", "your-source-bucket"),
            ("DEST_BUCKET", "your-dest-bucket"),
            ("DEST_PATH", "neuroglancer-volume"),
        ],
    ),
    (
        "Neuroglancer Volume Spec",
        [
            ("NG_SPEC_PATH", "mcns-v0.9-ng-specs.json"),
        ],
    ),
    (
        "Cloud Run Settings",
        [
            ("PARALLELISM", "100"),
            ("TASK_COUNT", "100"),
            ("MAX_RETRIES", "3"),
            ("TASK_TIMEOUT", "3600s"),
            ("MEMORY", "4Gi"),
            ("CPU", "2"),
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


def prompt_value(key: str, default: str) -> str:
    """Prompt user for a value, showing the default in brackets."""
    try:
        raw = input(f"  {key} [{default}]: ").strip()
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
    """Print a summary of the neuroglancer spec."""
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
            f"sharding: shard_bits={shard_bits} minishard_bits={minishard_bits} preshift_bits={preshift_bits}"
        )
    print()


def run_cmd(args: list, description: str) -> bool:
    """Run a shell command, printing description and streaming output."""
    print(f"\n{description}...")
    result = subprocess.run(args)
    if result.returncode != 0:
        print(f"  Failed (exit code {result.returncode})")
        return False
    return True


def build_cloud_run_yaml(env: dict, ng_spec_b64: str) -> str:
    """Generate the Cloud Run job YAML spec."""
    return f"""\
apiVersion: run.googleapis.com/v1
kind: Job
metadata:
  name: {env['JOB_NAME']}
  labels:
    cloud.googleapis.com/location: {env['REGION']}
spec:
  template:
    spec:
      parallelism: {env['PARALLELISM']}
      taskCount: {env['TASK_COUNT']}
      template:
        spec:
          maxRetries: {env['MAX_RETRIES']}
          taskTimeoutSeconds: {env['TASK_TIMEOUT'].rstrip('s')}
          containers:
          - image: gcr.io/{env['PROJECT_ID']}/{env['JOB_NAME']}
            env:
            - name: SOURCE_BUCKET
              value: "{env['SOURCE_BUCKET']}"
            - name: DEST_BUCKET
              value: "{env['DEST_BUCKET']}"
            - name: DEST_PATH
              value: "{env['DEST_PATH']}"
            - name: NG_SPEC
              value: "{ng_spec_b64}"
            - name: MAX_PROCESSING_TIME
              value: "55"
            - name: POLLING_INTERVAL
              value: "10"
            resources:
              limits:
                cpu: "{env['CPU']}"
                memory: "{env['MEMORY']}"
"""


def main():
    print("\n=== TensorStore Export — Cloud Run Deployment ===\n")

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

    # Build flat defaults dict from SECTIONS
    all_defaults = {}
    for _, fields in SECTIONS:
        for key, builtin_default in fields:
            all_defaults[key] = env.get(key, builtin_default)

    # Interactive prompting by section
    final = {}
    ng_spec = None

    for section_name, fields in SECTIONS:
        print(f"\n--- {section_name} ---")
        for key, builtin_default in fields:
            default = env.get(key, builtin_default)
            final[key] = prompt_value(key, default)

        # After the NG spec section, load and display the spec
        if section_name == "Neuroglancer Volume Spec":
            spec_path = final["NG_SPEC_PATH"]
            print(f"\n  Reading {spec_path}...")
            ng_spec = load_ng_spec(spec_path)
            display_spec_summary(ng_spec)

    # Offer to save
    try:
        save_answer = input("Save updated values to .env? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        save_answer = "n"

    if save_answer != "n":
        save_env(ENV_FILE, final)
        print(f"  Saved to {ENV_FILE}")

    # Encode the ng spec for the Cloud Run env var
    ng_spec_json = json.dumps(ng_spec, separators=(",", ":"))
    ng_spec_b64 = base64.b64encode(ng_spec_json.encode()).decode()

    # Build and push Docker image
    image = f"gcr.io/{final['PROJECT_ID']}/{final['JOB_NAME']}"

    if not run_cmd(
        ["docker", "build", "-t", image, str(PROJECT_ROOT)],
        f"Building Docker image: {image}",
    ):
        sys.exit(1)

    if not run_cmd(
        ["docker", "push", image],
        f"Pushing image to {image}",
    ):
        sys.exit(1)

    # Create/update Cloud Run job
    yaml_content = build_cloud_run_yaml(final, ng_spec_b64)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        if not run_cmd(
            [
                "gcloud", "run", "jobs", "replace", yaml_path,
                f"--region={final['REGION']}",
                f"--project={final['PROJECT_ID']}",
            ],
            "Creating/updating Cloud Run job",
        ):
            sys.exit(1)
    finally:
        os.unlink(yaml_path)

    print(f"\nDone.")
    print(f"  Run:  gcloud run jobs execute {final['JOB_NAME']} --region={final['REGION']} --project={final['PROJECT_ID']}")
    print(f"  Logs: gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name={final['JOB_NAME']}\" --project={final['PROJECT_ID']} --limit=100")


if __name__ == "__main__":
    main()
