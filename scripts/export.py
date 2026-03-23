#!/usr/bin/env python3
"""
Export DVID shards to neuroglancer precomputed via Cloud Run.

Single command that scans Arrow source files, assigns shards to
memory-appropriate tiers, writes per-task manifests to GCS, and
launches a Cloud Run job per tier.

Usage:
    pixi run export
    pixi run export --dry-run
    pixi run export --wait
    pixi run export --label-type supervoxels
    pixi run export --downres 10
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE
from scripts.precompute_manifest import (
    TIER_CPU,
    DEFAULT_TIER_MAX_TASKS,
    list_arrow_files,
    assign_tiers,
    distribute_tasks,
)


def _get_image(env: dict) -> str:
    """Get the Docker image URI from .env (set by deploy)."""
    return env.get("DOCKER_IMAGE", "")


def _create_or_update_job(job_name: str, image: str, env: dict,
                          ng_spec_b64: str, memory: str, cpu: int,
                          tasks: int, manifest_uri: str,
                          label_type: str, downres: str) -> bool:
    """Create or update a Cloud Run job for a tier."""
    env_vars = {
        "SOURCE_PATH": env["SOURCE_PATH"],
        "DEST_PATH": env["DEST_PATH"],
        "NG_SPEC": ng_spec_b64,
        "SCALES": env.get("SCALES", "0"),
        "LABEL_TYPE": label_type,
        "WORKER_MEMORY_GIB": str(_parse_memory_gib(memory)),
        "MANIFEST_URI": manifest_uri,
        "MAX_PROCESSING_TIME": env.get("MAX_PROCESSING_TIME", "1440"),
    }
    if downres:
        env_vars["DOWNRES_SCALES"] = downres

    # Write env vars to temp YAML (NG_SPEC contains chars that break
    # gcloud's --set-env-vars comma-separated format).
    env_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="cloudrun-env-"
    )
    for k, v in env_vars.items():
        env_file.write(f"{k}: '{v}'\n")
    env_file.close()

    cmd = [
        "gcloud", "run", "jobs", "create", job_name,
        f"--image={image}",
        f"--region={env['REGION']}",
        f"--project={env['PROJECT_ID']}",
        f"--tasks={tasks}",
        f"--parallelism={tasks}",
        f"--max-retries={env.get('MAX_RETRIES', '3')}",
        f"--task-timeout={env.get('TASK_TIMEOUT', '86400s')}",
        f"--memory={memory}",
        f"--cpu={cpu}",
        f"--env-vars-file={env_file.name}",
        # Gen 2 execution environment: container filesystem is disk-backed
        # (not tmpfs).  Writes to /mnt/staging use disk, not RAM — critical
        # for TensorStore batched RMW that would otherwise OOM.
        "--execution-environment=gen2",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 and "already exists" in result.stderr:
            cmd[3] = "update"  # replace "create" with "update"
            result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    finally:
        os.unlink(env_file.name)


def _execute_job(job_name: str, project: str, region: str,
                 tasks: int, wait: bool) -> bool:
    """Execute a Cloud Run job."""
    cmd = [
        "gcloud", "run", "jobs", "execute", job_name,
        f"--region={region}",
        f"--project={project}",
        f"--tasks={tasks}",
    ]
    cmd.append("--wait" if wait else "--async")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def _parse_memory_gib(memory_str: str) -> float:
    """Parse Cloud Run memory string to GiB."""
    s = memory_str.strip()
    if s.endswith("Gi"):
        return float(s[:-2])
    if s.endswith("Mi"):
        return float(s[:-2]) / 1024
    return float(s)


def main():
    parser = argparse.ArgumentParser(
        description="Export DVID shards to neuroglancer precomputed via Cloud Run.",
    )
    parser.add_argument(
        "--scales",
        help="Comma-separated scales to include (default: from .env SCALES)",
    )
    parser.add_argument(
        "--label-type",
        choices=["labels", "supervoxels"],
        help='Label type: "labels" for agglomerated (default), "supervoxels" for raw IDs',
    )
    parser.add_argument(
        "--downres",
        help="Comma-separated scales to generate by downsampling previous scale",
    )
    parser.add_argument(
        "--tiers",
        help="Override max tasks per tier as GiB:maxTasks pairs. "
             "E.g., 4:3000,8:50.  Default tier limits are used for unspecified tiers.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and show tier assignments without writing manifests or launching jobs",
    )
    parser.add_argument(
        "--wait", action="store_true",
        help="Block until all jobs complete (default: launch async)",
    )
    args = parser.parse_args()

    env = load_env(ENV_FILE) if ENV_FILE.exists() else load_env(ENV_EXAMPLE)

    source_path = env.get("SOURCE_PATH", "")
    project = env.get("PROJECT_ID", "")
    region = env.get("REGION", "us-central1")
    base_name = env.get("BASE_JOB_NAME", "")

    if not source_path or not project or project == "your-gcp-project":
        print("Error: Run 'pixi run deploy' first to configure .env.")
        sys.exit(1)

    scales_str = args.scales or env.get("SCALES", "0")
    scales = [int(s.strip()) for s in scales_str.split(",")]
    label_type = args.label_type or env.get("LABEL_TYPE", "labels")
    downres = args.downres or env.get("DOWNRES_SCALES", "")

    # Parse tier overrides
    max_tasks = dict(DEFAULT_TIER_MAX_TASKS)
    if args.tiers:
        for pair in args.tiers.split(","):
            gib_s, _, tasks_s = pair.partition(":")
            max_tasks[int(gib_s)] = int(tasks_s) if tasks_s else 1000

    # --- Step 1: Scan Arrow files ---
    print(f"Scanning Arrow files across {len(scales)} scales...")
    all_files = list_arrow_files(source_path, scales)
    print(f"  Found {len(all_files)} Arrow files")
    print(f"  Memory formula: 1.5 * arrow_size + 2 GiB (local-disk staging)")

    if not all_files:
        print("No Arrow files found. Check SOURCE_PATH and SCALES in .env.")
        sys.exit(1)

    # --- Step 2: Assign to tiers ---
    tier_map = assign_tiers(all_files, max_tasks)

    print(f"\nTier assignments:")
    for gib in sorted(tier_map.keys()):
        shards = tier_map[gib]
        tier_max = max_tasks.get(gib, 1000)
        num_tasks = min(tier_max, len(shards))
        total_bytes = sum(s for _, _, s in shards)
        max_arrow = max(s for _, _, s in shards)
        cpu = TIER_CPU.get(gib, 2)
        print(f"  {gib}Gi (cpu={cpu}): {len(shards)} shards, {num_tasks} tasks, "
              f"total={total_bytes/1e9:.1f}GB, max_arrow={max_arrow/1e6:.0f}MB")

    if args.dry_run:
        print("\n(dry run — no manifests written, no jobs launched)")
        return

    # --- Step 3: Write per-task manifests to GCS ---
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from google.cloud import storage
    storage_client = storage.Client()
    bucket_name, source_prefix = source_path.replace("gs://", "").split("/", 1)
    gcs_bucket = storage_client.bucket(bucket_name)

    tier_info = {}  # gib -> (manifest_uri, num_tasks)
    total_manifests = 0

    # Prepare all manifest uploads across tiers
    all_uploads = []  # (gib, task_idx, blob_path, json_bytes)
    for gib in sorted(tier_map.keys()):
        shards = tier_map[gib]
        tier_max = max_tasks.get(gib, 1000)
        tasks = distribute_tasks(shards, tier_max)
        num_tasks = len(tasks)

        tier_prefix = f"{source_prefix}/manifests/tier-{gib}gi"
        tier_uri = f"{source_path}/manifests/tier-{gib}gi"
        tier_info[gib] = (tier_uri, num_tasks)

        for task_idx, shard_list in tasks.items():
            blob_path = f"{tier_prefix}/task-{task_idx}.json"
            json_bytes = json.dumps(shard_list, separators=(",", ":"))
            all_uploads.append((gib, blob_path, json_bytes))

    print(f"\nWriting {len(all_uploads)} task manifests to GCS...")

    def _upload_one(blob_path_and_data):
        blob_path, data = blob_path_and_data
        gcs_bucket.blob(blob_path).upload_from_string(
            data, content_type="application/json",
        )

    uploaded = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = {
            pool.submit(_upload_one, (blob_path, data)): gib
            for gib, blob_path, data in all_uploads
        }
        for future in as_completed(futures):
            future.result()  # propagate exceptions
            uploaded += 1
            if uploaded % 500 == 0 or uploaded == len(all_uploads):
                print(f"  {uploaded}/{len(all_uploads)} manifests written")

    for gib in sorted(tier_info.keys()):
        tier_uri, num_tasks = tier_info[gib]
        print(f"  {gib}Gi: {num_tasks} tasks → {tier_uri}/")

    # --- Step 4: Get Docker image ---
    image = _get_image(env)
    if not image:
        print("Error: DOCKER_IMAGE not set in .env. Run 'pixi run deploy' first.")
        sys.exit(1)
    print(f"\nUsing image: {image}")

    # Load NG_SPEC for env vars
    ng_spec_path = env.get("NG_SPEC_PATH", "")
    if ng_spec_path:
        spec_path = Path(ng_spec_path)
        if not spec_path.is_absolute():
            spec_path = Path(__file__).resolve().parent.parent / spec_path
        ng_spec_b64 = base64.b64encode(
            json.dumps(json.loads(spec_path.read_text()), separators=(",", ":")).encode()
        ).decode()
    else:
        ng_spec_b64 = ""

    # --- Step 5: Create/update and execute per-tier jobs ---
    print(f"\nLaunching {len(tier_info)} tier job(s)...")
    for gib in sorted(tier_info.keys()):
        manifest_uri, num_tasks = tier_info[gib]
        cpu = TIER_CPU.get(gib, 2)
        memory = f"{gib}Gi"
        job_name = f"{base_name}-tier-{gib}gi"

        ok = _create_or_update_job(
            job_name, image, env, ng_spec_b64,
            memory, cpu, num_tasks, manifest_uri,
            label_type, downres,
        )
        if not ok:
            print(f"  {job_name}: FAILED to create/update")
            continue

        ok = _execute_job(job_name, project, region, num_tasks, args.wait)
        if ok:
            print(f"  {job_name}: launched ({num_tasks} tasks, {memory}, cpu={cpu})")
        else:
            print(f"  {job_name}: FAILED to execute")

    print(f"\nMonitor progress:")
    print(f"  pixi run export-status")
    print(f"  pixi run export-errors")
    print(f"  pixi run export-errors -- --details | grep 'memory profile'")


if __name__ == "__main__":
    main()
