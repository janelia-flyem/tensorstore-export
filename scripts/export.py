#!/usr/bin/env python3
"""
Export DVID shards to neuroglancer precomputed via Cloud Run.

Single command that scans Arrow source files, assigns shards to
memory-appropriate tiers, writes per-task manifests to GCS, and
launches a Cloud Run job per tier.

Usage:
    pixi run export
    pixi run export --dry-run
    pixi run export --async
    pixi run export --label-type supervoxels
    pixi run export --downres 10
    pixi run export --downres 2 --only-missing
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deploy import load_env, ENV_FILE, ENV_EXAMPLE
from scripts.precompute_manifest import (
    TIER_CPU,
    DEFAULT_TIER_MAX_TASKS,
    list_arrow_files,
    assign_tiers,
    distribute_tasks,
    generate_downres_manifests,
)
from scripts.aggregate_predicted_labels import aggregate_labels


SHARDS_PER_CHECK_TASK = 100  # shards per Cloud Run task for zero-check
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_RESET = "\033[0m"


def _format_elapsed(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _print_downres_success_summary(scale_timings: list[tuple[int, float]],
                                   total_seconds: float) -> None:
    print(f"\n{ANSI_GREEN}Downres pipeline completed successfully.{ANSI_RESET}")
    print("Scales:")
    for scale, elapsed in scale_timings:
        print(f"  s{scale}: {_format_elapsed(elapsed)}")
    print(f"Total: {_format_elapsed(total_seconds)}")


def _fail_downres(scale: int | None, message: str,
                  scale_timings: list[tuple[int, float]],
                  total_start: float) -> None:
    print(f"\n{ANSI_RED}Downres pipeline failed.{ANSI_RESET}")
    if scale is not None:
        print(f"Failed at: s{scale}")
    print(message)
    if scale_timings:
        print("Completed scales before failure:")
        for done_scale, elapsed in scale_timings:
            print(f"  s{done_scale}: {_format_elapsed(elapsed)}")
    print(f"Elapsed before failure: {_format_elapsed(time.monotonic() - total_start)}")
    sys.exit(1)


def _remove_zero_shards(all_files: list, env: dict, source_path: str) -> list:
    """Filter out empty shards by running a Cloud Run job to check Arrow metadata.

    Uploads a manifest of all shards, launches a Cloud Run job that checks
    each shard's labels/supervoxels columns, collects the empty list from
    Cloud Logging, and returns the filtered file list.
    """
    import math
    from google.cloud import storage

    image = _get_image(env)
    if not image:
        print("  Warning: DOCKER_IMAGE not set, skipping --remove-zeros")
        return all_files

    project = env.get("PROJECT_ID", "")
    region = env.get("REGION", "us-central1")
    bucket_name, source_prefix = source_path.replace("gs://", "").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Build flat shard list and upload per-task manifests
    shard_list = [{"scale": s, "shard": n} for s, n, _, _ in all_files]
    num_tasks = math.ceil(len(shard_list) / SHARDS_PER_CHECK_TASK)
    manifest_prefix = f"{source_prefix}/manifests-check-empty"

    print(f"\n  Uploading {num_tasks} zero-check manifests ({len(shard_list)} shards)...")
    for i in range(num_tasks):
        chunk = shard_list[i * SHARDS_PER_CHECK_TASK:(i + 1) * SHARDS_PER_CHECK_TASK]
        blob = bucket.blob(f"{manifest_prefix}/task-{i}.json")
        blob.upload_from_string(
            json.dumps(chunk, separators=(",", ":")),
            content_type="application/json",
        )

    manifest_uri = f"{source_path}/manifests-check-empty"
    job_name = "check-empty-shards"

    # Delete existing job if any
    subprocess.run(
        ["gcloud", "run", "jobs", "delete", job_name,
         "--project", project, "--region", region, "--quiet"],
        capture_output=True,
    )

    # Create and execute
    print(f"  Launching Cloud Run job: {job_name} ({num_tasks} tasks)...")
    create_cmd = [
        "gcloud", "run", "jobs", "create", job_name,
        "--project", project, "--region", region,
        "--image", image,
        "--tasks", str(num_tasks),
        "--task-timeout", "600s",
        "--max-retries", "1",
        "--memory", "2Gi", "--cpu", "1",
        "--set-env-vars",
        f"SOURCE_PATH={source_path},MANIFEST_URI={manifest_uri}",
        "--command", "python",
        "--args", "scripts/check_empty_shards.py,--worker",
    ]
    result = subprocess.run(create_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Warning: failed to create zero-check job: {result.stderr[:200]}")
        print("  Continuing without zero filtering.")
        return all_files

    exec_result = subprocess.run(
        ["gcloud", "run", "jobs", "execute", job_name,
         "--project", project, "--region", region, "--wait"],
        capture_output=True, text=True,
    )
    if exec_result.returncode != 0:
        print(f"  Warning: zero-check job failed: {exec_result.stderr[:200]}")
        print("  Continuing without zero filtering.")
        return all_files

    # Collect empty shards from Cloud Logging
    print("  Collecting results from Cloud Logging...")
    log_result = subprocess.run(
        ["gcloud", "logging", "read",
         'resource.type="cloud_run_job" AND '
         'resource.labels.job_name="check-empty-shards" AND '
         'jsonPayload.event="Shard is empty"',
         "--project", project,
         "--limit", "50000",
         "--format", "json"],
        capture_output=True, text=True,
    )
    if log_result.returncode != 0:
        print(f"  Warning: failed to read logs: {log_result.stderr[:200]}")
        return all_files

    entries = json.loads(log_result.stdout)
    empty_set = set()
    for entry in entries:
        jp = entry.get("jsonPayload", {})
        if jp.get("shard") and jp.get("scale") is not None:
            empty_set.add((jp["scale"], jp["shard"]))

    before = len(all_files)
    all_files = [f for f in all_files if (f[0], f[1]) not in empty_set]
    excluded = before - len(all_files)
    print(f"  Excluded {excluded} empty shards ({len(all_files)} remaining)")

    return all_files


def _get_image(env: dict) -> str:
    """Get the Docker image URI from .env (set by deploy)."""
    return env.get("DOCKER_IMAGE", "")


def _create_or_update_job(job_name: str, image: str, env: dict,
                          ng_spec_b64: str, memory: str, cpu: int,
                          tasks: int, manifest_uri: str,
                          label_type: str, downres: str,
                          downres_mode: bool = False) -> bool:
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
    if downres_mode:
        env_vars["DOWNRES_MODE"] = "1"

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
        # Cap parallelism to stay within project memory quota (~40 TB).
        f"--parallelism={min(tasks, int(40000 / _parse_memory_gib(memory)))}",
        f"--max-retries={env.get('MAX_RETRIES', '3')}",
        f"--task-timeout={env.get('TASK_TIMEOUT', '86400s')}",
        f"--memory={memory}",
        f"--cpu={cpu}",
        f"--env-vars-file={env_file.name}",
        # Gen 2 execution environment: required for full Linux compatibility
        # (cgroups, namespaces).  NOTE: the container filesystem is tmpfs
        # (in-memory), NOT disk-backed — writes to /mnt/staging consume
        # the container's memory budget.  The memory formula accounts for
        # the output shard file living on tmpfs.
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


def _launch_tier_jobs(tier_info, env, image, ng_spec_b64, base_name,
                      project, region, label_type, downres, wait,
                      job_suffix=""):
    """Create/update and execute Cloud Run jobs for each tier.

    Args:
        tier_info: dict mapping gib -> (manifest_uri, num_tasks)
        job_suffix: optional suffix inserted before '-tier-', e.g. 'retry'
                    produces job names like {base_name}-retry-tier-{N}gi.
    """
    suffix_part = f"-{job_suffix}" if job_suffix else ""
    print(f"\nLaunching {len(tier_info)} tier job(s)...")
    for gib in sorted(tier_info.keys()):
        manifest_uri, num_tasks = tier_info[gib]
        cpu = TIER_CPU.get(gib, 2)
        memory = f"{gib}Gi"
        job_name = f"{base_name}{suffix_part}-tier-{gib}gi"

        ok = _create_or_update_job(
            job_name, image, env, ng_spec_b64,
            memory, cpu, num_tasks, manifest_uri,
            label_type, downres,
        )
        if not ok:
            print(f"  {job_name}: FAILED to create/update")
            continue

        ok = _execute_job(job_name, project, region, num_tasks, wait)
        if ok:
            print(f"  {job_name}: launched ({num_tasks} tasks, {memory}, cpu={cpu})")
        else:
            print(f"  {job_name}: FAILED to execute")

    # Post-export verification when using --wait
    if wait:
        source_path = env.get("SOURCE_PATH", "").rstrip("/")
        dest_path = env.get("DEST_PATH", "").rstrip("/")
        ng_spec_path = env.get("NG_SPEC_PATH", "")
        scales_str = env.get("SCALES", "")

        if all([source_path, dest_path, ng_spec_path, scales_str]):
            print("\n--- Post-export verification ---")
            try:
                from scripts.verify_export import verify_all_scales, print_report
                scales = [int(s) for s in scales_str.split(",")]
                total_missing, results = verify_all_scales(
                    source_path, dest_path, ng_spec_path, scales)
                print_report(results)
                if total_missing > 0:
                    print(f"\n*** WARNING: {total_missing} DVID shards have "
                          f"no NG output ***")
                    print("Run 'pixi run verify-export' for details.")
            except Exception as e:
                print(f"\n  Verification failed: {e}")
                print("  Run 'pixi run verify-export' manually.")

    print("\nMonitor progress:")
    print("  pixi run export-status")
    print("  pixi run export-errors")


def _launch_from_manifests(args, env, source_path, project, region,
                           base_name, label_type, downres, ng_spec_b64):
    """Launch jobs from pre-built manifests (retry flow).

    Reads manifest files from {SOURCE_PATH}/{manifest_dir}/tier-{N}gi/
    to detect tiers and task counts, then creates and executes jobs.
    """
    from google.cloud import storage

    manifest_dir = args.manifest_dir.strip("/")
    bucket_name, source_prefix = source_path.replace("gs://", "").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    manifest_prefix = f"{source_prefix}/{manifest_dir}/"
    blobs = list(bucket.list_blobs(prefix=manifest_prefix))

    # Detect tiers and count tasks from manifest file names
    tier_info = {}  # gib -> (manifest_uri, num_tasks)
    for blob in blobs:
        if not blob.name.endswith(".json"):
            continue
        # Parse: .../manifests-retry/tier-16gi/task-42.json
        parts = blob.name.split("/")
        for p in parts:
            if p.startswith("tier-") and p.endswith("gi"):
                gib = int(p.replace("tier-", "").replace("gi", ""))
                uri = f"{source_path}/{manifest_dir}/tier-{gib}gi"
                if gib not in tier_info:
                    tier_info[gib] = [uri, 0]
                tier_info[gib][1] += 1
                break

    if not tier_info:
        print(f"No manifest files found under {source_path}/{manifest_dir}/")
        sys.exit(1)

    tier_info = {gib: (uri, count) for gib, (uri, count) in tier_info.items()}

    # Derive job name suffix from manifest dir
    suffix = args.job_suffix
    if not suffix:
        suffix = manifest_dir.replace("manifests-", "").replace("manifests", "retry")
        if not suffix:
            suffix = "retry"

    print(f"Using pre-built manifests from {manifest_dir}/:")
    for gib in sorted(tier_info.keys()):
        uri, num_tasks = tier_info[gib]
        cpu = TIER_CPU.get(gib, 2)
        print(f"  {gib}Gi (cpu={cpu}): {num_tasks} tasks")

    if args.dry_run:
        print("\n(dry run — no jobs launched)")
        return

    # Get Docker image
    image = _get_image(env)
    if not image:
        print("Error: DOCKER_IMAGE not set in .env. Run 'pixi run deploy' first.")
        sys.exit(1)
    print(f"\nUsing image: {image}")

    _launch_tier_jobs(
        tier_info, env, image, ng_spec_b64, base_name, project, region,
        label_type, downres, args.wait, job_suffix=suffix,
    )


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
        "--only-missing", action="store_true",
        help="For downres exports, generate manifests only for output shards "
             "missing from DEST_PATH at the target scale.",
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
        help="For non-downres exports, block until all jobs complete. "
             "Downres runs sequentially by default.",
    )
    parser.add_argument(
        "--async", dest="async_launch", action="store_true",
        help="For downres exports, launch jobs without sequential orchestration "
             "or per-scale verification.",
    )
    parser.add_argument(
        "--manifest-dir",
        help="Use pre-built manifests from this GCS subdirectory instead of "
             "scanning and partitioning shards. E.g., 'manifests-retry'. "
             "Created by 'pixi run find-failed -- --retry-tier <N>'.",
    )
    parser.add_argument(
        "--job-suffix",
        help="Suffix for Cloud Run job names (default: derived from --manifest-dir). "
             "E.g., 'retry' produces jobs named {BASE_JOB_NAME}-retry-tier-{N}gi.",
    )
    parser.add_argument(
        "--remove-zeros", action="store_true",
        help="Pre-filter empty shards (all-zero labels/supervoxels) via a "
             "Cloud Run job before building manifests. Adds ~2 minutes but "
             "avoids assigning empty shards to export tasks.",
    )
    parser.add_argument(
        "--downres-mode", action="store_true",
        help="Generate downres manifests and launch Cloud Run jobs with "
             "DOWNRES_MODE=1. Deprecated: --downres now implies this path. "
             "Uses the manifest chain approach: s0 shards -> s1 -> s2 -> ...",
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

    # --- Load NG_SPEC (needed by both paths) ---
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

    downres_mode = args.downres_mode or bool(downres)
    downres_wait = downres_mode and not args.async_launch

    if args.only_missing and not downres_mode:
        print("Error: --only-missing is only supported with --downres.")
        sys.exit(1)

    # --- Downres mode: generate manifests + launch with DOWNRES_MODE=1 ---
    if downres_mode:
        if not downres:
            print("Error: --downres must specify target scales.")
            sys.exit(1)
        if not ng_spec_path:
            print("Error: NG_SPEC_PATH must be configured in .env for downres.")
            sys.exit(1)

        downres_scales = [int(s.strip()) for s in downres.split(",")]
        spec_path_resolved = Path(ng_spec_path)
        if not spec_path_resolved.is_absolute():
            spec_path_resolved = Path(__file__).resolve().parent.parent / spec_path_resolved

        if downres_wait:
            image = _get_image(env)
            if not image and not args.dry_run:
                print("Error: DOCKER_IMAGE not set in .env. Run 'pixi run deploy' first.")
                sys.exit(1)
            if image:
                print(f"\nUsing image: {image}")

            from scripts.verify_export import verify_all_scales, print_report
            pipeline_start = time.monotonic()
            scale_timings = []

            for target_scale in downres_scales:
                scale_start = time.monotonic()
                if target_scale > 1:
                    print(f"\nAggregating label predictions for s{target_scale}...")
                    try:
                        aggregate_labels(
                            source_path, target_scale, str(spec_path_resolved))
                    except Exception as e:
                        _fail_downres(
                            target_scale,
                            f"Failed to aggregate labels for s{target_scale}: {e}",
                            scale_timings,
                            pipeline_start,
                        )

                print(f"\nGenerating downres manifests for scales [{target_scale}]...")
                t0 = time.monotonic()
                scale_results = generate_downres_manifests(
                    str(spec_path_resolved), source_path, env.get("DEST_PATH", ""),
                    scales, [target_scale],
                    max_tasks, only_missing=args.only_missing, dry_run=args.dry_run,
                )
                print(f"\nManifest generation took {time.monotonic() - t0:.1f}s")

                tier_info = scale_results.get(target_scale, {})
                if not tier_info:
                    print(f"\nNo manifests generated for s{target_scale}. "
                          "Checking whether the scale is already complete...")
                    total_missing, results = verify_all_scales(
                        source_path, env.get("DEST_PATH", ""),
                        str(spec_path_resolved), [target_scale])
                    print_report(results)
                    if total_missing > 0:
                        _fail_downres(
                            target_scale,
                            f"s{target_scale} still has {total_missing} missing NG "
                            "shard(s) but no manifests were generated.",
                            scale_timings,
                            pipeline_start,
                        )
                    scale_timings.append(
                        (target_scale, time.monotonic() - scale_start))
                    continue

                if args.dry_run:
                    continue

                print(f"\nLaunching downres jobs for scale {target_scale}...")
                launch_failed = False
                for gib in sorted(tier_info.keys()):
                    manifest_uri, num_tasks = tier_info[gib]
                    cpu = TIER_CPU.get(gib, 2)
                    memory = f"{gib}Gi"
                    job_name = f"{base_name}-downres-s{target_scale}-tier-{gib}gi"

                    ok = _create_or_update_job(
                        job_name, image, env, ng_spec_b64,
                        memory, cpu, num_tasks, manifest_uri,
                        label_type, "", downres_mode=True,
                    )
                    if not ok:
                        print(f"  {job_name}: FAILED to create/update")
                        launch_failed = True
                        continue

                    ok = _execute_job(job_name, project, region, num_tasks, True)
                    if ok:
                        print(f"  {job_name}: launched ({num_tasks} tasks, {memory}, cpu={cpu})")
                    else:
                        print(f"  {job_name}: FAILED to execute")
                        launch_failed = True

                if launch_failed:
                    _fail_downres(
                        target_scale,
                        f"Downres launch or execution failed for s{target_scale}.",
                        scale_timings,
                        pipeline_start,
                    )

                print(f"\nVerifying s{target_scale} export completeness...")
                total_missing, results = verify_all_scales(
                    source_path, env.get("DEST_PATH", ""),
                    str(spec_path_resolved), [target_scale])
                print_report(results)
                if total_missing > 0:
                    _fail_downres(
                        target_scale,
                        f"s{target_scale} finished with {total_missing} missing NG "
                        "shard(s). Stopping before the next scale.",
                        scale_timings,
                        pipeline_start,
                    )
                scale_timings.append((target_scale, time.monotonic() - scale_start))

            if args.dry_run:
                print("\n(dry run — no manifests written, no jobs launched)")
            else:
                _print_downres_success_summary(
                    scale_timings, time.monotonic() - pipeline_start)
            return

        print(f"Generating downres manifests for scales {downres_scales}...")
        t0 = time.monotonic()
        all_scale_results = generate_downres_manifests(
            str(spec_path_resolved), source_path, env.get("DEST_PATH", ""),
            scales, downres_scales,
            max_tasks, only_missing=args.only_missing, dry_run=args.dry_run,
        )
        elapsed = time.monotonic() - t0
        print(f"\nManifest generation took {elapsed:.1f}s")

        if args.dry_run:
            print("\n(dry run — no manifests written, no jobs launched)")
            return

        # Get Docker image
        image = _get_image(env)
        if not image:
            print("Error: DOCKER_IMAGE not set in .env. Run 'pixi run deploy' first.")
            sys.exit(1)
        print(f"\nUsing image: {image}")

        for target_scale in sorted(all_scale_results.keys()):
            tier_info = all_scale_results[target_scale]
            if not tier_info:
                continue

            print(f"\nLaunching downres jobs for scale {target_scale}...")
            for gib in sorted(tier_info.keys()):
                manifest_uri, num_tasks = tier_info[gib]
                cpu = TIER_CPU.get(gib, 2)
                memory = f"{gib}Gi"
                job_name = f"{base_name}-downres-s{target_scale}-tier-{gib}gi"

                ok = _create_or_update_job(
                    job_name, image, env, ng_spec_b64,
                    memory, cpu, num_tasks, manifest_uri,
                    label_type, "", downres_mode=True,
                )
                if not ok:
                    print(f"  {job_name}: FAILED to create/update")
                    continue

                ok = _execute_job(job_name, project, region, num_tasks, False)
                if ok:
                    print(f"  {job_name}: launched ({num_tasks} tasks, {memory}, cpu={cpu})")
                else:
                    print(f"  {job_name}: FAILED to execute")

        print("\nMonitor progress:")
        print("  pixi run export-status")
        print("  pixi run export-errors")
        return

    # --- Pre-built manifest path (retry flow) ---
    if args.manifest_dir:
        _launch_from_manifests(
            args, env, source_path, project, region, base_name,
            label_type, downres, ng_spec_b64,
        )
        return

    # --- Step 1: Scan Arrow files ---
    t0 = time.monotonic()
    print(f"Scanning Arrow files across {len(scales)} scales...")
    all_files = list_arrow_files(source_path, scales)
    print(f"  Found {len(all_files)} Arrow files")
    print("  Memory formula: arrow + 2 * shard_on_tmpfs + 2 GiB")

    if not all_files:
        print("No Arrow files found. Check SOURCE_PATH and SCALES in .env.")
        sys.exit(1)

    # --- Optional: filter empty shards via Cloud Run ---
    if args.remove_zeros:
        all_files = _remove_zero_shards(all_files, env, source_path)

    # --- Step 2: Assign to tiers ---
    tier_map = assign_tiers(all_files, max_tasks)

    print("\nTier assignments:")
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

    # Delete stale manifests from previous runs
    manifest_uri = f"{source_path}/manifests/"
    print(f"\nDeleting stale manifests under {manifest_uri} ...")
    result = subprocess.run(
        ["gsutil", "-m", "-q", "rm", "-r", f"{manifest_uri}"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("  Deleted.")
    elif "No URLs matched" in result.stderr or "CommandException" in result.stderr:
        print("  No stale manifests found.")

    tier_info = {}  # gib -> (manifest_uri, num_tasks)

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

    elapsed = time.monotonic() - t0
    for gib in sorted(tier_info.keys()):
        tier_uri, num_tasks = tier_info[gib]
        print(f"  {gib}Gi: {num_tasks} tasks → {tier_uri}/")
    print(f"\nManifest generation took {elapsed:.1f}s")

    # --- Step 4: Get Docker image ---
    image = _get_image(env)
    if not image:
        print("Error: DOCKER_IMAGE not set in .env. Run 'pixi run deploy' first.")
        sys.exit(1)
    print(f"\nUsing image: {image}")

    # --- Step 5: Create/update and execute per-tier jobs ---
    _launch_tier_jobs(
        tier_info, env, image, ng_spec_b64, base_name, project, region,
        label_type, downres, args.wait,
    )


if __name__ == "__main__":
    main()
