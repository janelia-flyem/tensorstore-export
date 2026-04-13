#!/usr/bin/env python3
"""Sample exported NG voxels and compare them against DVID batch labels."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export_value_verifier import (
    build_exposed_faces,
    build_shard_records,
    compare_export_and_dvid_points,
    fetch_dvid_labels,
    list_scale_shards,
    load_scale_params,
    map_export_point_to_dvid,
    open_precomputed_scale,
    sample_points_in_shards,
    sample_points_outside_shards,
    voxel_to_chunk_coords,
)

MAX_EXAMPLES = 10


def _format_point(point: tuple[int, int, int]) -> str:
    return f"({point[0]}, {point[1]}, {point[2]})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample exported neuroglancer segmentation voxels and compare "
            "them against DVID's batch /labels endpoint."
        )
    )
    parser.add_argument(
        "--points",
        type=int,
        default=1000,
        help=(
            "Number of inside-shard samples, and the same count for checks "
            "just outside shard boundaries."
        ),
    )
    parser.add_argument(
        "--scale",
        type=int,
        required=True,
        help="Export scale index to validate against the same DVID scale index.",
    )
    parser.add_argument(
        "--supervoxels",
        action="store_true",
        help="Query DVID with supervoxels=true instead of agglomerated labels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Points per DVID batch /labels request (default: 1000).",
    )
    parser.add_argument(
        "--tensorstore-concurrency",
        type=int,
        default=32,
        help="Maximum in-flight TensorStore chunk reads for export lookups (default: 32).",
    )
    parser.add_argument(
        "--z-compress",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Map export z to DVID z by multiplying by (N+1). "
            "Use the same value passed to export, e.g. --z-compress 1."
        ),
    )
    parser.add_argument("data_name", help="DVID labelmap instance name")
    parser.add_argument("uuid", help="DVID version UUID")
    parser.add_argument(
        "volume_path",
        help="NG precomputed sharded segmentation root (gs://... or local path)",
    )
    parser.add_argument("dvid_url", help="Base URL of the DVID server")
    args = parser.parse_args()

    if args.points <= 0:
        parser.error("--points must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.tensorstore_concurrency <= 0:
        parser.error("--tensorstore-concurrency must be > 0")
    if args.z_compress < 0:
        parser.error("--z-compress must be >= 0")

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    rng = random.Random(seed)

    all_scale_params = load_scale_params(args.volume_path)
    if args.scale not in all_scale_params:
        parser.error(f"--scale {args.scale} not present in export info")
    params = all_scale_params[args.scale]

    print(f"Volume: {args.volume_path}")
    print(f"DVID:   {args.dvid_url}")
    print(f"Data:   {args.data_name} @ {args.uuid}")
    print(f"Scale:  {args.scale} ({params['key']})")
    print(f"Mode:   {'supervoxels' if args.supervoxels else 'agglomerated labels'}")
    print(f"Z map:  export z -> dvid z * {args.z_compress + 1}")
    print(f"TS IO:  {args.tensorstore_concurrency} in-flight chunk reads")
    print(f"Seed:   {seed}")

    shard_numbers = list_scale_shards(args.volume_path, params["key"])
    if not shard_numbers:
        print("\nNo shard files found for the requested scale.")
        sys.exit(1)

    shard_records = build_shard_records(params, shard_numbers)
    inside_points = sample_points_in_shards(shard_records, args.points, rng)
    faces = build_exposed_faces(params, shard_records, shard_numbers)
    outside_points = sample_points_outside_shards(faces, args.points, rng)

    print(
        f"\nScale {args.scale} ({params['key']}): "
        f"{len(shard_numbers)} shard files, "
        f"{len(inside_points)} inside samples, "
        f"{len(outside_points)} boundary-outside samples"
    )

    store = open_precomputed_scale(args.volume_path, args.scale)
    inside_chunks = {voxel_to_chunk_coords(point, params) for point in inside_points}
    total_inside_batches = (len(inside_points) + args.batch_size - 1) // args.batch_size
    print(
        f"  Reading {len(inside_points)} inside samples from "
        f"{len(inside_chunks)} export chunks and pipelining them into "
        f"{total_inside_batches} DVID batch request(s)...",
        flush=True,
    )
    mismatches = compare_export_and_dvid_points(
        store,
        inside_points,
        [
            map_export_point_to_dvid(point, args.z_compress)
            for point in inside_points
        ],
        params,
        dvid_url=args.dvid_url,
        uuid=args.uuid,
        data_name=args.data_name,
        scale=args.scale,
        supervoxels=args.supervoxels,
        batch_size=args.batch_size,
        tensorstore_read_concurrency=args.tensorstore_concurrency,
        tensorstore_progress_step=100,
        tensorstore_progress_label="Read export values",
        dvid_batch_label="Queried DVID inside batches",
    )

    any_failures = False
    if mismatches:
        any_failures = True
        print(f"  *** {len(mismatches)} inside-point mismatches ***")
        for export_point, dvid_point, export_value, dvid_value in mismatches[:MAX_EXAMPLES]:
            print(
                f"    export={_format_point(export_point)} "
                f"dvid={_format_point(dvid_point)} "
                f"export_value={export_value} dvid_value={dvid_value}"
            )
        if len(mismatches) > MAX_EXAMPLES:
            print(f"    ... and {len(mismatches) - MAX_EXAMPLES} more")
    else:
        print("  Inside-point values match DVID")

    if not faces:
        print("  No exposed faces found; just-outside-boundary check skipped")
    else:
        dvid_outside_points = [
            map_export_point_to_dvid(point, args.z_compress) for point in outside_points
        ]
        total_outside_batches = (
            len(dvid_outside_points) + args.batch_size - 1
        ) // args.batch_size
        print(
            f"  Querying DVID for {len(dvid_outside_points)} points just "
            f"outside shard boundaries in {total_outside_batches} batch request(s)...",
            flush=True,
        )
        outside_values = fetch_dvid_labels(
            args.dvid_url,
            args.uuid,
            args.data_name,
            args.scale,
            dvid_outside_points,
            supervoxels=args.supervoxels,
            batch_size=args.batch_size,
            batch_label="Queried DVID boundary-outside batches",
        )
        outside_hits = []
        for export_point, dvid_point, dvid_value in zip(
            outside_points, dvid_outside_points, outside_values
        ):
            if dvid_value != 0:
                outside_hits.append((export_point, dvid_point, dvid_value))

        if outside_hits:
            any_failures = True
            print(
                f"  *** {len(outside_hits)} points just outside shard boundaries "
                f"returned non-zero labels ***"
            )
            for export_point, dvid_point, dvid_value in outside_hits[:MAX_EXAMPLES]:
                print(
                    f"    export={_format_point(export_point)} "
                    f"dvid={_format_point(dvid_point)} dvid_value={dvid_value}"
                )
            if len(outside_hits) > MAX_EXAMPLES:
                print(f"    ... and {len(outside_hits) - MAX_EXAMPLES} more")
        else:
            print(
                f"  Points just outside shard boundaries are absent from DVID "
                f"across {len(faces)} faces"
            )

    if any_failures:
        print("\nValue verification failed.")
        sys.exit(1)

    print("\nAll sampled values matched DVID.")


if __name__ == "__main__":
    main()
