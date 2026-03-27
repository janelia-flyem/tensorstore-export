#!/usr/bin/env python3
"""
Analyze v0.11 export data to derive a label-aware memory formula.

Correlates per-shard label profiles (from -labels.csv files) with actual
neuroglancer output shard sizes to build an empirical model for memory
estimation.  The output replaces the conservative KB_PER_CHUNK lookup in
precompute_manifest.py with a label-aware formula.

Data sources (all on GCS, read-only):
  - DVID Arrow shard files (for arrow_bytes)
  - Per-shard -labels.csv files (for chunk_count, total_labels, total_sv)
  - Neuroglancer precomputed shard files (for actual ng_output_bytes)
  - NG spec JSON (for scale-to-resolution mapping and sharding params)

Output:
  - analysis/v011_shard_memory.csv: full correlation dataset
  - Console report: per-scale stats, regression, tier comparison, formula

Usage:
    pixi run analyze-memory \\
      --source gs://flyem-male-cns/dvid-exports/mCNS-98d699/segmentation \\
      --labels gs://flyem-dvid-exports/mCNS-98d699/segmentation \\
      --dest gs://flyem-male-cns/v0.11/segmentation \\
      --ng-spec examples/mcns-v0.11-export-specs.json
"""

import argparse
import csv
import io
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


# ---------------------------------------------------------------------------
# Morton code / NG spec — centralized in src/ng_sharding.py
# ---------------------------------------------------------------------------

from src.ng_sharding import (  # noqa: E402
    dvid_to_ng_shard_number,
    load_ng_spec,
)


# ---------------------------------------------------------------------------
# GCS data collection
# ---------------------------------------------------------------------------

def list_blobs_with_sizes(prefix):
    """List blobs under a GCS prefix -> {filename: size_bytes}."""
    bucket_name, blob_prefix = _parse_gs(prefix)
    result = {}
    for blob in _get_gcs_client().bucket(bucket_name).list_blobs(
            prefix=blob_prefix):
        name = blob.name.split("/")[-1]
        result[name] = blob.size
    return result


def read_label_csvs(labels_path, scale, shard_names):
    """Read -labels.csv files in parallel -> {shard_name: stats_dict}."""
    prefix = f"{labels_path}/s{scale}/"
    bucket_name, blob_prefix = _parse_gs(prefix)
    bucket = _get_gcs_client().bucket(bucket_name)

    available = {}
    for blob in bucket.list_blobs(prefix=blob_prefix):
        if blob.name.endswith("-labels.csv"):
            name = blob.name.split("/")[-1].replace("-labels.csv", "")
            available[name] = blob

    to_read = [(name, available[name]) for name in shard_names
               if name in available]

    def _read_one(item):
        name, blob = item
        data = blob.download_as_text()
        reader = csv.DictReader(io.StringIO(data))
        labels_list = []
        sv_list = []
        unique_labels_list = []
        for row in reader:
            labels_list.append(int(row["num_labels"]))
            sv_list.append(int(row["num_supervoxels"]))
            # unique_labels column added in profiler v2; fall back to num_labels
            ul = row.get("unique_labels")
            unique_labels_list.append(int(ul) if ul else int(row["num_labels"]))
        if not labels_list:
            return name, None
        n = len(labels_list)
        return name, {
            "chunk_count": n,
            "total_labels": sum(labels_list),
            "mean_labels": sum(labels_list) / n,
            "max_labels": max(labels_list),
            "total_sv": sum(sv_list),
            "mean_sv": sum(sv_list) / n,
            "max_sv": max(sv_list),
            "total_unique_labels": sum(unique_labels_list),
            "mean_unique_labels": sum(unique_labels_list) / n,
            "max_unique_labels": max(unique_labels_list),
        }

    result = {}
    total = len(to_read)
    done = 0
    log_interval = max(1, total // 10)
    with ThreadPoolExecutor(max_workers=64) as pool:
        for name, stats in pool.map(_read_one, to_read):
            if stats:
                result[name] = stats
            done += 1
            if done % log_interval == 0 or done == total:
                print(f"    Read {done}/{total} label CSVs...", flush=True)

    return result


def process_scale(scale, source_path, labels_path, dest_path, scale_params):
    """Collect correlation data for one scale -> list of row dicts."""
    t0 = time.time()
    ng_key = scale_params["key"]

    print(f"  Scale {scale}: listing Arrow files...", flush=True)
    arrow_blobs = list_blobs_with_sizes(f"{source_path}/s{scale}/")
    arrow_files = {name.replace(".arrow", ""): size
                   for name, size in arrow_blobs.items()
                   if name.endswith(".arrow")}
    print(f"  Scale {scale}: {len(arrow_files)} Arrow files, "
          f"listing NG shards ({ng_key}/)...", flush=True)

    ng_blobs = list_blobs_with_sizes(f"{dest_path}/{ng_key}/")
    ng_by_number = {}
    for name, size in ng_blobs.items():
        if name.endswith(".shard"):
            num = int(name.replace(".shard", ""), 16)
            ng_by_number[num] = size
    print(f"  Scale {scale}: {len(ng_by_number)} NG shards, "
          f"reading {len(arrow_files)} label CSVs...", flush=True)

    label_data = read_label_csvs(
        labels_path, scale, list(arrow_files.keys()))

    # Build forward mapping and match
    rows = []
    matched = 0
    unmatched_ng = 0
    unmatched_labels = 0
    ng_collisions = defaultdict(list)

    for shard_name, arrow_bytes in arrow_files.items():
        ng_num = dvid_to_ng_shard_number(shard_name, scale_params)
        ng_collisions[ng_num].append(shard_name)
        ng_bytes = ng_by_number.get(ng_num)
        labels = label_data.get(shard_name)

        if ng_bytes is None:
            unmatched_ng += 1
            continue
        if labels is None:
            unmatched_labels += 1
            continue

        matched += 1
        rows.append({
            "scale": scale,
            "shard_name": shard_name,
            "arrow_bytes": arrow_bytes,
            "chunk_count": labels["chunk_count"],
            "total_labels": labels["total_labels"],
            "mean_labels": round(labels["mean_labels"], 1),
            "max_labels": labels["max_labels"],
            "total_sv": labels["total_sv"],
            "mean_sv": round(labels["mean_sv"], 1),
            "max_sv": labels["max_sv"],
            "total_unique_labels": labels["total_unique_labels"],
            "mean_unique_labels": round(labels["mean_unique_labels"], 1),
            "max_unique_labels": labels["max_unique_labels"],
            "ng_output_bytes": ng_bytes,
        })

    multi = sum(1 for v in ng_collisions.values() if len(v) > 1)
    elapsed = time.time() - t0
    print(f"  Scale {scale}: {matched} matched, {unmatched_ng} no-NG, "
          f"{unmatched_labels} no-labels "
          f"({len(arrow_files)} arrow, {len(ng_by_number)} NG, "
          f"{len(label_data)} labels) [{elapsed:.0f}s]")
    if multi:
        print(f"    WARNING: {multi} NG shards map to multiple DVID shards")

    return rows


# ---------------------------------------------------------------------------
# Statistics (no numpy dependency)
# ---------------------------------------------------------------------------

def _percentile(sorted_vals, pct):
    """Percentile from pre-sorted list."""
    if not sorted_vals:
        return 0
    idx = min(int(pct / 100 * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def _ls_1var(x, y):
    """Fit y = a*x (through origin).  Returns (a, R^2)."""
    n = len(x)
    if n == 0:
        return 0.0, 0.0
    xy = sum(xi * yi for xi, yi in zip(x, y))
    xx = sum(xi * xi for xi in x)
    a = xy / xx if xx else 0
    y_mean = sum(y) / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((yi - a * xi) ** 2 for xi, yi in zip(x, y))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0
    return a, r2


def _ls_2var(x1, x2, y):
    """Fit y = a*x1 + b*x2 (through origin).  Returns (a, b, R^2)."""
    n = len(x1)
    if n < 3:
        return 0.0, 0.0, 0.0
    a11 = sum(a * a for a in x1)
    a22 = sum(a * a for a in x2)
    a12 = sum(a * b for a, b in zip(x1, x2))
    b1 = sum(a * b for a, b in zip(x1, y))
    b2 = sum(a * b for a, b in zip(x2, y))
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-10:
        return 0.0, 0.0, 0.0
    a = (a22 * b1 - a12 * b2) / det
    b = (a11 * b2 - a12 * b1) / det
    y_mean = sum(y) / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((yi - a * x1i - b * x2i) ** 2
                 for x1i, x2i, yi in zip(x1, x2, y))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0
    return a, b, r2


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(rows):
    """Print full analysis report to console."""
    by_scale = defaultdict(list)
    for r in rows:
        by_scale[r["scale"]].append(r)

    # -- Per-scale summary --
    print("\n" + "=" * 80)
    print("PER-SCALE SUMMARY")
    print("=" * 80)

    for scale in sorted(by_scale):
        sr = by_scale[scale]
        n = len(sr)
        arrow_gb = sum(r["arrow_bytes"] for r in sr) / 1e9
        ng_gb = sum(r["ng_output_bytes"] for r in sr) / 1e9
        chunks = sum(r["chunk_count"] for r in sr)

        kb_per_chunk = sorted(
            r["ng_output_bytes"] / r["chunk_count"] / 1024 for r in sr)
        bytes_per_sv = sorted(
            r["ng_output_bytes"] / r["total_sv"]
            for r in sr if r["total_sv"] > 0)

        print(f"\n  Scale {scale}: {n} shards, {chunks:,} chunks")
        print(f"    Arrow: {arrow_gb:.1f} GB    NG output: {ng_gb:.1f} GB")
        print(f"    KB/chunk:  p5={_percentile(kb_per_chunk, 5):.0f}  "
              f"p50={_percentile(kb_per_chunk, 50):.0f}  "
              f"p95={_percentile(kb_per_chunk, 95):.0f}  "
              f"max={kb_per_chunk[-1]:.0f}")
        if bytes_per_sv:
            print(f"    Bytes/SV:  p5={_percentile(bytes_per_sv, 5):.0f}  "
                  f"p50={_percentile(bytes_per_sv, 50):.0f}  "
                  f"p95={_percentile(bytes_per_sv, 95):.0f}  "
                  f"max={bytes_per_sv[-1]:.0f}")

    # -- Regression --
    print("\n" + "=" * 80)
    print("REGRESSION: ng_output_bytes ~ f(label_profile)")
    print("=" * 80)

    best_per_scale = {}

    for scale in sorted(by_scale):
        sr = by_scale[scale]
        n = len(sr)
        if n < 10:
            print(f"\n  Scale {scale}: {n} shards (skipping — too few)")
            continue

        y = [r["ng_output_bytes"] for r in sr]
        sv = [r["total_sv"] for r in sr]
        lab = [r["total_labels"] for r in sr]
        ul = [r["total_unique_labels"] for r in sr]
        ch = [r["chunk_count"] for r in sr]

        a_sv, r2_sv = _ls_1var(sv, y)
        a_lab, r2_lab = _ls_1var(lab, y)
        a_ul, r2_ul = _ls_1var(ul, y)
        a2, b2, r2_2 = _ls_2var(ul, ch, y)

        # Relative prediction errors for the best single-var model
        models = [("total_sv", a_sv, r2_sv, sv),
                  ("total_unique_labels", a_ul, r2_ul, ul)]
        best_name, best_a, best_r2, best_x = max(models, key=lambda m: m[2])
        pred = [best_a * xi for xi in best_x]
        errs = sorted(abs(yi - pi) / yi for yi, pi in zip(y, pred) if yi > 0)

        print(f"\n  Scale {scale} ({n} shards):")
        print(f"    ng = {a_sv:.0f} * total_sv"
              f"                        R²={r2_sv:.4f}")
        print(f"    ng = {a_lab:.0f} * total_labels (list len)"
              f"        R²={r2_lab:.4f}")
        print(f"    ng = {a_ul:.0f} * total_unique_labels"
              f"           R²={r2_ul:.4f}")
        print(f"    ng = {a2:.0f} * total_unique_labels + {b2:.0f} * chunks"
              f"    R²={r2_2:.4f}")
        if errs:
            print(f"    Best ({best_name}) error: "
                  f"p50={_percentile(errs, 50) * 100:.1f}%  "
                  f"p95={_percentile(errs, 95) * 100:.1f}%  "
                  f"max={errs[-1] * 100:.1f}%")

        best_per_scale[scale] = {
            "a_sv": a_sv, "r2_sv": r2_sv,
            "a_ul": a_ul, "r2_ul": r2_ul,
        }

    # Global regression across all scales
    all_sv = [r["total_sv"] for r in rows]
    all_ul = [r["total_unique_labels"] for r in rows]
    all_y = [r["ng_output_bytes"] for r in rows]
    global_a_sv, global_r2_sv = _ls_1var(all_sv, all_y)
    global_a_ul, global_r2_ul = _ls_1var(all_ul, all_y)
    print(f"\n  GLOBAL: ng = {global_a_sv:.0f} * total_sv              R²={global_r2_sv:.4f}")
    print(f"  GLOBAL: ng = {global_a_ul:.0f} * total_unique_labels   R²={global_r2_ul:.4f}")

    # -- Tier comparison --
    print("\n" + "=" * 80)
    print("TIER DISTRIBUTION: OLD -> NEW -> ORACLE")
    print("=" * 80)

    KB_PER_CHUNK_OLD = {
        0: 150, 1: 200, 2: 280, 3: 400, 4: 530,
        5: 630, 6: 750, 7: 530, 8: 570, 9: 430,
    }
    TIERS = [4, 8, 16, 24, 32]

    def _pick_tier(mem_gib):
        for t in TIERS:
            if mem_gib <= t:
                return t
        return TIERS[-1]

    def _old(r):
        ag = r["arrow_bytes"] / (1 << 30)
        sg = r["chunk_count"] * KB_PER_CHUNK_OLD.get(r["scale"], 400) / (1 << 20)
        return (ag + 1.3 * sg + 1.5) * 1.3

    def _new_sv(r):
        ag = r["arrow_bytes"] / (1 << 30)
        coeff = best_per_scale.get(r["scale"], {}).get("a_sv", global_a_sv)
        ng = coeff * r["total_sv"] / (1 << 30)
        return (ag + ng + 1.5) * 1.15

    def _new_ul(r):
        ag = r["arrow_bytes"] / (1 << 30)
        coeff = best_per_scale.get(r["scale"], {}).get("a_ul", global_a_ul)
        ng = coeff * r["total_unique_labels"] / (1 << 30)
        return (ag + ng + 1.5) * 1.15

    def _oracle(r):
        ag = r["arrow_bytes"] / (1 << 30)
        ng = r["ng_output_bytes"] / (1 << 30)
        return (ag + ng + 1.5) * 1.15

    old_d = defaultdict(int)
    sv_d = defaultdict(int)
    ul_d = defaultdict(int)
    ora_d = defaultdict(int)
    for r in rows:
        old_d[_pick_tier(_old(r))] += 1
        sv_d[_pick_tier(_new_sv(r))] += 1
        ul_d[_pick_tier(_new_ul(r))] += 1
        ora_d[_pick_tier(_oracle(r))] += 1

    total = len(rows)
    print(f"\n  {'Tier':>8}  {'Old':>8}  {'New(SV)':>8}  {'New(UL)':>8}  {'Oracle':>8}")
    print(f"  {'----':>8}  {'---':>8}  {'------':>8}  {'------':>8}  {'------':>8}")
    for t in TIERS:
        o = old_d.get(t, 0)
        sv = sv_d.get(t, 0)
        ul = ul_d.get(t, 0)
        ora = ora_d.get(t, 0)
        print(f"  {t:>6}Gi  {o:>8}  {sv:>8}  {ul:>8}  {ora:>8}")
    print(f"  {'Total':>8}  {total:>8}  {total:>8}  {total:>8}  {total:>8}")

    # -- Overestimation analysis --
    print("\n" + "=" * 80)
    print("OVERESTIMATION ANALYSIS (formula / oracle ratio)")
    print("=" * 80)

    for label, fn in [("Old", _old), ("New(SV)", _new_sv),
                       ("New(UL)", _new_ul)]:
        print(f"\n  {label} formula:")
        for scale in sorted(by_scale):
            sr = by_scale[scale]
            ratios = sorted(fn(r) / _oracle(r) for r in sr if _oracle(r) > 0)
            if not ratios:
                continue
            print(f"    Scale {scale}: "
                  f"p50={_percentile(ratios, 50):.2f}x  "
                  f"p95={_percentile(ratios, 95):.2f}x  "
                  f"max={ratios[-1]:.2f}x")

    # -- Suggested formula --
    print("\n" + "=" * 80)
    print("SUGGESTED FORMULA FOR precompute_manifest.py")
    print("=" * 80)

    print(f"\n  Coefficients by scale:")
    print(f"  {'Scale':>7}  {'Bytes/SV':>10}  {'R²(SV)':>8}  "
          f"{'Bytes/UL':>10}  {'R²(UL)':>8}")
    for s in sorted(best_per_scale):
        info = best_per_scale[s]
        print(f"  {s:>7}  {info['a_sv']:>10.0f}  {info['r2_sv']:>8.4f}  "
              f"{info['a_ul']:>10.0f}  {info['r2_ul']:>8.4f}")

    print(f"\n  For agglomerated label exports (use unique_labels):")
    bps_ul = ", ".join(
        f"{s}: {best_per_scale[s]['a_ul']:.0f}"
        for s in sorted(best_per_scale))
    print(f"""
  BYTES_PER_UNIQUE_LABEL = {{{bps_ul}}}

  def estimate_memory_gib(arrow_bytes, total_unique_labels, scale):
      arrow_gib = arrow_bytes / (1 << 30)
      coeff = BYTES_PER_UNIQUE_LABEL.get(scale, {global_a_ul:.0f})
      ng_gib = total_unique_labels * coeff / (1 << 30)
      return (arrow_gib + ng_gib + 1.5) * 1.15
""")
    print(f"  For supervoxel exports (use total_sv):")
    bps_sv = ", ".join(
        f"{s}: {best_per_scale[s]['a_sv']:.0f}"
        for s in sorted(best_per_scale))
    print(f"""
  BYTES_PER_SV = {{{bps_sv}}}

  def estimate_memory_gib(arrow_bytes, total_sv, scale):
      arrow_gib = arrow_bytes / (1 << 30)
      coeff = BYTES_PER_SV.get(scale, {global_a_sv:.0f})
      ng_gib = total_sv * coeff / (1 << 30)
      return (arrow_gib + ng_gib + 1.5) * 1.15
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze v0.11 export to derive label-aware memory formula.",
    )
    parser.add_argument(
        "--source", required=True,
        help="GCS path to DVID Arrow shard files",
    )
    parser.add_argument(
        "--labels", required=True,
        help="GCS path to -labels.csv files",
    )
    parser.add_argument(
        "--dest", required=True,
        help="GCS path to neuroglancer precomputed output",
    )
    parser.add_argument(
        "--ng-spec", required=True,
        help="Local path to neuroglancer multiscale spec JSON",
    )
    parser.add_argument(
        "--scales", default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated scales (default: 0-9)",
    )
    parser.add_argument(
        "--output", default="analysis/v011_shard_memory.csv",
        help="Output CSV path (default: analysis/v011_shard_memory.csv)",
    )
    args = parser.parse_args()

    scales = [int(s) for s in args.scales.split(",")]
    scale_info = load_ng_spec(args.ng_spec)

    print("Analyzing v0.11 memory profile")
    print(f"  Source: {args.source}")
    print(f"  Labels: {args.labels}")
    print(f"  Dest:   {args.dest}")
    print(f"  Scales: {scales}")
    print()

    all_rows = []
    t0 = time.time()

    for scale in scales:
        if scale not in scale_info:
            print(f"  Scale {scale}: not in NG spec, skipping")
            continue
        rows = process_scale(
            scale, args.source, args.labels, args.dest, scale_info[scale])
        all_rows.extend(rows)

    elapsed = time.time() - t0
    print(f"\nData collection: {len(all_rows)} matched shards [{elapsed:.0f}s]")

    if not all_rows:
        print("No matched shards. Check paths and verify data exists.")
        sys.exit(1)

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scale", "shard_name", "arrow_bytes", "chunk_count",
        "total_labels", "mean_labels", "max_labels",
        "total_sv", "mean_sv", "max_sv",
        "total_unique_labels", "mean_unique_labels", "max_unique_labels",
        "ng_output_bytes",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(all_rows, key=lambda r: (r["scale"], r["shard_name"])):
            writer.writerow(r)

    print(f"Written: {output_path} ({len(all_rows)} rows)")

    analyze(all_rows)


if __name__ == "__main__":
    main()
