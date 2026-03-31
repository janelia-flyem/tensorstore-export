# Export Shard Optimization

## The Core Problem: Predicting Output Shard Size

Each Cloud Run task needs a memory tier (4, 8, 16, 24, or 32 GiB). The dominant
variable cost is the **output `.shard` file on tmpfs** — a neuroglancer
precomputed shard file using compressed_segmentation + gzip encoding,
byte-identical to what ends up on GCS. Predicting this file's size determines
which tier a shard needs. Too small → OOM kill. Too large → wasted cost.

**Our approach**: derive a formula that predicts output shard file size from
measurable input characteristics — number of chunks, number of unique labels,
number of supervoxels, and scale — then calibrate the constants from real
production data. As we export additional datasets, we refine the constants and
validate that the relationships hold.

### The formula

```
output_shard_bytes = f(num_chunks, num_unique_labels, num_supervoxels, scale)
```

Once we can predict output shard size, memory follows directly:

```
memory = input_in_ram + tmpfs(output_shard) + fixed_overhead
```

| Component | s0 (Arrow source) | s1+ (downres) |
|-----------|-------------------|---------------|
| **Input in RAM** | Arrow file (~1× file size, known) | Source read cache (0.25 GiB, bounded) |
| **tmpfs** | 2× output shard (RMW peak) | 1× output shard (fresh staging dir) |
| **Fixed overhead** | ~2.0 GiB (Python + BRAID + pyarrow + TS + GCS) | ~1.5 GiB (Python + TS + GCS) |

The Arrow file size is known from `gsutil ls`. The fixed overhead is measurable
from cgroup data. **The only unknown is the output shard file size.**

## Reference Datasets

We calibrate from real production exports. Each new dataset we export adds to
the calibration pool.

| Dataset | Source Arrow+CSV | Output NG shards | Labels |
|---------|-----------------|------------------|--------|
| **v0.11** | `gs://flyem-male-cns/dvid-exports/mCNS-98d699/segmentation` | `gs://flyem-male-cns/v0.11/segmentation` | Agglomerated (proofread) |
| **false-merge-corrected** | `gs://flyem-dvid-shards/mCNS-de91d3/segmentation` | `gs://flyem-ng-staging/false-merge-corrected/segmentation` | Merges removed |

Both use the same volume (mCNS, 94,088 × 78,317 × 134,576 at s0), same NG spec,
same chunk size (64³), same sharding parameters.

### Data available

| Data | v0.11 | false-merge-corrected |
|------|-------|-----------------------|
| Arrow file sizes | Yes | Yes |
| CSV chunk counts | Yes | Yes |
| Label profiles (`-labels.csv`) | Yes (s0-s9) | Yes (s0-s9) |
| Full correlation CSV | `analysis/v011_shard_memory.csv` (25,541 rows) | `analysis/fmc_shard_memory.csv` (25,739 rows) |

### Cross-dataset comparison

NG output sizes differ by 0–1.2% between v0.11 and FMC. The regression
coefficients (`BYTES_PER_UNIQUE_LABEL`, `BYTES_PER_SV`) are identical within
rounding at all scales with sufficient data (s0-s5), confirming these are
properties of the compressed_segmentation encoding, not dataset-specific.

FMC has slightly higher max bytes-per-chunk at s3-s9 (1–4% above v0.11),
reflecting marginally higher label density after merge removal.
`BYTES_PER_CHUNK` uses the max across both datasets.

## Regression Analysis

Fitting `ng_output_bytes = coeff × predictor` (least-squares through origin),
validated on both v0.11 (25,541 shards) and FMC (25,739 shards). Coefficients
shown are from FMC; v0.11 produces identical values within rounding.

| Scale | N | B/unique_label | R²(UL) | B/supervoxel | R²(SV) |
|-------|------|----------------|--------|--------------|--------|
| s0 | 21,690 | 394 | 0.956 | 348 | 0.959 |
| s1 | 3,294 | 341 | 0.962 | 288 | 0.970 |
| s2 | 597 | 153 | 0.977 | 125 | 0.984 |
| s3 | 120 | 55 | 0.990 | 44 | 0.991 |
| s4 | 25 | 23 | 0.996 | 18 | 0.993 |
| s5 | 8 | 13 | 0.999 | 10 | 0.997 |

**Key findings:**

1. **Label count is a much better predictor than chunk count.** R² > 0.95 at all
   scales for unique labels or supervoxels, vs 0.72–0.98 for chunk count.

2. **Chunk count alone is weak at s0** (R² = 0.725) because shards vary
   enormously in label density — a boundary shard may have 30,000 chunks of
   sparse tissue, while a dense interior shard has 30,000 chunks packed with
   labels. Same chunk count, wildly different output size.

3. **The per-scale coefficients vary significantly.** `B/unique_label` spans 30×
   (394 at s0 → 13 at s5). This means any formula needs per-scale constants.

4. **Coefficients are stable across datasets.** v0.11 and FMC produce identical
   regression coefficients despite FMC having merges removed. This confirms the
   coefficients are a property of compressed_segmentation encoding geometry, not
   the specific label state.

## Prediction Model

### Label-aware model (primary)

```
output_shard_bytes ≈ total_unique_labels × BYTES_PER_UNIQUE_LABEL[scale]
```

R² > 0.95 at all scales, validated across two datasets. The number of unique
labels per shard is obtainable for all scales through a cascading prediction
chain (see below).

### Label prediction chain

Each step pre-computes the next scale's per-chunk label predictions so the
manifest can use them directly:

1. `profile_shards.py` on s0 Arrow files → writes s0 -labels.csv (actual)
   AND s1 predicted labels (from set union of 8 parent s0 chunks per s1 chunk)
2. An aggregation step merges partial s1 predictions into per-shard files
3. s0→s1 manifest reads s1 -labels.csv → `BYTES_PER_UNIQUE_LABEL[1]` → tier
4. s0→s1 worker writes s1 output, reads back chunks, counts actual labels,
   groups 8 s1 chunks per s2 chunk → writes s2 partial predictions
5. Aggregation merges s2 partials → per-shard s2 -labels.csv
6. s1→s2 manifest reads s2 -labels.csv → tier assignment
7. And so on (worker at scale N produces sN+1 predictions, unless last scale)

All -labels.csv files are stored in the source Arrow+CSV bucket under
`s{N}/` directories (creating new scale directories as needed). This keeps
label metadata co-located with source data in the same single-region bucket.

### Chunk-count fallback (deprecated)

When no -labels.csv exists for a scale, falls back to a conservative
chunk-count model:

```
output_shard_bytes ≈ num_chunks × BYTES_PER_CHUNK[scale]
```

This uses the worst-case bytes-per-chunk observed across all shards at that
scale. It overestimates by 2–5× at median (20× at p95 for s0) because the
brain sparsely fills the bounding volume. Safe for tier placement but wastes
cost. The label-aware model should be used for all production exports.

## Derived Constants

### `BYTES_PER_UNIQUE_LABEL` (primary model)

Regression coefficients validated across both datasets (v0.11 and FMC produce
identical coefficients within rounding, confirming these are properties of the
compressed_segmentation encoding rather than dataset-specific):

| Scale | B/unique_label | R²(UL) | B/supervoxel | R²(SV) |
|-------|---------------|--------|-------------|--------|
| s0 | 394 | 0.956 | 348 | 0.959 |
| s1 | 341 | 0.962 | 288 | 0.970 |
| s2 | 153 | 0.977 | 125 | 0.984 |
| s3 | 55 | 0.990 | 44 | 0.991 |
| s4 | 23 | 0.996 | 18 | 0.993 |
| s5 | 13 | 0.999 | 10 | 0.997 |

### `BYTES_PER_CHUNK` (fallback only)

Used only when no -labels.csv exists for a scale.
`max(ng_output_bytes / chunk_count)` across all shards, taking the maximum
across both datasets. Overestimates by 2–5× at median due to sparse volume
fill, but never underestimates.

| Scale | Bytes/chunk | v0.11 | FMC |
|-------|-------------|-------|-----|
| s0 | 13,932 | 13,932 | 13,932 |
| s1 | 27,919 | 27,919 | 27,919 |
| s2 | 44,482 | 44,482 | 44,482 |
| s3 | 78,376 | 77,480 | 78,376 |
| s4 | 130,244 | 128,757 | 130,244 |
| s5 | 199,396 | 196,923 | 199,396 |
| s6 | 258,999 | 255,166 | 258,999 |
| s7 | 170,713 | 166,127 | 170,713 |
| s8 | 159,269 | 153,717 | 159,269 |
| s9 | 129,385 | 125,268 | 129,385 |

### Fixed overheads

| Component | Value | Measured from |
|-----------|-------|---------------|
| `SHARD_PROC_OVERHEAD_GIB` | 2.0 GiB | cgroup memory.current at worker startup (s0 processing) |
| `DOWNRES_OVERHEAD_GIB` | 1.5 GiB | Python + TensorStore + GCS + 256 MB source cache |

## How to Calibrate for a New Dataset

### Step 1: Profile source shards (get label counts)

```bash
pixi run launch-profiler \
  --source gs://bucket/dvid-exports/dataset/segmentation \
  --output gs://bucket/dvid-exports/dataset/segmentation \
  --tasks 200
```

Runs `profile_shards.py` on Cloud Run. Produces `{shard}-labels.csv` files with
per-chunk stats: `x, y, z, num_labels, num_supervoxels, unique_labels`.

### Step 2: Export and measure NG output

After the export completes, measure actual NG shard file sizes.

### Step 3: Run correlation analysis

```bash
pixi run analyze-memory \
  --source gs://bucket/dvid-exports/dataset/segmentation \
  --labels gs://bucket/dvid-exports/dataset/segmentation \
  --dest gs://bucket/ng-output/dataset/segmentation \
  --ng-spec examples/dataset-export-specs.json \
  --output analysis/dataset_shard_memory.csv
```

Produces a per-shard CSV and console report with regression coefficients.

### Step 4: Update constants

- Compare new `BYTES_PER_UNIQUE_LABEL` coefficients against existing values.
  Take the max across all datasets for the chunk-count model.
- If max shard sizes at any scale exceed current `BYTES_PER_CHUNK` values, update.
- If fixed overheads differ (check cgroup logs), update.

## Outstanding Work

1. **Validate downres memory predictions**: After running manifest-driven downres
   jobs, compare predicted memory (from formula) against actual peak memory (from
   cgroup "Shard memory peak" logs) to calibrate `DOWNRES_OVERHEAD_GIB`.

---

## Appendix: S0-Only Export Rationale

The pipeline exports only scale 0 from DVID and generates scales 1–9 via
TensorStore downsampling on Cloud Run. This saves ~4 hours per export due to
DVID's dramatic per-scale throughput degradation (12× slowdown at lower scales).

See `ExportShardsByDownresPlan.md` for the manifest-driven per-shard downres
implementation.

### Per-scale DVID throughput

| Scale | Blocks | Throughput (blocks/sec) | Relative |
|-------|--------|------------------------|----------|
| 0 | 510.5M | 11,570 | 1.00× |
| 1 | 66.6M | ~6,100 | 0.53× |
| 2 | 8.8M | 2,810 | 0.24× |
| 3 | 1.2M | 1,482 | 0.13× |
| 4+ | 0.2M | 979 | 0.08× |

### Time savings

| Component | All scales from DVID | S0-only + Cloud Run downres | Savings |
|-----------|---------------------|----------------------------|---------|
| DVID export | 12h 35m | 8h 16m | 4h 19m |
| GCS upload | ~1h | ~40m | ~20m |
| Cloud Run | 1h 18m | ~1h 30m | -12m |
| **Total** | **~15h** | **~10–11h** | **~4–5h** |

### Label fidelity

TensorStore's `ts.downsample(source, [2,2,2,1], "mode")` uses the same
majority-vote algorithm as DVID's downres. For a finalized dataset, the results
are equivalent.

### Parent→child scale size ratios

| Pair | Ratio (child/parent total NG bytes) |
|------|--------------------------------------|
| s0→s1 | 0.379 |
| s1→s2 | 0.259 |
| s2→s3 | 0.257 |
| s3→s4 | 0.266 |
| s4→s5 | 0.244 |

After s0→s1, the ratio stabilizes at ~0.25–0.26.
