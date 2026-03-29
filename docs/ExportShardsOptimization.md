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

### What data we have for each

| Data | v0.11 | false-merge-corrected |
|------|-------|-----------------------|
| Arrow file sizes | Yes | Yes |
| CSV chunk counts | Yes | Yes |
| Label profiles (`-labels.csv` from `profile_shards.py`) | **Yes** (21,834 files at `gs://flyem-dvid-exports/mCNS-98d699/segmentation/`) | **No** — needs `profile_shards.py` run |
| NG output shard sizes (per-shard) | **Yes** (25,541-row CSV at `analysis/v011_shard_memory.csv`) | **Totals/max/p95 per scale only** — no per-shard CSV |
| Full correlation CSV (input→output) | **Yes** | **No** — needs both label profiles and per-shard NG sizes |

### Measured NG output shard sizes

| Scale | v0.11 shards | v0.11 total | v0.11 max | FMC shards | FMC total | FMC max |
|-------|-------------|-------------|-----------|-----------|-----------|---------|
| s0 | 21,690 | 2,558 GB | 457 MB | 21,690 | 2,565 GB | 457 MB |
| s1 | 3,294 | 970 GB | 782 MB | 3,294 | 974 GB | 792 MB |
| s2 | 597 | 251 GB | 1,438 MB | 597 | 252 GB | 1,443 MB |
| s3 | 120 | 65 GB | 2,539 MB | 120 | 65 GB | 2,568 MB |
| s4 | 25 | 17 GB | 2,905 MB | 25 | 17 GB | 2,939 MB |
| s5 | 8 | 4.2 GB | 2,145 MB | 8 | 4.2 GB | 2,145 MB |
| s6 | 2 | 846 MB | 842 MB | 2 | 859 MB | 859 MB |
| s7 | 1 | 138 MB | — | 1 | 141 MB | — |
| s8 | 1 | 22 MB | — | 1 | 23 MB | — |
| s9 | 1 | 3.6 MB | — | 1 | 3.8 MB | — |

NG output sizes differ by 0–1.2% between the two datasets. However, we have
**not measured** whether the label counts differ — the false-merge-corrected
dataset removed proofread merges, but we don't know how many additional unique
labels that produced. Running `profile_shards.py` on both datasets would answer
this and tell us whether the regression coefficients are stable across label
states, or whether the similar NG sizes simply reflect similar label densities.

## Regression Analysis (v0.11, 25,541 shards)

From `analysis/v011_shard_memory.csv`, fitting `ng_output_bytes = coeff × predictor`
(least-squares through origin):

| Scale | N | B/chunk | R²(chunk) | B/unique_label | R²(UL) | B/supervoxel | R²(SV) |
|-------|------|---------|-----------|----------------|--------|--------------|--------|
| s0 | 21,533 | 5,275 | 0.725 | 395 | 0.956 | 347 | 0.958 |
| s1 | 3,261 | 15,602 | 0.806 | 342 | 0.962 | 287 | 0.970 |
| s2 | 591 | 31,102 | 0.890 | 153 | 0.976 | 125 | 0.983 |
| s3 | 118 | 60,633 | 0.953 | 55 | 0.990 | 44 | 0.990 |
| s4 | 25 | 114,754 | 0.982 | 23 | 0.996 | 18 | 0.992 |
| s5 | 8 | 195,951 | 0.999 | 13 | 0.999 | 10 | 0.996 |

**Key findings:**

1. **Label count is a much better predictor than chunk count.** R² > 0.95 at all
   scales for unique labels or supervoxels, vs 0.72–0.98 for chunk count.

2. **Chunk count alone is weak at s0** (R² = 0.725) because shards vary
   enormously in label density — a boundary shard may have 30,000 chunks of
   sparse tissue, while a dense interior shard has 30,000 chunks packed with
   labels. Same chunk count, wildly different output size.

3. **The per-scale coefficients vary significantly.** `B/chunk` spans 37×
   (5,275 at s0 → 195,951 at s5). `B/unique_label` spans 30× (395 at s0 → 13 at
   s5). This means any formula needs per-scale constants.

4. **`total_sv` and `total_labels` are identical in v0.11** (agglomerated labels =
   supervoxels for this particular export). They would differ for a
   supervoxel-only export.

## Two Prediction Tiers

### Tier 1: Chunk-count model (conservative, no profiling)

The actual chunk count per shard is available cheaply from the companion CSV
files in the DVID export (`wc -l` minus header). For downres scales, it comes
from `chunks_per_shard()` in `ng_sharding.py`.

```
output_shard_bytes ≈ num_chunks × BYTES_PER_CHUNK[scale]
```

This uses the worst-case bytes-per-chunk observed across all shards at that
scale. It overestimates for most shards because the brain sparsely fills the
bounding volume — boundary shards and tissue-edge shards have much lower label
density per chunk than the worst-case interior shard. At s0, the median
overestimate is 3× and the 95th percentile is 20×. But it never underestimates,
so it's safe for tier placement (overestimation only wastes memory headroom).

### Tier 2: Label-aware model (precise, requires profiling)

When per-shard label profiles are available (from running `profile_shards.py`
against the DVID Arrow files), use the label-based predictor:

```
output_shard_bytes ≈ total_unique_labels × BYTES_PER_UNIQUE_LABEL[scale]
```

This produces much tighter tier assignments (R² > 0.95 at all scales),
potentially saving 20–30% on Cloud Run costs by avoiding unnecessary bumps to
larger tiers.

For s1+ (downres), we don't have DVID Arrow files to profile. The chunk-count
model is the only option unless we develop a way to estimate child-scale label
counts from parent-scale data.

## Derived Constants

### `BYTES_PER_CHUNK` (chunk-count model)

`max(ng_output_bytes / chunk_count)` across all shards at each scale. This is
**not** `max_shard_file / full_shard_chunks` — the densest bytes-per-chunk shard
is often a partially-filled shard with high label density, not a full 32K-chunk
interior shard. (At s4, the densest shard has 22,565 chunks, not 32,768.)

From v0.11 analysis (25,541 shards):

| Scale | Bytes/chunk | Source shard | Chunks | Output |
|-------|-------------|-------------|--------|--------|
| s0 | 13,932 | 36864_43008_108544 | 32,768 | 457 MB |
| s1 | 27,919 | 26624_34816_49152 | 24,603 | 687 MB |
| s2 | 44,482 | 6144_10240_26624 | 1,558 | 69 MB |
| s3 | 77,480 | 6144_2048_2048 | 32,768 | 2,539 MB |
| s4 | 128,757 | 2048_0_0 | 22,565 | 2,905 MB |
| s5 | 196,923 | 0_0_2048 | 6,138 | 1,209 MB |
| s6 | 255,166 | 0_0_0 | 3,301 | 842 MB |
| s7 | 166,127 | 0_0_0 | 828 | 138 MB |
| s8 | 153,717 | 0_0_0 | 143 | 22 MB |
| s9 | 125,268 | 0_0_0 | 29 | 3.6 MB |

**Limitation**: this is a worst-case upper bound. The brain sparsely fills the
bounding volume, so most shards have much lower bytes-per-chunk than the max.
At s0, the median overestimate is 3.0× and the 95th percentile is 20×. This
means the chunk-count model reliably avoids under-allocation (OOM) but wastes
tier capacity for most shards. The label-aware model is much tighter.

### `BYTES_PER_UNIQUE_LABEL` (label-aware model)

Regression coefficients from v0.11 (single-dataset; needs validation with FMC):

| Scale | Coefficient | R² |
|-------|-------------|------|
| s0 | 395 | 0.956 |
| s1 | 342 | 0.962 |
| s2 | 153 | 0.976 |
| s3 | 55 | 0.990 |
| s4 | 23 | 0.996 |
| s5 | 13 | 0.999 |

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
