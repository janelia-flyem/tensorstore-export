# Memory Formula Optimization Plan

## Problem

The current memory estimation formula in `scripts/precompute_manifest.py` over-estimates by 2-3×, causing 64% of shards (16,717) to land in tier-16gi when actual peak memory was only 4.8 GiB. This makes tier-16gi the pipeline bottleneck at 1h18m wall time, while tier-4gi and tier-8gi finish in ~24 minutes with capacity to spare.

### Production results (mCNS v0.11)

| Tier | Shards | Actual max memory | Budget utilization |
|------|--------|-------------------|-------------------|
| 4 GiB | 5,169 | 1.2 G | 30% |
| 8 GiB | 4,027 | 3.0 G | 38% |
| 16 GiB | 16,717 | 4.8 G | 30% |
| 24 GiB | 196 | 8.7 G | 36% |
| 32 GiB | 16 | 9.5 G | 30% |

### Current formula

```python
KB_PER_CHUNK = {0: 150, 1: 200, 2: 280, 3: 400, 4: 530, 5: 630, ...}
shard_gib = chunk_count * KB_PER_CHUNK[scale] / (1024 * 1024)
memory_gib = (arrow_gib + 1.3 * shard_gib + 1.5) * 1.3
```

The compound 1.3× RMW overhead × 1.3× safety = 1.69× multiplier on the shard component, plus conservative `KB_PER_CHUNK` max values, produces estimates 2-3× above actual usage.

### Root cause: label density varies within a scale

The formula uses a single `KB_PER_CHUNK` per scale, but the neuroglancer compressed_segmentation output size depends heavily on the number of distinct labels per chunk — which varies enormously within a scale:

| Scale | SV/block mean | SV/block max | Labels/block mean | Labels/block max |
|-------|---------------|--------------|-------------------|------------------|
| 0 | 14 | 207 | 13 | 206 |
| 1 | 51 | 694 | 43 | 672 |
| 2 | 231 | 2,664 | 191 | 2,545 |
| 3 | 1,263 | 8,849 | 1,024 | 8,353 |
| 4 | 6,118 | 31,982 | 4,871 | 29,943 |
| 5 | 18,653 | 71,134 | 14,476 | 70,538 |

A scale 0 edge shard with 14 supervoxels/block compresses very differently than a scale 4 core shard with 6,000+ supervoxels/block. The current formula uses the max rate for the whole scale, which is correct for the densest shard but massively overestimates the typical shard.

## Data sources for optimization

We have three data sources that can be correlated to build an empirical model:

### 1. DVID export.log — per-scale aggregate statistics

Location: `gs://flyem-dvid-exports/mCNS-de91d3/segmentation/export.log` (new export)
Also: `gs://flyem-male-cns/dvid-exports/mCNS-98d699/segmentation/export.log` (v0.11)

Contains per-scale distributions of:
- Block sizes (compressed and uncompressed)
- Distinct supervoxels per block
- Distinct agglomerated labels per block
- Records per shard file, shard file sizes

### 2. Arrow shard files — per-chunk label profiles

Each Arrow record contains:
- `labels` (list\<uint64\>): agglomerated label IDs for this chunk
- `supervoxels` (list\<uint64\>): supervoxel IDs for this chunk
- `dvid_compressed_block` (binary): compressed block data
- `uncompressed_size` (uint32): pre-zstd block size

By reading the Arrow metadata (without decompressing blocks), we can build a per-shard profile:
- Number of chunks
- Per-chunk: `len(labels)`, `len(supervoxels)`, `uncompressed_size`
- Shard-level aggregates: total labels, max labels/chunk, mean labels/chunk

### 3. Neuroglancer output shard sizes — actual tmpfs consumption

Location: `gs://flyem-male-cns/v0.11/segmentation/s{0..9}/`

These are the actual neuroglancer precomputed shard files produced by the previous export. Each file's size is exactly what was on tmpfs during processing (since local-disk staging writes one shard file, then uploads it). Cross-referencing with the corresponding DVID Arrow shard by spatial coordinates gives us:

```
(arrow_size, chunk_count, label_profile) → actual_output_shard_size
```

### 4. Manifests — which shards went to which tiers

Location: `gs://flyem-male-cns/dvid-exports/mCNS-98d699/segmentation/manifests/`

Per-tier, per-scale shard assignments from the v0.11 export:

| Tier | Scale 0 | Scale 1 | Scale 2 | Scale 3 | Scale 4 | Scale 5+ |
|------|---------|---------|---------|---------|---------|----------|
| 4 GiB | 4,090 | 801 | 208 | 52 | 9 | 9 |
| 8 GiB | 3,355 | 535 | 109 | 22 | 5 | 1 |
| 16 GiB | 14,549 | 2,028 | 107 | 23 | 8 | 2 |
| 24 GiB | 0 | 0 | 182 | 13 | 0 | 1 |
| 32 GiB | 0 | 0 | 0 | 13 | 3 | 0 |

## Proposed approach

### Phase 1: Build shard profiles (new script)

Create `scripts/profile_shards.py` that:

1. For each Arrow shard in SOURCE_PATH, read the Arrow metadata to extract per-chunk:
   - `len(labels)` and `len(supervoxels)` (from the Arrow list columns — just lengths, no decompression needed)
   - `uncompressed_size`
2. Compute per-shard aggregates:
   - `total_labels = sum(len(labels) for each chunk)`
   - `max_labels_per_chunk = max(len(labels) for each chunk)`
   - `mean_labels_per_chunk`
   - Same for supervoxels
3. Write a shard profile JSON to `SOURCE_PATH/profiles/s{scale}/{shard_name}.json` (or a single consolidated file per scale)

This runs during `pixi run export` phase 1 (after scanning Arrow files, before assigning tiers) and is cached for reuse.

### Phase 2: Correlate with actual output sizes

Implemented in `scripts/analyze_memory.py` (`pixi run analyze-memory`).

Maps DVID shards to neuroglancer output shards using TensorStore's compressed Z-index
(not a standard Morton code — the compressed variant only interleaves bits for dimensions
that still have significant bits at each level, matching `EncodeCompressedZIndex`).

For each matched shard, builds a correlation row:
`(scale, shard_name, arrow_bytes, chunk_count, total_labels, total_sv, ng_output_bytes)`

Outputs:
- `analysis/v011_shard_memory.csv` — full dataset
- Console report with per-scale regression, tier comparison, and suggested formula

### Phase 3: New memory formula

Replace the current `KB_PER_CHUNK` lookup with a label-aware estimate:

```python
# Instead of: shard_gib = chunk_count * KB_PER_CHUNK[scale] / (1024 * 1024)
# Use: shard_gib = f(chunk_count, mean_labels_per_chunk, scale)
```

The function `f` would be derived from the Phase 2 regression. The hypothesis is that compressed_segmentation output size scales with the number of distinct labels in the 8×8×8 sub-blocks, not just the chunk count — but this needs to be validated empirically by correlating actual output shard sizes with per-shard label profiles (Phase 2).

### Phase 4: Integration into export pipeline

1. `pixi run export` phase 1: scan Arrow files AND compute shard profiles (cached)
2. `assign_tiers()` uses the new label-aware formula with shard profiles
3. Profile data is written to GCS alongside manifests for reuse on retries

## Memory model

The total memory for processing a DVID shard is:

```
memory = arrow_in_ram + output_shard_on_tmpfs + python_baseline + safety_margin
```

Where:
- `arrow_in_ram` ≈ Arrow file size (known from `gsutil ls`)
- `output_shard_on_tmpfs` = the neuroglancer shard file(s) written by TensorStore — this is the component that needs the label-aware model
- `python_baseline` ≈ 1.5 GiB (stable across all tiers)
- `safety_margin` = 1.3× on total (sufficient given the other components are empirically calibrated)

The RMW overhead (previously 1.3× on shard) is negligible with batched commits (BATCH_SIZE=100) because only one batch's worth of data is in flight during each commit cycle, not the full shard.

## Expected outcome

With a label-aware formula calibrated from v0.11 production data:
- Most tier-16gi shards (scale 0/1 with low label density) would move to tier-8gi
- Tier-24gi shards would move to tier-16gi
- The export bottleneck shifts from one overloaded tier to more even distribution
- Total wall time drops from ~1h18m to closer to ~30-40m
- Per-task cost drops (lower memory tiers cost less per second)
