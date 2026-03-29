# Export Optimization: S0-Only DVID Export + Cloud Run Downres

This document analyzes the per-scale throughput degradation observed in the mCNS
DVID `export-shards` and makes the case for exporting only scale 0 from DVID,
then generating scales 1–9 via TensorStore downsampling on Cloud Run.

## 1. The Problem: Dramatic Per-Scale Throughput Degradation

The mCNS export log shows that DVID's `export-shards` throughput degrades severely
at higher scales. The overall average of 12,962 blocks/sec reported in
`mCNS-ExportAnalysis.md` masks a 12× per-scale divergence:

| Scale | Blocks | Throughput (blocks/sec) | Relative | Time spent |
|-------|--------|------------------------|----------|------------|
| 0 | 510.5M | 11,570 | 1.00× | ~8h 16m (0 → 510M) |
| 1 (early Z) | ~20M | 7,650 | 0.66× | |
| 1 (mid Z) | ~20M | 8,930 | 0.77× | |
| 1 (late Z) | ~27M | 5,130 | 0.44× | |
| **1 total** | **66.6M** | **~6,100 avg** | **0.53×** | **~3h** |
| 2 | 8.8M | 2,810 | 0.24× | ~52m |
| 3 | 1.2M | 1,482 | 0.13× | ~13m |
| 4+ | 0.2M | 979 | 0.08× | ~4m |

**Measured from two concurrent DVID export instances** (identical chunk coordinates,
slightly different timestamps — likely two replicas or a resumed run).

### Why lower scales are slower

Lower-resolution scales have **more labels per block** because each block covers
more physical space. This means:

- **Larger DVID blocks**: Mean compressed block size grows from ~5 KB (scale 0)
  to ~265 KB (scale 5) — a 53× increase. Each block requires more zstd
  compression work and more I/O.
- **Larger label lists**: The `labels` and `supervoxels` fields in the Arrow
  schema are `list<uint64>`. At lower scales, blocks intersect more bodies, so
  these lists grow, increasing per-block Arrow serialization overhead.
- **Agglomerated label lookup**: The shard writer goroutines look up agglomerated
  labels from DVID's in-memory versioned label mapping for every block. More
  labels per block = more lookups per block.
- **Fewer shards, less parallelism**: Scale 0 has 21,994 shard writers running
  concurrently; scale 2 has 606; scale 4 has 25. The 50 `chunkHandler`
  goroutines feed into fewer and fewer shard writers, reducing pipeline
  parallelism.
- **Within-scale degradation**: Even within scale 1, throughput drops from
  ~8,900 blocks/sec at mid-Z to ~5,100 at late-Z. This likely reflects
  increasing block density (more label boundaries) deeper into the volume, plus
  potential Badger DB key-range effects.

## 2. Time Budget: Current vs S0-Only

### Current pipeline (all scales from DVID)

| Phase | Duration | Notes |
|-------|----------|-------|
| DVID export (s0–s9) | 12h 35m | Bottleneck |
| GCS upload (4.3 TB) | ~1h | Over Janelia–Google interconnect |
| Cloud Run conversion | 1h 18m | All scales in parallel |
| **Total** | **~15h** | |

### Proposed: S0-only export + Cloud Run downres

| Phase | Duration | Notes |
|-------|----------|-------|
| DVID export (s0 only) | **~8h 16m** | 510.5M blocks at ~11,570/sec |
| GCS upload (2.84 TB) | **~40m** | 66% of current upload |
| Cloud Run s0 conversion | ~1h | Same as current s0 portion |
| Cloud Run downres s1 | ~15–30m | TensorStore `ts.downsample` + `ts.copy` |
| Cloud Run downres s2–s9 | ~10–20m | Each scale 8× fewer voxels, highly parallel |
| **Total** | **~10–11h** | |

### Time savings breakdown

| Component | Current | Proposed | Savings |
|-----------|---------|----------|---------|
| DVID export | 12h 35m | 8h 16m | **4h 19m** (34%) |
| GCS upload | ~1h | ~40m | **~20m** |
| Cloud Run | 1h 18m | ~1h 30m | -12m (downres adds time) |
| **Total** | **~15h** | **~10–11h** | **~4–5h (27–33%)** |

The DVID export savings alone (4h 19m) dwarf the added Cloud Run downres time.

## 3. Cloud Run Downres: Already Implemented

The worker already has a complete `downres_scale()` implementation
(`src/worker.py:552–594`) using TensorStore's native downsampling:

```python
downsampled = ts.downsample(source, [2, 2, 2, 1], "mode")
ts.copy(downsampled, dest).result()
```

- Uses majority-vote mode — correct for uint64 segmentation labels
- Reads scale N-1 from the destination volume on GCS
- Writes scale N to the destination volume
- Activated via `DOWNRES_SCALES` env var or `--downres` CLI flag
- Already integrated into the worker's two-phase execution (Phase 1: shards,
  Phase 2: downres)

### Sequential dependency

Scale N+1 requires scale N to be fully written. This creates a sequential chain:

```
s0 conversion (parallel) → s1 downres → s2 downres → ... → s9 downres
```

However, each downres step processes 8× fewer voxels than the previous scale,
so the chain converges quickly:

| Scale | Voxels (relative to s0) | Estimated downres time |
|-------|------------------------|----------------------|
| 1 | 1/8 | 15–30m |
| 2 | 1/64 | 2–5m |
| 3 | 1/512 | <1m |
| 4–9 | <1/4096 | seconds each |

The total sequential downres chain (s1 through s9) should take **20–40 minutes**
on Cloud Run, far less than the 4h 19m saved on the DVID side.

## 4. Label Fidelity: DVID vs TensorStore Majority Vote

The previous recommendation in `mCNS-ExportAnalysis.md` §6 noted that "using
DVID's downres guarantees the neuroglancer volume exactly matches what DVID
serves." This is worth examining:

- **DVID's downres** uses majority vote on 2×2×2 neighborhoods, computed
  incrementally as mutations occur (`datatype/common/downres/`).
- **TensorStore's `ts.downsample` with `"mode"`** also performs majority vote on
  2×2×2 neighborhoods.

The algorithms are equivalent for a static dataset. Differences could arise only
if:
1. DVID's incremental downres has not been fully propagated (stale downres) —
   in which case the TensorStore result is actually *more* correct since it
   downsamples from the latest scale 0 data.
2. Tie-breaking differs (when a 2×2×2 block has no majority label). This is an
   edge case that does not affect practical neuroglancer visualization.

**For an export of a finalized dataset, TensorStore's downres from scale 0 is at
least as correct as DVID's pre-computed downres.**

## 5. Additional Benefits

### Reduced DVID server load
Exporting only scale 0 means the DVID server's Badger DB scan covers only the
scale 0 key range, avoiding the slower lower-scale key ranges entirely. This
frees the server sooner for other operations.

### Reduced GCS storage for source data
Only 2.84 TB of Arrow files (scale 0) instead of 4.30 TB (all scales) — a 34%
reduction in source bucket storage.

### Simpler DVID export configuration
No need to configure `num_scales` in the export spec. Export scale 0 only,
and the pipeline handles the rest.

### Overlap-friendly
The S0-only approach composes well with the "streaming upload" optimization
(start uploading and processing shards while DVID is still exporting). Since
scale 0 shards complete in ZYX strip order, Cloud Run workers can begin
converting early shards immediately, and the downres chain can start as soon
as scale 0 is fully written.

## 6. Revised Recommendation

The original recommendation ("time savings don't justify the added complexity")
was based on an assumption of uniform export throughput. The measured 12×
throughput degradation at lower scales changes the calculus:

| Factor | Original analysis | Updated with per-scale data |
|--------|------------------|---------------------------|
| DVID time saved | "~2.8 hours" | **4h 19m** (measured) |
| Added complexity | "Cloud Run would need to compute downres" | Already implemented |
| Sequential dependency | Noted as a concern | ~20–40m total, negligible vs savings |
| Label fidelity | "Guarantees match with DVID" | Equivalent for finalized datasets |

**Recommendation: Export only scale 0 from DVID and generate scales 1–9 via
TensorStore downsampling on Cloud Run.** This reduces total pipeline time from
~15 hours to ~10–11 hours with no code changes required — only configuration:

```bash
# DVID export spec: set num_scales=1 (or omit lower scales)
# Cloud Run:
pixi run export --scales 0 --downres 1,2,3,4,5,6,7,8,9
```

## 7. Open Questions

1. **Downres parallelism**: Currently each worker can downres one scale. Could
   we launch a dedicated Cloud Run job per scale in the chain (s1 job waits for
   s0 completion, s2 job waits for s1, etc.)? This would decouple downres from
   shard processing tasks.

2. **Partial overlap**: Can we start downres for scale 1 before *all* of scale 0
   is written? TensorStore reads from the destination volume — if scale 0 shards
   are written shard-by-shard, we might be able to downres completed regions
   incrementally. This would require careful coordination but could reduce the
   sequential gap.

3. **Validation**: Before production use, run a comparison: export all scales
   from DVID (current method) and export s0 + downres s1–s9 (proposed method),
   then diff the outputs at each scale to confirm equivalence.

