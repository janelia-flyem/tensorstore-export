# Export-Shards Per-Scale Throughput Analysis

This document analyzes the per-scale throughput degradation observed in the mCNS
DVID `export-shards` log, quantifying how block processing rate drops at lower
resolution scales.

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

