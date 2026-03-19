# mCNS Export Analysis and Cloud Run Pipeline Design

This document analyzes the real mCNS segmentation export from DVID, characterizes the Arrow shard files, evaluates the Arrow IPC format choice, and considers pipeline optimizations for the full DVID → neuroglancer precomputed path.

---

## 1. mCNS Export Shard Characteristics

The male CNS (mCNS) dataset was exported from DVID using the `export-shards` RPC command with the neuroglancer multiscale volume spec from `test_data/mcns-ng-specs.json`. The export produced 26,128 Arrow IPC shard files across 10 scales in 12 hours 35 minutes, totaling 587 million chunks and 4.3 TB on disk.

### Export summary

| | Value |
|-|-------|
| Total chunks exported | 587,173,098 |
| Total shard files | 26,128 |
| Total raw uint64 voxel data | 1,120 TB (1.09 PB) |
| Total DVID block data (pre-zstd) | 16.24 TB |
| DVID block compression ratio | 69× (vs raw uint64) |
| Total zstd-compressed block data | 3.75 TB |
| zstd compression ratio | 4.34× (vs DVID block) |
| **Combined block compression ratio** | **~300× (vs raw uint64)** |
| CSV index files | 0.29 GB |
| Arrow overhead (schema, record headers, per-record metadata) | ~0.55 TB |
| **Total on disk (Arrow + CSV)** | **4.30 TB** |
| Export duration | 12h 35m |
| Throughput | 12,962 chunks/sec |

**A note on compression and what's stored in the Arrow files**: Each 64³ label chunk has three size levels:
- **Full voxels**: 64³ × 8 bytes = 2 MB — the raw uint64 label array in memory
- **DVID block format**: variable (mean 26 KB at scale 0) — the sub-block indexed binary format with label tables and packed bit indices. This is what the export log calls "uncompressed" and what's stored in the `uncompressed_size` Arrow field
- **zstd of DVID block**: variable (mean 5 KB at scale 0) — this is what the export log calls "compressed" and what's stored in the `dvid_compressed_block` Arrow field

Each Arrow record stores: the zstd-compressed DVID block (binary), chunk coordinates (3 × int32), agglomerated label list (list\<uint64\>), supervoxel list (list\<uint64\>), and the pre-zstd size (uint32). The per-record metadata (coordinates, label lists, Arrow framing) adds ~0.55 TB of overhead across 587M records. Separate CSV index files (one per shard) map chunk coordinates to Arrow record numbers for lookup, adding 0.29 GB total.

The 16.24 TB "DVID block data" figure is the pre-zstd size, not the true in-memory voxel representation. The full voxel data for 587M chunks at 2 MB each would be 1.09 PB.

### Per-scale breakdown

| Scale | Volume size | Chunks | Shards | Arrow size | Mean shard | Max shard | zstd ratio |
|-------|------------|--------|--------|-----------|------------|-----------|------------|
| 0 | 94088 × 78317 × 134576 | 510.5M | 21,994 | 2.84 TB | 135 MB | 471 MB | 5.10× |
| 1 | 47044 × 39159 × 67288 | 66.6M | 3,364 | 1.03 TB | 321 MB | 856 MB | 3.13× |
| 2 | 23522 × 19580 × 33644 | 8.8M | 606 | 298 GB | 504 MB | 1.74 GB | 2.45× |
| 3 | 11761 × 9790 × 16822 | 1.2M | 123 | 96 GB | 799 MB | 3.77 GB | 1.91× |
| 4 | 5881 × 4895 × 8411 | 157K | 25 | 36 GB | 1.44 GB | 6.19 GB | 1.59× |
| 5 | 2941 × 2448 × 4206 | 22K | 8 | 12 GB | 1.47 GB | 6.18 GB | 1.54× |
| 6 | 1471 × 1224 × 2103 | 3.3K | 2 | 2.9 GB | 1.47 GB | 2.92 GB | 1.59× |
| 7 | 736 × 612 × 1052 | 828 | 2 | 600 MB | 300 MB | 597 MB | 1.60× |
| 8 | 368 × 306 × 526 | 143 | 2 | 113 MB | 57 MB | 112 MB | 1.49× |
| 9 | 184 × 153 × 263 | 29 | 2 | 18 MB | 9 MB | 18 MB | 1.61× |

The "zstd ratio" column is the compression of zstd over the DVID block format. The DVID block format itself is already a compact representation of the 2 MB raw voxel data (e.g., mean 26 KB at scale 0 — a ~77× reduction from the raw voxels before zstd is even applied).

### Key observations

**zstd compression ratio degrades at lower resolutions.** Scale 0 achieves 5.10× zstd compression over the DVID block format because many blocks are sparse (edge of the brain, large solid background regions with highly compressible DVID blocks). By scale 3 and below, most blocks are dense with many label boundaries, and the DVID block format already has higher entropy, achieving only ~1.6× additional zstd compression.

**Individual shard sizes increase dramatically at lower scales.** While scale 0's largest shard is 471 MB, scale 4 has shards up to 6.19 GB. This is because:
- The shard voxel extent (2048³) is the same at every scale, but each voxel at lower scales covers more physical space
- Fewer, larger shards contain more chunks with denser data
- Mean block size grows from 5 KB (scale 0 compressed) to 265 KB (scale 5 compressed)

**Scale 0 and 1 dominate the data volume** at 2.84 TB + 1.03 TB = 90% of the total 4.30 TB.

**Only 10 scales (s0–s9) were exported** despite requesting 11. The missing scale 10 may be below the minimum shard threshold or a configuration issue (to be investigated).

---

## 2. Arrow IPC Format: Streaming vs File

DVID's `export-shards` writes Arrow IPC **Streaming** format (not the File/Feather V2 format). The two formats carry identical data but differ structurally.

### Streaming format (what DVID writes)

- Records are written sequentially: schema message → record batch → record batch → ...
- **No footer** — the writer doesn't need to know how many records will exist upfront
- Can start writing immediately as blocks arrive from the database scan
- File is valid after every record batch flush; no clean close required
- Reader must scan sequentially from the beginning

### File format (Feather V2)

- Same record batches, plus a **footer** at the end with a table of byte offsets for every record batch
- Enables **random access**: seek directly to record batch N without reading 0 through N-1
- Writer must buffer footer metadata and write it at close time
- File is not valid until footer is written (requires clean close)

### Why streaming is the right choice

**For writing (DVID side)**: Blocks arrive one at a time from the ZYX database scan and are dispatched to shard writer goroutines. Each goroutine appends records immediately with zero buffering overhead. If a writer crashes mid-shard, other shards are unaffected. The streaming format is the natural fit for this append-only pattern.

**For reading (BRAID/Cloud Run side)**: The random-access advantage of the file format would only matter if workers needed to read individual chunks without loading the full shard. But the Cloud Run workflow is:

1. Load entire Arrow shard into memory via `read_all()`
2. Build in-memory CSV index for O(1) chunk lookup by coordinate
3. Process all chunks through TensorStore's `virtual_chunked` driver
4. Release the shard and move to the next one

Since `read_all()` materializes the complete table regardless of format, the streaming format has **no performance disadvantage**. Both formats produce the same in-memory Arrow Table.

The file format would only be preferable if we wanted to:
- Read individual records from very large shards without loading the rest (not our workflow)
- Support concurrent random-access readers on a single shard file (not needed — each worker processes one shard exclusively)

---

## 3. Cloud Run Worker Memory Sizing

The current deployment allocates 4 GB per Cloud Run worker. The memory requirement depends on which scale is being processed:

| Scale | Median shard | Max shard | Peak memory estimate |
|-------|-------------|-----------|---------------------|
| 0 | 135 MB | 471 MB | ~800 MB |
| 1 | 321 MB | 856 MB | ~1.2 GB |
| 2 | 504 MB | 1.74 GB | ~2.1 GB |
| 3 | 799 MB | 3.77 GB | ~4.1 GB |
| 4 | 1.44 GB | 6.19 GB | ~6.6 GB |
| 5+ | 1.47 GB | 6.18 GB | ~6.6 GB |

Peak memory estimate = max shard size + ~200 MB (Python runtime + TensorStore + decompression buffer).

**Recommendation**: Use different worker sizes per scale:
- **Scales 0–1** (25,358 shards, 96.9% of the work): **2 GB** workers
- **Scales 2–3** (729 shards): **4 GB** workers
- **Scales 4–5** (33 shards): **8 GB** workers
- **Scales 6–9** (8 shards): **8 GB** workers (or process on a single machine)

Alternatively, use **4 GB** workers for all scales and accept that scales 4+ would need larger instances. Since scales 4+ have only 33 shards total, this is a negligible cost impact.

---

## 4. Current Multiscale Handling

### What tensorstore-export currently does

The current `worker.py` processes shards at a **single scale** — it reads one shard file, writes the decompressed chunks to the neuroglancer precomputed volume at the shard's spatial offset, and moves on. The `ShardExportDesign.md` describes a **two-step architecture**:

1. **Step 1**: Cloud Run workers ingest all scale 0 shards into the neuroglancer precomputed volume
2. **Step 2**: A separate post-processing job reads the scale 0 data from GCS and generates downsampled scales using TensorStore's `ts.downsample(method='mode')`

### What DVID's export-shards actually produces

DVID exports **all scales simultaneously** in a single database scan. The `readBlocksZYX` function iterates through the key-value store in ZYX order across all requested scales (controlled by the `num_scales` parameter). Each block key encodes its scale level, so the scan naturally produces blocks at scales 0, 1, 2, etc.

The downsampled blocks at scale 1+ are pre-computed by DVID's label downres system (`datatype/common/downres/`), which uses a majority-vote algorithm on 2×2×2 neighborhoods. These are stored in the database alongside scale 0 blocks and exported in the same pass.

This means the scale 1+ Arrow shards already contain correctly downsampled segmentation — **no computation is needed** to produce lower-resolution scales. Each scale's shard files can be ingested directly into the corresponding scale of the neuroglancer precomputed volume.

### The current gap

The `worker.py` doesn't distinguish between scales. The shard files are organized by scale directory (`s0/`, `s1/`, `s2/`, ...) but the worker treats all shards identically — it reads the shard, extracts chunk coordinates, and writes to the destination volume. To support multiscale ingestion, workers need to:

1. Know which scale a shard belongs to (from its directory path)
2. Open the correct scale of the destination neuroglancer volume
3. Use the appropriate shard dimensions for that scale

---

## 5. End-to-End Pipeline Timing

The full pipeline from DVID to a viewable neuroglancer precomputed volume has three phases:

### Phase 1: DVID Export (12h 35m)

DVID scans the Badger database in ZYX order across all scales, compresses blocks with zstd, and writes Arrow streaming shards to local disk. This is I/O-bound on the database scan. Throughput: 12,962 chunks/sec (376 MB/sec uncompressed).

### Phase 2: Upload to GCS

The 4.3 TB of Arrow + CSV files must be uploaded from the DVID server to a GCS bucket.

| Method | Estimated time |
|--------|---------------|
| `gsutil -m rsync` at 1 Gbps | ~10 hours |
| `gsutil -m rsync` at 10 Gbps | ~1 hour |
| `gcloud storage cp` (optimized) at 10 Gbps | ~45 minutes |
| Direct-to-GCS writing (if DVID supported it) | 0 (overlapped with export) |

For a Janelia server with a 10 Gbps link to Google Cloud, the upload is roughly 1 hour. With a 1 Gbps link, it becomes a significant bottleneck.

### Phase 3: Cloud Run Processing

Each worker: download shard from GCS → decompress all chunks → write to neuroglancer precomputed volume on GCS.

| Parameter | Value |
|-----------|-------|
| Total shards (all scales) | 26,128 |
| Time per shard (estimate) | 30–90 seconds |
| With 200 parallel workers | ~26,128 / 200 = 131 shards/worker |
| Estimated Cloud Run time | ~2–3 hours wall clock |

GCS is not an I/O bottleneck — it supports thousands of concurrent readers/writers with no single-object contention.

### Total pipeline time

| Phase | Optimistic (10 Gbps) | Conservative (1 Gbps) |
|-------|----------------------|----------------------|
| DVID export | 12.5 hr | 12.5 hr |
| Upload to GCS | 1 hr | 10 hr |
| Cloud Run processing | 2 hr | 3 hr |
| **Total** | **15.5 hr** | **25.5 hr** |

---

## 6. Optimization: Export Only Scale 0?

### The question

Since Cloud Run processing is highly parallel and GCS I/O is not a bottleneck, would it be better to:
- Export only scale 0 from DVID (saving export time and upload volume for scales 1–9)
- Have Cloud Run workers compute downsampled scales on-the-fly from the scale 0 data already written to GCS

### Analysis

**Savings from exporting only scale 0:**
- DVID export time: saves ~2.8 hours (scales 1–9 produced 76.7M chunks at ~13K/sec)
- Upload volume: saves ~1.5 TB (scales 1–9)
- Upload time: saves ~20 minutes at 10 Gbps, or ~3.3 hours at 1 Gbps

**Cost of Cloud Run downsampling:**
- TensorStore's `ts.downsample(method='mode')` can generate scale N+1 from scale N by reading 2×2×2 neighborhoods and picking the majority label
- Scale 1 has 3,364 shard-sized regions to generate, each reading 8 scale-0 chunks and writing one scale-1 chunk. With 200 workers this completes in minutes
- Scales 2+ are negligible

**However, there are important considerations against this approach:**

1. **Correctness guarantee**: DVID's downres algorithm (`labels.Block.Downres`) and TensorStore's `ts.downsample(method='mode')` must produce identical results. Both use majority-vote, but tie-breaking behavior may differ. Using DVID's pre-computed downres guarantees the neuroglancer volume exactly matches what DVID serves, which is important for scientific reproducibility.

2. **Sequential dependency**: Scale N+1 can only be generated after scale N is completely written. This means Cloud Run would need to process all scale 0 shards → wait → process scale 1 → wait → etc. With DVID-exported shards, all scales can be processed in a single parallel batch.

3. **Marginal savings**: The ~2.8-hour DVID export saving is modest compared to the ~12.5-hour total. The upload saving matters more if network bandwidth is limited.

4. **Simplicity**: Processing pre-computed shards at every scale uses the same code path — just point workers at each scale's directory. Computing downres on-the-fly requires new logic for reading from the neuroglancer volume, downsampling, and writing back — a different and more complex code path.

### Recommendation

**Use DVID's pre-computed downres shards (current approach)** for production pipelines where correctness and simplicity matter. The time savings from skipping lower-scale export don't justify the added complexity and potential for divergence.

**Consider scale-0-only export** for future optimizations if:
- Network bandwidth to GCS is limited (1 Gbps), making the 1.5 TB upload saving significant
- A verified equivalence test confirms DVID and TensorStore downres produce identical output
- The pipeline already needs to be modified for other reasons

### A better optimization target

The largest time savings would come from **overlapping the DVID export with the GCS upload and Cloud Run processing**. DVID's export-shards writes shard files as strips complete — early shards are available on disk hours before the export finishes. A concurrent upload process could stream completed shards to GCS incrementally, allowing Cloud Run workers to start processing while DVID is still exporting later regions. This could reduce the effective total time from 15.5 hours to ~13 hours by hiding the upload latency entirely.
