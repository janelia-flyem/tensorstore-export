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

### Current approach: tier-based memory partitioning

The pipeline uses a memory estimation formula to assign each shard to the smallest
Cloud Run memory tier that can handle it. The formula accounts for:

- **Arrow file in RAM**: loaded by BRAID's `ShardReader` (~1x file size)
- **Output shard on tmpfs**: Cloud Run Gen 2 uses in-memory filesystem, so the
  neuroglancer shard file being built consumes memory (chunks x compressed KB/chunk)
- **TensorStore RMW overhead**: ~1.3x shard size during batched read-modify-write commits
- **Python baseline**: ~1.5 GiB for runtime, libraries, decompression buffers
- **Safety margin**: 1.3x on the total

```
memory_gib = (arrow_gib + 1.3 * shard_on_tmpfs_gib + 1.5) * 1.3
```

Cloud Run CPU coupling constraints determine available tiers:

| Tier | CPU | Cloud Run constraint |
|------|-----|---------------------|
| 4 GiB | 2 | CPU=2 max 8 GiB |
| 8 GiB | 2 | CPU=2 max 8 GiB |
| 16 GiB | 4 | CPU=4 max 16 GiB |
| 24 GiB | 6 | CPU=6 max 24 GiB |
| 32 GiB | 8 | CPU=8 max 32 GiB |

### mCNS v0.11 production results (March 2026)

All 26,125 shards completed with **zero errors** and **zero OOM failures**.

| Tier | Shards | Tasks | Wall time | Avg memory | Max memory | Avg time/shard | Max time/shard |
|------|--------|-------|-----------|------------|------------|----------------|----------------|
| 4 GiB | 5,169 | 5,000 | 23m | 0.5 G | 1.2 G | 12s | 187s |
| 8 GiB | 4,027 | 4,027 | 24m | 0.8 G | 3.0 G | 89s | 413s |
| 16 GiB | 16,717 | 2,500* | 1h 18m | 1.3 G | 4.8 G | 391s | 933s |
| 24 GiB | 196 | 100 | 53m | 3.2 G | 8.7 G | 868s | 1,249s |
| 32 GiB | 16 | 16 | 52m | 6.0 G | 9.5 G | 1,431s | 1,867s |
| **Total** | **26,125** | **14,143** | **1h 18m** | | | | |

\* 5,000 tasks assigned but parallelism capped at 2,500 due to Cloud Run quota.

Key observations:
- **No tier exceeded 50% of its memory budget** (max 9.5 G in the 32 GiB tier).
  The memory formula is conservative, which is the right trade-off for avoiding OOM.
- **Tier-4gi and tier-8gi finished in ~24 minutes** despite having 9,196 shards,
  because their shards are small (mostly scale 0 edge shards with few chunks).
- **Tier-16gi dominated wall time** at 1h 18m with 16,717 shards (64% of all shards).
  These are the bulk scale 0 and scale 1 shards with 20K–33K chunks each.
- **Batched transactions worked well**: avg 309 batch writes per shard in tier-16gi
  (BATCH_SIZE=100, so ~30,900 chunks per shard on average).

### Earlier run with incorrect block size

A previous export run (with `compressed_segmentation_block_size: [64,64,64]` instead
of the correct `[8,8,8]`) had 580/4,027 tier-8gi task failures (~14% OOM rate). The
block size mismatch caused TensorStore to produce larger output chunks, pushing memory
over budget. After fixing to `[8,8,8]`, the same tier completed with zero failures
and max memory of only 3.0 G.

---

## 4. Multiscale Handling

DVID exports **all scales simultaneously** in a single database scan. The `readBlocksZYX` function iterates through the key-value store in ZYX order across all requested scales (controlled by the `num_scales` parameter). Each block key encodes its scale level, so the scan naturally produces blocks at scales 0, 1, 2, etc.

The downsampled blocks at scale 1+ are pre-computed by DVID's label downres system (`datatype/common/downres/`), which uses a majority-vote algorithm on 2x2x2 neighborhoods. These are stored in the database alongside scale 0 blocks and exported in the same pass. This means scale 1+ Arrow shards already contain correctly downsampled segmentation — **no computation is needed** to produce lower-resolution scales.

The pipeline processes all 10 scales in a single batch. Each worker receives a manifest listing (scale, shard) pairs. The worker opens the correct scale of the destination neuroglancer volume based on the shard's scale index, using the sharding parameters from the neuroglancer spec (`NG_SPEC` env var, base64-encoded).

---

## 5. End-to-End Pipeline Timing

The full pipeline from DVID to a viewable neuroglancer precomputed volume has three phases:

### Phase 1: DVID Export (12h 35m)

DVID scans the Badger database in ZYX order across all scales, compresses blocks with zstd, and writes Arrow streaming shards to local disk. This is I/O-bound on the database scan. Throughput: 12,962 chunks/sec (376 MB/sec uncompressed).

### Phase 2: Upload to GCS

The 4.3 TB of Arrow + CSV files must be uploaded from the DVID server to a GCS bucket. Actual upload was performed via `gsutil -m rsync` over the Janelia–Google Cloud interconnect.

### Phase 3: Cloud Run Processing (actual: 1h 18m)

The pipeline uses `pixi run precompute-manifest` to scan all Arrow files, estimate
memory per shard, assign shards to tiers, and distribute across tasks. Then
`pixi run export` launches one Cloud Run job per tier in parallel.

| Parameter | Value |
|-----------|-------|
| Total shards (all scales) | 26,125 |
| Total Cloud Run tasks | 14,143 |
| Memory tiers | 4, 8, 16, 24, 32 GiB |
| Wall clock (longest tier) | 1h 18m (tier-16gi) |
| Errors | 0 |
| OOM failures | 0 |

The tier-4gi and tier-8gi tiers finished in ~24 minutes. The bottleneck was
tier-16gi (16,717 shards across 5,000 tasks), which ran for 1h 18m. All tiers
run in parallel, so the total wall time equals the slowest tier.

### Total pipeline time (actual)

| Phase | Duration |
|-------|----------|
| DVID export | 12h 35m |
| Upload to GCS | ~1 hr |
| Cloud Run processing | 1h 18m |
| **Total** | **~15 hr** |

---

## 6. Optimization: Export Only Scale 0?

Since Cloud Run processing is highly parallel and the conversion is fast (1h 18m),
the bottleneck is the DVID export (12h 35m) and GCS upload (~1h).

Exporting only scale 0 would save ~2.8 hours of export time and ~1.5 TB of upload,
but introduces complexity (Cloud Run would need to compute downres) and a
sequential dependency (scale N+1 depends on scale N being fully written).

**Recommendation**: Use DVID's pre-computed downres shards (current approach). The
time savings don't justify the added complexity, and using DVID's downres guarantees
the neuroglancer volume exactly matches what DVID serves.

### A better optimization target

The largest time savings would come from **overlapping the DVID export with the GCS
upload and Cloud Run processing**. DVID's export-shards writes shard files as strips
complete — early shards are available on disk hours before the export finishes. A
concurrent upload process could stream completed shards to GCS incrementally, allowing
Cloud Run workers to start processing while DVID is still exporting.

## 7. Pipeline Operation

### Commands

```bash
pixi run deploy               # Build Docker image, push to GCR
pixi run precompute-manifest   # Scan shards, assign tiers, write manifests to GCS
pixi run export                # Launch Cloud Run jobs (one per tier)
pixi run export --dry-run      # Preview without launching
pixi run export-status         # Monitor progress (task counts, memory, timing)
pixi run export-errors         # Scan logs for chunk/shard errors
pixi run find-failed           # Identify failed shards for retry
```

### Monitoring

`pixi run export-status` queries Cloud Run execution status and Cloud Logging for
structured events. It shows:
- Per-tier task completion counts
- Per-tier shard and chunk completion (with totals from `summary.json`)
- Memory usage (avg/max) and timing stats from completed shards
- In-flight shard progress (chunks written, memory, elapsed time)
- Grand progress bar (if `summary.json` exists from manifest precomputation)

### Retry workflow

If tasks fail (OOM, transient errors), use `pixi run find-failed` to identify
incomplete shards and create retry manifests at a higher memory tier:

```bash
pixi run find-failed -- --retry-tier 16   # creates retry manifest at 16 GiB
pixi run export --manifest-dir manifests-retry --job-suffix retry
```

### Key configuration

All configuration lives in `.env` (see `.env.example`). The neuroglancer volume spec
JSON (`NG_SPEC_PATH`) is the single source of truth for volume geometry, sharding
parameters, and `compressed_segmentation_block_size`.

---

## 8. Post-Mortem: Excessive GCS Replication Charges (March 20–23, 2026)

### Incident summary

During the mCNS v0.11 export runs (March 20–23), the GCP billing SKU "Network Data
Transfer GCP Replication within Northern America" (AED0-3315-7B11) accrued significant unexpected charges. The cost dropped 99.3% after deploying the local-disk staging
fix on March 22.

| Date | Replication (GiB) | Event |
|------|------------------|------|
| March 20 | 809,654 |First production export runs begin |
| March 21 | 2,150,962 | Heaviest export day (full parallelism, tier-based manifests) |
| March 22 | 210,440 | `.result()` fix at 19:11; **local-disk staging at 22:46** |
| March 23 | ~15,769 | Running with local-disk staging |

### Root cause: TensorStore read-modify-write amplification × multi-region replication

Two factors compounded to produce catastrophic costs:

**1. TensorStore's shard-level read-modify-write (RMW) against GCS.**
Before the local-disk staging change (`184e6ff`), the worker opened TensorStore with
the GCS destination as the kvstore:

```python
"kvstore": self.config.dest_path   # "gs://flyem-male-cns/..."
```

TensorStore's sharded neuroglancer_precomputed driver performs a full shard-level RMW
on every transaction commit (see `docs/EfficientWrites.md` for the source-level
explanation). With memory-pressure-based batching, a shard with N chunks and B batch
commits produced:

- **Total bytes written to GCS** ≈ final_shard_size × (B+1)/2
- **Total bytes read from GCS** ≈ final_shard_size × (B-1)/2

For a typical tier-16gi shard with ~30,000 chunks and ~300 batch commits (memory
pressure triggering a commit roughly every 100 chunks), the write amplification
factor was approximately **150×** relative to the final shard size.

**2. Multi-region GCS bucket replication.**
The destination bucket `gs://flyem-male-cns` is configured as **multi-region (US)**.
Every byte written to a multi-region bucket is replicated across US data centers.
The Cloud Run tasks were running in `us-central1`, but the writes were replicated to
other US regions by GCS to satisfy the multi-region durability guarantee.

The SKU "Network Data Transfer GCP Replication within Northern America" charges
**$0.02/GiB** for this cross-region replication — applied to every byte of every
amplified RMW write.

**3. Additional amplifiers.**

- **Soft delete** was enabled on the bucket (7-day retention). Each intermediate
  shard overwrite from each RMW commit created a retained soft-deleted version.
  Post-incident bucket inspection found **4,572,931 soft-deleted objects totaling
  3,097 TiB (3.1 PiB)** — 881× the 3.52 TiB of live data — adding significant
  one-time storage charges over the retention window. See "Soft-delete storage
  impact" below.
- **The `.result()` bug** (fixed at `7169e8f`, March 22 19:11): before this fix,
  `ts.TensorStore.write()` futures were not `.result()`'d, causing ~31% data loss at
  scale 0 and complete data loss at scales 5–9. Failed exports were rerun,
  multiplying GCS write traffic.
- **Rapid iteration**: 40+ commits across March 20–22 fixed bugs and redeployed,
  each deployment reprocessing shards that had already been partially written.

### The fix: local-disk staging (`184e6ff`, March 22 22:46 EDT)

The fix rewrote `process_shard()` to write to local disk instead of directly to GCS:

```python
# Before: TensorStore writes directly to GCS (every commit = shard RMW on GCS)
"kvstore": self.config.dest_path        # "gs://flyem-male-cns/..."

# After: TensorStore writes to local staging (every commit = shard RMW on local disk)
"kvstore": {"driver": "file", "path": staging_dir}   # "/mnt/staging/s0_abc123"
```

All RMW cycles now happen locally (free). Only one final upload per shard to GCS via
`_upload_shard_files()`. This reduces GCS write traffic from `O(B × shard_size)` to
`1 × shard_size` — eliminating the write amplification entirely.

### Cost breakdown: what could have prevented this

| Mitigation | Notes |
|-----------|-------|
| Local-disk staging (what was deployed) | Eliminates RMW write amplification against GCS |
| Single-region bucket (us-central1) | No cross-region replication; this specific SKU becomes $0 |
| Both combined | Belt and suspenders |

Using a single-region bucket pinned to the same region as Cloud Run would have
avoided the replication SKU entirely. However, the RMW amplification would still
have incurred elevated GCS **operation** costs (Class A writes at $0.05/10K ops)
and potentially egress charges. The local-disk staging fix is the more fundamental
solution because it eliminates the amplified I/O pattern regardless of bucket
configuration.

### Soft-delete storage impact

The destination bucket `gs://flyem-male-cns` had soft delete enabled with the
default 7-day retention (604800 seconds). Each RMW overwrite of a shard file turned
the previous version into a soft-deleted object, retained and billed at multi-region
standard storage rates ($0.026/GB/month) until hard deletion.

### Best practices

**1. Never let TensorStore write sharded format directly to remote storage.**
TensorStore's sharded neuroglancer_precomputed driver performs a shard-level
read-modify-write on every transaction commit. Writing directly to GCS turns every
commit into a full shard download + upload cycle. Always stage to local disk and
upload the finished shard file once.

**2. Pin your GCS bucket region to match your compute region.**
Multi-region buckets (US, EU, ASIA) incur cross-region replication charges on every
write. If all compute runs in a single region (e.g., Cloud Run in `us-central1`),
use a single-region bucket in that same region. Multi-region only provides value
when readers are distributed across regions.

**3. Account for soft-delete costs on buckets receiving repeated overwrites.**
Soft delete retains every overwritten version for the retention period (default 7
days), billed at the same storage rate as live data. In this incident, the RMW
pattern created 4.57M retained versions totaling 3.1 PiB — an additional required storage charge
on top of the significant replication bill. For write-heavy pipelines under development, consider
disabling soft delete or reducing the retention period during initial runs, then
re-enabling it once the pipeline is stable.

More practically, we should separate a staging bucket from the distribution bucket. The staging bucket should be single zone and could enable a number of properties to improve latency and throughput like Hierarchical Namespace. The distribution bucket could remain multi-region and enable soft delete.

**4. Always `.result()` TensorStore write futures inside transactions.**
`ts.TensorStore.write()` returns a `WriteFutures` object. Without calling
`.result()`, the write may not be staged in the transaction before commit — causing
silent data loss. This bug caused the export to be rerun multiple times, amplifying
all the costs above. See the pitfall note in `CLAUDE.md`.

**5. Estimate GCS costs before large exports.**
Back-of-envelope before launching: if writing N shards with B batch commits each to
a multi-region bucket, the replication cost is roughly:

```
cost ≈ N × avg_shard_size × (B+1)/2 × $0.02/GiB
```

**6. Monitor billing daily during large jobs.**
GCP billing data is available with a ~24-hour delay. For multi-day export runs, check
the billing console daily. 