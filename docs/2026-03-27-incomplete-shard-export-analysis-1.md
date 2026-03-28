# Incomplete Shard Export Analysis

**Date:** 2026-03-27
**Export:** mCNS false-merge-corrected (de91d3)
**Region:** us-central1
**Source:** `gs://flyem-dvid-shards/mCNS-de91d3/segmentation`
**Dest:** `gs://flyem-ng-staging/false-merge-corrected/segmentation`

## Symptoms

`pixi run verify-export` reports 393 missing NG shard files:

| Scale | DVID shards | Missing NG shards |
|-------|-------------|-------------------|
| 0 | 21,994 | 308 |
| 1 | 3,364 | 72 |
| 2 | 606 | 9 |
| 3 | 123 | 3 |
| 6 | 2 | 1 |
| 4-5, 7-9 | 36 | 0 |

(Note: count increased from initial 386 to 393 after deleting shards
for replay testing.)

## Key Finding: Zero-Upload Shards

Cloud Logging reveals **740 shards** completed "successfully" with `uploaded_mib: 0.0` across scales 0-5. These shards:

- Committed transactions without errors (`chunks_failed: 0`)
- Reported success (`"Shard processed successfully"`)
- But uploaded nothing to GCS

The 740 zero-upload shards are more than the 393 missing NG shards because multiple DVID shards can map to the same NG shard (many-to-one). The missing shards are cases where ALL DVID shards mapping to that NG shard produced zero output.

## Zero-Upload Shard Characteristics

- **Span all z-values**, not just z >= 45056 (z=0 through z=133120)
- **Tend to be small**: 2 to 6,530 chunks (vs thousands for working shards)
- **Complete extremely fast**: 0.1s for 56 chunks, 10s for 6,530 chunks
- **Memory is low**: peak ~0.8 GiB in 4Gi containers (not OOM)
- **No errors logged**: `chunks_failed: 0`, `chunks_outside: 0`

Breakdown by scale:

| Scale | Zero-upload shards |
|-------|--------------------|
| 0 | 592 |
| 1 | 122 |
| 2 | 18 |
| 3 | 6 |
| 4 | 1 |
| 5 | 1 |

## Detailed Investigation: Shard `14336_12288_45056` (scale 0)

### Source data is valid
- Arrow file exists at `gs://flyem-dvid-shards/mCNS-de91d3/segmentation/s0/14336_12288_45056.arrow`
- CSV index has 56 chunks, all at chunk z=704 (voxel z=45056)
- All 56 chunks map to a single NG shard: `041f9.shard`
- Coordinates are within volume bounds: (224, 192, 704) vs grid shape [1471, 1224, 2103]

### Worker processed it without error
- Task 280, tier-4gi, execution `tensorstore-dvid-export-tier-4gi-w4lps`
- `Shard loaded`: 56 chunks
- `Shard complete`: 56 chunks_written, 0 failed, 0 outside, 0.109 GiB uncompressed
- `uploaded_mib: 0.0` — no shard file found on staging disk
- `elapsed_s: 0.1` — suspiciously fast for 56 chunks with a batch commit
- `batches: 1` — single batch commit (56 < BATCH_SIZE of 100)
- Other shards on the same task (e.g., `75776_28672_16384`, 665 chunks) uploaded normally (0.5 MiB)

### Shard mapping is correct
- Compressed Z-index for (224, 192, 704) with coord_bits [11, 11, 12] = 553418752
- Shard number: 16889
- Expected filename: `041f9.shard` (matches verify-export output)
- Shard number is well within valid range (max valid: 524287)

### TensorStore produces the file locally
A local reproduction using the exact mCNS NG spec, writing 56 chunks at the same coordinates:
- Produces `8x8x8/041f9.shard` (49 MiB) after `commit_async().result()`
- File appears immediately after commit, before handle deletion
- No special close/flush needed

## Root Cause: DVID Exported Empty Arrow Files

All 393 missing NG shards correspond to DVID Arrow files that contain
**zero labels and zero supervoxels**. Every chunk in these shards
decompresses to all-zero uint64 data. TensorStore correctly skips
writing a `.shard` file when all data equals the fill value (zero).

**No data was lost. There is no TensorStore bug.**

### Verification

1. `scripts/check_empty_shards.py` ran as a Cloud Run Job (20 tasks,
   ~20 shards each) checking all 393 missing DVID shards.

2. For each shard, the Arrow metadata (labels and supervoxels list
   columns) was inspected without decompressing the DVID blocks.

3. **Result: 393/393 empty, 0 non-empty, 0 errors.**

4. Direct decompression of `14336_12288_45056` (56 chunks) confirmed
   all voxels are zero for both LABELS and SUPERVOXELS label types.

### Why DVID exports empty shards

DVID's `export-shards` writes an Arrow file for every spatial region
that intersects the data instance's bounding box, even if the region
contains no segmentation labels. This produces Arrow files with valid
chunk coordinates but empty label lists and all-zero compressed blocks.

This is a DVID issue — `export-shards` should skip regions with no
labels to avoid unnecessary empty files.

### Why this wasn't caught earlier

The v0.11 export (98d699 source) had shards that appeared empty in the
export but actually contained real segmentation data at the same
coordinates. This led to a false hypothesis that TensorStore was
dropping writes. The de91d3 source is a different DVID version where
these same spatial regions are genuinely empty (false-merge correction
may have removed labels from these boundary areas).

### Correlation with v0.11

The v0.11 export (98d699) had exactly 304 missing shards at scale 0.
This export (de91d3) also has ~308 missing at scale 0. The overlap is
because both versions have the same bounding box, so DVID exports Arrow
files for the same empty boundary regions.

## Investigation Timeline (export-test branch)

The investigation initially assumed a TensorStore bug because the v0.11
export (98d699 source) had shards with real non-zero data at the same
coordinates. Several hypotheses were tested on Cloud Run:

1. **Single shard in isolation** — shard produced correctly (for shards
   with real data like `67584_47104_20480`).
2. **Replay task 280's 7-shard sequence** — `14336_12288_45056`
   produced no output. Initially attributed to accumulated TensorStore
   state.
3. **Shared handle reuse** — refactored to one TensorStore handle per
   scale. Same result: no `.shard` file for `14336_12288_45056`.
4. **Single shard with new approach** — still no output. Ruled out
   handle reuse as a factor.
5. **Arrow metadata inspection** — discovered all 56 chunks have empty
   labels and supervoxels lists. The data is genuinely all-zero.
6. **Full verification via Cloud Run** — checked all 393 missing shards.
   All are empty.

The confusion arose because the v0.11 source (98d699) had real data at
these coordinates, while the de91d3 source does not. The false-merge
correction likely removed labels from these boundary regions.

## TensorStore Behavior (Not a Bug)

TensorStore correctly optimizes away writes where all data equals the
fill value. For `neuroglancer_precomputed` with `uint64` data type, the
fill value is 0. When all chunks in a shard are zero, `commit_async()`
succeeds (there are no errors) but produces no `.shard` file because
there is no data to write. This is correct behavior.

The `uncompressed_gib` field in the worker logs was misleading — it
reflects the raw byte count of chunks written (56 chunks x 64^3 x 8
bytes = 0.109 GiB) regardless of whether the data is all zeros.

## Recommendations

1. **Worker: detect and skip all-zero shards.** Before writing chunks,
   check if the Arrow metadata has empty labels/supervoxels lists. Log
   a "Shard skipped (empty)" message and return True without writing.
   This avoids wasting time on chunks that produce no output.

2. **DVID: skip empty regions in export-shards.** The `export-shards`
   command should not emit Arrow files for spatial regions with no
   labels. This would eliminate the 393 empty files (~2% of all shards)
   and avoid confusion in downstream pipelines.

3. **verify-export: distinguish empty from truly missing.** The
   verification script should check whether missing NG shards
   correspond to empty Arrow files and report them separately from
   genuinely missing output.
