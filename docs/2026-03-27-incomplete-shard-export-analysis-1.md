# Incomplete Shard Export Analysis

**Date:** 2026-03-27
**Export:** mCNS false-merge-corrected (de91d3)
**Region:** us-central1
**Source:** `gs://flyem-dvid-shards/mCNS-de91d3/segmentation`
**Dest:** `gs://flyem-ng-staging/false-merge-corrected/segmentation`

## Symptoms

`pixi run verify-export` reports 386 missing NG shard files:

| Scale | DVID shards | Missing NG shards |
|-------|-------------|-------------------|
| 0 | 21,994 | 304 |
| 1 | 3,364 | 70 |
| 2 | 606 | 9 |
| 3 | 123 | 3 |
| 4-9 | 38 | 0 |

This matches the exact count from the v0.11 export (304 missing at scale 0 in z-slabs 45056-59392), but the new export is a completely different DVID version (de91d3 vs 98d699).

## Key Finding: Zero-Upload Shards

Cloud Logging reveals **740 shards** completed "successfully" with `uploaded_mib: 0.0` across scales 0-5. These shards:

- Processed real chunk data (non-zero `uncompressed_gib`)
- Committed transactions without errors (`chunks_failed: 0`)
- Reported success (`"Shard processed successfully"`)
- But uploaded nothing to GCS

The 740 zero-upload shards are more than the 386 missing NG shards because multiple DVID shards can map to the same NG shard (many-to-one). The 386 missing are cases where ALL DVID shards mapping to that NG shard produced zero output.

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

## What's Ruled Out

1. **OOM** — peak memory 0.81 GiB in 4Gi container
2. **Coordinate mismatch** — all chunks within volume bounds, mapping verified
3. **TensorStore sharding bug** — local reproduction works correctly
4. **Handle lifecycle** — TensorStore flushes on commit, not on close
5. **Zero/empty data** — 2 MiB/chunk uncompressed, matches expected 64^3 * 8 bytes
6. **Upload code bug** — `os.walk` correctly finds files in subdirectories; other shards on the same task uploaded fine
7. **Staging directory cleanup race** — cleanup is in `finally` block, after upload

## Open Hypotheses

1. **tmpfs flush race condition** — On Cloud Run Gen2 tmpfs, `commit_async().result()` may return before the filesystem has fully materialized the file. Working shards may be large enough that the writes are already flushed by the time the upload walk happens, while small shards (few chunks, fast commits) may hit a race window.

2. **TensorStore internal caching** — TensorStore may cache small shard files in memory and not write them to the file:// kvstore until some threshold or GC trigger. The production behavior might differ from local because of memory pressure or cgroup limits.

3. **Info file issue** — The local staging volume's `info` file is written fresh for each shard. If the file is somehow malformed or truncated on tmpfs, TensorStore might silently produce no output. However, we see no errors logged, and the same info content works for other shards.

4. **Small shard threshold** — TensorStore might have an optimization where it doesn't write shard files below some minimum size. The failing shards tend to be small (few chunks). However, the local test with 56 chunks produces a file.

## Correlation with v0.11

The v0.11 export (98d699) had exactly 304 missing shards at scale 0 in z-slabs 45056-59392. This new export (de91d3) also has 304 missing at scale 0. However, the Cloud Logging analysis shows zero-upload shards across ALL z-values (not just z >= 45056), suggesting the z-slab correlation in v0.11 was coincidental — those just happened to be the shards where no other DVID shard contributed to the same NG shard.

## Local Reproduction Attempts

### Test shard: `67584_47104_20480` (scale 0, 3 chunks, 3 KB Arrow)

Chosen for its tiny size. All 3 chunks map to NG shard `0a8b3.shard`.

**Manual reproduction** (`debug_zero_upload.py`): Downloaded shard from GCS,
replicated the exact write + commit sequence on local tmpfs (`/dev/shm`).
Result: `8x8x8/0a8b3.shard` (1,749 bytes) produced correctly.

**Production code path** (`debug_zero_upload_v2.py`): Ran the actual
`ShardProcessor.process_shard()` method with a mock upload handler on
local tmpfs. Result: shard file produced correctly.

**Conclusion:** The bug does not reproduce locally, even on tmpfs with the
exact same code, data, NG spec, and filesystem type. Something specific to
the Cloud Run Gen2 container environment causes the shard file to not
materialize after `commit_async().result()`.

### TensorStore fill-value hypothesis (DISPROVEN)

TensorStore does not write a shard file when all written data equals the
fill value (zero for uint64). Initial hypothesis: the failing shards contain
only background data. **This is wrong.** Direct inspection of failing shards
shows real segmentation data:

- `67584_47104_20480` (3 chunks): 3-10% non-zero, labels include 979073363,
  742360899, 798220710, 860600797
- `43008_43008_71680` (6 chunks): 6.3% non-zero across all chunks, 2-3
  unique labels per chunk

The user also confirmed no segmentation is visible at these coordinates in
neuroglancer or the original DVID source — suggesting these are real but
sparse regions at tissue boundaries. However, the data IS non-zero and
should produce shard output.

Verified locally: writing 3% non-zero data through the production code path
correctly produces a shard file. The bug is specific to Cloud Run.

## Changes Made

Added diagnostic logging to `_upload_shard_files()` in `src/worker.py`:
when no shard files are found in the staging directory, logs a warning
with the full directory listing. This will reveal on the next deploy
whether the shard file doesn't exist at all vs exists but is being missed.

Also changed `"Shard extends beyond volume"` from WARNING to INFO (expected
edge-of-volume behavior, not actionable).

## Next Steps

- Deploy with diagnostic logging and retry the failing shards
- If the staging dir truly has no shard file: investigate TensorStore's
  behavior in Cloud Run's specific container filesystem / cgroup configuration
- If the staging dir has the file: investigate the `os.walk` behavior
- Test with `file_io_sync: true` in TensorStore context to force synchronous writes
- Consider adding `os.sync()` after `commit_async().result()` as a workaround
