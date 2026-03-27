# Export Quality Control Plan

## Problem

The v0.11 mCNS export silently dropped 304 DVID shards (all in z-slabs 45056–59392) containing real segmentation data. These shards were assigned to Cloud Run tasks, but their neuroglancer output shard files were never written — and nobody noticed until months later during the memory formula analysis.

Current gaps:

1. **No post-export reconciliation**: `find_failed_shards.py` compares completed shards from Cloud Logging against source Arrow files, but it depends on log retention and has a 50K entry limit. It doesn't verify that NG shard files actually exist on GCS.
2. **Silent chunk clipping**: When a chunk falls outside `dest.shape` (line 394 of `worker.py`), the worker silently `continue`s with no log. If all chunks in a shard are clipped, the shard appears to succeed (returns `True`) with `chunks_written=0`.
3. **No coordinate validation**: The worker trusts that DVID chunk coordinates match the NG spec volume extents. There's no warning if chunks are outside bounds or if a DVID shard contains chunks that would map to multiple NG shards.
4. **Compressed Z-index correctness**: The DVID-to-neuroglancer shard mapping depends on a compressed Z-index (not a standard Morton code). The `analyze_v011_memory.py` script initially implemented a standard Morton code, producing completely wrong shard numbers. Any coordinate/sharding code must be verified against TensorStore's reference implementation.

## 1. Worker-Level Validation

### 1a. Log out-of-bounds chunks instead of silently skipping

**File**: `src/worker.py`, lines 388–395

Current:
```python
if x1 <= x0 or y1 <= y0 or z1 <= z0:
    continue  # chunk entirely outside volume
```

Change to:
```python
if x1 <= x0 or y1 <= y0 or z1 <= z0:
    chunks_outside += 1
    if chunks_outside == 1:  # log first occurrence with detail
        logger.warning("Chunk outside volume bounds",
                       scale=scale, shard=shard_name,
                       chunk_x=cx, chunk_y=cy, chunk_z=cz,
                       voxel_origin=(x0, y0, z0),
                       volume_shape=tuple(dest.shape[:3]))
    continue
```

And in the "Shard complete" log, include `chunks_outside`:
```python
logger.info("Shard complete", ..., chunks_outside=chunks_outside)
```

This makes it possible to query after an export:
```
textPayload=~"chunks_outside" AND textPayload!~"chunks_outside=0"
```

### 1b. Warn if a shard writes zero chunks

After the chunk loop, before upload:
```python
if chunks_written == 0:
    logger.error("Shard produced no output",
                 scale=scale, shard=shard_name,
                 total_chunks=reader.chunk_count,
                 chunks_outside=chunks_outside,
                 chunks_failed=chunks_failed)
    return False  # Don't silently succeed
```

Currently a shard with all chunks clipped returns `True` and uploads nothing. This should be treated as a failure.

### 1c. Validate chunk coordinates against volume bounds on shard load

After loading the shard reader, compute the expected bounds:
```python
vol_shape = dest.shape[:3]
shard_chunks = reader.available_chunks
max_voxel = [max(c[d] * CHUNK_VOXELS + CHUNK_VOXELS for c in shard_chunks)
             for d in range(3)]
if any(mv > vs for mv, vs in zip(max_voxel, vol_shape)):
    logger.warning("Shard extends beyond volume",
                   scale=scale, shard=shard_name,
                   shard_max_voxel=tuple(max_voxel),
                   volume_shape=tuple(vol_shape))
```

This is a cheap check that runs once per shard (not per chunk).

## 2. Post-Export Reconciliation

### 2a. GCS-based verification (`scripts/verify_export.py`)

A new script that directly verifies NG shard files exist on GCS — no dependency on Cloud Logging.

For each scale:
1. List all DVID Arrow files from the source path
2. Compute the expected NG shard number for each DVID shard (using the compressed Z-index)
3. List all `.shard` files in the NG destination
4. Report:
   - DVID shards with no corresponding NG shard (missing output)
   - NG shards with no corresponding DVID shard (orphaned output)
   - Count mismatches per scale

This is similar to what `analyze_v011_memory.py` does for the mapping, but focused on completeness verification rather than size correlation.

### 2b. Integrate into `export --wait` flow

When `pixi run export --wait` is used, after all Cloud Run jobs complete, automatically run the GCS-based verification and report any missing shards. This catches failures without requiring manual intervention.

## 3. Compressed Z-Index Reference Verification

### 3a. The problem

The neuroglancer precomputed sharded format uses a **compressed Z-index** (not a standard Morton code) to map chunk grid coordinates to shard numbers. The compressed variant only interleaves bits for dimensions that still have significant bits at each level:

```
Standard Morton:    always 3 bits per level (x, y, z)
Compressed Z-index: skip dimension d at level i when i >= bit_length(grid_shape[d] - 1)
```

For a volume with grid shape `[1471, 1224, 2103]`, the compressed Z-index uses `[11, 11, 12]` bits per dimension. At level 11, only z contributes. This changes ALL shard numbers compared to a standard Morton code.

### 3b. TensorStore reference implementation

The authoritative implementation is in:

- **`~/tensorstore/tensorstore/driver/neuroglancer_precomputed/metadata.cc`**
  - `GetCompressedZIndexBits()` — computes bits per dimension from volume shape and chunk size
  - `EncodeCompressedZIndex()` — the actual compressed Z-index encoding
  - `CompressedMortonBitIterator` — helper for iterating through bit levels
  - `GetShardChunkHierarchy()` — computes shard/minishard spatial extents

- **`~/tensorstore/tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded.cc`**
  - `GetChunkShardInfo()` — chunk ID → shard + minishard number
  - `GetShardKey()` — shard number → hex filename (with `shard_bits`-based zero-padding)

- **`~/tensorstore/tensorstore/driver/neuroglancer_precomputed/driver.cc`**
  - `GetChunkStorageKey()` — chunk grid index → compressed Z-index → chunk key

### 3c. Verification approach

Any Python code that computes shard numbers from chunk coordinates must be tested against TensorStore's implementation. Specifically:

1. **Unit test**: For each scale in the NG spec, pick several known chunk coordinates, compute the compressed Z-index and shard number in Python, and verify against the actual NG shard filenames on GCS.

2. **Cross-reference with DVID**: The DVID repo (`~/go-code/src/github.com/janelia-flyem/dvid/datatype/labelmap/export.go`) has its own `mortonCode()` function that was fixed in commit `245db412` to match TensorStore. Our Python implementation should produce identical results for test vectors.

3. **Shard filename format**: The NG shard filename uses `ceil(shard_bits / 4)` hex digits (from TensorStore's `GetShardKey()`). This varies per scale:
   - Scale 0 (`shard_bits=19`): 5 hex digits → `00000.shard` to `7ffff.shard`
   - Scale 5 (`shard_bits=4`): 1 hex digit → `0.shard` to `f.shard`
   - Scale 7+ (`shard_bits=0`): 1 hex digit → `0.shard`

   Note: TensorStore uses `CeilOfRatio(shard_bits, 4)` for the digit count. When `shard_bits=0`, this yields 0, but the actual filename has at least 1 digit — verify this edge case.

### 3d. Centralize the implementation

The compressed Z-index computation currently lives in `scripts/analyze_v011_memory.py`. It should be extracted into a shared module (e.g., `src/ng_sharding.py` or `braid/src/braid/ng_sharding.py`) so that:
- `scripts/verify_export.py` can use it for reconciliation
- `scripts/analyze_v011_memory.py` can use it for shard mapping
- `src/worker.py` can use it for per-shard validation
- Unit tests can verify it against TensorStore reference vectors

## 4. DVID Export Shard Validation

Before the tensorstore-export pipeline even runs, we should validate the DVID Arrow export itself. The DVID `export-shards` command assigns chunks to shards based on its own compressed Morton code implementation, which had bugs fixed as recently as March 2026 (commit `245db412`). Errors at this stage propagate silently through the entire pipeline.

### 4a. Validate chunk-to-shard assignment (`scripts/validate_dvid_export.py`)

A new script that reads each DVID Arrow shard's CSV index and verifies that every chunk in the shard actually belongs there according to the neuroglancer sharding spec:

1. For each scale, load the NG spec sharding parameters (preshift_bits, minishard_bits, shard_bits, grid shape)
2. For each DVID shard:
   a. Read the CSV index to get all chunk coordinates (cx, cy, cz)
   b. Compute the compressed Z-index for each chunk
   c. Derive the expected NG shard number: `compressed_z >> (preshift_bits + minishard_bits)`
   d. Verify ALL chunks in the DVID shard map to the **same** NG shard number
   e. Verify that NG shard number matches what the shard's origin coordinate predicts

Report:
- Shards where chunks map to multiple NG shards (indicates DVID shard assignment bug)
- Shards where the origin-based prediction disagrees with the chunk-based computation
- Chunks outside the NG spec volume bounds (coordinates beyond `size / chunk_size`)
- Per-scale summary: total shards checked, errors found, chunks outside bounds

This script uses the same centralized compressed Z-index implementation (section 3d) that the worker and analysis scripts use, ensuring consistency.

### 4b. Verify compressed Z-index against TensorStore

The validation script's correctness depends entirely on our compressed Z-index matching TensorStore's. To close this loop:

1. **Generate test vectors from TensorStore directly**: Write a small C++ or Python program using TensorStore's API that, for a given NG spec, computes the shard number for a set of chunk coordinates. This produces ground-truth (shard_number, chunk_x, chunk_y, chunk_z) tuples.

2. **Verify our Python implementation against these vectors**: For each test coordinate, assert that `compressed_z_index()` and the shard number derivation produce identical results.

3. **Verify DVID's Go implementation against the same vectors**: Run the same coordinates through DVID's `mortonCode()` and `computeShardID()` to confirm all three implementations (TensorStore C++, our Python, DVID Go) agree.

4. **Spot-check against actual GCS data**: For a sample of DVID shards, compute the expected NG shard filename and verify it exists on GCS with the expected number of hex digits (`ceil(shard_bits / 4)`).

Test vectors should cover edge cases:
- Coordinates at (0, 0, 0)
- Maximum coordinates for each scale (near grid boundary)
- Coordinates where grid dimensions differ significantly (triggering compressed vs standard Morton divergence)
- Scales with shard_bits=0 (single shard)

### 4c. Volume extent alignment

DVID's `export-shards` iterates all stored blocks via `ProcessRange()` without filtering by the NG spec volume extent. Blocks beyond the spec's `size` field are exported into Arrow shards that the tensorstore-export worker then clips or skips.

Options:
- **DVID side**: Add coordinate validation in `chunkHandler()` to skip blocks outside the NG spec `Size` field. This prevents exporting data that will never be used.
- **tensorstore-export side**: Treat DVID shards with chunks beyond volume bounds as expected (edge effect), but log and report them clearly (covered by sections 1a and 4a).

### 4d. Shard completeness tracking

The DVID export should record the total number of shards and chunks exported per scale (in the `export.log` or a summary JSON). The tensorstore-export pipeline should verify these counts match after processing — any discrepancy means shards were lost or duplicated.

## Implementation Priority

| Item | Effort | Impact | Priority |
|------|--------|--------|----------|
| 1b. Zero-chunk shard failure | Small | Prevents silent data loss | **High** |
| 1a. Log out-of-bounds chunks | Small | Diagnostic visibility | **High** |
| 4a. Validate DVID chunk-to-shard | Medium | Catches upstream bugs before export | **High** |
| 4b. Z-index cross-verification | Medium | Correctness confidence across all 3 impls | **High** |
| 2a. GCS-based verify script | Medium | Post-export validation | **High** |
| 3d. Centralize Z-index code | Small | Code quality, testability | **Medium** |
| 3c. Unit tests vs TensorStore | Medium | Correctness confidence | **Medium** |
| 1c. Per-shard bounds check | Small | Early warning | **Medium** |
| 2b. Integrate into --wait | Small | Automation | **Low** |
| 4c. DVID-side filtering | Medium | Cleaner exports | **Low** |
