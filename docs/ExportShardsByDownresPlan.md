# Plan: Fix downres_scale() to use tmpfs staging + manifest-driven per-shard processing

## Context

`downres_scale()` (`src/worker.py:552-594`) currently opens the destination volume
directly on GCS via `_open_dest_scale()`. While the neuro-volume-expert confirmed
that `ts.copy()` batches writes per shard (and skips reads entirely for full shards),
ALL writes still go directly to GCS. This means:
- Partial/boundary shards do one shard-level RMW against GCS
- Every shard write incurs GCS operation costs
- No local staging = no retry without re-uploading

The fix mirrors the existing shard-processing pattern: manifest-driven work
assignment, tmpfs staging per output shard, upload, cleanup, next shard.

**Bucket setup**: Single-region aligned source bucket (DVID Arrow shards) and
single-region aligned dest bucket (NG precomputed output, hierarchical namespace,
soft delete disabled). No multi-region replication concerns.

## Key TensorStore findings (from neuro-volume-expert)

1. `ts.copy()` without explicit transaction creates one implicit transaction per
   shard region — all chunks in a shard are batched, not per-chunk RMW
2. Full shards get **unconditional writes** (no read needed)
3. `ts.downsample(source, [2,2,2,1], "mode")` is fully lazy — no materialization
4. Read-side uses byte-range requests per chunk, NOT full shard downloads. Cache
   stores decoded minishard indices, not raw shard blobs. Memory-light.
5. Slicing `ts.copy()` to one shard's voxel bbox produces exactly one `.shard` file

## Files to modify

| File | Change |
|------|--------|
| `src/worker.py` | Rewrite `downres_scale()` → `downres_shard()`, add `run_downres()` loop |
| `src/ng_sharding.py` | Add `enumerate_shard_bboxes(scale_params)` to compute per-shard voxel extents |
| `scripts/precompute_manifest.py` | Add `--downres-scales` mode that enumerates output shards and estimates memory |
| `scripts/export.py` | Wire up downres manifest generation + Cloud Run job launch |

## Implementation

### 1. `src/ng_sharding.py` — add shard bbox enumeration

Add functions:

**`get_shard_shape_in_chunks(scale_params) -> (sx, sy, sz)`**
- Computes the shard volume size in chunks per dimension from
  `preshift_bits + minishard_bits` and the compressed Morton code bit layout
- The Morton code interleaves x,y,z bits, so the shard shape may be asymmetric
  for non-cubic volumes. Replicate the `CompressedMortonBitIterator` logic
  from `~/tensorstore/tensorstore/driver/neuroglancer_precomputed/metadata.cc`
  (the neuro-volume-expert provided a Python sketch for this)

**`shard_bbox(shard_number, scale_params) -> dict`**
- Given a shard number, compute its voxel bounding box:
  ```python
  {
      "shard_origin": (x0, y0, z0),  # voxel coordinates
      "shard_extent": (sx, sy, sz),  # voxel size (clipped to volume bounds)
      "shard_number": N,
      "num_chunks": K,               # actual chunks (< max for boundary shards)
  }
  ```
- Requires inverting the shard number back to spatial coordinates via the
  compressed Morton code structure

**`parent_shards_to_child_shards(parent_shard_numbers, parent_params, child_params) -> List[int]`**
- Given a list of existing shard numbers at scale N-1, compute the set of
  shard numbers needed at scale N
- For each parent shard: compute its chunk grid range, map to child scale
  coordinates (divide by 2 in each dim), compute child shard numbers via
  `compressed_z_index` → `chunk_shard_info`
- Deduplicate and return sorted list
- This is the core function for the sparse shard derivation chain:
  s0 shards (from DVID Arrow files) → s1 shards → s2 shards → ... → s9 shards

### 2. `src/worker.py` — rewrite downres to per-shard tmpfs staging

Replace `downres_scale()` with `downres_shard()`:

```python
def downres_shard(self, scale: int, shard_bbox: dict) -> bool:
    """Generate one output shard at `scale` by downsampling scale-1 from GCS.

    1. Open source scale (N-1) from GCS with bounded cache
    2. Create tmpfs staging dir with info file
    3. Open local staging TensorStore (file driver)
    4. ts.copy(downsampled[bbox], local_dest[bbox]).result()
    5. Upload shard file(s) to GCS
    6. Delete staging dir
    """
```

Key details:
- **Source**: Open scale N-1 from the DEST bucket on GCS (read-only). Use
  `cache_pool.total_bytes_limit` in the TensorStore Context to bound read-side
  memory (e.g., 256 MB for minishard index cache — this is lightweight since
  reads use byte-range requests, not full shard downloads)
- **Staging**: Create per-shard staging dir under `self._staging_base`, write
  the info JSON, open with `"kvstore": {"driver": "file", "path": staging_dir}`
- **Copy**: Slice both source and dest to the shard's voxel bbox:
  ```python
  x0, y0, z0 = shard_bbox["shard_origin"]
  sx, sy, sz = shard_bbox["shard_extent"]
  downsampled = ts.downsample(source, [2, 2, 2, 1], "mode")
  ts.copy(
      downsampled[x0:x0+sx, y0:y0+sy, z0:z0+sz, :],
      local_dest[x0:x0+sx, y0:y0+sy, z0:z0+sz, :]
  ).result()
  ```
- **Upload**: Reuse existing `upload_staging_dir()` method
- **Cleanup**: `shutil.rmtree(staging_dir)` after upload to free tmpfs

Add `run_downres()` as the top-level loop (parallel to `run()` for shard processing):
- Loads a downres manifest (same format: list of `{scale, shard_origin, shard_extent, ...}`)
- Iterates over assigned output shards, calling `downres_shard()` for each
- Respects `_should_continue()` time limit
- Logs progress (shard count, upload bytes, elapsed time)

### 3. `scripts/precompute_manifest.py` — downres manifest generation

Add `--downres-scales` flag. The key insight is that the volume is sparse — not
every possible shard position has data. We derive the output shard list from the
**parent scale's existing shards**, not from enumerating the full grid.

**Shard derivation chain:**
- **s0 shards**: derived from existing DVID Arrow source files (already the case)
- **s1 shards**: derived from which s0 NG shards exist on GCS (or from the s0 manifest)
- **s2 shards**: derived from which s1 NG shards exist
- **sN shards**: derived from which s(N-1) NG shards exist

For each parent shard at scale N-1:
1. Compute its voxel bounding box (from shard number → spatial extent)
2. Map to scale N coordinates (divide by 2 in each dimension)
3. Compute which scale-N shard number(s) the mapped region falls into
   (using `compressed_z_index` → `chunk_shard_info` from `ng_sharding.py`)
4. Collect and deduplicate → these are the output shards needed at scale N

This means the manifest generator needs to know which shards exist at the
parent scale. Two options:
- **From GCS**: List `.shard` files at the parent scale in the dest bucket
  (works if parent scale is already fully written)
- **From manifest chain**: For s1, use the s0 shard processing manifest to
  know which s0 shards were produced; for s2, use the s1 downres manifest; etc.
  This allows computing the full chain before any downres runs.

**Recommended: manifest chain approach.** This computes the entire downres
manifest set (s1 through s9) at manifest-generation time, before any Cloud Run
jobs launch. The s0 DVID source files determine s0 shards, which determine s1
shards, which determine s2 shards, etc.

For each derived output shard, estimate memory:
```
memory_gib = (source_read_cache_gib + output_shard_tmpfs_gib + baseline_gib) * safety
```
Where:
- `source_read_cache_gib`: bounded by `cache_pool` setting (~0.25 GiB)
- `output_shard_tmpfs_gib`: `num_chunks * KB_PER_CHUNK[scale]` (using existing
  KB_PER_CHUNK table). Shard sizes vary significantly — interior shards with
  dense segmentation are much larger than boundary shards. The KB_PER_CHUNK max
  values capture worst-case density.
- `baseline_gib`: ~1.5 GiB (Python + TensorStore + GCS client — lighter than
  shard processing since no Arrow/BRAID/pyarrow overhead)
- `safety`: 1.3x

Expect larger tiers at lower-resolution scales where blocks contain more labels.

Assigns to tiers and distributes across tasks (reuse `pick_tier()`,
`distribute_tasks()`).

Writes manifests with extended format:
```json
[{"scale": 1, "shard_origin": [0,0,0], "shard_extent": [2048,2048,2048],
  "shard_number": 0, "num_chunks": 32768}, ...]
```

### 4. `scripts/export.py` — launch downres jobs

Add a `--downres` mode that:
1. Runs downres manifest generation (or reads pre-computed manifests)
2. Launches one Cloud Run job per tier (same pattern as shard export)
3. Sets env vars: `DOWNRES_MODE=1`, `MANIFEST_URI=...`, `DEST_PATH=...`,
   `NG_SPEC=...`, `WORKER_MEMORY_GIB=...`

The worker detects `DOWNRES_MODE` and calls `run_downres()` instead of `run()`.

### 5. Memory analysis by scale

Output shard sizes grow at lower-resolution scales due to denser labels. Expected
tier distribution for downres (based on KB_PER_CHUNK from mCNS v0.11):

| Target scale | Source scale | Output shards | Expected tiers |
|-------------|-------------|---------------|----------------|
| 1 | 0 | 3,364 | Mostly 4-8 GiB |
| 2 | 1 | 606 | Mostly 8-16 GiB |
| 3 | 2 | 123 | 16-24 GiB |
| 4 | 3 | 25 | 16-32 GiB |
| 5+ | 4+ | <10 each | 4-8 GiB (few chunks) |

These are rough — the actual distribution depends on label density at each
position in the volume. The manifest generator computes per-shard estimates.

## What we are NOT changing

- The existing shard-processing path (Phase 1) is untouched
- The existing `downres_scale()` method stays for now (can be deprecated later)
- No changes to BRAID or the Arrow reader
- No changes to `setup_destination.py`

## Verification

1. **Unit test — shard shape**: Test `get_shard_shape_in_chunks()` against known
   mCNS v0.11 spec values (`examples/mcns-v0.11-export-specs.json`)

2. **Unit test — shard derivation chain**: Test `parent_shards_to_child_shards()`
   with the mCNS spec. Given the list of s0 shard files (from DVID export),
   verify the derived s1 shard count matches the actual 3,364 s1 shards produced
   by DVID. Similarly for s1→s2 (606), s2→s3 (123), etc. This validates both
   the sparse derivation and the compressed Morton code mapping between scales.

3. **Local test**: Run `downres_shard()` locally against a small test volume to
   verify it produces correct shard files on tmpfs and uploads them

4. **Manifest test**: Run `pixi run precompute-manifest --downres-scales 1,2,3`
   to generate the full chain. Verify shard counts, tier assignments, and that
   the chain correctly propagates sparsity (empty regions at s0 → no shards at s1+)

5. **Comparison test**: For a small region, compare the output of the new
   per-shard downres against the old `downres_scale()` to confirm identical voxel
   values
