# Plan: Custom Neuroglancer Shard Writer (Bypass TensorStore)

## Context

TensorStore's neuroglancer precomputed driver accumulates all chunk data in
memory before writing a shard file (~2MB/chunk raw uint64, 60GB for a 30K-chunk
shard). Batched local-disk RMW reduces this but still leaks ~1GiB/10K chunks
and produces 4.5GiB shard upload buffers. Combined with cross-shard memory
accumulation across sequential tasks, this causes OOM on Cloud Run even at 16Gi.

**Solution**: Build our own shard writer that encodes and writes chunks
incrementally. Memory stays flat because we:
1. Transcode each DVID block to compressed_segmentation + gzip in C (small output)
2. Append to a file/buffer, tracking offsets for the minishard index
3. Build the shard index at the end and prepend it
4. Upload the finished shard file to GCS

This approach needs only ~10MB of working memory per shard (one chunk at a time
plus the index metadata), regardless of chunk count.

### Block size issue discovered

The v0.11 export spec (`mcns-v0.11-export-specs.json`) does not include
`compressed_segmentation_block_size`. The worker's `setdefault` fills in
`[64, 64, 64]`, but `deploy.py` writes `[8, 8, 8]` to the GCS info file.
The data is encoded with 64^3 blocks while readers expect 8^3 — a silent
mismatch. The correct value (per the v0.9 reference spec) is **[8, 8, 8]**.

Similarly, the v0.11 spec lacks `data_encoding` in the sharding config (defaults
to `"raw"`), while the correct value from v0.9 is `"gzip"` — chunk data within
shards should be gzip-compressed.

The custom shard writer eliminates this class of bug by using explicit parameters
rather than relying on info file defaults.

## Architecture

### Data flow per shard:

```
For each chunk in shard assignment:
    BRAID reader (Arrow record)
        → dvid_to_cseg() in C:
            zstd decompress DVID block
            apply supervoxel → agglomerated label mapping
            DVID decode → uint64 ZYX volume
            compressed_segmentation encode (block_size 8×8×8)
            gzip compress
        → append gzipped cseg bytes to shard data buffer
        → record (chunk_id, offset, size) in minishard index
Build shard file:
    → encode minishard indices (delta-encoded, gzip compressed)
    → build shard index (16 bytes per minishard)
    → assemble: shard_index || chunk_data || minishard_indices
    → upload to GCS
```

The entire per-chunk encode path (zstd decompress → label mapping → DVID decode
→ cseg encode → gzip) executes in a single C call (`dvid_to_cseg` with
`BRAID_INPUT_ZSTD | BRAID_OUTPUT_GZIP` flags), avoiding Python/C boundary
crossings and numpy transpositions. See `braid/docs/DVID-to-cseg-transcoder.md`
for the transcoder's format details and verification.

### Shard file format (neuroglancer_uint64_sharded_v1):

```
┌─────────────────────────────────────┐
│ Shard Index                         │  num_minishards × 16 bytes
│   [min_offset, max_offset] per ms   │  (byte ranges into minishard indices)
├─────────────────────────────────────┤
│ Chunk Data                          │  gzipped compressed_segmentation chunks
│   chunk_0_data                      │  (concatenated, in minishard order)
│   chunk_1_data                      │
│   ...                               │
├─────────────────────────────────────┤
│ Minishard Indices                   │  gzip'd delta-encoded indices
│   minishard_0_index                 │  (chunk_ids, offsets, sizes)
│   minishard_1_index                 │
│   ...                               │
└─────────────────────────────────────┘
```

All byte offsets in the shard index are relative to the end of the shard index
itself (i.e., after the `num_minishards × 16` header bytes).

### Volume parameters (from v0.9 reference spec):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `encoding` | `compressed_segmentation` | uint64 labels |
| `compressed_segmentation_block_size` | `[8, 8, 8]` | 512 blocks per 64^3 chunk |
| `chunk_sizes` | `[[64, 64, 64]]` | One chunk per DVID block |
| `data_encoding` | `gzip` | Chunk data is gzip-compressed in the shard |
| `minishard_index_encoding` | `gzip` | Minishard indices are gzip-compressed |
| `hash` | `identity` | Morton code used directly (no hashing) |

### Key algorithms to implement:

1. **Compressed Morton code** — map (cx, cy, cz) chunk coordinates to uint64 key
   - Bits per dimension: `ceil(log2(ceil(volume_size[d] / chunk_size[d])))`
   - Interleave bits round-robin: x0,y0,z0, x1,y1,z1, ...

2. **Shard/minishard assignment** — partition Morton code
   - `hash_input = morton_code >> preshift_bits`
   - With identity hash: `minishard = hash_input & ((1 << minishard_bits) - 1)`
   - `shard = (hash_input >> minishard_bits) & ((1 << shard_bits) - 1)`

3. **Minishard index encoding** — delta-encoded arrays
   - Sort entries by chunk_id within each minishard
   - Three parallel arrays: chunk_id_deltas, offset_deltas, sizes
   - Each as uint64 little-endian, then gzip

4. **Shard index** — fixed-size header
   - `num_minishards = 2^minishard_bits` entries
   - Each entry: `[inclusive_min_offset, exclusive_max_offset]` (uint64le pair)
   - Offsets relative to end of shard index
   - Empty minishards: both offsets equal (zero-length range)

## Dependencies

No new third-party packages. Everything is already in the project:

- `braid.CSEGEncoder` — C transcoder: DVID block → gzip'd cseg bytes
  (replaces the `compressed-segmentation` pip package from the original plan)
- `gzip` (stdlib) — compress minishard indices
- `struct` (stdlib) — pack little-endian uint64 values
- `google-cloud-storage` — upload finished shard to GCS
- `braid.ShardReader` — read DVID Arrow shard files (existing)

The `compressed-segmentation` pip package is **not needed**. The BRAID transcoder
(`libbraid_codec.so`) produces byte-exact output matching TensorStore's encoder,
verified by 28 tests including decode-roundtrip through TensorStore as oracle.

## Files to Create/Modify

### New file: `src/shard_writer.py`
The core shard writer module:
- `class ShardSpec` — holds sharding parameters (preshift, minishard, shard bits,
  volume shape, chunk size, block_size, data_encoding, minishard_index_encoding)
- `compute_morton_code(cx, cy, cz, spec) -> uint64` — compressed Z-index
- `get_shard_and_minishard(morton_code, spec) -> (shard_id, minishard_id)`
- `class ShardBuilder` — accumulates encoded chunks, builds index, produces
  final shard bytes:
  - `add_chunk(cx, cy, cz, encoded_data: bytes)` — append pre-encoded chunk
  - `finalize() -> bytes` — build indices, return complete shard file
  - `shard_key() -> str` — hex shard filename (e.g., "070e7.shard")

Chunk encoding is done by the caller using `CSEGEncoder.dvid_to_cseg()` before
passing to `add_chunk()`. This keeps the shard writer format-agnostic.

### Modified: `src/worker.py`
Replace TensorStore write logic in `process_shard()`:
- Remove TensorStore transaction/batched-RMW code
- Remove local staging directory creation and tmpfs usage
- Use `CSEGEncoder.dvid_to_cseg()` to transcode each chunk in C
- Use `ShardBuilder` to accumulate encoded chunks and build the shard file
- Upload finished shard bytes directly to GCS via streaming upload
- Fix `compressed_segmentation_block_size` to `[8, 8, 8]` (not `[64, 64, 64]`)

### New file: `tests/test_shard_writer.py`
- Morton code correctness tests
- Shard/minishard assignment tests
- End-to-end: build a shard, read it back with TensorStore

## Memory Profile (Expected)

Per chunk:
- `dvid_to_cseg` internal buffers: ~2MB uint64 volume (transient, freed per call)
- Gzip'd cseg output: ~5-50KB per chunk (returned to Python)

Per shard:
- Minishard index metadata: ~24 bytes per chunk × 32K chunks = ~768KB
- Accumulated gzip'd chunk data: ~50-150MB for a full 32K-chunk shard
  (vs 4.5GB with TensorStore's raw buffering)
- Arrow file in RAM: unchanged (up to ~6.6GB for largest shards)

Total: Arrow file + ~200MB shard buffer + ~2MB transcoder working memory.
This is dramatically less than the current TensorStore RMW approach.

## Verification

1. **Unit test**: Write a small shard (100 chunks) with our writer, then read
   it back with TensorStore to verify correctness
2. **Comparison test**: Process the real test shard (258 chunks) with both
   TensorStore and our writer, compare chunk-by-chunk
3. **Memory test**: Monitor RSS during a large shard — should stay flat
4. **GCS integration**: Upload a shard, verify neuroglancer can render it
5. **Full export**: Run `pixi run export` with the custom writer, verify all
   scales produce correct output
