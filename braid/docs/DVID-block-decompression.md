# DVID Block Decompression: Performance Plan

## Problem

The Python DVID block decompressor in `braid/src/braid/decompressor.py` is too
slow for production use.  A single 64x64x64 block with 512 sub-blocks takes
~100ms in Python; a full shard of 32,768 chunks takes ~55 minutes.  With 200
Cloud Run workers processing ~131 shards each, jobs cannot complete within the
60-minute task timeout.

## Current Bottlenecks

The decompressor has four nested Python loops with per-voxel operations.
For a standard 64^3 block (gx=gy=gz=8, 512 sub-blocks of 8^3 voxels):

### 1. Per-voxel label mapping: O(S) per voxel

```python
# Lines 237-242 — called 262,144 times per chunk
sv_matches = np.where(supervoxels == label)[0]  # linear scan!
```

For each of the 262,144 voxels, `np.where()` does a full linear scan of the
supervoxels array.  Real data shows supervoxel lists are small per chunk
(median ~14, max ~105 in the mCNS test shard), so this is ~262K × 20 = ~5M
comparisons per chunk rather than the billions it would be with larger lists.
Still vastly worse than O(1) dict lookup (5M vs 262K operations), and the
`np.where()` call overhead per voxel is significant.  However, **Bottleneck #2
(Python function calls for bit unpacking) is likely the dominant cost** since
it involves 262,144 Python-level function calls with branching and byte
indexing per chunk.

### 2. Per-voxel bit extraction: Python function call overhead

```python
# Lines 224-234 — triple-nested Python loop, 262,144 iterations
index = get_packed_value(sb_values, bit_pos, bits)
```

Each call crosses the Python/C boundary for byte indexing and bit math.
262,144 Python function calls per chunk.

### 3. Per-voxel output assignment

```python
# Line 244 — scalar numpy assignment in a Python loop
output[base_z + z, base_y + y, base_x + x] = label
```

Individual element assignment bypasses numpy's vectorized operations.

### 4. Per-sub-block label gathering

```python
# Lines 213-216 — Python loop per sub-block
for i in range(num_sb_labels_cur):
    sb_labels[i] = labels[sb_indices[index_pos]]
```

Small loop (typically 1-20 iterations) but called 512 times.

## Block Characteristics in Production Data

Measured from real mCNS shard files across scales s0–s3, supplemented by
per-scale population statistics from the export log (587M chunks, 26,128
shards, 4.30 TB).

### Measured per-scale label statistics

All data from real mCNS shard files.  Three shards per scale (edge,
median-sized interior, densest) give coverage of the full range.

| Shard              | Chunks | Med. labels | Max labels | Med. labels/SB | Max SB labels | Max bits | Med. solid% |
|--------------------|--------|------------|-----------|---------------|--------------|---------|------------|
| s0 edge            | 258    | 14         | 105       | 1.5           | 11           | 4       | 62%        |
| s0 interior        | 32,750 | 20         | 112       | 2.2           | 15           | 4       | 39%        |
| s0 densest         | 32,768 | 32         | 121       | 2.6           | 15           | 4       | 27%        |
| s1 edge            | 154    | 1          | 1         | 1.0           | 1            | 1       | 100%       |
| s1 median          | 19,456 | 45         | 162       | 2.7           | 17           | 5       | 32%        |
| s1 densest         | 32,768 | 89         | 254       | 3.7           | 17           | 5       | 20%        |
| s2 edge            | 33     | 15         | 126       | 1.4           | 11           | 4       | 71%        |
| s2 median          | 14,437 | 101        | 570       | 2.9           | 26           | 5       | 34%        |
| s2 densest         | 32,768 | 401        | 1,233     | 6.8           | 33           | 6       | 6%         |
| s3 edge            | 18     | 31         | 144       | 1.7           | 17           | 5       | 82%        |
| s3 median          | 2,011  | 1,198      | 8,284     | 10.2          | 117          | 7       | 8%         |
| s3 densest         | 32,768 | 1,788      | 4,974     | 13.4          | 75           | 7       | 1%         |

| s9 densest         | 28     | 34,414     | 118,965   | 68.1          | 481          | 9       | 76%        |

The s9 densest shard has only 28 chunks but extreme label complexity.
The worst single chunk `(1,0,0)` has 118,965 unique labels, 232.9 avg
labels/sub-block, and 461 labels in its densest sub-block (9 bits/voxel).
The current `np.where()` does 31.2 billion comparisons for that one chunk.

**Column definitions:**
- "labels" = unique supervoxel IDs in the block-level label table
- "labels/SB" = average labels per 8x8x8 sub-block within a chunk
- "Max SB labels" = most labels in any single sub-block across all chunks in that shard
- "Max bits" = ceil(log2(max SB labels)), the widest bit-packed index
- "solid%" = fraction of sub-blocks with ≤1 label (no bit unpacking needed)

### Key findings

**Label complexity increases dramatically at coarser scales.**  s0 densest
blocks have median 32 labels; s3 densest has median 1,788 — a 56x increase.
Each coarser scale covers 8x the volume, aggregating more supervoxels.

**Sub-block label counts are much smaller than block-level counts** but
still scale significantly:
- s0: max 15 labels in any sub-block (4 bits) across 65K+ measured chunks
- s1: max 17 (5 bits)
- s2: max 33 (6 bits)
- s3: max 117 (7 bits)

**The solid sub-block fast path dominates at s0, vanishes at s3:**
- s0 interior: 39% solid → s0 densest: 27% solid
- s3 median: 8% solid → s3 densest: 0.8% solid
Since s0 has 87% of all chunks, solid path optimization still matters overall.

**The np.where() label mapping cost is catastrophic at higher scales.**
At s3 densest (median 1,788 labels), the current code does 1,788 comparisons
per voxel × 262,144 voxels = **469M comparisons per chunk**.  At s0 interior
(20 labels), it's 5.2M per chunk.  The dict pre-mapping fix eliminates this
entirely at all scales.

**The s3 densest shard is the hardest case.**  32,768 chunks, median 1,788
block labels, 13.4 avg labels/sub-block, only 0.8% solid.  Nearly every
sub-block requires 7-bit unpacking across all 512 voxels.  This is where
a C extension would provide the most benefit over numpy vectorization.

### Per-scale population statistics (from export.log)

| Scale | Mean uncomp | Max uncomp | Mean zstd | Max zstd | Records/shard (mean/max) | Total chunks |
|-------|------------|-----------|----------|---------|--------------------------|-------------|
| s0    | 26.2 KB    | 109.2 KB  | 5.15 KB  | 48.0 KB | 23,210 / 32,768          | 510,494,798 |
| s1    | 47.7 KB    | 154.6 KB  | 15.2 KB  | 86.2 KB | 19,787 / 32,768          | 66,564,145  |
| s2    | 77.0 KB    | 218.2 KB  | 31.5 KB  | 145 KB  | 14,468 / 32,768          | 8,767,633   |
| s3    | 126.5 KB   | 345.4 KB  | 66.2 KB  | 239 KB  | 9,452 / 32,768           | 1,162,703   |
| s4    | 227.7 KB   | 645.4 KB  | 143 KB   | 430 KB  | 6,296 / 23,092           | 157,407     |
| s5    | 407.5 KB   | 1.09 MB   | 265 KB   | 666 KB  | 2,759 / 10,906           | 22,076      |
| s6    | 601.6 KB   | 1.59 MB   | 378 KB   | 1.03 MB | 1,668 / 3,301            | 3,336       |
| s7    | 452.2 KB   | 1.81 MB   | 283 KB   | 1.20 MB | 414 / 814                | 828         |
| s8    | 454.5 KB   | 2.03 MB   | 305 KB   | 1.45 MB | 71 / 139                 | 143         |
| s9    | 365.5 KB   | 1.56 MB   | 227 KB   | 1.01 MB | 14 / 28                  | 29          |

### Implications for optimization

- **s0 is the priority** — 510M of 587M chunks (87%), moderate complexity
  (4 bits max per voxel), 27-39% solid sub-blocks at interior.
- **The label mapping fix is critical at all scales** but especially s3+
  where it eliminates 314M+ comparisons per chunk.
- **Bit unpacking at s0** needs to handle 0-4 bits efficiently.
  At s3, up to 7 bits.  NumPy vectorization handles both.
- **A C extension** would particularly help s3+ where sub-blocks are
  dense (10+ labels/SB, only 8% solid) and the per-voxel work is highest.

## DVID Compressed Block Format

Reference: `dvid/datatype/common/labels/compressed.go` (`MakeLabelVolume`)

```
Offset   Type        Count       Description
──────   ────        ─────       ───────────
0        uint32 LE   1           gx (sub-blocks in X, typically 8)
4        uint32 LE   1           gy (sub-blocks in Y, typically 8)
8        uint32 LE   1           gz (sub-blocks in Z, typically 8)
12       uint32 LE   1           N  (number of unique labels in block)
16       uint64 LE   N           Labels array (supervoxel IDs)

If N < 2: block is solid (fill with Labels[0] or 0), no further data.

If N >= 2:
16+N*8   uint16 LE   gx*gy*gz    NumSBLabels — labels per sub-block
         uint32 LE   sum(above)  SBIndices — indices into Labels array
         bytes       variable    SBValues — bit-packed voxel indices

SBValues layout per sub-block (if NumSBLabels[i] > 1):
  - bits_per_voxel = ceil(log2(NumSBLabels[i]))
  - 512 packed values of bits_per_voxel bits each
  - Padded to next byte boundary
  - Values are indices into the sub-block's local label palette
    (not the block-level Labels array — use SBIndices to map)
```

The inner loop order in Go is Z→Y→X with X contiguous in the output array.
Python's numpy `output[z, y, x]` with shape (nz, ny, nx) has the same memory
layout (C-order: X contiguous), so the formats match.

## Optimization Strategy: C Extension

We go directly to a C extension rather than an intermediate numpy
vectorization step.  BRAID targets Linux containers (Cloud Run) and
research Linux systems where a C compiler is always available.  A C
extension replaces the entire hot path in one step — there is no benefit
to also doing numpy vectorization since the C code handles everything
numpy would have done (bit unpacking, label lookup, array placement).

The existing Python `_make_label_volume` is kept as a reference
implementation for testing and as a fallback if the shared library
cannot be loaded.

### C function

Write `dvid_decompress_block()` in C, callable via `ctypes`.  Single
source file, ~100-150 lines.  Direct port of Go's `MakeLabelVolume`.
No external dependencies beyond a C compiler.

```c
// braid/csrc/dvid_decompress.c
//
// Takes the DVID compressed block bytes (after zstd decompression)
// plus the pre-mapped label array.  Writes the decompressed uint64
// volume directly to a pre-allocated output buffer.
//
// The label mapping (supervoxel → agglomerated) is done in Python
// before calling this function, keeping the C code simple.
void dvid_decompress_block(
    const uint8_t *data, size_t data_len,
    const uint64_t *mapped_labels, size_t num_labels,
    uint64_t *output, int gx, int gy, int gz
);
```

The function handles:
- Header parsing (gx, gy, gz, N are passed in but also validated against data)
- Sub-block metadata parsing (NumSBLabels, SBIndices)
- `getPackedValue` equivalent for bit unpacking (inline, same logic as Go)
- Direct output to pre-allocated uint64 buffer in ZYX order
- Solid sub-block fast path (0 or 1 labels → fill, no bit unpacking)
- Byte-boundary padding between sub-blocks

### Python wrapper in decompressor.py

```python
def _make_label_volume(self, compressed_data, agglo_labels, supervoxels, block_shape):
    # Pre-map labels in Python (simple dict lookup)
    labels = np.frombuffer(compressed_data[16:16 + N * 8], dtype='<u8')
    if agglo_labels is not None and supervoxels is not None:
        sv_to_agglo = dict(zip(supervoxels.tolist(), agglo_labels.tolist()))
        mapped = np.array([sv_to_agglo.get(int(l), int(l)) for l in labels],
                          dtype=np.uint64)
    else:
        mapped = labels

    # Call C extension for the hot path
    output = np.zeros(block_shape, dtype=np.uint64)
    _c_lib.dvid_decompress_block(
        compressed_data, len(compressed_data),
        mapped, len(mapped),
        output, gx, gy, gz,
    )
    return output
```

Label mapping stays in Python because:
- It runs once per chunk (not per voxel), so it's not a bottleneck
- The dict construction from the Arrow supervoxels/labels columns is
  natural in Python
- It keeps the C code focused on the format-specific hot path

### Build

```makefile
# braid/csrc/Makefile
libdvid_decompress.so: dvid_decompress.c
	$(CC) -O2 -shared -fPIC -o $@ $<
```

Built during `pixi install` via a pixi task, and during Docker build.
The shared library is loaded at import time with fallback to the Python
reference implementation if not found.

## Estimated Performance

Per-chunk estimates (64^3 block, median label counts for each scale):

| Implementation   | s0 (~30 labels) | s3 (~1800 labels) | s9 (~120K labels) |
|------------------|----------------|-------------------|-------------------|
| Current Python   | ~100 ms        | ~10 sec           | ~minutes          |
| C extension      | ~0.05 ms       | ~0.1 ms           | ~0.5 ms           |

Per-shard estimates (32K-chunk dense interior shard at s0):

| Implementation   | Time/shard | 131 shards (one worker) |
|------------------|-----------|------------------------|
| Current Python   | ~55 min   | ~5 days                |
| C extension      | ~1.6 sec  | ~3.5 min               |

These are rough estimates.  Actual times depend on label density, number of
unique labels per sub-block, and I/O overhead.  The label mapping (Python
dict lookup, once per chunk) adds negligible overhead compared to the
current np.where() approach.

## Benchmarking Plan

Add `braid/tests/test_bench_decompressor.py` using `time.perf_counter`:

1. **Micro-benchmark: single chunk decompression**
   - Use the real `fib19-64x64x64-sample1-block.dat.gz` test data
   - Compare: current Python reference vs C extension
   - Report: ms/chunk, speedup factor

2. **Macro-benchmark: full shard decompression**
   - Use the real `30720_24576_28672.arrow` shard (258 chunks)
   - Time all 258 chunks end-to-end
   - Report: total time, chunks/sec

3. **Correctness gate: voxel-exact comparison**
   - C extension must produce bit-identical output to the current Python
     implementation (which is verified against Go's MakeLabelVolume)
   - Run `test_real_data.py::TestGroundTruthRoundtrip` tests with C path

## Test Strategy

### Existing tests (must all pass unchanged)

- `test_decompressor.py` — 10 tests covering solid blocks, label mapping,
  error handling, block info
- `test_real_data.py` — 10 tests with ground truth verification against Go
  output, including voxel-exact comparison, corner voxels, sub-block boundaries
- `test_compression_layers.py` — 6 tests for zstd + DVID layer interaction
- `test_go_produced_shard.py` — 4 tests with real 258-chunk mCNS shard
- `test_integration.py` — 6 tests for Arrow reader + decompressor pipeline
- `test_e2e_precomputed.py` — 6 tests for full TensorStore pipeline

### New tests to add

1. **Bit unpacking edge cases**: 0-bit (solid), 1-bit (two labels),
   9-bit (max observed at s9), values spanning byte boundaries
2. **Large label count**: Sub-blocks with 256+ labels (9 bits per index),
   reflecting real s9 data
3. **Sparse sub-blocks**: Mix of solid (0/1 label) and multi-label sub-blocks
4. **C vs Python reference**: Assert voxel-exact match between C extension
   and Python reference on all test data
5. **Benchmark regression**: Assert C extension is at least 100x faster than
   Python reference on real shard data

## Implementation Order

1. **Write C extension** (`braid/csrc/dvid_decompress.c`) — direct port of
   Go's MakeLabelVolume
2. **Add build** (`braid/csrc/Makefile`, pixi task)
3. **Update decompressor.py** — pre-map labels in Python, call C for hot path,
   keep Python `_make_label_volume` as `_make_label_volume_reference`
4. **Run all existing tests** — verify bit-identical output
5. **Add benchmarks** — measure speedup on real data
6. **Add new edge case tests**

## Files to Modify

| File | Change |
|------|--------|
| `braid/csrc/dvid_decompress.c` | New: C decompression function |
| `braid/csrc/Makefile` | New: build shared library |
| `braid/src/braid/decompressor.py` | Pre-map labels, call C extension, keep Python reference |
| `braid/tests/test_bench_decompressor.py` | New: benchmarks comparing C vs Python |
| `braid/tests/test_decompressor.py` | Add edge case tests, C vs Python comparison |
| `pixi.toml` | Add `build-braid-c` task |
| `Dockerfile` | Build C extension during image build |

## Reference Implementations

- **Go**: `dvid/datatype/common/labels/compressed.go` — `MakeLabelVolume()`
  (lines 1568-1636), `getPackedValue()` (lines 2798-2815)
- **Python (current)**: `braid/src/braid/decompressor.py` — `_make_label_volume()`
  (lines 115-252)
- **Test ground truth**: `braid/tests/test_data/fib19-64x64x64-sample1.dat.gz`
  (raw uint64 volume from Go's decompressor)
