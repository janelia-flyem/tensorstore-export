# DVID-to-Compressed-Segmentation Transcoder

## Overview

The BRAID codec library (`libbraid_codec.so`) provides a C extension that transcodes
DVID compressed label blocks directly into neuroglancer compressed_segmentation bytes.
The input may be zstd-compressed (as stored in DVID Arrow export shards) and the output
may be gzip-compressed (as stored in neuroglancer precomputed shard files). Both
compression layers are handled in C, avoiding Python overhead.

The encoding algorithm was ported from TensorStore's reference implementation in
`neuroglancer_compressed_segmentation.cc` and verified byte-exact against its test
vectors and through decode-roundtrip using TensorStore as an oracle.

## Motivation

The existing export pipeline reads DVID Arrow shard files, decompresses each 64x64x64
label block through the BRAID Python library, and writes the resulting uint64 volume
into a local neuroglancer precomputed store via TensorStore. TensorStore handles the
compressed_segmentation encoding internally, commits each batch to a local tmpfs-backed
staging directory via read-modify-write, and the finished shard file is uploaded to GCS.

This works but has two costs:

1. **Memory**: On Cloud Run Gen 2 the local filesystem is tmpfs. The output shard file
   grows on tmpfs as chunks are committed, and during each read-modify-write commit
   TensorStore briefly holds both the old and new shard in memory.

2. **CPU**: The data path crosses Python/C boundaries multiple times per chunk
   (zstd decompress in Python, DVID decode in C, numpy transpose in Python, TensorStore
   encode in C++, TensorStore RMW commit in C++).

A C transcoder that goes directly from the Arrow record's compressed bytes to the final
gzip-compressed chunk bytes eliminates both the Python roundtrip and the TensorStore
dependency for encoding. This is a prerequisite for a custom shard writer that can
assemble neuroglancer shard files directly from compressed chunk data, bypassing
TensorStore's transaction system and its tmpfs memory overhead entirely.

## Source and Target Formats

### DVID Compressed Label Block

DVID stores segmentation data in a custom compressed format documented in
`dvid/datatype/common/labels/compressed.go`. Each block covers a 64x64x64 voxel
region subdivided into a grid of 8x8x8 sub-blocks. In the DVID export-shards Arrow
files, each block is zstd-compressed before being stored in the `dvid_compressed_block`
binary column.

After zstd decompression, the binary layout is:

```
Offset  Type           Count         Description
------  ----           -----         -----------
0       uint32 LE      1             gx: sub-blocks in X (typically 8)
4       uint32 LE      1             gy: sub-blocks in Y (typically 8)
8       uint32 LE      1             gz: sub-blocks in Z (typically 8)
12      uint32 LE      1             N: number of unique labels in block
16      uint64 LE      N             Labels array (supervoxel IDs)

--- If N < 2: block is solid; fill output with Labels[0] or 0, done. ---

16+N*8  uint16 LE      gx*gy*gz      NumSBLabels: label count per sub-block
        uint32 LE      sum(above)    SBIndices: indices into Labels per sub-block
        bytes          variable      SBValues: bit-packed voxel indices
```

Sub-blocks are iterated in Z-Y-X order (Z outermost). Within each sub-block,
voxel indices are bit-packed MSB-first (big-endian) with `ceil(log2(n_labels))`
bits per voxel, padded to a byte boundary between sub-blocks.

The decompressor (`dvid_decompress.c`) produces a flat uint64 array in ZYX memory
order: `output[z * ny * nx + y * nx + x]`.

### Neuroglancer Compressed Segmentation

The neuroglancer precomputed format supports a `compressed_segmentation` encoding
for segmentation volumes (uint32 or uint64). The format is documented in the
[neuroglancer source](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/sliceview/compressed_segmentation)
and implemented in TensorStore at
`tensorstore/internal/compression/neuroglancer_compressed_segmentation.cc`.

For a single-channel uint64 chunk with block_size (bz, by, bx), the binary layout is:

```
Offset  Type           Count              Description
------  ----           -----              -----------
0       uint32 LE      1                  Channel 0 offset (always 1, in uint32 units)

--- Channel data (all offsets below in uint32 units from here) ---

4       block_header   gz * gy * gx       Per-block headers (8 bytes each)
...     bytes          variable           Encoded values + label tables
```

Each block header is 8 bytes (2 x uint32 LE):

```
Word 0: table_base_offset (bits 0-23) | encoding_bits (bits 24-31)
Word 1: encoded_value_base_offset (bits 0-23) | padding (bits 24-31)
```

Blocks are numbered as `x + grid_x * (y + grid_y * z)` and iterated Z-Y-X
(Z outermost, X innermost), matching the DVID sub-block iteration order.

Within each block, the encoding process is:

1. Collect all unique uint64 labels, sort ascending.
2. Determine `encoding_bits`: 0 for a single unique value, otherwise the smallest
   power of 2 >= ceil(log2(num_unique)). Valid values: 0, 1, 2, 4, 8, 16.
3. Bit-pack each voxel's index (into the sorted label table) into 32-bit
   little-endian words. The voxel offset within the block is
   `x + bx * (y + by * z)` (X fastest), and the bit position is
   `(offset * encoding_bits) % 32` within word `(offset * encoding_bits) / 32`.
4. Write the label table as consecutive uint64 LE values (8 bytes each).

The encoded values area is always sized for the full block dimensions
(`ceil(encoding_bits * bx * by * bz / 32)` words), even for edge blocks
that are smaller than the block size. Unoccupied positions are zero-initialized,
which maps to index 0 (the numerically smallest label in the block), matching
TensorStore's "pad with lowest value" semantics.

### Key Differences Between the Formats

| Property | DVID | Neuroglancer cseg |
|----------|------|-------------------|
| Granularity | 8x8x8 sub-blocks always | Configurable block size (commonly 8x8x8) |
| Bit packing | Big-endian, MSB-first, byte-padded per sub-block | Little-endian, packed into 32-bit LE words |
| Bits per voxel | `ceil(log2(n))` | Smallest power of 2 >= `ceil(log2(n))` |
| Label table | One block-level table; sub-block palettes index into it | One table per block; tables may be deduplicated across blocks |
| Byte order | Little-endian integers, big-endian bit packing | Little-endian throughout |
| Outer compression | zstd | gzip (per shard spec `data_encoding`) |

Because of the bit-packing and table structure differences, direct transcoding between
the formats is not practical. The transcoder fully decompresses to an intermediate
uint64 volume and re-encodes, which is still extremely fast in C (~0.1ms per 8x8x8
block, ~50ms for a full 64^3 chunk with 512 blocks).

## Implementation

### C Functions

All three functions are in `braid/csrc/cseg_encode.c`, compiled together with
`dvid_decompress.c` into `libbraid_codec.so`.

#### `cseg_encode_chunk`

Encodes a uint64 ZYX-contiguous volume as single-channel compressed_segmentation.

```c
int cseg_encode_chunk(
    const uint64_t *input,        // ZYX contiguous, shape (nz, ny, nx)
    int nz, int ny, int nx,       // volume dimensions
    int bz, int by, int bx,       // block size
    uint8_t *output,              // pre-allocated output buffer
    size_t output_cap,            // buffer capacity in bytes
    size_t *output_size);         // [out] actual bytes written
```

The algorithm iterates blocks in Z-Y-X order. For each block:

1. Copy the block's voxels to a temporary buffer, sort, and deduplicate in-place
   to obtain the sorted unique label set. This is O(n log n) via `qsort` where n
   is the block voxel count. For 8x8x8 blocks (n=512) this takes ~5us.

2. Compute `encoding_bits` and check the deduplication cache. The cache is a
   linear array of previously-seen sorted label sets; lookup is O(d * m) where
   d is the number of distinct tables written so far and m is the label count.
   For typical segmentation data (many solid or low-diversity blocks, d ~30, m ~5),
   this is negligible.

3. If the label set is new, append both the encoded values and the label table
   to the output. If it was seen before, append only the encoded values and
   reuse the cached table offset in the block header.

4. For each voxel in the block, binary-search the sorted unique array to find
   the label's index, then pack that index into the correct bit position within
   the encoded values area.

5. Write the 8-byte block header referencing the encoded values offset and
   table offset (both in uint32 units relative to the channel data start).

#### `dvid_to_cseg`

Full pipeline from Arrow record bytes to shard-ready chunk data:

```c
int dvid_to_cseg(
    const uint8_t *dvid_data,     // DVID block bytes (optionally zstd-compressed)
    size_t dvid_len,
    const uint64_t *sv_map_keys,  // supervoxel IDs (mapping keys), or NULL
    const uint64_t *sv_map_vals,  // agglomerated label IDs (mapping values), or NULL
    size_t num_map_entries,        // number of entries in the mapping (0 if NULL)
    int bz, int by, int bx,       // cseg block size
    int flags,                     // BRAID_INPUT_ZSTD=1, BRAID_OUTPUT_GZIP=2
    uint8_t *output,
    size_t output_cap,
    size_t *output_size);
```

The DVID block header (gx, gy, gz, N, labels) is parsed internally — the caller
does not need to know the grid dimensions or label count.

Steps:

1. **Optional zstd decompression** (flag `BRAID_INPUT_ZSTD`): Uses
   `ZSTD_getFrameContentSize` to determine the decompressed size, allocates a
   buffer, and calls `ZSTD_decompress`. Linked against libzstd.

2. **Header parsing and label mapping**: Reads gx, gy, gz, N, and the block's
   internal label array (supervoxel IDs) from the DVID header. If a mapping is
   provided (`sv_map_keys` / `sv_map_vals`), each block label is looked up in
   the mapping and replaced with the corresponding agglomerated label ID.
   Labels not found in the mapping are kept as-is (identity). The mapping is
   applied via linear scan since the number of block labels is typically small
   (< 256). If `sv_map_keys` is NULL, labels are used unchanged.

3. **DVID label decode**: Calls the existing `dvid_decompress_block` function
   (from `dvid_decompress.c`) with the mapped labels to produce the intermediate
   uint64 ZYX volume.

4. **Compressed segmentation encode**: Calls `cseg_encode_chunk` on the
   intermediate volume. For gzip output, encodes into a temporary buffer first.

5. **Optional gzip compression** (flag `BRAID_OUTPUT_GZIP`): Uses zlib's
   `deflateInit2` with `windowBits = 15 + 16` to produce gzip-format output
   (not raw deflate). This matches the `data_encoding: "gzip"` option in
   neuroglancer's sharding spec.

#### `cseg_max_encoded_size`

Returns a conservative upper bound on the compressed_segmentation output size
(before gzip) for a given volume shape and block size. Used by the Python wrapper
to pre-allocate the output buffer.

```c
size_t cseg_max_encoded_size(int nz, int ny, int nx, int bz, int by, int bx);
```

The bound assumes worst-case 16-bit encoding with all voxels having distinct labels
in every block (no table deduplication). For a 64^3 chunk with 8^3 blocks this is
approximately 2.6 MB — well under the uncompressed volume size of 2 MB, so 4 MB
is a safe allocation.

### Table Deduplication

TensorStore's encoder uses an `absl::flat_hash_map<vector<Label>, uint32_t>` to
deduplicate label tables across blocks. The C implementation uses a simpler linear
scan: an array of up to 1024 entries, each storing a sorted label set (up to 512
labels) and its table offset. For each new block, the cache is scanned with `memcmp`
against the sorted label set.

This is O(d * m) per block where d is the number of distinct label sets seen so far
and m is the label count. In practice, segmentation chunks have ~20-50 distinct
label sets (many blocks are solid or share the same small set of labels), making
this effectively constant-time. For the common case of solid blocks (single label,
m=1), the comparison is a single 8-byte `memcmp`.

The deduplication is important for output compactness: in a typical 64^3 chunk
where 60-80% of the 512 blocks are solid background, deduplication saves
~400 * 8 = 3.2 KB of redundant table entries. More importantly, it matches
TensorStore's behavior, enabling byte-exact comparison in tests.

### Build

```makefile
libbraid_codec.so: dvid_decompress.c cseg_encode.c
    $(CC) $(CFLAGS) $(LDFLAGS) -shared -fPIC -o $@ $^ -lzstd -lz
```

Dependencies:
- **libzstd** (zstd decompression): Provided by the pixi/conda environment
  via the `zstandard` package. Headers at `$CONDA_PREFIX/include/zstd.h`.
- **libz** (gzip compression): System zlib, universally available.

The Makefile auto-detects `$CONDA_PREFIX` to set include and library paths.
The legacy `libdvid_decompress.so` is still built separately for backward
compatibility with the standalone DVID decompressor.

### Python Wrapper

`braid/src/braid/cseg_encoder.py` provides the `CSEGEncoder` class:

```python
from braid import CSEGEncoder

enc = CSEGEncoder()

# Standalone encode: numpy uint64 ZYX volume → cseg bytes
cseg_bytes = enc.encode_chunk(volume, block_size=(8, 8, 8))

# Fused pipeline: zstd DVID block → gzip cseg bytes (ready for shard file)
# Label mapping (supervoxel → agglomerated) is applied in C.
chunk_data = enc.dvid_to_cseg(
    dvid_data=arrow_record["dvid_compressed_block"],
    block_size=(8, 8, 8),
    supervoxels=arrow_record["supervoxels"],
    agglo_labels=arrow_record["labels"],
    zstd_input=True,
    gzip_output=True,
)
```

The wrapper uses ctypes to load `libbraid_codec.so` and manages output buffer
allocation via `cseg_max_encoded_size`.

## Memory Layout and Axis Order

The DVID decompressor outputs uint64 values in ZYX order (Z slowest, X fastest):

```
output[z * ny * nx + y * nx + x]
```

This matches numpy's C-contiguous layout for shape `(nz, ny, nx)` and is the
natural layout for the DVID sub-block iteration (Z-Y-X outermost to innermost).

The neuroglancer compressed_segmentation format specifies voxel offsets within
a block as `x + bx * (y + by * z)` (X fastest), which is the same traversal
order. The encoder reads from the ZYX input and packs into the neuroglancer
format without any transposition.

TensorStore's internal representation for neuroglancer precomputed volumes uses
an `[x, y, z, channel]` domain. The existing worker code transposes BRAID's ZYX
output to XYZ before writing to TensorStore. With the transcoder, this
transposition is unnecessary — the encoder reads ZYX data directly and produces
correctly-formatted compressed_segmentation output because the per-block encoding
uses coordinate-based offset calculation rather than relying on a particular
memory layout.

## Verification

The encoder is verified at five levels (28 tests total, ~2 seconds):

### 1. Byte-exact match against TensorStore test vectors (4 tests)

Test vectors from `neuroglancer_compressed_segmentation_test.cc` are translated
to Python and the encoder's output is compared byte-for-byte. This covers:
- 0-bit encoding (solid blocks)
- 1-bit encoding (two labels)
- 2-bit encoding (three labels)
- Table deduplication (multiple blocks sharing a label set)

### 2. TensorStore decode roundtrip (6 tests)

The encoder's output is written to a local neuroglancer precomputed volume and
read back using TensorStore's decoder. The decoded voxels are compared against
the original input. This is the gold-standard test because TensorStore is the
authoritative implementation of the format. Test cases include:

- 8^3, 64^3, and non-aligned volume shapes
- 8^3 and 64^3 block sizes
- Random label data with varying numbers of unique labels per block

### 3. Encoding-bits coverage (13 tests)

Systematic coverage of every `encoding_bits` value (0, 1, 2, 4, 8, 16) with
TensorStore decode roundtrip verification at each bit-width boundary:

| Test | Unique labels | encoding_bits | Notes |
|------|--------------|---------------|-------|
| 0-bit solid | 1 | 0 | Single label fills block |
| 1-bit two labels | 2 | 1 | Minimum multi-label |
| 2-bit three labels | 3 | 2 | First 2-bit case |
| 2-bit four labels | 4 | 2 | 2-bit boundary |
| 4-bit five labels | 5 | 4 | First 4-bit case |
| 4-bit sixteen labels | 16 | 4 | 4-bit boundary |
| 8-bit seventeen labels | 17 | 8 | First 8-bit case |
| 8-bit 256 labels | 256 | 8 | 8-bit boundary |
| 16-bit 257 labels | 257 | 16 | First 16-bit case |
| 16-bit all unique | 512 | 16 | Every voxel distinct |
| Large uint64 values | 7 | 4 | Values up to 2^64-1 |
| Label zero preservation | 2 | 1 | Background label 0 survives |
| 64^3 high diversity | ~500 | mixed | Multiple encoding_bits per chunk |

### 4. Real DVID data — full shard roundtrip (1 test, 258 chunks)

Every chunk in the real mCNS Arrow shard (`30720_24576_28672.arrow`, 258 chunks
with 2 to 93 labels each) is transcoded through the full fused pipeline with
supervoxel-to-agglomerated label mapping and verified voxel-by-voxel against
the Python decompressor's ground truth. This exercises the actual label
distributions from production data.

### 5. Fused DVID-to-cseg path (4 tests)

- **fib19 roundtrip**: Real DVID block (18 labels) through fused path, verified
  against ground truth via TensorStore decode.
- **Gzip output**: Same block with `BRAID_OUTPUT_GZIP`, verified that the
  gzip-decompressed bytes produce correct voxels.
- **Parity**: Fused `dvid_to_cseg` produces byte-identical output to the
  two-step path (Python DVID decompress + standalone `cseg_encode_chunk`).
- **Label mapping**: Synthetic supervoxel-to-agglomerated mapping (+1000000
  to each label) applied in C. Verified that (a) the output matches the Python
  decompressor with the same mapping, and (b) the output differs from the
  unmapped case — confirming the mapping was actually applied, not silently
  ignored.

## Files

| File | Description |
|------|-------------|
| `braid/csrc/cseg_encode.c` | C encoder: `cseg_encode_chunk`, `dvid_to_cseg`, `cseg_max_encoded_size` |
| `braid/csrc/dvid_decompress.c` | C DVID decompressor (unchanged, linked into `libbraid_codec.so`) |
| `braid/csrc/Makefile` | Builds `libbraid_codec.so` (both C files, `-lzstd -lz`) |
| `braid/src/braid/cseg_encoder.py` | Python ctypes wrapper: `CSEGEncoder` class |
| `braid/src/braid/__init__.py` | Exports `CSEGEncoder` |
| `braid/tests/test_cseg_encode.py` | 28 tests across five verification levels |
