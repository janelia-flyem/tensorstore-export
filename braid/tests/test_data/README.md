# Test Data Files

These files are copied from the [DVID repository](https://github.com/janelia-flyem/dvid) (`test_data/` directory) and are the same files used by Go unit tests in `datatype/common/labels/compressed_test.go`. Using identical test data lets us verify that BRAID's Python decompressor produces bit-identical output to Go's `MakeLabelVolume()`.

## File Formats

### Raw uint64 label volumes (`*-sample*.dat.gz`)

Gzip-compressed flat arrays of little-endian uint64 values representing a 64x64x64 voxel block. Each file is exactly 2,097,152 bytes uncompressed (64 x 64 x 64 x 8).

Voxels are stored in XYZ order with X varying fastest: index = `z * 64 * 64 + y * 64 + x`. This matches Go's memory layout and numpy's C-order `array[z, y, x]`.

| File | Source | Labels |
|------|--------|--------|
| `fib19-64x64x64-sample1.dat.gz` | FIB-19 (Drosophila) dataset | 18 unique labels |
| `fib19-64x64x64-sample2.dat.gz` | FIB-19, different region | Multi-label |
| `cx-64x64x64-sample1.dat.gz` | CX dataset | Multi-label |
| `cx-64x64x64-sample2.dat.gz` | CX, different region | Multi-label |

### DVID compressed block (`*-block.dat.gz`)

Gzip-compressed DVID compressed segmentation block — the output of Go's `Block.MarshalBinary()`. This is the binary format that BRAID's `DVIDDecompressor` parses.

| File | Source | Corresponds to |
|------|--------|----------------|
| `fib19-64x64x64-sample1-block.dat.gz` | `MakeBlock()` + `MarshalBinary()` of sample1 | `fib19-64x64x64-sample1.dat.gz` |

#### DVID block binary layout

```
Header:
  gx, gy, gz    : 3 x uint32 LE   # sub-block grid dimensions (8,8,8 for 64^3)
  N              : uint32 LE       # number of unique labels

Label table:
  labels         : N x uint64 LE   # the unique label values

(if N > 1, the following sub-block data is present)

Sub-block metadata:
  num_sb_labels  : gx*gy*gz x uint16 LE   # label count per sub-block

Sub-block label indices:
  sb_indices     : sum(num_sb_labels) x uint32 LE   # indices into label table

Packed voxel indices:
  For each sub-block with num_sb_labels > 1:
    512 x ceil(log2(num_sb_labels)) bits, padded to byte boundary
    Each value indexes into that sub-block's label indices
```

If N == 1, the block is "solid" (all voxels have the same label) and no sub-block data follows.

## How Ground Truth Verification Works

The Go test `TestBlockCompression` in `compressed_test.go` does:

1. Load raw uint64 volume from `.dat.gz`
2. Compress: `MakeBlock(rawBytes, Point3d{64,64,64})`
3. Serialize: `block.MarshalBinary()`
4. Deserialize: `block2.UnmarshalBinary(serialized)`
5. Decompress: `block2.MakeLabelVolume()`
6. Assert voxel-for-voxel equality with the original raw volume

BRAID's `test_real_data.py::TestGroundTruthRoundtrip` replicates step 5-6 in Python:

1. Load `fib19-64x64x64-sample1-block.dat.gz` (the serialized compressed block — step 3 output)
2. Wrap in zstd (as `export-shards` does in the shard pipeline)
3. Decompress with `DVIDDecompressor.decompress_block()`
4. Load `fib19-64x64x64-sample1.dat.gz` (the raw ground truth — step 1)
5. Assert all 262,144 voxels match exactly

This proves the Python decompressor is equivalent to Go's `MakeLabelVolume()`.

## Go-produced Arrow IPC shard files

These files were copied from a real mCNS `export-shards` run and test the full cross-language pipeline: Go Arrow writer → Python Arrow reader → DVID decompression.

| File | Format | Records | Source |
|------|--------|---------|--------|
| `30720_24576_28672.arrow` | Arrow IPC Stream | 258 | mCNS scale 1, edge shard |
| `30720_24576_28672.csv` | CSV index (x,y,z,rec) | 258 rows | Companion index |

**Important**: DVID writes Arrow IPC **Streaming** format (sequential), not the File format (random-access with footer). BRAID's `ShardReader` handles both formats transparently.

The shard origin `(30720, 24576, 28672)` is in voxel coordinates at scale 1. Each shard covers a 2048-voxel cube, so chunk coordinates fall in `[origin/64, (origin+2048)/64)` per dimension. This is a sparse edge shard from the boundary of the brain volume, containing a mix of empty (label 0) and labeled regions.

### Alternate CSV index files

| File | Format | Purpose |
|------|--------|---------|
| `30720_24576_28672-offsets.csv` | Transitional (`x,y,z,rec,offset,size` + `# schema_size=688`) | Legacy format with byte offsets added; `rec` column instead of `batch_idx` |
| `30720_24576_28672-newformat.csv` | New format (`x,y,z,offset,size,batch_idx` + `# schema_size=688`) | New-format index required by `ShardRangeReader`; `batch_idx` replaces `rec` |

These files index the same 258 chunks as `30720_24576_28672.csv` but include
byte offsets into the Arrow IPC stream, enabling `ShardRangeReader`'s
byte-range access. The `-offsets.csv` is a transitional format (superset of
old columns); `-newformat.csv` is the canonical new format that `_parse_csv_index`
auto-detects.

The test `test_go_produced_shard.py` verifies:
- Arrow schema compatibility between Go writer and Python reader
- CSV index coordinates match Arrow record fields
- All 258 chunks decompress to 64x64x64 uint64 arrays
- Decompressed supervoxel values are subsets of each chunk's supervoxel list
- Agglomerated label mapping produces values from each chunk's labels list
