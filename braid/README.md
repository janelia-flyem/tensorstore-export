# BRAID
A Python library for reading segmentation data from spatially-partitioned sharded Arrow files with efficient block-wise access and support for multiple levels of labels and compression backends.

## Overview

BRAID provides an interface for efficient Arrow-based reading of large-scale (> TB) segmentation data. It was created to handle management of segmentation between different services, exploiting Arrow's zero-copy, multi-language ecosystem. In particular, it can be combined with `virtual_chunked` driver functions in [tensorstore](https://github.com/Google/tensorstore) to allow very memory-efficient reformatting to a variety of large-scale segmentation formats (e.g., zarr version 3 or neuroglancer precomputed volumes), see [tensorstore-export](https://github.com/JaneliaSciComp/tensorstore-export). From a data perspective, there are three levels of organization from small to large scale:

1. **Block Records**: Individual chunks of 3D compressed segmentation data
2. **Shard Files**: Collections of block records in Arrow format with accompanying CSV chunk indices
3. **Dataset**: Multiple shard files that together form the complete volume

It supports:

- **Efficient packaging**: Shard files package billions of block records improving object counts.
- **Block-wise access**: Read individual blocks (chunks) by coordinates
- **Fast lookups**: Uses CSV indices for O(1) block coordinate lookups
- **Levels of labels**: Choose between agglomerated labels or supervoxel (atomic) labels
- **Pluggable compression**: Supports Zstd + DVID segmentation compression with extensible backend system
- **C extensions**: High-performance DVID decompression and DVID-to-neuroglancer transcoding
- **Error handling**: Comprehensive error handling and validation

**BRAID** is an acronym for **Block Records Arrow Indexed Dataset**. The name evokes the concept of braiding individual block records into shard files, which in turn are like braids that come together to form the complete dataset. This metaphor captures both the technical structure (records → shards → dataset) and the parallel nature of how workers process large-scale datasets. Currently, the shape of the braids are shard subvolumes as defined by a given dataset's parameters.

## Installation

BRAID is not published on PyPI. It is installed as part of the
[tensorstore-export](https://github.com/JaneliaSciComp/tensorstore-export)
project via [pixi](https://pixi.sh/):

```bash
# From the tensorstore-export root:
pixi install              # installs braid as an editable dependency
pixi run build-braid-c    # build C extensions (decompressor + transcoder)
```

For standalone development:

```bash
cd braid
pip install -e .[dev]       # editable install with dev dependencies
pip install -e .[gcs]       # with GCS support
```

## Quick Start

```python
from braid import ShardReader, LabelType

# Initialize reader with Arrow and CSV files
reader = ShardReader("shard_0_0_0.arrow", "shard_0_0_0.csv")

# Read a chunk with agglomerated labels
chunk_data = reader.read_chunk(64, 128, 0, label_type=LabelType.LABELS)
print(f"Chunk shape: {chunk_data.shape}")  # (64, 64, 64)
print(f"Data type: {chunk_data.dtype}")    # uint64

# Custom chunk shape (default is (64, 64, 64))
chunk_data = reader.read_chunk(64, 128, 0, chunk_shape=(32, 32, 32))

# Read the same chunk with supervoxel labels
supervoxel_data = reader.read_chunk(64, 128, 0, label_type=LabelType.SUPERVOXELS)

# Check available chunks
print(f"Total chunks: {reader.chunk_count}")
print(f"Available coordinates: {reader.available_chunks[:10]}")
print(f"Is empty shard: {reader.is_empty}")
```

## Reader Variants

BRAID provides two reader classes with the same public interface:

| Class | Memory | I/O | Use case |
|-------|--------|-----|----------|
| `ShardReader` | Full Arrow file in RAM | One download per shard | Worker processing all chunks in a shard |
| `ShardRangeReader` | Schema + one record batch | Byte-range reads per batch | Memory-constrained access to individual chunks |

`ShardRangeReader` requires the new-format CSV index (with `offset`, `size`,
and `batch_idx` columns plus a `# schema_size=N` header). It caches the most
recently fetched record batch, so consecutive chunks from the same batch avoid
extra I/O.

```python
from braid import ShardRangeReader, LabelType

reader = ShardRangeReader(
    "gs://bucket/shards/s0/0_0_0.arrow",
    "gs://bucket/shards/s0/0_0_0.csv",
)
for cx, cy, cz in reader.available_chunks:
    data = reader.read_chunk(cx, cy, cz)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design rationale.

## Features

### Label Type Selection

BRAID supports multiple label types available in shard files:

- **`LabelType.LABELS`**: Agglomerated label IDs (post-proofreading)
- **`LabelType.SUPERVOXELS`**: Original supervoxel IDs (pre-proofreading)

```python
# Read agglomerated labels
labels = reader.read_chunk(x, y, z, label_type=LabelType.LABELS)

# Read supervoxel labels
supervoxels = reader.read_chunk(x, y, z, label_type=LabelType.SUPERVOXELS)
```

### Chunk Information

Get metadata about chunks without decompressing them:

```python
if reader.has_chunk(x, y, z):
    info = reader.get_chunk_info(x, y, z)
    print(f"Coordinates: {info['coordinates']}")
    print(f"Record index: {info['record_index']}")
    print(f"Labels count: {info['labels_count']}")
    print(f"Supervoxels count: {info['supervoxels_count']}")
    print(f"Compressed size: {info['compressed_size']} bytes")
    print(f"Uncompressed size: {info['uncompressed_size']} bytes")

    # Get raw chunk data without decompression
    raw_data = reader.read_chunk_raw(x, y, z)
    print(f"Raw labels: {raw_data['labels']}")
    print(f"Raw supervoxels: {raw_data['supervoxels']}")
```

### Error Handling

BRAID provides specific exceptions for different error conditions:

```python
from braid import BraidError, ChunkNotFoundError, DecompressionError

try:
    chunk = reader.read_chunk(x, y, z)
except ChunkNotFoundError:
    print(f"Chunk ({x}, {y}, {z}) not found in shard")
except DecompressionError as e:
    print(f"Failed to decompress chunk: {e}")
except BraidError as e:
    print(f"BRAID error: {e}")  # catches all braid-specific errors
```

## File Format

BRAID reads sharded Arrow files with CSV coordinate indices:

### Arrow IPC File (`.arrow`)
Contains the actual chunk data with schema:
- `chunk_x`, `chunk_y`, `chunk_z`: Chunk coordinates (int32)
- `labels`: Agglomerated label IDs (list of uint64)
- `supervoxels`: Supervoxel IDs (list of uint64)
- `dvid_compressed_block`: Compressed chunk data (binary)
- `uncompressed_size`: Size after decompression (uint32)

### CSV Index File (`.csv`)

BRAID auto-detects two CSV index formats:

**Old format** (legacy, used by `ShardReader`):
```csv
x,y,z,rec
64,64,0,0
128,64,0,1
64,128,0,2
```

**New format** (used by both readers, required for `ShardRangeReader`):
```csv
# schema_size=688
x,y,z,offset,size,batch_idx
64,64,0,688,792,0
128,64,0,1480,2624,0
64,128,0,4104,1192,0
```

The `# schema_size=N` header declares how many bytes of Arrow schema
precede the first record batch, enabling byte-range reads without
downloading the full file. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for details.

## Compression and Transcoding

### DVIDDecompressor

Decompresses DVID compressed segmentation blocks (zstd + DVID format).
Uses a C extension (`libdvid_decompress.so`) for the inner loop when
available, falling back to pure Python. See
[docs/DVID-block-decompression.md](docs/DVID-block-decompression.md).

```python
from braid import DVIDDecompressor

decompressor = DVIDDecompressor()
volume = decompressor.decompress_block(
    compressed_data,
    agglo_labels=labels,
    supervoxels=supervoxels,
    block_shape=(64, 64, 64),
)
```

### CSEGEncoder

Fused DVID-to-neuroglancer `compressed_segmentation` transcoder. Requires
the C extension (`libbraid_codec.so`); build with `pixi run build-braid-c`.
See [docs/DVID-to-cseg-transcoder.md](docs/DVID-to-cseg-transcoder.md).

```python
from braid import CSEGEncoder

encoder = CSEGEncoder()

# Encode a uint64 volume directly
cseg_bytes = encoder.encode_chunk(volume, block_size=(8, 8, 8))

# Fused path: DVID block -> compressed_segmentation (no intermediate array)
cseg_bytes = encoder.dvid_to_cseg(
    dvid_data,
    supervoxels=sv_ids,
    agglo_labels=agglo_ids,
    zstd_input=True,
    gzip_output=True,
)
```

## C Extensions

BRAID includes two C shared libraries for performance-critical paths:

| Library | Purpose | Speedup |
|---------|---------|---------|
| `libdvid_decompress.so` | DVID block decompression inner loop | ~600x vs Python |
| `libbraid_codec.so` | DVID-to-cseg fused transcoder + cseg encoder | Required for `CSEGEncoder` |

Build both with:
```bash
pixi run build-braid-c
```

Source files are in `csrc/`. When the `.so` files are not available,
`DVIDDecompressor` falls back to pure Python; `CSEGEncoder` raises
`RuntimeError` at construction.

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.py`: Basic local file reading
- `gcs_usage.py`: Reading files from Google Cloud Storage

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for I/O design decisions,
the GCS backend choice, and future streaming optimization plans.

## Development

BRAID is developed as part of the
[tensorstore-export](https://github.com/JaneliaSciComp/tensorstore-export)
repository. The recommended workflow uses [pixi](https://pixi.sh/):

```bash
# Clone the parent repository
git clone https://github.com/JaneliaSciComp/tensorstore-export.git
cd tensorstore-export

# Install all dependencies (including braid as editable)
pixi install

# Build C extensions
pixi run build-braid-c

# Run tests
pixi run test-braid    # unit + integration tests
pixi run test-bench    # C vs Python benchmarks
pixi run test-e2e      # end-to-end precomputed roundtrip
pixi run test-all      # everything (112 tests)

# Lint
pixi run lint          # ruff check
```

For standalone development without pixi:
```bash
cd braid
pip install -e .[dev]
pytest tests/
```

## License

Janelia provides this under a [modified BSD 3-Clause License](LICENSE).
