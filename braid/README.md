# BRAID
A Python library for reading segmentation data from spatially-partitioned sharded Arrow files with efficient block-wise access and support for multiple label types and compression backends.

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
- **Error handling**: Comprehensive error handling and validation

**BRAID** is an acronym for **Block Records Arrow Indexed Dataset**. The name evokes the concept of braiding individual block records into shard files, which in turn are like braids that come together to form the complete dataset. This metaphor captures both the technical structure (records → shards → dataset) and the parallel nature of how workers process large-scale datasets. Currently, the shape of the braids are shard subvolumes as defined by a given dataset's parameters.

## Installation

```bash
pip install braid
```

For GCS support:
```bash
pip install braid[gcs]
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

# Read the same chunk with supervoxel labels
supervoxel_data = reader.read_chunk(64, 128, 0, label_type=LabelType.SUPERVOXELS)

# Check available chunks
print(f"Total chunks: {reader.chunk_count}")
print(f"Available coordinates: {reader.available_chunks[:10]}")
```

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
# Check if a chunk exists
if reader.has_chunk(x, y, z):
    # Get chunk metadata
    info = reader.get_chunk_info(x, y, z)
    print(f"Labels count: {info['labels_count']}")
    print(f"Compressed size: {info['compressed_size']} bytes")
    
    # Get raw chunk data
    raw_data = reader.read_chunk_raw(x, y, z)
    print(f"Raw labels: {raw_data['labels']}")
    print(f"Raw supervoxels: {raw_data['supervoxels']}")
```

### Error Handling

BRAID provides specific exceptions for different error conditions:

```python
from braid import ChunkNotFoundError, DecompressionError

try:
    chunk = reader.read_chunk(x, y, z)
except ChunkNotFoundError:
    print(f"Chunk ({x}, {y}, {z}) not found in shard")
except DecompressionError as e:
    print(f"Failed to decompress chunk: {e}")
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
Maps chunk coordinates to Arrow record indices:
```csv
x,y,z,rec
64,64,0,0
128,64,0,1
64,128,0,2
```

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.py`: Basic local file reading
- `gcs_usage.py`: Reading files from Google Cloud Storage

## Compression Backends

BRAID currently supports:

- **DVID Compression**: The custom DVID block compression format with zstd
- **Extensible Design**: Plugin architecture for additional compression backends

The DVID compression format uses:
- Block-level label lists with sub-block indices
- Variable-bit encoding for efficient storage
- Support for sparse sub-blocks
- zstd compression of the structured data

## Development

```bash
# Clone repository
git clone https://github.com/your-org/braid.git
cd braid

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## License

Janelia provides this under a [modified BSD 3-Clause License](LICENSE). 

