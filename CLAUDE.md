# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a catch-all development repository containing three main components:

1. **research/**: Code and documentation for AI-driven development and analysis
2. **braid/**: Initial formulation of the BRAID library (will be committed to https://github.com/JaneliaSciComp/braid)
3. **src/**: The main tensorstore-export application for massively parallel processing of Arrow shard files

The primary application (`src/`) uses TensorStore and BRAID to run Google Cloud Run tasks that export Arrow shard files (with CSV indices) from GCS to Neuroglancer precomputed segmentation volumes.

## Development Commands

### BRAID Library Development (braid/)
```bash
# Navigate to braid directory
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

# Run tests with coverage
pytest -v --cov=braid --cov-report=term-missing
```

### TensorStore Export Application (src/)
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires environment variables)
python main.py

# Docker build
docker build -t tensorstore-export .

# Deploy to Cloud Run
./scripts/deploy.sh
```

## Architecture

### TensorStore Export System (`src/`)
The main application creates massively parallel Cloud Run workers that:

1. **Poll GCS**: Workers look for available shards in `unprocessed/` directory
2. **Process Shards**: Each worker takes one shard (Arrow + CSV files) and processes it completely
3. **Export to Neuroglancer**: Outputs neuroglancer precomputed segmentation volume shards
4. **Job Management**: Uses atomic file movement (`unprocessed/` → `processing/` → `finished/`) to prevent race conditions

#### Key Components
- `worker.py`: Main Cloud Run worker with GCS polling and job queue management
- `shard_reader.py`: Interface for reading DVID Arrow shard files
- `tensorstore_adapter.py`: TensorStore integration using `virtual_chunked` driver for memory efficiency

### BRAID Library (`braid/`)
A standalone Python library for efficient reading of segmentation data from spatially-partitioned sharded Arrow files. Features:

- **Block-wise Access**: Read individual 64x64x64 chunks by coordinates
- **Dual Label Support**: Both agglomerated labels and supervoxel data
- **Memory Efficient**: Decompresses only requested blocks
- **DVID Compression**: Handles custom DVID block compression with zstd

### Data Flow Architecture
1. **Input**: DVID exports produce paired files per shard:
   - `.arrow`: Arrow IPC file with compressed segmentation blocks
   - `.csv`: Coordinate index mapping (x,y,z,rec)
2. **Processing**: Cloud Run workers use BRAID + TensorStore's `virtual_chunked` driver
3. **Output**: Neuroglancer precomputed volume shards in target GCS location

## File Structure

```
├── main.py                    # Cloud Run entry point
├── requirements.txt           # Dependencies for tensorstore-export
├── Dockerfile                 # Container for Cloud Run deployment
├── src/                       # TensorStore export application
│   ├── worker.py             # Main worker with GCS job management
│   ├── shard_reader.py       # DVID shard reading interface
│   └── tensorstore_adapter.py # TensorStore virtual_chunked integration
├── braid/                     # BRAID library (future separate repo)
│   ├── src/braid/            # Library source code
│   ├── pyproject.toml        # Build configuration
│   ├── README.md             # Library documentation
│   └── tests/                # Unit tests
├── research/                  # AI development tools and analysis
├── docs/                     # Technical documentation
│   ├── ShardExportDesign.md  # Distributed system architecture
│   ├── EfficientWrites.md    # Memory optimization strategies
│   └── ArrowParallelWrite.md # Arrow integration details
└── scripts/deploy.sh         # Cloud Run deployment script
```

## Distributed Processing Strategy

### Job Queue Management
- Workers continuously poll GCS `unprocessed/` prefix for available shards
- Atomic file movement prevents race conditions between workers
- Each worker processes exactly one shard before polling for the next
- Failed processing moves shards to `failed/` prefix for debugging

### Scalability
- Designed for thousands of parallel Cloud Run workers
- Each worker has minimal memory requirements due to block-wise processing
- No central coordinator required - workers self-organize via GCS file operations

## Configuration

### Environment Variables (Cloud Run Workers)
- `SOURCE_BUCKET`: GCS bucket containing Arrow shard files in `unprocessed/`
- `DEST_BUCKET`: Target GCS bucket for Neuroglancer volume
- `DEST_PATH`: Path within destination bucket
- `TOTAL_VOLUME_SHAPE`: Volume dimensions as "z,y,x"
- `SHARD_SHAPE`: Shard dimensions (default: 2048,2048,2048)
- `CHUNK_SHAPE`: Block dimensions (default: 64,64,64)

### Data Formats
- **Arrow Schema**: `chunk_x`, `chunk_y`, `chunk_z`, `labels`, `supervoxels`, `dvid_compressed_block`, `uncompressed_size`
- **CSV Index**: `x,y,z,rec` mapping chunk coordinates to Arrow record indices
- **Output**: Neuroglancer precomputed format with info files and shard data

## Memory Efficiency Design

The system prioritizes minimal memory usage for high parallelism:
- TensorStore `virtual_chunked` driver provides on-demand chunk loading
- BRAID library decompresses only requested 64x64x64 blocks
- Workers avoid loading entire 2048³ shards into memory
- Supports processing TB-scale datasets with modest Cloud Run memory allocations