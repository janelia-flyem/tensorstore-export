# tensorstore-export

Converts DVID `export-shards` Arrow IPC files into neuroglancer precomputed segmentation volumes using massively parallel Google Cloud Run workers.

## Overview

[DVID](https://github.com/janelia-flyem/dvid) can export segmentation data as spatially-partitioned Arrow IPC files using the `export-shards` RPC command. Each shard file contains compressed 64x64x64 blocks covering a subvolume of the dataset, accompanied by a CSV coordinate index.

This project converts those shard files into [neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) volumes that can be viewed in [Neuroglancer](https://github.com/google/neuroglancer). The system is designed for terabyte-scale datasets (e.g., the male CNS at 94088 x 78317 x 134576 uint64 voxels) using thousands of parallel Cloud Run workers.

## Architecture

The system has three components:

1. **[BRAID](braid/)** (Block Records Arrow Indexed Dataset) — A standalone Python library for reading sharded Arrow IPC files with block-level random access and DVID block decompression.

2. **TensorStore Adapter** (`src/tensorstore_adapter.py`) — Bridges BRAID readers to TensorStore's `virtual_chunked` driver, enabling memory-efficient on-demand chunk loading without materializing full shard volumes.

3. **Cloud Run Worker** (`src/worker.py`) — Distributed worker that polls for available shards on GCS, processes them, and writes to a shared neuroglancer precomputed volume.

## Data Flow

```
DVID export-shards      Cloud Run Workers         Neuroglancer Volume
  (Arrow IPC + CSV)  -->  (BRAID + TensorStore)  -->  (precomputed on GCS)
       |                        |                            |
  Per-shard files         virtual_chunked             Shared output
  on GCS                  block-wise reads            volume on GCS
```

### Job Coordination

Workers self-coordinate without a central orchestrator using GCS prefix-based state:

- `gs://<bucket>/unprocessed/` — shard files waiting to be processed
- `gs://<bucket>/processing/` — claimed by a worker (atomic move acts as lock)
- `gs://<bucket>/finished/` — successfully processed

If a move fails due to contention, the worker tries the next shard.

## Shard File Format

Each DVID shard consists of two files:

- **`{origin}.arrow`** — Arrow IPC file containing compressed segmentation blocks
- **`{origin}.csv`** — CSV index mapping chunk coordinates to Arrow record indices

### Arrow Schema

| Field | Type | Description |
|-------|------|-------------|
| `chunk_x` | int32 | X chunk coordinate |
| `chunk_y` | int32 | Y chunk coordinate |
| `chunk_z` | int32 | Z chunk coordinate |
| `labels` | list\<uint64\> | Agglomerated label IDs |
| `supervoxels` | list\<uint64\> | Original supervoxel IDs |
| `dvid_compressed_block` | binary | Zstd-compressed DVID block data |
| `uncompressed_size` | uint32 | Size before compression |

### CSV Index

```csv
x,y,z,rec
0,0,0,0
64,0,0,1
64,64,0,2
```

## Quick Start

### Prerequisites

- Python >= 3.10
- Google Cloud SDK (for deployment)
- Docker (for Cloud Run)

### Install Dependencies

```bash
# Main application
pip install -r requirements.txt

# BRAID library (for development)
cd braid && pip install -e .[dev]
```

### Local Development

```bash
# Run BRAID tests
cd braid && pytest -v

# Run with coverage
cd braid && pytest -v --cov=braid --cov-report=term-missing
```

### Deploy to Cloud Run

Configure environment variables in `scripts/deploy.sh`, then:

```bash
# Set your GCP project
export PROJECT_ID=your-gcp-project
export SOURCE_BUCKET=your-source-bucket
export DEST_BUCKET=your-dest-bucket

./scripts/deploy.sh
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SOURCE_BUCKET` | GCS bucket with Arrow shard files | required |
| `DEST_BUCKET` | GCS bucket for neuroglancer output | required |
| `DEST_PATH` | Path within destination bucket | required |
| `TOTAL_VOLUME_SHAPE` | Volume dimensions as "z,y,x" | required |
| `SHARD_SHAPE` | Shard dimensions | 2048,2048,2048 |
| `CHUNK_SHAPE` | Block dimensions | 64,64,64 |
| `RESOLUTION` | Voxel resolution in nm | 8,8,8 |

## Multi-Scale Volumes

The distributed pipeline processes only full-resolution (Scale 0) data. Downsampled scales are generated in a separate post-processing step using TensorStore's built-in downsampling, reading from the completed Scale 0 data on GCS. See `docs/ShardExportDesign.md` for details.

## Project Structure

```
tensorstore-export/
├── main.py                      # Cloud Run entry point
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container for Cloud Run
├── scripts/deploy.sh            # Deployment script
├── src/                         # Main application
│   ├── worker.py               # Cloud Run worker with GCS job management
│   ├── shard_reader.py         # DVID shard reading interface
│   └── tensorstore_adapter.py  # TensorStore virtual_chunked integration
├── braid/                       # BRAID library (future: github.com/JaneliaSciComp/braid)
│   ├── src/braid/              # Library source
│   │   ├── reader.py           # ShardReader class
│   │   ├── decompressor.py     # DVID block decompressor
│   │   └── exceptions.py       # Custom exceptions
│   ├── tests/                  # Test suite
│   └── pyproject.toml          # Package configuration
├── docs/                        # Technical documentation
│   ├── ShardExportDesign.md    # Distributed system architecture
│   ├── ExportShards.md         # DVID export format specification
│   ├── EfficientWrites.md      # Memory optimization strategies
│   └── ArrowParallelWrite.md   # Arrow integration details
└── config/                      # Neuroglancer volume specifications
```

## Related Projects

- [DVID](https://github.com/janelia-flyem/dvid) — Source data server with `export-shards` command
- [BRAID](https://github.com/JaneliaSciComp/braid) — Standalone library (will be published separately)
- [TensorStore](https://github.com/google/tensorstore) — Efficient multi-dimensional array storage
- [Neuroglancer](https://github.com/google/neuroglancer) — Web-based volumetric data viewer

## License

Janelia provides this under a modified BSD 3-Clause License.
