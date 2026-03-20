# tensorstore-export

Converts DVID `export-shards` Arrow IPC files into neuroglancer precomputed segmentation volumes using massively parallel Google Cloud Run workers.

## Overview

[DVID](https://github.com/janelia-flyem/dvid) can export segmentation data as spatially-partitioned Arrow IPC files using the `export-shards` RPC command. Each shard file contains compressed 64x64x64 blocks covering a subvolume of the dataset, accompanied by a CSV coordinate index.

This project converts those shard files into [neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) volumes that can be viewed in [Neuroglancer](https://github.com/google/neuroglancer). The system is designed for terabyte-scale datasets (e.g., the male CNS at 94088 x 78317 x 134576 uint64 voxels) using thousands of parallel Cloud Run workers.

## Architecture

1. **[BRAID](braid/)** (Block Records Arrow Indexed Dataset) — A standalone Python library for reading sharded Arrow IPC files with block-level random access and DVID block decompression. Reads directly from local disk or GCS via PyArrow's native filesystem — no temp files or intermediate copies.

2. **Cloud Run Worker** (`src/worker.py`) — Distributed worker that reads DVID shard files from GCS via BRAID, decompresses each chunk, and writes to the neuroglancer precomputed volume via TensorStore. Scale-aware: can process DVID export shards at any scale, or generate downsampled scales from the previous scale's data.

3. **Deploy Script** (`scripts/deploy.py`) — Interactive deployment tool that reads configuration from `.env`, parses the neuroglancer volume spec JSON, and creates Cloud Run jobs with appropriate memory sizing per scale.

## Data Flow

```
DVID export-shards          Cloud Run Workers              Neuroglancer Volume
  (Arrow IPC + CSV)    -->    (BRAID + TensorStore)    -->   (precomputed on GCS)
  per-scale dirs on GCS       read directly from GCS         shared output volume
  s0/, s1/, s2/, ...          no temp files                  multi-scale
```

### Multi-Scale Processing

Workers support two modes for generating each scale:

**From DVID export shards** (default): DVID's `export-shards` pre-computes downsampled blocks at each scale using majority-vote downres. Workers ingest these directly — no computation needed, and the output exactly matches what DVID serves.

**From previous scale** (downres mode): For scales not materialized in DVID (e.g., if `MaxDownresLevel` was set lower than the number of scales in the spec), workers can generate scale N by reading scale N-1 from the neuroglancer precomputed volume on GCS and downsampling with `ts.downsample(method='mode')`. This requires scale N-1 to be fully written first.

## Quick Start

### Prerequisites

- [pixi](https://pixi.sh) — package manager (handles Python, dependencies, and tasks)
- [Docker](https://www.docker.com/) — for building Cloud Run container images
- [gcloud CLI](https://cloud.google.com/sdk) — for deploying to Cloud Run

### Install and Test

```bash
# Install all dependencies (Python, BRAID, TensorStore, etc.)
pixi install

# Run all tests (65 tests, ~55 seconds)
pixi run test-all
```

### Configure

Copy `.env.example` to `.env` and fill in your GCP project and data paths:

```bash
cp .env.example .env
# Edit .env with your settings
```

The neuroglancer volume spec JSON (same file used for DVID's `export-shards` command) is the single source of truth for volume geometry and sharding parameters. Point `NG_SPEC_PATH` in `.env` to this file.

### Deploy

```bash
# Interactive deployment — prompts for settings with .env defaults,
# writes neuroglancer info file, builds Docker image, creates Cloud Run job
pixi run deploy
```

The deploy script shows per-scale shard size estimates and recommends memory tiers:

```
--- Neuroglancer Volume Spec ---
  Reading ng-specs.json...
  Type: segmentation, data_type: uint64, 10 scale(s)
  Scale 0: 94088x78317x134576 @ 8nm, shard_bits=19 ...

  Estimated shard sizes (from sharding params):
    Scale 0: shard=2048^3, up to 32768 chunks, ~160 MB max
    Scale 1: shard=2048^3, up to 32768 chunks, ~320 MB max
    ...
```

### Execute

```bash
# Run the Cloud Run job
gcloud run jobs execute <job-name> --region=<region> --project=<project>
```

### One-Time Setup

Before the first deployment, the neuroglancer `info` file must exist at the destination:

```bash
# Write info file to GCS (also done automatically by pixi run deploy)
pixi run setup-destination
```

## Configuration

All deployment-specific values come from `.env` (not committed) and the ng spec JSON. Nothing is hardcoded to a specific dataset.

| Variable | Description | Example |
|----------|-------------|---------|
| `PROJECT_ID` | GCP project ID | `my-project` |
| `REGION` | GCP region | `us-central1` |
| `SOURCE_PATH` | GCS URI to shard export (contains s0/, s1/, ...) | `gs://mybucket/exports/segmentation` |
| `DEST_PATH` | GCS URI for neuroglancer precomputed output | `gs://mybucket/v1.0/precomputed` |
| `NG_SPEC_PATH` | Local path to neuroglancer volume spec JSON | `ng-specs.json` |
| `SCALES` | Scales to process from DVID shards | `0,1` |
| `DOWNRES_SCALES` | Scales to generate by downsampling previous scale | `10` |
| `PARALLELISM` | Number of parallel Cloud Run workers | `200` |
| `MEMORY` | Memory per worker | `2Gi` |

### Memory Sizing by Scale

Shard sizes grow at lower resolutions because each shard covers more physical volume. Recommended memory per worker:

| Scales | Typical shards | Recommended memory |
|--------|---------------|-------------------|
| 0–1 | 135–321 MB median | 2 GiB |
| 2–3 | 504–799 MB median | 4 GiB |
| 4+ | 1.4+ GB median | 8 GiB |

Deploy separate Cloud Run jobs per memory tier, or use a single job sized for the largest scale.

## Shard File Format

Each DVID shard consists of two files written in Arrow IPC Streaming format:

- **`{origin}.arrow`** — Arrow IPC stream containing compressed segmentation blocks
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
| `uncompressed_size` | uint32 | Size of DVID block before zstd |

BRAID reads these files directly from GCS using PyArrow's native C++ filesystem — no temp files, no Python-level copies. Arrow IPC data flows through Arrow's C++ runtime; CSV indices are parsed by `pyarrow.csv`.

## Project Structure

```
tensorstore-export/
├── main.py                          # Cloud Run entry point
├── pixi.toml                        # Pixi workspace (deps, tasks)
├── Dockerfile                       # Container for Cloud Run
├── .env.example                     # Configuration template
├── scripts/
│   ├── deploy.py                   # Interactive deployment
│   └── setup_destination.py        # One-time info file setup
├── src/
│   ├── worker.py                   # Cloud Run worker
│   └── tensorstore_adapter.py      # TensorStore helpers
├── braid/                           # BRAID library
│   ├── src/braid/
│   │   ├── reader.py               # ShardReader (local + GCS)
│   │   ├── decompressor.py         # DVID block decompressor
│   │   └── exceptions.py           # Custom exceptions
│   └── tests/
│       ├── test_real_data.py       # Ground truth vs Go decompressor
│       ├── test_go_produced_shard.py  # Real DVID export shard tests
│       ├── test_e2e_precomputed.py # Full pipeline tests
│       └── test_data/              # Real DVID test data
└── docs/
    ├── mCNS-ExportAnalysis.md      # Real export data analysis
    ├── ExportObservabilityPlan.md   # DVID export error detection
    ├── ShardExportDesign.md         # Distributed system design
    └── ExportShards.md             # DVID export format spec
```

## Related Projects

- [DVID](https://github.com/janelia-flyem/dvid) — Source data server with `export-shards` command
- [BRAID](https://github.com/JaneliaSciComp/braid) — Standalone library (will be published separately)
- [TensorStore](https://github.com/google/tensorstore) — Efficient multi-dimensional array storage
- [Neuroglancer](https://github.com/google/neuroglancer) — Web-based volumetric data viewer

## License

Janelia provides this under a modified BSD 3-Clause License.
