# tensorstore-export

Converts DVID `export-shards` Arrow IPC files into neuroglancer precomputed segmentation volumes using massively parallel Google Cloud Run workers.

## Overview

[DVID](https://github.com/janelia-flyem/dvid) exports segmentation data as spatially-partitioned Arrow IPC shard files via the `export-shards` RPC command. This project converts those files into [neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) volumes viewable in [Neuroglancer](https://github.com/google/neuroglancer). Designed for terabyte-scale datasets using hundreds of parallel Cloud Run workers.

[BRAID](braid/) (Block Records Arrow Indexed Dataset) reads shard files directly from GCS via `google-cloud-storage` — no temp files needed.

## Getting Started

### Prerequisites

- [pixi](https://pixi.sh) — manages Python, dependencies, and tasks
- [Docker](https://www.docker.com/) — builds Cloud Run container images
- [gcloud CLI](https://cloud.google.com/sdk) — deploys to Cloud Run

After installing gcloud, authenticate for both CLI commands and Python libraries:

```bash
gcloud auth login                        # for gcloud CLI commands
gcloud auth application-default login    # for Python libraries (GCS, TensorStore)
```

### Install

```bash
pixi install
pixi run build-braid-c  # build C decompressor extension
pixi run test-all       # 66 tests
```

### Deploy

Run `pixi run deploy` to be guided through all required configuration. It prompts for GCS paths, reads the neuroglancer volume spec JSON, writes the destination `info` file, builds the Docker image, and creates the Cloud Run job.

```bash
pixi run deploy
```

```
=== TensorStore Export — Cloud Run Deployment ===

--- GCP Settings ---
  PROJECT_ID [your-gcp-project]: my-project
  REGION [us-central1]: ↵

--- Data Settings ---
  SOURCE_PATH [gs://...]: gs://mybucket/exports/segmentation
  DEST_PATH [gs://...]: gs://mybucket/v1.0/precomputed

--- Neuroglancer Volume Spec ---
  NG_SPEC_PATH [ng-specs.json]: cns3-ng-specs.json

  Type: segmentation, data_type: uint64, 10 scale(s)
  Scale 0: 94088x78317x134576 @ 8nm, shard_bits=19 ...
  ...

--- Deployment ---
  SCALES [0,1]: ↵
  MEMORY [2Gi]: ↵
  ...

Writing neuroglancer info file ...
Building Docker image ...
Creating 10 per-scale Cloud Run jobs...
  tensorstore-dvid-export-s0: created
  tensorstore-dvid-export-s1: created
  ...

Done.
```

All values are saved to `.env` for future runs. You can also edit `.env` directly (see `.env.example` for all options).

### Generate Scales

After deploying, execute per-scale Cloud Run jobs. Each scale has its own job
(`{BASE_JOB_NAME}-s0`, `-s1`, ...) so they can run in parallel with independent
resource profiles:

```bash
# Process all scales from .env defaults
pixi run generate-scale

# Process s0 with 800 parallel workers
pixi run generate-scale --scales 0 --tasks 800

# Process multiple scales simultaneously
pixi run generate-scale --scales 0,1,2 --tasks 200

# Higher scales need more memory (larger blocks)
pixi run generate-scale --scales 3 --tasks 50 --memory 16Gi

# Generate scale 10 by downsampling the already-written scale 9
pixi run generate-scale --downres 10

# Export supervoxel IDs instead of agglomerated labels
pixi run generate-scale --scales 0 --label-type supervoxels
```

Options for `generate-scale`:

| Option | Description | Default |
|--------|-------------|---------|
| `--scales` | Scales to process from DVID shards | from `.env` |
| `--downres` | Output scales to generate by downsampling (e.g., `10` reads scale 9 to produce scale 10) | none |
| `--label-type` | `labels` (agglomerated, default) or `supervoxels` (raw IDs) | `labels` |
| `--tasks` | Number of parallel worker tasks | from `.env` |
| `--memory` | Memory per worker (e.g., `8Gi`) | from `.env` |
| `--cpu` | CPUs per worker | from `.env` |
| `--wait` | Block until the job completes | async (return immediately) |

By default, workers export **agglomerated labels** — the standard segmentation view where proofreading merges are applied. Use `--label-type supervoxels` to export the raw supervoxel IDs from the DVID blocks instead.

### Checking for Errors

Check for errors at any time — during execution for a live snapshot, or after completion for the final summary:

```bash
# Summary of errors across all scales (latest execution each)
pixi run export-errors

# Errors for a specific scale only
pixi run export-errors --scale 0

# Full details of every failed chunk
pixi run export-errors --details

# Errors from all executions (not just the latest)
pixi run export-errors --all
```

The script queries per-scale Cloud Run jobs (`{BASE_JOB_NAME}-s0`, `-s1`, ...),
auto-detects the latest execution for each, and aggregates results. Individual
chunk failures are logged as `"Chunk failed"` events with coordinates, so they
don't abort the entire shard.

### Multi-Scale Processing

Workers support two sources for each scale:

**From DVID export shards** (default): DVID pre-computes downsampled blocks at each scale. Workers ingest them directly — no computation needed, output exactly matches what DVID serves.

**From previous scale** (downres mode): For scales not materialized in DVID, `--downres N` generates scale N by reading the already-written scale N-1 from the destination volume and downsampling 2× in each dimension using majority vote. Scale N-1 must be fully written first.

### Memory Sizing

Arrow shard file sizes grow at coarser scales (from mCNS export data):

| Scale | Mean shard size | Max shard size | Recommended memory |
|-------|----------------|---------------|-------------------|
| 0     | 135 MB         | 470 MB        | 4 GiB             |
| 1     | 321 MB         | 898 MB        | 4 GiB             |
| 2     | 504 MB         | 1.8 GB        | 8 GiB             |
| 3     | 800 MB         | 3.8 GB        | 16 GiB            |
| 4+    | 1.4 GB         | 6.2 GB        | 16 GiB            |

Use `--memory` on `generate-scale` to set per-scale memory. Since each scale
has its own Cloud Run job, different scales can run simultaneously with
different memory profiles.

## Configuration

All settings live in `.env` (not committed). See `.env.example` for the full list.

| Variable | Description | Example |
|----------|-------------|---------|
| `SOURCE_PATH` | GCS URI to DVID shard export (contains s0/, s1/, ...) | `gs://mybucket/exports/seg` |
| `DEST_PATH` | GCS URI for neuroglancer precomputed output | `gs://mybucket/v1.0/precomputed` |
| `NG_SPEC_PATH` | Neuroglancer volume spec JSON (same as DVID export-shards) | `cns3-ng-specs.json` |
| `BASE_JOB_NAME` | Cloud Run job name prefix (per-scale: `{name}-s0`, `-s1`, ...) | `tensorstore-dvid-export` |
| `SCALES` | Scales to ingest from DVID shards | `0,1,2,3` |
| `DOWNRES_SCALES` | Scales to generate by downsampling previous scale | `10` |
| `TASKS` | Default parallel worker tasks per scale | `200` |
| `MEMORY` | Default memory per worker | `4Gi` |
| `CPU` | Default CPUs per worker | `2` |

## Architecture

```
DVID export-shards           Cloud Run Workers              Neuroglancer Volume
  Arrow IPC + CSV        →     BRAID reads from GCS      →   precomputed on GCS
  per-scale: s0/, s1/...       google-cloud-storage           multi-scale output
```

Each worker processes one shard at a time: download Arrow+CSV from GCS, decompress chunks via the DVID block decompressor, transpose ZYX→XYZ, and write to the neuroglancer precomputed volume via TensorStore. See [braid/docs/ARCHITECTURE.md](braid/docs/ARCHITECTURE.md) for I/O design decisions.

## Project Structure

```
tensorstore-export/
├── pixi.toml                        # Dependencies and tasks
├── .env.example                     # Configuration template
├── Dockerfile                       # Cloud Run container
├── scripts/
│   ├── deploy.py                   # Interactive deployment
│   ├── generate_scale.py           # Execute Cloud Run job
│   ├── export_errors.py            # Query error logs
│   └── setup_destination.py        # Info file setup (also called by deploy)
├── src/
│   ├── worker.py                   # Cloud Run worker
│   └── tensorstore_adapter.py      # TensorStore helpers
├── braid/                           # BRAID library (will be its own repo)
│   ├── src/braid/
│   │   ├── reader.py               # ShardReader (local + GCS)
│   │   ├── decompressor.py         # DVID block decompressor
│   │   └── exceptions.py
│   ├── tests/                      # 66 tests including ground truth verification
│   └── docs/ARCHITECTURE.md        # I/O design decisions
└── docs/
    ├── mCNS-ExportAnalysis.md      # Real export data analysis
    ├── ExportObservabilityPlan.md   # DVID export error detection plan
    └── ShardExportDesign.md        # System design document
```

## Related Projects

- [DVID](https://github.com/janelia-flyem/dvid) — Source data server with `export-shards` command
- [BRAID](https://github.com/JaneliaSciComp/braid) — Standalone Arrow shard reader (will be published separately)
- [TensorStore](https://github.com/google/tensorstore) — Multi-dimensional array storage
- [Neuroglancer](https://github.com/google/neuroglancer) — Web-based volumetric data viewer

## License

Janelia provides this under a modified BSD 3-Clause License.
