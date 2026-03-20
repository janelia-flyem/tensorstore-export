# tensorstore-export

Converts DVID `export-shards` Arrow IPC files into neuroglancer precomputed segmentation volumes using massively parallel Google Cloud Run workers.

## Overview

[DVID](https://github.com/janelia-flyem/dvid) exports segmentation data as spatially-partitioned Arrow IPC shard files via the `export-shards` RPC command. This project converts those files into [neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) volumes viewable in [Neuroglancer](https://github.com/google/neuroglancer). Designed for terabyte-scale datasets using hundreds of parallel Cloud Run workers.

[BRAID](braid/) (Block Records Arrow Indexed Dataset) reads shard files directly from GCS via PyArrow's native C++ filesystem — no temp files or intermediate copies.

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
pixi run test-all   # 65 tests, ~55 seconds
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
Creating Cloud Run job ...

Done.
```

All values are saved to `.env` for future runs. You can also edit `.env` directly (see `.env.example` for all options).

### Generate Scales

After deploying, execute the Cloud Run job:

```bash
# Process scales from .env defaults (typically scales 0,1 with agglomerated labels)
pixi run generate-scale

# Process specific scales
pixi run generate-scale --scales 0,1

# Process higher scales with more memory
pixi run generate-scale --scales 2,3 --memory 4Gi

# Generate scale 10 by downsampling the already-written scale 9
# (for scales not exported by DVID, e.g., when MaxDownresLevel < spec scale count)
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
| `--memory` | Memory per worker override | from `.env` |
| `--wait` | Block until the job completes | async (return immediately) |

By default, workers export **agglomerated labels** — the standard segmentation view where proofreading merges are applied. Use `--label-type supervoxels` to export the raw supervoxel IDs from the DVID blocks instead.

### Multi-Scale Processing

Workers support two sources for each scale:

**From DVID export shards** (default): DVID pre-computes downsampled blocks at each scale. Workers ingest them directly — no computation needed, output exactly matches what DVID serves.

**From previous scale** (downres mode): For scales not materialized in DVID, `--downres N` generates scale N by reading the already-written scale N-1 from the destination volume and downsampling 2× in each dimension using majority vote. Scale N-1 must be fully written first.

### Memory Sizing

Shard sizes grow at lower resolutions. The deploy script estimates sizes from the sharding params and suggests memory per worker:

| Scales | Typical shard size | Recommended memory |
|--------|-------------------|-------------------|
| 0–1 | 135–321 MB | 2 GiB |
| 2–3 | 504–799 MB | 4 GiB |
| 4+ | 1.4+ GB | 8 GiB |

Use `--memory` on `generate-scale` to override for a specific execution, or deploy separate jobs per memory tier.

## Configuration

All settings live in `.env` (not committed). See `.env.example` for the full list.

| Variable | Description | Example |
|----------|-------------|---------|
| `SOURCE_PATH` | GCS URI to DVID shard export (contains s0/, s1/, ...) | `gs://mybucket/exports/seg` |
| `DEST_PATH` | GCS URI for neuroglancer precomputed output | `gs://mybucket/v1.0/precomputed` |
| `NG_SPEC_PATH` | Neuroglancer volume spec JSON (same as DVID export-shards) | `cns3-ng-specs.json` |
| `SCALES` | Scales to ingest from DVID shards | `0,1` |
| `DOWNRES_SCALES` | Scales to generate by downsampling previous scale | `10` |
| `MEMORY` | Memory per Cloud Run worker | `2Gi` |
| `TASKS` | Number of parallel Cloud Run worker tasks | `200` |

## Architecture

```
DVID export-shards           Cloud Run Workers              Neuroglancer Volume
  Arrow IPC + CSV        →     BRAID reads from GCS      →   precomputed on GCS
  per-scale: s0/, s1/...       PyArrow C++ native I/O         multi-scale output
```

BRAID reads Arrow IPC and CSV files directly from GCS using `pyarrow.fs` — data flows through Arrow's C++ runtime with zero Python-level copies. Each worker processes one shard at a time: load table into memory, decompress chunks via the DVID block decompressor, transpose ZYX→XYZ, and write to the neuroglancer precomputed volume via TensorStore.

## Project Structure

```
tensorstore-export/
├── pixi.toml                        # Dependencies and tasks
├── .env.example                     # Configuration template
├── Dockerfile                       # Cloud Run container
├── scripts/
│   ├── deploy.py                   # Interactive deployment
│   └── setup_destination.py        # Info file setup (also called by deploy)
├── src/
│   ├── worker.py                   # Cloud Run worker
│   └── tensorstore_adapter.py      # TensorStore helpers
├── braid/                           # BRAID library
│   ├── src/braid/
│   │   ├── reader.py               # ShardReader (local + GCS via pyarrow.fs)
│   │   ├── decompressor.py         # DVID block decompressor
│   │   └── exceptions.py
│   └── tests/                      # 65 tests including ground truth verification
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
