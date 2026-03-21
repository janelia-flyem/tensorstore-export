# tensorstore-export

Converts DVID `export-shards` Arrow IPC files into neuroglancer precomputed segmentation volumes using massively parallel Google Cloud Run workers.

## Overview

[DVID](https://github.com/janelia-flyem/dvid) exports segmentation data as spatially-partitioned Arrow IPC shard files via the `export-shards` RPC command. This project converts those files into [neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) volumes viewable in [Neuroglancer](https://github.com/google/neuroglancer). Designed for terabyte-scale datasets using hundreds of parallel Cloud Run workers.

[BRAID](braid/) (Block Records Arrow Indexed Dataset) reads and decompresses the shard files, with a C extension for high-throughput DVID block decompression.

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

Run `pixi run deploy` to be guided through all required configuration. It prompts for GCS paths, reads the neuroglancer volume spec JSON, writes the destination `info` file (with `compressed_segmentation` encoding), builds the Docker image, and creates the Cloud Run job.

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
Creating per-scale Cloud Run jobs...
Done.
```

All values are saved to `.env` for future runs. You can also edit `.env` directly (see `.env.example` for all options).

### Export

After deploying, run the export. This single command scans all Arrow source
files, assigns shards to memory tiers, writes per-task manifests to GCS,
and launches a Cloud Run job per tier:

```bash
pixi run export
```

```
Scanning Arrow files across 10 scales...
  Found 26128 Arrow files
  Memory formula: 2.5 * arrow_size + 1 GiB

Tier assignments:
  2Gi (cpu=1): 10181 shards, 2000 tasks, total=372.1GB, max_arrow=99MB
  4Gi (cpu=2): 15915 shards, 2600 tasks, total=2804.5GB, max_arrow=1200MB
  8Gi (cpu=2):    22 shards,   22 tasks, total=43.8GB, max_arrow=1900MB
  16Gi (cpu=4):    8 shards,    8 tasks, total=30.2GB, max_arrow=4050MB
  32Gi (cpu=8):    2 shards,    2 tasks, total=13.3GB, max_arrow=6647MB

Writing per-task manifests...
  2Gi: 2000 task manifests → gs://mybucket/exports/seg/manifests/tier-2gi/
  4Gi: 2600 task manifests → gs://mybucket/exports/seg/manifests/tier-4gi/
  ...

Launching 5 tier job(s)...
  export-tier-2gi: launched (2000 tasks, 2Gi, cpu=1)
  export-tier-4gi: launched (2600 tasks, 4Gi, cpu=2)
  export-tier-8gi: launched (22 tasks, 8Gi, cpu=2)
  ...
```

Each task reads only its own small manifest file
(`task-0.json`, `task-1.json`, ...) listing the shards it should process.
Tasks within a tier are load-balanced by total Arrow file bytes.

Options for `export`:

| Option | Description | Default |
|--------|-------------|---------|
| `--scales` | Scales to include | from `.env` |
| `--label-type` | `labels` (agglomerated, default) or `supervoxels` (raw IDs) | `labels` |
| `--downres` | Scales to generate by downsampling previous scale | none |
| `--tiers` | Override max tasks per tier (e.g., `4:3000,8:50`) | auto |
| `--dry-run` | Scan and show tiers without writing manifests or launching | |
| `--wait` | Block until all jobs complete | async |

Use `pixi run precompute-manifest --dry-run` to preview tier assignments
without launching any jobs.

By default, workers export **agglomerated labels** — the standard segmentation view where proofreading merges are applied. Use `--label-type supervoxels` to export the raw supervoxel IDs from the DVID blocks instead.

### Monitoring Progress

```bash
# Status of all per-scale jobs (elapsed time, task counts)
pixi run export-status

# Status of a specific scale
pixi run export-status --scale 0
```

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

### Mining Memory Profiles

After an export, mine the structured logs to understand actual memory
usage per shard and tune tier boundaries for future runs:

```bash
# Memory profile summary per shard (peak RSS, batches, elapsed time)
pixi run export-errors -- --details | grep "Shard memory profile"

# Wall-clock progress snapshots (memory trajectory over time)
pixi run export-errors -- --details | grep "Shard progress"

# Memory pressure events (transaction forced to commit early)
pixi run export-errors -- --details | grep "memory pressure"
```

Key fields in `"Shard memory profile"` events:

| Field | Description |
|-------|-------------|
| `peak_memory_gib` | Maximum cgroup memory usage during shard processing |
| `memory_limit_gib` | Container memory limit (from Cloud Run `--memory`) |
| `batches` | Number of transaction commits (1 = ideal single-write) |
| `uncompressed_gib` | Total uncompressed chunk data written |
| `elapsed_s` | Wall-clock time to process the shard |

If `batches > 1` for many shards, increase the memory tier. If
`peak_memory_gib` is well below `memory_limit_gib` for all shards in a
tier, you can safely reduce the tier to save cost.

### Multi-Scale Processing

Workers support two sources for each scale:

**From DVID export shards** (default): DVID pre-computes downsampled blocks at each scale. Workers ingest them directly — no computation needed, output exactly matches what DVID serves.

**From previous scale** (downres mode): For scales not materialized in DVID, `--downres N` generates scale N by reading the already-written scale N-1 from the destination volume and downsampling 2× in each dimension using majority vote. Scale N-1 must be fully written first.

### Memory Sizing

BRAID's ShardReader fully downloads the Arrow IPC file into memory (not
lazy/streamed — pyarrow's native GCS filesystem is unreliable on Cloud Run).
The Arrow file size is the dominant memory variable.

Memory needed per shard ≈ `2.5 × arrow_file_size + 1 GiB`:

| Component | Size | Notes |
|-----------|------|-------|
| Arrow file (BRAID ShardReader) | 1× arrow_size | Fully in RAM via `blob.download_as_bytes()` |
| Transaction buffer (TensorStore Cords) | ~1-1.5× arrow_size | compressed_segmentation encoded chunks |
| Python, TensorStore, OS overhead | ~1 GiB | Runtime, encoding buffers, etc. |

The `precompute-manifest` command uses this formula to assign shards to
memory tiers automatically.  Workers monitor actual container memory via
cgroup (`/sys/fs/cgroup/memory.current`) and will commit the transaction
early if memory pressure is detected, so the estimate doesn't need to be
exact — the cgroup safety valve prevents OOM.

Arrow shard file sizes from the mCNS dataset:

| Scale | Count  | Mean   | Max     |
|-------|--------|--------|---------|
| 0     | 21,994 | 142 MB | 493 MB  |
| 1     | 3,364  | 337 MB | 898 MB  |
| 2     | 606    | 528 MB | 1.9 GB  |
| 3     | 123    | 838 MB | 4.1 GB  |
| 4     | 25     | 1.5 GB | 6.6 GB  |
| 5     | 8      | 1.6 GB | 6.6 GB  |
| 6+    | 9      | ≤1.6 GB| 3.1 GB  |

99.9% of shards (26,096 of 26,128) fit in 4 GiB workers.

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
  per-scale: s0/, s1/...       google-cloud-storage           compressed_segmentation
```

Each worker processes one shard at a time: download Arrow+CSV from GCS, decompress chunks via the DVID block decompressor, transpose ZYX→XYZ, and write to the neuroglancer precomputed volume via TensorStore.

**Transaction batching**: All chunk writes for a shard are accumulated in a
single explicit TensorStore transaction and committed as one shard write to
GCS.  Without this, each chunk triggers a full shard read-modify-write
(O(N²) I/O).  Workers monitor container memory via cgroup and commit early
only if approaching the memory limit.

**Tier-based task assignment**: `pixi run export` scans Arrow file sizes,
estimates memory requirements (`2.5 × arrow_size + 1 GiB`), and assigns
shards to memory tiers (1–32 GiB).  Each tier gets its own Cloud Run job
with appropriate memory and CPU.  Each task downloads only its own small
manifest file listing the shards it should process.  Tasks within a tier
are load-balanced by Arrow file size.

See [braid/docs/ARCHITECTURE.md](braid/docs/ARCHITECTURE.md) for I/O design decisions.

## Project Structure

```
tensorstore-export/
├── pixi.toml                        # Dependencies and tasks
├── .env.example                     # Configuration template
├── Dockerfile                       # Cloud Run container
├── scripts/
│   ├── deploy.py                   # Interactive deployment (image, info file, base job)
│   ├── export.py                   # Scan, manifest, launch (single command)
│   ├── precompute_manifest.py      # Tier assignment and manifest generation
│   ├── export_errors.py            # Query error logs
│   └── export_status.py            # Job status overview
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
├── examples/
│   └── run-all-scales.sh           # Full export example (tier-based)
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
