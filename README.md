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

Run `pixi run deploy` to be guided through all required configuration. It prompts for GCS paths, reads the neuroglancer volume spec JSON, validates bucket settings (creating the destination bucket if needed — see [GCS Bucket Setup](#gcs-bucket-setup)), writes the destination `info` file, and builds the Docker image.

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
  SCALES [0,1,2,3,4,5,6,7,8,9]: ↵
  ...

Writing neuroglancer info file ...
Building Docker image ...
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
  Found 26125 Arrow files
  Memory formula: (arrow + 1.3 * shard_on_tmpfs + 1.5) * 1.3

Tier assignments:
  4Gi (cpu=2): 5169 shards, 5000 tasks, total=41.5GB, max_arrow=629MB
  8Gi (cpu=2): 4027 shards, 4027 tasks, total=234.6GB, max_arrow=1997MB
  16Gi (cpu=4): 16717 shards, 5000 tasks, total=4121.0GB, max_arrow=3402MB
  24Gi (cpu=6): 196 shards, 100 tasks, total=275.4GB, max_arrow=6634MB
  32Gi (cpu=8): 16 shards, 16 tasks, total=58.2GB, max_arrow=6647MB

Writing per-task manifests...
  4Gi: 5000 task manifests → gs://mybucket/exports/seg/manifests/tier-4gi/
  8Gi: 4027 task manifests → gs://mybucket/exports/seg/manifests/tier-8gi/
  ...

Launching 5 tier job(s)...
  export-tier-4gi: launched (5000 tasks, 4Gi, cpu=2)
  export-tier-8gi: launched (4027 tasks, 8Gi, cpu=2)
  export-tier-16gi: launched (5000 tasks, 16Gi, cpu=4)
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
pixi run export-status    # task counts, memory, timing, in-flight shards
pixi run export-errors    # error summary across all tiers
pixi run export-errors --details  # full details of every failed chunk
```

`export-status` queries Cloud Run execution status and Cloud Logging for
structured events. It shows per-tier shard/chunk completion, memory usage,
timing stats, and in-flight shard progress. If `summary.json` exists (written
by `precompute-manifest`), it also shows a grand progress bar with total
chunks written vs expected.

### Retry workflow

If tasks fail (OOM, transient errors), identify incomplete shards and retry
at a higher memory tier:

```bash
pixi run find-failed -- --retry-tier 16
pixi run export --manifest-dir manifests-retry --job-suffix retry
```

### Multi-Scale Processing

Workers support two sources for each scale:

**From DVID export shards** (default): DVID pre-computes downsampled blocks at each scale. Workers ingest them directly — no computation needed, output exactly matches what DVID serves.

**From previous scale** (downres mode): For scales not materialized in DVID, `--downres N` generates scale N by reading the already-written scale N-1 from the destination volume and downsampling 2× in each dimension using majority vote. Scale N-1 must be fully written first.

### Memory Sizing

Memory per shard is dominated by two components: the Arrow file in RAM and
the output neuroglancer shard on tmpfs (Cloud Run Gen 2 uses in-memory
filesystem). The formula used by `precompute-manifest`:

```
memory_gib = (arrow_gib + 1.3 * shard_on_tmpfs_gib + 1.5) * 1.3
```

| Component | Size | Notes |
|-----------|------|-------|
| Arrow file | 1x arrow_size | Fully in RAM via `blob.download_as_bytes()` |
| Output shard on tmpfs | chunks x KB/chunk | Grows during batched RMW commits |
| TensorStore RMW overhead | ~1.3x shard size | During commit |
| Python baseline | ~1.5 GiB | Runtime, libraries, buffers |
| Safety margin | 1.3x total | |

The `precompute-manifest` command assigns shards to tiers (4/8/16/24/32 GiB)
automatically. In the mCNS v0.11 production run, all 26,125 shards completed
with zero OOM failures. No tier exceeded 50% of its memory budget.

## Configuration

All settings live in `.env` (not committed). See `.env.example` for the full list.

| Variable | Description | Example |
|----------|-------------|---------|
| `SOURCE_PATH` | GCS URI to DVID shard export (contains s0/, s1/, ...) | `gs://mybucket/exports/seg` |
| `DEST_PATH` | GCS URI for neuroglancer precomputed output | `gs://mybucket/v1.0/precomputed` |
| `NG_SPEC_PATH` | Neuroglancer volume spec JSON (same as DVID export-shards) | `mcns-v0.11-export-specs.json` |
| `BASE_JOB_NAME` | Cloud Run job name prefix (tier jobs: `{name}-tier-4gi`, ...) | `tensorstore-dvid-export` |
| `SCALES` | Scales to process (comma-separated) | `0,1,2,3,4,5,6,7,8,9` |

## GCS Bucket Setup

Use **single-region** GCS buckets in the same region as your Cloud Run tasks. This avoids cross-region replication charges that can be catastrophic at scale (see the [$63K incident post-mortem](docs/mCNS-ExportAnalysis.md#8-post-mortem-excessive-gcs-replication-charges-march-2023)).

```bash
# Source bucket (for DVID export-shards output)
gcloud storage buckets create gs://myproject-export-shards \
  --location=us-east4 \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access

# Destination bucket (for neuroglancer precomputed output)
# deploy auto-creates this if it doesn't exist
gcloud storage buckets create gs://myproject-precomputed \
  --location=us-east4 \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access \
  --enable-hierarchical-namespace \
  --no-soft-delete
```

**Key settings:**

| Setting | Value | Why |
|---------|-------|-----|
| Location type | Single-region | Cloud Run distributes tasks across zones randomly. Single-region gives consistent performance with no inter-zone costs. Multi-region incurs $0.02/GiB replication on every write. |
| Soft delete | Disabled | TensorStore's batched write pattern creates intermediate shard versions. Soft delete retains all of them, potentially generating petabytes of retained data. `pixi run deploy` auto-disables this if detected. |
| Uniform access | Enabled | Simplifies IAM; no per-object ACLs needed. |
| Hierarchical namespace | Enabled (dest) | Improves throughput for the thousands of parallel shard file uploads to the destination bucket. Not needed for the source bucket since DVID writes shards sequentially. Must be set at bucket creation time. |

**Why not zonal?** Zonal buckets offer higher peak throughput for same-zone access, but Cloud Run tasks can land in any zone within the region. A zonal bucket in `us-east4-c` would incur inter-zone transfer costs (~$0.01/GiB) when tasks land in other zones. For this pipeline's CPU-bound workload (decompression + label mapping), the GCS I/O difference is negligible.

**Recommended bucket layout:**

| Bucket | Purpose | Settings |
|--------|---------|----------|
| Source (e.g., `myproject-export-shards`) | DVID Arrow+CSV shard files | Single-region, same as `REGION` |
| Destination (e.g., `myproject-precomputed`) | Neuroglancer precomputed output | Single-region, same as `REGION`, HNS on, soft delete off |
| Distribution (optional) | Public/team access to finished volumes | Multi-region, soft delete on |

Use **separate buckets for source and destination** to isolate read and write workloads — this avoids throughput contention on a single bucket during large exports with thousands of parallel workers. Both must be in the same region as Cloud Run to avoid egress charges. Copy finished volumes to a distribution bucket afterward via `gcloud storage cp -r`.

`pixi run deploy` validates the destination bucket at deploy time: creates it if missing, warns on region mismatches, and auto-disables soft delete.

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

**Tier-based task assignment**: `pixi run precompute-manifest` scans Arrow
file sizes and chunk counts, estimates memory requirements, and assigns
shards to memory tiers (4–32 GiB).  Each tier gets its own Cloud Run job
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
│   ├── deploy.py                   # Interactive deployment (image, info file)
│   ├── export.py                   # Scan, manifest, launch (single command)
│   ├── precompute_manifest.py      # Tier assignment, manifest + summary generation
│   ├── export_status.py            # Job status, memory, progress tracking
│   ├── export_errors.py            # Query error logs
│   ├── find_failed_shards.py       # Identify failed shards for retry
│   ├── compute_offsets.py          # Pre-compute Arrow byte offsets for range reads
│   └── bucket_utils.py            # GCS bucket validation and configuration
├── src/
│   ├── worker.py                   # Cloud Run worker
│   └── tensorstore_adapter.py      # TensorStore helpers
├── braid/                           # BRAID library
│   ├── src/braid/
│   │   ├── reader.py               # ShardReader + ShardRangeReader
│   │   ├── decompressor.py         # DVID block decompressor
│   │   ├── cseg_encoder.py         # DVID-to-cseg transcoder (Python wrapper)
│   │   └── exceptions.py
│   ├── csrc/
│   │   ├── dvid_decompress.c       # C DVID block decompressor
│   │   └── cseg_encode.c           # C compressed_segmentation encoder
│   ├── tests/                      # Tests including ground truth verification
│   └── docs/ARCHITECTURE.md        # I/O design decisions
├── examples/
│   └── mcns-v0.11-export-specs.json # mCNS neuroglancer volume spec
└── docs/
    ├── mCNS-ExportAnalysis.md      # Export data analysis + production results
    ├── PrecomputeShardOffsets.md    # Two-phase memory separation design
    ├── CustomShardWriter.md        # Future: bypass TensorStore with BRAID transcoder
    └── ShardExportDesign.md        # System design document
```

## Related Projects

- [DVID](https://github.com/janelia-flyem/dvid) — Source data server with `export-shards` command
- [BRAID](https://github.com/JaneliaSciComp/braid) — Standalone Arrow shard reader (will be published separately)
- [TensorStore](https://github.com/google/tensorstore) — Multi-dimensional array storage
- [Neuroglancer](https://github.com/google/neuroglancer) — Web-based volumetric data viewer

## License

Janelia provides this under a modified BSD 3-Clause License.
