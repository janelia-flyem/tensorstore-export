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
  SCALES [0]: ↵
  DOWNRES_SCALES [1,2,3,4,5,6,7,8,9]: ↵
  ...

Writing neuroglancer info file ...
Building Docker image ...
Done.
```

All values are saved to `.env` for future runs. You can also edit `.env` directly (see `.env.example` for all options).

### Export

There are two supported export approaches:

1. **Direct from DVID at every scale**
   Export the scales listed in `SCALES` directly from DVID Arrow shards.
2. **s0 from DVID, then s1+ by TensorStore downres**
   Export `s0` from DVID, then generate later scales from the previous
   neuroglancer scale using TensorStore downsampling.

#### Approach 1: Direct from DVID

Set `SCALES` in `.env` to the DVID-materialized scales you want to export and
leave `DOWNRES_SCALES` blank, then run:

```bash
pixi run export
```

```
Scanning Arrow files across 1 scales...
  Found 21690 Arrow files

Tier assignments:
  4Gi (cpu=2): 19204 shards, 5000 tasks
  8Gi (cpu=2): 2321 shards, 2321 tasks
  16Gi (cpu=4): 162 shards, 162 tasks
  24Gi (cpu=6): 3 shards, 3 tasks

Launching 4 tier job(s)...
  tensorstore-dvid-export-tier-4gi: launched (5000 tasks, 4Gi, cpu=2)
  ...
```

Each task reads its own small manifest file (`task-0.json`, `task-1.json`, ...)
listing the shards it should process. Tasks within a tier are load-balanced by
Arrow file size.

#### Approach 2: s0 from DVID, then downres later scales

Set `.env` so that:

```bash
SCALES=0
DOWNRES_SCALES=1,2,3,4,5,6,7,8,9
```

Then run:

```bash
pixi run export
```

This now runs the downres pipeline sequentially by default:

- for each target scale, aggregate predicted labels first
- generate manifests for that scale
- launch the downres jobs for that scale
- verify the completed scale before moving on
- stop immediately if any scale fails verification or launch

At the end, the command prints a per-scale timing summary and total elapsed
time.

For one-off or partial runs, you can still target specific scales directly:

```bash
# Generate s1 from s0
pixi run export --downres 1

# Generate s3 from existing s2 output
pixi run export --downres 3

# Retry only missing output shards for s2
pixi run export --downres 2 --only-missing
```

For standalone later-scale runs, `aggregate-labels` is still available as a
manual step if you want the label summary written ahead of time:

```bash
pixi run aggregate-labels --target-scale 3
```

`aggregate-labels` produces per-shard label counts that the manifest
generator uses for tighter memory tier assignment (label-aware model).
Without it, the chunk-count model is used, which is more conservative.

#### Export options

| Option | Description | Default |
|--------|-------------|---------|
| `--scales` | Source scales with DVID Arrow shards | from `.env` |
| `--downres-mode` | Deprecated compatibility flag; `--downres` already implies downres mode | |
| `--downres` | Target scales for downres (e.g., `1` or `1,2,3,...,9`) | from `.env` |
| `--only-missing` | For downres, generate manifests only for missing output shards | off |
| `--label-type` | `labels` (agglomerated) or `supervoxels` (raw IDs) | `labels` |
| `--tiers` | Override max tasks per tier (e.g., `4:3000,8:50`) | auto |
| `--dry-run` | Show tier assignments without writing manifests or launching | |
| `--wait` | For non-downres export, block until jobs complete | async |
| `--async` | For downres export, disable sequential orchestration and launch jobs immediately | off |

Use `pixi run precompute-manifest --dry-run` to preview tier assignments
without launching any jobs.

### Monitoring Progress

```bash
pixi run export-status    # task counts, memory, timing, in-flight shards
pixi run export-errors    # error summary across all tiers
pixi run export-errors --details  # full details of every failed chunk
```

Both commands work for s0 export jobs and downres jobs. For downres jobs,
they query the appropriate log event names (`Downres shard complete` vs
`Shard complete`) automatically.

`export-status` queries Cloud Run execution status and Cloud Logging for
structured events. It shows per-tier shard/chunk completion, memory usage,
timing stats, and in-flight shard progress. If `summary.json` exists (written
by `precompute-manifest`), it also shows a grand progress bar with total
chunks written vs expected.

**Note:** Cloud Run can take 15-30 minutes to reconcile task statuses after
containers exit. Tasks may show as "running" in `export-status` even though
the containers have already exited with code 0. This is a platform-side delay
— no billing occurs after container exit. Use `pixi run verify-export` to
confirm output completeness independently of task status.

### Post-Export Verification

Verify that every DVID source shard produced an NG output shard on GCS:

```bash
pixi run verify-export                    # all scales
pixi run verify-export -- --scales 0,1,2  # specific scales
pixi run verify-export -- --json-report report.json
```

This compares DVID Arrow files against actual `.shard` files on GCS using the
compressed Z-index mapping — no dependency on Cloud Logging.

To validate the DVID export itself before running the pipeline (checks that
all chunks in each DVID shard map to the same NG shard number):

```bash
pixi run validate-dvid
pixi run validate-dvid -- --scales 0 --sample 100
```

### Retry workflow

If tasks fail (OOM, transient errors), identify incomplete shards and retry
at a higher memory tier:

```bash
pixi run find-failed -- --retry-tier 16
pixi run export --manifest-dir manifests-retry --job-suffix retry
```

For downres retries, the normal path is:

```bash
pixi run export --downres 2 --only-missing
```

That regenerates manifests only for missing destination shards at the target
scale and launches the manifest-driven downres worker path.

### Memory Sizing

Cloud Run Gen 2 uses in-memory filesystem (tmpfs), so the output shard file
consumes container memory. `precompute-manifest` assigns shards to memory
tiers (4/8/16/24/32 GiB) automatically based on estimated peak memory.

**s0 export** (from DVID Arrow shards):

```
memory = arrow_in_ram + 2 * shard_on_tmpfs + 2.0 GiB overhead
```

| Component | Notes |
|-----------|-------|
| Arrow file in RAM | Fully loaded via `blob.download_as_bytes()` |
| Output shard on tmpfs (2x) | RMW during batched transaction commits |
| Fixed overhead (2.0 GiB) | Python + pyarrow + BRAID + TensorStore + GCS client |

**Downres** (from previous scale):

```
memory = (raw_batch + 2 * shard_on_tmpfs + fixed_overhead + commit_spike) * safety_factor
```

| Component | Notes |
|-----------|-------|
| Raw batch arrays | One Z-plane of raw uint64 arrays: N^(2/3) chunks × 2 MiB (e.g., 1024 × 2 MiB = 2 GiB for 32³ shards) |
| Output shard on tmpfs (2×) | RMW during batched transaction commits. Coexists with raw arrays in cgroup memory. |
| Fixed overhead | Python + TensorStore + source/staging caches + label readback |
| Commit spike | Hidden TensorStore encode/commit spike, modeled as a scale-based floor |
| Safety factor | Multiplies the subtotal to cover additional source-side working set |

Writes are batched one Z-plane at a time with explicit transaction commits.
Between `write()` and `commit()`, TensorStore holds raw uint64 arrays in
its ChunkCache — cache_pool eviction does NOT apply to explicit
transactions. All three terms are additive in cgroup memory (raw arrays +
tmpfs shard file + overhead coexist simultaneously).

When per-shard label counts are available (from `aggregate-labels`), the
**label-aware model** replaces the chunk-count tmpfs estimate with a tighter
linear prediction based on total unique labels per shard.

## Configuration

All settings live in `.env` (not committed). See `.env.example` for the full list.

| Variable | Description | Example |
|----------|-------------|---------|
| `SOURCE_PATH` | GCS URI to DVID shard export (contains s0/, s1/, ...) | `gs://mybucket/exports/seg` |
| `DEST_PATH` | GCS URI for neuroglancer precomputed output | `gs://mybucket/v1.0/precomputed` |
| `NG_SPEC_PATH` | Neuroglancer volume spec JSON (same as DVID export-shards) | `mcns-export-specs.json` |
| `BASE_JOB_NAME` | Cloud Run job name prefix (tier jobs: `{name}-tier-4gi`, ...) | `tensorstore-dvid-export` |
| `SCALES` | Source scales with DVID Arrow shards | `0` |
| `DOWNRES_SCALES` | Scales to generate by downsampling (optional) | `1,2,3,4,5,6,7,8,9` |

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

**Direct-export workers** process one shard at a time: download Arrow+CSV from GCS, decompress chunks via the DVID block decompressor, transpose ZYX→XYZ, and write to the neuroglancer precomputed volume via TensorStore.

**Downres workers** generate scales s1-s9 by reading the previous scale from the destination volume on GCS, downsampling 2x in each dimension (majority vote), writing the output shard to local tmpfs staging, then uploading to GCS. Each worker also reads back the output chunks to compute unique label counts for next-scale memory prediction.

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
│   ├── export.py                   # Scan, manifest, launch (s0 and downres)
│   ├── precompute_manifest.py      # Tier assignment, manifest + summary generation
│   ├── aggregate_predicted_labels.py # Merge label predictions for tier assignment
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
│   └── mcns-export-specs.json       # mCNS neuroglancer volume spec
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
