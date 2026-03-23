# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Converts DVID `export-shards` Arrow IPC files into neuroglancer precomputed segmentation volumes on GCS using massively parallel Cloud Run jobs.

Two main components:
1. **src/**: The Cloud Run worker application (TensorStore + BRAID)
2. **braid/**: Python library for reading DVID's spatially-partitioned Arrow shard files (installed as an editable dependency)

Supporting directories: `scripts/` (deployment and orchestration), `examples/` (export spec files), `docs/` (design documents).

## Development Commands

This project uses [pixi](https://prefix.dev/) for dependency management and task running.

```bash
pixi run test-braid          # Unit tests for braid library
pixi run test-bench           # Benchmark tests for braid decompressor
pixi run test-e2e             # End-to-end precomputed output test
pixi run test-all             # All braid tests
pixi run lint                 # Ruff linter on src/, braid/, scripts/
pixi run build-braid-c        # Build braid C extension (decompressor)

pixi run deploy               # Build Docker image and push to GCR
pixi run export               # Full export: manifest + Cloud Run jobs
pixi run export --dry-run     # Show what would be launched
pixi run export-status        # Monitor running Cloud Run jobs
pixi run export-errors        # Scan logs for export errors
pixi run precompute-manifest  # Generate per-task shard manifests
```

## Architecture

### Worker (`src/worker.py`)

Each Cloud Run task:
1. Reads its shard assignments from a pre-computed manifest on GCS (or self-partitions via task index)
2. For each assigned shard, reads the Arrow IPC file and CSV index from GCS
3. Decompresses DVID blocks using the BRAID library
4. Writes chunks into a TensorStore transaction (neuroglancer precomputed with compressed_segmentation encoding)
5. Commits the transaction, which writes the output shard file to local staging (tmpfs — consumes memory), then uploads to GCS
6. Optionally generates downres scales by reading the previous scale from the destination volume

### Key Source Files
- `src/worker.py`: Cloud Run worker — shard processing, transaction management, cgroup memory monitoring
- `src/tensorstore_adapter.py`: Helper for opening neuroglancer precomputed volumes via TensorStore
- `braid/src/braid/reader.py`: Arrow shard reader with CSV index lookup
- `braid/src/braid/decompressor.py`: DVID block decompression (zstd + label encoding)
- `scripts/export.py`: Orchestrator — scans shards, builds tier-based manifests, launches Cloud Run jobs
- `scripts/deploy.py`: Docker build and GCR push
- `scripts/precompute_manifest.py`: Generates per-task manifest files partitioned into memory tiers
- `scripts/setup_destination.py`: Creates the neuroglancer precomputed info file on GCS

### Export Pipeline

```
pixi run deploy    →  builds Docker image, pushes to GCR, stores URI in .env
pixi run export    →  scans source shards → groups into memory tiers →
                      writes per-task manifests to GCS → launches one
                      Cloud Run job per tier
```

Tier-based manifests group shards by estimated memory so that small-shard tasks use less memory (and cost less) than large-shard tasks.

### BRAID Library (`braid/`)

Pure Python (with optional C extension) for reading DVID Arrow shard files:
- Block-wise access: reads individual 64x64x64 chunks by coordinate
- Dual label support: agglomerated labels and supervoxel data
- DVID block decompression: zstd + custom label encoding
- CSV index: maps (x,y,z) to Arrow record offsets

## Configuration

All configuration lives in `.env` (not committed; see `.env.example` for the template). Key variables:

| Variable | Description |
|----------|-------------|
| `SOURCE_PATH` | GCS URI to DVID Arrow shard export (contains `s0/`, `s1/`, ...) |
| `DEST_PATH` | GCS URI for neuroglancer precomputed output |
| `NG_SPEC_PATH` | Local path to neuroglancer multiscale JSON spec |
| `SCALES` | Comma-separated scale indices to process (e.g., `0,1,2,3`) |
| `DOWNRES_SCALES` | Scales to generate by downsampling (optional) |
| `PROJECT_ID` | GCP project |
| `REGION` | GCP region |
| `BASE_JOB_NAME` | Prefix for Cloud Run job names |

The worker itself reads `SOURCE_PATH`, `DEST_PATH`, `NG_SPEC` (base64-encoded), `SCALES`, `MANIFEST_URI`, `LABEL_TYPE`, `WORKER_MEMORY_GIB` from its environment (set by `scripts/export.py`).

## TensorStore Pitfall: Always `.result()` Writes in Transactions

`ts.TensorStore.write()` returns a `Future`, not a completed operation. Inside a
transaction, you **must** call `.result()` on every write to ensure it is staged
before `txn.commit_async().result()`. Without it, the commit silently drops any
writes whose futures haven't resolved yet — producing partial/empty output with
no errors logged.
