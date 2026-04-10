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
pixi run export               # Full pipeline: Arrow export + downres (skips existing)
pixi run export --dry-run     # Show what would be launched
pixi run export --overwrite   # Re-export all shards even if output exists
pixi run export --z-compress 1 --scales 0  # Z decimation for anisotropic exports
pixi run export-status        # Monitor running Cloud Run jobs
pixi run export-errors        # Scan logs for export errors
pixi run verify-export        # Verify all DVID shards have NG output
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
pixi run export    →  unified pipeline:
                      1. Scans Arrow source shards for SCALES
                      2. Checks DEST_PATH — skips already-exported shards
                      3. Groups remaining shards into memory tiers
                      4. Writes per-task manifests to GCS, launches Cloud Run jobs
                      5. If DOWNRES_SCALES is set: waits for Arrow export,
                         then runs downres for each scale sequentially
```

Tier-based manifests group shards by estimated memory so that small-shard tasks use less memory (and cost less) than large-shard tasks.

By default, only missing shards are exported (both Arrow and downres phases check the destination for existing output). Use `--overwrite` to force re-export. Use `--downres <scales>` to run downres-only without re-scanning Arrow source files.

### BRAID Library (`braid/`)

Pure Python (with optional C extension) for reading DVID Arrow shard files:
- Two reader variants: `ShardReader` (full-file load) and `ShardRangeReader` (GCS byte-range reads with batch caching)
- Block-wise access: reads individual 64x64x64 chunks by coordinate
- Dual label support: agglomerated labels and supervoxel data
- DVID block decompression: zstd + custom label encoding
- CSV index: maps (x,y,z) to Arrow byte offsets and batch positions

## Configuration

All configuration lives in `.env` (not committed; see `.env.example` for the template). **Always check `.env` first** to find the active dataset paths (SOURCE_PATH, DEST_PATH, etc.) rather than guessing or searching for them. Key variables:

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

## Anisotropic Datasets (Z Compression)

When the DVID source was exported with Z-doubled data (e.g., original [16,16,30] nm segmentation doubled to [16,16,15] nm), the `--z-compress N` flag decimates Z slices to produce native-resolution output. `--z-compress 1` keeps every 2nd Z slice (stride=2), halving the Z dimension.

This requires two export passes since only s0 needs Z decimation — from s1 onward, the DVID 2x downsample in Z collapses the doubled slices back to the original resolution:

```bash
# 1. Export s0 with Z decimation (NG_SPEC_PATH points to native-Z spec)
pixi run export --scales 0 --z-compress 1

# 2. Export s1+ without Z compression (source data already matches)
pixi run export --scales 1,2,3,4,5,6,7,8

# Verification also needs the flag for s0
pixi run verify-export --scales 0 --z-compress 1
pixi run verify-export --scales 1,2,3,4,5,6,7,8
```

The `--z-compress` flag also adjusts the skip-existing shard mapping so that re-runs correctly detect which output shards are already present. See `examples/fish2_seg_zdoubled_spec.json` (source) and `examples/fish2_seg_native_z_spec.json` (output) for reference.

## TensorStore Pitfall: Always `.result()` Writes in Transactions

`ts.TensorStore.write()` returns a `Future`, not a completed operation. Inside a
transaction, you **must** call `.result()` on every write to ensure it is staged
before `txn.commit_async().result()`. Without it, the commit silently drops any
writes whose futures haven't resolved yet — producing partial/empty output with
no errors logged.
