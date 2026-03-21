# Example Neuroglancer Volume Data & Script

These are dataset-specific run script and neuroglancer multiscale volume spec JSON files used with DVID's `export-shards` command. They serve as reference examples — copy and adapt for your own dataset.

## Files

| File | Dataset | Status |
|------|---------|--------|
| `mcns-v0.11-export-specs.json` | Male CNS (mCNS) segmentation | Correct sharding params per scale |
| `mcns-v0.9-ng-specs-INCORRECT.json` | Male CNS segmentation (v0.9) | **Incorrect** — shard_bits/minishard_bits/preshift_bits are identical across all scales instead of decreasing. Do not use for export. |
| `mcns-v0.9-grayscale-ng-specs.json` | Male CNS grayscale (v0.9) | Grayscale (uint8) volume spec |
| `run-all-scales.sh` | Male CNS | Correct sizing of Google Cloud Run tasks to process in hours |

## Usage

Point `NG_SPEC_PATH` in your `.env` to one of these files (or your own):

```bash
NG_SPEC_PATH=examples/mcns-v0.11-export-specs.json
```

Or provide the path during `pixi run deploy` when prompted.

## Sharding Parameter Rules

For a correctly configured spec, the total bits (`shard_bits + minishard_bits + preshift_bits`) must equal the total chunk coordinate bits at each scale. See `docs/mCNS-ExportAnalysis.md` for the derivation.
