# Pre-compute Arrow Shard Byte Offsets

## Problem

The export pipeline's memory budget is dominated by the Arrow file. Each Cloud
Run worker downloads the entire DVID Arrow shard file into memory (up to ~6.6 GB
for the largest s0 shards), then holds it while simultaneously building the
output neuroglancer shard on tmpfs. This forces complex tier-based memory
partitioning (4/8/16/24/32 GiB tiers) and still produces ~14% OOM failures in
the 8 GiB tier.

## Solution: Two-Phase Approach

Separate the memory-intensive work into two independent phases:

### Phase 1: Compute byte offsets (one-time preprocessing)

Scan each Arrow IPC streaming file to find the byte offset and size of every
record batch message. Write a companion `-offsets.csv` file alongside the Arrow
file on GCS.

Memory: one Arrow file at a time + trivial overhead. This is a sequential scan
with no output accumulation, so it can run on any machine with enough RAM for
the largest Arrow file. Since it's one-time, it can use a single high-memory
Cloud Run job or even a local workstation.

### Phase 2: Export neuroglancer precomputed (the actual conversion)

Use `ShardRangeReader` (already implemented in `braid/src/braid/reader.py`)
instead of `ShardReader`. The range reader never loads the full Arrow file.
For each chunk, it does a GCS byte-range read to fetch only the schema (~200
bytes, cached) plus the specific record batch (~5-50 KB).

Memory per chunk: ~50 KB (record batch) + ~2 MB (transcoder working buffer).
Memory per shard: accumulated output data (~50-150 MB) + index metadata (~768 KB).
Total: ~200 MB regardless of Arrow file size.

**The Arrow file size drops out of the memory formula entirely.**

This means every export task fits in the 4 GiB tier (or lower with the custom
shard writer), eliminating tier-based partitioning and the OOM failures that
come with memory underestimation.

## Offset CSV Format

Produced by `scripts/compute_offsets.py`, stored alongside the Arrow file:

```
# schema_size=184
x,y,z,rec,offset,size
480,384,448,0,184,3712
480,384,512,1,3896,3648
480,448,448,2,7544,4120
...
```

- **Comment header**: `# schema_size=N` — byte size of the Arrow schema message.
  The reader fetches bytes `[0, N)` to get the schema, then caches it.
- **Columns**: `x,y,z` (chunk coordinates), `rec` (record index), `offset`
  (byte offset of the record batch in the Arrow file), `size` (byte size of
  the record batch message including header padding).
- **File size**: ~15 bytes per row. A 258-chunk shard produces a ~4 KB offset
  CSV. A 32,768-chunk shard produces ~500 KB.

## Current Implementation

### `scripts/compute_offsets.py`

Registered as `pixi run compute-offsets`. Scans all Arrow files across
configured scales (`SCALES` in `.env`) and uploads offset CSVs to GCS.

```bash
pixi run compute-offsets                  # all scales from .env
pixi run compute-offsets --scales 0,1     # specific scales
pixi run compute-offsets --dry-run        # show what would be processed
```

Features:
- **Idempotent**: skips shards where `-offsets.csv` already exists
- **Parallel**: 8 concurrent download/scan/upload threads (`--workers N`)
- **Cloud Run ready**: supports `CLOUD_RUN_TASK_INDEX` / `CLOUD_RUN_TASK_COUNT`
  for distributing across a Cloud Run job

### `braid.ShardRangeReader`

Drop-in replacement for `ShardReader` with the same public interface. Requires
the pre-computed offset CSV.

```python
from braid import ShardRangeReader, LabelType

reader = ShardRangeReader(arrow_uri, csv_uri, offsets_uri)
for cx, cy, cz in reader.available_chunks:
    data = reader.read_chunk(cx, cy, cz, label_type=LabelType.LABELS)
```

Verified to produce identical output to `ShardReader` for all 258 chunks in the
real test shard (9 parity tests in `braid/tests/test_range_reader.py`).

## Scaling the Offset Computation

Running `pixi run compute-offsets` locally works for a modest number of shards
but is slow for the full dataset (~14,000 shards across 10 scales, some multi-GB).
Each shard requires downloading the full Arrow file to scan the IPC messages.

For production, deploy as a Cloud Run job:

```bash
# Example: 500 tasks, each processing ~28 shards
gcloud run jobs create compute-offsets \
    --image=$DOCKER_IMAGE \
    --tasks=500 \
    --parallelism=100 \
    --memory=8Gi \
    --cpu=2 \
    --task-timeout=3600s \
    --args="python,scripts/compute_offsets.py"
```

The 8 GiB memory is sufficient because each task only holds one Arrow file at a
time (the largest s0 shards are ~6.6 GB). The task partitioning via
`CLOUD_RUN_TASK_INDEX` ensures no two tasks process the same shard.

Estimated wall time: ~30 minutes with 100 parallel tasks (each shard takes
~10-30 seconds: download + scan + upload the tiny CSV).

## Integration with Export Pipeline

Once offset CSVs are pre-computed, the export worker can be switched from
`ShardReader` to `ShardRangeReader` by adding the offset CSV path:

```python
# In worker.py _open_shard_with_retry():
arrow_uri = f"{source_path}/s{scale}/{shard_name}.arrow"
csv_uri = f"{source_path}/s{scale}/{shard_name}.csv"
offsets_uri = f"{source_path}/s{scale}/{shard_name}-offsets.csv"

if config.use_range_reads:
    return ShardRangeReader(arrow_uri, csv_uri, offsets_uri)
else:
    return ShardReader(arrow_uri, csv_uri)
```

A config flag (`USE_RANGE_READS` env var or similar) keeps both code paths
available for testing and fallback.

### Trade-offs

| | ShardReader (current) | ShardRangeReader |
|---|---|---|
| **Init cost** | One large GCS download (seconds to minutes) | Two small CSV downloads (~ms) |
| **Per-chunk cost** | In-memory table lookup (~us) | GCS range read (~50-100ms) |
| **Memory** | Entire Arrow file (~100 MB to 6.6 GB) | Schema + one record batch (~50 KB) |
| **GCS requests** | 2 per shard (Arrow + CSV) | 2 + N per shard (CSVs + schema + N chunks) |
| **Prerequisite** | None | Pre-computed offset CSVs |

The range reader trades GCS request count for memory. For a 258-chunk shard,
that's 260 small requests vs 2 large ones. GCS per-request overhead is ~50ms,
so the total latency increase is ~13 seconds — negligible compared to the
minutes spent encoding chunks.

For the largest shards (32K chunks), the 32K GCS requests add ~27 minutes of
round-trip latency. This can be mitigated by batching nearby chunks into a
single range read (fetch a contiguous range covering multiple record batches).
This optimization can be added later if needed.

## Files

| File | Description |
|------|-------------|
| `scripts/compute_offsets.py` | Scans Arrow files, writes offset CSVs to GCS |
| `braid/src/braid/reader.py` | `ShardRangeReader` class (alongside existing `ShardReader`) |
| `braid/tests/test_range_reader.py` | 9 parity tests against `ShardReader` |
