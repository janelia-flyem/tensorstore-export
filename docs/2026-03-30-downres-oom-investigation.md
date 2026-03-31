# Downres OOM Investigation

Date: 2026-03-30
Branch: `s0-derived`

## Scope

This note summarizes the current `--downres-mode` implementation in
`tensorstore-export`, the TensorStore code paths it relies on, and the most
likely reasons the current memory-tier model can under-predict Cloud Run Task
RSS during downres.

Active dataset configuration from `.env` at the time of review:

- `SOURCE_PATH=gs://flyem-dvid-shards/mCNS-d79556/segmentation`
- `DEST_PATH=gs://flyem-ng-staging/v1.0/segmentation`
- `NG_SPEC_PATH=/home/katzw/tensorstore-export/examples/mcns-export-specs.json`
- `SCALES=0`

## Current Downres Path

The manifest-driven downres worker is implemented in
[`src/worker.py`](/home/katzw/tensorstore-export/src/worker.py#L659).

Per output shard, `downres_shard()` does:

1. Open scale `N-1` from the destination bucket on GCS with a 256 MiB cache
   limit.
2. Construct a lazy `ts.downsample(source, [2,2,2,1], "mode")` view.
3. Create a fresh local Neuroglancer store in a per-shard tmpfs staging dir.
4. For each output Z-plane, run one explicit TensorStore transaction:
   `local_dest.with_transaction(txn)[bbox].write(downsampled[bbox]).result()`
   then `txn.commit_async().result()`.
5. Re-open the staged shard and read back every output chunk to compute actual
   label counts for the next scale.
6. Upload the staged shard file(s) to GCS and delete the staging dir.

Relevant code:

- Source open + downsample view:
  [`src/worker.py`](/home/katzw/tensorstore-export/src/worker.py#L707)
- Local staged destination:
  [`src/worker.py`](/home/katzw/tensorstore-export/src/worker.py#L729)
- One-Z-plane explicit transactions:
  [`src/worker.py`](/home/katzw/tensorstore-export/src/worker.py#L742)
- Post-write label readback:
  [`src/worker.py`](/home/katzw/tensorstore-export/src/worker.py#L809)

## What The Current Model Assumes

The design doc frames downres memory as mostly:

`source cache + output shard tmpfs + fixed overhead`

and explicitly says the only unknown is output shard size:

- [`docs/ExportShardsOptimization.md`](/home/katzw/tensorstore-export/docs/ExportShardsOptimization.md#L23)
- [`docs/ExportShardsOptimization.md`](/home/katzw/tensorstore-export/docs/ExportShardsOptimization.md#L35)

The code currently estimates downres memory as:

`raw_batch_gib + 2 * tmpfs_gib + DOWNRES_OVERHEAD_GIB`

with:

- `DOWNRES_OVERHEAD_GIB = 1.0`
  [`scripts/precompute_manifest.py`](/home/katzw/tensorstore-export/scripts/precompute_manifest.py#L124)
- `raw_batch_gib` derived from `chunk_count ** (2/3)` assuming one output
  Z-plane worth of raw `uint64` chunks
  [`scripts/precompute_manifest.py`](/home/katzw/tensorstore-export/scripts/precompute_manifest.py#L232)

There is a mismatch between doc and code here:

- The optimization doc still says downres tmpfs is `1× output shard`
  [`docs/ExportShardsOptimization.md`](/home/katzw/tensorstore-export/docs/ExportShardsOptimization.md#L29)
- The implementation estimate already uses `2× tmpfs` for downres
  [`scripts/precompute_manifest.py`](/home/katzw/tensorstore-export/scripts/precompute_manifest.py#L235)

That doc mismatch is not the main problem, but it is worth fixing because it
obscures which terms are already accounted for.

## TensorStore Behavior That Matters

### 1. Sharded Neuroglancer writes are shard-level RMW

TensorStore documents that writes to a sharded
`neuroglancer_precomputed` volume are coalesced per shard transaction, but the
shard still commits via read-modify-write against the shard key:

- [`/home/katzw/tensorstore/tensorstore/transaction_impl.h`](/home/katzw/tensorstore/tensorstore/transaction_impl.h#L157)

The sharded kvstore writeback path explicitly merges new encoded chunks with the
existing encoded shard contents:

- [`/home/katzw/tensorstore/tensorstore/kvstore/neuroglancer_uint64_sharded/neuroglancer_uint64_sharded.cc`](/home/katzw/tensorstore/tensorstore/kvstore/neuroglancer_uint64_sharded/neuroglancer_uint64_sharded.cc#L751)

For a fresh local staging dir this usually means unconditional writes early, but
after the first batch it still means rebuilding the complete shard state as the
file grows.

### 2. `cache_pool.total_bytes_limit` helps cache eviction, not explicit transaction state

The worker comments already note this correctly:

- [`src/worker.py`](/home/katzw/tensorstore-export/src/worker.py#L742)

This is important because the source and destination stores each specify a
256 MiB cache limit, but the explicit transaction can still hold much more
memory than that while the batch is being staged and committed.

### 3. Cache pool docs are read-oriented

TensorStore's sharded kvstore schema docs describe `cache_pool` mainly in terms
of avoiding extra reads for shard and minishard indices:

- [`/home/katzw/tensorstore/tensorstore/kvstore/neuroglancer_uint64_sharded/schema.yml`](/home/katzw/tensorstore/tensorstore/kvstore/neuroglancer_uint64_sharded/schema.yml#L24)

That lines up with the repo's current source-side assumption, but it should not
be interpreted as a hard upper bound for total process memory during write
transactions.

### 4. Compressed-segmentation encoding allocates a new encoded output buffer per chunk

The chunk encoder writes encoded output into a fresh `std::string` before
wrapping it in an `absl::Cord`:

- [`/home/katzw/tensorstore/tensorstore/driver/neuroglancer_precomputed/chunk_encoding.cc`](/home/katzw/tensorstore/tensorstore/driver/neuroglancer_precomputed/chunk_encoding.cc#L327)

That means a batch can temporarily hold:

- raw decoded arrays for the pending writes
- encoded per-chunk chunk payloads
- encoded shard state assembled for writeback

The current model mainly captures the first and third terms.

## Likely Reasons The Current Formula Under-Predicts

## 1. The worker adds a full post-write readback phase that the tier model treats as overhead

After the shard is fully written, the worker re-opens the local staged volume
and reads every chunk back to compute exact label sets for the next scale:

- [`src/worker.py`](/home/katzw/tensorstore-export/src/worker.py#L809)

This phase allocates chunk arrays again and also stores Python `set`s of labels
for the shard in `actual_labels`. For dense low-resolution chunks, that Python
object graph can be non-trivial. The estimator rolls this into a fixed 1.0 GiB
overhead, but this term scales with shard content, not just process startup.

## 2. `raw_batch_gib = N^(2/3) * 2 MiB` is probably a lower bound, not a peak model

The estimate assumes one output Z-plane worth of raw output chunks.

What the actual write path needs at peak is closer to:

- source chunks needed to compute the output plane
- output chunk arrays staged in the explicit transaction
- encoded chunk payloads produced during `EncodeCompressedSegmentationChunk`
- shard-level merge state during `MergeForWriteback`

Because downsample is lazy, that source-side working set depends on how
TensorStore schedules reads for the write request. The current formula does not
include an explicit source-array term beyond the 256 MiB cache cap.

## 3. The label-aware model predicts final shard bytes well, but not transient memory

`BYTES_PER_UNIQUE_LABEL` looks useful for predicting final shard size, but OOM is
caused by transient peak RSS during:

- downsample read/decode
- transaction staging
- compressed-segmentation encode
- shard merge/writeback
- label readback

The optimization doc correctly notes that validating downres predictions
against observed peak memory is still outstanding:

- [`docs/ExportShardsOptimization.md`](/home/katzw/tensorstore-export/docs/ExportShardsOptimization.md#L231)

## 4. One output Z-plane may still be too large for worst-case shards

The worker always batches one full output chunk plane in Z:

- [`src/worker.py`](/home/katzw/tensorstore-export/src/worker.py#L748)

That keeps the batch count manageable, but it also means X and Y are unbounded
within the shard. For large interior shards, the transaction width in X/Y may be
the actual driver of peak memory. If OOMs are clustered in dense interior shards,
reducing batch shape in X/Y as well as Z is the first runtime lever I would try.

## Most Plausible Working Hypothesis

The OOMs are not primarily because the final `.shard` size prediction is wrong.
They are more likely because the peak RSS during one batched downres commit is:

`source working set + transaction raw arrays + encode scratch + shard merge state + tmpfs shard + label readback`

while the current tiering model is approximately:

`raw_batch + 2 * final_shard + 1 GiB`

That is close in structure, but it compresses several data-dependent terms into a
single fixed overhead that is probably too small for the worst `s0 -> s1`
interior shards.

## Immediate Next Checks

1. Compare `peak_memory_gib` vs `tmpfs_mib` from the `Downres write batch` logs.
   If peak RSS exceeds `2 * tmpfs + raw_batch` by a large margin, the missing
   term is real and should be modeled explicitly.
2. Split label prediction from the critical write path temporarily.
   Disable the post-write readback step and see how much peak memory drops.
3. Try smaller write batches than one full output Z-plane.
   The cleanest experiment is halving the batch in X or Y for the largest tier.
4. Calibrate from actual downres logs, not final shard sizes.
   The current regression work is good for shard-size prediction but not yet a
   peak-memory model.

## Suggested Code Directions

If the goal is to stabilize production quickly, I would test these in order:

1. Remove label readback from `downres_shard()` and do prediction in a separate
   pass or a lower-memory approximation.
2. Add adaptive batch sizing based on current cgroup memory rather than a fixed
   one-Z-plane batch.
3. Log estimated `raw_batch_gib`, final `tmpfs_mib`, and observed
   `peak_memory_gib` together per shard so the estimator can be fitted to the
   actual path.
4. If needed, inspect TensorStore scheduling around the downsample read path in
   more detail, but the repo-side readback and fixed batch geometry already look
   like the first two things to change.
