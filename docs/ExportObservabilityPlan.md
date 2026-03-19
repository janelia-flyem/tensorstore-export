# Plan: Improving export-shards Observability

The `export-shards` command processes hundreds of millions of blocks over many hours. Errors during this process — dropped blocks, failed writes, corrupt shards — are currently logged via `dvid.Errorf()` but not tracked, aggregated, or surfaced in a way that makes them actionable. A shard can silently lose chunks and still appear complete.

This document proposes concrete improvements to make export failures visible and verifiable.

---

## Current State

### How errors are handled today

The shard writer goroutine (`shardWriter.start()`) processes blocks from a channel in a loop:

```go
for block := range w.ch {
    w.writeIndexCSV(...)
    if err := w.writeBlock(block, &ab); err != nil {
        dvid.Errorf("Error writing block %s (record %d) to shard file %s: %v", ...)
    }
}
```

When `writeBlock` fails:
- The error is logged to the DVID log
- The block is **silently dropped** — the CSV index entry was already written but the Arrow record was not
- The loop continues processing subsequent blocks
- The shard writer's `defer` cleanup closes the file normally and reports "finished after writing N records"

### What can go wrong

| Failure | Effect | Current detection |
|---------|--------|-------------------|
| `writeBlock` error (Arrow serialization) | Block dropped from Arrow file; CSV index has a dangling record reference | Log-only |
| `writeIndexCSV` error (disk I/O) | CSV index missing entry; Arrow record exists but is unreachable | Log-only |
| Disk full | Write errors cascade across all active shard writers | Log-only, no early abort |
| `UnmarshalBinary` failure in `chunkHandler` | Entire block skipped before reaching any shard writer | Log-only |
| `DeserializeData` failure in `chunkHandler` | Same as above | Log-only |
| Arrow IPC writer close error | Shard file may be truncated/corrupt | Log-only |
| CSV index flush error | Index file may be incomplete | Log-only |

### Why this matters

The downstream pipeline (tensorstore-export) trusts the Arrow+CSV files. Missing chunks become zero-filled regions in the neuroglancer precomputed volume. For scientific data, silent gaps in segmentation are worse than a failed export — they can go unnoticed during analysis.

---

## Proposed Improvements

### 1. Per-shard error counter

Track errors within each shard writer and include the count in the completion message and metrics.

**In `shardWriter`:**
```go
type shardWriter struct {
    // ... existing fields ...
    errorCount  uint64  // blocks that failed to write
    csvErrors   uint64  // CSV index write failures
}
```

**In the write loop:**
```go
if err := w.writeBlock(block, &ab); err != nil {
    dvid.Errorf("Error writing block %s (record %d) to shard file %s: %v", ...)
    w.errorCount++
}
```

**In the completion log:**
```go
if w.errorCount > 0 {
    dvid.Criticalf("Shard writer for file %s finished with %d ERRORS out of %d blocks",
        fname, w.errorCount, w.recordNum+w.errorCount)
} else {
    dvid.Infof("Shard writer for file %s finished after writing %d records", fname, w.recordNum)
}
```

**In `shardReport`:**
```go
type shardReport struct {
    // ... existing fields ...
    errorCount uint64
}
```

### 2. Don't write CSV index before Arrow record succeeds

Currently, `writeIndexCSV` is called before `writeBlock`. If the block write fails, the CSV has a dangling entry pointing to a record that doesn't exist. Reverse the order:

```go
for block := range w.ch {
    currentRecord := w.recordNum
    if err := w.writeBlock(block, &ab); err != nil {
        dvid.Errorf("Error writing block %s (record %d) to shard file %s: %v", ...)
        w.errorCount++
        continue  // skip the CSV entry
    }
    w.recordNum++
    w.writeIndexCSV(block.ChunkCoord[0], block.ChunkCoord[1], block.ChunkCoord[2], currentRecord)
}
```

### 3. Expected vs actual record count per shard

For each shard, compute the expected number of non-empty chunks based on the volume extents and shard boundaries. Compare against the actual record count at completion.

**In `shardReport`:**
```go
type shardReport struct {
    // ... existing fields ...
    expectedRecords uint64  // chunks in the shard region that had data sent to the writer channel
    actualRecords   uint64  // records successfully written to Arrow
    errorCount      uint64  // blocks that failed to write
}
```

The `chunkHandler` already knows every block it sends to a shard writer. Track sends per shard ID and compare at close time.

### 4. Export summary with error aggregation

The `exportMetrics.writeLog()` function writes a per-shard summary to `export.log`. Extend it to:

- Flag shards with errors at the top of the file
- Include a global error summary
- Write a machine-readable JSON manifest alongside the human-readable log

**`export.log` additions:**
```
=== EXPORT SUMMARY ===
Total shards: 21,994
Shards with errors: 3
Total blocks dropped: 47

SHARDS WITH ERRORS:
  s0/36864_43008_108544.arrow: 2 errors (32766/32768 records written)
  s0/55296_69632_100352.arrow: 44 errors (28668/28712 records written)
  s1/30720_24576_28672.arrow: 1 error (257/258 records written)
```

**`export-manifest.json`:**
```json
{
  "start_time": "2026-03-19T00:00:00Z",
  "end_time": "2026-03-19T11:00:00Z",
  "total_shards": 25474,
  "total_records": 554000000,
  "total_errors": 47,
  "scales": [
    {
      "scale": 0,
      "shards": 21994,
      "records": 550000000,
      "errors": 46,
      "shards_with_errors": ["s0/36864_43008_108544.arrow", "s0/55296_69632_100352.arrow"]
    }
  ]
}
```

The JSON manifest serves two purposes:
- **Human review**: quickly check if any errors occurred without parsing the full log
- **Machine consumption**: the tensorstore-export pipeline can read this manifest before processing, skipping or flagging shards with known errors

### 5. Validate Arrow+CSV consistency at close time

When a shard writer finishes, perform a lightweight integrity check:

```go
// After closing the Arrow file, reopen and verify record count
func (w *shardWriter) validate() error {
    f, err := os.Open(w.f.Name())
    if err != nil {
        return fmt.Errorf("cannot reopen for validation: %v", err)
    }
    defer f.Close()

    reader, err := ipc.NewReader(f)
    if err != nil {
        return fmt.Errorf("cannot create Arrow reader: %v", err)
    }
    defer reader.Release()

    var arrowRecords uint64
    for reader.Next() {
        arrowRecords++
    }
    if arrowRecords != w.recordNum {
        return fmt.Errorf("Arrow record count %d != expected %d", arrowRecords, w.recordNum)
    }
    return nil
}
```

This adds a small amount of I/O at close time (re-reading the file) but catches truncation or corruption. It could be gated behind a `--validate` flag if the overhead is a concern.

### 6. Early abort on critical failures

Some errors should stop the export rather than silently continuing:

- **Disk full**: If any write returns `ENOSPC`, abort the entire export immediately rather than producing hundreds of corrupt shards. Check for this in `writeBlock` and propagate via a shared context cancellation.
- **High error rate**: If a shard exceeds a threshold (e.g., >1% of blocks failing), log a critical warning. If the global error rate exceeds a threshold, consider aborting.

```go
// In chunkHandler, check for global abort
select {
case <-ctx.Done():
    return  // export aborted
default:
}
```

---

## Implementation Priority

| Change | Effort | Impact | Priority |
|--------|--------|--------|----------|
| Per-shard error counter + log | Small | High — makes errors visible | 1 |
| Reorder CSV/Arrow writes | Small | Medium — prevents dangling CSV entries | 2 |
| Export summary with error counts | Small | High — single place to check for problems | 3 |
| JSON manifest | Medium | High — enables machine-readable validation | 4 |
| Early abort on disk full | Medium | High — prevents cascade of corrupt shards | 5 |
| Arrow+CSV validation at close | Medium | Medium — catches corruption | 6 |
| Expected vs actual record tracking | Medium | Medium — detects missing blocks | 7 |

Changes 1–3 are straightforward modifications to `export.go` that could be done in a single commit. Change 4 (JSON manifest) integrates with the tensorstore-export pipeline. Changes 5–7 are more involved but address rarer failure modes.

---

## Appendix: Checking Existing Export Logs for Errors

Until the above improvements are implemented, here's how to audit an existing `export-shards` run from the DVID log. All error messages from the export code path use `dvid.Errorf`, which prefixes log lines with `ERR` or `ERROR` depending on the logging configuration.

### Quick check: any errors at all?

```bash
grep -i "error\|errorf\|failed\|critical" dvid.log | grep -i "export\|shard\|block\|arrow\|csv\|index"
```

### Specific error patterns to search for

**Block-level failures (data lost from a shard):**

```
Error writing block (X,Y,Z) (record N) to shard file PATH
```
The block was dropped from the Arrow file. The CSV index may have a dangling entry for record N.

```
failed to unmarshal block data for (X,Y,Z)
```
The DVID compressed block couldn't be parsed. The entire block is skipped before reaching any shard writer.

```
Unable to deserialize block (X,Y,Z) in data "NAME"
```
The raw storage value couldn't be deserialized (gzip/lz4 decompression failed). Block is skipped.

```
Nil data for label block (X,Y,Z) in data "NAME"
```
A key existed in the database but had no value. Block is skipped.

```
Couldn't decode label block key ... for data "NAME"
```
A storage key couldn't be parsed into a scale + block coordinate. Block is skipped.

**Shard writer failures (entire shard may be corrupt):**

```
Error closing Arrow IPC writer for PATH
```
The Arrow streaming file may be truncated. Downstream readers may fail to parse this shard.

```
Error flushing index PATH
```
The CSV index file may be incomplete. Some chunks present in the Arrow file won't be discoverable via the index.

```
Error closing shard file PATH
```
The Arrow file wasn't closed cleanly. May be truncated.

```
failed to open shard file PATH
```
A shard writer couldn't be created — all blocks for this shard are lost.

```
failed to start shard writer for PATH
```
Same as above — the shard writer goroutine failed to initialize.

**Infrastructure failures:**

```
error initializing store
```
The underlying key-value database couldn't be opened. The entire export fails.

```
problem during process range
```
A database scan encountered an error. Some blocks in the affected range may be missing.

**Memory/resource issues:**

```
Arrow allocator leak for PATH: N bytes still allocated
```
An Arrow memory leak was detected after closing a shard writer. Not a data loss issue but indicates a bug.

```
error closing zstd encoder for PATH
```
The zstd compressor didn't close cleanly. Unlikely to affect data already written but indicates resource exhaustion.

### Verifying shard completeness

To check whether a specific shard has the expected number of records:

```bash
# Count CSV index entries (subtract 1 for header)
wc -l < PATH.csv  # should be N+1 where N is the expected record count

# Compare against the "finished after writing N records" log line
grep "finished.*PATH" dvid.log
```

If the CSV line count doesn't match the logged record count, the CSV index is inconsistent with the Arrow data.

### Bulk check across all shards

```bash
# Find all "finished" lines and extract record counts
grep "Shard writer for file.*finished" dvid.log | \
    sed 's/.*file \(.*\) finished after writing \([0-9]*\) records/\1 \2/' | \
    while read file count; do
        csv="${file%.arrow}.csv"
        if [ -f "$csv" ]; then
            csv_count=$(($(wc -l < "$csv") - 1))
            if [ "$csv_count" -ne "$count" ]; then
                echo "MISMATCH: $file arrow=$count csv=$csv_count"
            fi
        else
            echo "MISSING CSV: $csv"
        fi
    done
```

This script compares the Arrow record count (from the log) against the CSV line count for every shard. Any mismatch indicates a write-ordering bug or a failed flush.
