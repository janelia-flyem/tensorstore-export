# BRAID Architecture

## I/O Strategy

BRAID's `ShardReader` accepts both local file paths and GCS URIs (`gs://bucket/path`).
The I/O backend is selected by path scheme:

| Path type | Backend | Notes |
|-----------|---------|-------|
| Local (`/path/to/file`) | `pathlib.Path.read_bytes()` | Standard file I/O, full file materialized in memory |
| GCS (`gs://bucket/path`) | `google.cloud.storage` | Lazy import; install with `pip install braid[gcs]` |

Both backends download the full file into Python bytes, then parse via
`pa.BufferReader` into Arrow's IPC reader (for `.arrow` files) or CSV reader
(for `.csv` index files).

### Why not pyarrow.fs for GCS?

PyArrow ships a native C++ GCS filesystem (`pyarrow.fs.GcsFileSystem`) built on
Google's C++ Cloud SDK and libcurl.  In theory this is the ideal path: it returns
`NativeFile` objects that stay in Arrow's C++ memory space, enabling zero-copy
IPC deserialization without Python bytes intermediaries.

In practice, **it fails intermittently on Google Cloud Run** with:

```
google::cloud::Status(UNKNOWN: Permanent error in Read(): EasyPause()
  - CURL error [81]=Socket not ready for send/recv)
```

This was discovered in March 2026 during the mCNS dataset export (26K shard
files, 4.3 TB, 200 Cloud Run tasks).  Details:

- **Intermittent**: ~50% of first attempts fail; different files, different workers
- **Retries help partially**: Exponential backoff (5 attempts, 1/2/4/8/16s waits)
  recovered some shards, but many still failed after exhausting all retries
- **Root cause**: libcurl socket state management inside Cloud Run's containerized
  network stack.  The error comes from PyArrow's C++ runtime, not Python.
- **Not reproducible locally** or on standard GCE VMs — specific to Cloud Run

The `google-cloud-storage` Python SDK (`blob.download_as_bytes()`) uses
`requests`/`urllib3` instead of libcurl and works reliably in the same
environment.

If PyArrow fixes this in a future release, the GCS backend could be switched
back to `pyarrow.fs` to regain the zero-copy streaming path (see below).

## Future: Streaming Batch Iteration

The current `ShardReader` calls `read_all()` to materialize the entire Arrow
table in memory.  For a 160 MB shard file at scale 0, this means ~160 MB+ of
Arrow table data resident in memory before any chunks are processed.

A more memory-efficient architecture would iterate record batches without
materializing the full table:

```python
# Instead of:
table = ipc.open_stream(buf).read_all()

# Stream batches one at a time:
reader = ipc.open_stream(buf)
for batch in reader:
    # process this batch's chunks, then discard
    ...
```

This matters most for:

1. **Local VAST storage with large files** — processing multi-GB shards where
   peak memory is constrained.  Combined with `pyarrow.fs.LocalFileSystem`,
   which returns `NativeFile` objects for true zero-copy streaming through
   Arrow's C++ runtime, this eliminates both the Python bytes copy and the
   full-table materialization.

2. **TensorStore `virtual_chunked` driver** — instead of pre-loading all chunks
   and writing them sequentially, a `virtual_chunked` source can produce chunk
   data on demand.  The batch iterator feeds chunks as they're needed, and
   TensorStore orchestrates the writes.  This is the natural architecture for
   memory-efficient format conversion (neuroglancer precomputed, zarr v3, etc.).

For GCS on Cloud Run, streaming is less beneficial since `blob.download_as_bytes()`
already materializes the file in memory before parsing.  But for local I/O, the
combination of `pyarrow.fs.LocalFileSystem` + batch iteration would keep memory
at O(one batch) instead of O(entire file).

### Restoring pyarrow.fs for local paths

The current implementation uses `Path.read_bytes()` for local files — simple and
correct, but forces an extra Python bytes copy.  Restoring
`pyarrow.fs.LocalFileSystem` for local paths would enable:

- `NativeFile` objects staying in Arrow's C++ memory space
- Streaming IPC deserialization without Python intermediaries
- Compatibility with the batch iteration pattern above

This is a natural optimization to make when implementing streaming iteration.

## Known Use Cases

- **tensorstore-export**: Cloud Run + GCS, converting DVID Arrow IPC shards to
  neuroglancer precomputed volumes.  Uses `google.cloud.storage` backend.
- **fVDB IndexGrid**: Local VAST storage, generating NanoVDB IndexGrid topology
  files + sidecar supervoxel data.  Will benefit from the streaming + pyarrow.fs
  local path.
