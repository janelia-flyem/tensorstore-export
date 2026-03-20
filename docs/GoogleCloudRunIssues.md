# Google Cloud Run Issues

## PyArrow GCS Filesystem: CURL Error 81

**Date discovered:** 2026-03-19
**Status:** Worked around by switching to `google-cloud-storage`

### Problem

PyArrow's native C++ GCS filesystem (`pyarrow.fs.GcsFileSystem`, which uses
Google's C++ Cloud SDK and libcurl internally) fails intermittently on Google
Cloud Run with:

```
google::cloud::Status(UNKNOWN: Permanent error in Read(): EasyPause()
  - CURL error [81]=Socket not ready for send/recv)
```

The error occurs when reading files from GCS via `pyarrow.fs.FileSystem.from_uri("gs://...")`.
It is **intermittent** — some reads succeed while others fail, even for the same
file on different workers. Retrying with exponential backoff (up to 5 attempts,
1/2/4/8/16 second waits) helps some cases but many shards still fail after
exhausting all retries.

### Context

- 200 Cloud Run tasks reading ~130 Arrow IPC files each (160 MB average for s0)
- PyArrow 18.x on Python 3.12, Cloud Run gen2 (8Gi memory, 4 CPU)
- The error comes from libcurl inside PyArrow's C++ runtime, not from Python
- This error does **not** occur with `google-cloud-storage` (the Python GCS SDK),
  which uses `requests`/`urllib3` rather than libcurl

### What We Tried

1. **Direct `pyarrow.fs` reads** — failed intermittently on ~50% of first attempts
2. **Retry with exponential backoff** (5 retries) — some shards recovered,
   many still failed after all retries
3. **`google-cloud-storage` download to memory** — **works reliably**

### Solution

Replaced `pyarrow.fs.GcsFileSystem` with `google.cloud.storage` for all GCS
reads in BRAID's `ShardReader`. The approach:

1. Download the full file bytes via `blob.download_as_bytes()`
2. Parse from memory using `pa.BufferReader(data)` → `ipc.open_file()` / `ipc.open_stream()`
3. CSV parsed via `pcsv.read_csv(pa.BufferReader(data))`

This adds one memory copy compared to PyArrow's zero-copy native path, but with
8Gi per worker and files averaging 160MB, this is not a constraint.

Local file paths continue to use standard Python file I/O (`Path.read_bytes()`),
so tests and local development are unaffected.

### Affected Files

- `braid/src/braid/reader.py` — GCS reads via `google-cloud-storage` instead of `pyarrow.fs`
- `src/worker.py` — retry logic still present for transient GCS errors

### References

- PyArrow issue tracker: search for "CURL error 81" / "EasyPause"
- This appears related to Cloud Run's network stack and how libcurl manages
  socket state in containerized environments with connection pooling
