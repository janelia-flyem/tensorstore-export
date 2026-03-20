# Google Cloud Run Issues

## PyArrow GCS Filesystem: CURL Error 81

PyArrow's native C++ GCS filesystem (`pyarrow.fs.GcsFileSystem`) fails
intermittently on Cloud Run with CURL error 81.  The fix and full details
are documented in BRAID's architecture doc since the I/O layer lives there:

See **[braid/docs/ARCHITECTURE.md](../braid/docs/ARCHITECTURE.md)** — section
"Why not pyarrow.fs for GCS?"
