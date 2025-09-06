# Design for a Highly Parallel Shard Export System using TensorStore

This document outlines a design for converting a very large dataset, stored in custom-formatted, sharded Arrow files, into a single, coherent Neuroglancer precomputed volume. The system is designed to be highly parallel, with many independent workers processing shards concurrently without requiring a central coordinator.

## 1. The Core Problem: Memory-Efficient Custom Data Transformation

The initial challenge is to process a large dataset (e.g., 2048x2048x2048 `uint64` values) that is too large to fit in memory (64 GB). The data is stored in a sharded Arrow file format where each chunk (e.g., 64x64x64) requires custom decompression logic.

**Data Format per Chunk:**
*   A list of `uint64` labels (a lookup table).
*   A custom compressed binary blob that represents indices into the label table.

A naive approach would require loading the entire 2048^3 volume, which is not feasible.

## 2. The Solution: `tensorstore.virtual_chunked`

TensorStore's `virtual_chunked` driver is the ideal solution. It allows us to define a large, virtual TensorStore array that is backed by a Python function. TensorStore handles all the scheduling, I/O, and chunk management, while our custom function provides the logic for reading and decompressing a single chunk on-the-fly.

*(For a basic overview of this single-machine setup, see `EfficientWrites.md`)*

---

## 3. Evolving to a Highly Parallel, Distributed Design

The real-world requirement is to process a much larger volume (e.g., `[94088, 78317, 134576]`) composed of tens of thousands of `2048^3` shard files. This requires a distributed system where many workers (e.g., on Google Cloud Run) can process shards in parallel and write to a single, shared destination volume.

This distributed workflow fundamentally changes the design from a single large copy to many small, targeted writes.

### 3.1. The Core Concept: One Global Volume, Many Partial Writes

The key to the distributed design is that **all workers write to the same, single destination TensorStore volume, but each worker only writes to its assigned slice.**

*   **Global Destination:** The destination Neuroglancer volume is defined with the **full, final dimensions** of the entire dataset. This is critical for creating a single, coherent `info` metadata file on GCS that describes the complete volume.
*   **Worker-Specific Source:** Each worker creates a temporary, virtual source for **one shard file at a time**. The domain of this source is small (e.g., `[2048, 2048, 2048]`).
*   **Targeted Copy:** The worker copies its small source into the correct **offset** within the giant destination volume.

TensorStore's write operations are atomic at the chunk level, making this approach safe from race conditions.

### 3.2. State Management for Workers

To coordinate the work without a central database, a simple and robust state management system can be implemented directly on GCS using directory structures.

*   `gs://<bucket>/unprocessed/`: All shard files start here.
*   `gs://<bucket>/processing/`: A worker transactionally moves a shard file here before it begins processing. This acts as a lock.
*   `gs://<bucket>/finished/`: After a worker successfully processes a shard, it moves the file here.

---

## 4. Optimal Arrow Shard File Design for Fast Random Access

A sequential scan of a large Arrow file to find a specific chunk would be a major performance bottleneck, especially since TensorStore may request chunks in a non-sequential (e.g., Morton) order. The shard file format must be optimized for fast, random-access reads.

### 4.1. Use the Arrow IPC File Format

The key is to use the **Arrow IPC File Format** (also known as Feather V2) instead of the Arrow Streaming Format. The IPC File Format is designed for random access; it contains a footer that acts as a built-in index, storing the exact byte offset and length of every Record Batch in the file.

### 4.2. One Record Batch Per Chunk

Each `64x64x64` chunk will be stored as a single `pyarrow.RecordBatch`. Since each shard is `2048x2048x2048` voxels, it is composed of a `32x32x32` grid of chunks. Therefore, each shard Arrow file will contain `32 * 32 * 32 = 32,768` record batches.

### 4.3. Companion Index File

To avoid having each worker read metadata from the Arrow file to build a coordinate-to-batch map, we will pre-compute a companion index file. For each `shard_name.arrow`, we will generate `shard_name.arrow.index.json`.

This tiny JSON file will contain a `32x32x32` array where the value at `[z][y][x]` is the integer index of the record batch corresponding to that chunk. This allows for an instant lookup.

### 4.4. Shard File Schema

The `pyarrow.Table` within the shard file will use the following schema:

```python
import pyarrow as pa

SCHEMA = pa.schema([
    # Chunk coordinates relative to the shard origin (0-31)
    pa.field('chunk_coord_x', pa.int32(), nullable=False),
    pa.field('chunk_coord_y', pa.int32(), nullable=False),
    pa.field('chunk_coord_z', pa.int32(), nullable=False),
    
    # Custom data fields
    pa.field('labels', pa.list_(pa.uint64()), nullable=False),
    pa.field('supervoxels', pa.list_(pa.uint64()), nullable=False),
    pa.field('payload', pa.binary(), nullable=False)
])
```

---

## 5. Implementation for a Distributed Worker (Revised)

Here is the revised step-by-step logic for a single worker, incorporating the optimal Arrow reading strategy.

### 5.1. Define the Global Destination (Common to all workers)

This part of the design remains unchanged. Every worker uses the same spec with the full dataset dimensions.

```python
import tensorstore as ts
import numpy as np

# This is the GLOBAL shape of the final dataset
total_shape = [94088, 78317, 134576] 
chunk_shape = [64, 64, 64]

# Every worker uses the SAME destination spec
dest_spec = {
    'driver': 'neuroglancer_precomputed',
    'kvstore': {'driver': 'gcs', 'bucket': 'your-bucket', 'path': 'path/to/global-volume'},
    'metadata': { # ... metadata ... 
    },
    'open': True, 'create': True, 'delete_existing': False
}

global_dest_dataset = ts.open(dest_spec).result()
```

### 5.2. The Worker's Main Processing Loop with Efficient Reading

The worker's `read_single_shard_chunk` function is now implemented to be highly efficient.

```python
import pyarrow.ipc as ipc
import gcsfs
import json
import functools

# Assume this is your custom logic
def your_custom_decompressor(compressed_data: bytes, labels: list) -> np.ndarray:
    # ... decompression logic ...
    pass

@functools.lru_cache(maxsize=10) # Cache a few open readers/indices
def get_shard_data(shard_gcs_path, fs):
    """Reads the index and opens the Arrow file reader, caching the result."""
    index_path = shard_gcs_path + ".index.json"
    with fs.open(index_path, 'r') as f:
        index_map = json.load(f)["zyx_to_batch_index"]
    
    arrow_file = fs.open(shard_gcs_path, 'rb')
    reader = ipc.open_file(arrow_file)
    return reader, index_map

def read_single_shard_chunk(transform: ts.IndexTransform, shard_gcs_path: str) -> ts.ReadResult:
    fs = gcsfs.GCSFileSystem()

    # 1. Get the cached reader and index map for this shard.
    reader, index_map = get_shard_data(shard_gcs_path, fs)

    # 2. Calculate the chunk coordinate within the 32x32x32 grid.
    origin = transform.input_origin
    chunk_x, chunk_y, chunk_z = origin[0] // 64, origin[1] // 64, origin[2] // 64

    # 3. Find the record batch index via instant lookup.
    batch_index = index_map[chunk_z][chunk_y][chunk_x]

    # 4. Read ONLY that one record batch (fast random access).
    record_batch = reader.get_record_batch(batch_index)
    
    # 5. Extract data and decompress.
    data = record_batch.to_pydict()
    decompressed_chunk = your_custom_decompressor(data['payload'][0], data['labels'][0])
    
    return ts.ReadResult(decompressed_chunk, transform=transform)

# --- Worker's Main Loop ---
# The main loop remains structurally the same, but now calls the efficient
# read function. (Code for moving files and setting up the copy is omitted for brevity).

# for shard_gcs_path in assigned_shard_files:
    # ... move file to 'processing' ...
    # ... get shard_global_origin ...
    # ... create source_spec with the efficient read_single_shard_chunk ...
    # ... define dest_slice ...
    # ... ts.copy(source, dest_slice).result() ...
    # ... move file to 'finished' ...
```