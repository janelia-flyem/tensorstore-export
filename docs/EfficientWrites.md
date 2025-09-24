# Efficiently Writing of a Shard using TensorStore

This document introduces the core concepts of using `tensorstore.virtual_chunked` for custom, memory-efficient data processing. It outlines how to use TensorStore to efficiently process a large shard of data that requires custom, on-the-fly decoding, without ever needing to load the entire dataset into memory.

The core idea is to use the `tensorstore.virtual_chunked` driver. This driver allows you to define a large, virtual TensorStore array that is backed by a Python function you provide. TensorStore will call your function whenever it needs a specific chunk of data, and your function is responsible for reading, decompressing, and returning just that single chunk.

## The Problem

We have a large dataset (e.g., 2048x2048x2048 `uint64` values) stored in a sharded Arrow file format. For each chunk (e.g., 64x64x64), the Arrow file contains:
1.  A list of `uint64` agglomerated labels.
2.  A list of `uint64` unique supervoxel labels.
2.  A custom compressed binary blob.

To get the final `uint64` array for a chunk, we must read these three fields and apply a bespoke decompression algorithm. Loading the entire 2048^3 volume (64 GB) into memory is not efficient.

## The Solution: `tensorstore.virtual_chunked`

We can solve this by creating a virtual TensorStore array. TensorStore will manage the chunking, parallelism, and writing to the destination, while our custom Python code will handle the specialized decompression for each chunk as it's needed.

Here is a step-by-step guide.

### Step 1: Write a "Read Chunk" Function

First, we need a Python function that can read and decode a single `64x64x64` chunk. For this to be efficient in a parallel environment, the source Arrow files must be written in the **Arrow IPC File Format** (not the streaming format) and contain an index in their metadata to allow for fast random access.

The following function is the recommended implementation for a worker that needs to read a specific chunk from a single shard file.

```python
import tensorstore as ts
import numpy as np
import pyarrow.ipc as ipc
import gcsfs
import json
import functools

# Assume this is your custom logic
def your_custom_decompressor(compressed_data: bytes, labels: list) -> np.ndarray:
    """
    Your bespoke decompression logic goes here.
    """
    # In reality, this would be your complex C++ binding or Python logic.
    raw_indices = np.frombuffer(compressed_data, dtype=np.uint8).reshape((64, 64, 64))
    label_array = np.array(labels, dtype=np.uint64)
    return label_array[raw_indices]

@functools.lru_cache(maxsize=10) # Cache a few open readers/indices
def get_shard_data_from_metadata(shard_gcs_path, fs):
    """Reads the index from metadata and opens the Arrow file reader."""
    arrow_file = fs.open(shard_gcs_path, 'rb')
    reader = ipc.open_file(arrow_file)
    
    # Read the index from the Arrow file's metadata
    metadata = reader.schema.metadata
    if b'chunk_index_json' not in metadata:
        raise RuntimeError(f"Index not found in metadata for shard: {shard_gcs_path}")
    index_json_string = metadata[b'chunk_index_json']
    index_map = json.loads(index_json_string)
    
    return reader, index_map

def read_single_shard_chunk(transform: ts.IndexTransform, shard_gcs_path: str) -> ts.ReadResult:
    """
    This function is called by TensorStore to retrieve a single chunk from a
    specific, pre-defined shard file.
    """
    fs = gcsfs.GCSFileSystem()

    # 1. Get the cached reader and index map for this shard.
    reader, index_map = get_shard_data_from_metadata(shard_gcs_path, fs)

    # 2. Calculate the chunk coordinate within the shard's 32x32x32 grid.
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
```

### Step 2: Define the `tensorstore.virtual` Spec

Next, we wrap our `read_chunk` function in a TensorStore spec. This spec describes the virtual dataset to TensorStore. Note that we use a `lambda` to pass the specific path of the shard file we are processing to our read function.

```python
# The dimensions of a single shard
shard_shape = [2048, 2048, 2048]
chunk_shape = [64, 64, 64]
path_to_one_shard = 'gs://your-bucket/processing/shard_001.arrow'

virtual_source_spec = {
    'driver': 'virtual_chunked',
    # Use a lambda to parameterize the read function with the shard path
    'read': lambda transform: read_single_shard_chunk(transform, path_to_one_shard),
    'dtype': 'uint64',
    'domain': {'shape': shard_shape},
    'chunk_template': {'shape': chunk_shape},
    'transactional': False 
}
```

### Step 3: Put It All Together in the Copy Operation

Finally, we use this virtual spec as the source for a `tensorstore.copy` operation. The destination can be any TensorStore-supported driver, such as Neuroglancer on GCS.

```python
# Define the destination spec (e.g., Neuroglancer on GCS)
dest_spec = {
    'driver': 'neuroglancer_precomputed',
    'kvstore': {'driver': 'gcs', 'bucket': 'your-gcs-bucket', 'path': 'path/to/dataset'},
    'metadata': {
        'data_type': 'uint64',
        'num_channels': 1,
        'scales': [{
            'key': '2048_2048_2048',
            'size': shard_shape,
            'resolution': [8, 8, 8], # Example resolution in nm
            'chunk_sizes': [chunk_shape]
        }]
    }
}

# Open the virtual source and the real destination
source_dataset = ts.open(virtual_source_spec).result()
dest_dataset = ts.open(dest_spec, create=True, delete_existing=True).result()

print(f"Starting copy for shard: {path_to_one_shard}...")

# Execute the copy. TensorStore will now call `read_single_shard_chunk`
# in parallel for all the chunks needed to fill the volume.
copy_future = ts.copy(source_dataset, dest_dataset)
copy_future.result()

print("Copy complete!")
```

## Summary of Benefits

-   **Memory Efficiency:** You never hold the full shard array in memory. Memory usage is limited to the handful of chunks being processed concurrently.
-   **Separation of Concerns:** TensorStore handles the complex scheduling and I/O, allowing you to provide a clean function that decodes a single chunk.
-   **Performance:** By using the Arrow IPC File Format with a metadata index, the `read_single_shard_chunk` function can perform fast, random-access reads, which is critical for performance.
-   **Flexibility:** This pattern is incredibly flexible and forms the core of the highly parallel worker design described in `ShardExportDesign.md`.