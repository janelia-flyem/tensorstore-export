# Efficiently Writing Large Datasets with Custom Logic using TensorStore

This document outlines how to use TensorStore to efficiently process a large dataset that requires custom, on-the-fly decoding, without ever needing to load the entire dataset into memory. This is a common scenario when dealing with specialized or compressed data formats.

The core idea is to use the `tensorstore.virtual_chunked` driver. This driver allows you to define a large, virtual TensorStore array that is backed by a Python function you provide. TensorStore will call your function whenever it needs a specific chunk of data, and your function is responsible for reading, decompressing, and returning just that single chunk.

## The Problem

We have a large dataset (e.g., 2048x2048x2048 `uint64` values) stored in a sharded Arrow file format. For each chunk (e.g., 64x64x64), the Arrow file contains:
1.  A list of `uint64` labels.
2.  A custom compressed binary blob.

To get the final `uint64` array for a chunk, we must read these two fields and apply a bespoke decompression algorithm. Loading the entire 2048^3 volume (64 GB) into memory is not feasible.

## The Solution: `tensorstore.virtual_chunked`

We can solve this by creating a virtual TensorStore array. TensorStore will manage the chunking, parallelism, and writing to the destination (e.g., Neuroglancer on GCS), while our custom Python code will handle the specialized decompression for each chunk as it's needed.

Here is a step-by-step guide.

### Step 1: Write a "Read Chunk" Function

First, we need a Python function that can read and decode a single `64x64x64` chunk. TensorStore will provide this function with an `IndexTransform` object, which tells our function the exact byte offsets and indices for the chunk being requested.

```python
import tensorstore as ts
import numpy as np
import pyarrow.feather as feather
import functools

# Assume you have a function that can decompress your binary data.
# It takes the compressed bytes and the label map and returns a numpy array.
def your_custom_decompressor(compressed_data: bytes, labels: list) -> np.ndarray:
    """
    Your bespoke decompression logic goes here.
    
    For this demonstration, we'll simulate a simple lookup table decompression.
    """
    # In reality, this would be your complex C++ binding or Python logic.
    raw_indices = np.frombuffer(compressed_data, dtype=np.uint8).reshape((64, 64, 64))
    label_array = np.array(labels, dtype=np.uint64)
    return label_array[raw_indices]

# Use a cache to avoid re-reading the same Arrow shard if it contains
# multiple 64^3 chunks that TensorStore might request separately.
@functools.lru_cache(maxsize=128)
def get_arrow_table(shard_path: str) -> 'pyarrow.Table':
    """Reads and caches an Arrow file from disk."""
    print(f"Reading Arrow shard: {shard_path}")
    return feather.read_table(shard_path)

def read_chunk(transform: ts.IndexTransform) -> ts.ReadResult:
    """
    This function is called by TensorStore to retrieve a single chunk.
    """
    # 1. Determine which chunk to read from the transform.
    # The transform's `input_origin` will tell us the starting index
    # of the chunk, e.g., [0, 64, 128].
    chunk_origin = transform.input_origin
    
    # You need a way to map this chunk_origin to the correct Arrow file
    # and the correct row within that file. This logic is specific to your
    # sharding scheme.
    #
    # For this example, let's assume a simple mapping:
    shard_index = chunk_origin[0] // 64  # Example logic
    row_in_shard = chunk_origin[1] // 64 # Example logic
    
    shard_path = f"/path/to/your/shards/shard_{shard_index}.arrow"
    
    # 2. Read the data from the Arrow file.
    arrow_table = get_arrow_table(shard_path)
    
    # 3. Extract the relevant fields for the specific row.
    row = arrow_table.slice(row_in_shard, 1)
    labels = row.column('labels')[0].as_py()
    compressed_data = row.column('compressed_data')[0].as_py()

    # 4. Decompress the data using your custom logic.
    decompressed_chunk = your_custom_decompressor(compressed_data, labels)
    
    # 5. Return the chunk to TensorStore.
    # `transform.domain` specifies the shape of the requested chunk.
    # We must return a NumPy array of exactly this shape.
    return ts.ReadResult(decompressed_chunk, transform=transform)
```

### Step 2: Define the `tensorstore.virtual` Spec

Next, we wrap our `read_chunk` function in a TensorStore spec. This spec describes the full, virtual dataset to TensorStore, including its total dimensions, data type, and chunk layout.

```python
# The full dimensions of your dataset
total_shape = [2048, 2048, 2048]
chunk_shape = [64, 64, 64]

virtual_source_spec = {
    'driver': 'virtual_chunked',
    'read': read_chunk,
    'dtype': 'uint64',
    'domain': {'shape': total_shape},
    'chunk_template': {'shape': chunk_shape},
    # This allows TensorStore to request chunks in parallel
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
            'size': total_shape,
            'resolution': [8, 8, 8], # Example resolution in nm
            'chunk_sizes': [chunk_shape]
        }]
    }
}

# Open the virtual source and the real destination
source_dataset = ts.open(virtual_source_spec).result()
dest_dataset = ts.open(dest_spec, create=True, delete_existing=True).result()

print("Starting copy from virtual Arrow source to GCS...")

# Execute the copy. TensorStore will now call your `read_chunk` function
# in parallel for all the 64x64x64 chunks needed to fill the volume.
copy_future = ts.copy(source_dataset, dest_dataset)
copy_future.result()

print("Copy complete!")
```

## Summary of Benefits

-   **Memory Efficiency:** You never hold the full 2048^3 array in memory. Memory usage is limited to the handful of `64x64x64` chunks being processed concurrently.
-   **Separation of Concerns:** TensorStore handles the complex scheduling, I/O, and parallelism, allowing you to provide a clean function that does one thing: decode a single chunk.
-   **Performance:** TensorStore calls your `read_chunk` function from a thread pool, giving you parallel processing for free. Using an `lru_cache` can further optimize file reads.
-   **Flexibility:** This pattern is incredibly flexible. Your source could be anything—a database, a proprietary file format, or even a procedural generator—as long as you can represent it with a Python function.
