# Architectural Guide: Parallel Ingestion of Pre-Sharded Data into a Neuroglancer Volume

This document outlines a scalable architecture for ingesting a large, pre-sharded segmentation dataset into a cohesive Neuroglancer volume using TensorStore.

## 1. Problem Statement

We have a very large 3D segmentation volume that has been pre-processed into a collection of distinct data files, where each file corresponds to a single shard of the final volume (e.g., a 2048x2048x2048 voxel block). These data files, stored in a format like Apache Arrow in Google Cloud Storage (GCS), contain compressed segmentation data (`uint64` labels).

The goal is to design a system that can write these thousands of individual shard files into a single, queryable Neuroglancer precomputed sharded volume in GCS. The system must be highly parallel to handle massive scale (10,000+ shards) and robust enough to ensure data integrity.

## 2. Architectural Approach with TensorStore

The core of our approach is to leverage TensorStore to abstract away the complexities of the Neuroglancer sharded format. Instead of having each writer task understand how to create, modify, and write shard files, the tasks will operate on a virtual, global 3D array representing the entire volume.

### The Core Concept: Operate in Global Voxel Space

Each writer task will be responsible for a specific region of the volume. Its logic will be simple:

1.  Calculate the bounding box of its assigned region in the **global voxel coordinate system**.
2.  Read and decompress its assigned input data file into a NumPy array.
3.  Write that NumPy array to its calculated bounding box in the TensorStore volume.

TensorStore's Neuroglancer driver will handle all the backend complexity, including identifying the correct shard file in GCS, performing atomic read-modify-write operations with optimistic concurrency, and managing data compression. This makes the writer logic simple, scalable, and robust.

### Workflow Phases

The process is divided into two distinct phases:

1.  **Phase 1: Initialization (One-Time Setup)**: A single script is run once to create the `info` file in GCS. This file defines the metadata for the entire volume (dimensions, data type, sharding parameters, etc.), effectively creating the "canvas" for the parallel writers.

2.  **Phase 2: Parallel Data Ingestion (Writer Tasks)**: Thousands of independent, stateless writer tasks are launched. Each task is responsible for ingesting one of the pre-computed data files into the volume.

---

## 3. Implementation Details

### Phase 1: Volume Initialization Script

This script should be run once before any writer tasks are launched. It connects to GCS and writes the `info` file that defines the volume's structure.

```python
import tensorstore as ts
import numpy as np

# Define the full specification of your volume.
# This should be created ONCE and saved to GCS.
spec = {
    'driver': 'neuroglancer_precomputed',
    'kvstore': {
        'driver': 'gcs',
        'bucket': 'your-gcs-bucket',
        'path': 'path/to/your/volume/'
    },
    # The context specifies concurrency and cache limits.
    'context': {
        'cache_pool': {'total_bytes_limit': 100000000}, # 100MB
        'data_copy_concurrency': {'limit': 16}
    },
    # The scale_metadata defines the properties of a single resolution level.
    'scale_metadata': {
        'size': [94088, 78317, 134576],
        'resolution': [8, 8, 8],
        'encoding': 'compressed_segmentation',  # Standard for uint64 segmentation.
        'chunk_size': [64, 64, 64],
        'sharding': {
            '@type': 'neuroglancer_uint64_sharded_v1',
            'hash': 'identity',
            'preshift_bits': 9,
            'shard_bits': 19,
            'minishard_bits': 6,
            'minishard_index_encoding': 'gzip',
        }
    },
    # The multiscale_metadata provides top-level information.
    'multiscale_metadata': {
        'data_type': 'uint64',
        'num_channels': 1,
        'type': 'segmentation'
    },
    # These flags tell TensorStore to create the volume and overwrite if it exists.
    'create': True,
    'delete_existing': True,
}

# This command connects to GCS and writes the info file.
future = ts.open(spec)
future.result()

print("Neuroglancer info file created successfully in GCS.")
```

### Phase 2: Logic for a Single Parallel Writer Task

Each writer task (e.g., a Google Cloud Run instance) will execute the following logic. It is given a unique `SHARD_ID` that corresponds to an input data file.

```python
import tensorstore as ts
import numpy as np
import os

def get_voxel_bbox_for_shard_id(shard_id: int, shard_bits: int, preshift_bits: int):
    """
    Calculates the global voxel bounding box for a given shard ID.
    
    NOTE: This logic is critical and must correctly map a 1D shard ID
    to its corresponding 3D position in the global volume based on the
    sharding parameters. The implementation depends on how shard IDs are
    assigned to the 3D grid of shards.
    """
    # The total number of bits defining a shard's size along one dimension
    total_bits = preshift_bits + shard_bits
    
    # The size of a shard in voxels along one dimension (e.g., 2048)
    shard_dim_size = 1 << total_bits
    
    # Example decoding logic: Assumes a 3D shard grid of size (sx, sy, sz)
    # sx = total_volume_x // shard_dim_size
    # sy = total_volume_y // shard_dim_size
    # z_index = shard_id // (sx * sy)
    # y_index = (shard_id // sx) % sy
    # x_index = shard_id % sx
    
    # For this example, we'll use a simplified 1D layout for demonstration.
    x_index, y_index, z_index = shard_id, 0, 0
    
    origin = (
        x_index * shard_dim_size,
        y_index * shard_dim_size,
        z_index * shard_dim_size
    )
    shape = (shard_dim_size, shard_dim_size, shard_dim_size)
    
    print(f"Shard {shard_id} maps to origin: {origin}, shape: {shape}")
    
    # Return a tuple of slice objects for indexing
    return (
        slice(origin[0], origin[0] + shape[0]),
        slice(origin[1], origin[1] + shape[1]),
        slice(origin[2], origin[2] + shape[2]),
    )

# --- Main logic for one Writer Task ---

# 1. Identify the shard to process (e.g., from an environment variable)
shard_id_to_process = int(os.getenv("SHARD_ID", "0"))

# 2. Define the spec for opening the existing volume.
#    Note that `create` is false. All writers point to the same volume.
volume_spec = {
    'driver': 'neuroglancer_precomputed',
    'kvstore': {
        'driver': 'gcs',
        'bucket': 'your-gcs-bucket',
        'path': 'path/to/your/volume/'
    },
    'context': {
        'cache_pool': {'total_bytes_limit': 100000000},
        'data_copy_concurrency': {'limit': 16}
    },
    'open': True, # Use 'open' to connect to an existing volume
}

# 3. Open a handle to the global volume.
volume = ts.open(volume_spec).result()

# 4. Calculate the voxel bounding box for this task's shard.
#    These parameters MUST match the values from the info file.
SHARD_BITS = 19
PRESHIFT_BITS = 9
voxel_slices = get_voxel_bbox_for_shard_id(shard_id_to_process, SHARD_BITS, PRESHIFT_BITS)
shard_shape = tuple(s.stop - s.start for s in voxel_slices)

# 5. Read and decompress the source data into a NumPy array.
#    This is where logic to read the Arrow file from GCS would go.
#    The final array's shape MUST match the `shard_shape`.
print(f"Reading and preparing data of shape: {shard_shape}")
data_array = np.random.randint(0, 1000, size=shard_shape, dtype=np.uint64)

# 6. Write the data to the correct slice of the global volume.
#    This operation is asynchronous and returns a future.
print(f"Writing data for shard {shard_id_to_process}...")
write_future = volume[voxel_slices].write(data_array)

# 7. Block until the write is durably stored in GCS.
#    This is a crucial step to ensure the task doesn't exit prematurely.
#    .result() will raise an exception if the write failed.
write_future.result()

print(f"Successfully wrote data for shard {shard_id_to_process}.")
```

## 4. Key Benefits of this Architecture

-   **Massive Scalability**: The architecture is inherently parallel. You can scale to thousands of concurrent writers simply by launching more tasks.
-   **Simplicity of Logic**: The writer tasks are simple and stateless. They do not need to understand the Neuroglancer sharded format, locking, or concurrency management.
-   **Robustness and Data Integrity**: TensorStore handles atomic updates to the shard files in GCS, ensuring that concurrent writes do not corrupt the data. Its transactional guarantees provide a high degree of reliability.
-   **Efficiency**: By writing directly to the final GCS location, we avoid intermediate storage or complex file-moving operations.