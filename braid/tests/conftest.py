"""
Pytest configuration and shared fixtures for braid tests.
"""

import csv
import pytest
import tempfile
import struct
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import zstandard as zstd


@pytest.fixture
def zstd_compressor():
    """Provide a zstd compressor for tests."""
    return zstd.ZstdCompressor()


@pytest.fixture
def solid_dvid_block():
    """Create a simple solid DVID block for testing."""
    def _create_block(label_value: int = 42) -> bytes:
        header = struct.pack('<IIII', 8, 8, 8, 1)  # gx=8, gy=8, gz=8, numLabels=1
        label_data = struct.pack('<Q', label_value)
        return header + label_data
    return _create_block


@pytest.fixture
def sample_arrow_data(solid_dvid_block, zstd_compressor):
    """Create sample Arrow data with compressed DVID blocks."""
    # Create test DVID blocks
    dvid_block1 = solid_dvid_block(42)
    dvid_block2 = solid_dvid_block(99)

    # Compress with zstd
    compressed1 = zstd_compressor.compress(dvid_block1)
    compressed2 = zstd_compressor.compress(dvid_block2)

    # Create Arrow table
    schema = pa.schema([
        pa.field('chunk_x', pa.int32(), nullable=False),
        pa.field('chunk_y', pa.int32(), nullable=False),
        pa.field('chunk_z', pa.int32(), nullable=False),
        pa.field('labels', pa.list_(pa.uint64()), nullable=False),
        pa.field('supervoxels', pa.list_(pa.uint64()), nullable=False),
        pa.field('dvid_compressed_block', pa.binary(), nullable=False),
        pa.field('uncompressed_size', pa.uint32(), nullable=False)
    ])

    data = {
        'chunk_x': [0, 1],
        'chunk_y': [0, 0],
        'chunk_z': [0, 0],
        'labels': [[1000, 1042], [2000, 2099]],
        'supervoxels': [[0, 42], [50, 99]],
        'dvid_compressed_block': [compressed1, compressed2],
        'uncompressed_size': [len(dvid_block1), len(dvid_block2)]
    }

    return pa.table(data, schema=schema)


@pytest.fixture
def two_label_dvid_block():
    """Create a two-label DVID block split along Z axis.

    Bottom half (z < 32) gets label_a, top half (z >= 32) gets label_b.
    Each 8x8x8 sub-block is solid (single label).
    """
    def _create_block(label_a: int = 10, label_b: int = 20) -> bytes:
        gx, gy, gz = 8, 8, 8
        header = struct.pack('<IIII', gx, gy, gz, 2)
        labels_data = struct.pack('<QQ', label_a, label_b)
        num_sub_blocks = gx * gy * gz
        num_sb_labels = struct.pack('<' + 'H' * num_sub_blocks, *([1] * num_sub_blocks))
        sb_indices = []
        for sz in range(gz):
            for sy in range(gy):
                for sx in range(gx):
                    sb_indices.append(0 if sz < gz // 2 else 1)
        sb_indices_data = struct.pack('<' + 'I' * num_sub_blocks, *sb_indices)
        return header + labels_data + num_sb_labels + sb_indices_data
    return _create_block


@pytest.fixture
def temp_shard_files(sample_arrow_data):
    """Create temporary Arrow and CSV files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        arrow_path = temp_path / "test_shard.arrow"
        csv_path = temp_path / "test_shard.csv"

        # Write Arrow file
        with open(arrow_path, 'wb') as f:
            with ipc.new_file(f, sample_arrow_data.schema) as writer:
                writer.write_table(sample_arrow_data)

        # Write CSV index
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z', 'rec'])
            writer.writerow([0, 0, 0, 0])  # First chunk
            writer.writerow([1, 0, 0, 1])  # Second chunk

        yield arrow_path, csv_path