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


def _write_arrow_streaming(path, table, batch_size=1):
    """Write an Arrow IPC streaming file, grouping rows into batches.

    Returns (schema_size, batches) where batches is a list of
    (offset, size, num_rows) for each record batch.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from scripts.compute_offsets import scan_record_offsets

    with open(path, "wb") as f:
        writer = ipc.new_stream(f, table.schema)
        for start in range(0, len(table), batch_size):
            end = min(start + batch_size, len(table))
            batch = table.slice(start, end - start).to_batches()[0]
            writer.write_batch(batch)
        writer.close()

    # Scan the written file to find actual byte offsets (the streaming
    # writer buffers the schema, so we can't capture positions during write).
    data = Path(path).read_bytes()
    schema_size, raw_offsets = scan_record_offsets(data)

    batches = []
    total_rows = len(table)
    for i, (_, off, sz) in enumerate(raw_offsets):
        start = i * batch_size
        nrows = min(batch_size, total_rows - start)
        batches.append((off, sz, nrows))

    return schema_size, batches


def _write_new_csv(csv_path, table, schema_size, batches, batch_size=1):
    """Write a new-format CSV index with offset/size/batch_idx."""
    xs = table.column("chunk_x").to_pylist()
    ys = table.column("chunk_y").to_pylist()
    zs = table.column("chunk_z").to_pylist()

    with open(csv_path, "w", newline="") as f:
        f.write(f"# schema_size={schema_size}\n")
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "offset", "size", "batch_idx"])
        for i in range(len(table)):
            batch_num = i // batch_size
            offset, size, _ = batches[batch_num]
            batch_idx = i % batch_size
            writer.writerow([xs[i], ys[i], zs[i], offset, size, batch_idx])


@pytest.fixture
def temp_shard_files(sample_arrow_data):
    """Create temporary Arrow and CSV files (batch_size=1, new format)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        arrow_path = temp_path / "test_shard.arrow"
        csv_path = temp_path / "test_shard.csv"

        schema_size, batches = _write_arrow_streaming(
            arrow_path, sample_arrow_data, batch_size=1)
        _write_new_csv(csv_path, sample_arrow_data, schema_size, batches,
                       batch_size=1)

        yield arrow_path, csv_path


@pytest.fixture
def temp_shard_files_batched(sample_arrow_data):
    """Create temporary Arrow and CSV files with batch_size=2.

    Both chunks are in a single record batch, distinguished by batch_idx.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        arrow_path = temp_path / "test_shard.arrow"
        csv_path = temp_path / "test_shard.csv"

        schema_size, batches = _write_arrow_streaming(
            arrow_path, sample_arrow_data, batch_size=2)
        _write_new_csv(csv_path, sample_arrow_data, schema_size, batches,
                       batch_size=2)

        yield arrow_path, csv_path


@pytest.fixture
def temp_shard_files_old_csv(sample_arrow_data):
    """Create temporary Arrow and CSV files with old-format CSV (x,y,z,rec)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        arrow_path = temp_path / "test_shard.arrow"
        csv_path = temp_path / "test_shard.csv"

        # Write Arrow file (streaming, batch_size=1)
        _write_arrow_streaming(arrow_path, sample_arrow_data, batch_size=1)

        # Write old-format CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z", "rec"])
            writer.writerow([0, 0, 0, 0])
            writer.writerow([1, 0, 0, 1])

        yield arrow_path, csv_path