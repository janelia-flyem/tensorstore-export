"""
Tests for BRAID Shard Reader.
"""

import pytest
import tempfile
import csv
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from braid import ShardReader, LabelType
from braid.exceptions import (
    BraidError, ChunkNotFoundError, InvalidShardFormatError
)


@pytest.fixture
def sample_shard_files():
    """Create sample Arrow and CSV files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        arrow_path = temp_path / "test_shard.arrow"
        csv_path = temp_path / "test_shard.csv"
        
        # Create sample Arrow data
        schema = pa.schema([
            pa.field('chunk_x', pa.int32(), nullable=False),
            pa.field('chunk_y', pa.int32(), nullable=False),
            pa.field('chunk_z', pa.int32(), nullable=False),
            pa.field('labels', pa.list_(pa.uint64()), nullable=False),
            pa.field('supervoxels', pa.list_(pa.uint64()), nullable=False),
            pa.field('dvid_compressed_block', pa.binary(), nullable=False),
            pa.field('uncompressed_size', pa.uint32(), nullable=False)
        ])
        
        # Sample data for 3 chunks
        data = {
            'chunk_x': [0, 64, 128],
            'chunk_y': [0, 0, 64], 
            'chunk_z': [0, 0, 0],
            'labels': [[1, 2, 3], [4, 5], [6]],
            'supervoxels': [[100, 200, 300], [400, 500], [600]],
            'dvid_compressed_block': [b'\x00' * 100, b'\x01' * 50, b'\x02' * 25],
            'uncompressed_size': [1000, 800, 600]
        }
        
        table = pa.table(data, schema=schema)
        
        # Write Arrow file
        with open(arrow_path, 'wb') as f:
            with ipc.new_file(f, schema) as writer:
                writer.write_table(table)
        
        # Write CSV index
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z', 'rec'])
            writer.writerow([0, 0, 0, 0])
            writer.writerow([64, 0, 0, 1])
            writer.writerow([128, 64, 0, 2])
        
        yield arrow_path, csv_path


def test_reader_initialization(sample_shard_files):
    """Test reader initialization with valid files."""
    arrow_path, csv_path = sample_shard_files
    
    reader = ShardReader(arrow_path, csv_path)
    
    assert reader.chunk_count == 3
    assert len(reader.available_chunks) == 3
    assert (0, 0, 0) in reader.available_chunks
    assert (64, 0, 0) in reader.available_chunks
    assert (128, 64, 0) in reader.available_chunks


def test_reader_initialization_missing_files():
    """Test reader initialization with missing files."""
    with pytest.raises(BraidError, match="Arrow file not found"):
        ShardReader("nonexistent.arrow", "nonexistent.csv")


def test_has_chunk(sample_shard_files):
    """Test chunk existence checking."""
    arrow_path, csv_path = sample_shard_files
    reader = ShardReader(arrow_path, csv_path)
    
    assert reader.has_chunk(0, 0, 0) == True
    assert reader.has_chunk(64, 0, 0) == True
    assert reader.has_chunk(999, 999, 999) == False


def test_get_chunk_info(sample_shard_files):
    """Test getting chunk metadata."""
    arrow_path, csv_path = sample_shard_files
    reader = ShardReader(arrow_path, csv_path)
    
    info = reader.get_chunk_info(0, 0, 0)
    
    assert info['coordinates'] == (0, 0, 0)
    assert info['record_index'] == 0
    assert info['chunk_x'] == 0
    assert info['chunk_y'] == 0
    assert info['chunk_z'] == 0
    assert info['labels_count'] == 3
    assert info['supervoxels_count'] == 3
    assert info['compressed_size'] == 100


def test_get_chunk_info_not_found(sample_shard_files):
    """Test getting info for non-existent chunk."""
    arrow_path, csv_path = sample_shard_files
    reader = ShardReader(arrow_path, csv_path)
    
    with pytest.raises(ChunkNotFoundError):
        reader.get_chunk_info(999, 999, 999)


def test_read_chunk_raw(sample_shard_files):
    """Test reading raw chunk data."""
    arrow_path, csv_path = sample_shard_files
    reader = ShardReader(arrow_path, csv_path)
    
    raw_data = reader.read_chunk_raw(0, 0, 0)
    
    assert raw_data['coordinates'] == (0, 0, 0)
    assert raw_data['labels'] == [1, 2, 3]
    assert raw_data['supervoxels'] == [100, 200, 300]
    assert len(raw_data['compressed_data']) == 100
    assert raw_data['uncompressed_size'] == 1000


def test_read_chunk_with_labels(sample_shard_files):
    """Test reading chunk with agglomerated labels."""
    arrow_path, csv_path = sample_shard_files
    reader = ShardReader(arrow_path, csv_path)

    # Mock the decompressor to return predictable data.
    # Signature must match DVIDDecompressor.decompress_block kwargs.
    def mock_decompress(compressed_data, agglo_labels=None, supervoxels=None, block_shape=(64,64,64)):
        # Return array filled with first agglomerated label
        return np.full(block_shape, agglo_labels[0], dtype=np.uint64)

    reader._decompressor.decompress_block = mock_decompress

    chunk = reader.read_chunk(0, 0, 0, label_type=LabelType.LABELS)

    assert chunk.shape == (64, 64, 64)
    assert chunk.dtype == np.uint64
    assert np.all(chunk == 1)  # Should be filled with first label


def test_read_chunk_with_supervoxels(sample_shard_files):
    """Test reading chunk with supervoxel labels."""
    arrow_path, csv_path = sample_shard_files
    reader = ShardReader(arrow_path, csv_path)

    # Mock the decompressor. For SUPERVOXELS, reader passes agglo_labels=supervoxels.
    def mock_decompress(compressed_data, agglo_labels=None, supervoxels=None, block_shape=(64,64,64)):
        return np.full(block_shape, agglo_labels[0], dtype=np.uint64)

    reader._decompressor.decompress_block = mock_decompress

    chunk = reader.read_chunk(0, 0, 0, label_type=LabelType.SUPERVOXELS)

    assert chunk.shape == (64, 64, 64)
    assert chunk.dtype == np.uint64
    assert np.all(chunk == 100)  # Should be filled with first supervoxel


def test_read_chunk_not_found(sample_shard_files):
    """Test reading non-existent chunk."""
    arrow_path, csv_path = sample_shard_files
    reader = ShardReader(arrow_path, csv_path)
    
    with pytest.raises(ChunkNotFoundError):
        reader.read_chunk(999, 999, 999)


def test_invalid_csv_format():
    """Test handling of invalid CSV format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        arrow_path = temp_path / "test.arrow"
        csv_path = temp_path / "test.csv"
        
        # Create minimal valid Arrow file
        schema = ShardReader.EXPECTED_SCHEMA
        data = {field.name: [] for field in schema}
        table = pa.table(data, schema=schema)
        
        with open(arrow_path, 'wb') as f:
            with ipc.new_file(f, schema) as writer:
                writer.write_table(table)
        
        # Create invalid CSV (missing required columns)
        with open(csv_path, 'w') as f:
            f.write("invalid,headers\n1,2\n")
        
        with pytest.raises(InvalidShardFormatError, match="missing required columns"):
            ShardReader(arrow_path, csv_path)


def test_reader_string_representation(sample_shard_files):
    """Test string representation of reader."""
    arrow_path, csv_path = sample_shard_files
    reader = ShardReader(arrow_path, csv_path)
    
    repr_str = repr(reader)
    assert "ShardReader" in repr_str
    assert str(arrow_path) in repr_str
    assert str(csv_path) in repr_str
    assert "chunks=3" in repr_str


def test_reader_length(sample_shard_files):
    """Test len() function on reader."""
    arrow_path, csv_path = sample_shard_files
    reader = ShardReader(arrow_path, csv_path)
    
    assert len(reader) == 3