#!/usr/bin/env python3
"""
Integration tests for braid library functionality.

Tests the full pipeline from Arrow/CSV files through decompression.
"""

import unittest
import tempfile
from pathlib import Path
import csv
import struct
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import zstandard as zstd

from braid import ShardReader, LabelType
from braid.exceptions import ChunkNotFoundError


class TestBraidIntegration(unittest.TestCase):
    """Integration tests for the full braid pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.zstd_compressor = zstd.ZstdCompressor()

    def _create_dvid_block(self, label: int) -> bytes:
        """Create a simple DVID solid block."""
        header = struct.pack('<IIII', 8, 8, 8, 1)
        label_data = struct.pack('<Q', label)
        return header + label_data

    def _create_test_shard(self):
        """Create test Arrow and CSV files."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        arrow_path = temp_path / "test_shard.arrow"
        csv_path = temp_path / "test_shard.csv"

        # Create DVID blocks
        dvid_block1 = self._create_dvid_block(42)
        dvid_block2 = self._create_dvid_block(99)

        # Compress with zstd
        compressed1 = self.zstd_compressor.compress(dvid_block1)
        compressed2 = self.zstd_compressor.compress(dvid_block2)

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
            'labels': [[1000, 1042], [2000, 2099]],  # Agglomerated labels
            'supervoxels': [[0, 42], [50, 99]],      # Supervoxel labels
            'dvid_compressed_block': [compressed1, compressed2],
            'uncompressed_size': [len(dvid_block1), len(dvid_block2)]
        }

        table = pa.table(data, schema=schema)

        # Write Arrow file
        with open(arrow_path, 'wb') as f:
            with ipc.new_file(f, table.schema) as writer:
                writer.write_table(table)

        # Write CSV index
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z', 'rec'])
            writer.writerow([0, 0, 0, 0])  # First chunk
            writer.writerow([1, 0, 0, 1])  # Second chunk

        return arrow_path, csv_path

    def test_full_pipeline_labels(self):
        """Test the full pipeline using agglomerated labels."""
        arrow_path, csv_path = self._create_test_shard()

        try:
            reader = ShardReader(arrow_path, csv_path)

            # Test chunk exists
            self.assertTrue(reader.has_chunk(0, 0, 0))
            self.assertTrue(reader.has_chunk(1, 0, 0))
            self.assertFalse(reader.has_chunk(2, 0, 0))

            # Test reading with agglomerated labels
            chunk_data = reader.read_chunk(0, 0, 0, label_type=LabelType.LABELS)

            self.assertEqual(chunk_data.shape, (64, 64, 64))
            self.assertEqual(chunk_data.dtype, np.uint64)
            # Should be mapped: supervoxel 42 -> index 1 -> agglo label 1042
            self.assertTrue(np.all(chunk_data == 1042))

            # Test second chunk
            chunk_data2 = reader.read_chunk(1, 0, 0, label_type=LabelType.LABELS)
            # Should be mapped: supervoxel 99 -> index 1 -> agglo label 2099
            self.assertTrue(np.all(chunk_data2 == 2099))

        finally:
            # Cleanup
            arrow_path.unlink()
            csv_path.unlink()
            arrow_path.parent.rmdir()

    def test_full_pipeline_supervoxels(self):
        """Test the full pipeline using supervoxel labels."""
        arrow_path, csv_path = self._create_test_shard()

        try:
            reader = ShardReader(arrow_path, csv_path)

            # Test reading with supervoxel labels (no mapping)
            chunk_data = reader.read_chunk(0, 0, 0, label_type=LabelType.SUPERVOXELS)

            self.assertEqual(chunk_data.shape, (64, 64, 64))
            self.assertEqual(chunk_data.dtype, np.uint64)
            # Should be unmapped: supervoxel 42 directly
            self.assertTrue(np.all(chunk_data == 42))

        finally:
            # Cleanup
            arrow_path.unlink()
            csv_path.unlink()
            arrow_path.parent.rmdir()

    def test_chunk_not_found(self):
        """Test handling of missing chunks."""
        arrow_path, csv_path = self._create_test_shard()

        try:
            reader = ShardReader(arrow_path, csv_path)

            with self.assertRaises(ChunkNotFoundError):
                reader.read_chunk(99, 99, 99)

        finally:
            # Cleanup
            arrow_path.unlink()
            csv_path.unlink()
            arrow_path.parent.rmdir()

    def test_chunk_info(self):
        """Test chunk metadata extraction."""
        arrow_path, csv_path = self._create_test_shard()

        try:
            reader = ShardReader(arrow_path, csv_path)

            info = reader.get_chunk_info(0, 0, 0)

            self.assertEqual(info['coordinates'], (0, 0, 0))
            self.assertEqual(info['record_index'], 0)
            self.assertEqual(info['chunk_x'], 0)
            self.assertEqual(info['chunk_y'], 0)
            self.assertEqual(info['chunk_z'], 0)
            self.assertGreater(info['compressed_size'], 0)

        finally:
            # Cleanup
            arrow_path.unlink()
            csv_path.unlink()
            arrow_path.parent.rmdir()


if __name__ == '__main__':
    unittest.main()