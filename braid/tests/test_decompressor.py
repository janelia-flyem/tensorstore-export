#!/usr/bin/env python3
"""
Test suite for the DVID block decompressor in braid.

Tests both the DVID segmentation compression and zstd compression layers.
"""

import struct
import unittest
from pathlib import Path
from typing import List, Optional
import numpy as np
import zstandard as zstd

from braid.decompressor import DVIDDecompressor, bits_for, get_packed_value, DecompressionError


class TestHelperFunctions(unittest.TestCase):
    """Test the helper functions used in DVID decompression."""

    def test_bits_for(self):
        """Test the bits_for function."""
        self.assertEqual(bits_for(0), 0)
        self.assertEqual(bits_for(1), 0)
        self.assertEqual(bits_for(2), 1)
        self.assertEqual(bits_for(4), 2)
        self.assertEqual(bits_for(8), 3)
        self.assertEqual(bits_for(16), 4)

    def test_get_packed_value(self):
        """Test the get_packed_value function."""
        data = bytes([0b11010011, 0b10110101])  # Two test bytes

        # Get 3 bits starting at bit position 0
        value = get_packed_value(data, 0, 3)
        self.assertEqual(value, 0b110)  # First 3 bits of first byte

        # Get 4 bits starting at bit position 2
        value = get_packed_value(data, 2, 4)
        self.assertEqual(value, 0b0100)  # Bits 2-5 of first byte

        # Test zero bits
        value = get_packed_value(data, 0, 0)
        self.assertEqual(value, 0)


class TestDVIDDecompressor(unittest.TestCase):
    """Test the DVID block decompressor."""

    def setUp(self):
        """Set up test fixtures."""
        self.decompressor = DVIDDecompressor()
        self.zstd_compressor = zstd.ZstdCompressor()

    def _create_solid_block(self, label: int) -> bytes:
        """Helper to create a DVID solid block."""
        header = struct.pack('<IIII', 8, 8, 8, 1)  # gx=8, gy=8, gz=8, numLabels=1
        label_data = struct.pack('<Q', label)
        return header + label_data

    def _create_zstd_compressed_block(self, dvid_data: bytes) -> bytes:
        """Helper to apply zstd compression to DVID data."""
        return self.zstd_compressor.compress(dvid_data)

    def test_solid_block_decompression(self):
        """Test decompression of a solid block."""
        # Create DVID block and compress with zstd
        dvid_block = self._create_solid_block(42)
        compressed_block = self._create_zstd_compressed_block(dvid_block)

        result = self.decompressor.decompress_block(compressed_block)

        self.assertEqual(result.shape, (64, 64, 64))
        self.assertEqual(result.dtype, np.uint64)
        self.assertTrue(np.all(result == 42))

    def test_label_mapping(self):
        """Test label mapping from supervoxels to agglomerated labels."""
        dvid_block = self._create_solid_block(42)
        compressed_block = self._create_zstd_compressed_block(dvid_block)

        supervoxels = [0, 42, 100, 200]
        agglo_labels = [1000, 1042, 1100, 1200]

        # Test without mapping
        result_no_mapping = self.decompressor.decompress_block(compressed_block)
        self.assertTrue(np.all(result_no_mapping == 42))

        # Test with mapping
        result_mapped = self.decompressor.decompress_block(
            compressed_block,
            agglo_labels=agglo_labels,
            supervoxels=supervoxels
        )
        self.assertTrue(np.all(result_mapped == 1042))  # 42 -> index 1 -> 1042

    def test_empty_block(self):
        """Test handling of empty blocks."""
        result = self.decompressor.decompress_block(b'')
        self.assertEqual(result.shape, (64, 64, 64))
        self.assertTrue(np.all(result == 0))

    def test_block_info_extraction(self):
        """Test block metadata extraction."""
        dvid_block = self._create_solid_block(42)
        compressed_block = self._create_zstd_compressed_block(dvid_block)

        info = self.decompressor.get_block_info(compressed_block)

        self.assertEqual(info['type'], 'solid')
        self.assertEqual(info['subblocks'], (8, 8, 8))
        self.assertEqual(info['label_count'], 1)
        self.assertEqual(info['block_size'], (64, 64, 64))
        self.assertIn('zstd_compressed_size', info)
        self.assertIn('dvid_uncompressed_size', info)

    def test_compression_layers(self):
        """Test that both compression layers work correctly."""
        dvid_block = self._create_solid_block(123)

        # Test sizes
        dvid_size = len(dvid_block)
        compressed_size = len(self._create_zstd_compressed_block(dvid_block))

        # For small blocks, zstd might add overhead, but for larger blocks it should compress
        self.assertGreater(dvid_size, 0)
        self.assertGreater(compressed_size, 0)

        # Test round-trip
        compressed_block = self._create_zstd_compressed_block(dvid_block)
        result = self.decompressor.decompress_block(compressed_block)
        self.assertTrue(np.all(result == 123))

    def test_invalid_data_handling(self):
        """Test handling of invalid data."""
        # Test invalid zstd data
        with self.assertRaises(DecompressionError):
            self.decompressor.decompress_block(b'invalid_zstd_data')

        # Test valid zstd but invalid DVID data
        invalid_dvid = b'invalid_dvid_data'
        invalid_compressed = self.zstd_compressor.compress(invalid_dvid)

        with self.assertRaises(DecompressionError):
            self.decompressor.decompress_block(invalid_compressed)

    def test_different_block_shapes(self):
        """Test that block shape validation works."""
        # Create a block with wrong dimensions
        wrong_header = struct.pack('<IIII', 4, 4, 4, 1)  # 32x32x32 instead of 64x64x64
        label_data = struct.pack('<Q', 42)
        wrong_block = self._create_zstd_compressed_block(wrong_header + label_data)

        with self.assertRaises(DecompressionError):
            self.decompressor.decompress_block(wrong_block)


class TestRealDataIntegration(unittest.TestCase):
    """Test with real DVID data from test_data/ directory."""

    TEST_DATA_DIR = Path(__file__).parent / "test_data"

    def setUp(self):
        """Set up test fixtures."""
        self.decompressor = DVIDDecompressor()
        self.zstd_compressor = zstd.ZstdCompressor()

    def test_real_dvid_data(self):
        """Test decompression of real DVID compressed block."""
        import gzip
        block_path = self.TEST_DATA_DIR / 'fib19-64x64x64-sample1-block.dat.gz'
        if not block_path.exists():
            self.skipTest("Real DVID test data not available")

        with gzip.open(block_path, 'rb') as f:
            real_dvid_data = f.read()

        # Apply zstd compression to simulate shard format
        zstd_compressed = self.zstd_compressor.compress(real_dvid_data)

        # Test decompression
        result = self.decompressor.decompress_block(zstd_compressed)

        self.assertEqual(result.shape, (64, 64, 64))
        self.assertEqual(result.dtype, np.uint64)

        # Should have multiple unique labels
        unique_labels = np.unique(result)
        self.assertGreater(len(unique_labels), 1)

        # Test compression efficiency
        compression_ratio = len(real_dvid_data) / len(zstd_compressed)
        self.assertGreater(compression_ratio, 1.0)


if __name__ == '__main__':
    unittest.main()