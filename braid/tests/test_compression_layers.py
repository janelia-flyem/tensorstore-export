#!/usr/bin/env python3
"""
Test the two-layer compression architecture in braid.

This module specifically tests the interaction between:
1. DVID segmentation compression (inner layer)
2. zstd compression (outer layer)
"""

import struct
import unittest
import numpy as np
import zstandard as zstd

from braid.decompressor import DVIDDecompressor


class TestCompressionLayers(unittest.TestCase):
    """Test the two-layer compression architecture."""

    def setUp(self):
        """Set up test fixtures."""
        self.decompressor = DVIDDecompressor()
        self.zstd_compressor = zstd.ZstdCompressor()

    def test_compression_layer_separation(self):
        """Test that we can clearly separate the two compression layers."""
        # Create DVID compressed block (inner layer)
        dvid_header = struct.pack('<IIII', 8, 8, 8, 1)
        dvid_label = struct.pack('<Q', 42)
        dvid_block = dvid_header + dvid_label

        # Apply zstd compression (outer layer)
        zstd_compressed = self.zstd_compressor.compress(dvid_block)

        # Verify sizes
        self.assertEqual(len(dvid_block), 24)  # 4*4 + 8 = 24 bytes
        self.assertGreater(len(zstd_compressed), 0)

        # Test decompression
        result = self.decompressor.decompress_block(zstd_compressed)
        self.assertEqual(result.shape, (64, 64, 64))
        self.assertTrue(np.all(result == 42))

    def test_compression_efficiency(self):
        """Test compression efficiency with different data patterns."""
        test_cases = [
            (42, "solid_block"),
            (0, "zero_block"),
            (999999999, "large_value"),
        ]

        for label_value, description in test_cases:
            with self.subTest(description=description):
                # Create DVID block
                dvid_header = struct.pack('<IIII', 8, 8, 8, 1)
                dvid_label = struct.pack('<Q', label_value)
                dvid_block = dvid_header + dvid_label

                # Apply zstd compression
                zstd_compressed = self.zstd_compressor.compress(dvid_block)

                # Test decompression
                result = self.decompressor.decompress_block(zstd_compressed)
                self.assertTrue(np.all(result == label_value))

    def test_real_data_compression_efficiency(self):
        """Test compression efficiency with real DVID data."""
        try:
            import gzip
            # Load real DVID data
            with gzip.open('../research/fib19-64x64x64-sample1-block.dat.gz', 'rb') as f:
                real_dvid_data = f.read()

            # Apply zstd compression
            zstd_compressed = self.zstd_compressor.compress(real_dvid_data)

            # Calculate compression metrics
            dvid_size = len(real_dvid_data)
            zstd_size = len(zstd_compressed)
            compression_ratio = dvid_size / zstd_size

            # Verify compression is beneficial
            self.assertGreater(compression_ratio, 2.0)  # Should get at least 2x compression

            # Test decompression works
            result = self.decompressor.decompress_block(zstd_compressed)
            self.assertEqual(result.shape, (64, 64, 64))

            print(f"\nReal data compression results:")
            print(f"  DVID size: {dvid_size:,} bytes")
            print(f"  zstd size: {zstd_size:,} bytes")
            print(f"  Compression ratio: {compression_ratio:.2f}x")

        except (FileNotFoundError, ImportError):
            self.skipTest("Real DVID test data not available")

    def test_label_mapping_through_layers(self):
        """Test that label mapping works correctly through both compression layers."""
        # Create DVID block with specific label
        test_label = 123
        dvid_header = struct.pack('<IIII', 8, 8, 8, 1)
        dvid_label = struct.pack('<Q', test_label)
        dvid_block = dvid_header + dvid_label

        # Apply zstd compression
        zstd_compressed = self.zstd_compressor.compress(dvid_block)

        # Set up label mapping
        supervoxels = [0, test_label, 456, 789]
        agglo_labels = [1000, 1123, 1456, 1789]

        # Test mapping
        result = self.decompressor.decompress_block(
            zstd_compressed,
            agglo_labels=agglo_labels,
            supervoxels=supervoxels
        )

        # Should map: 123 -> index 1 in supervoxels -> agglo_labels[1] = 1123
        self.assertTrue(np.all(result == 1123))

    def test_error_handling_in_layers(self):
        """Test error handling for each compression layer."""
        from braid.decompressor import DecompressionError

        # Test zstd layer error
        with self.assertRaises(DecompressionError) as cm:
            self.decompressor.decompress_block(b'invalid_zstd_data')
        self.assertIn("zstd", str(cm.exception).lower())

        # Test DVID layer error (valid zstd, invalid DVID)
        invalid_dvid = b'invalid_dvid_header_data'
        invalid_zstd_compressed = self.zstd_compressor.compress(invalid_dvid)

        with self.assertRaises(DecompressionError):
            self.decompressor.decompress_block(invalid_zstd_compressed)

    def test_block_info_through_layers(self):
        """Test that block info extraction works through both layers."""
        # Create test block
        dvid_header = struct.pack('<IIII', 8, 8, 8, 2)  # Multi-label block
        dvid_labels = struct.pack('<QQ', 100, 200)
        dvid_block = dvid_header + dvid_labels

        # Apply zstd compression
        zstd_compressed = self.zstd_compressor.compress(dvid_block)

        # Extract info
        info = self.decompressor.get_block_info(zstd_compressed)

        # Verify info includes both layers
        self.assertEqual(info['type'], 'compressed')  # Multi-label
        self.assertEqual(info['label_count'], 2)
        self.assertIn('zstd_compressed_size', info)
        self.assertIn('dvid_uncompressed_size', info)
        self.assertEqual(info['zstd_compressed_size'], len(zstd_compressed))
        self.assertEqual(info['dvid_uncompressed_size'], len(dvid_block))


if __name__ == '__main__':
    unittest.main()