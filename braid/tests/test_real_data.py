"""
Ground truth tests using real DVID compressed blocks from the DVID repository.

These tests verify that BRAID's Python decompressor produces bit-identical
output to Go's MakeLabelVolume(), using the same test data files from
dvid/test_data/.

Test data files (in tests/test_data/):
    fib19-64x64x64-sample1-block.dat.gz
        Gzipped raw DVID compressed block (18 labels, 64x64x64).
        This is the MarshalBinary() output of a Block created from sample1.

    fib19-64x64x64-sample1.dat.gz
    fib19-64x64x64-sample2.dat.gz
    cx-64x64x64-sample1.dat.gz
    cx-64x64x64-sample2.dat.gz
        Gzipped raw uint64 label volumes (64x64x64 = 2,097,152 bytes each).
        Flat little-endian uint64 arrays in z*64*64 + y*64 + x order.
        Used as ground truth for roundtrip verification.
"""

import gzip
from pathlib import Path

import numpy as np
import pytest
import zstandard as zstd

from braid.decompressor import DVIDDecompressor

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def load_raw_volume(filename: str) -> np.ndarray:
    """Load a gzipped raw uint64 label volume as a (64,64,64) ZYX array."""
    with gzip.open(TEST_DATA_DIR / filename, "rb") as f:
        data = f.read()
    assert len(data) == 64 * 64 * 64 * 8, f"Expected 2097152 bytes, got {len(data)}"
    return np.frombuffer(data, dtype="<u8").reshape((64, 64, 64))


def load_compressed_block(filename: str) -> bytes:
    """Load a gzipped DVID compressed block as raw bytes."""
    with gzip.open(TEST_DATA_DIR / filename, "rb") as f:
        return f.read()


@pytest.fixture
def decompressor():
    return DVIDDecompressor()


@pytest.fixture
def zstd_compressor():
    return zstd.ZstdCompressor()


# ---- Ground truth roundtrip: compressed block vs raw volume ----

class TestGroundTruthRoundtrip:
    """
    Verify Python decompression matches Go's MakeLabelVolume() by comparing
    the decompressed fib19-sample1 block against the raw ground truth volume.

    This mirrors the Go test in compressed_test.go:blockTest() which:
    1. Loads raw uint64 volume from .dat.gz
    2. Compresses with MakeBlock()
    3. Serializes with MarshalBinary()
    4. Deserializes with UnmarshalBinary()
    5. Decompresses with MakeLabelVolume()
    6. Asserts voxel-for-voxel equality with checkLabels()
    """

    def test_fib19_sample1_voxel_exact(self, decompressor, zstd_compressor):
        """Decompress fib19-sample1 block and compare voxel-for-voxel with ground truth."""
        ground_truth = load_raw_volume("fib19-64x64x64-sample1.dat.gz")
        compressed_block = load_compressed_block("fib19-64x64x64-sample1-block.dat.gz")

        # Wrap in zstd as the shard pipeline does
        zstd_compressed = zstd_compressor.compress(compressed_block)

        # Decompress with Python
        result = decompressor.decompress_block(zstd_compressed)

        assert result.shape == (64, 64, 64)
        assert result.dtype == np.uint64

        # Voxel-for-voxel comparison — the core assertion
        mismatches = np.where(result != ground_truth)
        num_mismatches = len(mismatches[0])
        if num_mismatches > 0:
            # Report first few mismatches for debugging
            for i in range(min(5, num_mismatches)):
                z, y, x = mismatches[0][i], mismatches[1][i], mismatches[2][i]
                pytest.fail(
                    f"Mismatch at ({x},{y},{z}): "
                    f"expected {ground_truth[z, y, x]}, got {result[z, y, x]}. "
                    f"Total mismatches: {num_mismatches} / {64**3}"
                )

    def test_fib19_sample1_label_set(self, decompressor, zstd_compressor):
        """Verify the decompressed block has the expected set of unique labels."""
        ground_truth = load_raw_volume("fib19-64x64x64-sample1.dat.gz")
        compressed_block = load_compressed_block("fib19-64x64x64-sample1-block.dat.gz")
        zstd_compressed = zstd_compressor.compress(compressed_block)

        result = decompressor.decompress_block(zstd_compressed)

        expected_labels = set(np.unique(ground_truth))
        actual_labels = set(np.unique(result))
        assert actual_labels == expected_labels, (
            f"Label set mismatch.\n"
            f"  Missing: {expected_labels - actual_labels}\n"
            f"  Extra: {actual_labels - expected_labels}"
        )

    def test_fib19_sample1_label_counts(self, decompressor, zstd_compressor):
        """Verify per-label voxel counts match ground truth exactly."""
        ground_truth = load_raw_volume("fib19-64x64x64-sample1.dat.gz")
        compressed_block = load_compressed_block("fib19-64x64x64-sample1-block.dat.gz")
        zstd_compressed = zstd_compressor.compress(compressed_block)

        result = decompressor.decompress_block(zstd_compressed)

        gt_labels, gt_counts = np.unique(ground_truth, return_counts=True)
        res_labels, res_counts = np.unique(result, return_counts=True)

        gt_hist = dict(zip(gt_labels, gt_counts))
        res_hist = dict(zip(res_labels, res_counts))

        assert gt_hist == res_hist


# ---- Raw volume properties ----

class TestRawVolumeProperties:
    """Verify basic properties of the raw test volumes."""

    @pytest.mark.parametrize("filename,min_labels", [
        ("fib19-64x64x64-sample1.dat.gz", 2),
        ("fib19-64x64x64-sample2.dat.gz", 2),
        ("cx-64x64x64-sample1.dat.gz", 2),
        ("cx-64x64x64-sample2.dat.gz", 2),
    ])
    def test_volume_shape_and_labels(self, filename, min_labels):
        """Each raw volume is 64^3 uint64 with multiple labels."""
        vol = load_raw_volume(filename)
        assert vol.shape == (64, 64, 64)
        assert vol.dtype == np.uint64
        unique = np.unique(vol)
        assert len(unique) >= min_labels, (
            f"{filename}: expected >= {min_labels} labels, got {len(unique)}"
        )


# ---- Compressed block properties ----

class TestCompressedBlockProperties:
    """Verify the DVID compressed block can be parsed and decompressed."""

    def test_fib19_block_header(self):
        """Verify DVID block header fields."""
        import struct
        block = load_compressed_block("fib19-64x64x64-sample1-block.dat.gz")
        gx, gy, gz, num_labels = struct.unpack("<IIII", block[:16])
        assert (gx, gy, gz) == (8, 8, 8), f"Expected (8,8,8) sub-blocks, got ({gx},{gy},{gz})"
        assert num_labels == 18, f"Expected 18 labels, got {num_labels}"

    def test_fib19_block_decompresses(self, decompressor, zstd_compressor):
        """Block decompresses to correct shape and dtype."""
        block = load_compressed_block("fib19-64x64x64-sample1-block.dat.gz")
        zstd_compressed = zstd_compressor.compress(block)

        result = decompressor.decompress_block(zstd_compressed)
        assert result.shape == (64, 64, 64)
        assert result.dtype == np.uint64
        assert len(np.unique(result)) > 1, "Expected multi-label block"

    def test_fib19_block_info(self, decompressor, zstd_compressor):
        """Block info extraction gives correct metadata."""
        block = load_compressed_block("fib19-64x64x64-sample1-block.dat.gz")
        zstd_compressed = zstd_compressor.compress(block)

        info = decompressor.get_block_info(zstd_compressed)
        assert info["type"] == "compressed"
        assert info["subblocks"] == (8, 8, 8)
        assert info["label_count"] == 18
        assert info["block_size"] == (64, 64, 64)


# ---- Spot-check specific voxels against ground truth ----

class TestSpotCheckVoxels:
    """
    Spot-check individual voxel values at specific coordinates.
    These tests catch off-by-one errors in sub-block indexing and
    bit-packing that a full-volume comparison might make hard to debug.
    """

    def test_corner_voxels(self, decompressor, zstd_compressor):
        """Check all 8 corner voxels of the 64^3 block."""
        ground_truth = load_raw_volume("fib19-64x64x64-sample1.dat.gz")
        block = load_compressed_block("fib19-64x64x64-sample1-block.dat.gz")
        result = decompressor.decompress_block(zstd_compressor.compress(block))

        corners = [
            (0, 0, 0), (63, 0, 0), (0, 63, 0), (0, 0, 63),
            (63, 63, 0), (63, 0, 63), (0, 63, 63), (63, 63, 63),
        ]
        for z, y, x in corners:
            assert result[z, y, x] == ground_truth[z, y, x], (
                f"Corner ({x},{y},{z}): expected {ground_truth[z,y,x]}, got {result[z,y,x]}"
            )

    def test_sub_block_boundaries(self, decompressor, zstd_compressor):
        """Check voxels at sub-block boundaries (multiples of 8)."""
        ground_truth = load_raw_volume("fib19-64x64x64-sample1.dat.gz")
        block = load_compressed_block("fib19-64x64x64-sample1-block.dat.gz")
        result = decompressor.decompress_block(zstd_compressor.compress(block))

        # Check every sub-block boundary voxel
        for sz in range(8):
            for sy in range(8):
                for sx in range(8):
                    z, y, x = sz * 8, sy * 8, sx * 8
                    assert result[z, y, x] == ground_truth[z, y, x], (
                        f"Sub-block boundary ({x},{y},{z}): "
                        f"expected {ground_truth[z,y,x]}, got {result[z,y,x]}"
                    )

    def test_sub_block_last_voxels(self, decompressor, zstd_compressor):
        """Check last voxel of each sub-block (catches off-by-one in bit packing)."""
        ground_truth = load_raw_volume("fib19-64x64x64-sample1.dat.gz")
        block = load_compressed_block("fib19-64x64x64-sample1-block.dat.gz")
        result = decompressor.decompress_block(zstd_compressor.compress(block))

        for sz in range(8):
            for sy in range(8):
                for sx in range(8):
                    z, y, x = sz * 8 + 7, sy * 8 + 7, sx * 8 + 7
                    assert result[z, y, x] == ground_truth[z, y, x], (
                        f"Sub-block last voxel ({x},{y},{z}): "
                        f"expected {ground_truth[z,y,x]}, got {result[z,y,x]}"
                    )
