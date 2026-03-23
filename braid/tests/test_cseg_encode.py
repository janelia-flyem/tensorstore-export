"""
Tests for the neuroglancer compressed_segmentation encoder.

Layer 1: Byte-exact test vectors from TensorStore's C++ test suite.
Layer 2: TensorStore roundtrip (encode with our C ext, decode with TensorStore).
Layer 3: Fused DVID-to-cseg path with real test data.
"""

import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from braid.cseg_encoder import CSEGEncoder

TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture
def encoder():
    return CSEGEncoder()


def u32_le_bytes(*words):
    """Pack uint32 values as little-endian bytes."""
    return b"".join(struct.pack("<I", w) for w in words)


def u64_le_bytes(*words):
    """Pack uint64 values as little-endian bytes."""
    return b"".join(struct.pack("<Q", w) for w in words)


# ---- Layer 1: Byte-exact TensorStore test vectors ----


class TestByteExactVectors:
    """Verify byte-exact match against TensorStore's EncodeChannels tests.

    Test vectors from:
    tensorstore/internal/compression/neuroglancer_compressed_segmentation_test.cc
    """

    def test_1channel_1block_2values(self, encoder):
        """EncodeChannelsTest::Basic1Channel1Block — 1 channel, 1 block, 2 labels.

        Input: [4, 0, 4, 0] as shape (1, 2, 2) in ZYX order.
        Block size (1, 2, 2).
        """
        data = np.array([[[4, 0], [4, 0]]], dtype=np.uint64)  # shape (1,2,2)
        result = encoder.encode_chunk(data, block_size=(1, 2, 2))

        expected = (
            u32_le_bytes(1)                  # channel offset = 1
            + u32_le_bytes(3 | (1 << 24))    # block header word0: table=3, bits=1
            + u32_le_bytes(2)                # block header word1: enc_val_off=2
            + u32_le_bytes(0b0101)           # encoded values: [4,0,4,0] → [1,0,1,0]
            + u64_le_bytes(0, 4)             # table: [0, 4] sorted
        )
        assert result == expected, (
            f"Mismatch.\n  Got:      {result.hex()}\n  Expected: {expected.hex()}"
        )

    def test_solid_block(self, encoder):
        """All voxels same label → encoding_bits=0, single table entry."""
        data = np.full((2, 2, 2), 42, dtype=np.uint64)
        result = encoder.encode_chunk(data, block_size=(2, 2, 2))

        # channel_offset=1, block header: table_off=2, bits=0, enc_off=2
        # (enc_off == table_off since there are 0 encoded words)
        expected = (
            u32_le_bytes(1)                  # channel offset
            + u32_le_bytes(2 | (0 << 24))    # table=2, bits=0
            + u32_le_bytes(2)                # enc_val_off=2
            + u64_le_bytes(42)               # table: [42]
        )
        assert result == expected

    def test_3values_2bit_encoding(self, encoder):
        """Three unique labels → 2-bit encoding."""
        # shape (1, 2, 2): values [4, 3, 5, 4] in ZYX order
        data = np.array([[[4, 3], [5, 4]]], dtype=np.uint64)
        result = encoder.encode_chunk(data, block_size=(1, 2, 2))

        # Sorted labels: [3, 4, 5] → indices: 4→1, 3→0, 5→2, 4→1
        # Offsets: x=0,y=0: val=4 idx=1 off=0; x=1,y=0: val=3 idx=0 off=1
        #          x=0,y=1: val=5 idx=2 off=2; x=1,y=1: val=4 idx=1 off=3
        # Packed 2-bit: (1<<0)|(0<<2)|(2<<4)|(1<<6) = 0b01100001
        expected = (
            u32_le_bytes(1)                  # channel offset
            + u32_le_bytes(3 | (2 << 24))    # table=3, bits=2
            + u32_le_bytes(2)                # enc_val_off=2
            + u32_le_bytes(0b01100001)       # encoded values
            + u64_le_bytes(3, 4, 5)          # table: [3, 4, 5]
        )
        assert result == expected

    def test_table_dedup(self, encoder):
        """Multiple blocks with same labels should share a value table."""
        # 4 blocks of (1,2,2) stacked along Z: shape (4,2,2)
        # Block 0 and block 2 both have labels {1,3} → should share table
        data = np.array([
            [[3, 3], [3, 3]],  # block 0: solid {3}
            [[3, 3], [3, 3]],  # block 1: solid {3}
            [[3, 3], [3, 3]],  # block 2: solid {3}
            [[3, 3], [3, 3]],  # block 3: solid {3}
        ], dtype=np.uint64)
        result = encoder.encode_chunk(data, block_size=(1, 2, 2))

        # All 4 blocks should point to the same table offset.
        # Parse block headers (after 4-byte channel header):
        headers = []
        for i in range(4):
            off = 4 + i * 8
            w0 = struct.unpack_from("<I", result, off)[0]
            w1 = struct.unpack_from("<I", result, off + 4)[0]
            tbl = w0 & 0xFFFFFF
            bits = (w0 >> 24) & 0xFF
            headers.append((tbl, bits, w1))

        # All should have bits=0 and same table offset
        assert all(h[1] == 0 for h in headers), "All blocks should be 0-bit"
        assert len(set(h[0] for h in headers)) == 1, "All blocks should share one table"


# ---- Layer 2: TensorStore roundtrip ----


class TestTensorStoreRoundtrip:
    """Encode with our C extension, decode with TensorStore, verify voxels."""

    @staticmethod
    def _decode_with_tensorstore(encoded_bytes, volume_shape, block_size):
        """Use TensorStore to decode compressed_segmentation bytes."""
        import json
        import os
        import tempfile
        import tensorstore as ts

        nz, ny, nx = volume_shape
        bz, by, bx = block_size

        with tempfile.TemporaryDirectory() as tmpdir:
            info = {
                "@type": "neuroglancer_multiscale_volume",
                "data_type": "uint64",
                "num_channels": 1,
                "type": "segmentation",
                "scales": [{
                    "key": "s0",
                    "size": [nx, ny, nz],
                    "resolution": [8, 8, 8],
                    "chunk_sizes": [[nx, ny, nz]],
                    "encoding": "compressed_segmentation",
                    "compressed_segmentation_block_size": [bx, by, bz],
                    "voxel_offset": [0, 0, 0],
                }],
            }
            with open(os.path.join(tmpdir, "info"), "w") as f:
                json.dump(info, f)

            # Write chunk file: key is "0-{nx}_0-{ny}_0-{nz}"
            s0_dir = os.path.join(tmpdir, "s0")
            os.makedirs(s0_dir)
            chunk_key = f"0-{nx}_0-{ny}_0-{nz}"
            with open(os.path.join(s0_dir, chunk_key), "wb") as f:
                f.write(encoded_bytes)

            store = ts.open({
                "driver": "neuroglancer_precomputed",
                "kvstore": {"driver": "file", "path": tmpdir},
                "scale_index": 0,
                "open": True,
            }).result()

            # TensorStore domain is [x, y, z, channel]; read and transpose to ZYX
            arr = store[:, :, :, 0].read().result()  # shape (nx, ny, nz)
            return np.array(arr).transpose(2, 1, 0)  # → (nz, ny, nx)

    def test_solid_roundtrip(self, encoder):
        data = np.full((8, 8, 8), 99, dtype=np.uint64)
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = self._decode_with_tensorstore(encoded, (8, 8, 8), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)

    def test_two_label_roundtrip(self, encoder):
        data = np.zeros((8, 8, 8), dtype=np.uint64)
        data[:4, :, :] = 1
        data[4:, :, :] = 2
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = self._decode_with_tensorstore(encoded, (8, 8, 8), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)

    def test_many_labels_roundtrip(self, encoder):
        """Many distinct labels per block (tests 8-bit encoding)."""
        rng = np.random.default_rng(42)
        labels = rng.integers(1, 200, size=(8, 8, 8), dtype=np.uint64)
        encoded = encoder.encode_chunk(labels, block_size=(8, 8, 8))
        decoded = self._decode_with_tensorstore(encoded, (8, 8, 8), (8, 8, 8))
        np.testing.assert_array_equal(decoded, labels)

    def test_64cube_8block_roundtrip(self, encoder):
        """Full 64^3 chunk with 8^3 blocks (production config)."""
        rng = np.random.default_rng(123)
        data = rng.integers(0, 50, size=(64, 64, 64), dtype=np.uint64)
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = self._decode_with_tensorstore(encoded, (64, 64, 64), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)

    def test_64cube_64block_roundtrip(self, encoder):
        """Full 64^3 chunk with 64^3 block (single block)."""
        rng = np.random.default_rng(456)
        data = rng.integers(0, 10, size=(64, 64, 64), dtype=np.uint64)
        encoded = encoder.encode_chunk(data, block_size=(64, 64, 64))
        decoded = self._decode_with_tensorstore(encoded, (64, 64, 64), (64, 64, 64))
        np.testing.assert_array_equal(decoded, data)

    def test_non_aligned_shape_roundtrip(self, encoder):
        """Volume not evenly divisible by block_size (edge blocks)."""
        rng = np.random.default_rng(789)
        data = rng.integers(0, 20, size=(10, 12, 14), dtype=np.uint64)
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = self._decode_with_tensorstore(encoded, (10, 12, 14), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)


# ---- Layer 2b: Encoding-bits coverage and edge cases ----


class TestEncodingBitsCoverage:
    """Ensure every encoding_bits value (0,1,2,4,8,16) roundtrips correctly,
    and test edge cases the random tests might miss."""

    def test_0bit_single_label(self, encoder):
        """0-bit: 1 unique label (solid block)."""
        data = np.full((8, 8, 8), 7, dtype=np.uint64)
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
            encoded, (8, 8, 8), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)

    def test_1bit_two_labels(self, encoder):
        """1-bit: exactly 2 unique labels."""
        data = np.zeros((8, 8, 8), dtype=np.uint64)
        data[::2, :, :] = 999
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
            encoded, (8, 8, 8), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)

    def test_2bit_three_labels(self, encoder):
        """2-bit: 3 unique labels."""
        rng = np.random.default_rng(100)
        data = rng.choice([10, 20, 30], size=(8, 8, 8)).astype(np.uint64)
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
            encoded, (8, 8, 8), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)

    def test_2bit_four_labels(self, encoder):
        """2-bit: exactly 4 unique labels (boundary)."""
        rng = np.random.default_rng(101)
        data = rng.choice([1, 2, 3, 4], size=(8, 8, 8)).astype(np.uint64)
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
            encoded, (8, 8, 8), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)

    def test_4bit_five_labels(self, encoder):
        """4-bit: 5 unique labels (first case requiring 4 bits)."""
        rng = np.random.default_rng(102)
        data = rng.choice([10, 20, 30, 40, 50], size=(8, 8, 8)).astype(np.uint64)
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
            encoded, (8, 8, 8), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)

    def test_4bit_sixteen_labels(self, encoder):
        """4-bit: exactly 16 unique labels (boundary)."""
        rng = np.random.default_rng(103)
        labels = np.arange(1, 17, dtype=np.uint64)
        data = rng.choice(labels, size=(8, 8, 8)).astype(np.uint64)
        assert len(np.unique(data)) <= 16
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
            encoded, (8, 8, 8), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)

    def test_8bit_seventeen_labels(self, encoder):
        """8-bit: 17 unique labels (first case requiring 8 bits)."""
        rng = np.random.default_rng(104)
        labels = np.arange(100, 117, dtype=np.uint64)
        data = rng.choice(labels, size=(8, 8, 8)).astype(np.uint64)
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
            encoded, (8, 8, 8), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)

    def test_8bit_256_labels(self, encoder):
        """8-bit: exactly 256 unique labels (boundary)."""
        rng = np.random.default_rng(105)
        labels = np.arange(1000, 1256, dtype=np.uint64)
        data = rng.choice(labels, size=(8, 8, 8)).astype(np.uint64)
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
            encoded, (8, 8, 8), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)

    def test_16bit_257_labels(self, encoder):
        """16-bit: 257 unique labels (first case requiring 16 bits)."""
        rng = np.random.default_rng(106)
        labels = np.arange(5000, 5257, dtype=np.uint64)
        data = rng.choice(labels, size=(8, 8, 8)).astype(np.uint64)
        assert len(np.unique(data)) >= 200  # high diversity
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
            encoded, (8, 8, 8), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)

    def test_16bit_all_unique(self, encoder):
        """16-bit: every voxel has a distinct label (512 unique in 8^3)."""
        data = np.arange(512, dtype=np.uint64).reshape((8, 8, 8))
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
            encoded, (8, 8, 8), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)

    def test_large_uint64_values(self, encoder):
        """Labels spanning the full uint64 range (real segmentation uses large IDs)."""
        rng = np.random.default_rng(107)
        # Mix of small and very large uint64 values
        labels = np.array([0, 1, 2**32 - 1, 2**32, 2**48, 2**63, 2**64 - 1],
                          dtype=np.uint64)
        data = rng.choice(labels, size=(8, 8, 8)).astype(np.uint64)
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
            encoded, (8, 8, 8), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)

    def test_label_zero_preservation(self, encoder):
        """Label 0 (background) must survive encoding correctly."""
        data = np.zeros((8, 8, 8), dtype=np.uint64)
        data[3, 4, 5] = 42  # single non-zero voxel
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
            encoded, (8, 8, 8), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)
        assert decoded[0, 0, 0] == 0
        assert decoded[3, 4, 5] == 42

    def test_64cube_high_diversity(self, encoder):
        """64^3 chunk with 8^3 blocks and high per-block label diversity.

        Exercises 4-bit, 8-bit, and 16-bit encoding across different blocks
        within the same chunk.
        """
        rng = np.random.default_rng(108)
        data = rng.integers(0, 500, size=(64, 64, 64), dtype=np.uint64)
        encoded = encoder.encode_chunk(data, block_size=(8, 8, 8))
        decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
            encoded, (64, 64, 64), (8, 8, 8))
        np.testing.assert_array_equal(decoded, data)


class TestRealShardRoundtrip:
    """Roundtrip test using the real 258-chunk Arrow shard from test_data."""

    def test_all_chunks_in_real_shard(self, encoder):
        """Decode every chunk in 30720_24576_28672.arrow via dvid_to_cseg,
        then verify each through TensorStore decode.

        This exercises real-world label distributions: from 2-label solid
        blocks through 93-label dense blocks.
        """
        import io
        import pyarrow.ipc as ipc
        from braid.decompressor import DVIDDecompressor

        arrow_path = TEST_DATA_DIR / "30720_24576_28672.arrow"
        if not arrow_path.exists():
            pytest.skip("Real shard test data not available")

        with open(arrow_path, "rb") as f:
            reader = ipc.open_stream(io.BytesIO(f.read()))
            table = reader.read_all()

        decomp = DVIDDecompressor()
        errors = []

        for i in range(table.num_rows):
            labels_list = table.column("labels")[i].as_py()
            sv_list = table.column("supervoxels")[i].as_py()
            block_data = table.column("dvid_compressed_block")[i].as_py()

            # Build supervoxel → agglomerated mapping
            agglo_labels = np.array(labels_list, dtype=np.uint64)
            supervoxels = np.array(sv_list, dtype=np.uint64)

            # Get ground truth via two-step: decompress_block handles
            # zstd + DVID decode + label mapping internally
            gt_volume = decomp.decompress_block(
                block_data,
                agglo_labels=agglo_labels,
                supervoxels=supervoxels,
            )

            # Encode with our transcoder — mapping is now done in C
            encoded = encoder.dvid_to_cseg(
                block_data,
                block_size=(8, 8, 8),
                supervoxels=supervoxels,
                agglo_labels=agglo_labels,
                zstd_input=True, gzip_output=False,
            )

            # Decode with TensorStore
            decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
                encoded, (64, 64, 64), (8, 8, 8))

            if not np.array_equal(decoded, gt_volume):
                n_mismatch = np.sum(decoded != gt_volume)
                errors.append(
                    f"chunk {i}: {n_mismatch}/{64**3} voxels differ, "
                    f"N={num_labels} labels"
                )

        assert not errors, (
            f"{len(errors)}/{table.num_rows} chunks failed:\n"
            + "\n".join(errors[:10])
        )


# ---- Layer 3: Fused DVID-to-cseg path ----


class TestDVIDtoCseg:
    """Test the fused dvid_to_cseg path using real DVID test data."""

    @staticmethod
    def _load_raw_volume(filename):
        with gzip.open(TEST_DATA_DIR / filename, "rb") as f:
            data = f.read()
        return np.frombuffer(data, dtype="<u8").reshape((64, 64, 64))

    @staticmethod
    def _load_compressed_block(filename):
        with gzip.open(TEST_DATA_DIR / filename, "rb") as f:
            return f.read()

    def test_fib19_fused_roundtrip(self, encoder):
        """DVID block → dvid_to_cseg → TensorStore decode → matches ground truth."""
        ground_truth = self._load_raw_volume("fib19-64x64x64-sample1.dat.gz")
        dvid_block = self._load_compressed_block("fib19-64x64x64-sample1-block.dat.gz")

        import zstandard as zstd
        zstd_data = zstd.ZstdCompressor().compress(dvid_block)

        # No mapping (identity) — fib19 block labels are used as-is
        encoded = encoder.dvid_to_cseg(
            zstd_data,
            block_size=(8, 8, 8),
            zstd_input=True, gzip_output=False,
        )

        decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
            encoded, (64, 64, 64), (8, 8, 8),
        )
        np.testing.assert_array_equal(decoded, ground_truth)

    def test_fused_with_gzip(self, encoder):
        """Test gzip output flag: dvid_to_cseg with BRAID_OUTPUT_GZIP."""
        dvid_block = self._load_compressed_block("fib19-64x64x64-sample1-block.dat.gz")

        import zstandard as zstd
        zstd_data = zstd.ZstdCompressor().compress(dvid_block)

        gzipped = encoder.dvid_to_cseg(
            zstd_data,
            block_size=(8, 8, 8),
            zstd_input=True, gzip_output=True,
        )

        # Verify it's valid gzip by decompressing
        raw = gzip.decompress(gzipped)
        assert len(raw) > 0, "Gzip output should decompress to non-empty bytes"

        # The decompressed bytes should be valid compressed_segmentation
        ground_truth = self._load_raw_volume("fib19-64x64x64-sample1.dat.gz")
        decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
            raw, (64, 64, 64), (8, 8, 8),
        )
        np.testing.assert_array_equal(decoded, ground_truth)

    def test_parity_fused_vs_twostep(self, encoder):
        """Fused dvid_to_cseg should produce identical output to decompress+encode."""
        import zstandard as zstd

        dvid_block = self._load_compressed_block("fib19-64x64x64-sample1-block.dat.gz")
        zstd_data = zstd.ZstdCompressor().compress(dvid_block)

        # Two-step: decompress DVID via Python, then encode
        from braid.decompressor import DVIDDecompressor
        decomp = DVIDDecompressor()
        volume = decomp.decompress_block(zstd_data)
        two_step = encoder.encode_chunk(volume, block_size=(8, 8, 8))

        # Fused: dvid_to_cseg (zstd input, identity mapping)
        fused = encoder.dvid_to_cseg(
            zstd_data,
            block_size=(8, 8, 8),
            zstd_input=True, gzip_output=False,
        )

        assert two_step == fused, (
            f"Fused vs two-step mismatch: {len(fused)} vs {len(two_step)} bytes"
        )

    def test_label_mapping(self, encoder):
        """Verify supervoxel -> agglomerated mapping is applied correctly.

        Use the fib19 block whose internal labels are supervoxel IDs.
        Provide a mapping that remaps each supervoxel to a new value,
        then verify the output matches what the Python decompressor
        produces with the same mapping.
        """
        import zstandard as zstd

        dvid_block = self._load_compressed_block("fib19-64x64x64-sample1-block.dat.gz")
        zstd_data = zstd.ZstdCompressor().compress(dvid_block)

        # Parse block header to get the supervoxel labels
        num_labels = struct.unpack_from("<I", dvid_block, 12)[0]
        block_svs = np.frombuffer(dvid_block, dtype="<u8",
                                  count=num_labels, offset=16)

        # Create a synthetic mapping: remap each supervoxel to sv + 1000000
        sv_keys = block_svs.copy()
        agglo_vals = block_svs + 1000000

        # Fused path with mapping (zstd input)
        encoded = encoder.dvid_to_cseg(
            zstd_data,
            block_size=(8, 8, 8),
            supervoxels=sv_keys,
            agglo_labels=agglo_vals,
            zstd_input=True, gzip_output=False,
        )

        # Ground truth: Python decompressor with the same mapping
        from braid.decompressor import DVIDDecompressor
        decomp = DVIDDecompressor()
        gt_volume = decomp.decompress_block(
            zstd_data,
            agglo_labels=agglo_vals,
            supervoxels=sv_keys,
        )

        decoded = TestTensorStoreRoundtrip._decode_with_tensorstore(
            encoded, (64, 64, 64), (8, 8, 8))
        np.testing.assert_array_equal(decoded, gt_volume)

        # Verify the mapping was actually applied (not identity)
        original_gt = self._load_raw_volume("fib19-64x64x64-sample1.dat.gz")
        assert not np.array_equal(decoded, original_gt), \
            "Mapping should produce different labels than identity"
