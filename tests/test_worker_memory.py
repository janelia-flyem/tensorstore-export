"""Tests for worker memory tracking and cgroup reading."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.worker import _read_cgroup_memory


class TestReadCgroupMemory:
    """Test _read_cgroup_memory with simulated cgroup files."""

    def test_returns_3_tuple(self):
        """Always returns (current, limit, peak) tuple."""
        result = _read_cgroup_memory()
        assert len(result) == 3

    def test_missing_cgroup_files_returns_zeros(self):
        """Returns (0, 0, 0) when no cgroup files exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            current, limit, peak = _read_cgroup_memory()
        assert current == 0
        assert limit == 0
        assert peak == 0

    def test_simulated_cgroup_files(self, tmp_path):
        """Simulate cgroup v2 files and verify all three values are read."""
        current_file = tmp_path / "memory.current"
        max_file = tmp_path / "memory.max"
        peak_file = tmp_path / "memory.peak"

        current_file.write_text("2147483648\n")  # 2 GiB
        max_file.write_text("4294967296\n")       # 4 GiB
        peak_file.write_text("3221225472\n")      # 3 GiB

        # Patch the file paths used by _read_cgroup_memory
        cgv2_current = str(current_file)
        cgv2_max = str(max_file)
        cgv2_peak = str(peak_file)

        original_open = open

        def mock_open(path, *args, **kwargs):
            path_str = str(path)
            if path_str == "/sys/fs/cgroup/memory.current":
                return original_open(cgv2_current, *args, **kwargs)
            elif path_str == "/sys/fs/cgroup/memory.max":
                return original_open(cgv2_max, *args, **kwargs)
            elif path_str == "/sys/fs/cgroup/memory.peak":
                return original_open(cgv2_peak, *args, **kwargs)
            raise FileNotFoundError(path_str)

        with patch("builtins.open", side_effect=mock_open):
            current, limit, peak = _read_cgroup_memory()

        assert current == 2147483648
        assert limit == 4294967296
        assert peak == 3221225472

    def test_max_value_of_max_means_no_limit(self, tmp_path):
        """memory.max containing 'max' means no limit."""
        current_file = tmp_path / "memory.current"
        max_file = tmp_path / "memory.max"

        current_file.write_text("1000000\n")
        max_file.write_text("max\n")

        original_open = open

        def mock_open(path, *args, **kwargs):
            path_str = str(path)
            if path_str == "/sys/fs/cgroup/memory.current":
                return original_open(str(current_file), *args, **kwargs)
            elif path_str == "/sys/fs/cgroup/memory.max":
                return original_open(str(max_file), *args, **kwargs)
            raise FileNotFoundError(path_str)

        with patch("builtins.open", side_effect=mock_open):
            current, limit, peak = _read_cgroup_memory()

        assert current == 1000000
        assert limit == 0  # "max" means no limit
        assert peak == 0   # no peak file


# =========================================================================
# Compressed Z-index tests — verified against TensorStore C++ reference
#
# Reference sources:
#   ~/tensorstore/tensorstore/driver/neuroglancer_precomputed/metadata.cc
#     - GetCompressedZIndexBits()
#     - EncodeCompressedZIndex()
#   ~/tensorstore/tensorstore/driver/neuroglancer_precomputed/metadata_test.cc
#     - TEST(GetCompressedZIndexBitsTest, Basic)
#     - TEST(EncodeCompressedZIndexTest, Basic)
#     - TEST(GetChunksPerVolumeShardFunctionTest, AllShardsFull)
#   ~/tensorstore/tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded.cc
#     - GetChunkShardInfo(), GetSplitShardInfo(), GetShardKey()
# =========================================================================

from src.ng_sharding import (
    compressed_z_index,
    get_compressed_z_index_bits,
    chunk_shard_info,
    dvid_to_ng_shard_number,
    ng_shard_filename,
    load_ng_spec,
    load_ng_spec_from_dict,
    get_shard_chunk_hierarchy,
    shard_origin_in_chunks,
    chunks_per_shard,
    shard_bbox,
    parent_shards_to_child_shards,
    enumerate_shard_bboxes,
)


class TestGetCompressedZIndexBits:
    """Test get_compressed_z_index_bits against TensorStore's
    GetCompressedZIndexBitsTest::Basic test vectors."""

    def test_tensorstore_vector_1(self):
        """TensorStore test: shape=[0, 0xffffffff, kMaxFiniteIndex], chunk=[20, 1, 1].

        grid = [0, 0xffffffff, kMaxFiniteIndex]
        bits = [bit_width(max(0, 0-1)), bit_width(0xfffffffe), bit_width(kMaxFiniteIndex-1)]
             = [0, 32, 62]

        kMaxFiniteIndex is 2^62 - 2 in TensorStore, so grid = 2^62 - 2,
        val = 2^62 - 3, bit_width = 62.
        """
        # Shape 0 -> grid 0 -> max(0, -1) = 0 -> bit_width 0
        assert get_compressed_z_index_bits([0, 0xFFFFFFFF, (1 << 62) - 2],
                                           [20, 1, 1]) == [0, 32, 62]

    def test_tensorstore_vector_2(self):
        """TensorStore test: shape=[79, 80, 144], chunk=[20, 20, 12]."""
        # grid = [4, 4, 12], val = [3, 3, 11], bits = [2, 2, 4]
        assert get_compressed_z_index_bits([79, 80, 144], [20, 20, 12]) == [2, 2, 4]

    def test_mcns_scale0(self):
        """mCNS v0.11 scale 0: size=[94088, 78317, 134576], chunk=[64,64,64]."""
        bits = get_compressed_z_index_bits([94088, 78317, 134576], [64, 64, 64])
        assert bits == [11, 11, 12]

    def test_single_chunk_dimension(self):
        """grid_shape[d]==1 means 0 bits for that dimension."""
        # shape=50, chunk=50 -> grid=1 -> val=0 -> bits=0
        assert get_compressed_z_index_bits([50, 100, 200], [50, 50, 50]) == [0, 1, 2]


class TestCompressedZIndex:
    """Test compressed_z_index against TensorStore's
    EncodeCompressedZIndexTest::Basic test vectors."""

    # --- TensorStore metadata_test.cc exact test vectors ---
    # bits = [4, 2, 1]

    def test_ts_origin(self):
        assert compressed_z_index((0, 0, 0), (4, 2, 1)) == 0

    def test_ts_unit_x(self):
        assert compressed_z_index((1, 0, 0), (4, 2, 1)) == 1

    def test_ts_unit_y(self):
        assert compressed_z_index((0, 1, 0), (4, 2, 1)) == 2

    def test_ts_vector_4(self):
        """EncodeCompressedZIndex({0b10, 0b0, 0b1}, [4,2,1]) == 0b1100."""
        assert compressed_z_index((0b10, 0b0, 0b1), (4, 2, 1)) == 0b1100

    def test_ts_vector_5(self):
        """EncodeCompressedZIndex({0b10, 0b11, 0b0}, [4,2,1]) == 0b11010."""
        assert compressed_z_index((0b10, 0b11, 0b0), (4, 2, 1)) == 0b11010

    def test_ts_vector_6(self):
        """EncodeCompressedZIndex({0b1001, 0b10, 0b1}, [4,2,1]) == 0b1010101."""
        assert compressed_z_index((0b1001, 0b10, 0b1), (4, 2, 1)) == 0b1010101

    # --- Equal-bits tests (standard Morton equivalence) ---

    def test_origin_equal_bits(self):
        assert compressed_z_index((0, 0, 0), (11, 11, 12)) == 0

    def test_unit_x_equal_bits(self):
        assert compressed_z_index((1, 0, 0), (3, 3, 3)) == 1

    def test_unit_y_equal_bits(self):
        assert compressed_z_index((0, 1, 0), (3, 3, 3)) == 2

    def test_unit_z_equal_bits(self):
        assert compressed_z_index((0, 0, 1), (3, 3, 3)) == 4

    def test_standard_morton_when_equal_bits(self):
        """When all dimensions have equal bits, compressed == standard Morton."""
        assert compressed_z_index((1, 1, 1), (2, 2, 2)) == 0b000_111
        assert compressed_z_index((2, 1, 0), (2, 2, 2)) == 0b000_010 | (1 << 3)

    # --- Compressed-specific tests (unequal bits) ---

    def test_compressed_skips_exhausted_dimensions(self):
        """When z has more bits than x,y, compressed Z-index skips them."""
        result = compressed_z_index((1, 1, 3), (1, 1, 2))
        assert result == 0b1111  # 15

        result = compressed_z_index((0, 0, 2), (1, 1, 2))
        assert result == 0b1000  # 8

    def test_different_from_standard_morton(self):
        """Compressed Z-index differs from standard Morton when dims unequal."""
        # (0, 0, 4) with coord_bits (1, 1, 3)
        # Standard Morton: z[2] at position 2*3=6 -> 64
        # Compressed: z[2] at position 4 -> 16 (because x,y exhausted at level 1)
        result = compressed_z_index((0, 0, 4), (1, 1, 3))
        assert result == (1 << 4)  # 16, NOT 64 (standard Morton)


class TestChunkShardInfo:
    """Test chunk_shard_info against TensorStore's GetChunkShardInfo +
    GetSplitShardInfo, derived from GetChunksPerVolumeShardFunctionTest
    test vectors in metadata_test.cc.

    Test setup from TensorStore:
      ShardingSpec: hash=identity, preshift=1, minishard=2, shard=3
      Volume: [99, 98, 97], Chunk: [50, 25, 13]
      Grid: [2, 4, 8], z_bits: [1, 2, 3]

    Shard origins (verified in AllShardsFull + PartialShards1Dim):
      Shard 0: chunk origin (0, 0, 0)
      Shard 1: chunk origin (0, 2, 0)
      Shard 2: chunk origin (0, 0, 2)
      Shard 3: chunk origin (0, 2, 2)
      Shard 4: chunk origin (0, 0, 4)
      Shard 5: chunk origin (0, 2, 4)
      Shard 6: chunk origin (0, 0, 6)
      Shard 7: chunk origin (0, 2, 6)
    """

    # z_bits for all these tests
    BITS = (1, 2, 3)
    # sharding params
    PRESHIFT = 1
    MINISHARD = 2
    SHARD = 3

    def _shard_for_chunk(self, cx, cy, cz):
        z = compressed_z_index((cx, cy, cz), self.BITS)
        shard, minishard = chunk_shard_info(z, self.PRESHIFT, self.MINISHARD, self.SHARD)
        return shard

    def test_shard_0_origin(self):
        assert self._shard_for_chunk(0, 0, 0) == 0

    def test_shard_1_origin(self):
        assert self._shard_for_chunk(0, 2, 0) == 1

    def test_shard_2_origin(self):
        assert self._shard_for_chunk(0, 0, 2) == 2

    def test_shard_3_origin(self):
        assert self._shard_for_chunk(0, 2, 2) == 3

    def test_shard_4_origin(self):
        assert self._shard_for_chunk(0, 0, 4) == 4

    def test_shard_5_origin(self):
        assert self._shard_for_chunk(0, 2, 4) == 5

    def test_shard_6_origin(self):
        assert self._shard_for_chunk(0, 0, 6) == 6

    def test_shard_7_origin(self):
        assert self._shard_for_chunk(0, 2, 6) == 7

    def test_multiple_chunks_same_shard(self):
        """All chunks within a shard's spatial region map to the same shard.

        Shard shape is (2, 2, 2) chunks.  Shard 0 covers chunks
        (0..1, 0..1, 0..1) -- all 8 should map to shard 0.
        """
        for cx in range(2):
            for cy in range(2):
                for cz in range(2):
                    assert self._shard_for_chunk(cx, cy, cz) == 0, \
                        f"chunk ({cx},{cy},{cz}) expected shard 0"

    def test_invalid_shard_beyond_range(self):
        """Shard number never exceeds 2^shard_bits - 1."""
        # Max chunk in grid [2, 4, 8] is (1, 3, 7)
        shard = self._shard_for_chunk(1, 3, 7)
        assert 0 <= shard < (1 << self.SHARD)

    def test_shard_bits_zero(self):
        """With shard_bits=0, everything maps to shard 0."""
        z = compressed_z_index((5, 3, 7), (4, 4, 4))
        shard, _ = chunk_shard_info(z, preshift_bits=9, minishard_bits=4, shard_bits=0)
        assert shard == 0

    def test_minishard_derivation(self):
        """Verify minishard assignment matches TensorStore's split logic.

        For shard 0: chunk (0, 1, 0) should have minishard != 0 since it
        differs from (0, 0, 0) only in the minishard bits.
        """
        z0 = compressed_z_index((0, 0, 0), self.BITS)
        z1 = compressed_z_index((0, 1, 0), self.BITS)

        _, ms0 = chunk_shard_info(z0, self.PRESHIFT, self.MINISHARD, self.SHARD)
        _, ms1 = chunk_shard_info(z1, self.PRESHIFT, self.MINISHARD, self.SHARD)

        assert ms0 != ms1  # different chunks, different minishards


class TestNgShardFilename:
    """Test ng_shard_filename against TensorStore's GetShardKey.

    GetShardKey: StrFormat("%0*x.shard", CeilOfRatio(shard_bits, 4), num)
    """

    def test_shard_bits_19(self):
        """mCNS scale 0: shard_bits=19 -> 5 hex digits."""
        assert ng_shard_filename(25029, 19) == "061c5.shard"

    def test_shard_bits_16(self):
        """mCNS scale 1: shard_bits=16 -> 4 hex digits."""
        assert ng_shard_filename(0, 16) == "0000.shard"
        assert ng_shard_filename(0xFFFF, 16) == "ffff.shard"

    def test_shard_bits_4(self):
        """mCNS scale 5: shard_bits=4 -> 1 hex digit."""
        assert ng_shard_filename(15, 4) == "f.shard"
        assert ng_shard_filename(0, 4) == "0.shard"

    def test_shard_bits_3(self):
        """TensorStore test case: shard_bits=3 -> 1 hex digit."""
        assert ng_shard_filename(5, 3) == "5.shard"

    def test_shard_bits_1(self):
        """mCNS scale 6: shard_bits=1 -> 1 hex digit."""
        assert ng_shard_filename(0, 1) == "0.shard"
        assert ng_shard_filename(1, 1) == "1.shard"

    def test_shard_bits_0(self):
        """mCNS scales 7-9: shard_bits=0 -> CeilOfRatio(0,4)=0, but
        printf("%0*x", 0, 0) still produces "0"."""
        assert ng_shard_filename(0, 0) == "0.shard"

    def test_shard_bits_8(self):
        """shard_bits=8 -> 2 hex digits."""
        assert ng_shard_filename(42, 8) == "2a.shard"
        assert ng_shard_filename(0xFF, 8) == "ff.shard"

    def test_shard_bits_12(self):
        """shard_bits=12 -> 3 hex digits."""
        assert ng_shard_filename(0xFF, 12) == "0ff.shard"


class TestDvidToNgShardNumber:
    """Test the full DVID shard name -> NG shard number pipeline."""

    def test_mcns_scale0_known_mapping(self):
        """Verified against GCS: shard 10240_40960_43008 -> 061c5.shard."""
        spec_path = str(Path(__file__).parent.parent / "examples" /
                        "mcns-export-specs.json")
        scale_info = load_ng_spec(spec_path)
        s0 = scale_info[0]

        assert s0["coord_bits"] == [11, 11, 12]
        assert s0["preshift_bits"] == 9
        assert s0["minishard_bits"] == 6
        assert s0["shard_bits"] == 19

        shard_num = dvid_to_ng_shard_number("10240_40960_43008", s0)
        assert shard_num == 25029
        assert ng_shard_filename(shard_num, s0["shard_bits"]) == "061c5.shard"

    def test_origin_shard(self):
        """Origin (0,0,0) should map to shard 0."""
        spec_path = str(Path(__file__).parent.parent / "examples" /
                        "mcns-export-specs.json")
        scale_info = load_ng_spec(spec_path)
        s0 = scale_info[0]

        shard_num = dvid_to_ng_shard_number("0_0_0", s0)
        assert shard_num == 0


class TestLoadNgSpec:
    """Test NG spec loading and consistency."""

    @pytest.fixture
    def spec_path(self):
        return str(Path(__file__).parent.parent / "examples" /
                   "mcns-export-specs.json")

    def test_load_ng_spec_has_10_scales(self, spec_path):
        spec = load_ng_spec(spec_path)
        assert len(spec) == 10

    def test_load_ng_spec_from_dict_matches(self, spec_path):
        """load_ng_spec and load_ng_spec_from_dict produce identical results."""
        import json
        from_file = load_ng_spec(spec_path)
        with open(spec_path) as f:
            from_dict = load_ng_spec_from_dict(json.load(f))
        assert from_file == from_dict

    def test_all_scales_total_bits_fit(self, spec_path):
        """For every scale, total Z-index bits <= preshift + minishard + shard.

        This is a necessary condition for rectangular shard regions (identity hash).
        Verified against TensorStore's GetShardChunkHierarchy constraint.
        """
        spec = load_ng_spec(spec_path)
        for i, params in spec.items():
            total_z_bits = sum(params["coord_bits"])
            total_shard_bits = (params["preshift_bits"] +
                                params["minishard_bits"] +
                                params["shard_bits"])
            assert total_z_bits <= total_shard_bits, \
                f"Scale {i}: {total_z_bits} Z-bits > {total_shard_bits} shard bits"

    def test_scale0_grid_shape(self, spec_path):
        """Verify grid shape computation against known values."""
        spec = load_ng_spec(spec_path)
        s0 = spec[0]
        # 94088/64 = 1470.125 -> ceil = 1471
        # 78317/64 = 1223.703 -> ceil = 1224
        # 134576/64 = 2102.75 -> ceil = 2103
        assert s0["grid_shape"] == [1471, 1224, 2103]

    def test_max_shard_number_within_shard_bits(self, spec_path):
        """For each scale, the max possible shard number fits in shard_bits."""
        spec = load_ng_spec(spec_path)
        for i, params in spec.items():
            grid = params["grid_shape"]
            # Max chunk coordinate
            max_coord = [g - 1 for g in grid]
            max_z = compressed_z_index(max_coord, params["coord_bits"])
            shard, _ = chunk_shard_info(
                max_z,
                params["preshift_bits"],
                params["minishard_bits"],
                params["shard_bits"],
            )
            if params["shard_bits"] > 0:
                assert shard < (1 << params["shard_bits"]), \
                    f"Scale {i}: shard {shard} >= 2^{params['shard_bits']}"
            else:
                assert shard == 0, f"Scale {i}: shard_bits=0 but shard={shard}"


# =========================================================================
# Shard chunk hierarchy tests — verified against TensorStore C++ reference
#
# Reference:
#   ~/tensorstore/tensorstore/driver/neuroglancer_precomputed/metadata_test.cc
#     - TEST(GetChunksPerVolumeShardFunctionTest, AllShardsFull)
#     - TEST(GetChunksPerVolumeShardFunctionTest, PartialShards1Dim)
#     - TEST(GetChunksPerVolumeShardFunctionTest, PartialShards2Dims)
# =========================================================================


class TestGetShardChunkHierarchy:
    """Test get_shard_chunk_hierarchy against TensorStore's
    GetChunksPerVolumeShardFunction test vectors."""

    # TensorStore test setup:
    #   ShardingSpec: hash=identity, preshift=1, minishard=2, shard=3
    #   Volume: [99, 98, 97], Chunk: [50, 25, 13]
    #   Grid: [2, 4, 8], z_bits: [1, 2, 3]
    TS_PARAMS = {
        "coord_bits": [1, 2, 3],
        "preshift_bits": 1,
        "minishard_bits": 2,
        "shard_bits": 3,
        "grid_shape": [2, 4, 8],
        "chunk_size": [50, 25, 13],
        "vol_size": [99, 98, 97],
    }

    def test_shard_shape_in_chunks(self):
        """Shard shape should be [2, 2, 2] chunks for the TS test case."""
        h = get_shard_chunk_hierarchy(self.TS_PARAMS)
        assert h["shard_shape_in_chunks"] == [2, 2, 2]

    def test_minishard_shape_in_chunks(self):
        """Minishard shape: preshift=1 consumes 1 bit from x (the first dim
        with bits), giving [2, 1, 1]."""
        h = get_shard_chunk_hierarchy(self.TS_PARAMS)
        assert h["minishard_shape_in_chunks"] == [2, 1, 1]

    def test_all_shards_full(self):
        """AllShardsFull: all 8 shards have 8 chunks."""
        for sn in range(8):
            assert chunks_per_shard(sn, self.TS_PARAMS) == 8

    def test_partial_shards_1dim(self):
        """PartialShards1Dim: grid [2, 4, 7], shards 6-7 have 4 chunks."""
        params = dict(self.TS_PARAMS, grid_shape=[2, 4, 7])
        # Shards 0-5: 8 chunks each, shards 6-7: 4 chunks each
        expected = [8, 8, 8, 8, 8, 8, 4, 4]
        for sn in range(8):
            assert chunks_per_shard(sn, params) == expected[sn], \
                f"shard {sn}: expected {expected[sn]}, got {chunks_per_shard(sn, params)}"

    def test_partial_shards_2dims(self):
        """PartialShards2Dims: grid [2, 3, 7], mixed chunk counts."""
        params = dict(self.TS_PARAMS, grid_shape=[2, 3, 7])
        # Shards with y=2..3 (shard 1,3,5,7) lose 1 of 2 y-chunks
        expected = [8, 4, 8, 4, 8, 4, 4, 2]
        for sn in range(8):
            assert chunks_per_shard(sn, params) == expected[sn], \
                f"shard {sn}: expected {expected[sn]}, got {chunks_per_shard(sn, params)}"

    def test_mcns_scale0_shard_shape(self):
        """mCNS v0.11 scale 0: shard shape should be [32, 32, 32] chunks."""
        spec_path = str(Path(__file__).parent.parent / "examples" /
                        "mcns-export-specs.json")
        spec = load_ng_spec(spec_path)
        h = get_shard_chunk_hierarchy(spec[0])
        assert h["shard_shape_in_chunks"] == [32, 32, 32]

    def test_mcns_scale0_minishard_shape(self):
        """mCNS v0.11 scale 0: minishard shape should be [8, 8, 8] chunks."""
        spec_path = str(Path(__file__).parent.parent / "examples" /
                        "mcns-export-specs.json")
        spec = load_ng_spec(spec_path)
        h = get_shard_chunk_hierarchy(spec[0])
        assert h["minishard_shape_in_chunks"] == [8, 8, 8]

    def test_mcns_shard_shape_covers_grid(self):
        """For every scale, shard shape * max_shards >= grid_shape per dim.

        The shard shape times the number of shards along each dimension
        must cover the full grid. This is a necessary condition for
        correct spatial coverage.
        """
        spec_path = str(Path(__file__).parent.parent / "examples" /
                        "mcns-export-specs.json")
        spec = load_ng_spec(spec_path)
        for i, params in spec.items():
            h = get_shard_chunk_hierarchy(params)
            for d in range(3):
                ss = h["shard_shape_in_chunks"][d]
                gs = h["grid_shape_in_chunks"][d]
                assert ss > 0, f"Scale {i}: shard_shape[{d}] = 0"
                assert ss <= gs, \
                    f"Scale {i}: shard_shape[{d}] = {ss} > grid[{d}] = {gs}"


class TestShardOriginAndBbox:
    """Test shard_origin_in_chunks and shard_bbox."""

    TS_PARAMS = TestGetShardChunkHierarchy.TS_PARAMS

    def test_shard_0_origin(self):
        assert shard_origin_in_chunks(0, self.TS_PARAMS) == [0, 0, 0]

    def test_shard_1_origin(self):
        """Shard 1 -> y offset by 2 chunks."""
        assert shard_origin_in_chunks(1, self.TS_PARAMS) == [0, 2, 0]

    def test_shard_2_origin(self):
        """Shard 2 -> z offset by 2 chunks."""
        assert shard_origin_in_chunks(2, self.TS_PARAMS) == [0, 0, 2]

    def test_all_shard_origins_unique(self):
        """All 8 shard origins should be distinct."""
        origins = [tuple(shard_origin_in_chunks(sn, self.TS_PARAMS))
                   for sn in range(8)]
        assert len(set(origins)) == 8

    def test_shard_bbox_voxel_extent(self):
        """shard_bbox returns voxel coordinates, clipped to volume."""
        bb = shard_bbox(0, self.TS_PARAMS)
        assert bb["shard_origin"] == [0, 0, 0]
        # shard_extent: min(2*50, 99-0)=99, min(2*25, 98-0)=50, min(2*13, 97-0)=26
        assert bb["shard_extent"] == [99, 50, 26]
        assert bb["num_chunks"] == 8

    def test_shard_bbox_boundary(self):
        """Boundary shard has clipped extent.

        Shard 7 origin is [0, 2, 6] in chunk space because x has only 1 bit
        which gets consumed by preshift, so x-origin is always 0.
        Voxel origin = (0, 50, 78), extent clipped to vol bounds.
        """
        bb = shard_bbox(7, self.TS_PARAMS)
        assert bb["shard_origin"] == [0, 50, 78]
        # shard_extent: min(2*50, 99-0)=99, min(2*25, 98-50)=48, min(2*13, 97-78)=19
        assert bb["shard_extent"] == [99, 48, 19]

    def test_mcns_shard0_bbox(self):
        """mCNS v0.11 scale 0 shard 0: origin at (0,0,0), extent = 32*64 = 2048."""
        spec_path = str(Path(__file__).parent.parent / "examples" /
                        "mcns-export-specs.json")
        spec = load_ng_spec(spec_path)
        bb = shard_bbox(0, spec[0])
        assert bb["shard_origin"] == [0, 0, 0]
        assert bb["shard_extent"] == [2048, 2048, 2048]
        assert bb["num_chunks"] == 32768  # 32^3


class TestParentShardsToChildShards:
    """Test the shard derivation chain for downres."""

    @pytest.fixture
    def spec(self):
        spec_path = str(Path(__file__).parent.parent / "examples" /
                        "mcns-export-specs.json")
        return load_ng_spec(spec_path)

    def test_single_parent_produces_children(self, spec):
        """A single parent shard should produce 1 or more child shards."""
        children = parent_shards_to_child_shards([0], spec[0], spec[1])
        assert len(children) >= 1
        assert 0 in children  # origin shard maps to origin

    def test_child_shards_in_valid_range(self, spec):
        """All derived child shards must be valid shard numbers."""
        children = parent_shards_to_child_shards([0, 1, 2], spec[0], spec[1])
        max_shard = 1 << spec[1]["shard_bits"]
        for sn in children:
            assert 0 <= sn < max_shard, f"Invalid child shard {sn}"

    def test_derivation_chain_decreases_shard_count(self, spec):
        """Each downres level should have fewer or equal shards."""
        # Start with all possible s0 shards (full grid)
        s0_max = 1 << spec[0]["shard_bits"]
        s0_shards = [sn for sn in range(s0_max) if chunks_per_shard(sn, spec[0]) > 0]

        prev_shards = s0_shards
        for target_scale in range(1, 5):
            children = parent_shards_to_child_shards(
                prev_shards, spec[target_scale - 1], spec[target_scale]
            )
            assert len(children) <= len(prev_shards), \
                f"s{target_scale} has more shards ({len(children)}) than " \
                f"s{target_scale-1} ({len(prev_shards)})"
            prev_shards = children

    def test_empty_parent_produces_no_children(self, spec):
        """No parent shards -> no child shards."""
        children = parent_shards_to_child_shards([], spec[0], spec[1])
        assert children == []

    def test_enumerate_shard_bboxes_total_chunks(self, spec):
        """Total chunks from enumerate_shard_bboxes should match grid volume.

        Uses scale 5 (shard_bits=4 -> 16 shards) for speed.
        """
        params = spec[5]
        bboxes = enumerate_shard_bboxes(params)
        total = sum(bb["num_chunks"] for bb in bboxes)
        grid = params["grid_shape"]
        assert total == grid[0] * grid[1] * grid[2]
