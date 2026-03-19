"""
Tests using Arrow IPC shard files produced by DVID's Go export-shards command.

These files were copied from a real mCNS export run, verifying that BRAID's
Python Arrow reader and DVID decompressor are compatible with Go's Arrow
writer output.

Test data (in tests/test_data/):
    30720_24576_28672.arrow — Arrow IPC stream (258 records, edge shard from mCNS scale 1)
    30720_24576_28672.csv   — CSV chunk index (x,y,z,rec)
"""

from pathlib import Path

import numpy as np
import pytest

from braid import ShardReader, LabelType
from braid.decompressor import DVIDDecompressor

TEST_DATA_DIR = Path(__file__).parent / "test_data"
ARROW_FILE = TEST_DATA_DIR / "30720_24576_28672.arrow"
CSV_FILE = TEST_DATA_DIR / "30720_24576_28672.csv"


@pytest.fixture
def go_shard():
    """Open the Go-produced shard with ShardReader."""
    if not ARROW_FILE.exists() or not CSV_FILE.exists():
        pytest.skip("Go-produced shard test data not available")
    return ShardReader(ARROW_FILE, CSV_FILE)


class TestGoProducedShardReader:
    """Verify BRAID can read Arrow IPC files produced by DVID's Go export-shards."""

    def test_reader_opens(self, go_shard):
        """ShardReader successfully opens Go-produced Arrow+CSV files."""
        assert go_shard is not None
        assert go_shard.chunk_count > 0

    def test_chunk_count_matches_csv(self, go_shard):
        """Number of chunks matches CSV line count (258 records)."""
        assert go_shard.chunk_count == 258

    def test_schema_fields(self, go_shard):
        """Arrow schema has all expected fields."""
        expected_fields = {
            "chunk_x", "chunk_y", "chunk_z",
            "labels", "supervoxels",
            "dvid_compressed_block", "uncompressed_size",
        }
        actual_fields = {field.name for field in go_shard._table.schema}
        assert expected_fields.issubset(actual_fields), (
            f"Missing fields: {expected_fields - actual_fields}"
        )

    def test_csv_coordinates_match_arrow(self, go_shard):
        """Every CSV index entry matches the Arrow record's chunk coordinates."""
        for (cx, cy, cz), rec_idx in go_shard._chunk_index.items():
            arrow_x = go_shard._table["chunk_x"][rec_idx].as_py()
            arrow_y = go_shard._table["chunk_y"][rec_idx].as_py()
            arrow_z = go_shard._table["chunk_z"][rec_idx].as_py()
            assert (arrow_x, arrow_y, arrow_z) == (cx, cy, cz), (
                f"CSV coord ({cx},{cy},{cz}) at rec {rec_idx} != "
                f"Arrow coord ({arrow_x},{arrow_y},{arrow_z})"
            )

    def test_chunk_coordinates_in_shard_range(self, go_shard):
        """All chunk coordinates fall within the expected shard region.

        Shard origin is (30720, 24576, 28672) in voxels.  At scale 1 with
        2048-voxel shard dims, chunk coords should be in the range
        [origin/64, (origin+2048)/64) per dimension.
        """
        origin = (30720, 24576, 28672)
        shard_dim = 2048
        chunk_size = 64
        for cx, cy, cz in go_shard.available_chunks:
            for coord, orig, dim_name in [(cx, origin[0], "X"), (cy, origin[1], "Y"), (cz, origin[2], "Z")]:
                lo = orig // chunk_size
                hi = (orig + shard_dim) // chunk_size
                assert lo <= coord < hi, (
                    f"Chunk {dim_name}={coord} outside shard range [{lo}, {hi})"
                )


class TestGoProducedShardDecompression:
    """Verify decompression of chunks from Go-produced shards."""

    def test_first_chunk_decompresses(self, go_shard):
        """First chunk decompresses to correct shape and dtype."""
        coords = go_shard.available_chunks[0]
        chunk = go_shard.read_chunk(*coords, label_type=LabelType.SUPERVOXELS)
        assert chunk.shape == (64, 64, 64)
        assert chunk.dtype == np.uint64

    def test_all_chunks_decompress(self, go_shard):
        """Every chunk in the shard decompresses without error."""
        failures = []
        for cx, cy, cz in go_shard.available_chunks:
            try:
                chunk = go_shard.read_chunk(cx, cy, cz, label_type=LabelType.SUPERVOXELS)
                assert chunk.shape == (64, 64, 64)
                assert chunk.dtype == np.uint64
            except Exception as e:
                failures.append(f"({cx},{cy},{cz}): {e}")
        assert not failures, f"{len(failures)} chunks failed:\n" + "\n".join(failures[:5])

    def test_supervoxels_match_label_list(self, go_shard):
        """Decompressed voxel values are a subset of the chunk's supervoxel list."""
        # Test a sample of chunks
        for coords in go_shard.available_chunks[:20]:
            cx, cy, cz = coords
            chunk = go_shard.read_chunk(cx, cy, cz, label_type=LabelType.SUPERVOXELS)
            raw = go_shard.read_chunk_raw(cx, cy, cz)
            supervoxels = set(raw["supervoxels"])
            voxel_values = set(np.unique(chunk))
            assert voxel_values.issubset(supervoxels), (
                f"Chunk ({cx},{cy},{cz}): voxel values {voxel_values - supervoxels} "
                f"not in supervoxel list"
            )

    def test_label_mapping(self, go_shard):
        """Agglomerated label mapping produces values from the labels list."""
        for coords in go_shard.available_chunks[:20]:
            cx, cy, cz = coords
            chunk = go_shard.read_chunk(cx, cy, cz, label_type=LabelType.LABELS)
            raw = go_shard.read_chunk_raw(cx, cy, cz)
            labels = set(raw["labels"])
            voxel_values = set(np.unique(chunk))
            assert voxel_values.issubset(labels), (
                f"Chunk ({cx},{cy},{cz}): mapped values {voxel_values - labels} "
                f"not in labels list"
            )

    def test_nonzero_voxels_exist(self, go_shard):
        """At least some chunks contain non-zero label data (not empty)."""
        nonzero_count = 0
        for coords in go_shard.available_chunks[:50]:
            chunk = go_shard.read_chunk(*coords, label_type=LabelType.SUPERVOXELS)
            if np.any(chunk != 0):
                nonzero_count += 1
        assert nonzero_count > 0, "All sampled chunks were empty (all zeros)"
