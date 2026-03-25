"""
Tests for ShardRangeReader — byte-range-read Arrow reader.

Verifies that ShardRangeReader produces identical output to ShardReader
for all chunks in the real test shard, and that batch caching works
correctly for multi-row record batches.
"""

import csv
import io
import sys
from pathlib import Path

import numpy as np
import pytest

# compute_offsets is in the project's scripts/ dir, not a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from braid.reader import ShardReader, ShardRangeReader, LabelType
from scripts.compute_offsets import scan_record_offsets

TEST_DATA_DIR = Path(__file__).parent / "test_data"

ARROW_PATH = TEST_DATA_DIR / "30720_24576_28672.arrow"
CSV_PATH = TEST_DATA_DIR / "30720_24576_28672.csv"
NEW_CSV_PATH = TEST_DATA_DIR / "30720_24576_28672-newformat.csv"


@pytest.fixture(scope="module")
def new_format_csv():
    """Generate a new-format CSV from the old CSV + Arrow file.

    Combines the old (x,y,z,rec) mapping with scanned byte offsets into
    the new (x,y,z,offset,size,batch_idx) format.  Since the old data
    has batch_size=1, batch_idx is always 0.
    """
    if NEW_CSV_PATH.exists():
        return NEW_CSV_PATH

    arrow_bytes = ARROW_PATH.read_bytes()
    schema_size, offsets = scan_record_offsets(arrow_bytes)

    # Read old CSV to get (x,y,z) -> rec mapping
    csv_data = CSV_PATH.read_text()
    reader = csv.DictReader(io.StringIO(csv_data))

    out = io.StringIO()
    out.write(f"# schema_size={schema_size}\n")
    writer = csv.writer(out)
    writer.writerow(["x", "y", "z", "offset", "size", "batch_idx"])
    for row in reader:
        rec = int(row["rec"])
        if rec < len(offsets):
            _, off, sz = offsets[rec]
            writer.writerow([row["x"], row["y"], row["z"], off, sz, 0])

    NEW_CSV_PATH.write_text(out.getvalue())
    return NEW_CSV_PATH


@pytest.fixture(scope="module")
def full_reader():
    """ShardReader that downloads the full Arrow file (baseline)."""
    if not ARROW_PATH.exists():
        pytest.skip("Test shard data not available")
    return ShardReader(str(ARROW_PATH), str(CSV_PATH))


@pytest.fixture(scope="module")
def range_reader(new_format_csv):
    """ShardRangeReader using byte-range reads (new-format CSV)."""
    if not ARROW_PATH.exists():
        pytest.skip("Test shard data not available")
    return ShardRangeReader(str(ARROW_PATH), str(new_format_csv))


class TestRangeReaderInterface:
    """Verify ShardRangeReader has the same interface as ShardReader."""

    def test_chunk_count(self, full_reader, range_reader):
        assert range_reader.chunk_count == full_reader.chunk_count

    def test_available_chunks(self, full_reader, range_reader):
        assert set(range_reader.available_chunks) == set(full_reader.available_chunks)

    def test_has_chunk(self, full_reader, range_reader):
        for cx, cy, cz in list(full_reader.available_chunks)[:10]:
            assert range_reader.has_chunk(cx, cy, cz) == True
        assert range_reader.has_chunk(999999, 999999, 999999) == False

    def test_len(self, full_reader, range_reader):
        assert len(range_reader) == len(full_reader)


class TestRangeReaderParity:
    """Verify ShardRangeReader produces identical output to ShardReader."""

    def test_all_chunks_labels(self, full_reader, range_reader):
        """Every chunk decoded via range read matches full-file read."""
        errors = []
        for cx, cy, cz in full_reader.available_chunks:
            expected = full_reader.read_chunk(cx, cy, cz, LabelType.LABELS)
            actual = range_reader.read_chunk(cx, cy, cz, LabelType.LABELS)
            if not np.array_equal(expected, actual):
                n = np.sum(expected != actual)
                errors.append(f"({cx},{cy},{cz}): {n} voxels differ")
        assert not errors, (
            f"{len(errors)}/{full_reader.chunk_count} chunks differ:\n"
            + "\n".join(errors[:10])
        )

    def test_all_chunks_supervoxels(self, full_reader, range_reader):
        """Supervoxel label type also matches."""
        coords = list(full_reader.available_chunks)[:20]  # sample
        for cx, cy, cz in coords:
            expected = full_reader.read_chunk(cx, cy, cz, LabelType.SUPERVOXELS)
            actual = range_reader.read_chunk(cx, cy, cz, LabelType.SUPERVOXELS)
            np.testing.assert_array_equal(actual, expected)

    def test_read_chunk_raw_parity(self, full_reader, range_reader):
        """Raw chunk data matches between readers."""
        coords = list(full_reader.available_chunks)[:20]  # sample
        for cx, cy, cz in coords:
            expected = full_reader.read_chunk_raw(cx, cy, cz)
            actual = range_reader.read_chunk_raw(cx, cy, cz)
            assert actual["labels"] == expected["labels"]
            assert actual["supervoxels"] == expected["supervoxels"]
            assert actual["compressed_data"] == expected["compressed_data"]
            assert actual["uncompressed_size"] == expected["uncompressed_size"]


class TestRangeReaderErrors:
    """Verify proper error handling."""

    def test_missing_chunk(self, range_reader):
        from braid.exceptions import ChunkNotFoundError
        with pytest.raises(ChunkNotFoundError):
            range_reader.read_chunk(999999, 999999, 999999)

    def test_invalid_coordinates(self, range_reader):
        from braid.exceptions import InvalidCoordinateError
        with pytest.raises(InvalidCoordinateError):
            range_reader.read_chunk(-1, 0, 0)


class TestBatchCaching:
    """Verify batch caching for multi-row record batches."""

    def test_batch_size_1_no_cache_reuse(self, range_reader):
        """With batch_size=1, each chunk has a unique batch — no cache reuse."""
        coords = list(range_reader.available_chunks)[:5]
        range_reader._batch_fetches = 0
        for cx, cy, cz in coords:
            range_reader.read_chunk_raw(cx, cy, cz)
        assert range_reader._batch_fetches == len(coords)

    def test_batch_size_gt1_cache_reuse(self, temp_shard_files_batched):
        """With batch_size=2, two chunks share a batch — second is cached."""
        arrow_path, csv_path = temp_shard_files_batched
        reader = ShardRangeReader(str(arrow_path), str(csv_path))

        # Both chunks should be in the same batch
        coords = reader.available_chunks
        assert len(coords) == 2

        reader._batch_fetches = 0
        reader.read_chunk_raw(*coords[0])
        assert reader._batch_fetches == 1

        reader.read_chunk_raw(*coords[1])
        assert reader._batch_fetches == 1  # cache hit — no new fetch

    def test_batched_read_matches_full_reader(self, temp_shard_files_batched):
        """Batched ShardRangeReader produces same output as ShardReader."""
        arrow_path, csv_path = temp_shard_files_batched

        full = ShardReader(str(arrow_path), str(csv_path))
        rng = ShardRangeReader(str(arrow_path), str(csv_path))

        for cx, cy, cz in full.available_chunks:
            expected = full.read_chunk_raw(cx, cy, cz)
            actual = rng.read_chunk_raw(cx, cy, cz)
            assert actual["labels"] == expected["labels"]
            assert actual["supervoxels"] == expected["supervoxels"]
            assert actual["compressed_data"] == expected["compressed_data"]
