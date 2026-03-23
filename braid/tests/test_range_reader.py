"""
Tests for ShardRangeReader — byte-range-read Arrow reader.

Verifies that ShardRangeReader produces identical output to ShardReader
for all chunks in the real test shard, using pre-computed byte offsets.
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
OFFSETS_PATH = TEST_DATA_DIR / "30720_24576_28672-offsets.csv"


@pytest.fixture(scope="module")
def offset_csv():
    """Generate the offset CSV from the test Arrow file if it doesn't exist.

    This runs once per test session. In production, offsets are pre-computed
    by `pixi run compute-offsets` and stored on GCS alongside the Arrow files.
    """
    if OFFSETS_PATH.exists():
        return OFFSETS_PATH

    # Generate offsets from the Arrow file
    arrow_bytes = ARROW_PATH.read_bytes()
    schema_size, offsets = scan_record_offsets(arrow_bytes)

    # Read the CSV index to get (x,y,z) -> rec mapping
    csv_data = CSV_PATH.read_text()
    reader = csv.DictReader(io.StringIO(csv_data))

    out = io.StringIO()
    out.write(f"# schema_size={schema_size}\n")
    writer = csv.writer(out)
    writer.writerow(["x", "y", "z", "rec", "offset", "size"])
    for row in reader:
        rec = int(row["rec"])
        if rec < len(offsets):
            _, off, sz = offsets[rec]
            writer.writerow([row["x"], row["y"], row["z"], rec, off, sz])

    OFFSETS_PATH.write_text(out.getvalue())
    return OFFSETS_PATH


@pytest.fixture(scope="module")
def full_reader():
    """ShardReader that downloads the full Arrow file (baseline)."""
    if not ARROW_PATH.exists():
        pytest.skip("Test shard data not available")
    return ShardReader(str(ARROW_PATH), str(CSV_PATH))


@pytest.fixture(scope="module")
def range_reader(offset_csv):
    """ShardRangeReader using byte-range reads."""
    if not ARROW_PATH.exists():
        pytest.skip("Test shard data not available")
    return ShardRangeReader(
        str(ARROW_PATH), str(CSV_PATH), str(offset_csv))


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
