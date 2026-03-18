"""
End-to-end test: Arrow IPC shard files -> neuroglancer precomputed volume.

Creates synthetic DVID shard files with known uint64 label regions, converts
them to a neuroglancer precomputed volume via TensorStore, and reads back
the volume to verify correctness.
"""

import csv
import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import zstandard as zstd

ts = pytest.importorskip("tensorstore")

from braid import ShardReader, LabelType


# ---- Helpers for creating synthetic DVID compressed blocks ----

def create_solid_dvid_block(label: int) -> bytes:
    """Create a DVID compressed block where all voxels have the same label."""
    # Header: gx=8, gy=8, gz=8, numLabels=1
    header = struct.pack('<IIII', 8, 8, 8, 1)
    label_data = struct.pack('<Q', label)
    return header + label_data


def create_two_label_dvid_block(label_a: int, label_b: int) -> bytes:
    """
    Create a DVID compressed block split along Z.

    Bottom half (z < 32) gets label_a, top half (z >= 32) gets label_b.
    Each 8x8x8 sub-block is solid (single label), so no packed voxel indices.
    """
    gx, gy, gz = 8, 8, 8
    num_labels = 2

    # Header
    header = struct.pack('<IIII', gx, gy, gz, num_labels)

    # Label table
    labels_data = struct.pack('<QQ', label_a, label_b)

    # NumSBLabels: each sub-block has exactly 1 label
    num_sub_blocks = gx * gy * gz  # 512
    num_sb_labels = struct.pack('<' + 'H' * num_sub_blocks, *([1] * num_sub_blocks))

    # SBIndices: sub-blocks iterate in sz, sy, sx order.
    # sz < 4 (bottom half) -> index 0 (label_a), sz >= 4 -> index 1 (label_b)
    sb_indices = []
    for sz in range(gz):
        for sy in range(gy):
            for sx in range(gx):
                sb_indices.append(0 if sz < gz // 2 else 1)
    sb_indices_data = struct.pack('<' + 'I' * num_sub_blocks, *sb_indices)

    # No packed voxel values needed (num_sb_labels_cur == 1 for all sub-blocks)
    return header + labels_data + num_sb_labels + sb_indices_data


def create_test_shard(
    tmp_dir: Path,
    name: str,
    chunks: list,
) -> tuple:
    """
    Create an Arrow IPC shard file and CSV index from a list of chunk specs.

    Args:
        tmp_dir: Directory to write files into
        name: Base name for the shard files
        chunks: List of dicts with keys:
            x, y, z: chunk coordinates
            labels: list of uint64 agglomerated label IDs
            supervoxels: list of uint64 supervoxel IDs
            dvid_block: uncompressed DVID block bytes

    Returns:
        (arrow_path, csv_path)
    """
    compressor = zstd.ZstdCompressor()

    schema = pa.schema([
        pa.field('chunk_x', pa.int32(), nullable=False),
        pa.field('chunk_y', pa.int32(), nullable=False),
        pa.field('chunk_z', pa.int32(), nullable=False),
        pa.field('labels', pa.list_(pa.uint64()), nullable=False),
        pa.field('supervoxels', pa.list_(pa.uint64()), nullable=False),
        pa.field('dvid_compressed_block', pa.binary(), nullable=False),
        pa.field('uncompressed_size', pa.uint32(), nullable=False),
    ])

    arrow_path = tmp_dir / f"{name}.arrow"
    csv_path = tmp_dir / f"{name}.csv"

    # Build Arrow table
    data = {
        'chunk_x': [],
        'chunk_y': [],
        'chunk_z': [],
        'labels': [],
        'supervoxels': [],
        'dvid_compressed_block': [],
        'uncompressed_size': [],
    }

    for c in chunks:
        compressed = compressor.compress(c['dvid_block'])
        data['chunk_x'].append(c['x'])
        data['chunk_y'].append(c['y'])
        data['chunk_z'].append(c['z'])
        data['labels'].append(c['labels'])
        data['supervoxels'].append(c['supervoxels'])
        data['dvid_compressed_block'].append(compressed)
        data['uncompressed_size'].append(len(c['dvid_block']))

    table = pa.table(data, schema=schema)

    # Write Arrow IPC file
    with open(arrow_path, 'wb') as f:
        with ipc.new_file(f, schema) as writer:
            writer.write_table(table)

    # Write CSV index
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y', 'z', 'rec'])
        for rec, c in enumerate(chunks):
            w.writerow([c['x'], c['y'], c['z'], rec])

    return arrow_path, csv_path


# ---- Fixtures ----

@pytest.fixture
def volume_128_shard(tmp_path):
    """
    Create a 128x128x128 test volume (2x2x2 chunks of 64^3) as a single shard.

    Label layout (using supervoxel IDs):
        chunk (0,0,0): solid label 100
        chunk (1,0,0): solid label 200
        chunk (0,1,0): solid label 300
        chunk (1,1,0): solid label 400
        chunk (0,0,1): two labels - bottom-Z=500, top-Z=600
        chunk (1,0,1): solid label 700
        chunk (0,1,1): solid label 800
        chunk (1,1,1): solid label 900
    """
    chunks = [
        # Layer z=0 (chunk z coordinate 0)
        {
            'x': 0, 'y': 0, 'z': 0,
            'supervoxels': [100],
            'labels': [1100],  # agglomerated IDs
            'dvid_block': create_solid_dvid_block(100),
        },
        {
            'x': 1, 'y': 0, 'z': 0,
            'supervoxels': [200],
            'labels': [1200],
            'dvid_block': create_solid_dvid_block(200),
        },
        {
            'x': 0, 'y': 1, 'z': 0,
            'supervoxels': [300],
            'labels': [1300],
            'dvid_block': create_solid_dvid_block(300),
        },
        {
            'x': 1, 'y': 1, 'z': 0,
            'supervoxels': [400],
            'labels': [1400],
            'dvid_block': create_solid_dvid_block(400),
        },
        # Layer z=1 (chunk z coordinate 1)
        {
            'x': 0, 'y': 0, 'z': 1,
            'supervoxels': [500, 600],
            'labels': [1500, 1600],
            'dvid_block': create_two_label_dvid_block(500, 600),
        },
        {
            'x': 1, 'y': 0, 'z': 1,
            'supervoxels': [700],
            'labels': [1700],
            'dvid_block': create_solid_dvid_block(700),
        },
        {
            'x': 0, 'y': 1, 'z': 1,
            'supervoxels': [800],
            'labels': [1800],
            'dvid_block': create_solid_dvid_block(800),
        },
        {
            'x': 1, 'y': 1, 'z': 1,
            'supervoxels': [900],
            'labels': [1900],
            'dvid_block': create_solid_dvid_block(900),
        },
    ]

    arrow_path, csv_path = create_test_shard(tmp_path, "test_shard", chunks)
    return arrow_path, csv_path, chunks


# ---- Tests ----

class TestBraidReadback:
    """Verify BRAID correctly reads back the synthetic shard data."""

    def test_solid_block_supervoxels(self, volume_128_shard):
        """Solid block reads back correct supervoxel IDs."""
        arrow_path, csv_path, _ = volume_128_shard
        reader = ShardReader(arrow_path, csv_path)

        chunk = reader.read_chunk(0, 0, 0, label_type=LabelType.SUPERVOXELS)
        assert chunk.shape == (64, 64, 64)
        assert chunk.dtype == np.uint64
        assert np.all(chunk == 100)

    def test_solid_block_labels(self, volume_128_shard):
        """Solid block reads back correct agglomerated label IDs."""
        arrow_path, csv_path, _ = volume_128_shard
        reader = ShardReader(arrow_path, csv_path)

        chunk = reader.read_chunk(0, 0, 0, label_type=LabelType.LABELS)
        assert np.all(chunk == 1100)

    def test_two_label_block_supervoxels(self, volume_128_shard):
        """Two-label block reads back correct spatial label pattern."""
        arrow_path, csv_path, _ = volume_128_shard
        reader = ShardReader(arrow_path, csv_path)

        chunk = reader.read_chunk(0, 0, 1, label_type=LabelType.SUPERVOXELS)
        assert chunk.shape == (64, 64, 64)

        # Bottom half of Z (z < 32) should be 500
        assert np.all(chunk[:32, :, :] == 500)
        # Top half of Z (z >= 32) should be 600
        assert np.all(chunk[32:, :, :] == 600)

    def test_two_label_block_labels(self, volume_128_shard):
        """Two-label block maps supervoxels to agglomerated labels correctly."""
        arrow_path, csv_path, _ = volume_128_shard
        reader = ShardReader(arrow_path, csv_path)

        chunk = reader.read_chunk(0, 0, 1, label_type=LabelType.LABELS)
        assert np.all(chunk[:32, :, :] == 1500)
        assert np.all(chunk[32:, :, :] == 1600)

    def test_all_chunks_present(self, volume_128_shard):
        """All 8 chunks are present in the shard."""
        arrow_path, csv_path, _ = volume_128_shard
        reader = ShardReader(arrow_path, csv_path)
        assert reader.chunk_count == 8


class TestE2EPrecomputed:
    """
    End-to-end: Arrow IPC -> BRAID -> TensorStore -> neuroglancer precomputed -> readback.
    """

    def _write_info_file(self, tmp_dir: str, volume_shape):
        """Write a neuroglancer precomputed info file with the correct chunk sizes."""
        info = {
            '@type': 'neuroglancer_multiscale_volume',
            'data_type': 'uint64',
            'num_channels': 1,
            'type': 'segmentation',
            'scales': [{
                'key': 's0',
                'size': list(volume_shape),  # [x, y, z] for neuroglancer
                'resolution': [8, 8, 8],
                'chunk_sizes': [[64, 64, 64]],
                'encoding': 'raw',
                'voxel_offset': [0, 0, 0],
            }],
        }
        with open(os.path.join(tmp_dir, 'info'), 'w') as f:
            json.dump(info, f)

    def _open_precomputed_volume(self, tmp_dir: str):
        """Open a local neuroglancer precomputed volume."""
        spec = {
            'driver': 'neuroglancer_precomputed',
            'kvstore': {'driver': 'file', 'path': tmp_dir},
            'scale_index': 0,
            'open': True,
        }
        return ts.open(spec).result()

    def _write_precomputed_volume(self, tmp_dir: str, reader: ShardReader, volume_shape, label_type: LabelType):
        """Write all chunks from a BRAID reader to a local neuroglancer precomputed volume."""
        self._write_info_file(tmp_dir, volume_shape)
        dest = self._open_precomputed_volume(tmp_dir)

        # Write each chunk
        for (cx, cy, cz) in reader.available_chunks:
            chunk_data = reader.read_chunk(cx, cy, cz, label_type=label_type)
            # chunk_data is (64,64,64) in ZYX order
            # neuroglancer precomputed domain is [x, y, z, channel]
            transposed = chunk_data.transpose(2, 1, 0)  # ZYX -> XYZ

            x0 = cx * 64
            y0 = cy * 64
            z0 = cz * 64
            dest[x0:x0+64, y0:y0+64, z0:z0+64, 0].write(transposed).result()

    def _read_precomputed_volume(self, tmp_dir: str):
        """Open a local neuroglancer precomputed volume for reading (without channel dim)."""
        vol = self._open_precomputed_volume(tmp_dir)
        # Index out the channel dimension for simpler assertions
        return vol[:, :, :, 0]

    def test_solid_blocks_roundtrip(self, volume_128_shard):
        """Solid-label chunks survive the full Arrow -> precomputed -> readback pipeline."""
        arrow_path, csv_path, _ = volume_128_shard
        reader = ShardReader(arrow_path, csv_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # neuroglancer precomputed size is [x, y, z]
            self._write_precomputed_volume(tmp_dir, reader, [128, 128, 128], LabelType.SUPERVOXELS)
            vol = self._read_precomputed_volume(tmp_dir)

            # Read back chunk (0,0,0) — should be solid 100
            data_000 = vol[0:64, 0:64, 0:64].read().result()
            assert np.all(data_000 == 100), f"Expected 100, got unique values {np.unique(data_000)}"

            # Read back chunk (1,0,0) — should be solid 200
            data_100 = vol[64:128, 0:64, 0:64].read().result()
            assert np.all(data_100 == 200), f"Expected 200, got unique values {np.unique(data_100)}"

            # Read back chunk (0,1,0) — should be solid 300
            data_010 = vol[0:64, 64:128, 0:64].read().result()
            assert np.all(data_010 == 300), f"Expected 300, got unique values {np.unique(data_010)}"

            # Read back chunk (1,1,0) — should be solid 400
            data_110 = vol[64:128, 64:128, 0:64].read().result()
            assert np.all(data_110 == 400), f"Expected 400, got unique values {np.unique(data_110)}"

    def test_two_label_block_roundtrip(self, volume_128_shard):
        """Two-label block maintains spatial label pattern through the full pipeline."""
        arrow_path, csv_path, _ = volume_128_shard
        reader = ShardReader(arrow_path, csv_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_precomputed_volume(tmp_dir, reader, [128, 128, 128], LabelType.SUPERVOXELS)
            vol = self._read_precomputed_volume(tmp_dir)

            # Chunk (0,0,1) has label_a=500 in bottom Z, label_b=600 in top Z
            # In neuroglancer precomputed [x, y, z] indexing:
            # z=64..95 (bottom half of chunk) should be 500
            # z=96..127 (top half of chunk) should be 600
            data_001 = vol[0:64, 0:64, 64:128].read().result()

            bottom = data_001[:, :, :32]  # z offsets 0..31 within chunk (z=64..95 in volume)
            top = data_001[:, :, 32:]     # z offsets 32..63 within chunk (z=96..127 in volume)

            assert np.all(bottom == 500), (
                f"Bottom Z half expected 500, got unique values {np.unique(bottom)}"
            )
            assert np.all(top == 600), (
                f"Top Z half expected 600, got unique values {np.unique(top)}"
            )

    def test_agglomerated_labels_roundtrip(self, volume_128_shard):
        """Agglomerated label mapping is applied correctly through the full pipeline."""
        arrow_path, csv_path, _ = volume_128_shard
        reader = ShardReader(arrow_path, csv_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_precomputed_volume(tmp_dir, reader, [128, 128, 128], LabelType.LABELS)
            vol = self._read_precomputed_volume(tmp_dir)

            # Chunk (0,0,0) with agglomerated labels should be 1100
            data_000 = vol[0:64, 0:64, 0:64].read().result()
            assert np.all(data_000 == 1100), f"Expected 1100, got unique values {np.unique(data_000)}"

            # Chunk (1,1,1) with agglomerated labels should be 1900
            data_111 = vol[64:128, 64:128, 64:128].read().result()
            assert np.all(data_111 == 1900), f"Expected 1900, got unique values {np.unique(data_111)}"

    def test_two_label_agglomerated_roundtrip(self, volume_128_shard):
        """Two-label block with agglomerated mapping through the full pipeline."""
        arrow_path, csv_path, _ = volume_128_shard
        reader = ShardReader(arrow_path, csv_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_precomputed_volume(tmp_dir, reader, [128, 128, 128], LabelType.LABELS)
            vol = self._read_precomputed_volume(tmp_dir)

            data_001 = vol[0:64, 0:64, 64:128].read().result()
            bottom = data_001[:, :, :32]
            top = data_001[:, :, 32:]

            assert np.all(bottom == 1500), f"Expected 1500, got unique values {np.unique(bottom)}"
            assert np.all(top == 1600), f"Expected 1600, got unique values {np.unique(top)}"

    def test_full_volume_integrity(self, volume_128_shard):
        """Read the entire volume at once and verify all regions."""
        arrow_path, csv_path, _ = volume_128_shard
        reader = ShardReader(arrow_path, csv_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_precomputed_volume(tmp_dir, reader, [128, 128, 128], LabelType.SUPERVOXELS)
            vol = self._read_precomputed_volume(tmp_dir)

            # Read entire volume
            full = vol[0:128, 0:128, 0:128].read().result()
            assert full.shape == (128, 128, 128)
            assert full.dtype == np.uint64

            # Spot check each chunk's center voxel
            expected = {
                (32, 32, 32): 100,    # chunk (0,0,0)
                (96, 32, 32): 200,    # chunk (1,0,0)
                (32, 96, 32): 300,    # chunk (0,1,0)
                (96, 96, 32): 400,    # chunk (1,1,0)
                (32, 32, 80): 500,    # chunk (0,0,1) bottom-Z half
                (32, 32, 112): 600,   # chunk (0,0,1) top-Z half
                (96, 32, 96): 700,    # chunk (1,0,1)
                (32, 96, 96): 800,    # chunk (0,1,1)
                (96, 96, 96): 900,    # chunk (1,1,1)
            }
            for (x, y, z), expected_label in expected.items():
                actual = int(full[x, y, z])
                assert actual == expected_label, (
                    f"At ({x},{y},{z}): expected {expected_label}, got {actual}"
                )
