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


@pytest.fixture
def volume_zdoubled_shard(tmp_path):
    """
    Create a 128x128x256 test volume (2x2x4 chunks) simulating Z-doubled data.

    This models DVID export shards at [16,16,15] nm where the original [16,16,30]
    segmentation was doubled in Z.  Source chunks come in Z pairs:

        z=0, z=1: same spatial data (Z-doubled pair A)
        z=2, z=3: same spatial data (Z-doubled pair B)

    Within each block, Z slices are duplicated to simulate the doubling:
    original Z slice i appears at both z=2i and z=2i+1 within the 64-voxel block.

    With --z-compress 1 (stride=2), the output should be a 128x128x128 volume
    where each output chunk [0,64) in Z is populated from the pair and matches
    the un-doubled original.
    """
    def make_zdoubled_solid_block(label):
        """Solid block — Z-doubling is a no-op since all voxels are identical."""
        return create_solid_dvid_block(label)

    def make_zdoubled_two_label_block(label_a, label_b):
        """Two-label block with Z-doubled slices.

        Original (un-doubled): label_a for z<32, label_b for z>=32.
        Z-doubled: pairs of identical slices, so the sub-block boundary
        is the same (z < gz//2 → label_a, z >= gz//2 → label_b).
        After z-compress 1 (keep z=0,2,4,...), we recover the original
        32-slice-each split, but mapped to 32 output voxels: z<16 → label_a,
        z>=16 → label_b.
        """
        return create_two_label_dvid_block(label_a, label_b)

    # Z pair A: chunks z=0 and z=1 (identical data)
    pair_a_labels = [
        (0, 0, 100, [100], [100]),    # solid
        (1, 0, 200, [200], [200]),    # solid
        (0, 1, 300, [300], [300]),    # solid
        (1, 1, 400, [400], [400]),    # solid
    ]
    # Z pair B: chunks z=2 and z=3 (identical data)
    pair_b_labels = [
        (0, 0, None, [500, 600], [500, 600]),  # two-label
        (1, 0, 700, [700], [700]),              # solid
        (0, 1, 800, [800], [800]),              # solid
        (1, 1, 900, [900], [900]),              # solid
    ]

    chunks = []
    for z_src in (0, 1):  # pair A
        for cx, cy, label, svs, labs in pair_a_labels:
            chunks.append({
                'x': cx, 'y': cy, 'z': z_src,
                'supervoxels': svs, 'labels': labs,
                'dvid_block': make_zdoubled_solid_block(label),
            })
    for z_src in (2, 3):  # pair B
        for cx, cy, label, svs, labs in pair_b_labels:
            if label is None:
                block = make_zdoubled_two_label_block(svs[0], svs[1])
            else:
                block = make_zdoubled_solid_block(label)
            chunks.append({
                'x': cx, 'y': cy, 'z': z_src,
                'supervoxels': svs, 'labels': labs,
                'dvid_block': block,
            })

    arrow_path, csv_path = create_test_shard(tmp_path, "zdoubled_shard", chunks)
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


class TestE2EZCompress:
    """
    End-to-end with Z compression: Z-doubled source -> z_compress=1 -> readback.

    Verifies the worker's Z decimation and coordinate mapping logic by writing
    a Z-doubled 128x128x256 source volume (4 Z-chunks) through a z-compress=1
    pipeline into a 128x128x128 output volume.  Each source block's 64 Z voxels
    are decimated to 32, and pairs of adjacent source Z-chunks merge into one
    output chunk via TensorStore transactions.
    """

    CHUNK = 64
    Z_STRIDE = 2  # z_compress=1 → stride 2
    OUT_Z_PER_CHUNK = CHUNK // Z_STRIDE  # 32

    def _write_info_file(self, tmp_dir: str, volume_shape):
        info = {
            '@type': 'neuroglancer_multiscale_volume',
            'data_type': 'uint64',
            'num_channels': 1,
            'type': 'segmentation',
            'scales': [{
                'key': 's0',
                'size': list(volume_shape),
                'resolution': [8, 8, 8],
                'chunk_sizes': [[64, 64, 64]],
                'encoding': 'raw',
                'voxel_offset': [0, 0, 0],
            }],
        }
        with open(os.path.join(tmp_dir, 'info'), 'w') as f:
            json.dump(info, f)

    def _open_precomputed_volume(self, tmp_dir: str):
        spec = {
            'driver': 'neuroglancer_precomputed',
            'kvstore': {'driver': 'file', 'path': tmp_dir},
            'scale_index': 0,
            'open': True,
        }
        return ts.open(spec).result()

    def _write_z_compressed(self, tmp_dir, reader, output_shape, label_type):
        """Write all chunks with Z decimation, mirroring the worker's logic."""
        self._write_info_file(tmp_dir, output_shape)
        dest = self._open_precomputed_volume(tmp_dir)

        C = self.CHUNK
        out_z = self.OUT_Z_PER_CHUNK
        z_stride = self.Z_STRIDE

        txn = ts.Transaction()
        for (cx, cy, cz) in reader.available_chunks:
            chunk_data = reader.read_chunk(cx, cy, cz, label_type=label_type)
            # Decimate Z (BRAID returns ZYX order)
            chunk_data = chunk_data[::z_stride, :, :]
            transposed = chunk_data.transpose(2, 1, 0)  # ZYX -> XYZ

            x0, y0, z0 = cx * C, cy * C, cz * out_z
            x1 = min(x0 + C, output_shape[0])
            y1 = min(y0 + C, output_shape[1])
            z1 = min(z0 + out_z, output_shape[2])
            clipped = transposed[:x1 - x0, :y1 - y0, :z1 - z0]
            dest.with_transaction(txn)[x0:x1, y0:y1, z0:z1, 0].write(
                clipped).result()

        txn.commit_async().result()

    def _read_volume(self, tmp_dir):
        vol = self._open_precomputed_volume(tmp_dir)
        return vol[:, :, :, 0]

    def test_solid_blocks_z_compress(self, volume_zdoubled_shard):
        """Solid-label Z-doubled pairs produce correct output after z-compress."""
        arrow_path, csv_path, _ = volume_zdoubled_shard
        reader = ShardReader(arrow_path, csv_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_z_compressed(
                tmp_dir, reader, [128, 128, 128], LabelType.SUPERVOXELS)
            vol = self._read_volume(tmp_dir)

            # Source z=0 → output Z [0,32), source z=1 → output Z [32,64)
            # Both have the same labels, so the full output chunk is uniform.
            data_000 = vol[0:64, 0:64, 0:64].read().result()
            assert np.all(data_000 == 100), (
                f"Expected 100, got unique: {np.unique(data_000)}")

            data_100 = vol[64:128, 0:64, 0:64].read().result()
            assert np.all(data_100 == 200)

            data_010 = vol[0:64, 64:128, 0:64].read().result()
            assert np.all(data_010 == 300)

            data_110 = vol[64:128, 64:128, 0:64].read().result()
            assert np.all(data_110 == 400)

    def test_two_label_block_z_compress(self, volume_zdoubled_shard):
        """Two-label block Z pattern is preserved through z-compress pipeline."""
        arrow_path, csv_path, _ = volume_zdoubled_shard
        reader = ShardReader(arrow_path, csv_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_z_compressed(
                tmp_dir, reader, [128, 128, 128], LabelType.SUPERVOXELS)
            vol = self._read_volume(tmp_dir)

            # Chunk (0,0) in XY, Z pair B (source z=2,3 → output Z [64,128)):
            # Source block has label_a=500 in bottom Z half, label_b=600 in top.
            # After z-compress, each source block contributes 32 output Z voxels,
            # with the label split at the midpoint (16 voxels each).
            #
            # Source z=2 → output Z [64,96):  z<80 → 500, z>=80 → 600
            # Source z=3 → output Z [96,128): z<112 → 500, z>=112 → 600
            data = vol[0:64, 0:64, 64:128].read().result()

            assert np.all(data[:, :, 0:16] == 500), "z=2 bottom half"
            assert np.all(data[:, :, 16:32] == 600), "z=2 top half"
            assert np.all(data[:, :, 32:48] == 500), "z=3 bottom half"
            assert np.all(data[:, :, 48:64] == 600), "z=3 top half"

    def test_full_volume_z_compress(self, volume_zdoubled_shard):
        """Spot-check the entire Z-compressed output volume."""
        arrow_path, csv_path, _ = volume_zdoubled_shard
        reader = ShardReader(arrow_path, csv_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_z_compressed(
                tmp_dir, reader, [128, 128, 128], LabelType.SUPERVOXELS)
            vol = self._read_volume(tmp_dir)

            full = vol[0:128, 0:128, 0:128].read().result()
            assert full.shape == (128, 128, 128)

            # Each solid source block pair merges into one uniform output chunk.
            expected = {
                # Pair A: output Z [0,64) — solid labels
                (32, 32, 16): 100,    # from source z=0, center of 32-voxel slab
                (32, 32, 48): 100,    # from source z=1, center of 32-voxel slab
                (96, 32, 16): 200,
                (32, 96, 16): 300,
                (96, 96, 48): 400,
                # Pair B: output Z [64,128)
                (96, 32, 80): 700,    # from source z=2
                (96, 32, 112): 700,   # from source z=3
                (32, 96, 80): 800,
                (96, 96, 112): 900,
                # Two-label block: z-compress preserves within-block split
                (32, 32, 72): 500,    # z=2 bottom half
                (32, 32, 88): 600,    # z=2 top half
                (32, 32, 104): 500,   # z=3 bottom half
                (32, 32, 120): 600,   # z=3 top half
            }
            for (x, y, z), label in expected.items():
                actual = int(full[x, y, z])
                assert actual == label, (
                    f"At ({x},{y},{z}): expected {label}, got {actual}"
                )
