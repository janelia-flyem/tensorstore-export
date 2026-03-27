"""Tests for the per-scale TensorStore handle reuse pattern.

Exercises the core fix: multiple DVID shards writing through a single
pre-opened TensorStore handle into a shared staging directory, then
reading back to verify all shards produced output.
"""

import csv
import json
import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import zstandard as zstd

ts = pytest.importorskip("tensorstore")

from src.worker import (
    ShardProcessor,
    WorkerConfig,
    BATCH_SIZE,
    CHUNK_VOXELS,
)
from braid import LabelType


# ---- Helpers (adapted from test_e2e_precomputed.py) ----

def _create_solid_dvid_block(label: int) -> bytes:
    """Create a DVID compressed block where all voxels have the same label."""
    header = struct.pack('<IIII', 8, 8, 8, 1)
    label_data = struct.pack('<Q', label)
    return header + label_data


def _create_shard(tmp_dir: Path, name: str, chunks: list) -> tuple:
    """Create an Arrow IPC shard file + CSV index from chunk specs.

    Each chunk spec is a dict with keys: x, y, z, labels, supervoxels, dvid_block.
    Returns (arrow_path, csv_path).
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

    data = {k: [] for k in [
        'chunk_x', 'chunk_y', 'chunk_z', 'labels', 'supervoxels',
        'dvid_compressed_block', 'uncompressed_size',
    ]}

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

    arrow_path = tmp_dir / f"{name}.arrow"
    csv_path = tmp_dir / f"{name}.csv"

    with open(arrow_path, 'wb') as f:
        with ipc.new_file(f, schema) as writer:
            writer.write_table(table)

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y', 'z', 'rec'])
        for rec, c in enumerate(chunks):
            w.writerow([c['x'], c['y'], c['z'], rec])

    return arrow_path, csv_path


def _write_info_file(staging_dir: str, volume_shape: list):
    """Write a neuroglancer precomputed info file with compressed_segmentation + sharding.

    Matches the production encoding used by the real export pipeline.
    """
    info = {
        '@type': 'neuroglancer_multiscale_volume',
        'data_type': 'uint64',
        'num_channels': 1,
        'type': 'segmentation',
        'scales': [{
            'key': 's0',
            'size': volume_shape,
            'resolution': [8, 8, 8],
            'chunk_sizes': [[64, 64, 64]],
            'encoding': 'compressed_segmentation',
            'compressed_segmentation_block_size': [8, 8, 8],
            'voxel_offset': [0, 0, 0],
            'sharding': {
                '@type': 'neuroglancer_uint64_sharded_v1',
                'hash': 'identity',
                'preshift_bits': 9,
                'minishard_bits': 6,
                'shard_bits': 19,
                'minishard_index_encoding': 'gzip',
                'data_encoding': 'gzip',
            },
        }],
    }
    with open(os.path.join(staging_dir, 'info'), 'w') as f:
        json.dump(info, f)
    return info


def _open_local_volume(staging_dir: str, scale_index: int = 0):
    """Open a local neuroglancer precomputed TensorStore handle."""
    return ts.open({
        'driver': 'neuroglancer_precomputed',
        'kvstore': {'driver': 'file', 'path': staging_dir},
        'scale_index': scale_index,
        'open': True,
    }).result()


# ---- Fixtures ----

@pytest.fixture
def shard_source_dir(tmp_path):
    """Create two separate DVID shards that map to different spatial regions.

    Shard A covers chunks at (0,0,0) and (1,0,0) — labels 100, 200
    Shard B covers chunks at (0,1,0) and (1,1,0) — labels 300, 400

    Both shards should write to the same NG output shard file since they're
    in the same 128x128x128 region.
    """
    source_dir = tmp_path / "source" / "s0"
    source_dir.mkdir(parents=True)

    shard_a_chunks = [
        {'x': 0, 'y': 0, 'z': 0, 'supervoxels': [100], 'labels': [100],
         'dvid_block': _create_solid_dvid_block(100)},
        {'x': 1, 'y': 0, 'z': 0, 'supervoxels': [200], 'labels': [200],
         'dvid_block': _create_solid_dvid_block(200)},
    ]
    _create_shard(source_dir, "0_0_0", shard_a_chunks)

    shard_b_chunks = [
        {'x': 0, 'y': 1, 'z': 0, 'supervoxels': [300], 'labels': [300],
         'dvid_block': _create_solid_dvid_block(300)},
        {'x': 1, 'y': 1, 'z': 0, 'supervoxels': [400], 'labels': [400],
         'dvid_block': _create_solid_dvid_block(400)},
    ]
    _create_shard(source_dir, "0_64_0", shard_b_chunks)

    return tmp_path / "source"


# ---- Tests ----

class TestSharedHandleMultiShard:
    """Verify that multiple DVID shards writing through one TensorStore handle
    all produce correct output — the core scenario that was broken before."""

    def test_two_shards_one_handle(self, tmp_path, shard_source_dir):
        """Two shards written through the same handle both produce data."""
        staging_dir = str(tmp_path / "staging")
        os.makedirs(staging_dir)
        info = _write_info_file(staging_dir, [128, 128, 128])

        dest = _open_local_volume(staging_dir)

        # Process shard A
        from braid import ShardReader
        reader_a = ShardReader(
            str(shard_source_dir / "s0" / "0_0_0.arrow"),
            str(shard_source_dir / "s0" / "0_0_0.csv"),
        )
        txn = ts.Transaction()
        for cx, cy, cz in reader_a.available_chunks:
            chunk = reader_a.read_chunk(cx, cy, cz, label_type=LabelType.SUPERVOXELS)
            transposed = chunk.transpose(2, 1, 0)
            x0, y0, z0 = cx * 64, cy * 64, cz * 64
            dest.with_transaction(txn)[x0:x0+64, y0:y0+64, z0:z0+64, 0].write(
                transposed).result()
        txn.commit_async().result()

        # Process shard B through the SAME handle
        reader_b = ShardReader(
            str(shard_source_dir / "s0" / "0_64_0.arrow"),
            str(shard_source_dir / "s0" / "0_64_0.csv"),
        )
        txn = ts.Transaction()
        for cx, cy, cz in reader_b.available_chunks:
            chunk = reader_b.read_chunk(cx, cy, cz, label_type=LabelType.SUPERVOXELS)
            transposed = chunk.transpose(2, 1, 0)
            x0, y0, z0 = cx * 64, cy * 64, cz * 64
            dest.with_transaction(txn)[x0:x0+64, y0:y0+64, z0:z0+64, 0].write(
                transposed).result()
        txn.commit_async().result()

        # Read back: all four chunks should have correct data
        vol = dest[:, :, :, 0]
        assert np.all(vol[0:64, 0:64, 0:64].read().result() == 100)
        assert np.all(vol[64:128, 0:64, 0:64].read().result() == 200)
        assert np.all(vol[0:64, 64:128, 0:64].read().result() == 300)
        assert np.all(vol[64:128, 64:128, 0:64].read().result() == 400)

    def test_shard_files_accumulate_on_disk(self, tmp_path, shard_source_dir):
        """Shard files accumulate in the shared staging dir across commits."""
        staging_dir = str(tmp_path / "staging")
        os.makedirs(staging_dir)
        _write_info_file(staging_dir, [128, 128, 128])

        dest = _open_local_volume(staging_dir)

        from braid import ShardReader
        for shard_name in ["0_0_0", "0_64_0"]:
            reader = ShardReader(
                str(shard_source_dir / "s0" / f"{shard_name}.arrow"),
                str(shard_source_dir / "s0" / f"{shard_name}.csv"),
            )
            txn = ts.Transaction()
            for cx, cy, cz in reader.available_chunks:
                chunk = reader.read_chunk(cx, cy, cz, label_type=LabelType.SUPERVOXELS)
                transposed = chunk.transpose(2, 1, 0)
                x0, y0, z0 = cx * 64, cy * 64, cz * 64
                dest.with_transaction(txn)[x0:x0+64, y0:y0+64, z0:z0+64, 0].write(
                    transposed).result()
            txn.commit_async().result()

        # With compressed_segmentation + sharding, TensorStore writes .shard files.
        shard_files = []
        for root, _dirs, files in os.walk(staging_dir):
            for fn in files:
                if fn.endswith(".shard"):
                    fp = os.path.join(root, fn)
                    size = os.path.getsize(fp)
                    shard_files.append((fn, size))
                    assert size > 0, f"Shard file {fn} is empty"
        assert len(shard_files) >= 1, "No .shard files produced in staging dir"

    def test_many_sequential_shards(self, tmp_path):
        """Process 8 shards sequentially through one handle (exceeds the
        threshold where the old per-shard-open pattern would fail)."""
        source_dir = tmp_path / "source" / "s0"
        source_dir.mkdir(parents=True)

        # Create 8 shards, each with 1 chunk at a different position
        for i in range(8):
            cx, cy, cz = i % 4, (i // 4) % 4, 0
            label = (i + 1) * 100
            chunks = [{
                'x': cx, 'y': cy, 'z': cz,
                'supervoxels': [label], 'labels': [label],
                'dvid_block': _create_solid_dvid_block(label),
            }]
            _create_shard(source_dir, f"{cx*64}_{cy*64}_{cz*64}", chunks)

        staging_dir = str(tmp_path / "staging")
        os.makedirs(staging_dir)
        _write_info_file(staging_dir, [256, 256, 64])
        dest = _open_local_volume(staging_dir)

        from braid import ShardReader
        for i in range(8):
            cx, cy, cz = i % 4, (i // 4) % 4, 0
            name = f"{cx*64}_{cy*64}_{cz*64}"
            reader = ShardReader(
                str(source_dir / f"{name}.arrow"),
                str(source_dir / f"{name}.csv"),
            )
            txn = ts.Transaction()
            for ccx, ccy, ccz in reader.available_chunks:
                chunk = reader.read_chunk(ccx, ccy, ccz, label_type=LabelType.SUPERVOXELS)
                transposed = chunk.transpose(2, 1, 0)
                x0, y0, z0 = ccx * 64, ccy * 64, ccz * 64
                dest.with_transaction(txn)[x0:x0+64, y0:y0+64, z0:z0+64, 0].write(
                    transposed).result()
            txn.commit_async().result()

        # Read back all 8 chunks
        vol = dest[:, :, :, 0]
        for i in range(8):
            cx, cy, cz = i % 4, (i // 4) % 4, 0
            label = (i + 1) * 100
            x0, y0, z0 = cx * 64, cy * 64, cz * 64
            data = vol[x0:x0+64, y0:y0+64, z0:z0+64].read().result()
            assert np.all(data == label), (
                f"Shard {i} (chunk {cx},{cy},{cz}): expected {label}, "
                f"got unique {np.unique(data)}"
            )


class TestProcessShardWithHandle:
    """Test ShardProcessor.process_shard() with a pre-opened handle."""

    def _make_processor(self, tmp_path, source_dir):
        """Create a ShardProcessor with mocked GCS clients."""
        config = WorkerConfig(
            source_path=str(source_dir),
            dest_path="gs://fake-bucket/fake-dest",
            ng_spec=_write_info_file(str(tmp_path / "info_scratch"), [128, 128, 128]),
            scales=[0],
            staging_path=str(tmp_path / "staging"),
        )
        # Patch GCS client construction
        with patch("src.worker.storage.Client"):
            processor = ShardProcessor(config)
        return processor

    def _make_processor(self, tmp_path, shard_source_dir):
        """Create a ShardProcessor that reads local files via patched source_path."""
        ng_spec = {
            '@type': 'neuroglancer_multiscale_volume',
            'data_type': 'uint64',
            'num_channels': 1,
            'type': 'segmentation',
            'scales': [{
                'key': 's0',
                'size': [128, 128, 128],
                'resolution': [8, 8, 8],
                'chunk_sizes': [[64, 64, 64]],
                'encoding': 'compressed_segmentation',
                'compressed_segmentation_block_size': [8, 8, 8],
                'voxel_offset': [0, 0, 0],
                'sharding': {
                    '@type': 'neuroglancer_uint64_sharded_v1',
                    'hash': 'identity',
                    'preshift_bits': 9,
                    'minishard_bits': 6,
                    'shard_bits': 19,
                    'minishard_index_encoding': 'gzip',
                    'data_encoding': 'gzip',
                },
            }],
        }
        config = WorkerConfig(
            source_path="gs://fake-source/export",
            dest_path="gs://fake-dest/output",
            ng_spec=ng_spec,
            scales=[0],
            staging_path=str(tmp_path / "staging_proc"),
        )
        with patch("src.worker.storage.Client"):
            processor = ShardProcessor(config)

        # Patch _open_shard_with_retry to read from local files
        real_source = str(shard_source_dir)
        original_open = processor._open_shard_with_retry
        def local_open(arrow_uri, csv_uri, **kwargs):
            # Replace gs://fake-source/export with the real local path
            arrow_local = arrow_uri.replace("gs://fake-source/export", real_source)
            csv_local = csv_uri.replace("gs://fake-source/export", real_source)
            from braid import ShardReader
            return ShardReader(arrow_local, csv_local)
        processor._open_shard_with_retry = local_open

        return processor

    def test_process_shard_returns_true(self, tmp_path, shard_source_dir):
        """process_shard with a pre-opened handle returns True on success."""
        staging_dir = str(tmp_path / "staging")
        os.makedirs(staging_dir)
        _write_info_file(staging_dir, [128, 128, 128])
        dest = _open_local_volume(staging_dir)

        processor = self._make_processor(tmp_path, shard_source_dir)
        result = processor.process_shard(0, "0_0_0", dest)
        assert result is True

        # Verify data was written
        vol = dest[:, :, :, 0]
        data = vol[0:64, 0:64, 0:64].read().result()
        assert np.all(data == 100)

    def test_process_two_shards_sequentially(self, tmp_path, shard_source_dir):
        """Two process_shard calls on the same handle both succeed."""
        staging_dir = str(tmp_path / "staging")
        os.makedirs(staging_dir)
        _write_info_file(staging_dir, [128, 128, 128])
        dest = _open_local_volume(staging_dir)

        processor = self._make_processor(tmp_path, shard_source_dir)
        assert processor.process_shard(0, "0_0_0", dest) is True
        assert processor.process_shard(0, "0_64_0", dest) is True

        # Both shards' data present
        vol = dest[:, :, :, 0]
        assert np.all(vol[0:64, 0:64, 0:64].read().result() == 100)
        assert np.all(vol[64:128, 0:64, 0:64].read().result() == 200)
        assert np.all(vol[0:64, 64:128, 0:64].read().result() == 300)
        assert np.all(vol[64:128, 64:128, 0:64].read().result() == 400)


class TestUploadStagingDir:
    """Test the upload_staging_dir method."""

    def test_upload_counts_bytes(self, tmp_path):
        """upload_staging_dir returns total bytes of non-info files."""
        staging_dir = str(tmp_path / "staging")
        os.makedirs(os.path.join(staging_dir, "s0"))

        # Write fake info file (should be skipped)
        with open(os.path.join(staging_dir, "info"), "w") as f:
            f.write('{"fake": true}')

        # Write a fake shard file
        shard_path = os.path.join(staging_dir, "s0", "abc123.shard")
        shard_data = b"x" * 1024
        with open(shard_path, "wb") as f:
            f.write(shard_data)

        config = WorkerConfig(
            source_path="gs://fake/source",
            dest_path="gs://fake/dest",
            scales=[0],
        )
        with patch("src.worker.storage.Client") as mock_client:
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_bucket.blob.return_value = mock_blob
            mock_client.return_value.bucket.return_value = mock_bucket

            processor = ShardProcessor(config)
            processor.dest_bucket_obj = mock_bucket

            uploaded = processor.upload_staging_dir(staging_dir)

        assert uploaded == 1024
        mock_blob.upload_from_filename.assert_called_once()

    def test_upload_skips_info_file(self, tmp_path):
        """The info file is not uploaded."""
        staging_dir = str(tmp_path / "staging")
        os.makedirs(staging_dir)

        with open(os.path.join(staging_dir, "info"), "w") as f:
            f.write('{"fake": true}')

        config = WorkerConfig(
            source_path="gs://fake/source",
            dest_path="gs://fake/dest",
            scales=[0],
        )
        with patch("src.worker.storage.Client") as mock_client:
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_bucket.blob.return_value = mock_blob
            mock_client.return_value.bucket.return_value = mock_bucket

            processor = ShardProcessor(config)
            processor.dest_bucket_obj = mock_bucket

            uploaded = processor.upload_staging_dir(staging_dir)

        assert uploaded == 0
        mock_blob.upload_from_filename.assert_not_called()

    def test_upload_warns_on_empty_dir(self, tmp_path):
        """Warns when no shard files are found."""
        staging_dir = str(tmp_path / "staging")
        os.makedirs(staging_dir)
        with open(os.path.join(staging_dir, "info"), "w") as f:
            f.write('{}')

        config = WorkerConfig(
            source_path="gs://fake/source",
            dest_path="gs://fake/dest",
            scales=[0],
        )
        with patch("src.worker.storage.Client") as mock_client:
            mock_bucket = MagicMock()
            mock_client.return_value.bucket.return_value = mock_bucket

            processor = ShardProcessor(config)
            processor.dest_bucket_obj = mock_bucket

            uploaded = processor.upload_staging_dir(staging_dir)

        assert uploaded == 0
