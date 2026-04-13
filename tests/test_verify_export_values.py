"""Tests for export value verification helpers."""

import hashlib
import random
import urllib.parse

import numpy as np

from src.export_value_verifier import (
    build_dvid_labels_url,
    build_exposed_faces,
    build_shard_records,
    compare_export_and_dvid_points,
    map_export_point_to_dvid,
    read_export_values,
    sample_points_in_shards,
    sample_points_outside_shards,
    voxel_to_shard_number,
)
from src.ng_sharding import load_ng_spec_from_dict


def _simple_scale_params() -> dict:
    spec = {
        "@type": "neuroglancer_multiscale_volume",
        "scales": [
            {
                "key": "s0",
                "size": [128, 64, 64],
                "chunk_sizes": [[64, 64, 64]],
                "sharding": {
                    "@type": "neuroglancer_uint64_sharded_v1",
                    "preshift_bits": 0,
                    "minishard_bits": 0,
                    "shard_bits": 1,
                },
            }
        ],
    }
    return load_ng_spec_from_dict(spec)[0]


def test_map_export_point_to_dvid_applies_z_compress():
    assert map_export_point_to_dvid((10, 20, 30), 0) == (10, 20, 30)
    assert map_export_point_to_dvid((10, 20, 30), 1) == (10, 20, 60)


def test_build_exposed_faces_finds_missing_neighbor():
    scale_params = _simple_scale_params()
    records = build_shard_records(scale_params, {0})

    faces = build_exposed_faces(scale_params, records, {0})

    assert len(faces) == 1
    face = faces[0]
    assert face.axis == 0
    assert face.direction == 1
    assert face.area == 64 * 64


def test_build_exposed_faces_skips_existing_neighbor():
    scale_params = _simple_scale_params()
    records = build_shard_records(scale_params, {0, 1})

    faces = build_exposed_faces(scale_params, records, {0, 1})

    assert faces == []


def test_sample_points_stay_in_expected_regions():
    scale_params = _simple_scale_params()
    records = build_shard_records(scale_params, {0})
    faces = build_exposed_faces(scale_params, records, {0})
    rng = random.Random(123)

    inside_points = sample_points_in_shards(records, 20, rng)
    outside_points = sample_points_outside_shards(faces, 20, rng)

    assert all(0 <= x < 64 and 0 <= y < 64 and 0 <= z < 64 for x, y, z in inside_points)
    assert all(x == 64 and 0 <= y < 64 and 0 <= z < 64 for x, y, z in outside_points)
    assert all(voxel_to_shard_number(point, scale_params) == 1 for point in outside_points)


def test_build_dvid_labels_url_includes_expected_query():
    body = b"[[1,2,3],[4,5,6]]"

    url = build_dvid_labels_url(
        "https://my-dvid-server/api/",
        "839a23",
        "segmentation",
        scale=2,
        supervoxels=True,
        body=body,
    )

    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)

    assert parsed.path == "/api/node/839a23/segmentation/labels"
    assert query["scale"] == ["2"]
    assert query["supervoxels"] == ["true"]
    assert query["hash"] == [hashlib.md5(body).hexdigest()]


class _ImmediateRead:
    def __init__(self, array):
        self._array = array

    def read(self):
        return self

    def result(self):
        return self._array


class _FakeStore:
    def __init__(self, array):
        self._array = array

    def __getitem__(self, key):
        return _ImmediateRead(self._array[key])


class _TrackingFuture:
    def __init__(self, array, tracker):
        self._array = array
        self._tracker = tracker
        self._tracker["pending"] += 1
        self._tracker["max_pending"] = max(
            self._tracker["max_pending"], self._tracker["pending"]
        )

    def result(self):
        self._tracker["pending"] -= 1
        return self._array


class _TrackingRead:
    def __init__(self, array, tracker):
        self._array = array
        self._tracker = tracker

    def read(self):
        return _TrackingFuture(self._array, self._tracker)


class _TrackingStore:
    def __init__(self, array, tracker):
        self._array = array
        self._tracker = tracker

    def __getitem__(self, key):
        return _TrackingRead(self._array[key], self._tracker)


def test_read_export_values_handles_singleton_channel_axis():
    scale_params = {
        "chunk_size": [2, 2, 2],
        "vol_size": [2, 2, 2],
    }
    array = np.arange(8, dtype=np.uint64).reshape(2, 2, 2)
    array = array[..., np.newaxis]
    store = _FakeStore(array)

    values = read_export_values(store, [(0, 0, 0), (1, 1, 1)], scale_params)

    assert values == [0, 7]


def test_read_export_values_uses_tensorstore_concurrency():
    scale_params = {
        "chunk_size": [1, 1, 1],
        "vol_size": [2, 2, 2],
    }
    array = np.arange(8, dtype=np.uint64).reshape(2, 2, 2)
    array = array[..., np.newaxis]
    tracker = {"pending": 0, "max_pending": 0}
    store = _TrackingStore(array, tracker)

    values = read_export_values(
        store,
        [(0, 0, 0), (1, 1, 1), (1, 0, 0)],
        scale_params,
        max_in_flight_reads=2,
    )

    assert values == [0, 7, 4]
    assert tracker["max_pending"] == 2


def test_compare_export_and_dvid_points_reports_mismatches(monkeypatch):
    scale_params = {
        "chunk_size": [2, 2, 2],
        "vol_size": [2, 2, 2],
    }
    array = np.arange(8, dtype=np.uint64).reshape(2, 2, 2)
    array = array[..., np.newaxis]
    store = _FakeStore(array)
    export_points = [(0, 0, 0), (1, 1, 1), (1, 0, 0)]
    dvid_points = list(export_points)

    def _fake_fetch_batch(
        base_url,
        uuid,
        data_name,
        scale,
        points,
        supervoxels,
        batch_index,
        total_batches,
        batch_label,
    ):
        mapping = {
            (0, 0, 0): 0,
            (1, 1, 1): 99,
            (1, 0, 0): 4,
        }
        return [mapping[p] for p in points]

    monkeypatch.setattr(
        "src.export_value_verifier.fetch_dvid_label_batch",
        _fake_fetch_batch,
    )

    mismatches = compare_export_and_dvid_points(
        store,
        export_points,
        dvid_points,
        scale_params,
        dvid_url="https://example.com",
        uuid="abc123",
        data_name="segmentation",
        scale=0,
        supervoxels=True,
        batch_size=2,
        tensorstore_progress_step=100,
    )

    assert mismatches == [((1, 1, 1), (1, 1, 1), 7, 99)]
