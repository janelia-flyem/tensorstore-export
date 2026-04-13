"""Helpers for validating exported NG segmentation values against DVID."""

from __future__ import annotations

import bisect
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import random
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict, deque
import math
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import tensorstore as ts

from src.ng_sharding import (
    chunk_shard_info,
    compressed_z_index,
    load_ng_spec_from_dict,
    shard_bbox,
)

_gcs_client = None


@dataclass(frozen=True)
class ShardRecord:
    """Existing shard geometry at one scale."""

    shard_number: int
    origin: tuple[int, int, int]
    extent: tuple[int, int, int]
    volume: int


@dataclass(frozen=True)
class FaceCandidate:
    """A one-voxel-thick plane immediately outside an existing shard."""

    axis: int
    direction: int
    origin: tuple[int, int, int]
    extent: tuple[int, int, int]
    area: int


@dataclass
class PointProgress:
    """Track point-based progress and emit timing updates."""

    label: str
    total_points: int
    step: int = 100
    processed: int = 0
    started_at: float = 0.0
    last_report_at: float = 0.0
    last_report_points: int = 0

    def __post_init__(self):
        now = time.perf_counter()
        if self.started_at == 0.0:
            self.started_at = now
        if self.last_report_at == 0.0:
            self.last_report_at = self.started_at

    def advance(self, delta: int) -> None:
        """Advance progress and emit updates at each step boundary."""
        if delta <= 0:
            return

        prev_processed = self.processed
        self.processed += delta
        if self.step <= 0:
            return

        prev_mark = prev_processed // self.step
        new_mark = self.processed // self.step
        if self.processed == self.total_points:
            new_mark = max(new_mark, prev_mark + 1)

        while prev_mark < new_mark:
            prev_mark += 1
            report_target = min(prev_mark * self.step, self.total_points)
            now = time.perf_counter()
            delta_points = report_target - self.last_report_points
            interval = max(now - self.last_report_at, 1e-9)
            total_elapsed = now - self.started_at
            rate = delta_points / interval
            print(
                f"  {self.label}: {report_target}/{self.total_points} points "
                f"in {total_elapsed:.2f}s "
                f"(last {delta_points} in {interval:.2f}s, {rate:.0f} pts/s)",
                flush=True,
            )
            self.last_report_at = now
            self.last_report_points = report_target


def _get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage

        _gcs_client = storage.Client()
    return _gcs_client


def is_gs_uri(path: str) -> bool:
    return path.startswith("gs://")


def parse_gs_uri(path: str) -> tuple[str, str]:
    """Split ``gs://bucket/prefix`` into (bucket, prefix)."""
    rest = path[len("gs://") :]
    bucket_name, _, blob_path = rest.partition("/")
    return bucket_name, blob_path.rstrip("/")


def _local_root(path: str) -> Path:
    return Path(path).expanduser().resolve()


def load_precomputed_info(volume_path: str) -> dict:
    """Load the NG precomputed ``info`` JSON from GCS or local disk."""
    if is_gs_uri(volume_path):
        bucket_name, prefix = parse_gs_uri(volume_path)
        blob_name = f"{prefix}/info" if prefix else "info"
        blob = _get_gcs_client().bucket(bucket_name).blob(blob_name)
        return json.loads(blob.download_as_text())

    info_path = _local_root(volume_path) / "info"
    return json.loads(info_path.read_text())


def open_precomputed_scale(volume_path: str, scale_index: int) -> ts.TensorStore:
    """Open a neuroglancer precomputed scale from GCS or local disk."""
    if is_gs_uri(volume_path):
        bucket_name, prefix = parse_gs_uri(volume_path)
        kvstore = {
            "driver": "gcs",
            "bucket": bucket_name,
            "path": prefix,
        }
    else:
        kvstore = {
            "driver": "file",
            "path": str(_local_root(volume_path)),
        }

    spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": kvstore,
        "scale_index": scale_index,
        "open": True,
    }
    return ts.open(spec).result()


def list_scale_shards(volume_path: str, scale_key: str) -> set[int]:
    """Return existing shard numbers for a scale."""
    if is_gs_uri(volume_path):
        bucket_name, prefix = parse_gs_uri(f"{volume_path.rstrip('/')}/{scale_key}")
        shard_numbers = set()
        for blob in _get_gcs_client().bucket(bucket_name).list_blobs(prefix=prefix):
            name = blob.name.rsplit("/", 1)[-1]
            if name.endswith(".shard"):
                shard_numbers.add(int(name[:-6], 16))
        return shard_numbers

    scale_dir = _local_root(volume_path) / scale_key
    return {int(p.stem, 16) for p in scale_dir.glob("*.shard")}


def map_export_point_to_dvid(
    point: tuple[int, int, int], z_compress: int = 0
) -> tuple[int, int, int]:
    """Map an export-space voxel coordinate into DVID scale-space."""
    return (point[0], point[1], point[2] * (z_compress + 1))


def voxel_to_chunk_coords(
    point: tuple[int, int, int], scale_params: dict
) -> tuple[int, int, int]:
    """Map a voxel coordinate to chunk coordinates."""
    chunk_size = scale_params["chunk_size"]
    return (
        point[0] // chunk_size[0],
        point[1] // chunk_size[1],
        point[2] // chunk_size[2],
    )


def voxel_to_shard_number(point: tuple[int, int, int], scale_params: dict) -> int:
    """Map a voxel coordinate to its NG shard number."""
    chunk_coords = voxel_to_chunk_coords(point, scale_params)
    morton = compressed_z_index(chunk_coords, scale_params["coord_bits"])
    shard_number, _ = chunk_shard_info(
        morton,
        scale_params["preshift_bits"],
        scale_params["minishard_bits"],
        scale_params["shard_bits"],
    )
    return shard_number


def build_shard_records(scale_params: dict, shard_numbers: Iterable[int]) -> list[ShardRecord]:
    """Compute voxel bounding boxes for the existing shards."""
    records = []
    for shard_number in sorted(shard_numbers):
        bbox = shard_bbox(shard_number, scale_params)
        extent = tuple(bbox["shard_extent"])
        volume = extent[0] * extent[1] * extent[2]
        if volume <= 0:
            continue
        records.append(
            ShardRecord(
                shard_number=shard_number,
                origin=tuple(bbox["shard_origin"]),
                extent=extent,
                volume=volume,
            )
        )
    return records


def _build_cumulative_weights(weights: list[int]) -> tuple[list[int], int]:
    total = 0
    cumulative = []
    for weight in weights:
        total += weight
        cumulative.append(total)
    return cumulative, total


def _weighted_choice_index(
    cumulative: list[int], total: int, rng: random.Random
) -> int:
    pick = rng.randrange(total)
    return bisect.bisect_right(cumulative, pick)


def sample_points_in_shards(
    shard_records: list[ShardRecord], count: int, rng: random.Random
) -> list[tuple[int, int, int]]:
    """Sample random voxel points from existing shard volumes."""
    if not shard_records or count <= 0:
        return []

    weights = [record.volume for record in shard_records]
    cumulative, total = _build_cumulative_weights(weights)
    points = []
    for _ in range(count):
        record = shard_records[_weighted_choice_index(cumulative, total, rng)]
        point = tuple(
            record.origin[dim] + rng.randrange(record.extent[dim])
            for dim in range(3)
        )
        points.append(point)
    return points


def build_exposed_faces(
    scale_params: dict,
    shard_records: list[ShardRecord],
    existing_shards: set[int],
) -> list[FaceCandidate]:
    """Find shard faces whose adjacent voxel plane is in a missing shard."""
    vol_size = scale_params["vol_size"]
    faces = []
    for record in shard_records:
        for axis in range(3):
            other_axes = [dim for dim in range(3) if dim != axis]
            area = record.extent[other_axes[0]] * record.extent[other_axes[1]]
            if area <= 0:
                continue
            for direction in (-1, 1):
                outside_coord = (
                    record.origin[axis] - 1
                    if direction < 0
                    else record.origin[axis] + record.extent[axis]
                )
                if outside_coord < 0 or outside_coord >= vol_size[axis]:
                    continue
                probe = list(record.origin)
                probe[axis] = outside_coord
                if voxel_to_shard_number(tuple(probe), scale_params) in existing_shards:
                    continue
                faces.append(
                    FaceCandidate(
                        axis=axis,
                        direction=direction,
                        origin=record.origin,
                        extent=record.extent,
                        area=area,
                    )
                )
    return faces


def sample_points_outside_shards(
    faces: list[FaceCandidate], count: int, rng: random.Random
) -> list[tuple[int, int, int]]:
    """Sample points that lie one voxel beyond an exposed shard face."""
    if not faces or count <= 0:
        return []

    weights = [face.area for face in faces]
    cumulative, total = _build_cumulative_weights(weights)
    points = []
    for _ in range(count):
        face = faces[_weighted_choice_index(cumulative, total, rng)]
        coords = [0, 0, 0]
        for dim in range(3):
            if dim == face.axis:
                coords[dim] = (
                    face.origin[dim] - 1
                    if face.direction < 0
                    else face.origin[dim] + face.extent[dim]
                )
            else:
                coords[dim] = face.origin[dim] + rng.randrange(face.extent[dim])
        points.append(tuple(coords))
    return points


def read_export_values(
    store: ts.TensorStore,
    points: list[tuple[int, int, int]],
    scale_params: dict,
    progress: PointProgress | None = None,
    max_in_flight_reads: int = 1,
) -> list[int]:
    """Read export values for the requested voxel coordinates."""
    if not points:
        return []
    if max_in_flight_reads <= 0:
        raise ValueError("max_in_flight_reads must be > 0")

    chunk_size = scale_params["chunk_size"]
    vol_size = scale_params["vol_size"]
    by_chunk = defaultdict(list)
    for idx, point in enumerate(points):
        chunk_coords = voxel_to_chunk_coords(point, scale_params)
        by_chunk[chunk_coords].append((idx, point))

    values = [0] * len(points)
    inflight = deque()

    def _submit_chunk_read(chunk_coords, items):
        origin = [chunk_coords[dim] * chunk_size[dim] for dim in range(3)]
        stop = [
            min(origin[dim] + chunk_size[dim], vol_size[dim]) for dim in range(3)
        ]
        future = store[
            origin[0] : stop[0], origin[1] : stop[1], origin[2] : stop[2]
        ].read()
        inflight.append((future, origin, items))

    def _resolve_chunk_read():
        future, origin, items = inflight.popleft()
        chunk_array = future.result()
        for idx, point in items:
            local = tuple(point[dim] - origin[dim] for dim in range(3))
            value = np.asarray(chunk_array[local])
            if value.ndim > 0:
                if value.size != 1:
                    raise RuntimeError(
                        "Expected a scalar export value at "
                        f"{point}, got shape {value.shape}"
                    )
                value = value.reshape(())
            values[idx] = int(value)
        if progress is not None:
            progress.advance(len(items))

    for chunk_coords, items in by_chunk.items():
        _submit_chunk_read(chunk_coords, items)
        if len(inflight) >= max_in_flight_reads:
            _resolve_chunk_read()

    while inflight:
        _resolve_chunk_read()
    return values


def fetch_dvid_label_batch(
    base_url: str,
    uuid: str,
    data_name: str,
    scale: int,
    points: list[tuple[int, int, int]],
    supervoxels: bool,
    batch_index: int,
    total_batches: int,
    batch_label: str,
) -> list[int]:
    """Fetch a single DVID batch and log timing for that request."""
    body = json.dumps(points, separators=(",", ":")).encode("utf-8")
    url = build_dvid_labels_url(
        base_url, uuid, data_name, scale, supervoxels, body
    )
    request = urllib.request.Request(
        url,
        data=body,
        method="GET",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"DVID label request failed for scale {scale}: "
            f"HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach DVID server: {exc}") from exc

    if len(payload) != len(points):
        raise RuntimeError(
            f"DVID returned {len(payload)} labels for {len(points)} points "
            f"at scale {scale}"
        )

    elapsed = time.perf_counter() - started_at
    print(
        f"  {batch_label}: batch {batch_index}/{total_batches}, "
        f"{len(points)} points in {elapsed:.2f}s",
        flush=True,
    )
    return [int(value) for value in payload]


def normalize_dvid_base_url(base_url: str) -> str:
    """Normalize a DVID base URL while tolerating a trailing ``/api``."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/api"):
        normalized = normalized[:-4]
    return normalized


def build_dvid_labels_url(
    base_url: str,
    uuid: str,
    data_name: str,
    scale: int,
    supervoxels: bool,
    body: bytes,
) -> str:
    """Build the DVID batch-labels URL including query parameters."""
    query = {
        "scale": str(scale),
        "hash": hashlib.md5(body).hexdigest(),
    }
    if supervoxels:
        query["supervoxels"] = "true"
    base = normalize_dvid_base_url(base_url)
    return (
        f"{base}/api/node/{uuid}/{data_name}/labels?"
        f"{urllib.parse.urlencode(query)}"
    )


def fetch_dvid_labels(
    base_url: str,
    uuid: str,
    data_name: str,
    scale: int,
    points: list[tuple[int, int, int]],
    supervoxels: bool,
    batch_size: int = 1000,
    batch_label: str = "Queried DVID batches",
) -> list[int]:
    """Fetch labels from DVID's batch ``/labels`` endpoint."""
    results = []
    total_batches = math.ceil(len(points) / batch_size) if points else 0
    for batch_index, start in enumerate(range(0, len(points), batch_size), start=1):
        batch = points[start : start + batch_size]
        results.extend(
            fetch_dvid_label_batch(
                base_url,
                uuid,
                data_name,
                scale,
                batch,
                supervoxels,
                batch_index=batch_index,
                total_batches=total_batches,
                batch_label=batch_label,
            )
        )
    return results


def compare_export_and_dvid_points(
    store: ts.TensorStore,
    export_points: list[tuple[int, int, int]],
    dvid_points: list[tuple[int, int, int]],
    scale_params: dict,
    *,
    dvid_url: str,
    uuid: str,
    data_name: str,
    scale: int,
    supervoxels: bool,
    batch_size: int = 1000,
    tensorstore_read_concurrency: int = 1,
    tensorstore_progress_step: int = 100,
    tensorstore_progress_label: str = "Read export values",
    dvid_batch_label: str = "Queried DVID batches",
) -> list[tuple[tuple[int, int, int], tuple[int, int, int], int, int]]:
    """Compare export values to DVID values with batched pipelining."""
    if len(export_points) != len(dvid_points):
        raise ValueError("export_points and dvid_points must have the same length")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if tensorstore_read_concurrency <= 0:
        raise ValueError("tensorstore_read_concurrency must be > 0")
    if not export_points:
        return []

    mismatches = []
    total_batches = math.ceil(len(export_points) / batch_size)
    progress = PointProgress(
        label=tensorstore_progress_label,
        total_points=len(export_points),
        step=tensorstore_progress_step,
    )

    def _submit_dvid_batch(
        executor: ThreadPoolExecutor,
        batch_points: list[tuple[int, int, int]],
        batch_index: int,
    ) -> Future:
        return executor.submit(
            fetch_dvid_label_batch,
            dvid_url,
            uuid,
            data_name,
            scale,
            batch_points,
            supervoxels,
            batch_index,
            total_batches,
            dvid_batch_label,
        )

    pending_future = None
    pending_batch = None

    with ThreadPoolExecutor(max_workers=1) as executor:
        for batch_index, start in enumerate(range(0, len(export_points), batch_size), start=1):
            export_batch = export_points[start : start + batch_size]
            dvid_batch = dvid_points[start : start + batch_size]
            export_values = read_export_values(
                store,
                export_batch,
                scale_params,
                progress=progress,
                max_in_flight_reads=tensorstore_read_concurrency,
            )

            if pending_future is not None and pending_batch is not None:
                dvid_values = pending_future.result()
                pending_export_batch, pending_dvid_batch, pending_export_values = pending_batch
                for export_point, dvid_point, export_value, dvid_value in zip(
                    pending_export_batch,
                    pending_dvid_batch,
                    pending_export_values,
                    dvid_values,
                ):
                    if export_value != dvid_value:
                        mismatches.append(
                            (export_point, dvid_point, export_value, dvid_value)
                        )

            pending_future = _submit_dvid_batch(executor, dvid_batch, batch_index)
            pending_batch = (export_batch, dvid_batch, export_values)

        if pending_future is not None and pending_batch is not None:
            dvid_values = pending_future.result()
            pending_export_batch, pending_dvid_batch, pending_export_values = pending_batch
            for export_point, dvid_point, export_value, dvid_value in zip(
                pending_export_batch,
                pending_dvid_batch,
                pending_export_values,
                dvid_values,
            ):
                if export_value != dvid_value:
                    mismatches.append(
                        (export_point, dvid_point, export_value, dvid_value)
                    )

    return mismatches


def load_scale_params(volume_path: str) -> dict[int, dict]:
    """Load NG scale metadata from the destination volume's ``info`` file."""
    info = load_precomputed_info(volume_path)
    return load_ng_spec_from_dict(info)
