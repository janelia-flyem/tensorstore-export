"""Neuroglancer precomputed sharding calculations.

Implements the compressed Z-index and shard number derivation used by
TensorStore's neuroglancer_precomputed driver.  Every function here is
designed to produce bit-identical results to the C++ reference in:

  ~/tensorstore/tensorstore/driver/neuroglancer_precomputed/metadata.cc
  ~/tensorstore/tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded.cc

Key reference functions and their Python equivalents:

  C++ GetCompressedZIndexBits       ->  get_compressed_z_index_bits
  C++ EncodeCompressedZIndex        ->  compressed_z_index
  C++ GetChunkShardInfo             ->  chunk_shard_info  (identity hash only)
  C++ GetShardKey                   ->  ng_shard_filename
  C++ CompressedMortonBitIterator   ->  _CompressedMortonBitIterator
  C++ GetShardChunkHierarchy        ->  get_shard_chunk_hierarchy
  C++ GetChunksPerVolumeShardFunction -> chunks_per_shard / shard_origin_in_chunks
"""

import json
from typing import Dict, List, Sequence, Tuple


def get_compressed_z_index_bits(
    shape: Sequence[int], chunk_size: Sequence[int]
) -> List[int]:
    """Compute bits per dimension for compressed Z-index encoding.

    Matches TensorStore's GetCompressedZIndexBits() exactly:
      bits[i] = bit_width(max(0, CeilOfRatio(shape[i], chunk_size[i]) - 1))

    Args:
        shape: Volume size in voxels, (x, y, z).
        chunk_size: Chunk size in voxels, (x, y, z).

    Returns:
        List of 3 ints: number of bits per dimension.
    """
    bits = []
    for i in range(3):
        grid = -(-shape[i] // chunk_size[i])  # CeilOfRatio
        val = max(0, grid - 1)
        bits.append(val.bit_length())  # bit_width
    return bits


def compressed_z_index(
    coords: Sequence[int], coord_bits: Sequence[int]
) -> int:
    """Compressed Z-index matching TensorStore's EncodeCompressedZIndex.

    Unlike a standard Morton code, the compressed variant only interleaves
    bits for dimensions that still have significant bits at each level.

    Args:
        coords: (x, y, z) chunk grid coordinates.
        coord_bits: (bx, by, bz) number of bits per dimension,
                    from get_compressed_z_index_bits().

    Returns:
        The compressed Z-index (uint64).
    """
    max_bit = max(coord_bits)
    code = 0
    j = 0
    for bit in range(max_bit):
        for dim in range(3):
            if bit < coord_bits[dim]:
                code |= ((coords[dim] >> bit) & 1) << j
                j += 1
    return code


def chunk_shard_info(
    chunk_id: int,
    preshift_bits: int,
    minishard_bits: int,
    shard_bits: int,
) -> Tuple[int, int]:
    """Map a chunk ID to (shard_number, minishard_number).

    Matches TensorStore's GetChunkShardInfo + GetSplitShardInfo for the
    identity hash function.  Neuroglancer precomputed always uses identity.

    Args:
        chunk_id: The compressed Z-index of the chunk.
        preshift_bits: Bits to right-shift before hashing.
        minishard_bits: Number of minishard bits.
        shard_bits: Number of shard bits.

    Returns:
        (shard_number, minishard_number) tuple.
    """
    # ShiftRightUpTo64 equivalent
    if preshift_bits >= 64:
        hash_input = 0
    else:
        hash_input = chunk_id >> preshift_bits

    # Identity hash -> hash_output == hash_input
    hash_output = hash_input

    # GetLowBitMask and split
    combined_bits = minishard_bits + shard_bits
    if combined_bits >= 64:
        shard_and_minishard = hash_output
    else:
        shard_and_minishard = hash_output & ((1 << combined_bits) - 1)

    if minishard_bits >= 64:
        minishard = shard_and_minishard
    else:
        minishard = shard_and_minishard & ((1 << minishard_bits) - 1)

    if minishard_bits >= 64:
        shifted = 0
    else:
        shifted = shard_and_minishard >> minishard_bits

    if shard_bits == 0:
        shard = 0
    elif shard_bits >= 64:
        shard = shifted
    else:
        shard = shifted & ((1 << shard_bits) - 1)

    return shard, minishard


def ng_shard_filename(shard_number: int, shard_bits: int) -> str:
    """Compute the neuroglancer shard filename.

    Matches TensorStore's GetShardKey():
      absl::StrFormat("%0*x.shard", CeilOfRatio(shard_bits, 4), shard_number)

    Args:
        shard_number: The shard number.
        shard_bits: Number of shard bits from the sharding spec.

    Returns:
        Filename like "061c5.shard".
    """
    hex_digits = -(-shard_bits // 4)  # CeilOfRatio(shard_bits, 4)
    return f"{shard_number:0{hex_digits}x}.shard"


class _CompressedMortonBitIterator:
    """Python port of TensorStore's CompressedMortonBitIterator.

    Walks through bits in compressed Morton code order.  At each step,
    cycles through dimensions x(0)->y(1)->z(2), skipping any dimension
    that has exhausted its z_index_bits.

    Source:
      ~/tensorstore/tensorstore/driver/neuroglancer_precomputed/metadata.cc
    """

    __slots__ = ("z_index_bits", "cur_bit_for_dim", "dim_i")

    def __init__(self, z_index_bits: Sequence[int]):
        self.z_index_bits = list(z_index_bits)
        self.cur_bit_for_dim = [0, 0, 0]
        self.dim_i = 0

    def _get_next_dim(self) -> int:
        """Advance dim_i to the next dimension that still has bits."""
        while self.cur_bit_for_dim[self.dim_i] == self.z_index_bits[self.dim_i]:
            self.dim_i = (self.dim_i + 1) % 3
        return self.dim_i

    def _next(self):
        """Consume one bit from the current dimension, rotate to the next."""
        self.cur_bit_for_dim[self.dim_i] += 1
        self.dim_i = (self.dim_i + 1) % 3

    def advance(self, n: int):
        """Advance by *n* compressed-Morton bits."""
        for _ in range(n):
            self._get_next_dim()
            self._next()

    def get_cell_shape(self, grid_shape: Sequence[int]) -> List[int]:
        """Shape of the cell covered by bits consumed so far.

        Returns min(grid_shape[i], 1 << cur_bit_for_dim[i]) per dim.
        """
        return [
            min(grid_shape[i], 1 << self.cur_bit_for_dim[i])
            for i in range(3)
        ]


def get_shard_chunk_hierarchy(scale_params: dict) -> dict:
    """Compute the shard chunk hierarchy for a given scale.

    This is the Python equivalent of TensorStore's GetShardChunkHierarchy().
    It determines how the compressed Morton code bits are partitioned into
    preshift (within-minishard), minishard, and shard portions, and computes
    the minishard and shard shapes in chunks.

    Args:
        scale_params: Dict with keys: coord_bits, preshift_bits,
                      minishard_bits, shard_bits, grid_shape.

    Returns:
        Dict with keys:
            z_index_bits: bits per dimension (same as coord_bits)
            grid_shape_in_chunks: grid shape
            minishard_shape_in_chunks: [sx, sy, sz] minishard extent
            shard_shape_in_chunks: [sx, sy, sz] shard extent
            non_shard_bits: total preshift+minishard bits (clamped)
            shard_bits: effective shard bits (clamped)

    Raises:
        ValueError: If total bits exceed the sharding spec capacity.
    """
    z_index_bits = list(scale_params["coord_bits"])
    grid_shape = list(scale_params["grid_shape"])
    preshift_bits = scale_params["preshift_bits"]
    minishard_bits = scale_params["minishard_bits"]
    shard_bits = scale_params["shard_bits"]

    total_z_index_bits = sum(z_index_bits)

    if total_z_index_bits > preshift_bits + minishard_bits + shard_bits:
        raise ValueError(
            f"total_z_index_bits ({total_z_index_bits}) > "
            f"preshift_bits + minishard_bits + shard_bits "
            f"({preshift_bits} + {minishard_bits} + {shard_bits}): "
            f"shards do not correspond to rectangular regions"
        )

    within_minishard_bits = min(preshift_bits, total_z_index_bits)
    non_shard_bits = min(
        minishard_bits + preshift_bits, total_z_index_bits
    )
    effective_shard_bits = min(shard_bits, total_z_index_bits - non_shard_bits)

    bit_it = _CompressedMortonBitIterator(z_index_bits)

    # Advance past within-minishard bits to get minishard shape.
    bit_it.advance(within_minishard_bits)
    minishard_shape = bit_it.get_cell_shape(grid_shape)

    # Advance past remaining non-shard bits to get shard shape.
    bit_it.advance(non_shard_bits - within_minishard_bits)
    shard_shape = bit_it.get_cell_shape(grid_shape)

    return {
        "z_index_bits": z_index_bits,
        "grid_shape_in_chunks": grid_shape,
        "minishard_shape_in_chunks": minishard_shape,
        "shard_shape_in_chunks": shard_shape,
        "non_shard_bits": non_shard_bits,
        "shard_bits": effective_shard_bits,
    }


def shard_origin_in_chunks(
    shard_number: int, scale_params: dict
) -> List[int]:
    """Compute the chunk-space origin of a shard.

    This is the inverse mapping: given a shard number, return the (x, y, z)
    origin in chunk coordinates of the rectangular region that shard covers.

    Matches the shard-to-origin logic in TensorStore's
    GetChunksPerVolumeShardFunction.

    Args:
        shard_number: The shard number.
        scale_params: Dict with keys: coord_bits, preshift_bits,
                      minishard_bits, shard_bits, grid_shape.

    Returns:
        [ox, oy, oz] chunk-space origin of the shard.
    """
    hierarchy = get_shard_chunk_hierarchy(scale_params)
    z_index_bits = hierarchy["z_index_bits"]
    non_shard_bits = hierarchy["non_shard_bits"]
    effective_shard_bits = hierarchy["shard_bits"]

    bit_it = _CompressedMortonBitIterator(z_index_bits)
    # Skip past the non-shard bits (preshift + minishard).
    bit_it.advance(non_shard_bits)

    origin = [0, 0, 0]
    for bit_i in range(effective_shard_bits):
        dim_i = bit_it._get_next_dim()
        if (shard_number >> bit_i) & 1:
            origin[dim_i] |= 1 << bit_it.cur_bit_for_dim[dim_i]
        bit_it._next()

    return origin


def chunks_per_shard(
    shard_number: int, scale_params: dict
) -> int:
    """Compute the number of chunks in a specific shard.

    Matches TensorStore's GetChunksPerVolumeShardFunction.  For shards at
    the volume boundary, the count is reduced because the shard extends
    beyond the grid.

    Args:
        shard_number: The shard number.
        scale_params: Dict with keys: coord_bits, preshift_bits,
                      minishard_bits, shard_bits, grid_shape.

    Returns:
        Number of chunks in this shard.
    """
    hierarchy = get_shard_chunk_hierarchy(scale_params)
    z_index_bits = hierarchy["z_index_bits"]
    grid_shape = hierarchy["grid_shape_in_chunks"]
    non_shard_bits = hierarchy["non_shard_bits"]
    effective_shard_bits = hierarchy["shard_bits"]

    if (shard_number >> effective_shard_bits) != 0:
        return 0  # Invalid shard number

    bit_it = _CompressedMortonBitIterator(z_index_bits)
    bit_it.advance(non_shard_bits)
    cell_shape = bit_it.get_cell_shape(grid_shape)

    origin = [0, 0, 0]
    for bit_i in range(effective_shard_bits):
        dim_i = bit_it._get_next_dim()
        if (shard_number >> bit_i) & 1:
            origin[dim_i] |= 1 << bit_it.cur_bit_for_dim[dim_i]
        bit_it._next()

    num_chunks = 1
    for dim_i in range(3):
        num_chunks *= min(grid_shape[dim_i] - origin[dim_i], cell_shape[dim_i])
    return num_chunks


def shard_bbox(shard_number: int, scale_params: dict) -> dict:
    """Compute the voxel bounding box for a shard.

    Args:
        shard_number: The shard number.
        scale_params: Dict with keys: coord_bits, preshift_bits,
                      minishard_bits, shard_bits, grid_shape, chunk_size,
                      vol_size.

    Returns:
        Dict with keys:
            shard_number: the input shard number
            shard_origin: (x0, y0, z0) voxel coordinates
            shard_extent: (sx, sy, sz) voxel size (clipped to volume bounds)
            num_chunks: actual number of chunks in this shard
    """
    hierarchy = get_shard_chunk_hierarchy(scale_params)
    shard_shape = hierarchy["shard_shape_in_chunks"]
    chunk_size = scale_params["chunk_size"]
    vol_size = scale_params["vol_size"]

    origin_chunks = shard_origin_in_chunks(shard_number, scale_params)
    num_chunks = chunks_per_shard(shard_number, scale_params)

    # Voxel origin
    voxel_origin = [origin_chunks[d] * chunk_size[d] for d in range(3)]

    # Voxel extent: shard shape in voxels, clipped to volume bounds
    voxel_extent = [
        min(shard_shape[d] * chunk_size[d], vol_size[d] - voxel_origin[d])
        for d in range(3)
    ]

    return {
        "shard_number": shard_number,
        "shard_origin": voxel_origin,
        "shard_extent": voxel_extent,
        "num_chunks": num_chunks,
    }


def enumerate_shard_bboxes(scale_params: dict) -> List[dict]:
    """Enumerate all non-empty shard bboxes for a scale.

    Args:
        scale_params: Dict with keys from _parse_scale_params.

    Returns:
        List of shard_bbox dicts for every shard that has >0 chunks.
    """
    shard_bits = scale_params["shard_bits"]
    max_shards = 1 << shard_bits if shard_bits > 0 else 1
    result = []
    for sn in range(max_shards):
        nc = chunks_per_shard(sn, scale_params)
        if nc > 0:
            result.append(shard_bbox(sn, scale_params))
    return result


def parent_shards_to_child_shards(
    parent_shard_numbers: List[int],
    parent_params: dict,
    child_params: dict,
) -> List[int]:
    """Derive child-scale shard numbers from parent-scale shard numbers.

    For each parent shard, computes its chunk grid range, maps those
    coordinates to the child scale (divide by 2 in each dim), and
    computes the child shard numbers that overlap.

    This is the core function for the sparse shard derivation chain:
    s0 shards (from DVID Arrow files) -> s1 shards -> s2 shards -> ...

    Args:
        parent_shard_numbers: List of shard numbers at scale N-1.
        parent_params: Scale params for scale N-1.
        child_params: Scale params for scale N.

    Returns:
        Sorted list of unique shard numbers at scale N.
    """
    parent_hierarchy = get_shard_chunk_hierarchy(parent_params)
    parent_shard_shape = parent_hierarchy["shard_shape_in_chunks"]
    parent_grid = parent_hierarchy["grid_shape_in_chunks"]

    child_grid = child_params["grid_shape"]
    child_coord_bits = child_params["coord_bits"]

    child_shard_set = set()

    for parent_sn in parent_shard_numbers:
        # Parent shard's chunk-space origin and extent
        p_origin = shard_origin_in_chunks(parent_sn, parent_params)
        p_extent = [
            min(parent_shard_shape[d], parent_grid[d] - p_origin[d])
            for d in range(3)
        ]

        # Map parent chunk range to child chunk coordinates (divide by 2).
        # Parent chunk (cx, cy, cz) at parent scale covers voxels that
        # correspond to child chunk (cx // 2, cy // 2, cz // 2).
        c_min = [p_origin[d] // 2 for d in range(3)]
        c_max = [
            min((p_origin[d] + p_extent[d] - 1) // 2, child_grid[d] - 1)
            for d in range(3)
        ]

        # Find all child shard numbers that overlap this range.
        # For efficiency, compute the child shard hierarchy to step by
        # shard-shape rather than iterating every chunk.
        child_hierarchy = get_shard_chunk_hierarchy(child_params)
        child_shard_shape = child_hierarchy["shard_shape_in_chunks"]

        # Step through child chunk space at shard granularity
        cx = c_min[0]
        while cx <= c_max[0]:
            cy = c_min[1]
            while cy <= c_max[1]:
                cz = c_min[2]
                while cz <= c_max[2]:
                    # Compute child shard number for this chunk coord
                    morton = compressed_z_index(
                        (cx, cy, cz), child_coord_bits
                    )
                    shard, _ = chunk_shard_info(
                        morton,
                        child_params["preshift_bits"],
                        child_params["minishard_bits"],
                        child_params["shard_bits"],
                    )
                    child_shard_set.add(shard)
                    # Step by child shard shape to skip to next shard boundary
                    cz += child_shard_shape[2]
                cy += child_shard_shape[1]
            cx += child_shard_shape[0]

    return sorted(child_shard_set)


def shard_chunk_coords(
    shard_number: int, scale_params: dict
) -> List[Tuple[int, int, int]]:
    """Enumerate all (cx, cy, cz) chunk coordinates within a shard.

    Args:
        shard_number: The shard number.
        scale_params: Dict with keys: coord_bits, preshift_bits,
                      minishard_bits, shard_bits, grid_shape, chunk_size,
                      vol_size.

    Returns:
        List of (cx, cy, cz) chunk coordinate tuples.
    """
    hierarchy = get_shard_chunk_hierarchy(scale_params)
    shard_shape = hierarchy["shard_shape_in_chunks"]
    grid_shape = hierarchy["grid_shape_in_chunks"]

    origin = shard_origin_in_chunks(shard_number, scale_params)

    coords = []
    for dz in range(min(shard_shape[2], grid_shape[2] - origin[2])):
        for dy in range(min(shard_shape[1], grid_shape[1] - origin[1])):
            for dx in range(min(shard_shape[0], grid_shape[0] - origin[0])):
                coords.append((origin[0] + dx, origin[1] + dy, origin[2] + dz))
    return coords


def dvid_to_ng_shard_number(
    shard_name: str, scale_params: dict
) -> int:
    """Map a DVID shard name (e.g. "10240_40960_43008") to an NG shard number.

    Args:
        shard_name: DVID shard name in "x_y_z" voxel coordinate format.
        scale_params: Dict with keys: chunk_size, coord_bits, preshift_bits,
                      minishard_bits, shard_bits.

    Returns:
        The neuroglancer shard number.
    """
    x, y, z = (int(v) for v in shard_name.split("_"))
    chunk_size = scale_params["chunk_size"]
    cx = x // chunk_size[0]
    cy = y // chunk_size[1]
    cz = z // chunk_size[2]
    morton = compressed_z_index((cx, cy, cz), scale_params["coord_bits"])
    shard, _ = chunk_shard_info(
        morton,
        scale_params["preshift_bits"],
        scale_params["minishard_bits"],
        scale_params["shard_bits"],
    )
    return shard


# ---------------------------------------------------------------------------
# NG spec loading
# ---------------------------------------------------------------------------

def _parse_scale_params(scale_dict: dict) -> dict:
    """Extract sharding parameters from a single scale entry."""
    sh = scale_dict["sharding"]
    chunk_size = scale_dict["chunk_sizes"][0]
    vol_size = scale_dict["size"]
    grid_shape = [
        -(-vol_size[d] // chunk_size[d]) for d in range(3)
    ]
    coord_bits = get_compressed_z_index_bits(vol_size, chunk_size)
    return {
        "key": scale_dict["key"],
        "preshift_bits": sh["preshift_bits"],
        "minishard_bits": sh["minishard_bits"],
        "shard_bits": sh["shard_bits"],
        "chunk_size": chunk_size,
        "vol_size": vol_size,
        "grid_shape": grid_shape,
        "coord_bits": coord_bits,
    }


def load_ng_spec(spec_path: str) -> Dict[int, dict]:
    """Load NG spec file -> {scale_index: scale_params}."""
    with open(spec_path) as f:
        spec = json.load(f)
    return load_ng_spec_from_dict(spec)


def load_ng_spec_from_dict(spec_dict: dict) -> Dict[int, dict]:
    """Parse an already-loaded NG spec dict -> {scale_index: scale_params}."""
    result = {}
    for i, s in enumerate(spec_dict["scales"]):
        result[i] = _parse_scale_params(s)
    return result
