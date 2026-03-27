"""Neuroglancer precomputed sharding calculations.

Implements the compressed Z-index and shard number derivation used by
TensorStore's neuroglancer_precomputed driver.  Every function here is
designed to produce bit-identical results to the C++ reference in:

  ~/tensorstore/tensorstore/driver/neuroglancer_precomputed/metadata.cc
  ~/tensorstore/tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded.cc

Key reference functions and their Python equivalents:

  C++ GetCompressedZIndexBits  ->  get_compressed_z_index_bits
  C++ EncodeCompressedZIndex   ->  compressed_z_index
  C++ GetChunkShardInfo        ->  chunk_shard_info  (identity hash only)
  C++ GetShardKey              ->  ng_shard_filename
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
