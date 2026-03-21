"""
DVID Block Decompressor.

Implements the DVID compressed segmentation format decompression with support
for both agglomerated labels and supervoxel labels.

The hot path uses a C extension (libdvid_decompress.so) for performance.
Falls back to a pure-Python reference implementation if the shared library
is not available.  See braid/docs/DVID-block-decompression.md for details.
"""

import ctypes
import struct
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import zstandard as zstd
from .exceptions import DecompressionError

# ---------------------------------------------------------------------------
# C extension loading
# ---------------------------------------------------------------------------

_c_lib = None

def _load_c_lib():
    """Try to load the C decompression library."""
    global _c_lib
    if _c_lib is not None:
        return _c_lib

    # Look for the shared library relative to this file's package
    search_paths = [
        Path(__file__).resolve().parent.parent.parent / "csrc" / "libdvid_decompress.so",
        Path(__file__).resolve().parent / "libdvid_decompress.so",
    ]
    for path in search_paths:
        if path.exists():
            try:
                lib = ctypes.CDLL(str(path))
                lib.dvid_decompress_block.restype = ctypes.c_int
                lib.dvid_decompress_block.argtypes = [
                    ctypes.c_char_p,       # data
                    ctypes.c_size_t,       # data_len
                    ctypes.POINTER(ctypes.c_uint64),  # mapped_labels
                    ctypes.c_size_t,       # num_labels
                    ctypes.POINTER(ctypes.c_uint64),  # output
                    ctypes.c_int,          # gx
                    ctypes.c_int,          # gy
                    ctypes.c_int,          # gz
                ]
                _c_lib = lib
                return lib
            except OSError:
                continue
    return None


# ---------------------------------------------------------------------------
# Python reference helpers (used by fallback and tests)
# ---------------------------------------------------------------------------

def bits_for(n: int) -> int:
    """Calculate minimum bits needed to represent n-1 values (0 to n-1)."""
    if n < 2:
        return 0
    n -= 1
    bits = 0
    while n > 0:
        bits += 1
        n >>= 1
    return bits


def get_packed_value(data: bytes, bit_head: int, bits: int) -> int:
    """Extract a packed value from byte array at given bit position."""
    if bits == 0:
        return 0

    byte_pos = bit_head >> 3
    bit_pos = bit_head & 7

    LEFT_BIT_MASK = [0xFF, 0x7F, 0x3F, 0x1F, 0x0F, 0x07, 0x03, 0x01]

    if bit_pos + bits <= 8:
        right_shift = 8 - bit_pos - bits
        return (data[byte_pos] & LEFT_BIT_MASK[bit_pos]) >> right_shift
    else:
        index = (data[byte_pos] & LEFT_BIT_MASK[bit_pos]) << 8
        index |= data[byte_pos + 1]
        index >>= (16 - bit_pos - bits)
        return index


def _map_labels(labels: np.ndarray,
                agglo_labels: Optional[np.ndarray],
                supervoxels: Optional[np.ndarray]) -> np.ndarray:
    """Pre-map block-level labels from supervoxels to agglomerated IDs.

    Builds a dict once and maps all block labels upfront, replacing the
    per-voxel np.where() that was the #1 bottleneck in the old code.
    """
    if agglo_labels is None or supervoxels is None:
        return labels.copy()

    sv_to_agglo = dict(zip(supervoxels.tolist(), agglo_labels.tolist()))
    return np.array(
        [sv_to_agglo.get(int(lbl), int(lbl)) for lbl in labels],
        dtype=np.uint64
    )


# ---------------------------------------------------------------------------
# DVIDDecompressor
# ---------------------------------------------------------------------------

class DVIDDecompressor:
    """
    DVID Block decompressor implementation.

    Uses a C extension for the inner decompression loop when available,
    falling back to pure Python otherwise.

    Block Format Overview:
    - gx, gy, gz: Number of sub-blocks in each dimension (uint32 each)
    - N: Number of labels (uint32)
    - labels: N uint64 labels
    - sub-block data: Variable length compressed indices with bit packing
    """

    def __init__(self):
        """Initialize the DVID decompressor."""
        self._zstd_decompressor = zstd.ZstdDecompressor()
        self._c_lib = _load_c_lib()

    def decompress_block(self, compressed_data: bytes,
                        agglo_labels: Optional[List[int]] = None,
                        supervoxels: Optional[List[int]] = None,
                        block_shape: Tuple[int, int, int] = (64, 64, 64)) -> np.ndarray:
        """
        Decompress a two-layer compressed DVID block.

        Args:
            compressed_data: The zstd-compressed DVID block data
            agglo_labels: Optional list of agglomerated label IDs
            supervoxels: Optional list of supervoxel IDs
            block_shape: Shape of the output block (nz, ny, nx)

        Returns:
            Decompressed uint64 array of shape block_shape

        Raises:
            DecompressionError: If decompression fails
        """
        try:
            if len(compressed_data) == 0:
                return np.zeros(block_shape, dtype=np.uint64)

            try:
                # max_output_size is required when the zstd frame header
                # doesn't declare content size (valid zstd, but the
                # zstandard library raises without it).  DVID blocks are
                # at most ~2.1 MB decompressed; 16 MB is a safe ceiling.
                dvid_compressed_data = self._zstd_decompressor.decompress(
                    compressed_data, max_output_size=16 * 1024 * 1024
                )
            except Exception as e:
                raise DecompressionError(f"Failed to decompress zstd layer: {e}")

            return self._make_label_volume(dvid_compressed_data, agglo_labels, supervoxels, block_shape)
        except Exception as e:
            if isinstance(e, DecompressionError):
                raise
            raise DecompressionError(f"DVID block decompression failed: {e}")

    def _make_label_volume(
        self,
        compressed_data: bytes,
        agglo_labels: Optional[List[int]] = None,
        supervoxels: Optional[List[int]] = None,
        block_shape: Tuple[int, int, int] = (64, 64, 64)
    ) -> np.ndarray:
        """
        Decompress a DVID compressed block into a numpy array in ZYX order.

        Uses the C extension when available, otherwise falls back to the
        pure-Python reference implementation.
        """
        SUB_BLOCK_SIZE = 8

        if len(compressed_data) < 16:
            raise ValueError("Compressed data too short for header")

        gx, gy, gz, num_labels = struct.unpack('<IIII', compressed_data[:16])

        expected_shape = (gz * SUB_BLOCK_SIZE, gy * SUB_BLOCK_SIZE, gx * SUB_BLOCK_SIZE)
        if expected_shape != block_shape:
            raise ValueError(f"Block shape mismatch: expected {expected_shape}, got {block_shape}")

        if num_labels == 0:
            raise ValueError("Block has 0 labels, which is not allowed")

        # Parse and pre-map the block-level labels
        pos = 16
        labels_size = num_labels * 8
        if pos + labels_size > len(compressed_data):
            raise ValueError("Compressed data too short for labels")

        labels = np.frombuffer(compressed_data[pos:pos + labels_size], dtype='<u8')

        if agglo_labels is not None:
            agglo_labels = np.array(agglo_labels, dtype=np.uint64)
        if supervoxels is not None:
            supervoxels = np.array(supervoxels, dtype=np.uint64)

        mapped_labels = _map_labels(labels, agglo_labels, supervoxels)

        # Try C extension
        if self._c_lib is not None:
            return self._make_label_volume_c(
                compressed_data, mapped_labels, num_labels, gx, gy, gz, block_shape
            )

        # Fall back to Python reference
        return self._make_label_volume_reference(
            compressed_data, mapped_labels, num_labels, gx, gy, gz, block_shape
        )

    def _make_label_volume_c(
        self,
        compressed_data: bytes,
        mapped_labels: np.ndarray,
        num_labels: int,
        gx: int, gy: int, gz: int,
        block_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Decompress using the C extension."""
        output = np.zeros(block_shape, dtype=np.uint64)

        # Ensure mapped_labels is contiguous C-order uint64
        mapped_labels = np.ascontiguousarray(mapped_labels, dtype=np.uint64)

        ret = self._c_lib.dvid_decompress_block(
            compressed_data,
            len(compressed_data),
            mapped_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            num_labels,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            gx, gy, gz,
        )

        if ret != 0:
            raise ValueError(
                f"C decompression failed (ret={ret}) for block "
                f"gx={gx} gy={gy} gz={gz} num_labels={num_labels}"
            )

        return output

    def _make_label_volume_reference(
        self,
        compressed_data: bytes,
        mapped_labels: np.ndarray,
        num_labels: int,
        gx: int, gy: int, gz: int,
        block_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Pure-Python reference implementation.

        Equivalent to Go's MakeLabelVolume.  Kept for testing and as fallback
        when the C extension is not available.
        """
        SUB_BLOCK_SIZE = 8

        output = np.zeros(block_shape, dtype=np.uint64)

        if num_labels < 2:
            label = mapped_labels[0] if num_labels == 1 else 0
            output.fill(label)
            return output

        pos = 16 + num_labels * 8
        num_sub_blocks = gx * gy * gz

        num_sb_labels = np.frombuffer(compressed_data[pos:pos + num_sub_blocks * 2], dtype='<u2')
        pos += num_sub_blocks * 2

        total_sb_indices = np.sum(num_sb_labels, dtype=np.uint32)
        sb_indices = np.frombuffer(compressed_data[pos:pos + total_sb_indices * 4], dtype='<u4')
        pos += total_sb_indices * 4

        sb_values = compressed_data[pos:]

        sub_block_num_voxels = SUB_BLOCK_SIZE * SUB_BLOCK_SIZE * SUB_BLOCK_SIZE
        sb_labels = np.zeros(sub_block_num_voxels, dtype=np.uint64)

        index_pos = 0
        bit_pos = 0
        sub_block_num = 0

        for sz in range(gz):
            for sy in range(gy):
                for sx in range(gx):
                    num_sb_labels_cur = num_sb_labels[sub_block_num]
                    bits = bits_for(num_sb_labels_cur)

                    for i in range(num_sb_labels_cur):
                        sb_labels[i] = mapped_labels[sb_indices[index_pos]]
                        index_pos += 1

                    base_z = sz * SUB_BLOCK_SIZE
                    base_y = sy * SUB_BLOCK_SIZE
                    base_x = sx * SUB_BLOCK_SIZE

                    for z in range(SUB_BLOCK_SIZE):
                        for y in range(SUB_BLOCK_SIZE):
                            for x in range(SUB_BLOCK_SIZE):
                                if num_sb_labels_cur == 0:
                                    label = 0
                                elif num_sb_labels_cur == 1:
                                    label = sb_labels[0]
                                else:
                                    index = get_packed_value(sb_values, bit_pos, bits)
                                    label = sb_labels[index]
                                    bit_pos += bits

                                output[base_z + z, base_y + y, base_x + x] = label

                    if bit_pos % 8 != 0:
                        bit_pos += 8 - (bit_pos % 8)

                    sub_block_num += 1

        return output

    def get_block_info(self, compressed_data: bytes) -> dict:
        """
        Extract metadata from a two-layer compressed DVID block without full decompression.

        Args:
            compressed_data: The zstd-compressed DVID block data

        Returns:
            Dictionary with block metadata
        """
        try:
            if len(compressed_data) == 0:
                return {'type': 'empty', 'zstd_size': 0}

            try:
                dvid_data = self._zstd_decompressor.decompress(compressed_data)
            except Exception as e:
                return {
                    'type': 'zstd_error',
                    'error': str(e),
                    'zstd_size': len(compressed_data)
                }

            if len(dvid_data) < 16:
                return {'type': 'invalid', 'zstd_size': len(compressed_data), 'dvid_size': len(dvid_data)}

            gx, gy, gz, N = struct.unpack('<IIII', dvid_data[:16])

            return {
                'type': 'solid' if N <= 1 else 'compressed',
                'subblocks': (gx, gy, gz),
                'label_count': N,
                'block_size': (gx * 8, gy * 8, gz * 8),
                'zstd_compressed_size': len(compressed_data),
                'dvid_uncompressed_size': len(dvid_data)
            }

        except Exception as e:
            return {
                'type': 'error',
                'error': str(e),
                'zstd_compressed_size': len(compressed_data)
            }
