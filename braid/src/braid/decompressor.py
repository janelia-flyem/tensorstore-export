"""
DVID Block Decompressor.

Implements the DVID compressed segmentation format decompression with support
for both agglomerated labels and supervoxel labels.

This is a complete implementation of the DVID compressed block format,
equivalent to the Go MakeLabelVolume function.
"""

import struct
from typing import List, Tuple, Optional
import numpy as np
import zstandard as zstd
from .exceptions import DecompressionError


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

    # Left bit masks for each bit position in a byte
    LEFT_BIT_MASK = [0xFF, 0x7F, 0x3F, 0x1F, 0x0F, 0x07, 0x03, 0x01]

    if bit_pos + bits <= 8:
        # Index totally within this byte
        right_shift = 8 - bit_pos - bits
        return (data[byte_pos] & LEFT_BIT_MASK[bit_pos]) >> right_shift
    else:
        # Index spans byte boundaries
        index = (data[byte_pos] & LEFT_BIT_MASK[bit_pos]) << 8
        index |= data[byte_pos + 1]
        index >>= (16 - bit_pos - bits)
        return index


class DVIDDecompressor:
    """
    DVID Block decompressor implementation.

    This class handles the DVID compressed segmentation format as described in
    the DVID documentation. The format supports block-level label lists with
    sub-block indices for efficient storage.

    Block Format Overview:
    - gx, gy, gz: Number of sub-blocks in each dimension (uint32 each)
    - N: Number of labels (uint32)
    - labels: N uint64 labels
    - sub-block data: Variable length compressed indices with bit packing
    """

    def __init__(self):
        """Initialize the DVID decompressor."""
        self._zstd_decompressor = zstd.ZstdDecompressor()

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
            # First layer: decompress zstd to get raw DVID compressed block
            if len(compressed_data) == 0:
                # Empty block - return all zeros
                return np.zeros(block_shape, dtype=np.uint64)

            try:
                dvid_compressed_data = self._zstd_decompressor.decompress(compressed_data)
            except Exception as e:
                raise DecompressionError(f"Failed to decompress zstd layer: {e}")

            # Second layer: decompress DVID segmentation format
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

        This is equivalent to the Go MakeLabelVolume function.
        """
        SUB_BLOCK_SIZE = 8

        if len(compressed_data) < 16:
            raise ValueError("Compressed data too short for header")

        # Parse header: gx, gy, gz, numLabels (4 uint32s in little-endian)
        gx, gy, gz, num_labels = struct.unpack('<IIII', compressed_data[:16])

        # Verify this matches expected block shape
        expected_shape = (gz * SUB_BLOCK_SIZE, gy * SUB_BLOCK_SIZE, gx * SUB_BLOCK_SIZE)
        if expected_shape != block_shape:
            raise ValueError(f"Block shape mismatch: expected {expected_shape}, got {block_shape}")

        if num_labels == 0:
            raise ValueError("Block has 0 labels, which is not allowed")

        # Create output array in ZYX order
        output = np.zeros(block_shape, dtype=np.uint64)

        # Parse labels
        pos = 16
        labels_size = num_labels * 8
        if pos + labels_size > len(compressed_data):
            raise ValueError("Compressed data too short for labels")

        labels = np.frombuffer(compressed_data[pos:pos + labels_size], dtype='<u8')
        pos += labels_size

        # Convert to numpy arrays if needed
        if agglo_labels is not None:
            agglo_labels = np.array(agglo_labels, dtype=np.uint64)
        if supervoxels is not None:
            supervoxels = np.array(supervoxels, dtype=np.uint64)

        # If only 0 or 1 labels, fill entire array
        if num_labels < 2:
            label = labels[0] if num_labels == 1 else 0

            # Apply label mapping if provided
            if agglo_labels is not None and supervoxels is not None:
                sv_matches = np.where(supervoxels == label)[0]
                if len(sv_matches) > 0:
                    sv_idx = sv_matches[0]
                    if sv_idx < len(agglo_labels):
                        label = agglo_labels[sv_idx]

            output.fill(label)
            return output

        # Parse sub-block metadata
        num_sub_blocks = gx * gy * gz

        # NumSBLabels: number of labels per sub-block (uint16 each)
        num_sb_labels_size = num_sub_blocks * 2
        if pos + num_sb_labels_size > len(compressed_data):
            raise ValueError("Compressed data too short for NumSBLabels")

        num_sb_labels = np.frombuffer(compressed_data[pos:pos + num_sb_labels_size], dtype='<u2')
        pos += num_sb_labels_size

        # SBIndices: indices into Labels array (uint32 each)
        total_sb_indices = np.sum(num_sb_labels, dtype=np.uint32)
        sb_indices_size = total_sb_indices * 4
        if pos + sb_indices_size > len(compressed_data):
            raise ValueError("Compressed data too short for SBIndices")

        sb_indices = np.frombuffer(compressed_data[pos:pos + sb_indices_size], dtype='<u4')
        pos += sb_indices_size

        # SBValues: packed sub-block voxel indices
        sb_values = compressed_data[pos:]

        # Decompress each sub-block
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

                    # Get labels for this sub-block
                    for i in range(num_sb_labels_cur):
                        sb_labels[i] = labels[sb_indices[index_pos]]
                        index_pos += 1

                    # Calculate position in output array
                    base_z = sz * SUB_BLOCK_SIZE
                    base_y = sy * SUB_BLOCK_SIZE
                    base_x = sx * SUB_BLOCK_SIZE

                    # Decompress voxels in this sub-block
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

                                # Apply label mapping if provided
                                if agglo_labels is not None and supervoxels is not None:
                                    sv_matches = np.where(supervoxels == label)[0]
                                    if len(sv_matches) > 0:
                                        sv_idx = sv_matches[0]
                                        if sv_idx < len(agglo_labels):
                                            label = agglo_labels[sv_idx]

                                output[base_z + z, base_y + y, base_x + x] = label

                    # Pad to byte boundary
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

            # First decompress zstd layer to get DVID header
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

            # Parse DVID header
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