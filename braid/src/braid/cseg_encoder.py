"""
Neuroglancer compressed_segmentation encoder with DVID fused path.

Wraps the C extension in libbraid_codec.so which provides:
  - cseg_encode_chunk: uint64 ZYX volume → compressed_segmentation bytes
  - dvid_to_cseg: [zstd] DVID block → compressed_segmentation [gzip]
  - cseg_max_encoded_size: upper bound on output size
"""

import ctypes
from pathlib import Path

import numpy as np

# Flags matching C #defines in cseg_encode.c
BRAID_INPUT_ZSTD = 1
BRAID_OUTPUT_GZIP = 2

_c_lib = None


def _load_c_lib():
    """Load libbraid_codec.so, trying known paths."""
    global _c_lib
    if _c_lib is not None:
        return _c_lib

    search_paths = [
        Path(__file__).resolve().parent.parent.parent / "csrc" / "libbraid_codec.so",
        Path(__file__).resolve().parent / "libbraid_codec.so",
    ]

    for path in search_paths:
        if path.exists():
            try:
                lib = ctypes.CDLL(str(path))
            except OSError:
                continue

            lib.cseg_encode_chunk.restype = ctypes.c_int
            lib.cseg_encode_chunk.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),  # input
                ctypes.c_int, ctypes.c_int, ctypes.c_int,  # nz, ny, nx
                ctypes.c_int, ctypes.c_int, ctypes.c_int,  # bz, by, bx
                ctypes.c_char_p,                   # output
                ctypes.c_size_t,                   # output_cap
                ctypes.POINTER(ctypes.c_size_t),   # output_size
            ]

            lib.dvid_to_cseg.restype = ctypes.c_int
            lib.dvid_to_cseg.argtypes = [
                ctypes.c_char_p, ctypes.c_size_t,              # dvid_data, dvid_len
                ctypes.POINTER(ctypes.c_uint64),               # sv_map_keys (or NULL)
                ctypes.POINTER(ctypes.c_uint64),               # sv_map_vals (or NULL)
                ctypes.c_size_t,                               # num_map_entries
                ctypes.c_int, ctypes.c_int, ctypes.c_int,     # bz, by, bx
                ctypes.c_int,                                  # flags
                ctypes.c_char_p, ctypes.c_size_t,              # output, output_cap
                ctypes.POINTER(ctypes.c_size_t),               # output_size
            ]

            lib.cseg_max_encoded_size.restype = ctypes.c_size_t
            lib.cseg_max_encoded_size.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int,  # nz, ny, nx
                ctypes.c_int, ctypes.c_int, ctypes.c_int,  # bz, by, bx
            ]

            _c_lib = lib
            return lib

    return None


class CSEGEncoder:
    """Neuroglancer compressed_segmentation encoder with optional DVID fused path."""

    def __init__(self):
        self._lib = _load_c_lib()
        if self._lib is None:
            raise RuntimeError(
                "libbraid_codec.so not found. Run 'pixi run build-braid-c' first."
            )

    @staticmethod
    def max_encoded_size(shape, block_size=(8, 8, 8)):
        """Upper bound on encoded output size in bytes (before gzip).

        Args:
            shape: (nz, ny, nx) volume dimensions.
            block_size: (bz, by, bx) encoding block size.
        """
        lib = _load_c_lib()
        if lib is None:
            # Fallback calculation
            nz, ny, nx = shape
            bz, by, bx = block_size
            gz = (nz + bz - 1) // bz
            gy = (ny + by - 1) // by
            gx = (nx + bx - 1) // bx
            nb = gz * gy * gx
            bv = bz * by * bx
            return 4 + nb * 8 + nb * ((16 * bv + 31) // 32 + bv * 2) * 4
        return lib.cseg_max_encoded_size(*shape, *block_size)

    def encode_chunk(self, data, block_size=(8, 8, 8)):
        """Encode a uint64 ZYX volume to compressed_segmentation bytes.

        Args:
            data: uint64 array of shape (nz, ny, nx), C-contiguous, ZYX order.
            block_size: (bz, by, bx) encoding block size.

        Returns:
            bytes: compressed_segmentation encoded bytes (single channel).
        """
        data = np.ascontiguousarray(data, dtype=np.uint64)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array, got {data.ndim}D")
        nz, ny, nx = data.shape
        bz, by, bx = block_size

        max_size = self._lib.cseg_max_encoded_size(nz, ny, nx, bz, by, bx)
        output = ctypes.create_string_buffer(max_size)
        output_size = ctypes.c_size_t(0)

        ret = self._lib.cseg_encode_chunk(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            nz, ny, nx, bz, by, bx,
            output, max_size, ctypes.byref(output_size),
        )
        if ret != 0:
            raise RuntimeError("cseg_encode_chunk failed")

        return output.raw[:output_size.value]

    def dvid_to_cseg(self, dvid_data, block_size=(8, 8, 8),
                     supervoxels=None, agglo_labels=None,
                     zstd_input=True, gzip_output=False):
        """Convert a DVID compressed block to compressed_segmentation bytes.

        The DVID block header (gx, gy, gz, N, labels) is parsed internally.
        If a supervoxel-to-agglomerated mapping is provided, it is applied
        to the block's label array before encoding.

        Args:
            dvid_data: DVID block bytes (optionally zstd-compressed).
            block_size: (bz, by, bx) cseg encoding block size.
            supervoxels: uint64 array of supervoxel IDs (mapping keys).
                If None, block labels are used as-is (identity mapping).
            agglo_labels: uint64 array of agglomerated label IDs (mapping
                values, same length as supervoxels).
            zstd_input: True if dvid_data is zstd-compressed.
            gzip_output: True to gzip the output bytes.

        Returns:
            bytes: compressed_segmentation encoded bytes
                   (gzip-compressed if gzip_output=True).
        """
        bz, by, bx = block_size
        flags = 0
        if zstd_input:
            flags |= BRAID_INPUT_ZSTD
        if gzip_output:
            flags |= BRAID_OUTPUT_GZIP

        # Prepare mapping arrays (or NULL pointers for identity)
        if supervoxels is not None and agglo_labels is not None:
            sv_keys = np.ascontiguousarray(supervoxels, dtype=np.uint64)
            sv_vals = np.ascontiguousarray(agglo_labels, dtype=np.uint64)
            if len(sv_keys) != len(sv_vals):
                raise ValueError(
                    f"supervoxels ({len(sv_keys)}) and agglo_labels "
                    f"({len(sv_vals)}) must have the same length"
                )
            sv_keys_ptr = sv_keys.ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint64))
            sv_vals_ptr = sv_vals.ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint64))
            num_map = len(sv_keys)
        else:
            sv_keys_ptr = None
            sv_vals_ptr = None
            num_map = 0

        # Output buffer: use 64^3 with 8^3 blocks as conservative estimate
        # (the actual volume shape is parsed from the DVID header in C)
        max_size = self._lib.cseg_max_encoded_size(64, 64, 64, bz, by, bx)
        output = ctypes.create_string_buffer(max_size)
        output_size = ctypes.c_size_t(0)

        ret = self._lib.dvid_to_cseg(
            dvid_data, len(dvid_data),
            sv_keys_ptr, sv_vals_ptr, num_map,
            bz, by, bx,
            flags,
            output, max_size, ctypes.byref(output_size),
        )
        if ret != 0:
            raise RuntimeError("dvid_to_cseg failed")

        return output.raw[:output_size.value]
