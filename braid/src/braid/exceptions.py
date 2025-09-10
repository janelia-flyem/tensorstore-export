"""
Exception classes for BRAID library.
"""


class BraidError(Exception):
    """Base exception for BRAID library errors."""
    pass


class ChunkNotFoundError(BraidError):
    """Raised when a requested chunk is not found in the shard."""
    pass


class DecompressionError(BraidError):
    """Raised when block decompression fails."""
    pass


class InvalidShardFormatError(BraidError):
    """Raised when shard files have invalid format."""
    pass


class InvalidCoordinateError(BraidError):
    """Raised when invalid chunk coordinates are provided."""
    pass