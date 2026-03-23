"""
BRAID: Block Records Arrow Indexed Dataset

A Python library for reading chunked segmentation data from Arrow files with 
CSV coordinate indices. Supports efficient chunk-wise access and multiple 
compression backends.

BRAID provides a unified interface for reading sharded Arrow files where 
individual block records are "braided" together into indexed datasets.
"""

__version__ = "1.0.0"
__author__ = "BRAID Development Team"

from .reader import ShardReader, ShardRangeReader, LabelType
from .decompressor import DVIDDecompressor
from .cseg_encoder import CSEGEncoder
from .exceptions import BraidError, ChunkNotFoundError, DecompressionError

__all__ = [
    "ShardReader",
    "ShardRangeReader",
    "LabelType",
    "DVIDDecompressor",
    "CSEGEncoder",
    "BraidError",
    "ChunkNotFoundError",
    "DecompressionError",
]