"""
TensorStore DVID Shard Export Worker

A distributed system for converting DVID export shards to Neuroglancer
precomputed volumes.  Uses the BRAID library for Arrow IPC shard reading
and DVID block decompression.
"""

__version__ = "1.0.0"
