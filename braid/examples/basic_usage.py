#!/usr/bin/env python3
"""
Basic usage example for BRAID library.

This example demonstrates how to use BRAID to read chunks from 
sharded Arrow files with both agglomerated labels and supervoxels.
"""

from pathlib import Path
import numpy as np
from braid import ShardReader, LabelType


def main():
    """Demonstrate basic usage of BRAID ShardReader."""
    
    # Example shard files (update paths as needed)
    arrow_path = "shard_0_0_0.arrow"
    csv_path = "shard_0_0_0.csv"
    
    print("BRAID Shard Reader Example")
    print("=" * 40)
    
    try:
        # Initialize the reader
        print(f"Loading shard: {arrow_path}")
        reader = ShardReader(arrow_path, csv_path)
        
        print(f"✅ Successfully loaded shard with {reader.chunk_count} chunks")
        print(f"Reader: {reader}")
        print()
        
        # Show available chunks
        chunks = reader.available_chunks
        print(f"First 10 available chunks: {chunks[:10]}")
        print()
        
        if chunks:
            # Read a chunk with agglomerated labels
            x, y, z = chunks[0]
            print(f"Reading chunk at ({x}, {y}, {z}) with LABELS...")
            
            chunk_labels = reader.read_chunk(x, y, z, label_type=LabelType.LABELS)
            print(f"✅ Labels chunk shape: {chunk_labels.shape}")
            print(f"   Dtype: {chunk_labels.dtype}")
            print(f"   Unique values: {len(np.unique(chunk_labels))}")
            print(f"   Min/Max: {chunk_labels.min()} / {chunk_labels.max()}")
            print()
            
            # Read the same chunk with supervoxels
            print(f"Reading chunk at ({x}, {y}, {z}) with SUPERVOXELS...")
            chunk_supervoxels = reader.read_chunk(x, y, z, label_type=LabelType.SUPERVOXELS)
            print(f"✅ Supervoxels chunk shape: {chunk_supervoxels.shape}")
            print(f"   Dtype: {chunk_supervoxels.dtype}")
            print(f"   Unique values: {len(np.unique(chunk_supervoxels))}")
            print(f"   Min/Max: {chunk_supervoxels.min()} / {chunk_supervoxels.max()}")
            print()
            
            # Get chunk metadata
            print("Chunk metadata:")
            chunk_info = reader.get_chunk_info(x, y, z)
            for key, value in chunk_info.items():
                print(f"   {key}: {value}")
            print()
            
            # Get raw chunk data
            print("Raw chunk data:")
            raw_data = reader.read_chunk_raw(x, y, z)
            print(f"   Labels count: {len(raw_data['labels'])}")
            print(f"   Supervoxels count: {len(raw_data['supervoxels'])}")
            print(f"   Compressed size: {len(raw_data['compressed_data'])} bytes")
            print(f"   Uncompressed size: {raw_data['uncompressed_size']} bytes")
            print()
            
            # Test different chunks
            print("Testing multiple chunks...")
            for i, (x, y, z) in enumerate(chunks[:5]):
                if reader.has_chunk(x, y, z):
                    chunk_data = reader.read_chunk(x, y, z, LabelType.LABELS)
                    unique_labels = len(np.unique(chunk_data))
                    print(f"   Chunk {i+1} ({x}, {y}, {z}): {unique_labels} unique labels")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    print("\n✅ Example completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())