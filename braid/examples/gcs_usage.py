#!/usr/bin/env python3
"""
GCS usage example for BRAID library.

This example shows how to read shard files directly from Google Cloud Storage.
"""

import tempfile
import os
from pathlib import Path
from braid import ShardReader, LabelType


def download_gcs_file(gcs_path: str, local_path: str):
    """Download a file from GCS to local storage."""
    try:
        from google.cloud import storage
        
        # Parse GCS path
        if not gcs_path.startswith('gs://'):
            raise ValueError("GCS path must start with gs://")
        
        path_parts = gcs_path[5:].split('/', 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1]
        
        # Download file
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        
        return True
        
    except ImportError:
        print("❌ google-cloud-storage not installed. Install with: pip install google-cloud-storage")
        return False
    except Exception as e:
        print(f"❌ Failed to download {gcs_path}: {e}")
        return False


def main():
    """Demonstrate reading BRAID shards from GCS."""
    
    # Example GCS paths (update as needed)
    gcs_arrow_path = "gs://your-bucket/shards/shard_0_0_0.arrow"
    gcs_csv_path = "gs://your-bucket/shards/shard_0_0_0.csv"
    
    print("BRAID Shard Reader - GCS Example")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Download files from GCS
        local_arrow = temp_dir / "shard.arrow"
        local_csv = temp_dir / "shard.csv"
        
        print(f"Downloading from GCS...")
        print(f"  Arrow: {gcs_arrow_path}")
        print(f"  CSV: {gcs_csv_path}")
        
        if not download_gcs_file(gcs_arrow_path, str(local_arrow)):
            return 1
        if not download_gcs_file(gcs_csv_path, str(local_csv)):
            return 1
        
        print("✅ Files downloaded successfully")
        print()
        
        try:
            # Initialize reader with local copies
            reader = ShardReader(local_arrow, local_csv)
            
            print(f"✅ Loaded shard with {reader.chunk_count} chunks")
            
            # Process some chunks
            chunks = reader.available_chunks
            if chunks:
                x, y, z = chunks[0]
                
                # Compare labels vs supervoxels
                labels_chunk = reader.read_chunk(x, y, z, LabelType.LABELS)
                supervoxels_chunk = reader.read_chunk(x, y, z, LabelType.SUPERVOXELS)
                
                print(f"Chunk ({x}, {y}, {z}):")
                print(f"  Labels: {len(np.unique(labels_chunk))} unique values")
                print(f"  Supervoxels: {len(np.unique(supervoxels_chunk))} unique values")
                print(f"  Are they different? {not np.array_equal(labels_chunk, supervoxels_chunk)}")
            
        except Exception as e:
            print(f"❌ Error processing shard: {e}")
            return 1
    
    print("\n✅ GCS example completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())