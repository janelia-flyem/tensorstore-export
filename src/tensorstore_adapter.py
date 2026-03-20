"""
TensorStore helpers for neuroglancer precomputed volumes.

This module provides utility functions for opening neuroglancer precomputed
volumes on GCS via TensorStore.  The info file must already exist (created
by scripts/setup_destination.py).
"""

import structlog
import tensorstore as ts

logger = structlog.get_logger()


def open_precomputed_scale(bucket: str, path: str, scale_index: int) -> ts.TensorStore:
    """Open a neuroglancer precomputed volume at a specific scale.

    The info file must already exist at gs://<bucket>/<path>/info.

    Args:
        bucket: GCS bucket name
        path: Path within bucket to the precomputed volume root
        scale_index: Which scale to open (0 = full resolution)

    Returns:
        TensorStore handle for the requested scale
    """
    spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": {
            "driver": "gcs",
            "bucket": bucket,
            "path": path,
        },
        "scale_index": scale_index,
        "open": True,
    }
    store = ts.open(spec).result()
    logger.info("Opened precomputed scale",
                 bucket=bucket, path=path,
                 scale=scale_index, domain=str(store.domain))
    return store
