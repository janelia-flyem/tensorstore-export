#!/bin/bash

# Example of exporting segmentation for male CNS, 2026-03-21.
#
# Uses tier-based execution: shards are grouped by memory requirement
# (based on Arrow source file size) and each tier runs with appropriate
# Cloud Run resources.  This replaces per-scale jobs and avoids
# allocating 16Gi to shards that only need 4Gi (99.9% of shards).
#
# Prerequisites:
#   pixi run deploy       # create Cloud Run jobs + info file
#   pixi run precompute-manifest  # scan Arrow files, write manifests

set -e

echo "Step 1: Precompute manifests (scans Arrow file sizes)..."
pixi run precompute-manifest
echo

echo "Step 2: Execute by memory tier..."
echo

# Tier assignments (from mCNS v0.11 Arrow file analysis):
#   4Gi:  26,096 shards (99.9%) — all s0-s2, most s3-s9
#   8Gi:      27 shards         — large s3/s4/s5/s6
#  16Gi:       5 shards         — largest s3/s4/s5
#
# Note: task counts and tier boundaries will be refined after the first
# compressed_segmentation run provides real shard size data.

pixi run generate-scale --scales 0,1,2,3,4,5,6,7,8,9 \
    --tasks 2600 --memory 4Gi --cpu 2 \
    --manifest-uri gs://flyem-male-cns/dvid-exports/mCNS-98d699/segmentation/manifests/tier-4gi

pixi run generate-scale --scales 0,1,2,3,4,5,6,7,8,9 \
    --tasks 27 --memory 8Gi --cpu 2 \
    --manifest-uri gs://flyem-male-cns/dvid-exports/mCNS-98d699/segmentation/manifests/tier-8gi

pixi run generate-scale --scales 0,1,2,3,4,5,6,7,8,9 \
    --tasks 5 --memory 16Gi --cpu 4 \
    --manifest-uri gs://flyem-male-cns/dvid-exports/mCNS-98d699/segmentation/manifests/tier-16gi

echo
echo "All tiers launched. Monitor with:"
echo "  pixi run export-status"
echo "  pixi run export-errors"
echo
echo "Mine memory profiles after completion:"
echo "  pixi run export-errors -- --details | grep 'memory profile'"
