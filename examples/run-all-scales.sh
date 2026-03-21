#!/bin/bash

# Example of exporting segmentation for male CNS, 2026-03-20.
# Execute all scales in parallel with appropriate resource profiles.
# Each scale has its own Cloud Run job, so they run concurrently.

set -e

echo "Launching all scales..."
echo

pixi run generate-scale --scales 0 --tasks 2400 --memory 4Gi --cpu 2
pixi run generate-scale --scales 1 --tasks 600 --memory 4Gi --cpu 2
pixi run generate-scale --scales 2 --tasks 300 --memory 8Gi --cpu 2
pixi run generate-scale --scales 3 --tasks 150 --memory 16Gi --cpu 4
pixi run generate-scale --scales 4,5,6,7,8,9 --tasks 30 --memory 16Gi --cpu 4

echo
echo "All scales launched. Monitor with:"
echo "  pixi run export-status"
echo "  pixi run export-errors"
