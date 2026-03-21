#!/bin/bash

# Example of exporting segmentation for male CNS.
#
# Prerequisites:
#   pixi run deploy   # build image, write info file, create base Cloud Run job
#
# This single command scans all Arrow source files, assigns shards to
# memory tiers (1-32 GiB), writes per-task manifests to GCS, and
# launches a Cloud Run job per tier.

set -e

pixi run export

# After completion, mine memory profiles to tune tier assignments:
#   pixi run export-errors -- --details | grep "memory profile"
