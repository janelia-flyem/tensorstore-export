#!/bin/bash

# Cloud Run Job deployment script for TensorStore DVID Export Worker
# This script builds and deploys the worker as a Cloud Run Job

set -e

# Configuration - update these values for your deployment
PROJECT_ID="${PROJECT_ID:-your-gcp-project}"
JOB_NAME="${JOB_NAME:-tensorstore-dvid-export}"
REGION="${REGION:-us-central1}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${JOB_NAME}"

# Job configuration
MAX_RETRIES="${MAX_RETRIES:-3}"
TASK_TIMEOUT="${TASK_TIMEOUT:-3600s}"  # 1 hour
PARALLELISM="${PARALLELISM:-100}"      # Number of parallel workers
TASK_COUNT="${TASK_COUNT:-100}"        # Total number of job tasks

# Required environment variables for the job
SOURCE_BUCKET="${SOURCE_BUCKET:-your-source-bucket}"
DEST_BUCKET="${DEST_BUCKET:-your-dest-bucket}"
DEST_PATH="${DEST_PATH:-neuroglancer-volume}"
VOLUME_SHAPE="${VOLUME_SHAPE:-134576,78317,94088}"  # z,y,x for male CNS

echo "Deploying TensorStore DVID Export Worker..."
echo "Project: ${PROJECT_ID}"
echo "Job Name: ${JOB_NAME}"
echo "Region: ${REGION}"
echo "Image: ${IMAGE_NAME}"

# Build and push Docker image
echo "Building Docker image..."
docker build -t "${IMAGE_NAME}" .

echo "Pushing image to Container Registry..."
docker push "${IMAGE_NAME}"

# Create or update the Cloud Run Job
echo "Deploying Cloud Run Job..."
gcloud run jobs replace <(cat <<EOF
apiVersion: run.googleapis.com/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  labels:
    cloud.googleapis.com/location: ${REGION}
spec:
  template:
    spec:
      parallelism: ${PARALLELISM}
      taskCount: ${TASK_COUNT}
      template:
        spec:
          maxRetries: ${MAX_RETRIES}
          taskTimeoutSeconds: ${TASK_TIMEOUT%s}
          containers:
          - image: ${IMAGE_NAME}
            env:
            - name: SOURCE_BUCKET
              value: "${SOURCE_BUCKET}"
            - name: DEST_BUCKET
              value: "${DEST_BUCKET}"
            - name: DEST_PATH
              value: "${DEST_PATH}"
            - name: VOLUME_SHAPE
              value: "${VOLUME_SHAPE}"
            - name: SHARD_SHAPE
              value: "2048,2048,2048"
            - name: CHUNK_SHAPE
              value: "64,64,64"
            - name: RESOLUTION
              value: "8,8,8"
            - name: MAX_PROCESSING_TIME
              value: "55"
            - name: POLLING_INTERVAL
              value: "10"
            resources:
              limits:
                cpu: "2"
                memory: "4Gi"
          serviceAccountEmail: "${SERVICE_ACCOUNT:-default}"
EOF
) --region="${REGION}" --project="${PROJECT_ID}"

echo "✅ Deployment complete!"
echo ""
echo "To execute the job:"
echo "  gcloud run jobs execute ${JOB_NAME} --region=${REGION} --project=${PROJECT_ID}"
echo ""
echo "To monitor job execution:"
echo "  gcloud run jobs describe ${JOB_NAME} --region=${REGION} --project=${PROJECT_ID}"
echo ""
echo "To view logs:"
echo "  gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name=${JOB_NAME}\" --project=${PROJECT_ID} --limit=100"