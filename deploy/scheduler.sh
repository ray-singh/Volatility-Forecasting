#!/usr/bin/env bash
# Creates a Cloud Scheduler job that triggers volcast-ingest every Monday at 02:00 UTC.
# Run once after initial deploy.
#
# Usage: bash deploy/scheduler.sh <PROJECT_ID> <REGION>

set -euo pipefail

PROJECT_ID=${1:?Usage: $0 PROJECT_ID REGION}
REGION=${2:?Usage: $0 PROJECT_ID REGION}

gcloud scheduler jobs create http volcast-weekly-ingest \
  --project "$PROJECT_ID" \
  --location "$REGION" \
  --schedule "0 2 * * 1" \
  --uri "https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/volcast-ingest:run" \
  --message-body "{}" \
  --oauth-service-account-email "volcast-ingest-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --time-zone "UTC" \
  --attempt-deadline "3600s"

echo "✓ Cloud Scheduler job created: volcast-weekly-ingest"
