#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="${HOME}/reddit-gradadmissions-distress"
CODE_DIR="${PROJECT_DIR}/Code/exposure"
LOG_DIR="${PROJECT_DIR}/logs/exposure"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}" "${PROJECT_DIR}/Data/exposure"
export PYTHONUNBUFFERED=1

.venv/bin/python "${CODE_DIR}/extract_comment_events.py" \
    2>&1 | tee "${LOG_DIR}/extract_comment_events.log"

.venv/bin/python "${CODE_DIR}/build_user_cohorts.py" \
    2>&1 | tee "${LOG_DIR}/build_user_cohorts.log"

echo "Exposure and candidate-control extraction completed"
