#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

aws bedrock-runtime converse \
  --region "${AWS_REGION:-ap-northeast-1}" \
  --profile "${AWS_PROFILE:-rag}" \
  --model-id "${BEDROCK_MODEL_ID:-google.gemma-3-4b-it}" \
  --cli-input-json "file://${SCRIPT_DIR}/converse.json" \
  --no-cli-pager \
  --query "output.message.content[0].text" \
  --output text
