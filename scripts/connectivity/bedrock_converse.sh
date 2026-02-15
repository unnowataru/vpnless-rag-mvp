#!/usr/bin/env bash
set -euo pipefail

payload='{
  "messages": [
    { "role": "user", "content": [ { "text": "Reply with exactly: OK" } ] }
  ],
  "inferenceConfig": { "maxTokens": 32, "temperature": 0 }
}'

aws bedrock-runtime converse \
  --region "${AWS_REGION:-ap-northeast-1}" \
  --profile "${AWS_PROFILE:-rag}" \
  --model-id "${BEDROCK_MODEL_ID:-google.gemma-3-4b-it}" \
  --cli-input-json "${payload}" \
  --no-cli-pager \
  --query "output.message.content[0].text" \
  --output text
