aws bedrock-runtime converse `
  --region ap-northeast-1 `
  --profile rag `
  --model-id google.gemma-3-4b-it `
  --cli-input-json file://scripts/smoke/converse.json `
  --no-cli-pager `
  --query "output.message.content[0].text" `
  --output text