# VPNレスRAG (Plan A) グランドデザイン / Grand Design

## 1. Purpose / 目的
- EN: On-prem NFS is the source-of-truth. AWS Bedrock is used only for generation via outbound HTTPS.
- JP: 正本はオンプレNFS。AWS BedrockはアウトバウンドHTTPSで生成のみ利用。

## 2. Scope / スコープ
- EN: Local indexing + retrieval on-prem. No S3/KB/OpenSearch in AWS for MVP.
- JP: オンプレで索引/検索。MVPではAWS側にS3/KB/OpenSearchを常設しない。

## 3. Data policy / データ方針
- EN: Send only question + masked Top-K evidence chunks. Never send raw files.
- JP: 送信は「質問 + マスク済Top-K根拠」のみ。原本ファイルは送らない。

## 4. Cloud (AWS) / クラウド側
- Terraform-managed:
  - Budgets ($90; alerts 45/70/85)
  - IAM: `rag-bedrock-invoker` restricted to `google.gemma-3-4b-it`
- Manual:
  - Access key issuance (kept out of repo)

## 5. On-prem / オンプレ側
- Components:
  - Extractor (PDF/text)
  - Chunker
  - Embedder (local)
  - Vector index (FAISS)
  - Retriever (Top-K)
  - Sanitizer (mask/limits)
  - Bedrock client (Converse API)

## 6. Operations / 運用
- Smoke test: `scripts/smoke/*`
- Cost guard: Budget + token/call limits in app
