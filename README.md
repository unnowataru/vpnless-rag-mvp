# vpnless-rag-mvp (Plan A) / VPNレスRAG MVP (Plan A)

## EN: Overview
- Cloud: AWS Bedrock (Tokyo) + Terraform
- On-prem: Linux VM (NFS mount) + local vector index (FAISS, etc.)
- Policy: Keep source-of-truth on-prem. Never persist raw files on AWS.
  Send only: user question + masked Top-K evidence chunks.

## JP: 概要
- Cloud: AWS Bedrock(東京) + Terraform
- On-prem: Linux VM(NFSマウント) + ローカルベクトル索引(FAISS等)
- 方針: 正本はオンプレ。AWSへ原本は置かない。
  送信するのは「質問 + マスク済Top-K根拠チャンク」のみ。

---

## EN: Quick start (Windows)
### AWS profiles
- `tf-admin`: Terraform apply
- `rag`: Bedrock invoker (restricted to Gemma model ARN)

### Terraform
- Path: `infra/root`
- Budget: $90 (alerts: 45/70/85)
- IAM: `rag-bedrock-invoker` policy restricted to `google.gemma-3-4b-it`

### Smoke test
- Input: `scripts/smoke/converse.json`
- Run: `powershell -ExecutionPolicy Bypass -File scripts\smoke\bedrock_converse.ps1`

## JP: クイックスタート(Windows)
### AWSプロファイル
- `tf-admin`: Terraform適用用
- `rag`: Bedrock呼び出し専用(対象モデルをGemmaに限定)

### Terraform
- パス: `infra/root`
- 予算: $90(通知: 45/70/85)
- IAM: `rag-bedrock-invoker` は `google.gemma-3-4b-it` のみ許可

### スモークテスト
- 入力: `scripts/smoke/converse.json`
- 実行: `powershell -ExecutionPolicy Bypass -File scripts\smoke\bedrock_converse.ps1`

---

## EN/JP: Secrets policy / 秘密情報の扱い
- Do NOT store secrets in this repo.
- Store secrets outside the repo (example):
  - `C:\dev\_secrets\vpnless-rag-mvp\`
- Repo contains only templates (`*.example`, etc.)
