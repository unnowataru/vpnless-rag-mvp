# vpnless-rag-mvp

このリポジトリは **WSL（Linux）専用**で運用する前提です。  
オンプレを正本とし、AWS Bedrock は生成処理のみに使います。

## 1. 目的
- オンプレ側を source-of-truth とし、原本ファイルを AWS に保存しない。
- Bedrock には「質問 + マスク済み Top-K 根拠チャンク」のみ送信する。

## 2. スコープ（MVP）
- 索引作成・検索はローカル（WSL/Linux）で実行。
- AWS 側に S3 / Knowledge Base / OpenSearch を常設しない。
- Terraform では予算管理と Bedrock 呼び出し IAM のみを管理。

## 3. 前提環境（WSL）
- WSL2（Ubuntu 推奨）
- Python 3.10+
- AWS CLI v2
- Terraform 1.6+

## 4. リポジトリ構成
- `infra/root`: Terraform（Budget + IAM）
- `scripts/connectivity`: Bedrock 疎通テスト
- `scripts/rag`: ベクトル索引作成 + RAG 実行
- `scripts/audit`: Linux 監査情報収集

## 5. AWS プロファイル
- `tf-admin`: Terraform 実行用
- `rag`: Bedrock 推論実行用

例（`~/.aws/config`）:
```ini
[profile tf-admin]
region = ap-northeast-1

[profile rag]
region = ap-northeast-1
```

## 6. Terraform（WSL から実行）
```bash
cd infra/root
terraform init
terraform plan
terraform apply
```

実装済み内容:
- 月額予算: `90 USD`
- 通知閾値: `45 / 70 / 85`
- IAM ユーザー: `rag-bedrock-invoker`
- Bedrock 呼び出し許可モデル: `google.gemma-3-4b-it`, `google.gemma-3-27b-it`
- Rerank モデル: `amazon.rerank-v1:0`

## 7. Bedrock 疎通テスト（WSL）
```bash
bash scripts/connectivity/bedrock_converse.sh
```

環境変数で上書き可能:
- `AWS_PROFILE`（既定: `rag`）
- `AWS_REGION`（既定: `ap-northeast-1`）
- `BEDROCK_MODEL_ID`（既定: `google.gemma-3-4b-it`）

## 8. ベクトルRAG実行
1. 依存インストール
```bash
python3 -m pip install -r scripts/rag/requirements.txt
```

2. PDF から chunks.jsonl 生成
```bash
python3 scripts/rag/build_chunks_from_pdfs.py \
  --pdf-dir /home/user/dev/vpnless-rag-mvp/rag_data/pdfs \
  --out /home/user/dev/vpnless-rag-mvp/rag_data/index/chunks.jsonl
```

3. ベクトル索引作成
```bash
python3 scripts/rag/build_vector_index.py \
  --chunks /path/to/chunks.jsonl \
  --index-dir /path/to/rag_index
```

4. RAG 実行
```bash
python3 scripts/rag/rag_vector_cli.py \
  --index-dir /path/to/rag_index \
  "質問文"
```

`rag_vector_cli.py` の主要既定値:
- `--topk 5`
- `--rerank`（既定: 有効、無効化は `--no-rerank`）
- `--rerank-model amazon.rerank-v1:0`
- `--rerank-topn 0`（0 はベクトル候補を全件 rerank）
- `--answer-profile cost`（`cost=google.gemma-3-4b-it`, `high=google.gemma-3-27b-it`）
- `--bedrock-model`（明示指定時は `--answer-profile` より優先）
- `--max-context-chars 12000`
- `--max-tokens 512`
- `--region ap-northeast-1`
- `--profile rag`

## 9. Linux監査収集
```bash
bash scripts/audit/collect_linux.sh
```

## 10. 秘密情報と成果物
- 秘密情報はリポジトリ外で管理する。
- `.gitignore` で以下を除外:
  - `rag_data/`
  - `*.jsonl`, `*.pdf`, `*.faiss`, `*.npy`
  - `logs/`
