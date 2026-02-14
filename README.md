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
現実装は次のパイプラインです。
- PDF からテキスト抽出して `chunks.jsonl` を作成
- ローカルでベクトル索引（FAISS または numpy）を作成
- ベクトル上位候補を取得し、必要に応じて Bedrock Rerank で並び替え
- Bedrock Converse で最終回答を生成

1. 依存インストール
```bash
source /home/user/vpnless-rag-venv/bin/activate
python3 -m pip install -r scripts/rag/requirements.txt
```

2. PDF から chunks.jsonl 生成
```bash
python3 scripts/rag/build_chunks_from_pdfs.py \
  --pdf-dir /home/user/dev/vpnless-rag-mvp/rag_data/pdfs \
  --out /home/user/dev/vpnless-rag-mvp/rag_data/index/chunks.jsonl
```

`build_chunks_from_pdfs.py` の主要既定値:
- `--glob *.pdf`
- `--chunk-size 900`
- `--chunk-overlap 150`
- `--min-chars 80`

注記:
- OCR は行っていないため、画像だけの PDF は `No text chunks were extracted` になります。
- 現実装では `--pdf-dir` 配下の PDF をまとめて索引化します（文書種別の自動フィルタなし）。
  必要な文書だけを検索対象にする場合は、PDF 配置ディレクトリを分けるか `--glob` で絞ってください。

3. ベクトル索引作成
```bash
python3 scripts/rag/build_vector_index.py \
  --chunks /home/user/dev/vpnless-rag-mvp/rag_data/index/chunks.jsonl \
  --index-dir /home/user/dev/vpnless-rag-mvp/rag_data/index
```

`build_vector_index.py` の主要既定値:
- `--embedding-model intfloat/multilingual-e5-small`
- `--batch-size 64`
- `--backend faiss`

索引生成後の成果物:
- `vectors.faiss`（または `vectors.npy`）
- `metadata.jsonl`
- `manifest.json`

4. RAG 実行
```bash
python3 scripts/rag/rag_vector_cli.py \
  --index-dir /home/user/dev/vpnless-rag-mvp/rag_data/index \
  "質問文"
```

モデル切替例:
```bash
# コスト優先（既定）
python3 scripts/rag/rag_vector_cli.py \
  --index-dir /home/user/dev/vpnless-rag-mvp/rag_data/index \
  --answer-profile cost \
  "質問文"

# 精度優先（Gemma 27B）
python3 scripts/rag/rag_vector_cli.py \
  --index-dir /home/user/dev/vpnless-rag-mvp/rag_data/index \
  --answer-profile high \
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

挙動メモ:
- `Rerank` は候補の並び替えです。現実装では rerank で選ばれなかった候補も末尾に残します。
- `=== TOPK EVIDENCE ===` に表示された根拠だけを使って回答させるプロンプトです。
- 根拠が空の場合のみ `Evidence is insufficient.` を返します。

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

## 11. 現在の制約（検証プランPDFとの差分）
この README は **現在実装済みの挙動** を正として記載しています。  
`rag_data/appendix/AWS Bedrock & VAST DATA検証プラン.pdf` は検証計画であり、以下は現実装との差分です（本文では製品名を `VAST Data` と表記）。

| 項目 | 現在の実装状態 | 補足 |
|---|---|---|
| 検索基盤（VAST Data / NetApp / マネージド連携） | 未実装 | WSL ローカル索引（FAISS / numpy）のみ |
| ベクトルDB（マネージド） | 未実装 | OpenSearch Serverless / Aurora pgvector など未接続 |
| UI（OpenWebUI） | 未実装 | 実行インターフェースは CLI のみ |
| API 受け口（`/search`） | 未実装 | サービス化していない |
| ACL / metadata フィルタ検索 | 未実装 | 索引内の全チャンクが候補。文書種別での絞り込みは手動運用 |
| インデクシング更新 | 未実装（自動） | 手動で PDF 取り込み → `chunks.jsonl` 再生成 → 索引再作成 |
| 監査ログ設計（相関ID/TTL） | 未実装 | `scripts/audit/collect_linux.sh` での収集中心 |
| 情報最小化統制（Tier分類など） | 部分実装 | メール/電話の簡易マスクのみ |

## 12. 現時点で実装済みの範囲
- WSL ローカルでの PDF 取り込み、チャンク化、ベクトル索引作成。
- ベクトル検索 + Bedrock Rerank（`amazon.rerank-v1:0`）で候補並び替え。
- Bedrock Converse での回答生成（`cost` / `high` モデル切替）。
- Terraform による予算管理と Bedrock 呼び出し IAM 管理。

## 13. 今後の実装候補（優先順）
1. P0: metadata フィルタ（文書カテゴリ・有効日・部門など）を検索時に適用。
2. P0: `/search` API 化（CLI 依存を解消し、UI/外部連携可能にする）。
3. P1: VAST Data / NetApp 連携、またはマネージドベクトルDB連携（OpenSearch Serverless など）を追加。
4. P1: 差分更新前提のイベント駆動インデクシング。
5. P2: 監査ログ運用（相関ID、保持期間、マスク方針）の実装。
