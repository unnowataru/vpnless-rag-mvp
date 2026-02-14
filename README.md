# vpnless-rag-mvp

このリポジトリは **「オンプレに置いたPDFを根拠に、クラウドAI（AWS Bedrock）で回答するRAG」** を実装するためのものです。  
運用前提は **Linux（Ubuntu推奨。WSL上のUbuntuでも可）** で、原本データはオンプレに保持したまま、クラウドには最小限の情報だけを送ります。

動作イメージ:
- PDF原本はオンプレ側に置く（source-of-truth）
- オンプレ側で関連チャンク（Top-K根拠）を検索
- `質問 + Top-K根拠` だけを Bedrock に渡して回答生成

## 設計原則
本プロジェクトは、次の原則で構成しています。

- データ主権: 原本・メタデータ・権限情報の正本はオンプレ側に置く。
- 最小開示: クラウドへ送る情報は「質問 + 許可済みTop-K根拠」を原則とし、必要に応じてマスクで制御する。
- 統制と説明可能性: 最小権限と監査可能な運用を前提にし、将来の相関ID/TTL制御へ拡張可能な形で設計する。
- 可搬性: 計算リソースの配置は固定せず、将来のオンプレ回帰/ハイブリッド継続に対応できる構成を志向する。

## 目的
- オンプレ側を source-of-truth とし、原本ファイルを AWS に保存しない。
- Bedrock には「質問 + マスク済み Top-K 根拠チャンク」のみ送信する。

## スコープ（MVP）
- 索引作成・検索はローカル Linux で実行。
- AWS 側に S3 / Knowledge Base / OpenSearch を常設しない。
- Terraform では予算管理と Bedrock 呼び出し IAM のみを管理。

## 前提環境（Linux）
- Ubuntu 22.04+（WSL2 上の Ubuntu を含む）
- Python 3.10+
- AWS CLI v2
- Terraform 1.6+

## リポジトリ構成
- `infra/live/prod`: Terraform ルートモジュール（Budget + IAM）
- `infra/modules`: Terraform 共通モジュール
- `scripts/connectivity`: Bedrock 疎通テスト
- `scripts/rag`: ベクトル索引作成 + RAG 実行
- `scripts/audit`: Linux 監査情報収集

## AWS プロファイル
- `tf-admin`: Terraform 実行用
- `rag`: Bedrock 推論実行用

例（`~/.aws/config`）:
```ini
[profile tf-admin]
region = ap-northeast-1

[profile rag]
region = ap-northeast-1
```

## Terraform（Linux から実行）
```bash
cd infra/live/prod
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

`infra/root` からの移行手順は `infra/live/prod/README.md` を参照してください。

Terraform 管理対象:
- 月額予算: `90 USD`
- 通知閾値: `45 / 70 / 85`
- IAM ユーザー: `rag-bedrock-invoker`
- Bedrock 呼び出し許可モデル: `google.gemma-3-4b-it`, `google.gemma-3-27b-it`
- Rerank モデル: `amazon.rerank-v1:0`

## Bedrock 疎通テスト（Linux）
```bash
bash scripts/connectivity/bedrock_converse.sh
```

環境変数で上書き可能:
- `AWS_PROFILE`（既定: `rag`）
- `AWS_REGION`（既定: `ap-northeast-1`）
- `BEDROCK_MODEL_ID`（既定: `google.gemma-3-4b-it`）

## ベクトルRAG実行
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

2. PDF 配置先（先に作成）
```bash
mkdir -p /home/user/dev/vpnless-rag-mvp/rag_data/pdfs
mkdir -p /home/user/dev/vpnless-rag-mvp/rag_data/index
```

配置パス:
- Linux パス（推奨）: `/home/<linux-user>/dev/vpnless-rag-mvp/rag_data/pdfs`
- リポジトリ相対: `rag_data/pdfs`
- Windows から WSL を開く場合: `\\wsl.localhost\\Ubuntu-24.04\\home\\<linux-user>\\dev\\vpnless-rag-mvp\\rag_data\\pdfs`

3. PDF から chunks.jsonl 生成
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

4. ベクトル索引作成
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

5. RAG 実行
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

## Linux監査収集
```bash
bash scripts/audit/collect_linux.sh
```

## 秘密情報と成果物
- 秘密情報はリポジトリ外で管理する。
- `.gitignore` で以下を除外:
  - `rag_data/`
  - `*.jsonl`, `*.pdf`, `*.faiss`, `*.npy`
  - `logs/`

## 差分・実装計画（Issue）
検証プランPDFとの差分と今後の実装候補は、更新履歴を残しやすいよう GitHub Issue で管理します。

- 差分トラッカー: `#1` https://github.com/unnowataru/vpnless-rag-mvp/issues/1
- P0: metadataフィルタ検索: `#2` https://github.com/unnowataru/vpnless-rag-mvp/issues/2
- P0: `/search` API: `#3` https://github.com/unnowataru/vpnless-rag-mvp/issues/3
- P1: VAST Data / NetApp / マネージド連携: `#4` https://github.com/unnowataru/vpnless-rag-mvp/issues/4
- P1: イベント駆動インデクシング: `#5` https://github.com/unnowataru/vpnless-rag-mvp/issues/5
- P2: 監査ログ運用（相関ID・TTL・マスク）: `#6` https://github.com/unnowataru/vpnless-rag-mvp/issues/6
