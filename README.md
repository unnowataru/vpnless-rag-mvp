# vpnless-rag-mvp

このリポジトリは **「オンプレに置いたPDFを根拠に、クラウドAI（AWS Bedrock）で回答するRAG」** を実装するためのものです。  
推奨環境は **Linux（Ubuntu推奨。WSL上のUbuntuでも可）** で、原本データはオンプレに保持したまま、クラウドには最小限の情報だけを送ります。

動作イメージ:
- PDF原本はオンプレ側に置く（source-of-truth）
- オンプレ側で関連チャンク（Top-K根拠）を検索
- `質問 + Top-K根拠` だけを Bedrock に渡して回答生成

## 目次
- [クイックスタート（すぐに試すならこちら）](#quickstart)
- [設計原則](#design-principles)
- [スコープ（MVP）](#scope)
- [インフラ観点の配置と責務](#infra-roles)
- [データフロー（インフラ向け）](#data-flow)
- [動作環境（Linux）](#prerequisites)
- [AWSアカウント側の準備](#aws-account-prep)
- [セットアップ手順（詳細）](#initial-setup)
- [AWS認証（プロファイル / アクセスキー）](#aws-auth)
- [Terraform（Linux から実行）](#terraform)
- [Bedrock 疎通テスト（Linux）](#bedrock-connectivity)
- [ベクトルRAG実行](#vector-rag)
- [Linux監査収集](#linux-audit)
- [トラブル時にまず確認する項目](#troubleshooting)
- [秘密情報と成果物](#secrets-and-artifacts)
- [差分・実装計画（Issue）](#issues)

<a id="quickstart"></a>
## クイックスタート（すぐに試すならこちら）
```bash
cd /home/<linux-user>/dev/vpnless-rag-mvp
source /home/<linux-user>/vpnless-rag-venv/bin/activate

# 1) PDF投入
mkdir -p rag_data/pdfs rag_data/index

# 2) チャンク化
python3 scripts/rag/build_chunks_from_pdfs.py --pdf-dir rag_data/pdfs --out rag_data/index/chunks.jsonl

# 3) 索引作成
python3 scripts/rag/build_vector_index.py --chunks rag_data/index/chunks.jsonl --index-dir rag_data/index

# 4) 疎通テスト
bash scripts/connectivity/bedrock_converse.sh

# 5) RAG実行
python3 scripts/rag/rag_vector_cli.py --index-dir rag_data/index "質問文"
```

<a id="design-principles"></a>
## 設計原則
本プロジェクトは、次の原則で構成しています。

- データ主権: 原本・メタデータ・権限情報の正本はオンプレ側に置く。
- 最小開示: クラウドへ送る情報は「質問 + 許可済みTop-K根拠」を原則とし、必要に応じてマスクで制御する。
- 統制と説明可能性: 最小権限と監査可能な運用を基本にし、将来の相関ID/TTL制御へ拡張しやすい形で設計する。
- 可搬性: 計算リソースの配置は固定せず、将来のオンプレ回帰/ハイブリッド継続に対応できる構成を志向する。

## 目的
- オンプレ側を source-of-truth とし、原本ファイルを AWS に保存しない。
- Bedrock には「質問 + マスク済み Top-K 根拠チャンク」のみ送信する。

<a id="scope"></a>
## スコープ（MVP）
- 索引作成・検索はローカル Linux で実行。
- AWS 側に S3 / Knowledge Base / OpenSearch を常設しない。
- Terraform では予算管理と Bedrock 呼び出し IAM のみを管理。

<a id="infra-roles"></a>
## インフラ観点の配置と責務
| 区分 | 配置場所 | このMVPでの実体 | 常時課金 |
|---|---|---|---|
| 原本PDF | Linux ローカル（オンプレ側） | `rag_data/pdfs` | なし |
| チャンク/索引 | Linux ローカル（オンプレ側） | `rag_data/index/chunks.jsonl`, `vectors.faiss` など | なし |
| 生成AI実行 | AWS Bedrock | `converse`, `rerank` API 呼び出し | 呼び出し時のみ |
| クラウド永続ストレージ | AWS | 本MVPでは未使用（S3等を常設しない） | なし |
| クラウド常時稼働サーバー | AWS | 本MVPでは未使用（EC2/ECS/Lambda常駐なし） | なし |

責務境界:
- オンプレ側（Linux）責務: 原本保管、前処理、検索、索引更新。
- AWS側責務: 推論API提供、IAM認可、予算監視。

<a id="data-flow"></a>
## データフロー（インフラ向け）
1. `rag_data/pdfs` の PDF から `chunks.jsonl` を生成する。
2. `chunks.jsonl` からローカル索引（`vectors.faiss` または `vectors.npy`）を生成する。
3. 質問時にローカル索引から Top-K 根拠を検索し、必要に応じて Bedrock Rerank で再並び替えする。
4. Bedrock には `質問 + Top-K 根拠` のみを送信して回答を得る。

クラウドに送らないもの:
- PDF原本ファイルそのもの
- ローカル索引ファイルそのもの

<a id="prerequisites"></a>
## 動作環境（Linux）
- Ubuntu 22.04+（WSL2 上の Ubuntu を含む）
- Python 3.10+
- AWS CLI v2
- Terraform 1.6+

<a id="aws-account-prep"></a>
## AWSアカウント側の準備
- 利用リージョンで Bedrock の対象モデルが有効化されていること（`ap-northeast-1`）。
- `tf-admin` 用の認証情報があること（Terraformで IAM/Budget を作成できる権限）。
- 予算通知先メールを受信できること（Terraform の `terraform.tfvars` で指定）。

## リポジトリ構成
- `infra/live/prod`: Terraform ルートモジュール（Budget + IAM）
- `infra/modules`: Terraform 共通モジュール
- `scripts/connectivity`: Bedrock 疎通テスト
- `scripts/rag`: ベクトル索引作成 + RAG 実行
- `scripts/audit`: Linux 監査情報収集

<a id="initial-setup"></a>
## セットアップ手順（詳細）
1. 作業ディレクトリを作成して clone
```bash
mkdir -p /home/<linux-user>/dev
cd /home/<linux-user>/dev
git clone git@github.com:unnowataru/vpnless-rag-mvp.git
cd vpnless-rag-mvp
```
2. Python venv を作成して依存を導入
```bash
python3 -m venv /home/<linux-user>/vpnless-rag-venv
source /home/<linux-user>/vpnless-rag-venv/bin/activate
python3 -m pip install -r scripts/rag/requirements.txt
```
3. AWS CLI と Terraform を利用可能にする
```bash
aws --version
terraform --version
```
4. `tf-admin` プロファイルを設定（Terraform用）
5. Terraform を適用して `rag-bedrock-invoker` を作成
6. `rag-bedrock-invoker` のアクセスキーを発行し `rag` プロファイルを設定
7. Bedrock疎通テストと RAG 実行

<a id="aws-auth"></a>
## AWS認証（プロファイル / アクセスキー）
このプロジェクトでは、AWS CLI の `profile` を「用途ごとに分けた認証情報セット」として使います。
`profile` 自体は AWS 上のサーバー/ストレージではなく、Linux ホスト上のローカル設定です。

| プロファイル | 用途 | 想定IAM主体 | 主な権限 |
|---|---|---|---|
| `tf-admin` | Terraform適用（インフラ変更） | 管理者ユーザーまたは管理者ロール | IAM/Budget 作成更新権限 |
| `rag` | Bedrock推論実行（アプリ実行） | `rag-bedrock-invoker` | Bedrock Invoke / Rerank など最小権限 |

`profile` は次の2ファイルで構成されます。
- `~/.aws/config`: リージョン、出力形式などの設定
- `~/.aws/credentials`: Access Key ID / Secret Access Key

例（`~/.aws/config`）:
```ini
[profile tf-admin]
region = ap-northeast-1
output = json

[profile rag]
region = ap-northeast-1
output = json
```

例（`~/.aws/credentials`）:
```ini
[tf-admin]
aws_access_key_id = AKIAxxxxxxxxxxxxxxxx
aws_secret_access_key = xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

[rag]
aws_access_key_id = AKIAyyyyyyyyyyyyyyyy
aws_secret_access_key = yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
```

アクセスキーの意味:
- `aws_access_key_id`: キーの識別子（公開されても直接は使えない）
- `aws_secret_access_key`: 実質パスワード（漏えい禁止、発行時にしか完全表示されない）

運用の注意点:
- `tf-admin` と `rag` を共用しない（責務分離）。
- `rag` は `rag-bedrock-invoker` に限定し、最小権限で運用する。
- 古いキーは無効化/削除し、常時アクティブキーを増やしすぎない。
- 可能なら `tf-admin` は長期キーよりもロール/短期クレデンシャルを優先する。

設定後の確認:
```bash
aws sts get-caller-identity --profile tf-admin
aws sts get-caller-identity --profile rag
```

<a id="terraform"></a>
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

`rag` プロファイルをまだ作っていない場合:
1. `tf-admin` で Terraform を適用（上記）
2. AWSコンソールで IAM ユーザー `rag-bedrock-invoker` のアクセスキーを作成
3. `~/.aws/credentials` の `[rag]` に設定

state 管理（補足）:
- 現在はローカル state 運用（単独運用向け）。
- 複数人運用にする場合は、S3 backend + lock（DynamoDB等）へ移行する。

<a id="bedrock-connectivity"></a>
## Bedrock 疎通テスト（Linux）
```bash
bash scripts/connectivity/bedrock_converse.sh
```

環境変数で上書き可能:
- `AWS_PROFILE`（既定: `rag`）
- `AWS_REGION`（既定: `ap-northeast-1`）
- `BEDROCK_MODEL_ID`（既定: `google.gemma-3-4b-it`）

ネットワーク要件:
- Linux ホストから AWS API へ HTTPS(443) でアウトバウンド到達できること。
- 少なくとも STS / IAM / Bedrock Runtime / Bedrock Agent Runtime へ到達可能であること。
- 企業プロキシ環境では `HTTPS_PROXY` / `NO_PROXY` を設定し、AWS CLI が通信できること。

<a id="vector-rag"></a>
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

<a id="linux-audit"></a>
## Linux監査収集
```bash
bash scripts/audit/collect_linux.sh
```

<a id="troubleshooting"></a>
## トラブル時にまず確認する項目
1. 認証確認: `aws sts get-caller-identity --profile rag`
2. モデル疎通: `bash scripts/connectivity/bedrock_converse.sh`
3. データ確認: `rag_data/pdfs` に PDF があるか、`rag_data/index/chunks.jsonl` があるか
4. 索引確認: `rag_data/index/manifest.json` と `vectors.faiss`（または `vectors.npy`）があるか
5. 失敗時は `--no-rerank` で rerank 依存を切り離して原因を分離する

<a id="secrets-and-artifacts"></a>
## 秘密情報と成果物
- 秘密情報はリポジトリ外で管理する。
- `.gitignore` で以下を除外:
  - `rag_data/`
  - `*.jsonl`, `*.pdf`, `*.faiss`, `*.npy`
  - `logs/`

<a id="issues"></a>
## 差分・実装計画（Issue）
検証プランPDFとの差分と今後の実装候補は、更新履歴を残しやすいよう GitHub Issue で管理します。

- 差分トラッカー: `#1` https://github.com/unnowataru/vpnless-rag-mvp/issues/1
- P0: metadataフィルタ検索: `#2` https://github.com/unnowataru/vpnless-rag-mvp/issues/2
- P0: `/search` API: `#3` https://github.com/unnowataru/vpnless-rag-mvp/issues/3
- P1: VAST Data / NetApp / マネージド連携: `#4` https://github.com/unnowataru/vpnless-rag-mvp/issues/4
- P1: イベント駆動インデクシング: `#5` https://github.com/unnowataru/vpnless-rag-mvp/issues/5
- P2: 監査ログ運用（相関ID・TTL・マスク）: `#6` https://github.com/unnowataru/vpnless-rag-mvp/issues/6
