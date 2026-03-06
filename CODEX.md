# CODEX.md

## 目的

このリポジトリで Codex が実装するときの案件固有運用を定義する。

## 進め方

1. 先に `C:\dev\work-env` の SoT を読む
2. 次にこの repo の `repo-profile.yaml` を読む
3. 調査が必要なら `x-search` を使う
4. 実装、検証、コミットをこの repo で行う
5. 共通判断が増えたら `work-env` に同期する

## 調査ルール

- ブログやスライド向けの外部反応調査は `x-search` を優先する
- RAG 実装や Bedrock 接続確認はこの repo の scripts を優先する

## 出力契約

- 変更ファイル一覧
- 実装内容の要約
- テスト結果の要約
- 参照した SoT / context files の一覧
- コミットメッセージ
- 必要なら残課題と blocker
