# CLAUDE.md

## 目的

このリポジトリで Claude を呼ぶときの案件固有レビュー観点を定義する。

## 期待役割

- RAG と infra の責務分離確認
- Bedrock 送信範囲とデータ主権の観点整理
- docs と実装差分のレビュー

## 調査系との境界

- X 上の反応調査は `x-search` を優先する
- Claude は調査結果を設計や文章構成へ接続する

## 確認すべきこと

- `work-env` との整合性
- オンプレ SoT の維持
- AWS 認証や secrets の扱い
