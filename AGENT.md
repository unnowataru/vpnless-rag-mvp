# AGENT.md

## 目的

このリポジトリで動くエージェントに対し、案件固有差分と共通 SoT の関係を明示する。

## 共通前提

- 共通方針は `work-env` を優先する
- `work-env` は `C:\dev\work-env` にある前提で扱う
- 操作 UI は VS Code
- 実行オーケストレーターは Codex
- Claude は設計 / レビュー sidecar
- xAI は調査 sidecar
- `x-search` は blog / slide 向けの X 調査に使う

## 案件固有差分

- この repo はオンプレ PDF を根拠に AWS Bedrock で回答する RAG MVP を扱う
- 主な実行対象は `scripts/rag/` と `infra/`
- Linux / WSL 前提の手順を優先する

## 禁止

- 共通方針をこの repo 単独で上書きしない
- secrets をコミットしない
- `x-search` を実装オーケストレーション用途に使わない
