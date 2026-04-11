# Second Brain

Obsidian + Git + Claude Code で構築する個人ナレッジベース。

## フォルダ構成

```
00_Inbox/          — 未整理の思考・メモの入口
10_Journal/        — 日記・振り返り
20_Projects/       — ゴールを持つ進行中プロジェクト
30_Tech_Notes/     — 永続的な技術ナレッジ
40_Study_Materials/ — 勉強会資料・学習ノート
50_Business_Context/ — ビジネス文脈・意思決定ログ
60_Reports/        — 定期レポート（週次/月次/Codex）
90_System/         — テンプレート・運用ルール
99_Archives/       — 完了済み
```

**情報のライフサイクル**: Inbox → 整理 → 知識化 → Archive

## Claude Code コマンド

| コマンド | 用途 |
|----------|------|
| `/weekly-review` | 週次振り返りレポートを生成 |
| `/monthly-report` | 月次レポートを生成 |
| `/study-prep [トピック]` | 勉強会資料のドラフト作成 |
| `/inbox-process` | Inbox の未整理ノートを処理 |
| `/deep-research [トピック]` | トピックの深掘りリサーチ |
| `/kickoff [プロジェクト名]` | 新プロジェクト立ち上げ |

## エージェント

| エージェント | 役割 |
|-------------|------|
| socrates | ソクラテス的問答で思考を深める |
| curator | ナレッジベースの品質管理・整理 |
| reporter | 定期レポート自動生成 |

## Codex 連携

- Codex タスクの成果物 → `60_Reports/codex/`
- 勉強会資料のドラフト → `40_Study_Materials/drafts/`

## セットアップ

1. このリポジトリを Obsidian の Vault として開く
2. Claude Code でこのディレクトリに入る: `cd second-brain`
3. コマンドを使って知識を蓄積する
