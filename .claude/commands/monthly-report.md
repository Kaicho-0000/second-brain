月次レポートを生成する。確認不要、即実行。

1. `60_Reports/weekly/` の当月分の週次レポートを読み込む
2. `20_Projects/` のプロジェクト状況を確認する
3. `30_Tech_Notes/` と `40_Study_Materials/` の当月の変更を `git log --since="first day of this month"` で特定する
4. ディレクトリ `60_Reports/monthly/` がなければ作成する
5. `60_Reports/monthly/YYYY-MM.md` にレポートを生成する
6. レポート構成:
   ```
   ---
   tags: [report, monthly]
   created: "YYYY-MM-DD"
   period: "YYYY-MM"
   ---
   # Monthly Report — YYYY-MM
   ## Executive Summary
   ## プロジェクト進捗一覧
   ## 技術的な学び・成長
   ## 勉強会の実施/参加状況
   ## KPI（ノート数、完了タスク数）
   ## 来月の計画
   ## アクションアイテム
   ```
7. フロントマターと `[[wikilink]]` を付与する
8. 完了後、生成したファイルパスを報告する
