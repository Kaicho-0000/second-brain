今週の振り返りレポートを生成する。確認不要、即実行。

1. `10_Journal/` の今週分（月〜今日）の日記を読み込む。なければスキップ
2. `20_Projects/` のアクティブなプロジェクト（status: active）の進捗を確認する
3. `30_Tech_Notes/` と `40_Study_Materials/` の今週追加・更新されたファイルを `git log --since="last monday"` で特定する
4. ディレクトリ `60_Reports/weekly/` がなければ作成する
5. `60_Reports/weekly/` に `YYYY-Www.md` の命名でレポートを生成する（例: `2026-W16.md`）
6. レポート構成:
   ```
   ---
   tags: [report, weekly-review]
   created: "YYYY-MM-DD"
   period: "YYYY-Www"
   ---
   # Weekly Review — YYYY-Www
   ## 今週の成果サマリ
   ## 学んだこと
   ## 課題・ブロッカー
   ## 来週の優先事項
   ## アクションアイテム
   ```
7. 関連ノートへの `[[wikilink]]` を含める
8. 完了後、生成したファイルパスを報告する
