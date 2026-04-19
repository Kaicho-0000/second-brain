勉強会資料を作成する。確認不要、即実行。

トピック: $ARGUMENTS

1. `30_Tech_Notes/` と `40_Study_Materials/` から関連する既存ノートを検索する
2. 不足している情報は Web 検索（または fetch MCP）で補完する
3. `40_Study_Materials/` に `YYYY-MM-DD-kebab-topic.md` の命名で資料を作成する
4. テンプレート選択:
   - AI/ML 関連トピック → `90_System/templates/ai-study-material.md` を使用
   - その他 → `90_System/templates/study-material.md` を使用
5. 必須セクション:
   - 概念図（Mermaid）を最低 1 つ含める
   - コード例（Python）を含める
   - ハンズオン/演習セクションを含める
   - 対象者レベル: 明記する（デフォルト「中級者」）
   - 所要時間: デフォルト 30 分
6. 関連する Tech Notes がなければ `30_Tech_Notes/` にも補足ノートを作成する
7. 既存ノートへの `[[wikilink]]` を含める
8. 完了後、生成したファイルパス一覧を報告する
