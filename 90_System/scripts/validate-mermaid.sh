#!/usr/bin/env bash
# Mermaid コードブロックを抽出して構文チェックする
# Usage: validate-mermaid.sh <file.md>
set -euo pipefail

file="${1:?Usage: validate-mermaid.sh <file.md>}"
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

# Markdown から mermaid コードブロックを抽出
awk '/^```mermaid$/,/^```$/' "$file" | grep -v '^```' > "$tmpdir/diagram.mmd" 2>/dev/null || true

if [[ ! -s "$tmpdir/diagram.mmd" ]]; then
  echo "No mermaid blocks found in $file"
  exit 0
fi

# mmdc で検証（PNG出力を /dev/null に捨てる）
if mmdc -i "$tmpdir/diagram.mmd" -o "$tmpdir/out.png" 2>"$tmpdir/err.log"; then
  echo "OK: Mermaid syntax valid in $file"
else
  echo "ERROR: Mermaid syntax error in $file"
  cat "$tmpdir/err.log"
  exit 1
fi
