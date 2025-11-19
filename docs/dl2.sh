#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://docs.nvidia.com/metropolis/deepstream/7.1/"
TARGET_DIR="deepstream-7.1-docs"
LOGFILE="fetch.log"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "=== Start: $(date) ===" | tee -a "../$LOGFILE"

wget \
  --mirror \
  --convert-links \
  --adjust-extension \
  --page-requisites \
  --no-parent \
  --tries=5 \
  --retry-connrefused \
  --wait=1 \
  --limit-rate=250k \
  "$BASE_URL" \
  2>&1 | tee -a "../$LOGFILE"

echo "=== Download finished: $(date) ===" | tee -a "../$LOGFILE"

# Часто главный файл DeepStream — index.html
INDEX="docs.nvidia.com/metropolis/deepstream/7.1/index.html"

if [[ -f "$INDEX" ]]; then
    echo "OK: Основной index.html найден" | tee -a "../$LOGFILE"
else
    echo "WARN: index.html не найден — структура сайта может отличаться" | tee -a "../$LOGFILE"
fi

echo "=== Done ===" | tee -a "../$LOGFILE"
