#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://docs.nvidia.com/cuda/archive/12.6.0/"
TARGET_DIR="cuda-12.6.0-docs"
LOGFILE="fetch.log"

# Создаём папку, если её нет
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "=== Start: $(date) ===" | tee -a "../$LOGFILE"

# Основная загрузка
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

# Простая проверка — должен существовать индекс раздела
INDEX="docs.nvidia.com/cuda/archive/12.6.0/index.html"

if [[ -f "$INDEX" ]]; then
    echo "OK: Основной index.html найден" | tee -a "../$LOGFILE"
else
    echo "WARN: index.html не найден — структура сайта могла измениться" | tee -a "../$LOGFILE"
fi

echo "=== Done ===" | tee -a "../$LOGFILE"
