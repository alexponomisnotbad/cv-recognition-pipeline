#!/usr/bin/env bash
set -e

# Запуск основного стека (mediamtx + nats + vision-pipeline)
# Использование:
#   ./docker/up.sh

# Переходим в корень проекта
cd "$(dirname "$0")/.."

docker compose -f docker/docker-compose.yml up -d mediamtx nats vision-pipeline
