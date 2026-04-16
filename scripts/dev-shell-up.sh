#!/usr/bin/env bash
set -e

# Запуск dev-shell (интерактивный контейнер vision-pipeline)
# Использование:
#   ./docker/dev-shell-up.sh

cd "$(dirname "$0")/.."

docker compose -f docker/docker-compose.yml --profile test up -d dev-shell
