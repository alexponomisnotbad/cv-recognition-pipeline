#!/usr/bin/env bash
set -e

# Остановка основного стека
# Использование:
#   ./docker/down.sh

cd "$(dirname "$0")/.."

docker compose -f docker/docker-compose.yml down
