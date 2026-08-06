#!/usr/bin/env bash
set -e

# Запуск comparator
# Использование:
#   ./docker/comparator-up.sh

cd "$(dirname "$0")/.."

docker compose -f docker/docker-compose.yml up -d --build comparator