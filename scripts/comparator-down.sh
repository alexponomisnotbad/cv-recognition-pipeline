#!/usr/bin/env bash
set -e

# Остановка comparator
# Использование:
#   ./docker/comparator-down.sh

cd "$(dirname "$0")/.."

docker compose -f docker/docker-compose.yml stop comparator
