#!/usr/bin/env bash
set -e

# Остановка dev-shell
# Использование:
#   ./docker/dev-shell-down.sh

cd "$(dirname "$0")/.."

docker compose -f docker/docker-compose.yml stop dev-shell
