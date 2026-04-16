#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
docker compose -f docker/docker-compose.yml up -d vision-pipeline