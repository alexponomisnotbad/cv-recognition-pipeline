#!/bin/bash
# Сохранить все Docker образы для переноса на сервер amd64

mkdir -p docker-backups

echo "🔄 Сохранение docker образов..."

# 1. Скачиваемые образы (базовые)
# echo "  → bluenviron/mediamtx:latest"
# docker pull --platform linux/amd64 bluenviron/mediamtx:latest
# docker save bluenviron/mediamtx:latest | gzip > docker-backups/mediamtx-latest.tar.gz

# echo "  → nats:2.10-alpine"
# docker pull --platform linux/amd64 nats:2.10-alpine
# docker save nats:2.10-alpine | gzip > docker-backups/nats-2.10-alpine.tar.gz

# echo "  → python:3.12-slim"
# docker pull --platform linux/amd64 python:3.12-slim
# docker save python:3.12-slim | gzip > docker-backups/python-3.12-slim.tar.gz

# echo "  → postgres:16-alpine"
# docker pull --platform linux/amd64 postgres:16-alpine
# docker save postgres:16-alpine | gzip > docker-backups/postgres-16-alpine.tar.gz

# 2. Локально собранные образы (БЕЗ gzip - они уже большие)
# echo "  → docker-vision-pipeline:latest (local, ~3.4GB)"
# if docker save docker-vision-pipeline:latest > docker-backups/docker-vision-pipeline-latest.tar; then
#     echo "    ✅ Сохранён успешно"
# else
#     echo "    ❌ Ошибка при сохранении!"
#     exit 1
# fi

echo "  → docker-comparator:latest (local, ~458MB)"
if docker save docker-comparator:latest > docker-backups/docker-comparator-latest.tar; then
    echo "    ✅ Сохранён успешно"
else
    echo "    ❌ Ошибка при сохранении!"
    exit 1
fi

echo ""
echo "✅ Готово! Архивы в папке docker-backups/:"
ls -lh docker-backups/
du -sh docker-backups/