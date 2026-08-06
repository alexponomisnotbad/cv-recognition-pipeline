#!/bin/bash
# Загрузить все Docker образы

set -e

cd "$(dirname "$0")"
cd ../docker-backups

echo "📥 Загрузка docker образов..."
echo ""

# Обработка .tar.gz файлов
for file in *.tar.gz; do
    if [ -f "$file" ]; then
        echo "[$file]"
        gunzip -c "$file" | docker load
        echo "✅ Загружен"
        echo ""
    fi
done

# Обработка .tar файлов
for file in *.tar; do
    if [ -f "$file" ]; then
        echo "[$file]"
        docker load -i "$file"
        echo "✅ Загружен"
        echo ""
    fi
done

echo "✅ Готово!"
echo ""
echo "📊 Загруженные образы:"
docker images | grep -E "mediamtx|nats|python|postgres|docker-vision|docker-comparator"
