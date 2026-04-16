#!/usr/bin/env bash
# =============================================================================
# download_models.sh – загружает веса SAM1, SAM2 (и опционально YOLO)
#
# Использование:
#   bash scripts/download_models.sh
#   SAM1_MODEL=vit_b bash scripts/download_models.sh   # лёгкая версия SAM1
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SAM1_DIR="${SAM1_DIR:-$PROJECT_ROOT/models/sam1}"
SAM2_DIR="${SAM2_DIR:-$PROJECT_ROOT/models/sam2}"
YOLO_DIR="${YOLO_DIR:-$PROJECT_ROOT/models/yolo}"

mkdir -p "$SAM1_DIR" "$SAM2_DIR" "$YOLO_DIR"

# =============================================================================
echo "=== Загрузка весов SAM1 (segment_anything) ==="
# ViT-H (~2.4 ГБ) – самый точный, медленный на CPU
# ViT-L (~1.2 ГБ)
# ViT-B (~375 МБ) – быстрый, хорошо работает на CPU  ← рекомендуем для CPU
#
# Переключиться на ViT-B: SAM1_MODEL=vit_b bash scripts/download_models.sh
SAM1_URL_BASE="https://dl.fbaipublicfiles.com/segment_anything"
SAM1_MODEL="${SAM1_MODEL:-vit_h}"

declare -A SAM1_FILES=(
    ["vit_h"]="sam_vit_h_4b8939.pth"
    ["vit_l"]="sam_vit_l_0b3195.pth"
    ["vit_b"]="sam_vit_b_01ec64.pth"
)

SAM1_FILENAME="${SAM1_FILES[$SAM1_MODEL]}"
SAM1_TARGET="$SAM1_DIR/sam_vit_h.pth"   # имя фиксировано – совпадает с docker-compose

if [[ ! -f "$SAM1_TARGET" ]]; then
    echo "Скачиваем SAM1 ($SAM1_MODEL): $SAM1_FILENAME"
    wget -q --show-progress -O "$SAM1_TARGET" "$SAM1_URL_BASE/$SAM1_FILENAME"
    echo "OK: $SAM1_TARGET"
else
    echo "Уже есть: $SAM1_TARGET – пропускаем"
fi

# =============================================================================
echo ""
echo "=== Загрузка весов SAM2 ==="
# Tiny  (~39 МБ)   – CPU-friendly, меньшая точность
# Large (~224 МБ)  – точный, может быть медленным на CPU
SAM2_URL_BASE="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM2_TARGET="sam2.1_hiera_large.pt"

if [[ ! -f "$SAM2_DIR/$SAM2_TARGET" ]]; then
    echo "Скачиваем $SAM2_TARGET..."
    wget -q --show-progress -O "$SAM2_DIR/$SAM2_TARGET" "$SAM2_URL_BASE/$SAM2_TARGET"
    echo "OK: $SAM2_DIR/$SAM2_TARGET"
else
    echo "Уже есть: $SAM2_DIR/$SAM2_TARGET – пропускаем"
fi

# =============================================================================
echo ""
echo "=== Проверка YOLO-модели ==="
if [[ ! -f "$YOLO_DIR/best.pt" ]]; then
    echo "ВНИМАНИЕ: $YOLO_DIR/best.pt не найден."
    echo "Поместите обученную модель YOLO в эту директорию вручную."
else
    echo "OK: $YOLO_DIR/best.pt"
fi

echo ""
echo "=== Готово ==="
ls -lh "$SAM1_DIR" "$SAM2_DIR" "$YOLO_DIR"
