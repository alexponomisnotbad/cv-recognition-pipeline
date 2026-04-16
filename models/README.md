# models/

Эта директория содержит веса моделей, смонтированные в контейнеры как read-only тома.

```
models/
├── yolo/
│   └── best.pt          # обученная YOLO-модель для детекции детали
└── sam2/
    └── sam2.1_hiera_large.pt   # чекпоинт SAM2 (скачать: scripts/download_models.sh)
```

## Загрузка SAM2

```bash
bash src/vision-pipeline/scripts/download_models.sh
```

## Обучение YOLO

Обучите модель на своём датасете и скопируйте `best.pt` в `models/yolo/`.
