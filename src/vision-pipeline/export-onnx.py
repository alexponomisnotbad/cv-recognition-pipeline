# from ultralytics import YOLO

# model = YOLO("/app/models/yolo/part_detector/weights/best.pt")
# model.export(format="onnx", imgsz=640, opset=12) # экспорт детектора в ONNX

import torch
from utils.resnet import build_resnet50

model, device = build_resnet50(num_classes=3, pretrained=False, device="cpu")
state = torch.load("/app/models/resnet/resnet50_cls_430.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

dummy = torch.randn(1, 3, 224, 224)  # тот же размер и нормализация, что в train_resnet.py

torch.onnx.export(
    model,
    dummy,
    "/app/models/onnx/classifier-v1.onnx",
    input_names=["input"],
    output_names=["logits"],
    opset_version=12,
)