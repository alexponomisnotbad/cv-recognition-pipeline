#!/usr/bin/env python3
"""
Конвертер COCO датасета в YOLOv8 формат
"""
import json
import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

def convert_coco_to_yolo(coco_dir, output_dir, train_ratio=0.8):
    """Convert COCO dataset to YOLOv8 format"""
    
    coco_dir = Path(coco_dir)
    output_dir = Path(output_dir)
    
    # Load COCO annotations
    anno_file = coco_dir / "annotations" / "instances_default.json"
    with open(anno_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)  
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Get class names
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    num_classes = len(categories)
    
    # Create class index mapping
    class_idx = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}
    
    print(f"Classes: {categories}")
    print(f"Class mapping: {class_idx}")
    
    # Process images and annotations
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Split into train/val
    image_ids = list(image_annotations.keys())
    train_ids, val_ids = train_test_split(image_ids, train_size=train_ratio, random_state=42)
    
    print(f"Total images: {len(image_ids)}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
    
    # Process each image
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for img_id in ids:
            img_info = next(img for img in coco_data['images'] if img['id'] == img_id)
            img_filename = img_info['file_name']
            img_height = img_info['height']
            img_width = img_info['width']
            
            # Copy image
            src_img = coco_dir / "images" / "default" / img_filename
            dst_img = output_dir / "images" / split / img_filename
            if src_img.exists():
                shutil.copy(src_img, dst_img)
            
            # Create YOLO annotation
            anns = image_annotations[img_id]
            label_lines = []
            
            for ann in anns:
                cat_id = ann['category_id']
                class_id = class_idx[cat_id]
                bbox = ann['bbox']  # [x, y, width, height]
                
                # Convert to YOLO format (x_center, y_center, width, height) normalized
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height
                
                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save labels
            label_filename = img_filename.rsplit('.', 1)[0] + '.txt'
            label_path = output_dir / "labels" / split / label_filename
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))
    
    # Create data.yaml for YOLOv8
    yaml_content = f"""path: {output_dir}
train: images/train
val: images/val

nc: {num_classes}
names: {list(categories.values())}
"""
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Conversion complete!")
    print(f"✓ Data saved to: {output_dir}")
    print(f"✓ data.yaml created at: {yaml_path}")

if __name__ == "__main__":
    coco_dir = "/app/dataset/coco_dataset"
    output_dir = "/app/dataset/yolo_dataset"
    
    convert_coco_to_yolo(coco_dir, output_dir)
