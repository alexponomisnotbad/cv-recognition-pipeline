import xml.etree.ElementTree as ET
import json
from pathlib import Path
import os
import sys
from typing import Dict, List, Any

# Ваши пути
xml_path = Path(os.getenv("xml_dir", "/app/test_classification/test_1/annotations.xml"))
json_dir = Path(os.getenv("json_dir", "/app/output/test_pipeline/resnet50"))

def parse_cvat_labels(xml_path: Path) -> Dict[str, List[str]]:
    """
    Парсит XML файл CVAT for images 1.1
    Возвращает словарь: {имя_изображения: [список_меток]}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    result = {}
    
    for image in root.findall('image'):
        image_name = image.get('name')
        labels = []
        
        
        for elem in image:
            if elem.tag in ['box']:
                label = elem.get('label')
                if label:
                    labels.append({
                        'label': label,
                    })
        
        result[image_name] = label
    
    return result

def parse_json_predictions(json_file: Path) -> Dict[str, List[Dict]]:
    """
    Парсит JSON файл с предсказаниями
    Возвращает словарь: {имя_изображения: [список объектов с предсказаниями]}
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    result = {}
    for item in data:
        image_name = item.get('image')
        objects = item.get('objects', [])
        result[image_name] = objects
    
    return result

def compare_labels_and_predictions(
    ground_truth: Dict[str, List[Dict]],
    predictions: Dict[str, List[Dict]]
) -> Dict[str, Any]:
    """
    Сравнивает ground truth (из CVAT) и предсказания (из JSON)
    """
    results = {
        'total_images': len(predictions),
        'matched_images': 0,
        'total_objects': 0,
        'correct_predictions': 0,
        'per_image': {},
        'per_color': {}
    }
    
    for img_name, pred_objects in predictions.items():
        if img_name not in ground_truth:
            print(f"Warning: {img_name} not found in ground truth")
            continue
        
        gt_objects = ground_truth[img_name]
        results['matched_images'] += 1
        
        img_result = {
            'predicted_objects': len(pred_objects),
            'gt_objects': len(gt_objects),
            'matches': []
        }
        
        # Для каждого предсказанного объекта
        for pred_obj in pred_objects:
            results['total_objects'] += 1
            pred_color = pred_obj.get('color', 'unknown')
            
            
            matched = False
        
            gt_color = gt_objects
            print(f"Анализ {img_name[22:]}")
            print(f"Реальный объект {gt_color}")
            print(f"Предсказанный объект {pred_color}")
            if pred_color == gt_color:
                matched = True
                results['correct_predictions'] += 1
                
                # Статистика по цветам
                if pred_color not in results['per_color']:
                    results['per_color'][pred_color] = {'total': 0, 'correct': 0}
                results['per_color'][pred_color]['total'] += 1
                results['per_color'][pred_color]['correct'] += 1
                break
            
            if not matched:
                # Если не нашли соответствие
                if pred_color not in results['per_color']:
                    results['per_color'][pred_color] = {'total': 0, 'correct': 0}
                results['per_color'][pred_color]['total'] += 1
            
            img_result['matches'].append({
                'pred_color': pred_color,
                'matched': matched,
                'confidence': pred_obj.get('yolo_conf', 0),
                'sam_score': pred_obj.get('sam_score', 0)
            })
        
        results['per_image'][img_name] = img_result
    
    # Вычисляем точность
    if results['total_objects'] > 0:
        results['accuracy'] = results['correct_predictions'] / results['total_objects']
    else:
        results['accuracy'] = 0
    
    return results

def main():
    print("=== Парсинг CVAT XML ===")
    try:
        cvat_labels = parse_cvat_labels(xml_path)
        print(f"Загружено {len(cvat_labels)} изображений из CVAT")
    except Exception as e:
        print(f"Ошибка при парсинге XML: {e}")
        return
    
    print("\n=== Парсинг JSON предсказаний ===")
    try:
        # Если в командной строке передали имя файла, используем его.
        # Пример: python utils/metrics.py my_summary.json
        if len(sys.argv) > 1:
            json_file_path = json_dir / sys.argv[1]
            print(f"Используем JSON из аргумента командной строки: {json_file_path}")
        else:
            json_file_path = json_dir / 'summary_resnet.json'
            if not json_file_path.exists():
                # Попробуем найти любой json файл в директории
                json_files = list(json_dir.glob('*.json'))
                if json_files:
                    json_file_path = json_files[0]
                    print(f"Найден JSON файл: {json_file_path.name}")
                else:
                    print("JSON файл не найден")
                    return
        
        predictions = parse_json_predictions(json_file_path)
        print(f"Загружено {len(predictions)} изображений из JSON")
    except Exception as e:
        print(f"Ошибка при парсинге JSON: {e}")
        return
    
    print("\n=== Сравнение результатов ===")
    results = compare_labels_and_predictions(cvat_labels, predictions)
    
    # Вывод статистики
    print(f"\nСтатистика:")
    print(f"  Всего изображений в JSON: {results['total_images']}")
    print(f"  Изображений с ground truth: {results['matched_images']}")
    print(f"  Всего объектов: {results['total_objects']}")
    print(f"  Правильных предсказаний: {results['correct_predictions']}")
    print(f"  Точность (Accuracy): {results['accuracy']:.2%}")
    
    
    
    # Сохраняем результаты
    output_file = json_dir / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nРезультаты сохранены в: {output_file}")

if __name__ == "__main__":
    main()