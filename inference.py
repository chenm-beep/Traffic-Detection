"""
推理脚本 - 生成测试集预测结果
"""
import os
import torch
import argparse
import json
import pandas as pd
from tqdm import tqdm
import numpy as np

from config import Config
from dataset import TrafficLightDataset, create_dataloader
from model import create_model, load_checkpoint


def predict(model, dataloader, device, conf_threshold=0.5):
    """对测试集进行预测"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for images, metadata in tqdm(dataloader, desc='推理中'):
            images = [img.to(device) for img in images]
            
            # 推理
            predictions = model(images)
            
            # 处理每个预测结果
            for pred, meta in zip(predictions, metadata):
                image_id = meta['image_id']
                original_size = meta['original_size']
                original_width, original_height = original_size
                
                # 提取检测框
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()
                
                # 过滤低置信度检测
                valid_indices = scores >= conf_threshold
                boxes = boxes[valid_indices]
                scores = scores[valid_indices]
                labels = labels[valid_indices]
                
                # 将坐标缩放回原始图像尺寸
                scale_x = original_width / Config.IMAGE_SIZE
                scale_y = original_height / Config.IMAGE_SIZE
                
                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    
                    # 缩放坐标
                    x_min = int(x_min * scale_x)
                    y_min = int(y_min * scale_y)
                    x_max = int(x_max * scale_x)
                    y_max = int(y_max * scale_y)
                    
                    # 确保坐标在图像范围内
                    x_min = max(0, min(x_min, original_width - 1))
                    y_min = max(0, min(y_min, original_height - 1))
                    x_max = max(x_min + 1, min(x_max, original_width))
                    y_max = max(y_min + 1, min(y_max, original_height))
                    
                    all_predictions.append({
                        'image_id': image_id,
                        'bbox': [x_min, y_min, x_max, y_max],
                        'score': float(score),
                        'category_id': int(label)
                    })
    
    return all_predictions


def save_submission_csv(predictions, output_path):
    """保存为CSV格式提交文件"""
    # 根据竞赛要求调整格式
    # 常见格式：image_id, x_min, y_min, x_max, y_max, class_id, confidence
    rows = []
    for pred in predictions:
        rows.append({
            'image_id': pred['image_id'],
            'x_min': pred['bbox'][0],
            'y_min': pred['bbox'][1],
            'x_max': pred['bbox'][2],
            'y_max': pred['bbox'][3],
            'class_id': pred['category_id'],
            'confidence': pred['score']
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f'CSV提交文件已保存到: {output_path}')


def save_submission_json(predictions, output_path):
    """保存为JSON格式提交文件（交通灯检测竞赛格式）"""
    # 类别ID到标签名称的映射
    id_to_label = {
        1: 'red',
        2: 'yellow',
        3: 'green'
    }
    
    # 转换为竞赛要求的格式
    annotations = []
    for pred in predictions:
        image_id = pred['image_id']
        # 处理文件名：如果只是文件名，添加test_images\前缀
        if '\\' not in image_id and '/' not in image_id:
            filename = f"test_images\\{image_id}"
        else:
            # 确保使用Windows路径分隔符
            filename = image_id.replace('/', '\\')
        
        bbox = pred['bbox']
        category_id = pred['category_id']
        score = pred['score']
        
        # 获取标签名称
        label = id_to_label.get(category_id, 'unknown')
        
        annotations.append({
            'filename': filename,
            'conf': float(score),
            'box': {
                'xmin': float(bbox[0]),
                'ymin': float(bbox[1]),
                'xmax': float(bbox[2]),
                'ymax': float(bbox[3])
            },
            'label': label
        })
    
    submission = {
        'annotations': annotations
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(submission, f, indent=4, ensure_ascii=False)
    print(f'JSON提交文件已保存到: {output_path}')
    print(f'共 {len(annotations)} 个检测结果')


def save_submission_txt(predictions, output_path):
    """保存为TXT格式提交文件（每行一个检测结果）"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            bbox = pred['bbox']
            line = f"{pred['image_id']} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {pred['category_id']} {pred['score']:.4f}\n"
            f.write(line)
    print(f'TXT提交文件已保存到: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='交通灯检测推理')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--test-dir', type=str, default=None, help='测试集目录（覆盖配置）')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    parser.add_argument('--format', type=str, default=None, 
                       choices=['csv', 'json', 'txt'], help='输出格式')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--model-type', type=str, default='faster_rcnn',
                       choices=['faster_rcnn', 'yolo'], help='模型类型')
    args = parser.parse_args()
    
    # 创建目录
    Config.create_dirs()
    
    # 设置设备
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载模型
    print('加载模型...')
    model = create_model(
        model_type=args.model_type,
        num_classes=Config.NUM_CLASSES,
        pretrained=False
    )
    model = load_checkpoint(model, args.checkpoint, device)
    print('模型加载完成')
    
    # 创建测试数据集
    test_dir = args.test_dir or Config.TEST_DATA_DIR
    print(f'加载测试集: {test_dir}')
    test_dataset = TrafficLightDataset(
        image_dir=test_dir,
        annotation_dir=None,  # 测试集没有标注
        annotation_format='coco',
        image_size=Config.IMAGE_SIZE,
        is_train=False,
        use_augmentation=False
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    
    print(f'测试集大小: {len(test_dataset)}')
    
    # 进行预测
    print('开始推理...')
    predictions = predict(
        model, 
        test_loader, 
        device, 
        conf_threshold=args.conf_threshold or Config.CONF_THRESHOLD
    )
    
    print(f'共检测到 {len(predictions)} 个目标')
    
    # 保存结果
    output_format = args.format or Config.SUBMISSION_FORMAT
    if args.output:
        output_path = args.output
    else:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(
            Config.SUBMISSION_DIR, 
            f'submission_{timestamp}.{output_format}'
        )
    
    if output_format == 'csv':
        save_submission_csv(predictions, output_path)
    elif output_format == 'json':
        save_submission_json(predictions, output_path)
    elif output_format == 'txt':
        save_submission_txt(predictions, output_path)
    
    print('推理完成！')


if __name__ == '__main__':
    main()

