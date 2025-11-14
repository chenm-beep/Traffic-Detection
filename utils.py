"""
工具函数
"""
import torch
import numpy as np
from collections import defaultdict


class AverageMeter:
    """计算和存储平均值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, optimizer, epoch, best_map, filepath):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_map': best_map
    }
    torch.save(checkpoint, filepath)
    print(f'检查点已保存: {filepath}')


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算并集
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def calculate_map(predictions, targets, iou_threshold=0.5):
    """
    计算mAP (mean Average Precision)
    
    Args:
        predictions: 预测结果列表
        targets: 真实标注列表
        iou_threshold: IoU阈值
    
    Returns:
        mAP值
    """
    # 按类别计算AP
    class_aps = []
    
    # 检查输入
    if len(predictions) == 0 or len(targets) == 0:
        print("警告: predictions或targets为空")
        return 0.0
    
    # 获取所有类别
    all_classes = set()
    for pred in predictions:
        if len(pred['labels']) > 0:
            all_classes.update(pred['labels'].cpu().numpy().tolist())
    for target in targets:
        if len(target['labels']) > 0:
            all_classes.update(target['labels'].cpu().numpy().tolist())
    
    if len(all_classes) == 0:
        print("警告: 没有找到任何类别")
        return 0.0
    
    for class_id in all_classes:
        if class_id == 0:  # 跳过背景类
            continue
        
        # 收集该类别的所有预测和真实值
        pred_boxes = []
        target_boxes = []
        
        for pred, target in zip(predictions, targets):
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_boxes_tensor = pred['boxes'].cpu().numpy()
            
            target_labels = target['labels'].cpu().numpy()
            target_boxes_tensor = target['boxes'].cpu().numpy()
            
            # 提取该类别的预测
            class_mask = pred_labels == class_id
            for i in np.where(class_mask)[0]:
                pred_boxes.append({
                    'box': pred_boxes_tensor[i],
                    'score': pred_scores[i]
                })
            
            # 提取该类别的真实值
            class_mask = target_labels == class_id
            for i in np.where(class_mask)[0]:
                target_boxes.append({
                    'box': target_boxes_tensor[i]
                })
        
        if len(target_boxes) == 0:
            continue
        
        # 按置信度排序
        pred_boxes.sort(key=lambda x: x['score'], reverse=True)
        
        # 计算TP和FP
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        matched_targets = set()
        
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_target_idx = -1
            
            for j, target_box in enumerate(target_boxes):
                if j in matched_targets:
                    continue
                
                iou = calculate_iou(pred_box['box'], target_box['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = j
            
            if best_iou >= iou_threshold:
                tp[i] = 1
                matched_targets.add(best_target_idx)
            else:
                fp[i] = 1
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算精确率和召回率
        recalls = tp_cumsum / len(target_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # 计算AP（使用11点插值法）
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        class_aps.append(ap)
    
    # 计算mAP
    if len(class_aps) == 0:
        return 0.0
    
    map_score = np.mean(class_aps)
    return map_score


def visualize_predictions(image, predictions, class_names, save_path=None):
    """可视化预测结果"""
    import cv2
    import matplotlib.pyplot as plt
    
    # 绘制图像
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # 绘制检测框
    for pred in predictions:
        box = pred['bbox']
        score = pred['score']
        class_id = pred['category_id']
        
        x_min, y_min, x_max, y_max = box
        
        # 绘制边界框
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor='red',
            linewidth=2
        )
        ax.add_patch(rect)
        
        # 添加标签
        label = f"{class_names[class_id]}: {score:.2f}"
        ax.text(
            x_min, y_min - 5,
            label,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
            fontsize=10,
            color='white'
        )
    
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    
    plt.close()

