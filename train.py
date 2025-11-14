"""
训练脚本 - 修复版本
解决NaN/Inf loss和训练不稳定问题
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np

from config import Config
from dataset import TrafficLightDataset, create_dataloader
from model import create_model, load_checkpoint
from utils import save_checkpoint, AverageMeter, calculate_map


def validate_bboxes(targets):
    """验证bbox格式是否正确"""
    for target in targets:
        boxes = target['boxes']
        if len(boxes) > 0:
            # 检查bbox格式: [x_min, y_min, x_max, y_max]
            # 确保 x_max > x_min 且 y_max > y_min
            valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            # 确保坐标非负
            valid_mask = valid_mask & (boxes[:, 0] >= 0) & (boxes[:, 1] >= 0)
            # 确保坐标在合理范围内（假设图像尺寸不超过10000）
            valid_mask = valid_mask & (boxes[:, 2] < 10000) & (boxes[:, 3] < 10000)
            
            if not valid_mask.all():
                # 过滤无效bbox
                target['boxes'] = boxes[valid_mask]
                target['labels'] = target['labels'][valid_mask]
                target['area'] = target['area'][valid_mask]
                target['iscrowd'] = target['iscrowd'][valid_mask]
    return targets


def get_warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """创建warmup学习率调度器"""
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    
    return LambdaLR(optimizer, f)


def train_one_epoch(model, dataloader, optimizer, device, epoch, warmup_scheduler=None):
    """训练一个epoch - 改进版本"""
    model.train()
    loss_meter = AverageMeter()
    loss_classifier_meter = AverageMeter()
    loss_box_reg_meter = AverageMeter()
    loss_objectness_meter = AverageMeter()
    loss_rpn_box_reg_meter = AverageMeter()
    
    batch_count = 0
    skipped_batches = 0
    nan_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        batch_count += 1
        
        # 验证bbox格式
        targets = validate_bboxes(targets)
        
        # 将图像和targets移到设备
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 检查是否有有效标注
        has_valid_targets = False
        for target in targets:
            if len(target['boxes']) > 0:
                has_valid_targets = True
                break
        
        if not has_valid_targets:
            skipped_batches += 1
            continue
        
        # 前向传播
        try:
            loss_dict = model(images, targets)
            
            # 检查loss_dict是否有效
            if not isinstance(loss_dict, dict) or len(loss_dict) == 0:
                print(f"警告: 批次 {batch_idx} 返回无效loss_dict")
                skipped_batches += 1
                continue
            
            # 计算总loss
            losses = sum(loss for loss in loss_dict.values())
            
            # 检查loss是否为NaN或Inf
            if torch.isnan(losses) or torch.isinf(losses):
                nan_batches += 1
                print(f"\n警告: 批次 {batch_idx} 检测到NaN/Inf loss")
                print(f"Loss详情: {[(k, v.item() if torch.is_tensor(v) else v) for k, v in loss_dict.items()]}")
                continue
            
            # 检查loss是否过大（可能表示梯度爆炸）
            if losses.item() > 1000:
                print(f"\n警告: 批次 {batch_idx} loss过大: {losses.item():.4f}")
                # 仍然尝试训练，但会进行梯度裁剪
            
            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            
            # 梯度裁剪，防止梯度爆炸
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # 检查梯度是否异常
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"\n警告: 批次 {batch_idx} 检测到NaN/Inf梯度，跳过")
                nan_batches += 1
                continue
            
            optimizer.step()
            
            # 更新warmup学习率
            if warmup_scheduler is not None:
                warmup_scheduler.step()
            
            # 更新统计
            loss_meter.update(losses.item(), len(images))
            
            # 记录各个loss分量
            if 'loss_classifier' in loss_dict:
                loss_classifier_meter.update(loss_dict['loss_classifier'].item(), len(images))
            if 'loss_box_reg' in loss_dict:
                loss_box_reg_meter.update(loss_dict['loss_box_reg'].item(), len(images))
            if 'loss_objectness' in loss_dict:
                loss_objectness_meter.update(loss_dict['loss_objectness'].item(), len(images))
            if 'loss_rpn_box_reg' in loss_dict:
                loss_rpn_box_reg_meter.update(loss_dict['loss_rpn_box_reg'].item(), len(images))
            
            # 更新进度条
            postfix = {'loss': f'{loss_meter.avg:.4f}'}
            if loss_classifier_meter.count > 0:
                postfix['cls'] = f'{loss_classifier_meter.avg:.4f}'
            if loss_box_reg_meter.count > 0:
                postfix['box'] = f'{loss_box_reg_meter.avg:.4f}'
            if loss_objectness_meter.count > 0:
                postfix['obj'] = f'{loss_objectness_meter.avg:.4f}'
            pbar.set_postfix(postfix)
            
        except Exception as e:
            print(f"\n错误: 批次 {batch_idx} 训练失败: {str(e)}")
            skipped_batches += 1
            continue
    
    if skipped_batches > 0 or nan_batches > 0:
        print(f"\nEpoch {epoch} 统计: 跳过 {skipped_batches} 个批次, NaN批次 {nan_batches} 个")
    
    return loss_meter.avg


def validate(model, dataloader, device, score_threshold=0.05):
    """验证模型 - 改进版本"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Validating'):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # 推理
            try:
                predictions = model(images)
                
                # 过滤低置信度预测
                filtered_predictions = []
                for pred in predictions:
                    if len(pred['scores']) > 0:
                        mask = pred['scores'] > score_threshold
                        filtered_pred = {
                            'boxes': pred['boxes'][mask],
                            'labels': pred['labels'][mask],
                            'scores': pred['scores'][mask]
                        }
                    else:
                        filtered_pred = {
                            'boxes': pred['boxes'],
                            'labels': pred['labels'],
                            'scores': pred['scores']
                        }
                    filtered_predictions.append(filtered_pred)
                
                all_predictions.extend(filtered_predictions)
                all_targets.extend(targets)
            except Exception as e:
                print(f"验证时出错: {str(e)}")
                continue
    
    # 计算mAP
    if len(all_predictions) > 0 and len(all_targets) > 0:
        map_score = calculate_map(all_predictions, all_targets)
    else:
        map_score = 0.0
        print("警告: 验证时没有有效预测或目标")
    
    return map_score


def main():
    parser = argparse.ArgumentParser(description='交通灯检测模型训练')
    parser.add_argument('--config', type=str, default='config.py', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--model-type', type=str, default='faster_rcnn', 
                       choices=['faster_rcnn', 'yolo'], help='模型类型')
    parser.add_argument('--annotation-format', type=str, default='custom_json',
                       choices=['coco', 'yolo', 'voc', 'txt', 'custom_json', 'traffic_light'], 
                       help='标注格式 (默认: custom_json 用于交通灯检测竞赛)')
    args = parser.parse_args()
    
    # 创建目录
    Config.create_dirs()
    
    # 设置设备
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建数据集
    print('加载数据集...')
    train_dataset = TrafficLightDataset(
        image_dir=Config.TRAIN_DATA_DIR,
        annotation_dir=Config.ANNOTATIONS_DIR,
        annotation_format=args.annotation_format,
        image_size=Config.IMAGE_SIZE,
        is_train=True,
        use_augmentation=Config.USE_AUGMENTATION
    )
    
    # 划分训练集和验证集
    train_size = int((1 - Config.VAL_SPLIT) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    # Windows上num_workers设为0避免multiprocessing问题
    num_workers = 0 if os.name == 'nt' else Config.NUM_WORKERS
    train_loader = create_dataloader(
        train_dataset, 
        Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = create_dataloader(
        val_dataset,
        Config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}')
    
    # 详细检查数据加载情况
    print('\n检查数据加载情况...')
    sample_count = 0
    valid_sample_count = 0
    total_boxes = 0
    invalid_boxes = 0
    
    for i in range(min(20, len(train_dataset))):
        sample_count += 1
        try:
            image, target = train_dataset[i]
            boxes = target['boxes']
            
            if len(boxes) > 0:
                valid_sample_count += 1
                total_boxes += len(boxes)
                
                # 检查bbox有效性
                for box in boxes:
                    x_min, y_min, x_max, y_max = box
                    if x_max <= x_min or y_max <= y_min:
                        invalid_boxes += 1
                    if x_min < 0 or y_min < 0:
                        invalid_boxes += 1
                
                print(f'样本 {i}: {len(boxes)} 个标注框, 标签: {target["labels"].tolist()}')
        except Exception as e:
            print(f'样本 {i} 加载失败: {str(e)}')
    
    print(f'检查了 {sample_count} 个样本，其中 {valid_sample_count} 个有标注')
    print(f'总标注框数: {total_boxes}, 无效框数: {invalid_boxes}')
    
    if valid_sample_count == 0:
        print('错误: 没有找到任何有效标注！请检查train.json文件是否正确加载。')
        return
    
    if invalid_boxes > 0:
        print(f'警告: 发现 {invalid_boxes} 个无效bbox，训练时会自动过滤')
    
    # 创建模型
    print('\n创建模型...')
    model = create_model(
        model_type=args.model_type,
        num_classes=Config.NUM_CLASSES,
        pretrained=True
    )
    model.to(device)
    
    # 使用更低的学习率
    # Faster R-CNN通常需要较小的学习率，特别是使用预训练模型时
    initial_lr = Config.LEARNING_RATE * 0.1  # 降低10倍
    print(f'使用学习率: {initial_lr} (配置文件: {Config.LEARNING_RATE})')
    
    # 优化器 - 使用AdamW，但学习率较低
    optimizer = optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=0.0001,
        eps=1e-8
    )
    
    # 使用StepLR，每几个epoch降低学习率（更稳定）
    # 或者使用CosineAnnealingLR
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=Config.NUM_EPOCHS,
        eta_min=initial_lr * 0.01
    )
    
    # Warmup调度器（可选，前几个batch使用）
    warmup_iters = min(500, len(train_loader))  # 1个epoch的warmup
    warmup_scheduler = get_warmup_lr_scheduler(optimizer, warmup_iters, 0.1)
    
    # 恢复训练
    start_epoch = 0
    best_map = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_map = checkpoint.get('best_map', 0.0)
        print(f'从epoch {start_epoch}恢复训练')
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_map': [],
        'epochs': [],
        'learning_rates': []
    }
    
    # 训练循环
    print('\n开始训练...')
    global_step = 0
    
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            warmup_scheduler if global_step < warmup_iters else None
        )
        
        # 更新全局步数
        global_step += len(train_loader)
        
        # 更新学习率（warmup后使用主调度器）
        if global_step >= warmup_iters:
            scheduler.step()
        
        # 验证（每个epoch都验证）
        val_map = validate(model, val_loader, device, score_threshold=0.05)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch+1}/{Config.NUM_EPOCHS} - '
              f'Train Loss: {train_loss:.4f}, Val mAP: {val_map:.4f}, LR: {current_lr:.6f}')
        
        # 保存最佳模型
        if val_map > best_map:
            best_map = val_map
            save_checkpoint(
                model, optimizer, epoch, best_map,
                os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
            )
            print(f'✓ 保存最佳模型，mAP: {best_map:.4f}')
        
        # 保存历史
        history['train_loss'].append(train_loss)
        history['epochs'].append(epoch + 1)
        history['val_map'].append(val_map)
        history['learning_rates'].append(current_lr)
        
        # 定期保存检查点
        if (epoch + 1) % Config.SAVE_INTERVAL == 0:
            save_checkpoint(
                model, optimizer, epoch, best_map,
                os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            )
    
    # 保存训练历史
    history_path = os.path.join(Config.OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'\n训练完成！最佳mAP: {best_map:.4f}')
    print(f'训练历史已保存到: {history_path}')


if __name__ == '__main__':
    main()
