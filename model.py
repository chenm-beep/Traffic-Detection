"""
模型定义模块
支持多种目标检测模型：Faster R-CNN, YOLO等
"""
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T


class TrafficLightDetector(nn.Module):
    """交通灯检测模型"""
    
    def __init__(self, num_classes=4, backbone='resnet50', pretrained=True):
        """
        Args:
            num_classes: 类别数量（包括背景）
            backbone: 骨干网络
            pretrained: 是否使用预训练权重
        """
        super(TrafficLightDetector, self).__init__()
        
        if backbone == 'resnet50':
            # 使用Faster R-CNN with ResNet-50 FPN
            self.model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
            
            # 替换分类头
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, images, targets=None):
        """
        Args:
            images: 图像列表或tensor
            targets: 标注目标（训练时）
        
        Returns:
            训练时返回loss字典，推理时返回预测结果
        """
        if self.training:
            return self.model(images, targets)
        else:
            return self.model(images)


class YOLOModel(nn.Module):
    """YOLO模型封装（可选，需要安装ultralytics）"""
    
    def __init__(self, num_classes=4, model_size='yolov8n'):
        """
        Args:
            num_classes: 类别数量
            model_size: 模型大小 ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
        """
        super(YOLOModel, self).__init__()
        try:
            from ultralytics import YOLO
            self.model = YOLO(f'{model_size}.pt')
            self.model.model.nc = num_classes - 1  # YOLO不包括背景类
        except ImportError:
            raise ImportError("请安装ultralytics: pip install ultralytics")
    
    def forward(self, images, targets=None):
        if self.training:
            # YOLO训练需要特殊处理
            return self.model.train(data='', epochs=1, imgsz=640)
        else:
            results = self.model(images)
            return results


def create_model(model_type='faster_rcnn', num_classes=4, pretrained=True):
    """
    创建模型
    
    Args:
        model_type: 模型类型 ('faster_rcnn', 'yolo')
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    
    Returns:
        模型实例
    """
    if model_type == 'faster_rcnn':
        model = TrafficLightDetector(
            num_classes=num_classes,
            backbone='resnet50',
            pretrained=pretrained
        )
    elif model_type == 'yolo':
        model = YOLOModel(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

