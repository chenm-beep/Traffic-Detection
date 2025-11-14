"""
数据集加载模块
支持多种标注格式：COCO、YOLO、Pascal VOC等
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import xml.etree.ElementTree as ET
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TrafficLightDataset(Dataset):
    """交通灯检测数据集类"""
    
    def __init__(self, 
                 image_dir, 
                 annotation_dir=None, 
                 annotation_format='coco',
                 image_size=640,
                 is_train=True,
                 use_augmentation=True):
        """
        Args:
            image_dir: 图像目录路径
            annotation_dir: 标注文件目录路径
            annotation_format: 标注格式 ('coco', 'yolo', 'voc', 'txt')
            image_size: 输入图像尺寸
            is_train: 是否为训练集
            use_augmentation: 是否使用数据增强
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.annotation_format = annotation_format
        self.image_size = image_size
        self.is_train = is_train
        self.use_augmentation = use_augmentation
        
        # 加载图像列表
        self.image_files = self._load_image_files()
        
        # 加载标注
        if annotation_dir:
            self.annotations = self._load_annotations()
        else:
            self.annotations = {}
        
        # 数据增强
        if use_augmentation and is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    def _load_image_files(self):
        """加载图像文件列表"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        if os.path.isdir(self.image_dir):
            for filename in os.listdir(self.image_dir):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(filename)
        
        return sorted(image_files)
    
    def _load_annotations(self):
        """根据标注格式加载标注"""
        annotations = {}
        
        if self.annotation_format == 'coco':
            annotations = self._load_coco_annotations()
        elif self.annotation_format == 'yolo':
            annotations = self._load_yolo_annotations()
        elif self.annotation_format == 'voc':
            annotations = self._load_voc_annotations()
        elif self.annotation_format == 'txt':
            annotations = self._load_txt_annotations()
        elif self.annotation_format == 'custom_json' or self.annotation_format == 'traffic_light':
            annotations = self._load_custom_json_annotations()
        
        return annotations
    
    def _load_coco_annotations(self):
        """加载COCO格式标注"""
        annotations = {}
        
        # 查找JSON标注文件
        json_files = [f for f in os.listdir(self.annotation_dir) 
                     if f.endswith('.json')]
        
        for json_file in json_files:
            json_path = os.path.join(self.annotation_dir, json_file)
            with open(json_path, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            # 构建图像ID到文件名的映射
            image_id_to_filename = {img['id']: img['file_name'] 
                                   for img in coco_data['images']}
            
            # 构建图像ID到标注的映射
            image_id_to_anns = {}
            for ann in coco_data['annotations']:
                image_id = ann['image_id']
                if image_id not in image_id_to_anns:
                    image_id_to_anns[image_id] = []
                
                bbox = ann['bbox']  # [x, y, width, height]
                # 转换为 [x_min, y_min, x_max, y_max]
                bbox_pascal = [
                    bbox[0],
                    bbox[1],
                    bbox[0] + bbox[2],
                    bbox[1] + bbox[3]
                ]
                
                image_id_to_anns[image_id].append({
                    'bbox': bbox_pascal,
                    'category_id': ann['category_id']
                })
            
            # 转换为文件名到标注的映射
            for image_id, filename in image_id_to_filename.items():
                if image_id in image_id_to_anns:
                    annotations[filename] = image_id_to_anns[image_id]
        
        return annotations
    
    def _load_yolo_annotations(self):
        """加载YOLO格式标注（归一化坐标需要图像尺寸转换）"""
        annotations = {}
        
        for image_file in self.image_files:
            base_name = os.path.splitext(image_file)[0]
            txt_file = os.path.join(self.annotation_dir, base_name + '.txt')
            
            if os.path.exists(txt_file):
                # 读取图像尺寸
                image_path = os.path.join(self.image_dir, image_file)
                if os.path.exists(image_path):
                    img = cv2.imread(image_path)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                    else:
                        continue
                else:
                    continue
                
                bboxes = []
                with open(txt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            # YOLO格式：归一化的中心坐标和宽高
                            x_center_norm = float(parts[1])
                            y_center_norm = float(parts[2])
                            width_norm = float(parts[3])
                            height_norm = float(parts[4])
                            
                            # 转换为像素坐标
                            x_center = x_center_norm * img_width
                            y_center = y_center_norm * img_height
                            width = width_norm * img_width
                            height = height_norm * img_height
                            
                            # 转换为Pascal VOC格式 [x_min, y_min, x_max, y_max]
                            x_min = x_center - width / 2
                            y_min = y_center - height / 2
                            x_max = x_center + width / 2
                            y_max = y_center + height / 2
                            
                            bboxes.append({
                                'bbox': [x_min, y_min, x_max, y_max],
                                'category_id': class_id
                            })
                
                if bboxes:
                    annotations[image_file] = bboxes
        
        return annotations
    
    def _load_voc_annotations(self):
        """加载Pascal VOC格式标注"""
        annotations = {}
        
        for image_file in self.image_files:
            base_name = os.path.splitext(image_file)[0]
            xml_file = os.path.join(self.annotation_dir, base_name + '.xml')
            
            if os.path.exists(xml_file):
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                bboxes = []
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    # 这里需要根据实际类别名称映射到ID
                    class_id = self._class_name_to_id(class_name)
                    
                    bbox = obj.find('bndbox')
                    x_min = float(bbox.find('xmin').text)
                    y_min = float(bbox.find('ymin').text)
                    x_max = float(bbox.find('xmax').text)
                    y_max = float(bbox.find('ymax').text)
                    
                    bboxes.append({
                        'bbox': [x_min, y_min, x_max, y_max],
                        'category_id': class_id
                    })
                
                annotations[image_file] = bboxes
        
        return annotations
    
    def _load_txt_annotations(self):
        """加载TXT格式标注（每行：image_name x1 y1 x2 y2 class_id）"""
        annotations = {}
        
        txt_files = [f for f in os.listdir(self.annotation_dir) 
                    if f.endswith('.txt')]
        
        for txt_file in txt_files:
            txt_path = os.path.join(self.annotation_dir, txt_file)
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        image_name = parts[0]
                        bbox = [float(parts[1]), float(parts[2]), 
                               float(parts[3]), float(parts[4])]
                        class_id = int(parts[5])
                        
                        if image_name not in annotations:
                            annotations[image_name] = []
                        
                        annotations[image_name].append({
                            'bbox': bbox,
                            'category_id': class_id
                        })
        
        return annotations
    
    def _load_custom_json_annotations(self):
        """加载自定义JSON格式标注（交通灯检测竞赛格式）"""
        annotations = {}
        
        # 查找train.json文件的可能位置
        possible_paths = []
        
        # 1. 当前工作目录（项目根目录）
        current_dir = os.getcwd()
        possible_paths.append(os.path.join(current_dir, 'train.json'))
        
        # 2. annotation_dir
        if self.annotation_dir and os.path.exists(self.annotation_dir):
            possible_paths.append(os.path.join(self.annotation_dir, 'train.json'))
        
        # 3. image_dir的父目录
        if self.image_dir:
            parent_dir = os.path.dirname(self.image_dir)
            if os.path.exists(parent_dir):
                possible_paths.append(os.path.join(parent_dir, 'train.json'))
        
        # 4. 项目根目录（相对于image_dir）
        if self.image_dir:
            # 尝试向上查找项目根目录
            current_path = self.image_dir
            for _ in range(3):  # 最多向上查找3层
                parent = os.path.dirname(current_path)
                if parent == current_path:
                    break
                possible_paths.append(os.path.join(parent, 'train.json'))
                current_path = parent
        
        # 查找存在的文件
        json_path = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                json_path = path
                break
        
        if json_path is None:
            print("警告: 未找到train.json文件，尝试的路径:")
            for path in possible_paths:
                print(f"  - {path}")
            return annotations
        
        print(f"加载标注文件: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"错误: 无法读取train.json文件: {str(e)}")
            return annotations
        
        # 统计信息
        total_annotations = 0
        skipped_ignore = 0
        skipped_empty_inbox = 0
        skipped_difficult = 0
        valid_annotations = 0
        
        # 处理每个标注
        for ann in data.get('annotations', []):
            total_annotations += 1
            
            # 获取文件名（去除路径，只保留文件名）
            filename_with_path = ann.get('filename', '')
            # 处理Windows路径分隔符，统一转换为正斜杠
            filename_with_path = filename_with_path.replace('\\', '/')
            # 提取文件名（去除目录路径）
            filename = os.path.basename(filename_with_path)
            
            # 跳过ignore=1的标注
            if ann.get('ignore', 0) == 1:
                skipped_ignore += 1
                continue
            
            # 获取inbox（交通灯内部的颜色信息）
            inbox_list = ann.get('inbox', [])
            
            # 如果inbox为空，跳过
            if not inbox_list:
                skipped_empty_inbox += 1
                continue
            
            # 为每个inbox创建一个标注
            if filename not in annotations:
                annotations[filename] = []
            
            for inbox in inbox_list:
                color = inbox.get('color', '').lower()
                inbox_bbox = inbox.get('bndbox', {})
                
                # 跳过difficult或truncated的标注（可选）
                if inbox.get('difficult', 0) == 1 or inbox.get('truncated', 0) == 1:
                    skipped_difficult += 1
                    continue
                
                # 获取边界框坐标
                xmin = inbox_bbox.get('xmin', 0)
                ymin = inbox_bbox.get('ymin', 0)
                xmax = inbox_bbox.get('xmax', 0)
                ymax = inbox_bbox.get('ymax', 0)
                
                # 验证bbox有效性
                if xmax <= xmin or ymax <= ymin:
                    continue
                
                # 转换为类别ID
                class_id = self._class_name_to_id(color)
                
                # 只添加有效的类别（非背景）
                if class_id > 0:
                    annotations[filename].append({
                        'bbox': [xmin, ymin, xmax, ymax],
                        'category_id': class_id
                    })
                    valid_annotations += 1
        
        print(f"标注统计:")
        print(f"  总标注数: {total_annotations}")
        print(f"  跳过(ignore=1): {skipped_ignore}")
        print(f"  跳过(空inbox): {skipped_empty_inbox}")
        print(f"  跳过(difficult/truncated): {skipped_difficult}")
        print(f"  有效标注框数: {valid_annotations}")
        print(f"  有标注的图像数: {len(annotations)}")
        
        return annotations
    
    def _class_name_to_id(self, class_name):
        """将类别名称转换为ID"""
        class_mapping = {
            'red': 1,
            'yellow': 2,
            'green': 3,
            'background': 0
        }
        return class_mapping.get(class_name.lower(), 0)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # 准备标注
        if image_file in self.annotations:
            anns = self.annotations[image_file]
            bboxes = [ann['bbox'] for ann in anns]
            labels = [ann['category_id'] for ann in anns]
        else:
            bboxes = []
            labels = []
        
        # 应用数据增强/变换
        if bboxes:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']
        else:
            transformed = self.transform(image=image, bboxes=[], labels=[])
            image = transformed['image']
            bboxes = []
            labels = []
        
        # 转换为tensor格式
        if self.is_train:
            # 训练时需要返回标注（即使没有标注也要返回空tensor）
            if bboxes:
                # 验证和过滤无效bbox
                valid_bboxes = []
                valid_labels = []
                for bbox, label in zip(bboxes, labels):
                    x_min, y_min, x_max, y_max = bbox
                    # 确保bbox有效
                    if x_max > x_min and y_max > y_min and x_min >= 0 and y_min >= 0:
                        # 确保bbox在图像范围内（考虑resize后的尺寸）
                        if x_max <= self.image_size and y_max <= self.image_size:
                            valid_bboxes.append(bbox)
                            valid_labels.append(label)
                
                if valid_bboxes:
                    target = {
                        'boxes': torch.tensor(valid_bboxes, dtype=torch.float32),
                        'labels': torch.tensor(valid_labels, dtype=torch.long),
                        'image_id': torch.tensor([idx], dtype=torch.int64),
                        'area': torch.tensor([(b[2]-b[0])*(b[3]-b[1]) for b in valid_bboxes], dtype=torch.float32),
                        'iscrowd': torch.zeros(len(valid_bboxes), dtype=torch.int64)
                    }
                else:
                    # 所有bbox都无效，返回空tensor
                    target = {
                        'boxes': torch.zeros((0, 4), dtype=torch.float32),
                        'labels': torch.zeros((0,), dtype=torch.long),
                        'image_id': torch.tensor([idx], dtype=torch.int64),
                        'area': torch.zeros((0,), dtype=torch.float32),
                        'iscrowd': torch.zeros((0,), dtype=torch.int64)
                    }
            else:
                # 没有标注时返回空tensor
                target = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.zeros((0,), dtype=torch.long),
                    'image_id': torch.tensor([idx], dtype=torch.int64),
                    'area': torch.zeros((0,), dtype=torch.float32),
                    'iscrowd': torch.zeros((0,), dtype=torch.int64)
                }
            return image, target
        else:
            # 测试时只返回图像
            return image, {
                'image_id': image_file,
                'original_size': (original_width, original_height)
            }


def collate_fn(batch):
    """数据批处理函数（必须在模块级别定义，以便Windows multiprocessing可以pickle）"""
    if isinstance(batch[0][1], dict) and 'boxes' in batch[0][1]:
        # 训练数据
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets
    else:
        # 测试数据
        images = [item[0] for item in batch]
        metadata = [item[1] for item in batch]
        return images, metadata


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

