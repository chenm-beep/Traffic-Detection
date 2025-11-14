# 快速开始指南

## 1. 环境准备

### 安装Python依赖
```bash
pip install -r requirements.txt
```

### 初始化项目目录
```bash
python setup.py
```

## 2. 数据准备

### 下载数据集
1. 访问竞赛页面：https://www.datafountain.cn/competitions/554/datasets
2. 下载训练集和测试集
3. 解压到项目目录

### 组织数据目录
```
data/
├── train/              # 训练图像
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── test/               # 测试图像
│   ├── test1.jpg
│   └── ...
└── annotations/        # 标注文件
    ├── annotations.json  # COCO格式
    # 或
    ├── image1.txt      # YOLO格式
    └── ...
```

## 3. 配置调整

编辑 `config.py` 文件，根据实际数据调整：

```python
# 类别数量（根据实际竞赛调整）
NUM_CLASSES = 4  # 背景 + 3种交通灯状态

# 类别名称
CLASS_NAMES = ['background', 'red', 'yellow', 'green']

# 数据路径（如果与默认不同）
TRAIN_DATA_DIR = './data/train'
TEST_DATA_DIR = './data/test'
ANNOTATIONS_DIR = './data/annotations'
```

## 4. 训练模型

### 基本训练
```bash
python train.py
```

### 指定参数训练
```bash
python train.py \
    --model-type faster_rcnn \
    --annotation-format coco
```

### 支持的标注格式
- `coco`: COCO JSON格式
- `yolo`: YOLO TXT格式
- `voc`: Pascal VOC XML格式
- `txt`: 自定义TXT格式

## 5. 模型推理

### 生成提交文件
```bash
python inference.py \
    --checkpoint models/checkpoints/best_model.pth \
    --format csv \
    --conf-threshold 0.5
```

### 输出格式
- `csv`: CSV格式（推荐）
- `json`: JSON格式
- `txt`: TXT格式

## 6. 提交结果

1. 检查生成的提交文件：`submissions/submission_*.csv`
2. 根据竞赛要求验证格式
3. 上传到竞赛平台

## 常见问题

### Q: 训练时出现内存不足？
A: 减小 `BATCH_SIZE` 或 `IMAGE_SIZE`（在config.py中）

### Q: 如何查看训练进度？
A: 训练过程会显示loss和mAP，训练历史保存在 `outputs/training_history.json`

### Q: 如何调整模型参数？
A: 编辑 `config.py` 中的相关参数

### Q: 支持哪些模型？
A: 目前支持 Faster R-CNN，可选支持 YOLO（需安装ultralytics）

## 下一步

- 尝试不同的数据增强策略
- 调整超参数优化模型性能
- 使用模型集成提高准确率
- 根据验证集结果调整模型

