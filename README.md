# 交通灯检测竞赛项目

基于DataFountain平台的交通灯检测计算机视觉竞赛项目。

## 项目结构

```
CV_1/
├── config.py          # 配置文件
├── dataset.py         # 数据集加载模块
├── model.py           # 模型定义
├── train.py           # 训练脚本
├── inference.py       # 推理脚本
├── utils.py           # 工具函数
├── main.py            # 主程序入口
├── requirements.txt   # 依赖包
├── README.md          # 项目说明
├── data/              # 数据目录
│   ├── train/         # 训练图像
│   ├── test/          # 测试图像
│   └── annotations/   # 标注文件
├── models/            # 模型目录
│   └── checkpoints/   # 模型检查点
├── outputs/           # 输出目录
└── submissions/       # 提交文件目录
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 初始化项目目录（自动创建文件夹）

运行以下命令会自动创建所有必要的文件夹（data、models、outputs、submissions等）：

```bash
python setup.py
```

**注意**：如果不运行 `setup.py`，直接运行 `train.py` 或 `inference.py` 时也会自动创建这些文件夹。

### 3. 数据准备

1. 从竞赛页面下载数据集
2. 解压数据到 `data/` 目录
3. 确保目录结构如下：
   ```
   data/
   ├── train/          # 训练图像
   ├── test/           # 测试图像
   └── annotations/    # 标注文件（COCO/YOLO/VOC格式）
   ```

## 使用方法

### 训练模型

```bash
# 使用默认配置训练
python train.py

# 指定模型类型和标注格式
python train.py --model-type faster_rcnn --annotation-format coco

# 恢复训练
python train.py --resume models/checkpoints/checkpoint_epoch_50.pth
```

### 推理和生成提交文件

```bash
# 使用最佳模型进行推理
python inference.py --checkpoint models/checkpoints/best_model.pth

# 指定输出格式和路径
python inference.py \
    --checkpoint models/checkpoints/best_model.pth \
    --format csv \
    --output submissions/submission.csv \
    --conf-threshold 0.5
```

## 配置说明

在 `config.py` 中可以调整以下参数：

- **数据路径**: `TRAIN_DATA_DIR`, `TEST_DATA_DIR`, `ANNOTATIONS_DIR`
- **模型参数**: `IMAGE_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`
- **类别信息**: `NUM_CLASSES`, `CLASS_NAMES`
- **推理参数**: `CONF_THRESHOLD`, `IOU_THRESHOLD`

## 支持的标注格式

- **COCO格式**: JSON文件，包含images、annotations、categories
- **YOLO格式**: 每个图像对应一个.txt文件，格式为 `class_id x_center y_center width height`（归一化坐标）
- **Pascal VOC格式**: XML文件
- **TXT格式**: 每行格式为 `image_name x1 y1 x2 y2 class_id`

## 模型说明

### Faster R-CNN
- 使用ResNet-50作为骨干网络
- 支持FPN（特征金字塔网络）
- 预训练权重来自ImageNet

### YOLO（可选）
- 需要安装 `ultralytics` 包
- 支持YOLOv8系列模型

## 提交文件格式

根据竞赛要求，支持以下格式：

- **CSV格式**: `image_id, x_min, y_min, x_max, y_max, class_id, confidence`
- **JSON格式**: 包含predictions数组
- **TXT格式**: 每行一个检测结果

## 评估指标

- **mAP (mean Average Precision)**: 主要评估指标
- 支持IoU阈值可配置

## 注意事项

1. 确保GPU可用（推荐使用CUDA）
2. 根据实际数据集调整类别数量和名称
3. 根据竞赛要求调整提交文件格式
4. 建议使用数据增强提高模型泛化能力

## 常见问题

### Q: 如何修改类别数量？
A: 在 `config.py` 中修改 `NUM_CLASSES` 和 `CLASS_NAMES`

### Q: 如何切换不同的标注格式？
A: 在训练时使用 `--annotation-format` 参数指定格式

### Q: 如何调整模型输入尺寸？
A: 在 `config.py` 中修改 `IMAGE_SIZE` 参数

## 竞赛链接

- 竞赛主页: https://www.datafountain.cn/competitions/554
- 数据与评测: https://www.datafountain.cn/competitions/554/datasets

## 许可证

本项目仅用于学习和竞赛目的。

