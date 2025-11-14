# 数据格式使用指南

## 文件说明

### 1. train.json - 训练标注文件

这是竞赛提供的训练集标注文件，格式如下：

```json
{
    "annotations": [
        {
            "filename": "train_images\\00001.jpg",
            "bndbox": {
                "xmin": 1026.5,
                "ymin": 741.8,
                "xmax": 1077.5,
                "ymax": 910.9
            },
            "inbox": [
                {
                    "color": "red",
                    "shape": "0",
                    "bndbox": {
                        "xmin": 1037.2,
                        "ymin": 750.8,
                        "xmax": 1068.6,
                        "ymax": 800.3
                    },
                    "occluded": 0,
                    "truncated": 0,
                    "difficult": 0,
                    "value": -1
                }
            ],
            "truncated": 0,
            "occluded": 0,
            "ignore": 0
        }
    ]
}
```

**关键字段说明：**
- `filename`: 图像文件名（带路径）
- `bndbox`: 交通灯外框（整个交通灯的位置）
- `inbox`: 交通灯内部颜色信息数组（这是我们要检测的目标）
  - `color`: 颜色（"red", "green", "yellow"）
  - `bndbox`: 颜色区域的位置
- `ignore`: 是否忽略此标注（1=忽略，0=使用）

**使用方法：**
1. 将 `train.json` 放在项目根目录或 `data/annotations/` 目录
2. 训练时会自动查找并加载此文件

### 2. submit_example.json - 提交示例文件

这是竞赛要求的提交文件格式示例：

```json
{
    "annotations": [
        {
            "filename": "test_images\\00007.jpg",
            "conf": 1,
            "box": {
                "xmin": 1256.4,
                "ymin": 999.5,
                "xmax": 1267.3,
                "ymax": 1013
            },
            "label": "green"
        }
    ]
}
```

**关键字段说明：**
- `filename`: 测试图像文件名（必须包含 `test_images\` 前缀）
- `conf`: 置信度（0-1之间的浮点数）
- `box`: 边界框坐标
  - `xmin`, `ymin`, `xmax`, `ymax`: 边界框坐标
- `label`: 类别标签（"red", "green", "yellow"）

**使用方法：**
- 运行推理脚本后，会自动生成符合此格式的JSON文件
- 生成的文件可以直接提交到竞赛平台

## 数据组织

### 推荐的文件结构：

```
CV_1/
├── train.json                    # 训练标注文件（放在根目录或data/annotations/）
├── data/
│   ├── train/                     # 训练图像
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   └── ...
│   ├── test/                      # 测试图像
│   │   ├── 00007.jpg
│   │   └── ...
│   └── annotations/
│       └── submit_example.json    # 提交示例（参考用）
```

## 使用步骤

### 1. 准备数据

```bash
# 确保train.json在项目根目录或data/annotations/目录
# 确保训练图像在data/train/目录
# 确保测试图像在data/test/目录
```

### 2. 训练模型

```bash
# 使用默认的custom_json格式（会自动识别train.json）
python train.py

# 或者显式指定格式
python train.py --annotation-format custom_json
```

### 3. 生成提交文件

```bash
# 使用训练好的模型生成提交文件
python inference.py \
    --checkpoint models/checkpoints/best_model.pth \
    --format json

# 生成的文件会自动保存到 submissions/ 目录
# 格式完全符合竞赛要求，可以直接提交
```

## 注意事项

1. **train.json位置**：
   - 代码会自动在以下位置查找 `train.json`：
     - `data/annotations/train.json`
     - `data/train.json`（父目录）
     - 项目根目录的 `train.json`

2. **文件名处理**：
   - train.json中的文件名可能包含路径（如 `train_images\00001.jpg`）
   - 代码会自动提取纯文件名进行匹配

3. **忽略标注**：
   - `ignore=1` 的标注会被自动跳过
   - `inbox` 为空的标注也会被跳过

4. **提交文件格式**：
   - 必须使用JSON格式（`--format json`）
   - 文件名必须包含 `test_images\` 前缀
   - 标签必须是 "red", "green", "yellow" 之一

## 常见问题

**Q: train.json应该放在哪里？**
A: 可以放在项目根目录、`data/` 目录或 `data/annotations/` 目录，代码会自动查找。

**Q: 训练时提示找不到标注文件？**
A: 检查train.json是否在正确位置，文件名必须是 `train.json`（不区分大小写）。

**Q: 提交文件格式不对？**
A: 确保使用 `--format json` 参数，生成的文件格式会自动匹配submit_example.json。

**Q: 如何验证提交文件格式？**
A: 可以对比生成的JSON文件和submit_example.json的结构是否一致。










