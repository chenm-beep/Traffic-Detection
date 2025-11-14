"""
项目初始化脚本
创建必要的目录结构
"""
import os
from config import Config

def setup_project():
    """初始化项目目录结构"""
    print("正在创建项目目录结构...")
    
    # 创建所有必要的目录
    Config.create_dirs()
    
    print("✓ 目录结构创建完成！")
    print("\n项目目录结构:")
    print(f"  {Config.DATA_ROOT}/")
    print(f"    ├── train/          # 训练图像目录")
    print(f"    ├── test/           # 测试图像目录")
    print(f"    └── annotations/    # 标注文件目录")
    print(f"  {Config.MODEL_DIR}/")
    print(f"    └── checkpoints/    # 模型检查点目录")
    print(f"  {Config.OUTPUT_DIR}/   # 输出目录")
    print(f"  {Config.SUBMISSION_DIR}/  # 提交文件目录")
    
    print("\n下一步:")
    print("1. 从竞赛页面下载数据集")
    print("2. 将训练图像放入 data/train/")
    print("3. 将测试图像放入 data/test/")
    print("4. 将标注文件放入 data/annotations/")
    print("5. 运行训练: python train.py")
    print("6. 运行推理: python inference.py --checkpoint models/checkpoints/best_model.pth")

if __name__ == '__main__':
    setup_project()

