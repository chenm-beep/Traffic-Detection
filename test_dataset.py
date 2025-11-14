"""测试数据集加载"""
from dataset import TrafficLightDataset
from config import Config

print("测试数据集加载...")
print(f"图像目录: {Config.TRAIN_DATA_DIR}")
print(f"标注目录: {Config.ANNOTATIONS_DIR}")

ds = TrafficLightDataset(
    image_dir=Config.TRAIN_DATA_DIR,
    annotation_dir=Config.ANNOTATIONS_DIR,
    annotation_format='custom_json',
    image_size=512,
    is_train=True,
    use_augmentation=False
)

print(f'\n数据集大小: {len(ds)}')
print(f'标注数量: {len(ds.annotations)}')

if len(ds.annotations) > 0:
    print("\n前5个有标注的图像:")
    count = 0
    for filename, anns in list(ds.annotations.items())[:5]:
        print(f"  {filename}: {len(anns)} 个标注框")
        count += 1
    
    # 测试加载一个样本
    import random
    valid_indices = [i for i in range(len(ds)) if len(ds[i][1]['boxes']) > 0]
    if valid_indices:
        idx = random.choice(valid_indices[:10])
        img, target = ds[idx]
        print(f"\n测试样本 {idx}:")
        print(f"  图像形状: {img.shape}")
        print(f"  标注框数: {len(target['boxes'])}")
        if len(target['boxes']) > 0:
            print(f"  第一个框: {target['boxes'][0]}")
            print(f"  标签: {target['labels'][0]}")
else:
    print("\n警告: 没有找到任何标注！")

