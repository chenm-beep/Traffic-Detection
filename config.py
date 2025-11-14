"""
配置文件 - 交通灯检测竞赛
"""
import os

class Config:
    """项目配置类"""
    
    # 数据路径
    DATA_ROOT = './data'
    TRAIN_DATA_DIR = os.path.join(DATA_ROOT, 'train')
    TEST_DATA_DIR = os.path.join(DATA_ROOT, 'test')
    ANNOTATIONS_DIR = os.path.join(DATA_ROOT, 'annotations')
    
    # 模型路径
    MODEL_DIR = './models'
    CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
    OUTPUT_DIR = './outputs'
    SUBMISSION_DIR = './submissions'
    
    # 模型参数
    IMAGE_SIZE = 512  # 输入图像尺寸
    BATCH_SIZE = 2
    NUM_EPOCHS = 16
    LEARNING_RATE = 0.001  # 降低学习率，防止梯度爆炸导致NaN
    NUM_WORKERS = 4  # Windows上建议设为0，Linux/Mac可以设为4
    
    # 数据增强
    USE_AUGMENTATION = True
    
    # 类别信息（根据实际竞赛调整）
    NUM_CLASSES = 4  # 背景 + 3种交通灯状态（红、黄、绿）
    CLASS_NAMES = ['background', 'red', 'yellow', 'green']
    
    # 设备
    DEVICE = 'cuda'  # 'cuda' 或 'cpu'
    
    # 训练参数
    VAL_SPLIT = 0.2  # 验证集比例
    SAVE_INTERVAL = 10  # 每N个epoch保存一次模型
    
    # 推理参数
    CONF_THRESHOLD = 0.5  # 置信度阈值
    IOU_THRESHOLD = 0.5  # NMS IoU阈值
    
    # 提交文件格式
    SUBMISSION_FORMAT = 'json'  # 竞赛要求JSON格式
    
    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        dirs = [
            cls.DATA_ROOT,
            cls.TRAIN_DATA_DIR,
            cls.TEST_DATA_DIR,
            cls.ANNOTATIONS_DIR,
            cls.MODEL_DIR,
            cls.CHECKPOINT_DIR,
            cls.OUTPUT_DIR,
            cls.SUBMISSION_DIR
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

