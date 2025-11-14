"""
交通灯检测竞赛 - 主程序入口
"""
import argparse
import sys
from config import Config

def main():
    parser = argparse.ArgumentParser(description='交通灯检测竞赛项目')
    parser.add_argument('mode', choices=['train', 'inference'], 
                       help='运行模式: train 或 inference')
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print('=' * 50)
        print('开始训练模式')
        print('=' * 50)
        from train import main as train_main
        train_main()
    elif args.mode == 'inference':
        print('=' * 50)
        print('开始推理模式')
        print('=' * 50)
        print('请使用以下命令进行推理:')
        print('python inference.py --checkpoint <模型路径>')
        print('=' * 50)
        from inference import main as inference_main
        # 如果没有提供checkpoint，显示帮助信息
        if '--checkpoint' not in sys.argv:
            inference_parser = argparse.ArgumentParser(description='推理模式')
            inference_parser.add_argument('--checkpoint', required=True)
            inference_parser.print_help()
        else:
            inference_main()

if __name__ == '__main__':
    main()
