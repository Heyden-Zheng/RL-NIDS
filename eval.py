import json
import os
import numpy as np
from utils.data_loader import DataLoader
from models.rl_nids import RLNIDS
from utils.metrics import compute_metrics
from config import Config
'''
目前采用轻量级评估方案，无需执行该文件。
'''


def evaluate_model(model_path=None):
    # 初始化配置和数据加载器
    config = Config()
    data_loader = DataLoader()
    data = data_loader.load_data()

    # 初始化RL-NIDS模型
    rl_nids = RLNIDS(data_loader)

    # 加载预训练模型
    model_path = os.path.join(config.MODEL_PATH, 'rl_nids.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")

    # 准备输入数据
    value_embeddings = rl_nids.train_fvrl()  # 只需要FVRL部分
    X_test = rl_nids.prepare_nnrl_input(value_embeddings, data['X_test'])

    # 加载NNRL模型
    num_classes = len(np.unique(data['y_test']))
    rl_nids.load_model(model_path, X_test.shape[1], num_classes)

    # 在测试集上评估
    y_test = data['y_test']
    y_pred = rl_nids.predict(X_test)

    # 计算评估指标
    metrics = compute_metrics(y_test, y_pred)
    print("\nEvaluation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    evaluate_model()