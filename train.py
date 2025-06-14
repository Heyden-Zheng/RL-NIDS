import json
import os
from datetime import datetime
from utils.data_loader import DataLoader
from models.rl_nids import RLNIDS
from utils.metrics import compute_metrics
from config import Config


def main():
    # 初始化配置和数据加载器
    config = Config()
    data_loader = DataLoader()
    data = data_loader.load_data()  # 加载数据

    # 初始化RL-NIDS模型
    rl_nids = RLNIDS(data_loader)

    # 训练模型
    X_train, X_test = rl_nids.train()

    # 在测试集上评估
    y_test = data['y_test']
    y_pred = rl_nids.predict(X_test)

    # 计算评估指标
    metrics = compute_metrics(y_test, y_pred)
    print("\nTest Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # 保存结果到JSON文件

    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result = {
        "dataset": config.DATASET,
        "timestamp": timestamp,
        "batch_size": config.BATCH_SIZE,
        "metrics": metrics
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/result_{timestamp}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to results/result_{timestamp}.json")

    # 保存模型（带时间戳）
    if config.SAVE_MODEL:
        os.makedirs(config.MODEL_PATH, exist_ok=True)
        model_path = os.path.join(config.MODEL_PATH, f"rl_nids_{timestamp}.pth")
        rl_nids.save_model(model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()