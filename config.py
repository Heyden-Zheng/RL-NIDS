import os


class Config:
    # 数据集配置
    DATASET = 'NSL-KDD'  # 'NSL-KDD' 或 'AWID'
    DATA_PATH = os.path.join('data', DATASET)

    # FVRL模块配置
    FVRL_HIDDEN_DIM = 10 if DATASET == 'NSL-KDD' else 5
    FVRL_CLUSTER_NUMS = [5, 10, 15, 20, 25, 30]  # 多粒度聚类数量

    # NNRL模块配置
    NNRL_DIMS = [64, 32, 16, 5] if DATASET == 'NSL-KDD' else [128, 64, 32, 5]
    TRIPLET_MARGIN = 1.0  # 三元组损失边界
    TRIPLET_ALPHA = 0.5  # 三元组损失权重

    # 训练配置
    BATCH_SIZE = 256
    EPOCHS = 20
    LEARNING_RATE = 0.0006
    VALIDATION_SPLIT = 0.2

    # 其他配置
    SEED = 42
    SAVE_MODEL = True
    MODEL_PATH = 'models/saved_models'

    # 分类特征列名 (根据数据集不同而不同)
    if DATASET == 'NSL-KDD':
        # 9个分类特征
        categorical_cols = ['protocol_type', 'service', 'flag',
                            'land', 'logged_in', 'root_shell',
                            'su_attempted', 'is_host_login', 'is_guest_login']
    elif DATASET == 'AWID':
        categorical_cols = ['radiotap.channel.freq', 'radiotap.channel.type.cck',
                          'wlan.fc.type_subtype', 'wlan.fc.type', 'wlan.fc.subtype',
                          'wlan.fc.ds', 'wlan.fc.frag', 'wlan.fc.retry',
                          'wlan.fc.pwrmgt', 'wlan.fc.moredata', 'wlan.fc.protected',
                          'wlan.wep.key', 'wlan.qos.priority', 'wlan.qos.bit4']
    else:
        raise ValueError(f"Unsupported dataset: {DATASET}")



