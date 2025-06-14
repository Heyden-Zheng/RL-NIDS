import numpy as np
import torch
from models.fvrl import FVRL
from models.nnrl import NNRL
from config import Config
from sklearn.preprocessing import StandardScaler


class RLNIDS:
    """完整的RL-NIDS模型"""

    def __init__(self, data_loader):
        self.config = Config()
        self.data_loader = data_loader
        self.data = data_loader.load_data()
        self.fvrl = None
        self.nnrl = None
        self.triplet_generator = None

    def train_fvrl(self):
        """训练FVRL模块"""
        # 获取one-hot后的分类数据（在FVRL中，直接使用原始分类特征值）
        categorical_data = self.data['X_train'][:, len(self.data['numeric_cols']):]  # 行：取所有行，列：从第33列开始，前32列是数值特征，后面是分类特征

        # 初始化FVRL
        self.fvrl = FVRL(categorical_data, self.data['encoder'])

        # 构建耦合矩阵
        M0, Mc = self.fvrl.build_coupling_matrices()

        # 多粒度聚类
        C = self.fvrl.multi_grain_clustering(M0, Mc)  # # C 将作为后续特征交互模块的输入，提供多粒度的特征值聚类信息。

        # 训练自编码器
        autoencoder = self.fvrl.train_autoencoder(C)  # 自编码器通过降维-重建过程，提取特征值的本质结构，用于后续的特征交互建模。目标是学习特征值的低维表示。

        # 获取特征值嵌入
        value_embeddings = self.fvrl.get_value_embeddings(autoencoder, C)

        return value_embeddings

    def prepare_nnrl_input(self, value_embeddings, X):
        """准备NNRL输入数据（先拼接再标准化）"""
        # 分离数值特征和分类特征
        num_numeric = len(self.data['numeric_cols'])
        numeric_data = X[:, :num_numeric]  # 数值特征（未标准化）
        categorical_data = X[:, num_numeric:]  # 分类特征（One-Hot编码）

        # 获取每个样本的分类特征值嵌入
        sample_embeddings = []
        for sample in categorical_data:
            nonzero_indices = np.where(sample == 1)[0]  # 找到非零特征值索引
            if len(nonzero_indices) > 0:
                emb = np.mean(value_embeddings[nonzero_indices], axis=0)  # 均值聚合
            else:
                emb = np.zeros(value_embeddings.shape[1])  # 无有效特征值时填充零
            sample_embeddings.append(emb)

        # 拼接数值特征和分类特征嵌入
        nnrl_input = np.concatenate([numeric_data, np.array(sample_embeddings)],
                                    axis=1)  # 形状: (n_samples, num_numeric + FVRL_HIDDEN_DIM)

        # 标准化整个输入（数值特征 + 分类嵌入）
        if hasattr(self, 'nnrl_scaler'):  # 若已有标准化器，直接转换
            nnrl_input = self.nnrl_scaler.transform(nnrl_input)
        else:  # 首次调用时拟合标准化器
            self.nnrl_scaler = StandardScaler().fit(nnrl_input)
            nnrl_input = self.nnrl_scaler.transform(nnrl_input)

        return nnrl_input

    def train_nnrl(self, X_train, y_train):
        """训练NNRL模块"""

        # 初始化NNRL
        num_classes = len(np.unique(y_train))  # 标签类别数
        self.nnrl = NNRL(X_train.shape[1], num_classes)  # X_train.shape[1]：输入特征的维度（数值特征 + 分类特征嵌入的拼接维度）。

        # 训练循环
        for epoch in range(self.config.EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.EPOCHS}")

            # 随机打乱数据，提升泛化能力
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # 分批训练
            epoch_loss = 0.0
            for i in range(0, X_shuffled.shape[0], self.config.BATCH_SIZE):
                batch_X = X_shuffled[i:i + self.config.BATCH_SIZE]
                batch_y = y_shuffled[i:i + self.config.BATCH_SIZE]

                # 获取当前批次的标签
                batch_labels = batch_y

                # 执行训练步骤
                metrics = self.nnrl.train_step(batch_X, batch_y, batch_labels)
                epoch_loss += metrics['loss']

                print(
                    f"Batch {i // self.config.BATCH_SIZE + 1} - Loss: {metrics['loss']:.4f} - Cls Loss: {metrics['cls_loss']:.4f}",
                    end='\r')

            print(
                f"\nEpoch {epoch + 1} Average Loss: {epoch_loss / (X_shuffled.shape[0] / self.config.BATCH_SIZE):.4f}")  # 打印平均损失

    def train(self):
        """训练完整RL-NIDS模型"""
        print("Training FVRL module...")
        value_embeddings = self.train_fvrl()

        print("\nPreparing NNRL input...")
        X_train = self.prepare_nnrl_input(value_embeddings, self.data['X_train'])
        X_test = self.prepare_nnrl_input(value_embeddings, self.data['X_test'])

        print("\nTraining NNRL module...")
        self.train_nnrl(X_train, self.data['y_train'])

        return X_train, X_test

    def predict(self, X):
        """使用训练好的模型进行预测"""
        if self.nnrl is None:
            raise ValueError("Model not trained yet. Please call train() first.")
        return self.nnrl.predict(X)

    def get_representations(self, X):
        """获取数据对象的表示"""
        if self.nnrl is None:
            raise ValueError("Model not trained yet. Please call train() first.")
        return self.nnrl.get_representations(X)

    def save_model(self, path):
        """保存模型"""
        if self.nnrl is None:
            raise ValueError("Model not trained yet. Please call train() first.")
        torch.save(self.nnrl.state_dict(), path)

    def load_model(self, path, input_dim, num_classes):
        """加载模型"""
        self.nnrl = NNRL(input_dim, num_classes)
        self.nnrl.load_state_dict(torch.load(path))
        self.nnrl.eval()