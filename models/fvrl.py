import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from config import Config


class ValueAutoencoder(nn.Module):
    """特征值自编码器"""

    def __init__(self, input_dim, hidden_dim):
        super(ValueAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()  # 因输入 C 是One-Hot编码，输出需通过Sigmoid压缩到 [0,1]。
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class FVRL:
    """分类特征值表示学习模块（无监督）"""

    def __init__(self, categorical_data, encoder):
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categorical_data = categorical_data
        self.encoder = encoder
        self.value_categories = encoder.categories_
        self.value_names = []

        # 生成特征值名称列表（可以理解为“大的分类特征名+小的特征名。如：protocol_type_tcp”）
        for i, col in enumerate(self.config.categorical_cols):
            for val in self.value_categories[i]:
                self.value_names.append(f"{col}_{val}")

        self.m = len(self.value_names)  # 特征值总数（97个）

    def build_coupling_matrices(self):
        """构建值-值耦合矩阵"""
        # 计算互信息矩阵（计算两个特征值之间的归一化互信息 97*97）
        nmi_matrix = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(i, self.m):
                #  获取特征值对 (fi,vi) 和 (fj,vj) 如：fi=0 对应 protocol_type，vi='icmp'
                fi, vi = self._get_feature_and_value(i)
                fj, vj = self._get_feature_and_value(j)

                # 计算边际概率和联合概率
                p_vi = np.mean(self.categorical_data[:, fi] == vi)  # 对应公式(3.1)
                p_vj = np.mean(self.categorical_data[:, fj] == vj)  # 对应公式(3.1)
                p_vi_vj = np.mean((self.categorical_data[:, fi] == vi) &
                                  (self.categorical_data[:, fj] == vj))  # 联合概率，对应公式(3.2)

                if p_vi_vj == 0:
                    nmi = 0
                else:
                    # 计算互信息
                    mi = p_vi_vj * np.log(p_vi_vj / (p_vi * p_vj))  # 公式（3.4）
                    # 计算归一化互信息
                    h_vi = -p_vi * np.log(p_vi) if p_vi > 0 else 0  # 信息熵，公式（3.5）
                    h_vj = -p_vj * np.log(p_vj) if p_vj > 0 else 0
                    nmi = 2 * mi / (h_vi + h_vj) if (h_vi + h_vj) > 0 else 0  # 互信息（两个特征之间的关系），对应公式（3.3）

                # 对称填充nmi_matrix，因为是无序度量
                nmi_matrix[i, j] = nmi
                nmi_matrix[j, i] = nmi

        # 构建基于出现频率的耦合矩阵M0，衡量特征值 vi 的出现频率受特征值 vj 的影响，对应公式（3.6）（3.7）
        M0 = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                fi, vi = self._get_feature_and_value(i)
                fj, vj = self._get_feature_and_value(j)
                p_vi = np.mean(self.categorical_data[:, fi] == vi)
                p_vj = np.mean(self.categorical_data[:, fj] == vj)

                # 获取两个特征之间的NMI
                rho = nmi_matrix[i, j]
                # 添加小epsilon防止除以零
                epsilon = 1e-10
                if p_vj > epsilon:
                    M0[i, j] = rho * (p_vi / p_vj)
                else:
                    # 当p_vj接近0时，使用p_vi作为近似
                    M0[i, j] = rho * p_vi
                # M0[i, j] = nmi_matrix[i, j] * (p_vi / p_vj)

        # 构建基于共现的耦合矩阵Mc，衡量特征值 vi 和 vj 共同出现的概率受特征值 vi 的出现频率的影响 对应公式（3.8）（3.9）
        Mc = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                fi, vi = self._get_feature_and_value(i)
                fj, vj = self._get_feature_and_value(j)

                p_vi = np.mean(self.categorical_data[:, fi] == vi)
                p_vi_vj = np.mean((self.categorical_data[:, fi] == vi) &
                                  (self.categorical_data[:, fj] == vj))

                # 添加小epsilon防止除以零
                epsilon = 1e-10
                if p_vi > epsilon:
                    Mc[i, j] = p_vi_vj / p_vi
                else:
                    # 当p_vi接近0时，使用联合概率作为近似
                    Mc[i, j] = p_vi_vj

        return M0, Mc

    def _get_feature_and_value(self, index):
        """根据索引获取特征和值"""
        cum_sum = 0
        for fi, categories in enumerate(self.value_categories):
            if index < cum_sum + len(categories):
                return fi, categories[index - cum_sum]
            cum_sum += len(categories)
        raise IndexError("Index out of range")

    def multi_grain_clustering(self, M0, Mc):
        """多粒度聚类"""
        cluster_memberships = []  # 存储不同粒度和不同矩阵（M0 和 Mc）的聚类结果（One-Hot 编码形式）。cluster_memberships 是一个列表，包含 2 * len(FVRL_CLUSTER_NUMS) 个矩阵（M0 和 Mc 各对应多个粒度）。

        # 对M0进行多粒度聚类
        # 举例：若 k=5 且 clusters = [0, 2, 1, 0, ...]，则 membership 的每一行是 5 维 One-Hot 向量（如 [1,0,0,0,0] 表示属于第 0 类）。
        for k in self.config.FVRL_CLUSTER_NUMS:
            kmeans = KMeans(n_clusters=k, random_state=self.config.SEED)
            clusters = kmeans.fit_predict(M0)  # fit_predict(M0) 返回每个特征值的聚类标签（clusters），形状为 (m,)，其中 m 是特征值总数。
            membership = np.eye(k)[clusters]  # np.eye(k) 生成单位矩阵（大小为 k×k），np.eye(k)[clusters] 将聚类标签转换为 One-Hot 编码矩阵，形状为 (m, k)。
            cluster_memberships.append(membership)  # 将当前粒度的聚类成员矩阵加入 cluster_memberships。

        # 对Mc进行多粒度聚类
        for k in self.config.FVRL_CLUSTER_NUMS:
            kmeans = KMeans(n_clusters=k, random_state=self.config.SEED)
            clusters = kmeans.fit_predict(Mc)
            membership = np.eye(k)[clusters]  # one-hot编码
            cluster_memberships.append(membership)

        # 合并所有聚类结果
        C = np.concatenate(cluster_memberships, axis=1)
        return C

    def train_autoencoder(self, C, epochs=50, batch_size=256):
        """训练特征值自编码器"""
        # 将NumPy数组转换为PyTorch张量
        C_tensor = torch.FloatTensor(C).to(self.device)

        # 初始化模型
        input_dim = C.shape[1]
        model = ValueAutoencoder(input_dim, self.config.FVRL_HIDDEN_DIM).to(self.device)
        criterion = nn.BCELoss()  # 二元交叉熵
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)  # 自适应学习率优化器

        # 训练循环
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(C_tensor)
            loss = criterion(outputs, C_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:  # 每10轮打印一次损失值
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        # 返回训练完成的自编码器，通过 model.encoder(C_tensor)，可用于提取特征值的低维表示。
        return model

    def get_value_embeddings(self, model, C):
        """获取特征值嵌入"""
        # 通过自编码器的编码部分得到特征值的嵌入表示。
        with torch.no_grad():  # 模型前向推理无需计算梯度，减少内存消耗并加速推理过程
            C_tensor = torch.FloatTensor(C).to(self.device)  # 将输入的多粒度聚类矩阵 C 转换为PyTorch张量。
            embeddings = model.encoder(C_tensor).cpu().numpy()  # 将结果从gpu移回cpu
        return embeddings

