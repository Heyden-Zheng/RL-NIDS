import numpy as np
from collections import defaultdict
from config import Config


class TripletGenerator:
    """三元组生成器（批次内版本）"""

    def __init__(self, batch_labels):
        self.config = Config()
        self.batch_labels = batch_labels
        self.batch_size = len(batch_labels)
        self.class_indices = self._get_class_indices()

    def _get_class_indices(self):
        """获取当前批次中每个类别的样本索引"""
        class_indices = defaultdict(list)
        for idx, label in enumerate(self.batch_labels):
            class_indices[label].append(idx)
        return class_indices

    def generate_triplets(self, embeddings=None, n_triplets=None, hard_ratio=0.5):
        """
        生成三元组（基于当前批次）
        :param embeddings: 当前批次的嵌入表示
        :param n_triplets: 要生成的三元组数量（默认为批次大小）
        :param hard_ratio: 困难三元组比例
        """
        if n_triplets is None:
            n_triplets = self.batch_size

        triplets = []
        n_hard = int(n_triplets * hard_ratio)
        n_random = n_triplets - n_hard

        # 生成随机三元组（批次内）
        for _ in range(n_random):
            # 随机选择一个类别作为anchor和positive
            anchor_class = np.random.choice(list(self.class_indices.keys()))
            positive_class = anchor_class

            # 随机选择另一个类别作为negative
            negative_class = np.random.choice([c for c in self.class_indices.keys()
                                               if c != anchor_class])

            # 确保每个类都有样本
            if (len(self.class_indices[anchor_class]) > 0 and
                    len(self.class_indices[positive_class]) > 0 and
                    len(self.class_indices[negative_class]) > 0):
                anchor_idx = np.random.choice(self.class_indices[anchor_class])
                positive_idx = np.random.choice(self.class_indices[positive_class])
                negative_idx = np.random.choice(self.class_indices[negative_class])

                triplets.append((anchor_idx, positive_idx, negative_idx))

        # 如果提供了嵌入，生成困难三元组（批次内）
        if embeddings is not None and n_hard > 0:
            hard_triplets = self._generate_hard_triplets(embeddings, n_hard)
            triplets.extend(hard_triplets)

        return triplets

    def _generate_hard_triplets(self, embeddings, n_triplets):
        """生成困难三元组（基于当前批次）"""
        hard_triplets = []

        for _ in range(n_triplets):
            # 随机选择一个类别作为anchor和positive
            anchor_class = np.random.choice(list(self.class_indices.keys()))
            positive_class = anchor_class

            # 随机选择另一个类别作为negative
            negative_class = np.random.choice([c for c in self.class_indices.keys()
                                               if c != anchor_class])

            # 确保每个类都有样本
            if (len(self.class_indices[anchor_class]) == 0 or
                    len(self.class_indices[positive_class]) == 0 or
                    len(self.class_indices[negative_class]) == 0):
                continue

            # 从anchor类中随机选择一个样本作为anchor
            anchor_idx = np.random.choice(self.class_indices[anchor_class])
            anchor_embed = embeddings[anchor_idx]

            # 计算anchor与同类样本的距离
            positive_dists = []
            for pos_idx in self.class_indices[positive_class]:
                if pos_idx == anchor_idx:
                    continue
                dist = np.linalg.norm(anchor_embed - embeddings[pos_idx])
                positive_dists.append((pos_idx, dist))

            # 选择距离最远的positive（hard positive）
            if positive_dists:
                positive_dists.sort(key=lambda x: x[1], reverse=True)
                positive_idx = positive_dists[0][0]
            else:
                # 如果没有其他同类样本，随机选择一个
                positive_idx = np.random.choice(self.class_indices[positive_class])

            # 计算anchor与不同类样本的距离
            negative_dists = []
            for neg_idx in self.class_indices[negative_class]:
                dist = np.linalg.norm(anchor_embed - embeddings[neg_idx])
                negative_dists.append((neg_idx, dist))

            # 选择距离最近的negative（hard negative）
            if negative_dists:
                negative_dists.sort(key=lambda x: x[1])
                negative_idx = negative_dists[0][0]
            else:
                # 如果没有不同类样本，随机选择一个
                negative_idx = np.random.choice(self.class_indices[negative_class])

            hard_triplets.append((anchor_idx, positive_idx, negative_idx))

        return hard_triplets