import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from utils.triplet_generator import TripletGenerator


class NNRL(nn.Module):
    """对象表示学习模块，对应论文里的样本表征学习网络"""
    # 一个编码器（多层全连接网络）和一个分类器
    def __init__(self, input_dim, num_classes):
        super(NNRL, self).__init__()
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 编码器：将输入映射到低维表示空间。
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.config.NNRL_DIMS[0]),
            nn.ReLU(),
            nn.Linear(self.config.NNRL_DIMS[0], self.config.NNRL_DIMS[1]),
            nn.ReLU(),
            nn.Linear(self.config.NNRL_DIMS[1], self.config.NNRL_DIMS[2]),
            nn.ReLU(),
            nn.Linear(self.config.NNRL_DIMS[2], self.config.NNRL_DIMS[3]),
            nn.ReLU()
        )

        # 分类器：预测样本类别。
        self.classifier = nn.Sequential(
            nn.Linear(self.config.NNRL_DIMS[3], num_classes),
            nn.Softmax(dim=1)
        )

        # 损失函数和优化器
        self.cls_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)

    def forward(self, x):
        """前向传播"""
        representations = self.encoder(x)
        outputs = self.classifier(representations)
        return outputs, representations

    def triplet_loss(self, anchor, positive, negative):
        """计算三元组损失"""
        pos_dist = torch.sum((anchor - positive).pow(2), dim=1)
        neg_dist = torch.sum((anchor - negative).pow(2), dim=1)
        losses = torch.relu(pos_dist - neg_dist + self.config.TRIPLET_MARGIN)
        return torch.mean(losses)

    def train_step(self, X, y, batch_labels):
        """训练步骤"""
        self.train()
        self.optimizer.zero_grad()
        # print(X.shape)
        # print(y.shape)
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # 前向传播
        outputs, representations = self(X_tensor)
        reps_np = representations.detach().cpu().numpy()

        # 计算分类损失
        cls_loss = self.cls_criterion(outputs, y_tensor)

        # 初始化三元组生成器（使用当前批次的标签）
        triplet_generator = TripletGenerator(batch_labels)

        # 生成三元组
        triplets = triplet_generator.generate_triplets(embeddings=reps_np)

        if triplets:
            anchors = representations[[t[0] for t in triplets]]
            positives = representations[[t[1] for t in triplets]]
            negatives = representations[[t[2] for t in triplets]]

            tri_loss = self.triplet_loss(anchors, positives, negatives)
            total_loss = cls_loss + self.config.TRIPLET_ALPHA * tri_loss
        else:
            total_loss = cls_loss

        # 反向传播
        total_loss.backward()
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'cls_loss': cls_loss.item()
        }

    def predict(self, X):
        """预测"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs, _ = self(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()

    def get_representations(self, X):
        """获取表示"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            _, representations = self(X_tensor)
            return representations.cpu().numpy()