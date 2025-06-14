import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from config import Config


class DataLoader:
    '''数据加载与预处理'''
    def __init__(self):
        self.config = Config()
        self.categorical_cols = []
        self.numeric_cols = []
        self.label_col = 'label'

    def load_data(self):
        """加载并预处理数据集"""
        if self.config.DATASET == 'NSL-KDD':
            return self._load_nsl_kdd()
        elif self.config.DATASET == 'AWID':
            return self._load_awid()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.DATASET}")

    def _load_nsl_kdd(self):
        """加载NSL-KDD数据集"""

        # 定义所有列名（42列），原始数据没有header，所以给它加上
        columns = ['duration',  # 连接持续时间（秒），0 表示瞬时连接（如 UDP）。
                            'protocol_type',  # 协议类型（tcp, udp, icmp）。
                            'service',  # 目标主机服务类型（http, ftp, smtp 等），共 70 种。
                            'flag',  # 连接状态（如 SF 表示正常完成，REJ 表示被拒绝）。
                            'src_bytes',  # 从源主机到目标主机的数据字节数。
                            'dst_bytes',  # 从目标主机到源主机的数据字节数。
                            'land',  # 是否为同主机连接（1=是，0=否）。
                            'wrong_fragment',  # 错误分片数量（通常为 0，异常值可能表示攻击）。
                            'urgent',  # 紧急数据包数量（如 TCP URG 标志）。
                            'hot',  # 过去 2 秒内与相同主机的连接计数。
                            'num_failed_logins',  # 过去 2 秒内失败的登录尝试次数。
                            'logged_in',  # 是否成功登录（1=是，0=否）。
                            'num_compromised',  # 过去 2 秒内检测到的漏洞利用次数。
                            'root_shell',  # 是否获得 root shell（1=是，0=否）。
                            'su_attempted',  # 是否尝试 su 命令提权（1=是，0=否）。
                            'num_root',  # 过去 2 秒内 root 权限访问次数。
                            'num_file_creations',  # 过去 2 秒内文件创建操作次数。
                            'num_shells',  # 过去 2 秒内启动的 shell 数量。
                            'num_access_files',  # 过去 2 秒内敏感文件（如 /etc/passwd）访问次数。
                            'num_outbound_cmds',  # 过去 2 秒内外向 FTP 命令数（通常为 0）。
                            'is_host_login',  # 是否属于主机登录会话（1=是，0=否）。
                            'is_guest_login',  # 是否属于访客登录（1=是，0=否）。
                            'count',  # 过去 2 秒内与当前连接相同服务的连接数。
                            'srv_count',  # 过去 2 秒内与当前连接相同目标主机的连接数。
                            'serror_rate',
                            'srv_serror_rate',
                            'rerror_rate',
                            'srv_rerror_rate',
                            'same_srv_rate',
                            'diff_srv_rate',
                            'srv_diff_host_rate',
                            'dst_host_count',
                            'dst_host_srv_count',
                            'dst_host_same_srv_rate',
                            'dst_host_diff_srv_rate',
                            'dst_host_same_src_port_rate',
                            'dst_host_srv_diff_host_rate',
                            'dst_host_serror_rate',
                            'dst_host_srv_serror_rate',
                            'dst_host_rerror_rate',
                            'dst_host_srv_rerror_rate',
                            'attack_type'  # 连接类型：normal 攻击类型（如 smurf, back, sql_injection）
         ]

        # 定义9个分类特征
        self.categorical_cols = [
            'protocol_type', 'service', 'flag',
            'land', 'logged_in', 'root_shell',
            'su_attempted', 'is_host_login', 'is_guest_login']

        # 定义32个数值特征
        self.numeric_cols = [
            'duration', 'src_bytes', 'dst_bytes',
            'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'num_compromised', 'num_root',
            'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

        # # 分类特征在原始数据的索引
        # categorical_idx = [1, 2, 3, 6, 11, 13, 14, 20, 21]  # 9个分类特征
        # # 数值特征在原始数据的索引
        # numeric_idx = [0, 4, 5, 7, 8, 9, 10, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
        #                35, 36, 37, 38, 39, 40]  # 32个数值特征

        # 加载数据
        train_df = pd.read_csv(os.path.join(self.config.DATA_PATH, 'KDDTrain+.txt'), header=None, names=columns, usecols=range(42))
        test_df = pd.read_csv(os.path.join(self.config.DATA_PATH, 'KDDTest+.txt'), header=None, names=columns, usecols=range(42))

        # 合并训练和测试数据(虽然NSL-KDD数据集已经划分好训练测试集，但分布不均匀)
        df = pd.concat([train_df, test_df])

        # 处理标签
        df['attack_type'] = df['attack_type'].str.replace('.', '', regex=False)

        self.labels = df['attack_type'].unique()
        self.label_map = {label: idx for idx, label in enumerate(self.labels)}  # 定义label_map

        df[self.label_col] = df['attack_type'].map(self.label_map)  # 将字符型标签转换为数值型标签，比如normal为0，neptune为1。

        df = df.drop('attack_type', axis=1)  # 删除原字符标签列

        return self._preprocess_data(df)

    def _load_awid(self):
        """加载AWID数据集"""
        # 定义AWID的特征列（根据论文Table 3）
        self.categorical_cols = ['radiotap.channel.freq', 'radiotap.channel.type.cck',
                                 'wlan.fc.type_subtype', 'wlan.fc.type', 'wlan.fc.subtype',
                                 'wlan.fc.ds', 'wlan.fc.frag', 'wlan.fc.retry',
                                 'wlan.fc.pwrmgt', 'wlan.fc.moredata', 'wlan.fc.protected',
                                 'wlan.wep.key', 'wlan.qos.priority', 'wlan.qos.bit4']
        self.numeric_cols = ['frame.time_epoch', 'frame.time_delta', 'frame.time_relative',
                             'frame.len', 'radiotap.mactime', 'radiotap.datarate',
                             'radiotap.dbm_antsignal', 'wlan.duration']

        # 加载数据
        df = pd.read_csv(os.path.join(self.config.DATA_PATH, 'AWID_dataset.csv'))

        # 处理标签
        labels = df['class'].unique()
        label_map = {label: idx for idx, label in enumerate(labels)}
        df[self.label_col] = df['class'].map(label_map)

        return self._preprocess_data(df)

    def _preprocess_data(self, df):
        """数据预处理"""
        # 标准化数值特征
        scaler = StandardScaler()
        numeric_data = scaler.fit_transform(df[self.numeric_cols])

        # 对分类特征进行one-hot编码（注意：这个one-hot编码在FVRL模块中并不使用，而是为了比较方法准备的。在FVRL中，我们直接使用原始分类特征值）
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        categorical_data = encoder.fit_transform(df[self.categorical_cols])  # 9个大的分类特征下面细分为97个小的分类特征，比如protocol_type包含tcp, udp, icmp  service包含http, ftp, smtp...

        # 合并特征：合并数值和one-hot编码后的分类特征（用于比较方法），但RL-NIDS自己的流程中，分类特征会单独输入FVRL。
        X = np.concatenate([numeric_data, categorical_data], axis=1)  # 32个数值特征在前32列，后面的列就是one-hot后的分类特征
        y = df[self.label_col].values

        '''
        NSL-KDD数据集的训练集和测试集是已经分好的，无需再做拆分处理，直接返回即可。
        '''

        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.config.SEED, stratify=y)

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols,
            'label_map': self.label_map,
            'labels': self.labels,
            'scaler': scaler,
            'encoder': encoder
        }