import numpy as np
import math

# 隐私保护相关功能

class OUE:
    def __init__(self, epsilon, d, map_func=None):
        # 隐私预算
        self.epsilon = epsilon
        # 数据维度
        self.d = d
        # 映射函数,用于将数据映射到索引
        self.map_func = map_func if map_func is not None else lambda x: x

        # 聚合数据(真实计数)
        self.aggregated_data = np.zeros(self.d, dtype=int)
        # 无偏调整后的数据
        self.adjusted_data = np.zeros(self.d, dtype=int)

        # 用户数量
        self.n = 0

        # 1保持为1的概率
        self.p = 0.5
        # 0变为1的概率
        self.q = 1 / (math.pow(math.e, self.epsilon) + 1)

    def _aggregate(self, index):
        # 聚合数据,计数加1
        self.aggregated_data[index] += 1
        self.n += 1

    def privatise(self, data):
        # 将数据映射到索引并聚合
        index = self.map_func(data)
        self._aggregate(index)

    def adjust(self):
        """对聚合数据进行OUE (Optimized Unary Encoding)扰动处理和无偏估计调整
        
        处理步骤:
        1. 对原始值为1的数据进行扰动:
           - tmp_perturbed_1存储原始值为1的计数
           - 以概率p保持为1,以概率(1-p)变为0
        2. 对原始值为0的数据进行扰动:
           - tmp_perturbed_0存储原始值为0的计数
           - 以概率q变为1,以概率(1-q)保持为0
        3. 进行无偏估计调整:
           - 使用公式 (est_count - n*q)/(p-q) 消除扰动带来的偏差
           - n为总样本数,p和q为扰动概率参数
        """
        # 第一步:处理原始值为1的数据
        # 复制原始计数数据(每个维度上值为1的数量)
        tmp_perturbed_1 = np.copy(self.aggregated_data)
        # 对每个计数值进行二项分布采样,以概率p保持为1
        est_count = np.random.binomial(tmp_perturbed_1, self.p)

        # 第二步:处理原始值为0的数据
        # 计算每个维度上值为0的数量
        tmp_perturbed_0 = self.n - np.copy(self.aggregated_data)
        # 对值为0的数据进行二项分布采样,以概率q变为1
        est_count += np.random.binomial(tmp_perturbed_0, self.q)

        # 第三步:无偏估计调整
        # 使用OUE的无偏估计公式进行调整,消除扰动带来的统计偏差
        self.adjusted_data = (est_count - self.n * self.q)/(self.p-self.q)

    @property
    def non_negative_data(self):
        # 返回调整后的非负数据,将负值置为0
        data = np.zeros_like(self.adjusted_data)
        for i in range(len(self.adjusted_data)):
            if self.adjusted_data[i] > 0:
                data[i] = self.adjusted_data[i]
        return data
