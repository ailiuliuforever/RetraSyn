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
        # 对原始数据进行扰动处理
        # 如果原始值为1,则以概率p保持为1
        tmp_perturbed_1 = np.copy(self.aggregated_data)
        est_count = np.random.binomial(tmp_perturbed_1, self.p)

        # 如果原始值为0,则以概率q变为1
        tmp_perturbed_0 = self.n - np.copy(self.aggregated_data)
        est_count += np.random.binomial(tmp_perturbed_0, self.q)

        # 进行无偏估计调整
        self.adjusted_data = (est_count - self.n * self.q)/(self.p-self.q)

    @property
    def non_negative_data(self):
        # 返回调整后的非负数据,将负值置为0
        data = np.zeros_like(self.adjusted_data)
        for i in range(len(self.adjusted_data)):
            if self.adjusted_data[i] > 0:
                data[i] = self.adjusted_data[i]
        return data
