import random

import numpy as np
from grid import Grid, GridMap
import utils
from typing import List, Tuple
# 轨迹生成相关功能

class SynDB:
    def __init__(self):
        # 所有历史轨迹数据
        self.history_data: List[List[Tuple[Grid, int]]] = []
        # 尚未结束的轨迹
        self.current_data: List[List[Tuple[Grid, int]]] = []
        # 当前时间戳
        self.t = -1

    def generate_new_points(self,
                            markov_mat: np.ndarray,
                            grid_map: GridMap,
                            avg_len: float):
        """根据马尔可夫矩阵生成新的轨迹点"""
        self.t += 1
        for traj in self.current_data:
            # 获取轨迹最后一个网格
            prev_grid = traj[-1][0]
            # 获取相邻网格
            candidates = grid_map.get_adjacent(prev_grid)
            # 添加当前网格到候选网格列表中
            candidates.append(prev_grid.index)
            # 初始化候选网格的转移概率向量,长度为候选网格数量+1(额外的1用于表示轨迹结束)
            candidate_prob = np.zeros(len(candidates) + 1)
            # 获取当前网格在转移矩阵中的行索引
            row = utils.grid_index_map_func(prev_grid, grid_map)

            # 遍历每个候选网格,计算转移概率
            for k, (i, j) in enumerate(candidates):
                # 获取候选网格在转移矩阵中的列索引
                col = utils.grid_index_map_func(grid_map.map[i][j], grid_map)
                # 从转移矩阵中获取转移概率
                prob = markov_mat[row][col]

                # 如果概率为NaN,则设为0
                if np.isnan(prob):
                    candidate_prob[k] = 0
                else:
                    # 否则使用转移矩阵中的概率值
                    candidate_prob[k] = prob

            # 计算轨迹结束的概率
            col = -1
            prob = markov_mat[row][col]
            prob *= min(1.0, len(traj) / avg_len)
            candidate_prob[-1] = prob

            # 根据概率选择下一个网格或结束轨迹
            if candidate_prob.sum() < 0.00001:
                # 如果所有候选网格的概率之和接近0,保持在当前网格
                traj.append((prev_grid, self.t))
            else:
                # 归一化概率分布
                candidate_prob = candidate_prob / candidate_prob.sum()
                # 根据概率随机采样选择下一个网格或结束轨迹
                sample_id = np.random.choice(np.arange(len(candidate_prob)), p=candidate_prob)

                if sample_id == len(candidate_prob) - 1:
                    # 如果采样到最后一个概率,表示轨迹结束
                    continue
                # 获取采样到的网格坐标
                i, j = candidates[sample_id]
                # 将新的网格和时间戳添加到轨迹中
                traj.append((grid_map.map[i][j], self.t))

        # 将已结束的轨迹移到历史数据中
        new_curr_data = []
        for traj in self.current_data:  # 遍历当前活跃的轨迹
            if traj[-1][1] == self.t:   # 如果轨迹的最后一个时间戳等于当前时间
                new_curr_data.append(traj)  # 说明轨迹仍在继续,保留在当前数据中
            else:
                # 轨迹已经结束(最后时间戳小于当前时间),移到历史数据中保存
                self.history_data.append(traj)
        # 更新当前活跃轨迹列表
        self.current_data = new_curr_data

    def generate_new_points_baseline(self,
                                     markov_mat: np.ndarray,
                                     grid_map: GridMap):
        """基准方法生成新的轨迹点,不考虑轨迹结束事件"""
        self.t += 1
        for traj in self.current_data:
            prev_grid = traj[-1][0]
            candidates = grid_map.get_adjacent(prev_grid)
            # 添加自转移
            candidates.append(prev_grid.index)
            candidate_prob = np.zeros(len(candidates))
            row = utils.grid_index_map_func(prev_grid, grid_map)

            for k, (i, j) in enumerate(candidates):
                col = utils.grid_index_map_func(grid_map.map[i][j], grid_map)
                prob = markov_mat[row][col]

                if np.isnan(prob):
                    candidate_prob[k] = 0
                else:
                    candidate_prob[k] = prob

            if candidate_prob.sum() < 0.00001:
                sample_id = np.random.choice(np.arange(len(candidates)))
                i, j = candidates[sample_id]
                traj.append((grid_map.map[i][j], self.t))
            else:
                candidate_prob = candidate_prob / candidate_prob.sum()
                sample_id = np.random.choice(np.arange(len(candidate_prob)), p=candidate_prob)

                i, j = candidates[sample_id]
                traj.append((grid_map.map[i][j], self.t))

    def adjust_data_size(self,
                         markov_mat: np.ndarray,
                         target_n: int,
                         grid_map: GridMap,
                         quit_distribution: np.ndarray):
        """调整合成数据库大小到目标数量"""
        # 如果当前轨迹数量小于目标数量,需要添加新轨迹
        while self.n < target_n:
            # 根据马尔可夫矩阵最后一行(表示轨迹起始状态)计算进入分布概率
            prob = markov_mat[-1] / markov_mat[-1].sum()
            # 根据概率分布随机选择一个网格作为轨迹起点
            sample_id = np.random.choice(np.arange(grid_map.size), p=prob[:-1])
            # 将新轨迹添加到当前活跃轨迹列表中
            self.current_data.append([(utils.grid_index_inv_func(sample_id, grid_map), self.t)])

        # 如果当前轨迹数量大于目标数量,需要删除一些轨迹
        if self.n > target_n:
            # 如果结束分布概率和接近0,采用随机采样方式
            if np.sum(quit_distribution) < 1e-5:
                # 随机打乱当前轨迹列表
                random.shuffle(self.current_data)
                # 获取需要移除的轨迹
                sample_data = self.current_data[target_n:]

                # 移除每条轨迹的最后一个点(因为最后一个点是当前时刻)
                for idx, traj in enumerate(sample_data):
                    sample_data[idx] = traj[:-1]
                # 将移除的轨迹添加到历史数据中
                self.history_data.extend(sample_data)
                # 保留前target_n条轨迹
                self.current_data = self.current_data[:target_n]
            else:
                # 基于结束分布概率进行采样
                prob = np.zeros(self.n)
                # 计算每条轨迹的结束概率
                for i in range(self.n):
                    # 获取轨迹倒数第二个点所在网格的索引
                    row = utils.grid_index_map_func(self.current_data[i][-2][0], grid_map)
                    prob[i] = quit_distribution[row]
                # 添加小量避免除0
                prob += 1e-8
                # 归一化概率
                prob = prob / prob.sum()
                # 根据概率采样需要移除的轨迹索引
                sample_id = np.random.choice(np.arange(self.n), size=self.n - target_n, replace=False, p=prob)
                # 获取需要保留的轨迹索引
                non_sample_id = list(set(np.arange(self.n)) - set(sample_id))
                # 获取需要移除的轨迹
                new_history_add = [self.current_data[i] for i in sample_id]

                # 移除每条轨迹的最后一个点
                for idx, traj in enumerate(new_history_add):
                    new_history_add[idx] = traj[:-1]
                # 将移除的轨迹添加到历史数据中
                self.history_data.extend(new_history_add)
                # 更新当前活跃轨迹列表
                new_curr_data = [self.current_data[i] for i in non_sample_id]
                self.current_data = new_curr_data

    def random_initialize(self,
                          target_n: int,
                          grid_map: GridMap):
        """随机初始化合成数据库"""
        self.t = 0
        while self.n < target_n:
            sample_id = np.random.choice(np.arange(grid_map.size))
            self.current_data.append([(utils.grid_index_inv_func(sample_id, grid_map), self.t)])

    @property
    def n(self):
        """返回当前轨迹数量"""
        return len(self.current_data)

    @property
    def all_data(self):
        """返回所有轨迹数据"""
        d = self.history_data.copy()
        d.extend(self.current_data)
        return d


class Users:
    """
    User status:
    1: active(available), 0: inactive(not recycled), 2: sampled for reporting, -1: quitted
    """

    def __init__(self):
        self.users = {}

    def register(self, uid):
        try:
            self.users[uid]
        except KeyError:
            self.users[uid] = 1

    def sample(self, p):
        available_users = self.available_users
        sampled_users = random.sample(available_users, int(p * len(available_users)))
        for uid in sampled_users:
            self.users[uid] = 2
        return sampled_users

    def deactivate(self, uid):
        self.users[uid] = 0

    def remove(self, uid):
        self.users[uid] = -1

    def recycle(self, uid):
        if self.users[uid] != -1:
            self.users[uid] = 1

    @property
    def available_users(self):
        a_u = []
        for (uid, state) in self.users.items():
            if state == 1:
                a_u.append(uid)
        return a_u
