from typing import List, Tuple
from grid import Grid, GridMap
import numpy as np
import json
import math

# 工具函数
def list_to_dict(l: List):
    """将列表转换为字典
    
    参数:
        l: 输入列表
        
    返回:
        字典,其中键为列表元素,值为元素在列表中的索引
    """
    d = {}
    # 遍历列表,将元素作为键,索引作为值存入字典
    for (index, val) in enumerate(l):
        d[val] = index
    return d


def t_dataset_stats(dataset: List[List[Tuple[float, float, float, float, int]]], stats_name: str):
    """
    Used in budget-division strategy
    Get statistics of the transition-formed dataset, the name of the data file
    should be '{dataset_name}_transition.pkl'
    dataset: [[(x0, y0, x1, y1, flag), ...], ...]
    """
    xs, ys = [], []
    for t_l in dataset:
        for (x0, y0, x1, y1, flag) in t_l:
            if flag:
                continue
            xs.extend([x0, x1])
            ys.extend([y0, y1])
    stats = {'min_x': min(xs), 'min_y': min(ys), 'max_x': max(xs), 'max_y': max(ys)}
    with open(stats_name, 'w') as f:
        json.dump(stats, f)
    return stats


def tid_dataset_stats(dataset: List[List[Tuple[float, float, float, float, int, int]]], stats_name: str):
    """
    用于人口划分策略
    获取带有用户ID的轨迹数据集的统计信息,数据文件名应为'{dataset_name}_transition_id.pkl'
    dataset: [[(x0, y0, x1, y1, flag, uid), ...], ...] 
    其中:
    - x0,y0: 轨迹起点坐标
    - x1,y1: 轨迹终点坐标  
    - flag: 轨迹状态标记
    - uid: 用户ID
    """
    # 初始化存储所有x坐标和y坐标的列表
    xs, ys = [], []
    
    # 遍历数据集中的每个时间戳
    for t_l in dataset:
        # 遍历每个时间戳中的所有轨迹点
        for (x0, y0, x1, y1, flag, uid) in t_l:
            # 跳过标记为特殊状态的轨迹点
            if flag:
                continue
            # 将起点和终点的坐标添加到对应列表
            xs.extend([x0, x1])
            ys.extend([y0, y1])
    
    # 计算坐标的边界值
    stats = {'min_x': min(xs), 'min_y': min(ys), 'max_x': max(xs), 'max_y': max(ys)}
    
    # 将统计结果保存到JSON文件
    with open(stats_name, 'w') as f:
        json.dump(stats, f)
        
    return stats


def xy2grid(xy_list: List[Tuple[float, float]], grid_map: GridMap):
    """
    将坐标点列表转换为对应的网格列表
    
    参数:
        xy_list: 坐标点列表,每个元素为(x,y)元组
        grid_map: 网格地图对象
        
    返回:
        grid_list: 对应的网格列表
    """
    grid_list = []
    for pos in xy_list:  # 遍历每个坐标点
        found = False
        # 遍历网格地图中的每个网格
        for i in range(len(grid_map.map)):
            for j in range(len(grid_map.map[i])):
                # 判断坐标点是否在当前网格内
                if grid_map.map[i][j].in_cell(pos):
                    # 将找到的网格添加到结果列表
                    grid_list.append(grid_map.map[i][j])
                    # 设置找到标志
                    found = True
                    # 跳出循环  
                    break
            # 如果找到网格,跳出外层循环
            if found:
                break

    return grid_list


def xyt2grid(xy_list: List[Tuple[float, float, int]], grid_map: GridMap):
    grid_list = []
    for (x, y, t) in xy_list:
        found = False
        for i in range(len(grid_map.map)):
            for j in range(len(grid_map.map[i])):
                if grid_map.map[i][j].in_cell((x, y)):
                    grid_list.append((grid_map.map[i][j], t))
                    found = True
                    break
            if found:
                break

    return grid_list


def grid_index_map_func(g: Grid, grid_map: GridMap):
    """
    将网格映射到索引: (i, j) => int
    
    参数:
        g: 网格对象
        grid_map: 网格地图对象
        
    返回:
        网格的一维索引: i*列数+j,其中i和j是网格的行列索引
    """
    i, j = g.index
    return i * len(grid_map.map[0]) + j


def grid_index_inv_func(index: int, grid_map: GridMap):
    """
    Inverse function of grid_index_map_func
    """
    i = index // len(grid_map.map[0])
    j = index % len(grid_map.map[0])
    return grid_map.map[i][j]


def pair_grid_index_map_func(grid_pair: Tuple[Grid, Grid], grid_map: GridMap):
    """
    Map a pair of grid to index: (g1, g2) => (i1, i2) => int
    Firstly map (g1, g2) to a matrix of [N x N], where N is
    the total number of grids
    return: i1 * N + i2
    """
    g1, g2 = grid_pair
    index1 = grid_index_map_func(g1, grid_map)
    index2 = grid_index_map_func(g2, grid_map)

    return index1 * grid_map.size + index2


def kl_divergence(prob1, prob2):
    prob1 = np.asarray(prob1)
    prob2 = np.asarray(prob2)

    kl = np.log((prob1 + 1e-8) / (prob2 + 1e-8)) * prob1

    return np.sum(kl)


def js_divergence(prob1, prob2):
    prob1 = np.asarray(prob1)
    prob2 = np.asarray(prob2)

    avg_prob = (prob1 + prob2) / 2

    return 0.5 * kl_divergence(prob1, avg_prob) + 0.5 * kl_divergence(prob2, avg_prob)


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def allocation_p(e, w, alpha=1):
    return alpha / w * math.log(e + 1)


def get_travel_distance(t: List[Tuple[float, float, int]]):
    dist = 0
    for i in range(len(t) - 1):
        curr_p = (t[i][0], t[i][1])
        next_p = (t[i + 1][0], t[i + 1][1])
        dist += euclidean_distance(curr_p, next_p)

    return dist


def get_diameter(t: List[Tuple[float, float,int]]):
    max_d = 0
    for i in range(len(t)):
        for j in range(i+1, len(t)):
            max_d = max(max_d, euclidean_distance((t[i][0],t[i][1]), (t[j][0],t[j][1])))

    return max_d


def pass_through(t: List[Tuple[Grid, int]], g: Grid):
    for t_g in t:
        if t_g[0].index == g.index:
            return True

    return False


