from typing import Tuple, List
import random
# 网格化相关功能

class Grid:
    def __init__(self,
                 min_x: float,
                 min_y: float,
                 step_x: float,
                 step_y: float,
                 index: Tuple[int, int]):
        """
        网格类，表示地图中的一个网格单元
        
        参数:
            min_x: 网格左下角x坐标
            min_y: 网格左下角y坐标 
            step_x: 网格x方向长度
            step_y: 网格y方向长度
            index: 网格在矩阵中的索引位置(i,j)
        
        属性:
            min_x, min_y: 网格左下角坐标
            max_x, max_y: 网格右上角坐标
            index: 网格在矩阵中的位置索引
        """
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = min_x + step_x  # 计算右上角x坐标
        self.max_y = min_y + step_y  # 计算右上角y坐标
        self.index = index

    def in_cell(self, p: Tuple[float, float]):
        """
        判断一个点是否在当前网格内
        
        参数:
            p: 待判断的点坐标(x,y)
        返回:
            True: 点在网格内
            False: 点在网格外
        """
        if self.min_x <= p[0] <= self.max_x and self.min_y <= p[1] <= self.max_y:
            return True
        else:
            return False

    def sample_point(self):
        """
        在网格内随机采样一个点
        
        返回:
            (x,y): 采样点的坐标
        """
        x = self.min_x + random.random() * (self.max_x - self.min_x)
        y = self.min_y + random.random() * (self.max_y - self.min_y)
        return x, y

    def equal(self, other):
        """
        判断两个网格是否相同(基于索引判断)
        """
        return self.index == other.index

    def __eq__(self, other):
        """
        重载相等运算符,用于比较两个网格是否相同
        """
        if not type(other) == Grid:
            return False
        return self.index == other.index

    def __hash__(self):
        """
        重载哈希函数,使Grid对象可哈希
        """
        return hash(self.index)


class GridMap:
    def __init__(self,
                 n: int,
                 min_x: float,
                 min_y: float,
                 max_x: float,
                 max_y: float):
        """
        网格化后的地理地图
        参数:
            n: 网格数量
            min_x, min_y, max_x, max_y: 地图边界
        """
        # 边界扩展一点点,避免边界点判断问题
        min_x -= 1e-6
        min_y -= 1e-6
        max_x += 1e-6
        max_y += 1e-6
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        step_x = (max_x - min_x) / n  # 计算x方向网格步长
        step_y = (max_y - min_y) / n  # 计算y方向网格步长
        self.step_x = step_x
        self.step_y = step_y

        # 空间地图,n x n的网格矩阵
        self.map: List[List[Grid]] = list()
        for i in range(n):
            self.map.append(list())
            for j in range(n):
                self.map[i].append(Grid(min_x + step_x * i, min_y + step_y * j, step_x, step_y, (i, j)))

    def get_adjacent(self, g: Grid) -> List[Tuple[int, int]]:
        """
        获取网格g的8个相邻网格
        """
        i, j = g.index
        adjacent_index = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j + 1),
                          (i, j - 1), (i + 1, j + 1), (i + 1, j), (i + 1, j - 1)]
        adjacent_index_new = []
        # 移除超出边界的网格索引
        for index in adjacent_index:
            if len(self.map) > index[0] >= 0 and len(self.map[0]) > index[1] >= 0:
                adjacent_index_new.append(index)
        return adjacent_index_new

    def is_adjacent_grids(self, g1: Grid, g2: Grid):
        """判断两个网格是否相邻"""
        return True if g2.index in self.get_adjacent(g1) else False

    def get_list_map(self):
        """将二维网格矩阵转换为一维列表"""
        list_map = []
        for li in self.map:
            list_map.extend(li)
        return list_map

    def get_all_transition(self):
        """获取所有可能的转移(包括起点和终点转移)
        
        返回一个包含所有可能转移的列表,包括:
        1. 起点转移: flag=1,表示轨迹从该网格开始
        2. 普通转移: flag=0,包括:
           - 在同一网格内的转移
           - 相邻网格之间的转移
        3. 终点转移: flag=2,表示轨迹在该网格结束
        """
        transitions = []
        for g in self.get_list_map():
            # 起点转移 - 轨迹从该网格开始
            transitions.append((Transition(g, g, 1)))
            
            # 获取相邻网格
            adjacent_grids = self.get_adjacent(g)
            # 同一网格内的转移
            transitions.append(Transition(g, g, 0))
            # 添加到相邻网格的转移
            for (i, j) in adjacent_grids:
                transitions.append(Transition(g, self.map[i][j]))
                
            # 终点转移 - 轨迹在该网格结束
            transitions.append(Transition(g, g, 2))
        return transitions

    def get_normal_transition(self):
        """获取普通转移(不包括起点和终点转移)"""
        transitions = []
        for g in self.get_list_map():
            adjacent_grids = self.get_adjacent(g)
            transitions.append(Transition(g, g, 0))
            for (i, j) in adjacent_grids:
                transitions.append(Transition(g, self.map[i][j]))
        return transitions

    @property
    def size(self):
        """返回网格总数"""
        return len(self.map) * len(self.map[0])


class Transition:
    """表示轨迹中的转移
    
    属性:
        g1: 起始网格
        g2: 目标网格 
        flag: 转移类型
            0: 普通转移(相邻网格间或同一网格内)
            1: 起点转移(轨迹从该网格开始)
            2: 终点转移(轨迹在该网格结束)
    """
    def __init__(self, g1: Grid, g2: Grid, flag=0):
        self.g1 = g1  # 起始网格
        self.g2 = g2  # 目标网格
        self.flag = flag  # 转移类型标志

    def __eq__(self, other):
        """判断两个转移是否相等"""
        if not type(other) == Transition:
            return False
        return self.g1 == other.g1 and self.g2 == other.g2 and self.flag == other.flag

    def __hash__(self):
        """计算转移的哈希值,用于字典键和集合元素"""
        return hash(self.g1.index + self.g2.index + (self.flag,))
