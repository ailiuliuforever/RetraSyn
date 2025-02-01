import pickle

from ldp import OUE
from grid import Grid, GridMap, Transition
from typing import List, Tuple
import utils
import numpy as np
import math
from parse import args
import multiprocessing
import random
from syndb import SynDB
from logger.logger import ConfigParser
import lzma

config = ConfigParser(name='RetraSyn', save_dir='./')
logger = config.get_logger(config.exper_name)

CORES = multiprocessing.cpu_count() // 2
random.seed(2023)
np.random.seed(2023)

logger.info(args)


def spatial_decomposition(xy_l: List[Tuple[float, float, float, float, int]], gm: GridMap):
    """
    空间分解函数，将移动坐标转换为网格坐标
    
    参数:
        xy_l: 同一时间戳的移动列表，每个元素为(x0,y0,x1,y1,flag)的元组
            x0,y0: 起始点坐标
            x1,y1: 终止点坐标 
            flag: 标志位(0:普通轨迹点, 1:轨迹起点, 2:轨迹终点)
        gm: 网格地图对象
    
    返回:
        grid_list: 转换后的网格坐标列表
    """
    # 初始化网格列表,用于存储转换后的网格坐标
    grid_list = []
    # 遍历每个移动点
    for (x0, y0, x1, y1, flag) in xy_l:
        if flag == 0:
            # 普通轨迹点:将起点和终点坐标转换为对应的网格
            g0, g1 = utils.xy2grid([(x0, y0), (x1, y1)], gm)
            grid_list.append((g0, g1, flag))
        elif flag == 1:
            # 轨迹起点:只需要转换终点坐标为网格
            g1 = utils.xy2grid([(x1, y1)], gm)[0]
            grid_list.append((g1, g1, flag))
        else:
            # 轨迹终点:只需要转换起点坐标为网格
            g0 = utils.xy2grid([(x0, y0)], gm)[0]
            grid_list.append((g0, g0, flag))
    return grid_list


def split_traj(traj_stream: List[List[Tuple[Grid, Grid, int]]], gm: GridMap):
    """
    处理非相邻网格之间的转移
    如果两个网格(G1, G2, flag)不相邻,则将其拆分为:
    - 在时间t: (G1, end, 2)表示在G1结束
    - 在时间t+1: (start, G2, 1)表示在G2开始
    
    参数:
        traj_stream: 轨迹流,包含每个时间戳的网格转移列表
        gm: 网格地图对象,用于判断网格是否相邻
    
    返回:
        new_stream: 处理后的轨迹流
    """
    # 初始化新的轨迹流
    new_stream = []
    # 确保new_stream的长度至少等于traj_stream的长度
    # 因为可能需要在t+1时刻添加新的轨迹点
    while len(new_stream) <= len(traj_stream):
        new_stream.append([])
    
    # 遍历每个时间戳
    for t in range(len(traj_stream)):
        # 处理该时间戳下的每个网格转移
        for g1, g2, flag in traj_stream[t]:
            # 如果是轨迹起点或终点,直接添加
            if flag:
                new_stream[t].append((g1, g2, flag))
                continue
            
            # 如果两个网格不相等且不相邻,则拆分转移
            if not g1.equal(g2) and not gm.is_adjacent_grids(g1, g2):
                new_stream[t].append((g1, g1, 2))  # 在g1结束
                new_stream[t + 1].append((g2, g2, 1))  # 在g2开始
            else:
                # 相邻网格间的转移直接添加
                new_stream[t].append((g1, g2, flag))
    return new_stream


def generate_markov_matrix(markov_vec: np.ndarray, trans_domain: List[Transition]):
    """
    生成马尔可夫转移矩阵和终止分布
    
    参数:
        markov_vec: 马尔可夫向量,包含每个转移的概率
        trans_domain: 转移域列表,定义了所有可能的转移
        
    返回:
        markov_mat: 马尔可夫转移矩阵
        end_distribution: 轨迹终止分布
    """
    # 矩阵大小为网格数量+1(额外的1用于表示轨迹的开始和结束状态)
    n = grid_map.size + 1
    # 初始化马尔可夫转移矩阵
    markov_mat = np.zeros((n, n), dtype=float)
    # 初始化终止分布向量
    end_distribution = np.zeros(n - 1)
    
    # 遍历马尔可夫向量中的每个转移概率
    for k in range(len(markov_vec)):
        # 跳过概率为0或负数的转移
        if markov_vec[k] <= 0:
            continue

        # 获取当前转移对象
        trans = trans_domain[k]
        if not trans.flag:
            # 普通转移:从一个网格到另一个网格
            i = utils.grid_index_map_func(trans.g1, grid_map)
            j = utils.grid_index_map_func(trans.g2, grid_map)
        elif trans.flag == 1:
            # 轨迹起点:从虚拟起始状态转移到某个网格
            i = -1  # 最后一行表示轨迹起始状态
            j = utils.grid_index_map_func(trans.g2, grid_map)
        else:
            # 轨迹终点:从某个网格转移到虚拟终止状态
            i = utils.grid_index_map_func(trans.g1, grid_map)
            j = -1  # 最后一列表示轨迹终止状态
            end_distribution[i] = markov_vec[k]
        # 更新转移矩阵中对应位置的概率
        markov_mat[i][j] = markov_vec[k]

    # 按行归一化转移概率(加上小量1e-8避免除0)
    markov_mat = markov_mat / (markov_mat.sum(axis=1).reshape((-1, 1)) + 1e-8)
    # 归一化终止分布
    end_distribution = end_distribution / (end_distribution.sum() + 1e-8)
    return markov_mat, end_distribution


def convert_grid_to_raw(grid_db: List[List[Tuple[Grid, int]]]):
    def traj_grid_to_raw(traj: List[Tuple[Grid, int]]):
        xy_traj = []
        for (g, t) in traj:
            x, y = g.sample_point()
            xy_traj.append((x, y, t))
        return xy_traj

    raw_db = [traj_grid_to_raw(traj) for traj in grid_db]

    return raw_db


def RetraSyn(traj_stream, w: int, eps, trans_domain: List[Transition]):
    """RetraSyn算法的主要实现
    
    参数:
        traj_stream: 轨迹流数据
        w: 时间窗口大小
        eps: 隐私预算
        trans_domain: 转移域列表
    """
    # 将转移域列表转换为字典,方便查找
    trans_domain_map = utils.list_to_dict(trans_domain)

    # 初始化合成数据库
    synthetic_db = SynDB()
    # 存储每个时间戳的转移分布
    trans_distribution = []
    # 存储每个时间戳使用的隐私预算
    used_budget = []
    # 存储每个时间戳的发布结果
    release = []
    # 当前时间窗口内已使用的预算总和
    used_budget_in_curr_w = 0
    # 存储每个时间戳中显著转移的数量
    N_st = []

    # 预热阶段(前两个时间戳)
    for t in range(2):
        # 使用OUE机制进行隐私保护
        oue = OUE(eps / w, len(trans_domain), lambda x: trans_domain_map[x])

        # 对每个转移进行隐私化处理
        for (g1, g2, flag) in traj_stream[t]:
            trans = Transition(g1, g2, flag)
            oue.privatise(trans)
        oue.adjust()

        # 计算估计的转移计数
        est_counts = oue.non_negative_data / oue.n
        print(est_counts);

        # 根据估计计数生成马尔可夫转移矩阵和轨迹终止概率分布
        # markov_mat: 网格间转移概率矩阵
        # end_distribution: 轨迹在各网格结束的概率分布
        markov_mat, end_distribution = generate_markov_matrix(est_counts,trans_domain)
        # 保存当前时间戳的转移矩阵
        trans_distribution.append(markov_mat)
        # 计算并保存归一化后的转移概率分布
        release.append(est_counts / est_counts.sum())

        # 根据当前分布生成新的轨迹点
        # markov_mat: 网格间转移概率矩阵
        # grid_map: 网格地图
        # avg_lens[args.dataset]: 数据集的平均轨迹长度
        synthetic_db.generate_new_points(markov_mat, grid_map, avg_lens[args.dataset])


        # 调整合成数据库大小
        synthetic_db.adjust_data_size(markov_mat, len(traj_stream[t]), grid_map, end_distribution)

        used_budget.append(eps / w)
        used_budget_in_curr_w += eps / w
        N_st.append(0)

    # 主循环阶段
    for t in range(2, len(traj_stream)):
        if not len(traj_stream[t]):
            continue

        # 预算回收:移除窗口外的预算
        if t >= w:
            used_budget_in_curr_w -= used_budget[t - w]
        # 计算剩余预算
        eps_rm = eps - used_budget_in_curr_w

        # 计算偏差:当前发布与过去5个时间戳平均值的差异
        dev = np.abs(np.array(release[-1]) - np.average(release[max(0, t - 5):t], axis=0)).sum()

        # 计算分配比例
        cr = max(0.5, 1 - np.average(N_st[max(0, t - 5):t]) / len(trans_domain))
        p = utils.allocation_p(dev, w, alpha=8)
        p = min(p * cr, 0.6)
        eps_t = p * eps_rm

        # 使用OUE机制处理当前时间戳数据
        oue = OUE(eps_t, len(trans_domain), lambda x: trans_domain_map[x])

        for (g1, g2, flag) in traj_stream[t]:
            trans = Transition(g1, g2, flag)
            oue.privatise(trans)
        oue.adjust()
        f_hat = oue.non_negative_data / oue.n
        f_tilde = release[-1]

        # 选择显著模式:根据方差阈值判断
        variance = 4 * math.exp(eps_t) / (oue.n * (math.exp(eps_t) - 1) ** 2)
        select = (f_tilde - f_hat) ** 2 > variance

        # 合并显著模式和其他模式的计数
        counts = np.zeros(len(trans_domain))
        sig_counts = oue.non_negative_data / oue.n

        for i in range(len(select)):
            if select[i]:
                counts[i] = sig_counts[i]
            else:
                counts[i] = f_tilde[i] * sig_counts.sum()

        used_budget.append(eps_t)
        used_budget_in_curr_w += eps_t
        
        # 生成马尔可夫矩阵
        markov_mat, end_distribution = generate_markov_matrix(counts, trans_domain)

        # 根据当前分布生成新的轨迹点
        synthetic_db.generate_new_points(markov_mat, grid_map, avg_lens[args.dataset])

        # 检查进入分布:如果当前时间戳的进入分布为0,使用历史有效的进入分布
        if markov_mat[-1].sum() == 0:
            for i in range(t):
                if not trans_distribution[t - i - 1][-1].sum() == 0:
                    markov_mat[-1] = trans_distribution[t - i - 1][-1]
                    break

        # 调整合成数据库大小
        synthetic_db.adjust_data_size(markov_mat, len(traj_stream[t]), grid_map, end_distribution)

        trans_distribution.append(markov_mat)
        release.append(counts / counts.sum())

        N_st.append(np.sum(select))

        # 每处理100个时间戳打印一次进度
        if (t + 1) % 100 == 0:
            logger.info(f'{t + 1} timestamps processed')

    return synthetic_db


def lbd(traj_stream, w: int, eps: float, trans_domain: List[Transition]):
    trans_domain_map = utils.list_to_dict(trans_domain)
    release = []
    used_budget = []

    synthetic_db = SynDB()
    trans_distribution = []

    # randomly initialize synthetic database
    synthetic_db.random_initialize(len(traj_stream[0]), grid_map)
    # first timestamp
    eps_rm = eps / 2
    oue = OUE(eps_rm / 2, len(trans_domain), lambda x: trans_domain_map[x])

    for (g1, g2, flag) in traj_stream[1]:
        trans = Transition(g1, g2, flag)
        oue.privatise(trans)

    oue.adjust()
    est_counts = oue.non_negative_data / oue.n
    release.append(est_counts / est_counts.sum())

    # generate Markov matrix
    markov_mat, end_distribution = generate_markov_matrix(est_counts,
                                                          trans_domain)
    trans_distribution.append(markov_mat)

    # generate new points in synthetic data based on current distribution
    synthetic_db.generate_new_points_baseline(markov_mat, grid_map)

    used_budget.append(eps_rm / 2)
    used_budget_in_curr_w = eps_rm / 2

    for t in range(2, len(traj_stream)):
        if not len(traj_stream[t]):
            continue
        # set dissimilarity budget
        eps_1 = eps / (2 * w)
        # estimate c_t
        oue = OUE(eps_1, len(trans_domain), lambda x: trans_domain_map[x])

        for (g1, g2, flag) in traj_stream[t]:
            trans = Transition(g1, g2, flag)
            oue.privatise(trans)
        oue.adjust()
        c_bar = oue.non_negative_data / oue.n

        # calculate dissimilarity
        dis = np.mean((c_bar - release[-1]) ** 2)
        dis -= 4 * math.exp(eps_1) / (oue.n * (math.exp(eps_1) - 1) ** 2)

        if t >= w:
            # budget recovery
            used_budget_in_curr_w -= used_budget[t - w]
        eps_rm = eps / 2 - used_budget_in_curr_w

        err = 4 * math.exp(eps_rm / 2) / (oue.n * (math.exp(eps_rm / 2) - 1) ** 2)

        if dis > err:
            # perturbation
            oue = OUE(eps_rm/2, len(trans_domain), lambda x: trans_domain_map[x])

            for (g1, g2, flag) in traj_stream[t]:
                trans = Transition(g1, g2, flag)
                oue.privatise(trans)
            oue.adjust()
            est_counts = oue.non_negative_data / oue.n
            release.append(est_counts / est_counts.sum())
            used_budget.append(eps_rm / 2)
            used_budget_in_curr_w += eps_rm / 2
        else:
            # approximation
            release.append(release[-1])
            used_budget.append(0)

        markov_mat, end_distribution = generate_markov_matrix(release[-1], trans_domain)
        synthetic_db.generate_new_points_baseline(markov_mat, grid_map)

        trans_distribution.append(markov_mat)

        if (t + 1) % 100 == 0:
            logger.info(f'{t + 1} timestamps processed')
    return synthetic_db


def lba(traj_stream, w: int, eps: float, trans_domain: List[Transition]):
    trans_domain_map = utils.list_to_dict(trans_domain)
    release = []
    l: int = 0
    eps_l2 = 0

    synthetic_db = SynDB()
    trans_distribution = []

    # randomly initialize synthetic database
    synthetic_db.random_initialize(len(traj_stream[0]), grid_map)
    # first timestamp
    eps_2 = eps / (2 * w)
    oue = OUE(eps_2, len(trans_domain), lambda x: trans_domain_map[x])

    for (g1, g2, flag) in traj_stream[1]:
        trans = Transition(g1, g2, flag)
        oue.privatise(trans)
    oue.adjust()
    est_counts = oue.non_negative_data / oue.n
    release.append(est_counts/est_counts.sum())

    # generate Markov matrix
    markov_mat, end_distribution = generate_markov_matrix(est_counts,
                                                          trans_domain)
    trans_distribution.append(markov_mat)

    # generate new points in synthetic data based on current distribution
    synthetic_db.generate_new_points_baseline(markov_mat, grid_map)
    l = 1
    eps_l2 = eps_2

    for t in range(2, len(traj_stream)):
        if not len(traj_stream[t]):
            continue
        # set dissimilarity budget
        eps_1 = eps / (2 * w)
        # estimate c_t
        oue = OUE(eps_1, len(trans_domain), lambda x: trans_domain_map[x])

        for (g1, g2, flag) in traj_stream[t]:
            trans = Transition(g1, g2, flag)
            oue.privatise(trans)
        oue.adjust()
        c_bar = oue.non_negative_data / oue.n

        # calculate dissimilarity
        dis = np.mean((c_bar - release[-1]) ** 2)
        dis -= 4 * math.exp(eps_1) / (oue.n * (math.exp(eps_1) - 1) ** 2)

        # calculate nullified timestamps
        t_N = eps_l2 / (eps / (2 * w)) - 1

        if t - l <= t_N:
            # nullified timestamp
            release.append(release[-1])
        else:
            # calculate absorbed timestamps
            t_A = t - (l + t_N)
            eps_2 = eps / (2 * w) * min(t_A, w)
            err = 4 * math.exp(eps_2) / (oue.n * (math.exp(eps_2) - 1) ** 2)

            if dis > err:
                # perturbation
                oue = OUE(eps_2, len(trans_domain), lambda x: trans_domain_map[x])

                for (g1, g2, flag) in traj_stream[t]:
                    trans = Transition(g1, g2, flag)
                    oue.privatise(trans)
                oue.adjust()
                est_counts = oue.non_negative_data / oue.n
                release.append(est_counts/est_counts.sum())
                l = t
                eps_l2 = eps_2
            else:
                # approximation
                release.append(release[-1])

        markov_mat, end_distribution = generate_markov_matrix(release[-1], trans_domain)
        synthetic_db.generate_new_points_baseline(markov_mat, grid_map)

        trans_distribution.append(markov_mat)

        if (t + 1) % 100 == 0:
            logger.info(f'{t + 1} timestamps processed')

    return synthetic_db

avg_lens = {
    'tdrive': 13.61,
    'oldenburg': 59.98,
    'sanjoaquin': 55.3
}

timestamps = {
    'tdrive': 886,
    'oldenburg': 500,
    'sanjoaquin': 1000
}

# 打印日志,提示正在读取数据集
logger.info('Reading dataset...')
# 使用lzma解压并打开数据集文件,从中读取指定时间戳数量的数据
with lzma.open(f'../data/{args.dataset}_transition.xz', 'rb') as f:
    dataset = pickle.load(f)[:timestamps[args.dataset]]

# 计算数据集的统计信息(最小/最大x,y坐标),并保存到json文件中
stats = utils.t_dataset_stats(dataset, f'../data/{args.dataset}_stats.json')
grid_map = GridMap(args.grid_num,
                   stats['min_x'],
                   stats['min_y'],
                   stats['max_x'],
                   stats['max_y'])

# 打印日志,提示正在进行空间分解
logger.info('Spatial decomposition...')
if args.multiprocessing:
    # 定义多进程处理函数,用于并行处理空间分解
    def decomp_multi(xy_l):
        return spatial_decomposition(xy_l, grid_map)
    # 创建进程池
    pool = multiprocessing.Pool(CORES)
    # 使用进程池并行处理数据集
    grid_db = pool.map(decomp_multi, dataset)
    # 关闭进程池
    pool.close()
else:
    # 单进程顺序处理数据集
    grid_db = [spatial_decomposition(xy_l, grid_map) for xy_l in dataset]

# 处理非相邻网格之间的转移,将其拆分为结束和开始两个转移
# 例如: 如果两个网格G1和G2不相邻,则拆分为:
# - 在时间t: (G1, end, 2)表示在G1结束
# - 在时间t+1: (start, G2, 1)表示在G2开始
grid_db = split_traj(grid_db, grid_map)

# 根据选择的算法,生成合成数据
if args.method == 'retrasyn':
    logger.info('RetraSyn ...')
    syn_grid_db = RetraSyn(grid_db, args.w, args.epsilon,
                           grid_map.get_all_transition())
elif args.method == 'lbd':
    logger.info('LBD...')
    syn_grid_db = lbd(grid_db, args.w, args.epsilon, grid_map.get_all_transition())
elif args.method == 'lba':
    logger.info('LBA...')
    syn_grid_db = lba(grid_db, args.w, args.epsilon, grid_map.get_all_transition())
else:
    logger.info('Invalid method name!')
    exit()

# 将网格坐标形式的合成轨迹数据转换为原始坐标形式
syn_xy_db = convert_grid_to_raw(syn_grid_db.all_data)
# 将合成轨迹数据保存到文件
# 文件名格式: dataset_method_epsilon_gridnum_windowsize.pkl
# 例如: porto_retrasyn_1.0_g100_w10.pkl 表示:
# - 数据集: porto
# - 方法: retrasyn 
# - 隐私预算: 1.0
# - 网格数: 100
# - 时间窗口: 10
with open(
        f'../data/syn_data/{args.dataset}/{args.method}_{args.epsilon}_g{args.grid_num}_w{args.w}.pkl',
        'wb') as f:
    pickle.dump(syn_xy_db, f)
