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
from syndb import SynDB, Users
from logger.logger import ConfigParser
import lzma

config = ConfigParser(name='RetraSyn', save_dir='./')
logger = config.get_logger(config.exper_name)

CORES = multiprocessing.cpu_count() // 2
random.seed(2023)
np.random.seed(2023)

logger.info(args)


def spatial_decomposition(xy_l: List[Tuple[float, float, float, float, int, int]], gm: GridMap):
    """
    空间分解函数，将轨迹点坐标转换为网格坐标
    Args:
        xy_l: 同一时间戳的轨迹点列表，每个元素为(x0,y0,x1,y1,flag,uid)的元组
            x0,y0: 起始点坐标
            x1,y1: 终止点坐标 
            flag: 标志位(0:普通轨迹点, 1:轨迹起点, 2:轨迹终点)
            uid: 用户ID
        gm: 网格地图对象
    Returns:
        grid_list: 转换后的网格坐标列表
    """
    grid_list = []
    for (x0, y0, x1, y1, flag, uid) in xy_l:
        if flag == 0:  # 普通轨迹点,需要起始和终止网格
            g0, g1 = utils.xy2grid([(x0, y0), (x1, y1)], gm)
            grid_list.append((g0, g1, flag, uid))
        elif flag == 1:  # 轨迹起点,只需要终止网格
            g1 = utils.xy2grid([(x1, y1)], gm)[0]
            grid_list.append((g1, g1, flag, uid))
        else:  # 轨迹终点,只需要起始网格
            g0 = utils.xy2grid([(x0, y0)], gm)[0]
            grid_list.append((g0, g0, flag, uid))
    return grid_list


def split_traj(traj_stream: List[List[Tuple[Grid, Grid, int, int]]], gm: GridMap):
    """
    处理不相邻网格之间的转移
    如果两个网格G1和G2不相邻,则将转移(G1, G2, flag)拆分为:
    在时刻t: (G1, end, 2) - 表示在G1结束
    在时刻t+1: (start, G2, 1) - 表示在G2重新开始

    参数:
        traj_stream: 轨迹流,每个时刻包含多个轨迹点(g1,g2,flag,uid)
            g1: 起始网格
            g2: 终止网格
            flag: 标志位(0:普通转移, 1:轨迹起点, 2:轨迹终点)
            uid: 用户ID
        gm: 网格地图对象
    返回:
        new_stream: 处理后的轨迹流
    """
    new_stream = []
    while len(new_stream) <= len(traj_stream):
        new_stream.append([])
    for t in range(len(traj_stream)):
        for g1, g2, flag, uid in traj_stream[t]:
            # 如果是轨迹起点或终点,直接保留
            if flag:
                new_stream[t].append((g1, g2, flag, uid))
                continue
            # 如果两个网格不相等且不相邻,进行拆分
            if not g1.equal(g2) and not gm.is_adjacent_grids(g1, g2):
                new_stream[t].append((g1, g1, 2, uid))  # 在g1结束
                new_stream[t + 1].append((g2, g2, 1, uid))  # 在g2重新开始
            else:
                new_stream[t].append((g1, g2, flag, uid))
    return new_stream


def generate_markov_matrix(markov_vec: np.ndarray, trans_domain: List[Transition]):
    """生成马尔可夫转移矩阵和轨迹结束分布

    参数:
        markov_vec: 转移概率向量
        trans_domain: 转移域列表
    返回:
        markov_mat: 马尔可夫转移矩阵
        end_distribution: 轨迹结束分布
    """
    n = grid_map.size + 1  # 网格数量+1,包含虚拟起点和终点
    markov_mat = np.zeros((n, n), dtype=float)  # 初始化马尔可夫矩阵
    end_distribution = np.zeros(n - 1)  # 初始化结束分布
    for k in range(len(markov_vec)):
        if markov_vec[k] <= 0:
            continue

        # 在矩阵中找到对应的索引位置
        trans = trans_domain[k]
        if not trans.flag:  # 普通转移
            i = utils.grid_index_map_func(trans.g1, grid_map)
            j = utils.grid_index_map_func(trans.g2, grid_map)
        elif trans.flag == 1:  # 轨迹起点转移,位于矩阵最后一行
            i = -1
            j = utils.grid_index_map_func(trans.g2, grid_map)
        else:  # 轨迹终点转移,位于矩阵最后一列
            i = utils.grid_index_map_func(trans.g1, grid_map)
            j = -1
            end_distribution[i] = markov_vec[k]
        markov_mat[i][j] = markov_vec[k]

    # 按行归一化概率
    markov_mat = markov_mat / (markov_mat.sum(axis=1).reshape((-1, 1)) + 1e-8)
    end_distribution = end_distribution / (end_distribution.sum() + 1e-8)
    return markov_mat, end_distribution


def convert_grid_to_raw(grid_db: List[List[Tuple[Grid, int]]]):
    """将网格化的轨迹数据转换回原始坐标形式

    参数:
        grid_db: 网格化后的轨迹数据库,每条轨迹由(网格,时间戳)元组组成
    返回:
        raw_db: 转换后的原始坐标轨迹数据库
    """
    def traj_grid_to_raw(traj: List[Tuple[Grid, int]]):
        """将单条网格轨迹转换为原始坐标轨迹"""
        xy_traj = []
        for (g, t) in traj:
            # 在网格内随机采样一个点作为轨迹点坐标
            x, y = g.sample_point()
            xy_traj.append((x, y, t))
        return xy_traj

    # 对数据库中每条轨迹进行转换
    raw_db = [traj_grid_to_raw(traj) for traj in grid_db]

    return raw_db


def RetraSyn(traj_stream, w: int, eps, trans_domain: List[Transition]):
    """RetraSyn算法主函数

    参数:
        traj_stream: 轨迹流数据
        w: 时间窗口大小
        eps: 隐私预算
        trans_domain: 转移域列表
    返回:
        synthetic_db: 生成的合成数据库
    """
    # 将转移域列表转换为字典,方便查找
    trans_domain_map = utils.list_to_dict(trans_domain)

    # 初始化合成数据库和相关变量
    synthetic_db = SynDB()  # 合成数据库
    trans_distribution = []  # 存储每个时间戳的转移分布
    used_budget = []  # 存储每个时间戳使用的预算(采样的用户)
    release = []  # 存储每个时间戳的发布结果
    N_sp = []  # 存储每个时间戳的显著模式数量
    users = Users()  # 用户管理器
    quitted_users = []  # 存储退出的用户

    # 预热阶段(前两个时间戳)
    for t in range(2):
        # 初始化OUE机制
        oue = OUE(eps, len(trans_domain), lambda x: trans_domain_map[x])

        # 移除退出的用户
        for uid in quitted_users:
            users.remove(uid)
        quitted_users = []

        # 添加新用户并记录退出的用户
        for (_, _, flag, uid) in traj_stream[t]:
            users.register(uid)
            if flag == 2:  # flag=2表示轨迹终点,用户退出
                quitted_users.append(uid)

        # 按1/w的比例采样用户
        sampled_users = users.sample(1 / w)

        # 更新转移分布
        for (g1, g2, flag, uid) in traj_stream[t]:
            if not users.users[uid] == 2:  # 用户未被采样
                continue
            trans = Transition(g1, g2, flag)
            oue.privatise(trans)
        oue.adjust()

        # 计算估计的计数
        est_counts = oue.non_negative_data / oue.n

        # 生成马尔可夫转移矩阵
        markov_mat, end_distribution = generate_markov_matrix(est_counts,
                                                              trans_domain)
        trans_distribution.append(markov_mat)
        release.append(est_counts / est_counts.sum())

        # 根据当前分布生成新的轨迹点
        synthetic_db.generate_new_points(
            markov_mat, grid_map, avg_lens[args.dataset])

        # 调整合成数据库大小
        synthetic_db.adjust_data_size(markov_mat, len(
            traj_stream[t]), grid_map, end_distribution)

        # 更新预算使用情况
        used_budget.append(sampled_users)
        for uid in sampled_users:
            users.deactivate(uid)
        N_sp.append(0)

    # 主循环阶段
    for t in range(2, len(traj_stream)):
        if not len(traj_stream[t]):
            continue

        # 用户回收(超过时间窗口w的用户可以重新使用)
        if t >= w:
            for uid in used_budget[t - w]:
                users.recycle(uid)

        # 移除退出的用户
        for uid in quitted_users:
            users.remove(uid)
        quitted_users = []

        # 添加新用户并记录退出的用户
        for (g1, g2, flag, uid) in traj_stream[t]:
            users.register(uid)
            if flag == 2:
                quitted_users.append(uid)

        # 计算当前分布与历史平均分布的偏差
        dev = np.abs(
            np.array(release[-1]) - np.average(release[max(0, t - 5):t], axis=0)).sum()

        # 计算采样概率
        cr = max(
            0.5, 1 - np.average(N_sp[max(0, t - 5):t]) / len(trans_domain))
        p = utils.allocation_p(dev, w, alpha=8)
        p = min(p * cr, 0.6)
        sampled_users = users.sample(p)

        # 初始化OUE机制
        oue = OUE(eps, len(trans_domain), lambda x: trans_domain_map[x])

        # 更新转移分布
        for (g1, g2, flag, uid) in traj_stream[t]:
            if not users.users[uid] == 2:
                continue
            trans = Transition(g1, g2, flag)
            oue.privatise(trans)
        oue.adjust()

        # 如果没有采样到用户,使用上一时刻的分布
        if oue.n == 0:
            used_budget.append([])
            N_sp.append(0)
            counts = release[-1]
        else:
            # 计算当前估计分布和历史分布
            f_hat = oue.non_negative_data / oue.n
            f_tilde = release[-1]

            # 选择显著模式
            variance = 4 * math.exp(eps) / (oue.n * (math.exp(eps) - 1) ** 2)
            select = (f_tilde - f_hat) ** 2 > variance

            # 合并显著模式和非显著模式
            counts = np.zeros(len(trans_domain))
            sig_counts = oue.non_negative_data / oue.n

            for i in range(len(select)):
                if select[i]:  # 显著模式使用当前估计
                    counts[i] = sig_counts[i]
                else:  # 非显著模式使用历史分布
                    counts[i] = f_tilde[i] * sig_counts.sum()

            # 更新预算使用情况
            for uid in sampled_users:
                users.deactivate(uid)
            used_budget.append(sampled_users)
            N_sp.append(np.sum(select))

        # 生成马尔可夫转移矩阵
        markov_mat, end_distribution = generate_markov_matrix(
            counts, trans_domain)

        # 根据当前分布生成新的轨迹点
        synthetic_db.generate_new_points(
            markov_mat, grid_map, avg_lens[args.dataset])

        # 检查进入分布是否为0,如果为0则使用历史非0分布
        if markov_mat[-1].sum() == 0:
            for i in range(t):
                if not trans_distribution[t - i - 1][-1].sum() == 0:
                    markov_mat[-1] = trans_distribution[t - i - 1][-1]
                    break

        # 调整合成数据库大小
        synthetic_db.adjust_data_size(markov_mat, len(
            traj_stream[t]), grid_map, end_distribution)

        # 更新转移分布和发布结果
        trans_distribution.append(markov_mat)
        release.append(counts / counts.sum())

        # 每处理100个时间戳输出一次日志
        if (t + 1) % 100 == 0:
            logger.info(f'{t + 1} timestamps processed')

    return synthetic_db


def lpd(traj_stream, w: int, eps: float, trans_domain: List[Transition]):
    trans_domain_map = utils.list_to_dict(trans_domain)
    release = []
    used_budget_1 = []
    used_budget_2 = []

    synthetic_db = SynDB()
    trans_distribution = []
    users = Users()

    # randomly initialize synthetic database
    synthetic_db.random_initialize(len(traj_stream[0]), grid_map)
    # first timestamp
    # add new users
    for (g1, g2, flag, uid) in traj_stream[1]:
        users.register(uid)

    sampled_users = users.sample(1/4)
    oue = OUE(eps, len(trans_domain), lambda x: trans_domain_map[x])

    # update transition distribution
    for (g1, g2, flag, uid) in traj_stream[1]:
        if not users.users[uid] == 2:
            # user not sampled
            continue
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

    used_budget_2.append(sampled_users)
    for uid in sampled_users:
        users.deactivate(uid)
    used_budget_1.append([])
    quitted_users = []

    for t in range(2, len(traj_stream)-1):
        if not len(traj_stream[t]):
            continue
        # user recycling

        if t >= w:
            for uid in used_budget_1[t - w]:
                users.recycle(uid)
            for uid in used_budget_2[t-w]:
                users.recycle(uid)

        # remove quited users
        for uid in quitted_users:
            users.remove(uid)
        quitted_users = []

        for (g1, g2, flag, uid) in traj_stream[t]:
            users.register(uid)
            if flag == 2:
                quitted_users.append(uid)

        # set dissimilarity budget
        sampled_users = users.sample(1/(2*w))

        if len(sampled_users) == 0:
            release.append(release[-1])
            used_budget_2.append([])
            markov_mat, end_distribution = generate_markov_matrix(
                release[-1], trans_domain)
            synthetic_db.generate_new_points_baseline(markov_mat, grid_map)

            trans_distribution.append(markov_mat)
            continue

        # estimate c_t
        oue = OUE(eps, len(trans_domain), lambda x: trans_domain_map[x])

        for (g1, g2, flag, uid) in traj_stream[t]:
            if not users.users[uid] == 2:
                continue
            trans = Transition(g1, g2, flag)
            oue.privatise(trans)
        oue.adjust()

        for uid in sampled_users:
            users.deactivate(uid)
        used_budget_1.append(sampled_users)

        c_bar = oue.non_negative_data / oue.n

        # calculate dissimilarity
        dis = np.mean((c_bar - release[-1]) ** 2)
        dis -= 4 * math.exp(eps) / (oue.n * (math.exp(eps) - 1) ** 2)

        users_rm = len(users.available_users)//2
        err = 4 * math.exp(eps) / (int(1/2*users_rm)
                                   * (math.exp(eps) - 1) ** 2)

        if dis > err and users_rm > 0:
            sampled_users = users.sample(users_rm/2/len(users.available_users))
            # perturbation
            oue = OUE(eps, len(trans_domain), lambda x: trans_domain_map[x])
            used_users = []
            for (g1, g2, flag, uid) in traj_stream[t]:
                if not users.users[uid] == 2:
                    continue
                used_users.append(uid)
                trans = Transition(g1, g2, flag)
                oue.privatise(trans)
            oue.adjust()
            est_counts = oue.non_negative_data / oue.n
            release.append(est_counts / est_counts.sum())
            used_budget_2.append(sampled_users)
            for uid in sampled_users:
                users.deactivate(uid)
        else:
            # approximation
            release.append(release[-1])
            used_budget_2.append([])

        markov_mat, end_distribution = generate_markov_matrix(
            release[-1], trans_domain)
        synthetic_db.generate_new_points_baseline(markov_mat, grid_map)

        trans_distribution.append(markov_mat)

        if (t + 1) % 100 == 0:
            logger.info(f'{t + 1} timestamps processed')
    return synthetic_db

def lpa(traj_stream, w: int, eps: float, trans_domain: List[Transition]):
    trans_domain_map = utils.list_to_dict(trans_domain)
    l: int = 0
    eps_l2 = 0

    release = []
    used_budget_1 = []
    used_budget_2 = []

    synthetic_db = SynDB()
    trans_distribution = []
    users = Users()

    # randomly initialize synthetic database
    synthetic_db.random_initialize(len(traj_stream[0]), grid_map)
    # first timestamp
    # add new users
    for (g1, g2, flag, uid) in traj_stream[1]:
        users.register(uid)
    # first timestamp
    sampled_users = users.sample(1/(2*w))
    oue = OUE(eps, len(trans_domain), lambda x: trans_domain_map[x])

    for (g1, g2, flag, uid) in traj_stream[1]:
        if not users.users[uid] == 2:
            # user not sampled
            continue
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
    used_budget_2.append(sampled_users)
    for uid in sampled_users:
        users.deactivate(uid)
    used_budget_1.append([])
    quited_users = []

    l = 1
    eps_l2 = int(len(traj_stream[1])/(w*2))

    for t in range(2, len(traj_stream)-1):
        if not len(traj_stream[t]):
            continue
        # user recycling
        if t >= w:
            for uid in used_budget_1[t - w]:
                users.recycle(uid)
            for uid in used_budget_2[t - w]:
                users.recycle(uid)

        # remove quited users
        for uid in quited_users:
            users.remove(uid)
        quited_users = []

        for (g1, g2, flag, uid) in traj_stream[t]:
            users.register(uid)
            if flag == 2:
                quited_users.append(uid)

        # set dissimilarity budget
        sampled_users = users.sample(1 / (2 * w))

        if len(sampled_users) == 0:
            release.append(release[-1])
            used_budget_2.append([])
            markov_mat, end_distribution = generate_markov_matrix(
                release[-1], trans_domain)
            synthetic_db.generate_new_points_baseline(markov_mat, grid_map)

            trans_distribution.append(markov_mat)
            continue

        # estimate c_t
        oue = OUE(eps, len(trans_domain), lambda x: trans_domain_map[x])

        for (g1, g2, flag, uid) in traj_stream[t]:
            if not users.users[uid] == 2:
                continue
            trans = Transition(g1, g2, flag)
            oue.privatise(trans)
        oue.adjust()

        for uid in sampled_users:
            users.deactivate(uid)
        used_budget_1.append(sampled_users)
        c_bar = oue.non_negative_data / oue.n

        # calculate dissimilarity

        dis = np.mean((c_bar - release[-1]) ** 2)
        dis -= 4 * math.exp(eps) / (oue.n * (math.exp(eps) - 1) ** 2)

        # calculate nullified timestamps
        t_N = eps_l2 / (len(traj_stream[t]) / (2 * w)) - 1

        if t - l <= t_N:
            # nullified timestamp
            release.append(release[-1])
            used_budget_2.append([])
        else:
            # calculate absorbed timestamps
            t_A = t - (l + t_N)
            N_pp = int(len(traj_stream[t])/(w*2)) * min(t_A, w)
            err = 4 * math.exp(eps) / (N_pp * (math.exp(eps) - 1) ** 2)

            if dis > err:
                sampled_users = users.sample(N_pp/len(users.available_users))
                # perturbation
                oue = OUE(eps, len(trans_domain),
                          lambda x: trans_domain_map[x])

                for (g1, g2, flag, uid) in traj_stream[t]:
                    if not users.users[uid] == 2:
                        continue
                    trans = Transition(g1, g2, flag)
                    oue.privatise(trans)
                oue.adjust()
                est_counts = oue.non_negative_data / oue.n
                release.append(est_counts/est_counts.sum())
                used_budget_2.append(sampled_users)
                for uid in sampled_users:
                    users.deactivate(uid)
                l = t
                eps_l2 = N_pp
            else:
                # approximation
                release.append(release[-1])
                used_budget_2.append([])

        markov_mat, end_distribution = generate_markov_matrix(
            release[-1], trans_domain)
        synthetic_db.generate_new_points_baseline(markov_mat, grid_map)

        trans_distribution.append(markov_mat)

        if (t + 1) % 100 == 0:
            logger.info(f'{t + 1} timestamps processed')

    return synthetic_db


avg_lens = {
    'tdrive': 13.61,
    'porto_center': 12.00,
    'oldenburg': 59.98,
    'sanjoaquin': 55.3
}

timestamps = {
    'tdrive': 886,
    'oldenburg': 500,
    'sanjoaquin': 1000}

# 打印日志,提示正在读取数据集
logger.info('Reading dataset...')
# 使用lzma解压并打开数据集文件,从中读取指定时间戳数量的数据
with lzma.open(f'../data/{args.dataset}_transition_id.xz', 'rb') as f:
    dataset = pickle.load(f)[:timestamps[args.dataset]]


# 计算数据集的统计信息,并保存到json文件中
stats = utils.tid_dataset_stats(dataset, f'../data/{args.dataset}_stats.json')

# 创建网格地图对象,用于将坐标转换为网格索引
grid_map = GridMap(args.grid_num,
                   stats['min_x'],
                   stats['min_y'],
                   stats['max_x'],
                   stats['max_y'])

# 打印日志,提示正在进行空间分解
logger.info('Spatial decomposition...')
# 根据是否使用多进程来进行空间分解
if args.multiprocessing:
    # 定义多进程处理函数
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

grid_db = split_traj(grid_db, grid_map)


if args.method == 'retrasyn':
    logger.info('RetraSyn...')
    syn_grid_db = RetraSyn(grid_db, args.w, args.epsilon,
                           grid_map.get_all_transition())
elif args.method == 'lpd':
    logger.info('LPD...')
    syn_grid_db = lpd(grid_db, args.w, args.epsilon,
                      grid_map.get_all_transition())
elif args.method == 'lpa':
    logger.info('LPA...')
    syn_grid_db = lpa(grid_db, args.w, args.epsilon,
                      grid_map.get_all_transition())
else:
    logger.info('Invalid method name!')
    exit()

syn_xy_db = convert_grid_to_raw(syn_grid_db.all_data)
with open(
        f'../data/syn_data/{args.dataset}/{args.method}_{args.epsilon}_g{args.grid_num}_w{args.w}_p.pkl',
        'wb') as f:
    pickle.dump(syn_xy_db, f)
