import pickle
import lzma
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from parse import args
import utils
from grid import GridMap
import os

class TrajectoryVisualizer:
    def __init__(self, dataset_name, method, epsilon, grid_num, w):
        self.dataset_name = dataset_name
        self.method = method
        self.epsilon = epsilon
        self.grid_num = grid_num
        self.w = w
        
        # 加载原始数据集
        with lzma.open(f'../data/{dataset_name}_transition.xz', 'rb') as f:
            self.original_data = pickle.load(f)
            
        # 加载合成数据集
        syn_path = f'../data/syn_data/{dataset_name}/{method}_{epsilon}_g{grid_num}_w{w}.pkl'
        with open(syn_path, 'rb') as f:
            self.synthetic_data = pickle.load(f)
            
        # 加载统计信息
        stats = utils.t_dataset_stats(self.original_data, f'../data/{dataset_name}_stats.json')
        self.grid_map = GridMap(grid_num, stats['min_x'], stats['min_y'], 
                              stats['max_x'], stats['max_y'])
        
        # 设置绘图参数
        self.min_x = stats['min_x']
        self.max_x = stats['max_x']
        self.min_y = stats['min_y']
        self.max_y = stats['max_y']

    def show_trajectories(self):
        """显示所有轨迹的静态图"""
        # 创建图形和子图
        fig = plt.figure(figsize=(15, 7))
        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])  # 原始轨迹
        ax2 = fig.add_subplot(gs[0, 1])  # 合成轨迹
        
        # 设置坐标轴范围和网格
        for ax in [ax1, ax2]:
            ax.set_xlim(self.min_x, self.max_x)
            ax.set_ylim(self.min_y, self.max_y)
            
            # 绘制网格
            x_step = (self.max_x - self.min_x) / self.grid_num
            y_step = (self.max_y - self.min_y) / self.grid_num
            for i in range(self.grid_num + 1):
                ax.axvline(x=self.min_x + i * x_step, color='gray', linestyle='--', alpha=0.2)
                ax.axhline(y=self.min_y + i * y_step, color='gray', linestyle='--', alpha=0.2)
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        
        # 绘制原始轨迹
        start_points = []
        end_points = []
        for timestamp in self.original_data:
            for x0, y0, x1, y1, flag in timestamp:
                ax1.plot([x0, x1], [y0, y1], 'b-', alpha=0.1)
                if flag == 1:  # 起点
                    start_points.append((x1, y1))
                elif flag == 2:  # 终点
                    end_points.append((x0, y0))
        
        # 绘制起点和终点
        if start_points:
            x_starts, y_starts = zip(*start_points)
            ax1.scatter(x_starts, y_starts, c='g', s=10, alpha=0.5, label='Start Points')
        if end_points:
            x_ends, y_ends = zip(*end_points)
            ax1.scatter(x_ends, y_ends, c='r', s=10, alpha=0.5, label='End Points')
        
        # 绘制合成轨迹的所有点
        all_syn_points = []
        for traj in self.synthetic_data:
            points = [(x, y) for x, y, _ in traj]
            all_syn_points.extend(points)
            # 连接轨迹点
            if len(points) > 1:
                x_coords, y_coords = zip(*points)
                ax2.plot(x_coords, y_coords, 'r-', alpha=0.1)
        
        if all_syn_points:
            x_coords, y_coords = zip(*all_syn_points)
            ax2.scatter(x_coords, y_coords, c='r', s=1, alpha=0.1, label='Synthetic Points')
        
        ax1.set_title('Original Trajectories')
        ax2.set_title('Synthetic Trajectories')
        
        ax1.legend()
        ax2.legend()
        
        plt.suptitle(f'Trajectory Visualization\n{self.dataset_name.upper()} - Method: {self.method}, ε={self.epsilon}, Grid: {self.grid_num}x{self.grid_num}')
        plt.tight_layout()
        plt.show()

def main():
    try:
        # 创建可视化对象
        visualizer = TrajectoryVisualizer(
            dataset_name=args.dataset,
            method=args.method,
            epsilon=args.epsilon,
            grid_num=args.grid_num,
            w=args.w
        )
        
        # 显示所有轨迹
        visualizer.show_trajectories()
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise

if __name__ == '__main__':
    main() 