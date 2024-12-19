import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os
from datetime import datetime
import seaborn as sns
from matplotlib.patches import Rectangle

class TrainingLogger:
    def __init__(self, args):
        self.args = args
        self.rewards = []
        self.path_lengths = []
        self.avg_rewards = []
        self.avg_lengths = []
        self.window_size = 20
        
       
        plt.style.use('seaborn-v0_8')
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rcParams['figure.figsize'] = (12, 8)
        mpl.rcParams['figure.dpi'] = 300
        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.titlesize'] = 14
        mpl.rcParams['axes.labelsize'] = 12
        
        # 创建日志目录
        self.log_dir = f"logs/{args.algorithm}_{args.env_type}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置Seaborn样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")

    def log_episode(self, reward, path_length):
        """记录每个回合的训练数据"""
        self.rewards.append(reward)
        self.path_lengths.append(path_length)
        
        # 计算滑动平均
        if len(self.rewards) >= self.window_size:
            avg_reward = np.mean(self.rewards[-self.window_size:])
            avg_length = np.mean(self.path_lengths[-self.window_size:])
        else:
            avg_reward = np.mean(self.rewards)
            avg_length = np.mean(self.path_lengths)
            
        self.avg_rewards.append(avg_reward)
        self.avg_lengths.append(avg_length)
        
        # 打印训练信息
        current_episode = len(self.rewards)
        print(f"\n第 {current_episode} 回合训练信息:")
        print(f"├── 当前奖励: {reward:.2f}")
        print(f"├── 平均奖励: {avg_reward:.2f}")
        print(f"├── 路径长度: {path_length}")
        print(f"└── 平均长度: {avg_length:.2f}")

    def save_data(self):
  
        data = {
            'rewards': self.rewards,
            'path_lengths': self.path_lengths,
            'avg_rewards': self.avg_rewards,
            'avg_lengths': self.avg_lengths,
            'training_config': vars(self.args),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        filename = os.path.join(self.log_dir, 'training_data.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
 

    def plot_curves(self):
    
        episodes = range(1, len(self.rewards) + 1)
        
        # 创建图表
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        fig.suptitle(f'{self.args.algorithm.upper()} in {self.args.env_type} training',
                    fontsize=16, y=0.95)
        
        # 绘制奖励曲线
        ax1.plot(episodes, self.rewards, alpha=0.3, label='reward',
                color='#2ecc71', linewidth=1)
        ax1.plot(episodes, self.avg_rewards, label='avg reward',
                color='#e74c3c', linewidth=2)
        ax1.set_title('REWARD', pad=20)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('REWARD')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # 添加训练信息注释
        info_text = (f"ps:\n"
                     f"environment: {self.args.env_type}\n"
                     f"epoch: {len(self.rewards)}\n"
                     f"avg reward: {self.avg_rewards[-1]:.2f}")
        ax1.text(1.15, 0.5, info_text, transform=ax1.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # 绘制路径长度曲线
        ax2.plot(episodes, self.path_lengths, alpha=0.3, label='path length',
                color='#3498db', linewidth=1)
        ax2.plot(episodes, self.avg_lengths, label='avg path length',
                color='#9b59b6', linewidth=2)
        ax2.set_title('path length', pad=20)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('length')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # 添加路径统计信息
        length_info = (f"ps:\n"
                       f"avg path length: {np.mean(self.path_lengths):.2f}\n"
                       f"shortest path: {min(self.path_lengths):.2f}\n"
                       f"longest path: {max(self.path_lengths):.2f}")
        ax2.text(1.15, 0.5, length_info, transform=ax2.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.log_dir, 'training_curves.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"训练曲线已保存至: {self.log_dir}/training_curves.png")

    def plot_comparison(self):
   
        plt.clf()
        window_sizes = [5, 20, 50]
        episodes = range(1, len(self.rewards) + 1)
        
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        fig.suptitle(f'{self.args.algorithm.upper()} results ',
                    fontsize=16, y=0.95)
        
    
        colors = ['#1abc9c', '#3498db', '#9b59b6', '#f1c40f']
        

        ax1.plot(episodes, self.rewards, color='#95a5a6', alpha=0.3, 
                 label='data', linewidth=1)
        for idx, window in enumerate(window_sizes):
            smoothed = np.array([np.mean(self.rewards[max(0, i-window):i+1]) 
                                 for i in range(len(self.rewards))])
            ax1.plot(episodes, smoothed, color=colors[idx], 
                     label=f'{window}epoches average', linewidth=2)
        
        ax1.set_title('different window size reward curve', pad=20)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('reward')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # 绘制路径长度对比曲线
        ax2.plot(episodes, self.path_lengths, color='#95a5a6', alpha=0.3,
                 label='data', linewidth=1)
        for idx, window in enumerate(window_sizes):
            smoothed = np.array([np.mean(self.path_lengths[max(0, i-window):i+1])
                                 for i in range(len(self.path_lengths))])
            ax2.plot(episodes, smoothed, color=colors[idx],
                     label=f'{window}epoches average', linewidth=2)
        
        ax2.set_title('different window size path length curve', pad=20)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('length')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # 添加统计信息
        stats_info = (f"ps:\n"
                      f"epoch: {len(self.rewards)}\n"
                      f"final reward: {self.rewards[-1]:.2f}\n"
                      f"higest reward: {max(self.rewards):.2f}\n"
                      f"avg reward: {np.mean(self.rewards):.2f}")
        ax1.text(1.15, 0.5, stats_info, transform=ax1.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.log_dir, 'comparison_curves.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"对比曲线已保存至: {self.log_dir}/comparison_curves.png")

    def save_all(self):
        """保存所有数据和可视化结果"""
        self.save_data()
        self.plot_curves()
        self.plot_comparison()
        print("\n所有结果已保存完成！")





def show_obs(ax, x1, y1, z1, x2, y2, z2):
    xx, yy = np.meshgrid([x1, x2], [y1, y2])

    ax.plot_surface(xx, yy, np.full_like(xx, z1), color='gray', alpha=0.3)
    ax.plot_surface(xx, yy, np.full_like(xx, z2), color='gray', alpha=0.3)

    yy, zz = np.meshgrid([y1, y2], [z1, z2])
    ax.plot_surface(np.full_like(yy, x1), yy, zz, color='gray', alpha=0.3)
    ax.plot_surface(np.full_like(yy, x2), yy, zz, color='gray', alpha=0.3)

    xx, zz = np.meshgrid([x1, x2], [z1, z2])
    ax.plot_surface(xx, np.full_like(xx, y1), zz, color='gray', alpha=0.3)
    ax.plot_surface(xx, np.full_like(xx, y2), zz, color='gray', alpha=0.3)


def show_obs2(ax, coords, plane='xy'):
    (x1, y1, z1), (x2, y2, z2) = coords

    if plane == 'xy':
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='gray', alpha=0.3)
    elif plane == 'xz':
        rect = Rectangle((x1, z1), x2 - x1, z2 - z1, facecolor='gray', alpha=0.3)
    elif plane == 'yz':
        rect = Rectangle((y1, z1), y2 - y1, z2 - z1, facecolor='gray', alpha=0.3)
    ax.add_patch(rect)


def save(path_points, obstacles, env, args, episode_idx):
    save_dir = f"test_results/{args.algorithm}_{args.env_type}"
    os.makedirs(save_dir, exist_ok=True)

    path_points = np.array(path_points)

    fig = plt.figure(figsize=(20, 15))

    ax1 = fig.add_subplot(221, projection='3d')

    for (p1, p2) in obstacles:
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        show_obs(ax1, x1, y1, z1, x2, y2, z2)

    ax1.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], 'b-', label='路径', linewidth=2)
    ax1.scatter(path_points[0, 0], path_points[0, 1], path_points[0, 2],
                color='g', s=100, label='起点')
    ax1.scatter(path_points[-1, 0], path_points[-1, 1], path_points[-1, 2],
                color='r', s=100, label='终点')
    ax1.set_xlabel('X轴')
    ax1.set_ylabel('Y轴')
    ax1.set_zlabel('Z轴')
    ax1.set_title('3D路径图')
    ax1.set_xlim(0, env.grid_size)
    ax1.set_ylim(0, env.grid_size)
    ax1.set_zlim(0, env.grid_size)
    ax1.legend()

    ax2 = fig.add_subplot(222)

    for coords in obstacles:
        show_obs2(ax2, coords, plane='xy')
    ax2.plot(path_points[:, 0], path_points[:, 1], 'b-', linewidth=2)
    ax2.scatter(path_points[0, 0], path_points[0, 1], color='g', s=100, label='起点')
    ax2.scatter(path_points[-1, 0], path_points[-1, 1], color='r', s=100, label='终点')
    ax2.set_xlabel('X轴')
    ax2.set_ylabel('Y轴')
    ax2.set_title('俯视图 (X-Y平面)')
    ax2.grid(True)
    ax2.set_xlim(0, env.grid_size)
    ax2.set_ylim(0, env.grid_size)
    ax2.legend()

    ax3 = fig.add_subplot(223)

    for coords in obstacles:
        show_obs2(ax3, coords, plane='xz')
    ax3.plot(path_points[:, 0], path_points[:, 2], 'b-', linewidth=2)
    ax3.scatter(path_points[0, 0], path_points[0, 2], color='g', s=100, label='起点')
    ax3.scatter(path_points[-1, 0], path_points[-1, 2], color='r', s=100, label='终点')
    ax3.set_xlabel('X轴')
    ax3.set_ylabel('Z轴')
    ax3.set_title('主视图 (X-Z平面)')
    ax3.grid(True)
    ax3.set_xlim(0, env.grid_size)
    ax3.set_ylim(0, env.grid_size)
    ax3.legend()

    ax4 = fig.add_subplot(224)

    for coords in obstacles:
        show_obs2(ax4, coords, plane='yz')
    ax4.plot(path_points[:, 1], path_points[:, 2], 'b-', linewidth=2)
    ax4.scatter(path_points[0, 1], path_points[0, 2], color='g', s=100, label='起点')
    ax4.scatter(path_points[-1, 1], path_points[-1, 2], color='r', s=100, label='终点')
    ax4.set_xlabel('Y轴')
    ax4.set_ylabel('Z轴')
    ax4.set_title('侧视图 (Y-Z平面)')
    ax4.grid(True)
    ax4.set_xlim(0, env.grid_size)
    ax4.set_ylim(0, env.grid_size)
    ax4.legend()

    plt.tight_layout()

    save_path = f"{save_dir}/result_{episode_idx + 1}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_dir