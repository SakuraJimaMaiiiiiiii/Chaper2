import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.optim as optim
import numpy as np
from nn_net.net import BCNet
import copy
from tqdm import tqdm
from env.environment import Environment
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import random


class BehaviorCloning:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = BCNet(state_dim, action_dim, max_action).to(self.device)
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
    def select_action(self, state):
        self.net.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            mean, std = self.net(state)
            # 在测试时只使用均值
            action = mean.cpu().data.numpy().flatten()
        self.net.train()
        return action
            
    def train(self, states, actions):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        
        # 数据增强
        noise_scale = 0.1
        states = states + torch.randn_like(states) * noise_scale
        
        # 确保网络在训练模式
        self.net.train()
        
        # 获取预测的动作分布
        mean, std = self.net(states)
        
        # 分别计算各个损失组件
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions)
        nll_loss = -log_prob.mean()
        kl_div = -0.5 * (1 + 2*torch.log(std) - mean.pow(2) - std.pow(2))
        kl_loss = 0.01 * kl_div.mean()
        
        total_loss = nll_loss + kl_loss
        
        # 监控各个指标
        with torch.no_grad():
            mean_std = std.mean().item()
            mean_diff = (mean - actions).abs().mean().item()
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'nll_loss': nll_loss.item(),
            'kl_loss': kl_loss.item(),
            'mean_std': mean_std,
            'mean_diff': mean_diff
        }
        
    def save(self, filename):
        torch.save(self.net.state_dict(), filename)
        
    def load(self, filename):
        self.net.load_state_dict(torch.load(filename))

def train_BC(args, expert_data):
    print("\n开始BC训练...")
    print(f"环境: {args.env_type}")

    save_dir = f'results/models/BC/{args.env_type}'
    env = Environment(env_type=args.env_type)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = BehaviorCloning(state_dim, action_dim, max_action)
    
    # expert_data = np.load(f'expert_data/env3/Astar_data.npy', allow_pickle=True).item()
    all_expert_states = expert_data['states']
    all_expert_actions = expert_data['actions']
    
    episodes = args.max_episodes
    batch_size = 256
    return_list = []
    episode_list = []  # 用于记录episode编号
    
    print(f"\n{'='*20} 开始训练 {'='*20}")
    
    with tqdm(total=episodes, desc='训练进度') as pbar:
        for i in range(episodes):
            indices = np.random.choice(len(all_expert_states), batch_size, replace=False)
            batch_states = all_expert_states[indices]
            batch_actions = all_expert_actions[indices]
            
            loss_info = agent.train(batch_states, batch_actions)
            
            if (i + 1) % 10 == 0:
                agent_copy = copy.deepcopy(agent)
                current_reward = get_reward(agent_copy, env, 5)
                return_list.append(current_reward)
                episode_list.append(i + 1)  # 记录当前episode编号
                print(f"\nEpisode {i+1}:")
                print(f"Reward = {current_reward:.2f}")
                print(f"Total Loss = {loss_info['total_loss']:.4f}")
                print(f"NLL Loss = {loss_info['nll_loss']:.4f}")
                print(f"KL Loss = {loss_info['kl_loss']:.4f}")
                print(f"Mean Std = {loss_info['mean_std']:.4f}")
                print(f"Mean Action Diff = {loss_info['mean_diff']:.4f}")

                agent.save(f'{save_dir}/BC_{i}.pth')
            
            pbar.update(1)
    
    # 绘制训练曲线
    plot_training_curve(episode_list, return_list, args)
            
    return return_list

def plot_training_curve(episodes, rewards, args):
    """绘制训练曲线并保存"""
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    
    # 确保episodes和rewards长度相同
    episodes = np.array(episodes)
    rewards = np.array(rewards)
    
    # 绘制奖励曲线
    plt.plot(episodes, rewards, label='Reward', color='blue')
    
    # 添加移动平均线
    window_size = 10
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        # 确保x轴长度匹配
        moving_avg_episodes = episodes[window_size-1:len(moving_avg)+window_size-1]
        plt.plot(moving_avg_episodes, moving_avg, 
                label=f'Moving Average ({window_size})', 
                color='red', 
                linestyle='--')
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'BC Training Curve - {args.env_type}')
    plt.legend()
    
    # 创建保存目录
    save_dir = f'results/plots/BC/{args.env_type}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图片
    plt.savefig(f'{save_dir}/training_curve.png')
    plt.close()

def get_reward(agent, env, n_episodes=5):
    agent.net.eval()  # 评估时设置为评估模式
    total_reward = 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
    agent.net.train()  # 恢复训练模式
    return total_reward / n_episodes

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    import argparse

    # 设置中文字体
    try:
        font = FontProperties(fname=r"C:\Windows\Fonts\SimHei.ttf")
    except:
        print("警告：无法加载中文字体，将使用默认字体")
        font = FontProperties()
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", default="env3", type=str)
    parser.add_argument("--max_episodes", default=1000, type=int)
    parser.add_argument("--save_freq", default=10, type=int)
    parser.add_argument("--render", default=False, action='store_true')
    args = parser.parse_args()
    set_seed(args.seed)

    # 创建保存目录
    save_dir = f'results/models/BC/{args.env_type}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练并获取结果
    expert_data = np.load(f'expert_data/env3/Astar_data.npy', allow_pickle=True).item()
    return_list = train_BC(args, expert_data)
    
    # 保存最终模型
    env = Environment(env_type=args.env_type)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = BehaviorCloning(state_dim, action_dim, max_action)
    agent.save(f'{save_dir}/BC_final.pth')
    
    # 打印训练结果
    print("\n训练完成!")
    print(f"最终平均奖励: {return_list[-1]:.2f}")
    print(f"最高平均奖励: {max(return_list):.2f}")
    print(f"模型已保存至: {save_dir}/BC_final.pth")
    print(f"训练曲线已保存至: results/plots/BC/{args.env_type}/training_curve.png")

