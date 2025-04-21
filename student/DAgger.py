import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.optim as optim
import numpy as np
from nn_net.net import BCNet
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from env.environment import Environment
from student.get_data import load_td3_model, sample_td3_data, sample_expert_data
import random
from utils.utils import TrainingLogger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DAgger:
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
            action = mean.cpu().data.numpy().flatten()
        self.net.train()
        return action
    
    def train(self, states, actions):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        
        # 数据增强
        noise_scale = 0.05
        states = states + torch.randn_like(states) * noise_scale
        
        self.net.train()
        mean, std = self.net(states)
        
        # 计算负对数似然损失
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions)
        nll_loss = -log_prob.mean()
        
        # KL散度正则化
        kl_div = -0.5 * (1 + 2*torch.log(std) - mean.pow(2) - std.pow(2))
        kl_loss = 0.01 * kl_div.mean()
        
        total_loss = nll_loss + kl_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'nll_loss': nll_loss.item(),
            'kl_loss': kl_loss.item(),
            'mean_std': std.mean().item(),
            'mean_diff': (mean - actions).abs().mean().item()
        }
    
    def save(self, filename):
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        
    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def get_expert_action(env, state,agent):
    """获取专家动作（这里需要实现专家策略）"""
    action = agent.select_action(state)
    return action


def train_DAgger(args, expert_data):
    print("\n开始DAgger训练...")
    print(f"环境: {args.env_type}")
    
    env = Environment(env_type=args.env_type)
    env_dagger = Environment(env_type=args.env_type)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = DAgger(state_dim, action_dim, max_action)
    expert_agent = load_td3_model(env)
    logger = TrainingLogger(args)
    # 加载初始专家数据
    # expert_data = np.load(f'expert_data/env3/Astar_data.npy', allow_pickle=True).item()

    all_states = expert_data['states']
    all_actions = expert_data['actions']
    all_states = all_states[:10000]
    all_actions = all_actions[:10000]
    
    episodes = args.max_episodes
    dagger_iters = args.dagger_iters  # DAgger迭代次数
    batch_size = 256
    return_list = []
    episode_list = []
    episode = 0
    
    print(f"\n{'='*20} 开始训练 {'='*20}")
    
    with tqdm(total=episodes * dagger_iters, desc='训练进度') as pbar:
        for dagger_iter in range(dagger_iters):
            # 收集新数据
            new_states = []
            new_actions = []
            for i in range(10):
                state = env.reset()
                done = False
                while not done:
                    # 使用当前策略选择动作
                    action = agent.select_action(state)
                    # 获取专家动作
                    expert_action = get_expert_action(env_dagger, state,expert_agent)

                    new_states.append(state)
                    new_actions.append(expert_action)

                    next_state, _, done, _ = env.step(action)
                    state = next_state

                # 合并数据
            if len(new_states) > 0:
                new_states = np.array(new_states)
                new_actions = np.array(new_actions)
                all_states = np.concatenate([all_states, new_states])
                all_actions = np.concatenate([all_actions, new_actions])
            
            # 训练模型
            for i in range(episodes):
                indices = np.random.choice(len(all_states), batch_size, replace=False)
                batch_states = all_states[indices]
                batch_actions = all_actions[indices]
                
                loss_info = agent.train(batch_states, batch_actions)
                
                if (i + 1) % 10 == 0:
                    agent_copy = copy.deepcopy(agent)
                    current_reward = get_reward(agent_copy, env, 5)
                    return_list.append(current_reward)
                    episode_list.append(i + 1 + dagger_iter * episodes)
                    episode += 1
                    logger.log_episode(current_reward,episode)
                    print(f"\nDAgger Iter {dagger_iter + 1}, Episode {i + 1}:")
                    print(f"Reward = {current_reward:.2f}")
                    print(f"Dataset Size = {len(all_states)}")
                    print(f"Loss Info: {loss_info}")
                
                pbar.update(1)
    
    # 保存模型和绘制曲线
    save_dir = f'results/models/DAgger/{args.env_type}'
    os.makedirs(save_dir, exist_ok=True)
    agent.save(f'{save_dir}/DAgger_final.pth')
    logger.save_all()
    plot_training_curve(episode_list, return_list, args)
    
    return return_list

def get_reward(agent, env, n_episodes=5):
    agent.net.eval()
    total_reward = 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
    agent.net.train()
    return total_reward / n_episodes

def plot_training_curve(episodes, rewards, args):
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    
    episodes = np.array(episodes)
    rewards = np.array(rewards)
    
    plt.plot(episodes, rewards, label='Reward', color='blue')
    
    window_size = 10
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        moving_avg_episodes = episodes[window_size-1:len(moving_avg)+window_size-1]
        plt.plot(moving_avg_episodes, moving_avg, 
                label=f'Moving Average ({window_size})', 
                color='red', 
                linestyle='--')
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'DAgger Training Curve - {args.env_type}')
    plt.legend()
    
    save_dir = f'results/plots/DAgger/{args.env_type}'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/training_curve.png')
    plt.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", default="env3", type=str)
    parser.add_argument("--max_episodes", default=200, type=int)
    parser.add_argument("--dagger_iters", default=5, type=int)
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--render", default=False, action='store_true')
    parser.add_argument('--algorithm', type=str, default='BC',help='选择训练算法(ppo或td3)')
    parser.add_argument('--seed', type=int, default=22, choices=[42, 32, 22], help='随机种子')
    args = parser.parse_args()

    set_seed(args.seed)
    expert_data = np.load(f'expert_data/env3/Astar_data.npy', allow_pickle=True).item()

    return_list = train_DAgger(args, expert_data)