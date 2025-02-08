import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from env.environment import Environment
from student.get_data import load_td3_model, sample_td3_data, sample_expert_data
from utils.utils import TrainingLogger
from args import get_args
import random
device = torch.device("cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Policy(nn.Module):
    """简单的策略网络"""
    def __init__(self, state_dim, action_dim, max_action):
        super(Policy, self).__init__()
        self.max_action = max_action
        
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.policy(state) * self.max_action

class Discriminator(nn.Module):
    """简单的判别器"""
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state, action):
        return self.discriminator(torch.cat([state, action], dim=1))

class GAIL:
    def __init__(self, state_dim, action_dim, max_action):
        self.policy = Policy(state_dim, action_dim, max_action).to(device)
        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-4)
        
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.trajectory_states = []
        self.trajectory_actions = []
        
        # 添加模型保存路径
        self.save_dir = 'models/gail'
        os.makedirs(self.save_dir, exist_ok=True)
    
    def save_model(self, env_type, episode, reward):
        """
        保存模型
        Args:
            env_type: 环境类型
            episode: 当前回合数
            reward: 当前回合奖励
        """
        save_path = os.path.join(self.save_dir, f'{env_type}_episode_{episode}.pth')
        torch.save({
            'episode': episode,
            'policy_state_dict': self.policy.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'reward': reward
        }, save_path)
        
        # 同时保存最新的模型
        latest_path = os.path.join(self.save_dir, f'{env_type}_latest.pth')
        torch.save({
            'episode': episode,
            'policy_state_dict': self.policy.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'reward': reward
        }, latest_path)
        
        print(f"模型已保存: {save_path}")
    
    def load_model(self, env_type, episode=None):
        """
        加载模型
        Args:
            env_type: 环境类型
            episode: 指定回合数（如果为None则加载最新模型）
        Returns:
            bool: 是否成功加载模型
        """
        try:
            if episode is None:
                # 加载最新模型
                load_path = os.path.join(self.save_dir, f'{env_type}_latest.pth')
            else:
                # 加载指定回合的模型
                load_path = os.path.join(self.save_dir, f'{env_type}_episode_{episode}.pth')
            
            if not os.path.exists(load_path):
                print(f"未找到模型: {load_path}")
                return False
            
            checkpoint = torch.load(load_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
            
            print(f"成功加载模型: {load_path}")
            print(f"模型信息: Episode {checkpoint['episode']}, Reward {checkpoint['reward']:.2f}")
            return True
            
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return False
    
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action = self.policy(state).cpu().numpy()
        return action
    
    def store_transition(self, state, action):
        self.trajectory_states.append(state)
        self.trajectory_actions.append(action)
    
    def clear_trajectory(self):
        self.trajectory_states = []
        self.trajectory_actions = []
    
    def train(self, expert_states, expert_actions):
        # 如果没有足够的数据，不进行训练
        if len(self.trajectory_states) < 32:
            return 0, 0
        
        # 准备数据
        policy_states = torch.FloatTensor(self.trajectory_states).to(device)
        policy_actions = torch.FloatTensor(self.trajectory_actions).to(device)
        
        expert_idx = np.random.randint(0, len(expert_states), len(self.trajectory_states))
        expert_states_batch = torch.FloatTensor(expert_states[expert_idx]).to(device)
        expert_actions_batch = torch.FloatTensor(expert_actions[expert_idx]).to(device)
        
        # 训练判别器
        for _ in range(5):  # 多次更新判别器
            expert_probs = self.discriminator(expert_states_batch, expert_actions_batch)
            policy_probs = self.discriminator(policy_states, policy_actions)
            
            discriminator_loss = -(torch.log(expert_probs + 1e-8).mean() + 
                                 torch.log(1 - policy_probs + 1e-8).mean())
            
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
        
        # 训练策略
        policy_actions = self.policy(policy_states)
        policy_probs = self.discriminator(policy_states, policy_actions)
        policy_loss = -torch.log(policy_probs + 1e-8).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return discriminator_loss.item(), policy_loss.item()

def train_Gail(args):
    env = Environment(env_type=args.env_type)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    gail = GAIL(state_dim, action_dim, max_action)
    logger = TrainingLogger(args)
    
    # 尝试加载已有模型
    if args.load_model:
        gail.load_model(args.env_type)
    
    # 加载专家数据
    data_path = f'expert_data/{args.env_type}/TD3_data.npy'
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    if os.path.exists(data_path):
        print(f"加载已存在的专家数据: {data_path}")
        data = np.load(data_path, allow_pickle=True).item()
        expert_states = data['states']
        expert_actions = data['actions']
    else:
        print("收集专家数据中...")
        expert_states = []
        expert_actions = []
        total_samples = 500 * args.Gail_batch_size
        
        while len(expert_states) < total_samples:
            args.teacher = 'Astar'
            if args.sample_data == 'expert':
                _, states, actions = sample_expert_data(env, args, args.Gail_batch_size)
            elif args.sample_data == 'td3':
                _, states, actions = sample_td3_data(args, env, 
                                                   load_td3_model(env, device=device),
                                                   args.Gail_batch_size)
            expert_states.extend(states)
            expert_actions.extend(actions)
            print(f'已收集数据量: {len(expert_states)}')
            
        expert_states = np.array(expert_states[:total_samples])
        expert_actions = np.array(expert_actions[:total_samples])
        np.save(data_path, {'states': expert_states, 'actions': expert_actions})
    
    print(f"专家数据数量: {len(expert_states)}")
    
    # 训练循环
    max_episodes = 1000
    best_reward = float('-inf')
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        gail.clear_trajectory()
        
        # 收集一个episode的数据
        for step in range(200):
            # 选择动作
            action = gail.select_action(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储transition
            gail.store_transition(state, action)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # 训练GAIL
        d_loss, p_loss = gail.train(expert_states, expert_actions)
        
        # 记录和打印
        logger.log_episode(episode_reward, step + 1)
        
        # 每回合都保存模型
        gail.save_model(args.env_type, episode + 1, episode_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, "
                  f"D_Loss = {d_loss:.3f}, P_Loss = {p_loss:.3f}")
            
            # 保存最佳模型
            if episode_reward > best_reward:
                best_reward = episode_reward
                gail.save_model(args.env_type, episode + 1, episode_reward)
    
    logger.save_all()

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    # 添加模型加载参数
    args.load_model = False  # 可以通过命令行参数设置
    train_Gail(args)
