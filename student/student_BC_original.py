import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.optim as optim
from env.environment import Environment
import args
from nn_net.net import Actor
import numpy as np
import time
from tqdm import tqdm
import copy
from student.get_data import load_td3_model, sample_td3_data, sample_expert_data
import matplotlib.pyplot as plt
from utils.utils import TrainingLogger
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def get_reward(agent, env, n_episode):
    return_list = []
    agent.actor.to('cpu')
    for episode in range(n_episode):
        ep_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_return += reward
        return_list.append(ep_return)
    return np.mean(return_list)


class student_BC:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.step = 0
        # self.loss = torch.nn.CrossEntropyLoss()
        self.loss = torch.nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    # 输入专家数据
    def learn(self, expert_state, expert_action):
        expert_state = torch.FloatTensor(expert_state).to(device)
        expert_action = torch.FloatTensor(expert_action).to(device)
        BC_action = self.actor(expert_state).to(self.device)
        loss = self.loss(expert_action, BC_action)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step += 1

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename)

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename))

    def save_expert_data(self, states, actions, filename):
        """保存专家数据到文件"""
        data = {
            'states': states,
            'actions': actions
        }
        np.save(filename, data)
        print(f"专家数据已保存至: {filename}")

    def load_expert_data(self, filename):
        """从文件加载专家数据"""
        data = np.load(filename, allow_pickle=True).item()
        return data['states'], data['actions']



# args 为get_args
def train_BC(args):
    print("\n开始BC训练...")
    print(f"环境: {args.env_type}")
    print(f"设备: {device}")
    print(f"\n{'=' * 20} 开始训练 {'=' * 20}")

    env = Environment(env_type=args.env_type)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    logger = TrainingLogger(args)
    step = 0
    agent = student_BC(state_dim, action_dim, max_action)

    episodes = args.max_episodes
    batch_size = 256
    
    # 收集1000个batch的数据
    total_samples = 10 * batch_size
    data_path = f'expert_data/{args.env_type}/{args.teacher}_data.npy'
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # 检查是否存在已保存的专家数据
    if os.path.exists(data_path):
        print(f"加载已存在的专家数据: {data_path}")
        all_expert_states, all_expert_actions = agent.load_expert_data(data_path)
    else:
        print("收集专家数据中...")
        all_expert_states = []
        all_expert_actions = []

        times_get_data = 0
        while len(all_expert_states) < total_samples:
            if times_get_data == 0:
                args.teacher = 'Astar'
            elif times_get_data == 1:
                args.teacher = 'Astar'
            else:
                args.teacher = 'Astar'
            if args.sample_data == 'expert':
                _, expert_state, expert_action = sample_expert_data(env, args, batch_size)
            elif args.sample_data == 'td3':
                _, expert_state, expert_action = sample_td3_data(args, env, load_td3_model(env, device=device), batch_size)
            all_expert_states.extend(expert_state)
            all_expert_actions.extend(expert_action)
            if times_get_data == 2:
                times_get_data = 0
            times_get_data += 1
            print(f'已收集数据量: {len(all_expert_states)}')
        
        # 转换为numpy数组并保存
        all_expert_states = np.array(all_expert_states[:total_samples])
        all_expert_actions = np.array(all_expert_actions[:total_samples])
        agent.save_expert_data(all_expert_states, all_expert_actions, data_path)
    
    print(f"专家数据数量: {len(all_expert_states)}")
    
    return_list = []
    start_time = time.time()
    
    with tqdm(total=episodes, desc='训练进度') as pbar:
        for i in range(episodes):
            step += 1
            # 从收集的数据中随机采样一个batch
            indices = np.random.choice(total_samples, batch_size, replace=False)
            batch_states = all_expert_states[indices]
            batch_actions = all_expert_actions[indices]
            
            agent.learn(batch_states, batch_actions)
            
            # 评估当前模型
            agent_copy = copy.deepcopy(agent)
            current_reward = get_reward(agent_copy, env, 5)
            return_list.append(current_reward)
            logger.log_episode(current_reward, step)
            pbar.update(1)

        end_time = time.time()
        save_dir = f'BCresult/models/{args.env_type}/{args.seed}/{args.teacher}'
        save_path = os.path.join(save_dir, f"Actor_net_{args.teacher}.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        agent.save(save_path)
        logger.save_all()

    print(f"总用时: {end_time-start_time}")
    return return_list





if __name__ == '__main__':
    args = args.get_args()
    set_seed(args.seed)
    return_list = train_BC(args)
    iteration_list = list(range(len(return_list)))
    plt.plot(iteration_list, return_list)
    plt.xlabel('epoch')
    plt.ylabel('Returns')
    plt.title('BC on {}'.format(args.env_type))
    save_dir = r'.\Chaper2\student\logs\td3_env3'
    plt.savefig(os.path.join(save_dir, 'Return_curves.png'),
                bbox_inches='tight', dpi=300)




