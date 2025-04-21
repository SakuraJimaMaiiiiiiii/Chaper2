from agent.td3 import TD3
from agent.ppo import PPO
from teacher.get_pair import get_pair
import numpy as np
import random
from env.environment import Environment
from teacher.Astar_teacher import Astar_teacher
from args import get_test_args
# from train.train import PPOConfig
import torch


def load_td3_model(env, env_name,device='cpu'):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = TD3(state_dim, action_dim, max_action)

    model_path1 = f'../finalresult/savemodel/{env_name}/td3/Actor_net_step1000.pth'
    # model_path1 = r'..\finalresult\savemodel\env3\td3\Actor_net_step1000.pth'
    model_path2 = f'../finalresult/savemodel/{env_name}/td3/Critic1_net_step1000.pth'
    # model_path2 = r'..\finalresult\savemodel\env3\td3\Critic1_net_step1000.pth'
    model_path3 = f'../finalresult/savemodel/{env_name}/td3/Critic2_net_step1000.pth'
    # model_path3 = r'..\finalresult\savemodel\env3\td3\Critic2_net_step1000.pth'

    agent.load(model_path1, model_path2, model_path3, device)
    return agent


# def load_ppo_model(env, model_path, args):
#     """加载PPO模型"""
#     # 创建PPO配置
#     config = PPOConfig(args)
#     agent = PPO(env, config)
#     agent.ac.load_state_dict(torch.load(model_path,map_location='cpu'))
#     return agent


def sample_td3_data(args, env, agent, batch_size):
    states = []
    actions = []
    episodes = 10
    samples = 256

    for i in range(samples):
        for i in range(episodes):
            # print('*'*100)
            rewards = 0
            state = env.reset()
            while True:
                if args.render:
                    env.render()
                action = agent.select_action(state)
                noise = np.random.normal(scale=0.2,size = action.shape)
                action = action+noise

                # print(action)
                next_state, reward, done, info = env.step(action)
                # if reward >0:
                states.append(state)
                actions.append(action)
                rewards += reward
                state = next_state
                if done:
                    break
            print(rewards)
        env.close()
    assert batch_size <= len(states)
    index = random.sample(range(len(states)), batch_size)
    # states = [states[i] for i in index]
    # actions = [actions[i] for i in index]

    return index, np.array(states), np.array(actions)

def sample_PPO_data(args, env, agent, batch_size):
    states = []
    actions = []
    episodes = 1
    samples = 256
    for i in range(samples):
        for i in range(episodes):
            state = env.reset()
            while True:
                if args.render:
                    env.render()
                action = agent.select_action(state)
                states.append(state)
                actions.append(action)
                next_state, reward, done, info = env.step(action)
                state = next_state
                if done:
                    break
        env.close()
    assert batch_size <= len(states)
    index = random.sample(range(len(states)), batch_size)
    states = [states[i] for i in index]
    actions = [actions[i] for i in index]

    return index, np.array(states), np.array(actions)

def evaluate_astar_reward(args, path):
    """简化版的路径奖励评估，只关注位置相关的奖励"""
    total_reward = 0
    env_copy = Environment(env_type=args.env_type)
    
    # 重置环境
    env_copy.reset()
    
    # 直接设置初始位置
    env_copy.position = np.array(path[0])
    
    # 计算路径上每一步的奖励
    for i in range(len(path) - 1):
        # 计算位置差作为动作
        action = np.array(path[i + 1]) - np.array(path[i])
        
        # 执行动作
        _, reward, done, _ = env_copy.step(action)
        total_reward += reward
        
        # 直接设置下一个位置，避免累积误差
        env_copy.position = np.array(path[i + 1])
        
        if done:
            break
    
    return total_reward


# args = args.get_args()
# env = Environment(env_type=args.env_type)
# agent = load_td3_model(env)
# index, states, actions = sample_td3_data(env, agent, batch_size = 5)
# print(f"index:{index},actions{actions}")


def sample_expert_data(args, env, agent, batch_size):
    # 获取A*路径并评估
    # path = Astar_teacher(args)
    # if path is not None:
    #     reward = evaluate_astar_reward(args, path)
    #     print(f"\nA*路径奖励: {reward:.2f}")
    
    # 使用原有的get_pair获取训练数据
    pairs, states, actions = get_pair(env, args)

    
    return 0, states, actions  # 保持原有返回格式

# args = args.get_args()
# env = Environment(env_type=args.env_type)
# index, states, actions = sample_expert_data(env, args, batch_size = 100)
# print(f"index:{index},actions{actions}")

if __name__ == '__main__':
    import args

    'td3,ppo data'
    args = args.get_args()
    env = Environment(env_type=args.env_type)
    agent = load_td3_model(env)
    index, states, actions = sample_td3_data(args, env, agent, batch_size=5)
    print(f"index:{index},actions{actions}")



    'expert_data'
    # args = args.get_args()
    # env = Environment(env_type=args.env_type)
    # index, states, actions = sample_expert_data(env, args, batch_size = 100)
    # print(f"index:{index},actions{actions}")