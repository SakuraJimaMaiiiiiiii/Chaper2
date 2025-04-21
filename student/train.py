import os
import numpy as np
import torch
import random
from env.environment import Environment
from utils.utils import TrainingLogger
from args import get_args,get_test_args
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_model(model, env, n_episodes=5):
    """ 评估模型在环境中的表现 """
    total_reward = 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = model.select_action(state)  # 🔥 确保模型有 select_action 方法
            state, reward, done, _ = env.step(action)
            total_reward += reward
    return total_reward / n_episodes  # 计算平均奖励


def train_model(env_name, algo, seed, expert_data):
    """ 训练 GAIL / BC / DAgger """
    from student.BC import student_BC
    from student.GAIL import GAIL
    from student.DAgger import DAgger

    set_seed(seed)
    env = Environment(env_type=env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 选择模型
    if algo == "BC":
        model = student_BC(state_dim, action_dim, max_action)
    elif algo == "GAIL":
        model = GAIL(state_dim, action_dim, max_action)
    elif algo == "DAgger":
        model = DAgger(state_dim, action_dim, max_action)
    else:
        raise ValueError(f"未知算法: {algo}")

    logger = TrainingLogger(args)

    # 训练轮数
    max_episodes = 1000
    reward_list = []
    for episode in range(max_episodes):
        episode_reward = 0
        states, actions = expert_data
        if algo == "BC" :
            loss_info = model.learn(states, actions)  # BC & DAgger 训练方法是 learn()
        elif algo == "DAgger":
            loss_info = model.train(states, actions)
        elif algo == "GAIL":
            # model.clear_trajectory()
            state = env.reset()
            for step in range(200):
                action = model.select_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                model.store_transition(state, action)
                state = next_state
                if done:
                    break

            d_loss, p_loss = model.train(expert_data[0], expert_data[1], env)
            # print(d_loss,p_loss)

        # 评估当前模型
        if (episode + 1) % 10 == 0:
            reward = evaluate_model(model, env)
            reward_list.append(reward)
            # model.save_model(env_name, episode, reward)
            logger.log_episode(reward, episode + 1)
            print(f"[{env_name} - {algo} - Seed {seed}] Episode {episode + 1}, Reward: {reward:.2f}")
            # 创建字典


    # 保存模型
    model_dir = f"models/{env_name}/{algo}/seed_{2023}"
    os.makedirs(model_dir, exist_ok=True)
    reward_dict = {'rewards': reward_list}
    # dumps 将数据转换成字符串
    info_json = json.dumps(reward_dict)
    # 显示数据类型
    f = open(model_dir+'_rewards.json', 'w')
    f.write(info_json)
    model.save(f"{model_dir}/{algo}_final.pth")

    logger.save_all()
    return model


if __name__ == "__main__":
    args = get_args()
    test_args = get_test_args()
    env_list = ["env5"]
    # algos = [ "BC"]
    algos = ["GAIL",]
    seeds = [22]

    for env_name in env_list:
        args.env_type = env_name
        test_args.env_type = env_name
        for algo in algos:
            for seed in seeds:
                # 加载专家数据
                # expert_data_path = f"expert_data/{env_name}_astar.npy"
                expert_data_path = f"expert_data/{env_name}_td3.npy"
                if not os.path.exists(expert_data_path):
                    print(f"❌ 专家数据文件不存在: {expert_data_path}, 跳过训练！")
                    continue

                expert_data = np.load(expert_data_path, allow_pickle=True).item()

                states = np.array(expert_data["states"])
                actions = np.array(expert_data["actions"])

                train_model(env_name, algo, seed, (states, actions))
