import numpy as np
import torch
from agent.td3 import TD3
from env.environment import Environment
from args import get_args
from utils.utils import TrainingLogger
import random
import os
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_td3(args):
    print("\n开始TD3训练...")
    print(f"环境: {args.env_type}")
    print(f"设备: {device}")
    save_dir = f'results/models/td3model/{args.env_type}'
    save_path1 = f"{save_dir}/actor"
    save_path2 = f"{save_dir}/critic1"
    save_path3 = f"{save_dir}/critic2"
    args.algorithm = 'td3'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_path1, exist_ok=True)
    os.makedirs(save_path2, exist_ok=True)
    os.makedirs(save_path3, exist_ok=True)

    env = Environment(env_type=args.env_type)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        use_per=args.use_per,
        use_her=args.use_her,
        buffer_size=args.buffer_size,
        k_goals=args.k_goals,
        per_alpha=args.per_alpha,
        per_beta=args.per_beta,
        per_beta_increment=args.per_beta_increment,
        per_epsilon=args.per_epsilon,
        reward_scale=args.reward_scale
    )

    logger = TrainingLogger(args)

    exploration_noise = args.td3_noise
    episodes = args.max_episodes
    batch_size = args.td3_batch_size
    warmup_steps = args.td3_warmup_steps
    steps = 0

    print(f"\n{'=' * 20} 开始训练 {'=' * 20}")
    print(f"{'使用PER' if args.use_per else '不使用PER'}")
    print(f"{'使用HER' if args.use_her else '不使用HER'}")
    start_time = datetime.now()

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_transitions = []
        done = False

        while not done:
            if args.render:
                env.render()

            if steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
                action = (action + np.random.normal(0, exploration_noise, size=action_dim)
                          ).clip(-max_action, max_action)

            next_state, reward, done, _ = env.step(action)

            if args.use_her:
                episode_transitions.append((state, action, next_state, reward, done))
            else:
                agent.store_transition(state, action, next_state, reward, done)

            state = next_state
            episode_reward += reward
            steps += 1
            episode_steps += 1

            if len(agent.replay_buffer) > batch_size:
                agent.train(
                    batch_size=batch_size,
                    discount=args.td3_gamma,
                    tau=args.td3_tau,
                    policy_noise=args.td3_policy_noise,
                    noise_clip=args.td3_noise_clip,
                    policy_freq=args.td3_policy_freq
                )

        if args.use_her:
            agent.store_episode(episode_transitions)

        logger.log_episode(episode_reward, episode_steps)

        exploration_noise = max(0.01, exploration_noise * 0.995)

        if (episode + 1) % 50 == 0:
            actor_path = os.path.join(save_path1, f"Actor_net_step{episode+1}.pth")
            critic1_path = os.path.join(save_path2, f"Critic1_net_step{episode+1}.pth")
            critic2_path = os.path.join(save_path3, f"Critic2_net_step{episode+1}.pth")

            os.makedirs(os.path.dirname(actor_path), exist_ok=True)
            os.makedirs(os.path.dirname(critic1_path), exist_ok=True)
            os.makedirs(os.path.dirname(critic2_path), exist_ok=True)

            agent.save(actor_path, critic1_path, critic2_path)
            print(f"\n模型已保存至: actor:{save_path1} \n"
                  f"critic1:{save_path2} \n"
                  f"critic2:{save_path3}")

    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"\n{'=' * 20} 训练结束 {'=' * 20}")
    print(f"总用时: {training_time}")

    logger.save_all()

    env.close()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    args = get_args()

    set_seed(args.seed)

    print("\n训练配置:")
    print(f"算法: td3")
    print(f"环境: {args.env_type}")
    print(f"随机种子: {args.seed}")
    print(f"最大训练回合: {args.max_episodes}")



    train_td3(args)