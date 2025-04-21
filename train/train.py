import numpy as np
import torch
from agent.agent import PPO, TD3
from env.environment import Environment
from args import get_args
from utils.utils import TrainingLogger
import random
import os
from datetime import datetime


class PPOConfig:
    def __init__(self, args):
        self.gamma = args.ppo_gamma
        self.lam = args.ppo_lam
        self.clip_ratio = args.ppo_clip_ratio
        self.lr = args.ppo_lr
        self.train_iters = args.ppo_train_iters
        self.target_kl = args.ppo_target_kl
        self.batch_size = args.ppo_batch_size
        self.minibatch_size = args.ppo_minibatch_size
        self.max_ep_len = args.ppo_max_ep_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_ppo(args):
    print("\n开始PPO训练...")
    print(f"环境: {args.env_type}")
    print(f"设备: {device}")
    
    # 初始化环境和智能体
    env = Environment(env_type=args.env_type)
    config = PPOConfig(args)
    agent = PPO(env, config)
    logger = TrainingLogger(args)

    total_steps = 0
    max_episodes = args.max_episodes
    batch_size = config.batch_size

    print(f"\n{'='*20} 开始训练 {'='*20}")
    start_time = datetime.now()

    for ep in range(max_episodes):
        obs = env.reset()
        ep_rewards = []
        obs_buf = []
        act_buf = []
        adv_buf = []
        ret_buf = []
        logp_buf = []
        val_buf = []
        rewards = []
        values = []
        log_probs = []
        dones = []

        ep_len = 0
        ep_ret = 0

        while True:
            if args.render:
                env.render()

            action, log_prob = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            ep_ret += reward
            ep_len += 1

            obs_buf.append(obs)
            act_buf.append(action)
            rewards.append(reward)
            dones.append(done)
            logp_buf.append(log_prob)

            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                _, _, value = agent.ac(obs_tensor)
            values.append(value.item())

            obs = next_obs

            if done or ep_len == config.max_ep_len:
                next_value = 0 if done else agent.ac(torch.tensor(obs, dtype=torch.float32).to(device))[2].item()
                values.append(next_value)

                advantages = agent.compute_gae(rewards, values, dones)
                returns = [adv + val for adv, val in zip(advantages, values[:-1])]

                adv_buf.extend(advantages)
                ret_buf.extend(returns)
                total_steps += ep_len

                ep_rewards.append(ep_ret)
                
                # 记录训练数据
                logger.log_episode(ep_ret, ep_len)

                ep_ret = 0
                ep_len = 0
                rewards = []
                values = []
                log_probs = []
                dones = []

                obs = env.reset()

                if total_steps >= batch_size:
                    break

        # 转换数据为numpy数组
        obs_buf = np.array(obs_buf)
        act_buf = np.array(act_buf)
        adv_buf = np.array(adv_buf)
        ret_buf = np.array(ret_buf)
        logp_buf = np.array(logp_buf)

        # 标准化优势
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

        # 更新策略
        agent.update(obs_buf, act_buf, adv_buf, ret_buf, logp_buf)

  
        if (ep + 1) % 50 == 0:
            save_path = f"{args.model_path}/ppo_{args.env_type}_{ep+1}.pth"
            agent.save_model(save_path)
            print(f"\n模型已保存至: {save_path}")


    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"\n{'='*20} 训练结束 {'='*20}")
    print(f"总用时: {training_time}")
    
   
    final_model_path = f"{args.model_path}/ppo_{args.env_type}_final.pth"
    agent.save_model(final_model_path)
    print(f"最终模型已保存至: {final_model_path}")
    

    logger.save_all()
    
    env.close()

def train_td3(args):
    print("\n开始TD3训练...")
    print(f"环境: {args.env_type}")
    print(f"设备: {device}")
    
  
    env = Environment( env_type=args.env_type)
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

    print(f"\n{'='*20} 开始训练 {'='*20}")
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
            save_path = f"{args.model_path}/td3_{args.env_type}_{episode+1}"
            agent.save(save_path)
            print(f"\n模型已保存至: {save_path}")


    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"\n{'='*20} 训练结束 {'='*20}")
    print(f"总用时: {training_time}")

    final_model_path = f"{args.model_path}/td3_{args.env_type}_final"
    agent.save(final_model_path)
    print(f"最终模型已保存至: {final_model_path}")
    

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

    os.makedirs(args.model_path, exist_ok=True)

    set_seed(args.seed)
    
 
    print("\n训练配置:")
    print(f"算法: {args.algorithm.upper()}")
    print(f"环境: {args.env_type}")
    print(f"随机种子: {args.seed}")
    print(f"最大训练回合: {args.max_episodes}")

    if args.algorithm == 'ppo':
        train_ppo(args)
    else:
        train_td3(args)