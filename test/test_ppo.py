import numpy as np
import torch
from env.environment import Environment
from agent.ppo import PPO
from args import get_test_args
from train.train import PPOConfig
from utils.utils import save


model_path = r'E:\files\code\硕士论文code\Chaper2\train\results\models\ppomodel\env5\env5_ActorCritic_net_step900.pth'


def load_ppo_model(env, model_path, args):
    """加载PPO模型"""
    # 创建PPO配置
    config = PPOConfig(args)
    agent = PPO(env, config)
    agent.ac.load_state_dict(torch.load(model_path))
    return agent


def test_model():
    args = get_test_args()
    args.algorithm = 'ppo'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    env = Environment(render_mode=args.render, env_type=args.env_type)


    agent = load_ppo_model(env, model_path, args)


    print(f"\n{'=' * 20} 开始测试 {'=' * 20}")
    print(f"算法: PPO")
    print(f"环境: {args.env_type}")
    print(f"模型路径: {model_path}")

    total_rewards = []
    total_steps = []
    success_count = 0

    for i in range(args.test_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        path_points = [env.position.copy()]

        print(f"\n测试回合 {i + 1}/{args.test_episodes}")

        while True:
            if args.render:
                env.render()


            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                action, _, _ = agent.ac(state_tensor)
                action = action.cpu().numpy()

            next_state, reward, done, info = env.step(action)
            path_points.append(env.position.copy())

            episode_reward += reward
            steps += 1
            state = next_state

            if done:
                if info.get('distance_to_goal', 1) < env.delta / env.max_distance:
                    success_count += 1
                break

        total_rewards.append(episode_reward)
        total_steps.append(steps)

        print(f"回合奖励: {episode_reward:.2f}")
        print(f"步数: {steps}")
        save(path_points, env.obstacles, env, args, i)

    print(f"\n{'=' * 20} 测试结果 {'=' * 20}")
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"平均步数: {np.mean(total_steps):.2f} ± {np.std(total_steps):.2f}")
    print(f"成功率: {success_count / args.test_episodes * 100:.2f}%")


    env.close()


if __name__ == '__main__':
    test_model()