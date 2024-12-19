import numpy as np
import torch
from env.environment import Environment
from agent.td3 import TD3
from args import get_test_args
from utils.utils import save

model_path1 = r'E:\files\code\硕士论文code\Chaper2\train\results\models\td3model\env5\actor\Actor_net_step1000.pth'
model_path2 = r'E:\files\code\硕士论文code\Chaper2\train\results\models\td3model\env5\critic1\Critic1_net_step1000.pth'
model_path3 = r'E:\files\code\硕士论文code\Chaper2\train\results\models\td3model\env5\critic2\Critic2_net_step1000.pth'



def load_td3_model(env, model_path1, modelpath2, modelpath3):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = TD3(state_dim, action_dim, max_action)
    agent.load(model_path1, modelpath2, modelpath3)
    return agent



def test_model():
    args = get_test_args()
    args.algorithm = 'td3'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    env = Environment(render_mode=args.render, env_type=args.env_type)

    agent = load_td3_model(env, model_path1, model_path2, model_path3)



    print(f"\n{'=' * 20} 开始测试 {'=' * 20}")
    print(f"算法: td3")
    print(f"环境: {args.env_type}")
    print(f"模型路径: {model_path1}\n"
          f"{model_path2}\n"
          f"{model_path3}")

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

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            path_points.append(env.position.copy())

            episode_reward += reward
            steps += 1
            state = next_state

            if done:
                if info.get('distance_to_goal', 1.0) < env.delta / env.max_distance:
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