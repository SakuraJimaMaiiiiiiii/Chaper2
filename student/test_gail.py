import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.environment import Environment
import numpy as np
from utils.utils import save
from args import get_test_args
from student.GAIL import GAIL


def load_Gail_model(env, env_type):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent1 = GAIL(state_dim, action_dim, max_action)
    agent1.load_model(env_type, 999)
    return agent1


# args 为get_test_args
def test_gail(args):
    env = Environment(env_type=args.env_type)
    agent = load_Gail_model(env, args.env_type)

    print(f"\n{'=' * 20} 开始测试 {'=' * 20}")
    print(f"算法: gail")
    print(f"环境: {args.env_type}")


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

            if done or steps > 50:
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
    args = get_test_args()
    test_gail(args)