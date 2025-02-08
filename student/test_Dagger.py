import torch
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.environment import Environment
import numpy as np
from student.BC import BehaviorCloning as student_BC
from utils.utils import save
from args import get_test_args
from student.DAgger import DAgger
from student.GAIL import GAIL
from student.TD3_student import TD3_student

def load_DAgger_model(env, model_path):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = DAgger(state_dim, action_dim, max_action)
    agent.load(model_path)
    agent.net.to('cuda')
    return agent



# args 为get_test_args
def test_DAgger(args, model_path):
    env = Environment(env_type=args.env_type)
    agent = load_DAgger_model(env, model_path)

    print(f"\n{'=' * 20} 开始测试 {'=' * 20}")
    print(f"算法: BC")
    print(f"环境: {args.env_type}")
    print(f"模型路径: {model_path}\n")


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
    model_path = r'E:\files\论文\IL\student\results\models\DAgger\env3\DAgger_final.pth'
    test_DAgger(args, model_path)