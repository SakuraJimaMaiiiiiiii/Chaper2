import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.environment import Environment
import numpy as np
from utils.utils import save_3d, calculate_path_length
from args import get_test_args
from student.DAgger import DAgger
from curvature_plot3d import compute_total_curvature
import time



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
    print(f"算法: Dagger")
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
        start_time = time.time()
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
            save_3d(path_points, env.obstacles, env, args, i)

            '''
            保存路径
            '''
            dir = r'E:\files\code\硕士论文code\Chaper2'
            point_path = f"{dir}/Dagger.txt"
            with open(point_path, 'w') as f:
                for point in path_points:
                    f.write(f"{point}\n")
            print(f'路径已保存至{point_path}')

        end_time = time.time()
        print(f"\n{'=' * 20} 测试结果 {'=' * 20}")
        print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"平均步数: {np.mean(total_steps):.2f} ± {np.std(total_steps):.2f}")
        print(f"成功率: {success_count / args.test_episodes * 100:.2f}%")
        print(f"运行时间: {end_time - start_time:.2f} 秒")
        print(f"\n{'=' * 20} 路径长度 {'=' * 20} \n{calculate_path_length(path_points)}")
        print(f"\n{'=' * 20} 路径曲率 {'=' * 20} \n{compute_total_curvature(path_points)}")

        env.close()


if __name__ == '__main__':
    args = get_test_args()
    model_path = r'E:\files\code\硕士论文code\Chaper2\student\models\env5\DAgger\seed_42\DAgger_final.pth'
    test_DAgger(args, model_path)