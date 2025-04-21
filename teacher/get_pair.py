from teacher.Astar_teacher import Astar_teacher
from teacher.RRT_teacher import RRT_teacher
import args
from env.environment import Environment
import numpy as np


def get_sensor_readings(env, position):
    # 传感器读数函数
    readings = []
    for direction in env.sensor_directions:
        min_distance = env.sensor_range
        for i in np.linspace(0, env.sensor_range, num=10):
            probe_pos = position + direction * i
            if (probe_pos < 0).any() or (probe_pos > env.grid_size).any():
                min_distance = i
                break  # 超出边界
            is_obstacle = False
            for (x1, y1, z1), (x2, y2, z2) in env.obstacles:
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                z_min, z_max = min(z1, z2), max(z1, z2)
                if (x_min <= probe_pos[0] <= x_max and
                        y_min <= probe_pos[1] <= y_max and
                        z_min <= probe_pos[2] <= z_max):
                    min_distance = i
                    is_obstacle = True
                    break
            if is_obstacle:
                break
        readings.append(min_distance)
    return np.array(readings, dtype=np.float32)


def get_state(env, path):
    states = []
    for i in path:
        relative_position = (env.goal - i) / env.grid_size
        abs_distance = np.linalg.norm(i - env.goal)
        normalized_distance = abs_distance / env.max_distance
        direction_to_goal = (env.goal - i) / ((np.linalg.norm(env.goal - i) + 1e-8))
        sensor_readings = get_sensor_readings(env, i)
        state = np.concatenate(
            (relative_position, [normalized_distance], direction_to_goal, [abs_distance], sensor_readings))
        states.append(state.tolist())
    return states

# 动作到 n-1 除去终点的动作/把终点的动作对变成(0,0,0)
def get_action(env,path):
    actions = []
    for current, next in zip(path, path[1:]):
        action = np.array(next) - np.array(current)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        actions.append(action)
    actions.append([0, 0, 0])
    return actions

# 输出(s,a) , s , a

def evaluate_astar_reward(args, path):
    """修正后的A*路径奖励评估"""
    total_reward = 0
    env_copy = Environment(env_type=args.env_type)
    
    # 重置环境并设置初始状态
    env_copy.reset()
    env_copy.position = np.array(path[0])  # 设置起始位置
    
    # 执行路径并累积奖励
    for i in range(len(path) - 1):
        current_pos = np.array(path[i])
        next_pos = np.array(path[i+1])
        action = next_pos - current_pos
        
        # 确保动作在动作空间范围内
        action = np.clip(action, env_copy.action_space.low, env_copy.action_space.high)
        
        # 执行动作并获取奖励
        _, reward, done, _ = env_copy.step(action)
        total_reward += reward
        
        if done:
            break
            
        # 验证位置是否正确
        # if not np.allclose(env_copy.position, next_pos, atol=1e-3):
        #     print(f"Warning: Position mismatch at step {i}")
        #     print(f"Expected: {next_pos}")
        #     print(f"Got: {env_copy.position}")
    
    return total_reward

def get_pair(env, args):
    teacher = args.teacher
    samples = 1000
    pairs = []
    states = []
    actions = []
    path = []
    for i in range(samples):
        if teacher == 'Astar':
            path = Astar_teacher(args)
            path = list(map(tuple, path))
            path = path[::-1]
        elif teacher == 'RRT':
            path = RRT_teacher(args)
        elif teacher == 'RRTStar':
            path = RRT_teacher(args)
        elif teacher == 'RRTStarBidirectional':
            path = RRT_teacher(args)
        # if path is not None:
        #     reward = evaluate_astar_reward(args, path)
        #     print(f"\nA*路径奖励: {reward:.2f}")
            # if reward < 10 :
            #     continue

        if path:
            state = get_state(env, path)
            action = get_action(env, path)
            for j in range(len(path)):
                pair = (state[j][0], action[j][0])
                pairs.append(pair)
                states.append(state[j])
                actions.append(action[j])
    return pairs, states, actions



if __name__ == '__main__':
    args = args.get_args()
    env = Environment(env_type=args.env_type)

    pairs, states, actions = get_pair(env, args)

