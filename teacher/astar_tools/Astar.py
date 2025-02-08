import math
import numpy as np
import random



# 判断一个点是否为障碍物
def f_is_safe(point_f, env_f, epsilon = 0.5):
    for [bos_s, obs_e] in env_f:
        if point_f[1] > (bos_s[1]-epsilon) and point_f[2] > (bos_s[2]-epsilon) and point_f[0] > (bos_s[0]-epsilon):
            if point_f[1] < (obs_e[1] + epsilon) and point_f[2] < (obs_e[2]+epsilon) and point_f[0] < (obs_e[0]+epsilon):
                return False
    return True

# 判断两个点之间是否有点在障碍物里
def is_mid_safe(point1,point2, env, step=100):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    x_mid = np.linspace(x1, x2, step)
    y_mid = np.linspace(y1, y2, step)
    z_mid = np.linspace(z1, z2, step)

    for i in range(step):
        x_value = x_mid[i]
        y_value = y_mid[i]
        z_value = z_mid[i]
        point = (x_value, y_value, z_value)
        # 如果点在障碍物中
        if not f_is_safe(point,env):
            return False

    return True



# 获得相邻节点
def f_neighbors(node, random_step=True):
    """获得相邻节点，添加随机步长"""
    neighbors = []
    temp_x = node[0]
    temp_y = node[1]
    temp_z = node[2]
    
    # 基础步长
    base_step = 0.8
    
    # 可能的方向
    directions = [-1, 0, 1]
    
    for dx in directions:
        for dy in directions:
            for dz in directions:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                    
                if random_step:
                    # 添加随机扰动到步长
                    step_x = base_step * dx * (1 + 0.2 * (random.random() * 2 - 1))
                    step_y = base_step * dy * (1 + 0.2 * (random.random() * 2 - 1))
                    step_z = base_step * dz * (1 + 0.2 * (random.random() * 2 - 1))
                else:
                    step_x = base_step * dx
                    step_y = base_step * dy
                    step_z = base_step * dz
                
                neighbor = [temp_x + step_x, temp_y + step_y, temp_z + step_z]
                
                if neighbor[0] <= 0 or neighbor[1] <= 0 or neighbor[2] <= 0:
                    continue
                neighbors.append(neighbor)
    
    # 随机打乱邻居顺序
    random.shuffle(neighbors)
    return neighbors


# 计算距离
def f_distcal(Point_1, point_2):
    dx = Point_1[0] - point_2[0]
    dy = Point_1[1] - point_2[1]
    dz = Point_1[2] - point_2[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    return dist


# 从开列表中找到成本最低的节点
def f_minf(open_set):
    fval_min = 100000000
    for node_t in open_set:
        if fval_min > node_t[5]:
            fval_min = node_t[5]
            node_r = node_t
    return node_r


def f_isinClose(point, close_set):
    for node_t in close_set:
        if node_t[0] == point[0] and node_t[1] == point[1] and node_t[2] == point[2]:
            return True
    return False


# 从闭集合中找到路径
def f_path_get(close_set):
    path = []
    node_c = close_set[-1]
    point = node_c[0:3]
    path.append(point)
    while node_c[6] > 0:
        node_c = close_set[node_c[6]]
        path.append(node_c[0:3])
    node_c = close_set[0]
    path.append(node_c[0:3])
    return path


def add_random_noise(value, noise_level=0.1):
    """添加随机噪声，确保输入值不为None"""
    if value is None:
        return 0.0
    return float(value) * (1 + noise_level * (random.random() * 2 - 1))


def calculate_reward(current_pos, next_pos, goal, env):
    """计算从current_pos到next_pos的移动奖励"""
    try:
        grid_size = 12
        max_distance = np.sqrt(3 * (grid_size) ** 2)
        
        # 确保输入是numpy数组
        current_pos = np.array(current_pos, dtype=np.float32)
        next_pos = np.array(next_pos, dtype=np.float32)
        goal = np.array(goal, dtype=np.float32)
        
        # 潜在奖励：距离变化
        prev_distance = np.linalg.norm(current_pos - goal)
        current_distance = np.linalg.norm(next_pos - goal)
        potential_reward = (prev_distance - current_distance) / max_distance
        
        # 方向奖励：移动方向与目标方向的一致性
        movement = next_pos - current_pos
        movement_norm = np.linalg.norm(movement)
        
        direction_reward = 0.0
        if movement_norm > 1e-8:
            direction = movement / movement_norm
            desired_direction = (goal - next_pos)
            desired_direction_norm = np.linalg.norm(desired_direction)
            if desired_direction_norm > 1e-8:
                desired_direction = desired_direction / desired_direction_norm
                direction_reward = float(np.dot(direction, desired_direction))
        
        # 总奖励
        reward = float(10 * potential_reward + 0.5 * direction_reward - 0.01)
        return reward
        
    except Exception as e:
        print(f"Error in calculate_reward: {e}")
        return 0.0  # 发生错误时返回默认值


def get_safe_distance(point, env):
    """计算点到最近障碍物的距离"""
    try:
        min_distance = float('inf')
        point = np.array(point, dtype=np.float32)
        
        for obs_start, obs_end in env:
            # 计算点到障碍物的最短距离
            x_min = float(min(obs_start[0], obs_end[0]))
            x_max = float(max(obs_start[0], obs_end[0]))
            y_min = float(min(obs_start[1], obs_end[1]))
            y_max = float(max(obs_start[1], obs_end[1]))
            z_min = float(min(obs_start[2], obs_end[2]))
            z_max = float(max(obs_start[2], obs_end[2]))
            
            dx = max(x_min - point[0], 0, point[0] - x_max)
            dy = max(y_min - point[1], 0, point[1] - y_max)
            dz = max(z_min - point[2], 0, point[2] - z_max)
            
            distance = float(np.sqrt(dx*dx + dy*dy + dz*dz))
            min_distance = min(min_distance, distance)
        
        return float(min_distance)  # 确保返回float类型
    except Exception as e:
        print(f"Error in get_safe_distance: {e}")
        return float('inf')  # 出错时返回一个安全的默认值


# 路径规划
def Astar(start, goal, env, randomization=0.1):
    """添加随机性的A*算法"""
    random.seed()
    open_set = []
    close_set = []
    delta = 1.0
    
    node_s = np.array(start, dtype=np.float32)
    h = 0.0
    g = float(f_distcal(start, goal))
    safe_distance = float(get_safe_distance(start, env))
    f = float(h + g)
    node_s = [*node_s, h, g, f, -1, 0.0, safe_distance]  # 确保所有数值都是float类型
    open_set.append(node_s)
    
    for _ in range(100000):
        if random.random() < 0.15:
            candidates = sorted(open_set, key=lambda x: x[5])[:3]
            if candidates:
                node_c = random.choice(candidates)
                open_set.remove(node_c)
        else:
            node_c = f_minf(open_set)
            if node_c is None:
                return None
            open_set.remove(node_c)
            
        close_set.append(node_c)
        current_pos = np.array(node_c[:3], dtype=np.float32)
        
        neighbors = f_neighbors(node_c, random_step=True)
        
        for point in neighbors:
            if f_isinClose(point, close_set):
                continue
                
            next_pos = np.array(point[:3], dtype=np.float32)
            
            if not f_is_safe(point, env):
                continue
            if not is_mid_safe(current_pos, next_pos, env, step=10):
                continue
            
            safe_distance = float(get_safe_distance(next_pos, env))
            
            base_reward = float(calculate_reward(current_pos, next_pos, goal, env))
            reward = float(add_random_noise(base_reward, randomization))
            
            # 确保safe_distance是float类型进行比较
            if safe_distance < 0.2:
                reward -= float((0.2 - safe_distance) * 1.0)
            
            accumulated_reward = float(node_c[7] + reward)
            
            h = float(node_c[3] + 1)
            g = float(f_distcal(point, goal))
            
            safety_weight = float(add_random_noise(0.2, randomization))
            reward_weight = float(add_random_noise(0.15, randomization))
            
            f = float(h + g - reward_weight * accumulated_reward - safety_weight * safe_distance)
            f = float(add_random_noise(f, randomization * 0.5))
            
            node_n = [*point, h, g, f, len(close_set) - 1, accumulated_reward, safe_distance]
            
            if g <= delta:
                close_set.append(node_n)
                path = f_path_get(close_set)
                return path
                
            open_set.append(node_n)
    
    return None







