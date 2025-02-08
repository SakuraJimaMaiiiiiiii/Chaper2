import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from env.obstacles import env1, env2, env3, env4, env5

class Environment(gym.Env):
    def __init__(self, render_mode=False, env_type='env3'):
        super(Environment, self).__init__()

        rcParams['font.sans-serif'] = ['SimHei']
        rcParams['axes.unicode_minus'] = False

        # 根据环境类型设置起点终点和障碍物
        if env_type == 'env3':
            self.start = np.array([11, 1, 1], dtype=np.float32)
            self.goal = np.array([2, 10, 1], dtype=np.float32)
            self.obstacles = env3
            self.grid_size = 12

        elif env_type == 'env4':
            self.start = np.array([1, 1, 2], dtype=np.float32)
            self.goal = np.array([9, 5, 2], dtype=np.float32)
            self.obstacles = env4
            self.grid_size = 12

        elif env_type == 'env5':
            self.start = np.array([5, 3, 1], dtype=np.float32)
            self.goal = np.array([1, 1, 0], dtype=np.float32)
            self.obstacles = env5
            self.grid_size = 12

        elif env_type == 'env1':
            self.start = np.array([1, 1, 1], dtype=np.float32)
            self.goal = np.array([29, 29, 29], dtype=np.float32)
            self.obstacles = env1
            self.grid_size = 30

        elif env_type == 'env2':
            self.start = np.array([3, 15, 3], dtype=np.float32)
            self.goal = np.array([27, 15, 27], dtype=np.float32)
            self.obstacles = env2
            self.grid_size = 30


        self.position = self.start.copy()
        self.delta = 1.0  # 达到目标的距离阈值
        self.render_mode = render_mode
        self.max_steps = 200
        self.steps_taken = 0

        # 动作空间范围-2.0, 2.0
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32)     # shape: (3,)  shape[0]=3

        # 观测空间
        self.num_sensors = 12  # 模拟距离传感器
        self.sensor_range = 10.0
        self.max_distance = np.sqrt(3 * (self.grid_size) ** 2)    # 对角线长度 空间最远距离
        # 状态表示，归一化向量和标量距离
        obs_low = np.array([-1.0]*3 + [0.0]*1 + [-1.0]*3 + [0.0] + [0.0]*self.num_sensors, dtype=np.float32)
        obs_high = np.array([1.0]*3 + [1.0]*1 + [1.0]*3 + [1.0] + [1.0]*self.num_sensors, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)    # shape:(20,)

        # 传感器方向
        self.sensor_directions = np.array([
            [1, 0, 0],    # 正 x 方向
            [-1, 0, 0],   # 负 x 方向
            [0, 1, 0],    # 正 y 方向
            [0, -1, 0],   # 负 y 方向
            [0, 0, 1],    # 正 z 方向
            [0, 0, -1],   # 负 z 方向
            [1, 1, 0],    # x 和 y 正方向
            [1, -1, 0],   # x 正，y 负
            [-1, 1, 0],   # x 负，y 正
            [-1, -1, 0],  # x 和 y 负方向
            [0, 1, 1],    # y 和 z 正方向
            [0, -1, -1],  # y 和 z 负方向
        ], dtype=np.float32)

        # 归一化传感器方向     np.linalg.norm(vec)  计算向量的范数(长度) len = √x2+y2+z2
        self.sensor_directions = np.array([vec / np.linalg.norm(vec) for vec in self.sensor_directions])

        self.fig = None
        self.ax = None
        self.agent_plot = None

        # 代理路径
        self.path = []
        


    def reset(self):
        # 重置位置、步数和路径
        self.position = self.start.copy()
        self.steps_taken = 0
        self.path = [self.position.copy()]
        if self.render_mode:
            self._init_render()
        current_distance = np.linalg.norm(self.position - self.goal)    # len = √(x1-x2)2+(y1-y2)2+(z1-z2)2
        relative_position = (self.goal - self.position) / self.grid_size  # 相对位置 len/grid_size
        normalized_distance = current_distance / self.max_distance  # 归一化距离 len/√3grid_size  范围：(0,1)
        direction_to_goal = (self.goal - self.position) / (np.linalg.norm(self.goal - self.position) + 1e-8)  # 归一化方向向量 position -> goal
        abs_distance = current_distance   # 绝对距离
        sensor_readings = self._get_sensor_readings() / self.sensor_range  # 传感器读数
        # 构建观测
        state = np.concatenate((relative_position, [normalized_distance], direction_to_goal, [abs_distance], sensor_readings))
        return state

    def step(self, action):
        # 确保动作在动作空间内
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # 更新步数
        self.steps_taken += 1
        action = np.squeeze(action)

        # 记录之前的位置和距离
        previous_position = self.position.copy()
        previous_distance = np.linalg.norm(self.position - self.goal)

        # 用动作更新位置
        self.position += action
        self.position = np.clip(self.position, 0, self.grid_size - 1)

        # 碰撞标志
        collision = False

        # 沿着移动路径进行碰撞检测
        movement = self.position - previous_position
        movement_length = np.linalg.norm(movement)

        if movement_length > 0:
            num_samples = int(np.ceil(movement_length / 0.1))
            t_values = np.linspace(0, 1, num_samples + 1)         # 采样间隔 [0~1]
            for i, t in enumerate(t_values):
                interp_position = previous_position + t * movement
                
                in_obstacle = False
                for (x1, y1, z1), (x2, y2, z2) in self.obstacles:
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    z_min, z_max = min(z1, z2), max(z1, z2)
                    if (x_min <= interp_position[0] <= x_max and
                        y_min <= interp_position[1] <= y_max and
                        z_min <= interp_position[2] <= z_max):
                        in_obstacle = True
                        break
                if in_obstacle:
                    collision = True
                    # 将位置设置为碰撞前的最后安全位置
                    if i == 0:
                        # 在起始点就发生碰撞，保持原位
                        self.position = previous_position
                    else:
                        # 上一个安全位置
                        self.position = previous_position + t_values[i-1] * movement
                    break
        else:
            # 没有移动
            self.position = previous_position

      
        self.path.append(self.position.copy())

        '''
        奖励设计
        '''
        # 当前位置与目标的距离
        current_distance = np.linalg.norm(self.position - self.goal)

        # 奖励函数
        # 潜在奖励使用距离的差值作为潜在函数
        potential_reward = (previous_distance - current_distance) / self.max_distance

        # 密集奖励鼓励朝向目标的移动方向
        direction = (self.position - previous_position) / (np.linalg.norm(self.position - previous_position) + 1e-8)
        desired_direction = (self.goal - self.position) / (np.linalg.norm(self.goal - self.position) + 1e-8)
        direction_reward = np.dot(direction, desired_direction)        # <1

        # 总奖励
        reward = 10 * potential_reward + 0.5 * direction_reward           #  < 1

        
        if collision:
            reward -= 1.0  # 碰撞惩罚

        # 检查是否到达目标
        if current_distance < self.delta:
            reward += 5.0  # 到达目标的奖励
            done = True
        else:
            done = False

        # 最大步数限制
        if self.steps_taken >= self.max_steps:
            done = True

        # 获取传感器
        sensor_readings = self._get_sensor_readings() / self.sensor_range

        # 避障惩罚
        min_sensor_reading = np.min(sensor_readings)
        if min_sensor_reading < 0.2:
            reward -= (0.2 - min_sensor_reading) * 1.0  # 离障碍物太近的惩罚

        # 时间惩罚
        reward -= 0.01

        
        info = {'distance_to_goal': current_distance / self.max_distance, 'collision': collision}

        # 计算相对位置、归一化距离、方向向量和标量距离
        relative_position = (self.goal - self.position) / self.grid_size
        normalized_distance = current_distance / self.max_distance
        direction_to_goal = (self.goal - self.position) / (np.linalg.norm(self.goal - self.position) + 1e-8)
        scalar_distance = current_distance / self.max_distance

        # 构建观测
        state = np.concatenate((relative_position, [normalized_distance], direction_to_goal, [scalar_distance], sensor_readings))

        return state, reward, done, info

    def _get_sensor_readings(self):
        # 传感器读数函数
        readings = []
        for direction in self.sensor_directions:
            min_distance = self.sensor_range
            for i in np.linspace(0, self.sensor_range, num=10):
                probe_pos = self.position + direction * i
                if (probe_pos < 0).any() or (probe_pos > self.grid_size).any():
                    min_distance = i
                    break  # 超出边界
                is_obstacle = False
                for (x1, y1, z1), (x2, y2, z2) in self.obstacles:
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

    def _init_render(self):
      
        if self.fig is None or self.ax is None:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_zlim(0, self.grid_size)
            self.ax.set_xlabel('X 轴')
            self.ax.set_ylabel('Y 轴')
            self.ax.set_zlabel('Z 轴')

          
            self.ax.scatter(*self.start, color='green', s=20, label='起点')
            self.ax.scatter(*self.goal, color='red', s=20, label='终点')

           
            for (x1, y1, z1), (x2, y2, z2) in self.obstacles:
                self._draw_obstacle(x1, y1, z1, x2, y2, z2)

            self.ax.legend()

    def _draw_obstacle(self, x1, y1, z1, x2, y2, z2):
        # 障碍物
        x = [x1, x2]
        y = [y1, y2]
        z = [z1, z2]
        xx, yy = np.meshgrid(x, y)

    
        self.ax.plot_surface(xx, yy, np.full_like(xx, z1), color='gray', alpha=0.5)
        self.ax.plot_surface(xx, yy, np.full_like(xx, z2), color='gray', alpha=0.5)

        yy, zz = np.meshgrid(y, z)
        self.ax.plot_surface(np.full_like(yy, x1), yy, zz, color='gray', alpha=0.5)
        self.ax.plot_surface(np.full_like(yy, x2), yy, zz, color='gray', alpha=0.5)

        xx, zz = np.meshgrid(x, z)
        self.ax.plot_surface(xx, np.full_like(xx, y1), zz, color='gray', alpha=0.5)
        self.ax.plot_surface(xx, np.full_like(xx, y2), zz, color='gray', alpha=0.5)

    def render(self, mode='human'):
       
        if not self.render_mode:
            return

        if self.fig is None or self.ax is None:
            self._init_render()
        else:
            self.ax.clear()
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_zlim(0, self.grid_size)
            self.ax.set_xlabel('X 轴')
            self.ax.set_ylabel('Y 轴')
            self.ax.set_zlabel('Z 轴')

          
            self.ax.scatter(*self.start, color='green', s=20, label='起点')
            self.ax.scatter(*self.goal, color='red', s=20, label='终点')

        
            for (x1, y1, z1), (x2, y2, z2) in self.obstacles:
                self._draw_obstacle(x1, y1, z1, x2, y2, z2)

           
            path_array = np.array(self.path)
            self.ax.plot3D(path_array[:, 0], path_array[:, 1], path_array[:, 2], color='black', label='路径')
            self.ax.scatter(*self.position, color='black', s=20)

            self.ax.legend()

        plt.draw()
        plt.pause(0.01)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


if __name__ == '__main__':
    env = Environment(render_mode=True,env_type='env3')
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample() 
        obs, reward, done, info = env.step(action)
        env.render()
    env.close()
