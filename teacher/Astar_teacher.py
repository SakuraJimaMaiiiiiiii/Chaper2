from teacher.astar_tools.Astar import Astar
from args import get_args
from env.environment import Environment
from teacher.rrt_tools.plotting import Plot
from teacher.rrt_tools.search_space import SearchSpace
import numpy as np
from utils.utils import convert_obstacles
import time
import random
from utils.utils import calculate_path_length


start_time = time.time()
args = get_args()
random.seed(args.seed)
env = Environment(env_type=args.env_type)
obstacle = env.obstacles
start = (env.start).tolist()
goal = (env.goal).tolist()


print(f"\n{'=' * 20} 训练配置 {'=' * 20}")
print(f"\n算法: {args.teacher}")
print(f"环境: {args.env_type}")
print(f"随机种子: {args.seed}")




path = Astar(start, goal, obstacle)
path = list(map(tuple, path))

end_time = time.time()


print(f"\n{'=' * 20} 路径长度 {'=' * 20} \n{calculate_path_length(path)}\n")
print(f"运行时间: {end_time - start_time:.2f} 秒")





# plot
grid_size = env.grid_size
X_dimensions = np.array([(0, grid_size), (0, grid_size), (0, grid_size)])
obstacles = np.array(convert_obstacles(env.obstacles))
X = SearchSpace(X_dimensions, obstacles)


plot = Plot("Astar")
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles(X, obstacles)
plot.plot_start(X, start)
plot.plot_goal(X, goal)
plot.draw(auto_open=False)