import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from curvature_plot3d import compute_total_curvature,Plot3D

def Astar_teacher(args, Plot_=False):
    random.seed(int(time.time() * 1000))
    env = Environment(env_type=args.env_type)
    obstacle = env.obstacles
    start = (env.start).tolist()
    goal = (env.goal).tolist()
    args.teacher = 'Astar'

    # print(f"\n{'=' * 20} 训练配置 {'=' * 20}")
    # print(f"\n算法: {args.teacher}")
    # print(f"环境: {args.env_type}")
    # print(f"随机种子: {args.seed}")

    if args.teacher == 'Astar':
        path = Astar(start, goal, obstacle)
        path = list(map(tuple, path))
    else:
        print('请选择Astar算法')

    # print(f"\n{'=' * 20} 路径长度 {'=' * 20} \n{calculate_path_length(path)}\n")

    if Plot_== True:
        # plot
        grid_size = env.grid_size
        X_dimensions = np.array([(0, grid_size), (0, grid_size), (0, grid_size)])
        obstacles = np.array(convert_obstacles(env.obstacles))
        X = SearchSpace(X_dimensions, obstacles)

        plot = Plot("Astar_noc")
        if path is not None:
            plot.plot_path(X, path)
        plot.plot_obstacles(X, obstacles)
        plot.plot_start(X, start)
        plot.plot_goal(X, goal)
        plot.draw(auto_open=False)

    return path


if __name__ == '__main__':
    args = get_args()
    args.seed = 2026
    args.env_type = 'env5'
    env = Environment(env_type=args.env_type)
    random.seed(args.seed)
    start_time = time.time()
    path = Astar_teacher(args, Plot_=True)

    '''
    保存路径
    '''
    dir = r'E:\files\code\硕士论文code\Chaper2'
    file_path = f"{dir}/Apath.txt"
    with open(file_path, 'w') as f:
        for point in path:
            f.write(f"{point}\n")
    print(f'路径已保存至{file_path}')


    end_time = time.time()

    print(f"环境: {args.env_type}")
    print(f"随机种子: {args.seed}")
    print(f"\n{'=' * 20} 路径长度 {'=' * 20} \n{calculate_path_length(path)}\n")
    print(f"\n{'=' * 20} 路径曲率 {'=' * 20} \n{compute_total_curvature(path)}")
    print(f"运行时间: {end_time - start_time:.2f} 秒")

    Plot3D(path, env.obstacles, env, args, algorithm='Astar')




#     start_time = time.time()
#     args = get_args()
#     random.seed(args.seed)
#     env = Environment(env_type=args.env_type)
#     obstacle = env.obstacles
#     start = (env.start).tolist()
#     goal = (env.goal).tolist()
#     args.seed =42
#
#     print(f"\n{'=' * 20} 训练配置 {'=' * 20}")
#     print(f"\n算法: {args.teacher}")
#     print(f"环境: {args.env_type}")
#     print(f"随机种子: {args.seed}")
#
#
#
#
#     path = Astar(start, goal, obstacle)
#     path = list(map(tuple, path))
#
#     end_time = time.time()
#
#
#     print(f"\n{'=' * 20} 路径长度 {'=' * 20} \n{calculate_path_length(path)}\n")
#     print(f"运行时间: {end_time - start_time:.2f} 秒")
# #
#
#
#
#
# # plot
# grid_size = env.grid_size
# X_dimensions = np.array([(0, grid_size), (0, grid_size), (0, grid_size)])
# obstacles = np.array(convert_obstacles(env.obstacles))
# X = SearchSpace(X_dimensions, obstacles)
#
#
# plot = Plot("Astar")
# if path is not None:
#     plot.plot_path(X, path)
# plot.plot_obstacles(X, obstacles)
# plot.plot_start(X, start)
# plot.plot_goal(X, goal)
# plot.draw(auto_open=False)
