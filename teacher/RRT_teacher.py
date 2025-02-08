import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from teacher.rrt_tools.rrt import RRT
from teacher.rrt_tools.rrt_star import RRTStar
from teacher.rrt_tools.rrt_star_bid import RRTStarBidirectional
from teacher.rrt_tools.search_space import SearchSpace
from teacher.rrt_tools.plotting import Plot
from env.environment import Environment
from args import get_args
import random
from utils.utils import convert_obstacles, calculate_path_length
import time


def RRT_teacher(args, plot=False):
    random.seed(args.seed)
    env = Environment(env_type=args.env_type)

    X_dimensions = np.array([(0, env.grid_size), (0, env.grid_size), (0, env.grid_size)])
    obstacles = np.array(convert_obstacles(env.obstacles))
    start = tuple(env.start)  # starting location
    goal = tuple(env.goal)  # goal location

    q = 0.5  # length of tree edges
    r = 0.1  # length of smallest edge to check for intersection with obstacles
    max_samples = 1024  # max number of samples to take before timing out
    prc = 0.1  # probability of checking for a connection to goal

    # print(f"\n{'=' * 20} 训练配置 {'=' * 20}")
    # print(f"\n算法: {args.teacher}")
    # print(f"环境: {args.env_type}")
    # print(f"随机种子: {args.seed}")
    # print(f"最大采样回合: {max_samples}")

    # create Search Space
    X = SearchSpace(X_dimensions, obstacles)

    # create rrt_search

    if args.teacher == 'RRT':
        rrt = RRT(X, q, start, goal, max_samples, r, prc)
    elif args.teacher == 'RRTStar':
        rrt = RRTStar(X, q, start, goal, max_samples, r, prc)
    elif args.teacher == 'RRTStarBidirectional':
        rrt = RRTStarBidirectional(X, q, start, goal, max_samples, r, prc)
    else:
        print('请选择RRT类算法!')

    path = rrt.rrt_search()

    # print(f"\n{'=' * 20} 路径长度 {'=' * 20} \n{calculate_path_length(path)}\n")

    if plot == True:
        # plot
        plot = Plot("rrt_3d")
        plot.plot_tree(X, rrt.trees)
        if path is not None:
            plot.plot_path(X, path)
        plot.plot_obstacles(X, obstacles)
        plot.plot_start(X, start)
        plot.plot_goal(X, goal)
        plot.draw(auto_open=False)

    return path


if __name__ == '__main__':
    args = get_args()

    start_time = time.time()
    path = RRT_teacher(args, plot=False)
    end_time = time.time()
    print(f"运行时间: {end_time - start_time:.2f} 秒")

    args = get_args()
    random.seed(args.seed)

    env = Environment(env_type=args.env_type)
    grid_size = env.grid_size

    X_dimensions = np.array([(0, grid_size), (0, grid_size), (0, grid_size)])
    obstacles = np.array(convert_obstacles(env.obstacles))
    start = tuple(env.start)  # starting location
    goal = tuple(env.goal)  # goal location



    q = 1  # length of tree edges
    r = 0.1  # length of smallest edge to check for intersection with obstacles
    max_samples = 1024  # max number of samples to take before timing out
    prc = 0.1  # probability of checking for a connection to goal

    algorithm = args.teacher


    print(f"\n{'=' * 20} 训练配置 {'=' * 20}")
    print(f"\n算法: {args.teacher}")
    print(f"环境: {args.env_type}")
    print(f"随机种子: {args.seed}")
    print(f"最大采样回合: {max_samples}")




    # create Search Space
    X = SearchSpace(X_dimensions, obstacles)

    # create rrt_search

    if args.teacher == 'RRT':
        rrt = RRT(X, q, start, goal, max_samples, r, prc)
    elif args.teacher == 'RRTStar':
        rrt = RRTStar(X, q, start, goal, max_samples, r, prc)
    elif args.teacher == 'RRTStarBidirectional':
        rrt = RRTStarBidirectional(X, q, start, goal, max_samples, r, prc)
    else:
        print('请选择RRT类算法')

    path = rrt.rrt_search()
    print(f"\n{'=' * 20} 路径长度 {'=' * 20} \n{calculate_path_length(path)}")


    # plot
    plot = Plot("rrt_3d")
    plot.plot_tree(X, rrt.trees)
    if path is not None:
        plot.plot_path(X, path)
    plot.plot_obstacles(X, obstacles)
    plot.plot_start(X, start)
    plot.plot_goal(X, goal)
    plot.draw(auto_open=False)
