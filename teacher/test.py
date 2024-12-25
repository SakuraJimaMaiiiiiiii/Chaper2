import math
import random
from args import get_args


# 判断一个点是否为障碍物
def f_is_safe(point_f, env_f):
    for [bos_s, obs_e] in env_f:
        if point_f[1] > bos_s[1] and point_f[2] > bos_s[2] and point_f[0] > bos_s[0]:
            if point_f[1] < obs_e[1] and point_f[2] < obs_e[2] and point_f[0] < obs_e[0]:
                return False
    return True


# def f_is_safe(point_f, env_f):
#     for (x1, y1, z1), (x2, y2, z2) in env_f:
#         x_min, x_max = min(x1, x2), max(x1, x2)
#         y_min, y_max = min(y1, y2), max(y1, y2)
#         z_min, z_max = min(z1, z2), max(z1, z2)
#         if (x_min <= point_f[0] <= x_max and
#                 y_min <= point_f[1] <= y_max and
#                 z_min <= point_f[2] <= z_max):
#             return True
#     return False



# 获得相邻节点
def f_neighbors(node):
    neighbors = []
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            for dz in [-1, 1]:
                neighbor = [node[0] + dx, node[1] + dy, node[2] + dz]
                if (0 <= neighbor[0] <= 400) and (0 <= neighbor[1] <= 400) and (
                        0 <= neighbor[2] <= 400):  # 假设网格大小为.....
                    neighbors.append(neighbor)
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
    fval_min = 100000
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


# 路径规划
def Astar(start, goal, env):
    # 开闭环集合
    open_set = []
    close_set = []
    # 初始化开始节点和开列表
    node_s = start
    node_s.append(0)  # h  3
    dist_g = f_distcal(start, goal)
    node_s.append(dist_g)  # g  4
    node_s.append(dist_g)  # f = g + h    5
    node_s.append(0)  # 父节点
    open_set.append(node_s)
    for id_run in range(10000):
        node_c = f_minf(open_set)
        open_set.remove(node_c)  # 移除该节点
        close_set.append(node_c)  # 加入闭环列表

        neighbors = f_neighbors(node_c)  # 周围节点
        for point in neighbors:
            if f_isinClose(point, close_set):  # 判断节点是否已经搜索过
                continue
            if not f_is_safe(point, env):  # 判断节点是否处于障碍物
                continue
            h = node_c[3] + 1
            g = f_distcal(point, goal)
            f = h + g
            len_set = len(close_set)

            node_n = point
            node_n.append(h)  # h  3
            node_n.append(g)  # g
            node_n.append(f)  # f
            node_n.append(len_set - 1)  # f
            open_set.append(node_n)  # 新节点添加到开列表

            if g == 0:
                close_set.append(node_n)  # 加入闭环列表
                path = f_path_get(close_set)
                return path


# 起点和终点

env3 = [
            [(0, 0, 0), (4, 1, 1)],
            [(0, 1, 0), (3, 2, 1)],
            [(1, 0, 1), (2, 1, 3)],

            [(1, 2, 0), (2, 4, 3)],
            [(1, 4, 1), (2, 5, 2)],
            [(1, 5, 0), (2, 8, 3)],
            [(2, 7, 0), (4, 8, 3)],
            [(4, 7, 1), (7, 8, 2)],
            [(6, 7, 2), (7, 8, 3)],

            [(4, 3, 0), (5, 6, 3)],
            [(5, 5, 0), (7, 6, 1)],
            [(7, 3, 0), (8, 6, 2)],

            [(7, 9, 0), (8, 10, 3)],
            [(8, 8, 0), (9, 10, 3)],
            [(9, 7, 0), (10, 10, 3)],
]
start = [11, 1, 1]
goal = [2, 10, 1]


env4 = [
            [(0, 0, 0), (1, 1, 1)],

            [(0, 2, 1), (1, 4, 2)],
            [(0, 3, 0), (1, 6, 1)],

            [(3, 0, 0), (4, 1, 2)],
            [(3, 1, 1), (4, 2, 3)],
            [(3, 2, 2), (7, 3, 3)],

            [(3, 5, 0), (4, 6, 3)],
            [(3, 6, 0), (4, 7, 2)],
            [(3, 9, 0), (4, 10, 2)],
            [(3, 10, 0), (4, 11, 3)],
            [(3, 11, 0), (4, 12, 1)],
            [(4, 10, 0), (5, 11, 2)],
            [(5, 10, 0), (6, 11, 1)],


            [(5, 0, 0), (6, 3, 1)],
            [(5, 1, 1), (6, 2, 3)],

            [(6, 2, 2), (7, 7, 3)],
            [(6, 4, 0), (8, 7, 2)],
            [(5, 7, 0), (7, 8, 3)],

            [(8, 0, 0), (9, 1, 2)],
            [(9, 0, 0), (10, 2, 1)],
            [(10, 1, 0), (11, 3, 1)],
            [(11, 2, 0), (12, 4, 1)],

            [(10, 6, 0), (11, 7, 2)],
            [(11, 11, 0), (12, 12, 1)],

            [(8, 11, 0), (9, 12, 3)],
            [(9, 11, 1), (10, 12, 2)],
]
start1=[1, 1, 2]
goal1 = [9, 5, 2]


env5 = [
            [(0, 2, 0), (1, 3, 3)],

            [(3, 0, 0), (4, 4, 3)],
            [(3, 4, 0), (4, 5, 1)],
            [(3, 5, 0), (4, 9, 3)],

            [(6, 0, 0), (7, 1, 3)],
            [(7, 0, 0), (8, 1, 1)],
            [(6, 1, 2), (7, 12, 3)],

            [(6, 4, 0), (8, 12, 1)],
            [(6, 4, 1), (8, 5, 2)],
            [(6, 6, 1), (8, 7, 2)],
            [(6, 8, 2), (12, 9, 3)],
            [(6, 8, 1), (7, 12, 2)],
            [(8, 8, 0), (12, 9, 1)],
            [(11, 8, 1), (12, 9, 2)],

            [(9, 11, 0), (11, 12, 1)],

]


start2 =[5, 3, 1]
goal3 = [1, 1, 0]