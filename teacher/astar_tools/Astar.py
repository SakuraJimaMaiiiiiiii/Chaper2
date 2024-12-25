import math
import numpy as np



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
def f_neighbors(node):
    neighbors = []
    temp_x = node[0]
    temp_y = node[1]
    temp_z = node[2]
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbor = [temp_x + dx, temp_y + dy, temp_z + dz]
                if neighbor[0] <= 0 or neighbor[1] <= 0 or neighbor[2] <= 0:
                    continue
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
    for id_run in range(100000):
        node_c = f_minf(open_set)
        open_set.remove(node_c)  # 移除该节点
        close_set.append(node_c)  # 加入闭环列表

        neighbors = f_neighbors(node_c)  # 周围节点                      # 插入一条路径 看是否在障碍物里
        for point in neighbors:
            if f_isinClose(point, close_set):  # 判断节点是否已经搜索过
                continue
            if not f_is_safe(point, env):  # 判断节点是否处于障碍物
                continue
            if not is_mid_safe(node_c[:3], point[:3], env, step=10):
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

            if g <= 0.5:
                close_set.append(node_n)  # 加入闭环列表
                path = f_path_get(close_set)
                return path







