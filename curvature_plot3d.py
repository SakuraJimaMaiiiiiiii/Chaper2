import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
import ast


def compute_total_curvature(path):
    total_curvature = 0.0

    # 遍历路径中的每个小段（相邻点对）
    for i in range(1, len(path) - 1):
        # 获取相邻的三个点
        p1 = np.array(path[i - 1])
        p2 = np.array(path[i])
        p3 = np.array(path[i + 1])

        # 计算切向量 (p2 - p1) 和 (p3 - p2)
        T1 = p2 - p1
        T2 = p3 - p2

        # 计算加速度 (T2 - T1)
        A = T2 - T1

        # 计算局部曲率
        cross_product = np.linalg.norm(np.cross(T1, A))
        tangent_length = np.linalg.norm(T1)
        curvature = cross_product / (tangent_length ** 3)

        # 计算小段的弧长
        delta_s = np.linalg.norm(p2 - p1)

        # 将小段曲率乘以弧长，加到总曲率上
        total_curvature += curvature * delta_s

    return total_curvature


def show_obs(ax, x1, y1, z1, x2, y2, z2):
    xx, yy = np.meshgrid([x1, x2], [y1, y2])

    ax.plot_surface(xx, yy, np.full_like(xx, z1), color='gray', alpha=0.3)
    ax.plot_surface(xx, yy, np.full_like(xx, z2), color='gray', alpha=0.3)

    yy, zz = np.meshgrid([y1, y2], [z1, z2])
    ax.plot_surface(np.full_like(yy, x1), yy, zz, color='gray', alpha=0.3)
    ax.plot_surface(np.full_like(yy, x2), yy, zz, color='gray', alpha=0.3)

    xx, zz = np.meshgrid([x1, x2], [z1, z2])
    ax.plot_surface(xx, np.full_like(xx, y1), zz, color='gray', alpha=0.3)
    ax.plot_surface(xx, np.full_like(xx, y2), zz, color='gray', alpha=0.3)


def show_obs2(ax, coords, plane='xy'):
    (x1, y1, z1), (x2, y2, z2) = coords

    if plane == 'xy':
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='gray', alpha=0.3)
    elif plane == 'xz':
        rect = Rectangle((x1, z1), x2 - x1, z2 - z1, facecolor='gray', alpha=0.3)
    elif plane == 'yz':
        rect = Rectangle((y1, z1), y2 - y1, z2 - z1, facecolor='gray', alpha=0.3)
    ax.add_patch(rect)


def Plot3D(path_points, obstacles, env, args, algorithm):
    save_dir = f"Plot_result/{args.env_type}"
    os.makedirs(save_dir, exist_ok=True)

    path_points = np.array(path_points)

    fig = plt.figure(figsize=(20, 15))
    ax1 = fig.add_subplot(111, projection='3d')
    # ax1 = fig.add_subplot(221, projection='3d')

    for (p1, p2) in obstacles:
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        show_obs(ax1, x1, y1, z1, x2, y2, z2)

    ax1.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], 'b-', label='path', linewidth=5)
    ax1.scatter(path_points[0, 0], path_points[0, 1], path_points[0, 2],
                color='g', s=150, label='start')
    ax1.scatter(path_points[-1, 0], path_points[-1, 1], path_points[-1, 2],
                color='r', s=150, label='goal')
    ax1.set_xlabel('X-axis', fontsize=20)
    ax1.set_ylabel('Y-axis', fontsize=20)
    ax1.set_zlabel('Z-axis', fontsize=20)
    ax1.set_title('3DPlot', fontsize=40)
    ax1.set_xlim(0, env.grid_size)
    ax1.set_ylim(0, env.grid_size)
    ax1.set_zlim(0, env.grid_size)
    ax1.legend(fontsize=30)

    # ax2 = fig.add_subplot(222)
    #
    # for coords in obstacles:
    #     show_obs2(ax2, coords, plane='xy')
    # ax2.plot(path_points[:, 0], path_points[:, 1], 'b-', linewidth=2)
    # ax2.scatter(path_points[0, 0], path_points[0, 1], color='g', s=100, label='start')
    # ax2.scatter(path_points[-1, 0], path_points[-1, 1], color='r', s=100, label='goal')
    # ax2.set_xlabel('X-axis')
    # ax2.set_ylabel('Y-axis')
    # ax2.set_title('Top view (X-Y plane)')
    # ax2.grid(True)
    # ax2.set_xlim(0, env.grid_size)
    # ax2.set_ylim(0, env.grid_size)
    # ax2.legend()
    #
    # ax3 = fig.add_subplot(223)
    #
    # for coords in obstacles:
    #     show_obs2(ax3, coords, plane='xz')
    # ax3.plot(path_points[:, 0], path_points[:, 2], 'b-', linewidth=2)
    # ax3.scatter(path_points[0, 0], path_points[0, 2], color='g', s=100, label='start')
    # ax3.scatter(path_points[-1, 0], path_points[-1, 2], color='r', s=100, label='goal')
    # ax3.set_xlabel('X-axis')
    # ax3.set_ylabel('Z-axis')
    # ax3.set_title('Main view (X-Z plane)')
    # ax3.grid(True)
    # ax3.set_xlim(0, env.grid_size)
    # ax3.set_ylim(0, env.grid_size)
    # ax3.legend()
    #
    # ax4 = fig.add_subplot(224)
    #
    # for coords in obstacles:
    #     show_obs2(ax4, coords, plane='yz')
    # ax4.plot(path_points[:, 1], path_points[:, 2], 'b-', linewidth=2)
    # ax4.scatter(path_points[0, 1], path_points[0, 2], color='g', s=100, label='start')
    # ax4.scatter(path_points[-1, 1], path_points[-1, 2], color='r', s=100, label='goal')
    # ax4.set_xlabel('Y-axis')
    # ax4.set_ylabel('Z-axis')
    # ax4.set_title('Side view (Y-Z plane)')
    # ax4.grid(True)
    # ax4.set_xlim(0, env.grid_size)
    # ax4.set_ylim(0, env.grid_size)
    # ax4.legend()

    plt.tight_layout()

    save_path = f"{save_dir}/{algorithm}_{args.seed}.pdf"
    print(f'图像位置:{save_path}')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    return save_dir


def Plot_multi_3D(path_point1, path_point2, path_point3, path_point4, obstacles, env, args, algorithm='multi'):
    save_dir = f"Plot_result/{args.env_type}"
    os.makedirs(save_dir, exist_ok=True)

    path_points = np.array(path_point1)

    fig = plt.figure(figsize=(20, 15))
    ax1 = fig.add_subplot(111, projection='3d')
    # ax1 = fig.add_subplot(221, projection='3d')
    colors = ['m','g', 'b', 'r']
    labels = ['RRT', 'ppo', 'BC','Expert-Driven GAIL']

    path_list = [path_point1, path_point2, path_point3, path_point4]

    for (p1, p2) in obstacles:
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        show_obs(ax1, x1, y1, z1, x2, y2, z2)

    for i, path in enumerate(path_list):
        path_points = np.array(path)
        ax1.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], color=colors[i], label=labels[i], linewidth=5)



    ax1.scatter(np.array(path_point1)[0][0], np.array(path_point1)[0][1], np.array(path_point1)[0][2],
                color='g', s=150)
    ax1.scatter(np.array(path_point1)[-1][0], np.array(path_point1)[-1][1], np.array(path_point1)[-1][2],
                color='r', s=150)

    # env3
    # ax1.text(np.array(path_point1)[0][0]+0.5, np.array(path_point1)[0][1] + 0.5, np.array(path_point1)[0][2] + 0.5, 'Start', color='g', fontsize=25, fontweight='black')
    # ax1.text(np.array(path_point1)[-1][0]-1, np.array(path_point1)[-1][1]-0.5, np.array(path_point1)[-1][2] + 1, 'Goal', color='r', fontsize=25, fontweight='black')

    # env4
    # ax1.text(np.array(path_point1)[0][0] - 0.5, np.array(path_point1)[0][1] + 0.5, np.array(path_point1)[0][2] + 2.3,
    #          'Start', color='g', fontsize=25, fontweight='black')
    # ax1.text(np.array(path_point1)[-1][0] + 0.2, np.array(path_point1)[-1][1] + 0.4, np.array(path_point1)[-1][2] - 1,
    #          'Goal', color='r', fontsize=25, fontweight='black')

    # env5
    ax1.text(np.array(path_point1)[0][0] - 0.5, np.array(path_point1)[0][1] + 0.5, np.array(path_point1)[0][2] + 1.5,
             'Start', color='g', fontsize=25, fontweight='black')
    ax1.text(np.array(path_point1)[-1][0]-0.1, np.array(path_point1)[-1][1] + 0.4, np.array(path_point1)[-1][2] - 1,
             'Goal', color='r', fontsize=25, fontweight='black')


    ax1.set_xlabel('X-axis', fontsize=30)
    ax1.set_ylabel('Y-axis', fontsize=30)
    ax1.set_zlabel('Z-axis', fontsize=30)
    ax1.set_xlim(0, env.grid_size)
    ax1.set_ylim(0, env.grid_size)
    ax1.set_zlim(0, env.grid_size)
    ax1.legend(fontsize=30,framealpha=0,bbox_to_anchor=(0.94, 0.84))
    plt.tight_layout()

    save_path = f"{save_dir}/{algorithm}_{args.seed}.pdf"
    print(f'图像位置:{save_path}')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    return save_dir


def read_mult_txt(path_load):
    loaded_path = []
    for path in path_load:
        with open(path, 'r', encoding='utf-8') as f:
            loaded_path.append([ast.literal_eval(line.strip()) for line in f if line.strip()])  # 改为列表解析
    return loaded_path


if __name__ == '__main__':
    # # 测试路径
    # path =[(2.147781020127806, 9.647277545930477, 1.5531299672255348), (2.847905228256904, 9.647277545930477, 2.351905283084594), (2.847905228256904, 8.943842871687215, 3.196881914635459), (3.5570586503522255, 8.113167591489011, 3.99905813124809), (4.289857128527018, 7.3305083083921385, 3.99905813124809), (5.121849591950753, 6.690204315257715, 4.92976111043934), (6.024844464231268, 5.742215979266127, 4.1471050201073405), (6.941909879931881, 4.818407499446699, 4.1471050201073405), (7.72926180579055, 4.059703857271457, 3.3062811175539424), (8.489836899929838, 3.4100265095477376, 2.572473099608617), (9.258202222680435, 2.614466934949541, 1.7356381832462098), (10.178347201970144, 1.7680749685747852, 1.7356381832462098), (11.0, 1.0, 1.0)]
    #
    # total_curvature = compute_total_curvature(path)
    # print(f"Total Curvature: {total_curvature}")
    from args import get_args
    from env.environment import Environment
    from utils.utils import convert_obstacles, calculate_path_length
    import ast

    args = get_args()
    args.env_type = 'env5'
    env = Environment(env_type=args.env_type)
    grid_size = env.grid_size
    obstacles = np.array(convert_obstacles(env.obstacles))


    # env3
    # path1 = r'E:\files\code\硕士论文code\Chaper2\finalresult\3d_result\env3\txt\RRTpath.txt'
    # path2 = r'E:\files\code\硕士论文code\Chaper2\finalresult\3d_result\env3\txt\Dagger.txt'
    # path3 = r'E:\files\code\硕士论文code\Chaper2\finalresult\3d_result\env3\txt\GAIL.txt'
    # path4 = r'E:\files\code\硕士论文code\Chaper2\finalresult\3d_result\env3\txt\ppo.txt'

    # #env4
    # path1 = r'E:\files\code\硕士论文code\Chaper2\finalresult\3d_result\env4\txt\RRTpath.txt'
    # path2 =  r'E:\files\code\硕士论文code\Chaper2\finalresult\3d_result\env4\txt\ppo.txt'
    # path3 = r'E:\files\code\硕士论文code\Chaper2\finalresult\3d_result\env4\txt\GAIL.txt'
    # path4 = r'E:\files\code\硕士论文code\Chaper2\finalresult\3d_result\env4\txt\Dagger.txt'

    # env5
    path1 = r'E:\files\code\硕士论文code\Chaper2\finalresult\3d_result\env5\txt\RRTpath.txt'
    path2 =  r'E:\files\code\硕士论文code\Chaper2\finalresult\3d_result\env5\txt\Dagger.txt'
    path3 = r'E:\files\code\硕士论文code\Chaper2\finalresult\3d_result\env5\txt\GAIL.txt'
    path4 = r'E:\files\code\硕士论文code\Chaper2\finalresult\3d_result\env5\txt\Apath.txt'

    path_load = [path1, path2, path3, path4]

    load_path = read_mult_txt(path_load)
    #
    Plot_multi_3D(load_path[0], load_path[1], load_path[2], load_path[3], env.obstacles, env, args)
