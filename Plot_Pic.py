import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# 设置字体后端
import matplotlib
matplotlib.use('TKAgg')

# 数据文件路径列表，每个子列表表示一个算法，包含多个随机种子的结果
env_name_ = 'env3'
data_dirs = {
    "ppo": [
        f'./finalresult/savemodel/{env_name_}/ppo/training_data_seed22.json',
        f'./finalresult/savemodel/{env_name_}/ppo/training_data_seed32.json',
        f'./finalresult/savemodel/{env_name_}/ppo/training_data_seed42.json',
    ],
    # "td3": [
    #     f'./finalresult/savemodel/{env_name_}/td3/training_data_seed22.json',
    #     f'./finalresult/savemodel/{env_name_}/td3/training_data_seed32.json',
    #     f'./finalresult/savemodel/{env_name_}/td3/training_data_seed42.json',
    # ],
    "td3_HER": [
        f'./finalresult/savemodel/{env_name_}/td3_HER/training_data_seed22.json',
        f'./finalresult/savemodel/{env_name_}/td3_HER/training_data_seed32.json',
        f'./finalresult/savemodel/{env_name_}/td3_HER/training_data_seed42.json',
    ],
    "td3_PER": [
        f'./finalresult/savemodel/{env_name_}/td3_PER/training_data_seed22.json',
        f'./finalresult/savemodel/{env_name_}/td3_PER/training_data_seed32.json',
        f'./finalresult/savemodel/{env_name_}/td3_PER/training_data_seed42.json',
    ],

    "DAgger": [
        f'./student/models/{env_name_}/DAgger/seed_0_rewards.json',
        f'./student/models/{env_name_}/DAgger/seed_42_rewards.json',
        f'./student/models/{env_name_}/DAgger/seed_2023_rewards.json',
    ],
    "Expert-Driven GAIL": [
        f'./student/models/{env_name_}/GAIL/seed_0_rewards.json',
        f'./student/models/{env_name_}/GAIL/seed_42_rewards.json',
        f'./student/models/{env_name_}/GAIL/seed_2023_rewards.json',
    ],
    "BC": [
        f'./student/models/{env_name_}/BC/seed_0_rewards.json',
        f'./student/models/{env_name_}/BC/seed_42_rewards.json',
        f'./student/models/{env_name_}/BC/seed_2023_rewards.json',
    ],
}

# 数据平滑函数
def smooth(data, weight=0.9):
    smoothed = []
    last = data[0]  # 初始值
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# 设置绘图风格
sns.set(style='whitegrid', font_scale=1.2)
plt.figure(figsize=(12, 8))

# 自动生成颜色
colors = sns.color_palette("husl", len(data_dirs)-1)
colors.append('red')
labels = ['ppo','td3_HER','td3_PER','DAgger','BC','Expert-Driven GAIL']


# 遍历每个算法的数据
for idx, (algorithm, file_list) in enumerate(data_dirs.items()):
    all_rewards = []

    # 遍历每个随机种子的文件
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 假设每个文件的 "rewards" 是一个列表
        rewards = data["rewards"][:100]
        smoothed_rewards = smooth(rewards)
        all_rewards.append(smoothed_rewards)

    # 统一长度：裁剪为最短长度
    min_length = min(len(r) for r in all_rewards)
    all_rewards_trimmed = [r[:min_length] for r in all_rewards]

    # 计算均值和标准差
    all_rewards_array = np.array(all_rewards_trimmed)  # 转为NumPy数组
    mean_rewards = np.mean(all_rewards_array, axis=0)  # 计算均值
    std_rewards = np.std(all_rewards_array, axis=0)    # 计算标准差

    len_reward = range(1, len(mean_rewards) + 1)

    # 绘制每个算法的均值曲线
    sns.lineplot(
        x=len_reward,
        y=mean_rewards,
        color=colors[idx],
        label=f"{labels[idx]}",
        linewidth=2,
        zorder = 3

    )

    # 添加阴影表示标准差范围
    plt.fill_between(
        len_reward,
        np.array(mean_rewards) - np.array(std_rewards),
        np.array(mean_rewards) + np.array(std_rewards),
        color=colors[idx],
        alpha=0.2,
        label=None,
        zorder = 2
    )

# 设置图例和标签
plt.ylabel("Reward", fontsize=25)
plt.xlabel("Iterations ", fontsize=25)
plt.xticks(fontsize=20)  # 调整 X 轴刻度字体大小
plt.yticks(fontsize=20)  # 调整 Y 轴刻度字体大小

# 添加 x 轴和 y 轴的坐标线

# plt.axhline(y=-250, color='black', linewidth=1.5, linestyle='-')  # 添加水平轴线
# plt.axvline(x=0, color='black', linewidth=1.5, linestyle='-')  # 添加竖直轴线


# 显示横纵坐标网格线
plt.grid(which='major', axis='both', linestyle='--', linewidth=0.8, color='gray')  # 主网格线
plt.grid(which='minor', axis='both', linestyle=':', linewidth=0.5, color='gray')  # 次网格线（可选）

# 将图例放在右下角
plt.legend(loc='lower right', fontsize=19, bbox_to_anchor=(1, 0.03),  framealpha=0)

# 调整布局
# plt.tight_layout()
plt.ylim(-210, 20)  # 示例范围，需匹配你的实际数据
plt.xlim(0, 101)    # 确保x轴从0开始
plt.subplots_adjust(
    left=0.1,    # 左边距
    right=0.95,  # 右边距
    bottom=0.1,  # 下边距
    top=0.95     # 上边距
)

for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')  # 设置边框颜色
    spine.set_linewidth(2)  # 设置边框线宽


# 保存图像
plt.savefig(f'./final_adjusted_plot_{env_name_}.pdf', dpi=600, bbox_inches='tight')



