import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# 设置字体后端
import matplotlib
matplotlib.use('TKAgg')

# 数据文件路径列表，每个子列表表示一个算法，包含多个随机种子的结果
data_dirs = {
    "ppo": [
        './finalresult/savemodel/env3/ppo/training_data_seed22.json',
        './finalresult/savemodel/env3/ppo/training_data_seed32.json',
        './finalresult/savemodel/env3/ppo/training_data_seed42.json',
    ],
    "td3": [
        './finalresult/savemodel/env3/td3/training_data_seed22.json',
        './finalresult/savemodel/env3/td3/training_data_seed32.json',
        './finalresult/savemodel/env3/td3/training_data_seed42.json',
    ],
    "td3_HER": [
        './finalresult/savemodel/env3/td3_HER/training_data_seed22.json',
        './finalresult/savemodel/env3/td3_HER/training_data_seed32.json',
        './finalresult/savemodel/env3/td3_HER/training_data_seed42.json',
    ],
    "td3_PER": [
        './finalresult/savemodel/env3/td3_PER/training_data_seed22.json',
        './finalresult/savemodel/env3/td3_PER/training_data_seed32.json',
        './finalresult/savemodel/env3/td3_PER/training_data_seed42.json',
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
sns.set(style='darkgrid', font_scale=1.2)
plt.figure(figsize=(12, 8))

# 自动生成颜色
colors = sns.color_palette("husl", len(data_dirs))
labels = list(data_dirs.keys())

# 遍历每个算法的数据
for idx, (algorithm, file_list) in enumerate(data_dirs.items()):
    all_rewards = []

    # 遍历每个随机种子的文件
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 假设每个文件的 "rewards" 是一个列表
        rewards = data["rewards"]
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
        linewidth=2
    )

    # 添加阴影表示标准差范围
    plt.fill_between(
        len_reward,
        np.array(mean_rewards) - np.array(std_rewards),
        np.array(mean_rewards) + np.array(std_rewards),
        color=colors[idx],
        alpha=0.2,
        label=None
    )

# 设置图例和标签
plt.ylabel("Reward", fontsize=14)
plt.xlabel("Iteration steps", fontsize=14)
plt.title("Performance Comparison Across Algorithms", fontsize=16)

# 显示横纵坐标网格线
plt.grid(which='major', axis='both', linestyle='--', linewidth=0.8, color='gray')  # 主网格线
plt.grid(which='minor', axis='both', linestyle=':', linewidth=0.5, color='gray')  # 次网格线（可选）

# 将图例放在右下角
plt.legend(loc='lower right', fontsize=12)

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('./final_adjusted_plot.png', dpi=300)



