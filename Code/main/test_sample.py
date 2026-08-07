import numpy as np
import matplotlib.pyplot as plt

# 1. 假设这是你训练的步数 (X轴)
steps = np.array([0, 10000, 20000, 30000, 40000, 50000])

# 2. 假设这是你 3 个不同 Seed 跑出来的评估分数 (Y轴)
seed_1_rewards = np.array([10, 30, 60, 80, 95, 100])
seed_2_rewards = np.array([12, 25, 40, 75, 90, 98])
seed_3_rewards = np.array([8,  40, 70, 85, 92, 105])

# 将 3 个种子的数据堆叠成一个二维矩阵 (3行, 6列)
all_reward = np.array([seed_1_rewards,seed_2_rewards,seed_3_rewards])

# 3. 计算均值 (mean) 和 标准差 (std)
# axis=0 表示沿着"种子"所在的维度求平均，也就是求每一列的平均值
mean_reward = np.mean(all_reward,axis=0)
std_reward = np.std(all_reward,axis=0)

# 4. 开始画图
plt.figure(figsize=(8,5))

# 画中间的均值实线
plt.plot(steps,mean_reward,color="blue",label="ppo mean")

# 画阴影区域（核心代码就是这一句）
# y1 是下界 (mean - std), y2 是上界 (mean + std)
# alpha 是透明度，通常设为 0.2 到 0.3
plt.fill_between(steps,mean_reward-2*std_reward,mean_reward+2*std_reward,color="blue",alpha=0.2)

# 5. 美化图表 (发论文的标准格式)
plt.title("Learning Curve with Multiple Seeds")
plt.xlabel("Environment Steps")
plt.ylabel("Environment Steps")
plt.legend("Environment Steps")
plt.grid(True,alpha=0.6)
plt.tight_layout()

plt.show()
