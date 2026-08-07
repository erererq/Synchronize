import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

TIME_PER_STEP = 20 / 400  # 400 steps are approximately 20 s.

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 13,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

def plot_multiple_csvs():
    # 1. 定义你的文件路径和它们对应的图例名称 (Legend)
    # 请根据你实际的文件名修改这里
    csv_sources = {
        "Obs 2": Path("logs/hopfield/node2_obs2_step_by_step_data.csv"),
        "Obs 4": Path("logs/hopfield/node2_obs4_step_by_step_data.csv"),
        "Obs 6": Path("logs/hopfield/node2_obs6_step_by_step_data.csv")
    }

    # 2. 【核心架构：数据融合】
    df_list = []
    for label, path in csv_sources.items():
        if path.exists():
            df = pd.read_csv(path)
            # 关键魔法：在原表里强行塞入一列新数据，打上标签
            # 这样拼接到一起后，我们就知道每一行数据到底来自哪个文件了
            df["source"] = label 
            df_list.append(df)
        else:
            print(f"⚠️ 警告：找不到文件 {path}")
            
    if not df_list:
        print("❌ 致命错误：所有文件均未找到！")
        return

    # 用 concat 将列表里的所有 DataFrame 垂直拼成一个超级大表
    final_df = pd.concat(df_list, ignore_index=True)
    final_df["time_s"] = final_df["step"] * TIME_PER_STEP

    # 3. 准备画布
    plt.figure(figsize=(8, 5))

    # 4. 【降维打击：Seaborn 自动聚合绘图】
    # 我们只需告诉它：X轴是 step，Y轴是 error_l1
    # hue="source" 的意思是：请根据 "source" 这一列的不同值，自动帮我拆分成不同颜色的线！
    sns.lineplot(
        data=final_df,
        x="time_s",
        y="error_l1",
        # errorbar=("pi", 100),
        hue="source",   # <--- 自动拆分 3 条线的灵魂参数
        # linewidth=2
    )

    # 5. 视觉优化
    plt.yscale("log") # 误差下降曲线通常需要对数坐标系
    plt.xlabel("Time/s")
    plt.ylabel("L1 Error (Mean ± 95%CI)")
    # plt.title("Error Comparison Across Different Observations")
    
    # 将图例放到合适的位置，避免遮挡曲线
    plt.legend(title="Observation Type", loc="best")
    plt.grid(True, linestyle=":", alpha=0.6)

    # 6. 导出图片
    out_path = Path("logs/hopfield/comparison_plot.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.savefig("logs/hopfield/comparison_plot.pdf")
    print(f"✅ 三线对比图已成功生成：{out_path.absolute()}")

    # ==========================================
    # 论文表格数据提取区 (直接接在画图代码后面)
    # ==========================================
    print("\n📊 正在计算论文表格所需的核心指标...")

    # 1. 计算 MAE (Mean Absolute Error)
    # 物理意义：将该模型所有 step、所有 seed 的 error_l1 全部取平均
    mae_stats = final_df.groupby("source")["error_l1"].mean()
    print("\n=== 全局 MAE (越小越好) ===")
    print(mae_stats)

    # 2. 计算 Final L1 Error
    # 物理意义：只看收敛后的最终状态（即 step 达到最大值的那一步）的平均误差
    max_step = final_df["step"].max() # 找出总共有多少步
    
    # 极其优雅的布尔过滤：只把最后一步的数据捞出来，然后再分组求均值
    final_step_df = final_df[final_df["step"] == max_step]
    final_l1_stats = final_step_df.groupby("source")["error_l1"].mean()
    
    print(f"\n=== 最终步 (Step={max_step}) L1 Error ===")
    print(final_l1_stats)

if __name__ == "__main__":
    plot_multiple_csvs()
