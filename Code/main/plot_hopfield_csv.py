import csv
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

logs_dir = Path("logs")
logs_dir = logs_dir / "hopfield" / "test_results"

# 1. 读取 CSV 文件
csv_file = logs_dir / "metrics_summary.csv"
data = pd.read_csv(csv_file)
# 2. 提取数据
node1_mae = data["Node 1 MAE"]
node2_mae = data["Node 2 MAE"]
node3_mae = data["Node 3 MAE"]
node1_rmse = data["Node 1 RMSE"]
node2_rmse = data["Node 2 RMSE"]
node3_rmse = data["Node 3 RMSE"]
# 3. 计算均值
node1_mae_mean = node1_mae.mean()
node2_mae_mean = node2_mae.mean()
node3_mae_mean = node3_mae.mean()
node1_rmse_mean = node1_rmse.mean()
node2_rmse_mean = node2_rmse.mean()
node3_rmse_mean = node3_rmse.mean()
# 4. 打印结果
print(f"Node 1 MAE Mean: {node1_mae_mean:.4f}")
print(f"Node 2 MAE Mean: {node2_mae_mean:.4f}")
print(f"Node 3 MAE Mean: {node3_mae_mean:.4f}")
print(f"Node 1 RMSE Mean: {node1_rmse_mean:.4f}")
print(f"Node 2 RMSE Mean: {node2_rmse_mean:.4f}")
print(f"Node 3 RMSE Mean: {node3_rmse_mean:.4f}")