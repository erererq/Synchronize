import csv
from pathlib import Path

# 模拟深度学习训练后产生的一系列字典记录 (List of Dictionaries)
dl_experiment_logs = [
    {"model": "ResNet50", "epoch": 1, "loss": 0.85, "accuracy": 65.2},
    {"model": "ResNet50", "epoch": 2, "loss": 0.45, "accuracy": 82.1},
    {"model": "ResNet50", "epoch": 3, "loss": 0.21, "accuracy": 91.5}
]

def save_metrics_to_csv(records: list[dict], out_file: Path) -> None:
    """
    你的默写区：将 DL 实验记录保存为 CSV
    """
    # 1. 第一道防线：如果 records 是空的，直接 return
    if not records:
        return []
    # 2. 提取所有的列名（表头）。
    # 提示：提取 records 列表中第一行（第一个字典）的所有键，转为 list
    # fieldnames = ________
    fieldnames = list(records[0].keys())
    # 3. 创建输出路径的父文件夹 (parents=True, exist_ok=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    # 4. 使用 with 语句打开 out_file。
    # 关键参数设定：mode="w", newline="", encoding="utf-8"
    # (如果不写 newline=""，在 Windows 上每行数据之间会多出一个空行)
    # with open(______) as f:
    with open(out_file, mode="w", newline="", encoding="utf-8") as f:
        # 5. 实例化 csv.DictWriter，传入文件句柄 f 和刚刚拿到的表头 fieldnames
        # writer = ________
        writer = csv.DictWriter(f,fieldnames=fieldnames)
        # 6. 调用 writer 写入表头（列名）
        # writer.________()
        writer.writeheader()
        # 7. 调用 writer 一次性写入所有的多行数据
        # writer.________(records)
        writer.writerows(records)
    print(f"✅ 深度学习指标已存盘：{out_file.absolute()}")

if __name__ == "__main__":
    output_path = Path("results/resnet_metrics.csv")
    save_metrics_to_csv(dl_experiment_logs, output_path)

import csv
from pathlib import Path

def load_metrics_from_csv(in_file: Path) -> list[dict]:
    """
    你的默写区：从 CSV 安全地加载 DL 指标
    """
    # 1. 第一道防线：检查 in_file 是否存在 (.exists())，不存在则抛出 FileNotFoundError
    if not in_file.exists():
        raise FileNotFoundError
    loaded_records = []
    
    # 2. 使用 with 语句打开文件。mode="r", encoding="utf-8"
    # with open(______) as f:
    with open(in_file,mode="r",encoding="utf-8") as f:
        # 3. 实例化 csv.DictReader，传入文件句柄 f
        # reader = ________
        reader = csv.DictReader()
        # 4. 遍历 reader 中的每一行数据 (此时拿到的每一行已经是一个字典)
        # for row in reader:
        for row in reader:
            # 5. 【核心陷阱：类型清洗】CSV 读出来的全是字符串！
            # 请构建一个新的字典 cleaned_row，对数据进行类型还原：
            # "model" 保持原样 (字符串)
            # "epoch" 转为 int
            # "loss" 和 "accuracy" 转为 float
            # cleaned_row = {
            #     "model": row["model"],
            #     "epoch": ______,
            #     "loss": ______,
            #     "accuracy": ______
            # }
            cleaned_row={
                "model":row["model"],
                "epoch":int(row["epoch"]),
                "loss":float(row["loss"]),
                "accuracy":float(row["accuracy"])
            }
            # 6. 将 cleaned_row 追加到 loaded_records 列表中
            loaded_records.append(cleaned_row)
    # 7. 返回 loaded_records
    return loaded_records

def plot_dl_curves(records: list[dict]) -> None:
    print("\n成功加载 CSV 数据：")
    for r in records:
        print(f"Epoch {r['epoch']} | Loss: {r['loss']:.2f} (类型: {type(r['loss']).__name__})")
    print("✅ 画图所需的数据已准备完毕（注意观察它们的类型是否正确恢复）！")

if __name__ == "__main__":
    input_path = Path("results/resnet_metrics.csv")
    data = load_metrics_from_csv(input_path)
    plot_dl_curves(data)