# import matplotlib.pyplot as plt
# import numpy as np
# from pathlib import Path

# # ==========================================
# # 测试数据区（导师已为你准备好，无需修改，用于验证你的代码）
# # ==========================================
# DUMMY_NODES = [2, 3]
# DUMMY_RESULTS = {
#     2: {
#         "ppo_mean_error": np.exp(-np.linspace(0, 5, 200)) + np.random.normal(0, 0.01, 200),
#         "linear_error": np.exp(-np.linspace(0, 2, 200)) + np.random.normal(0, 0.05, 200),
#         "step_time": 0.05
#     },
#     3: {
#         "ppo_mean_error": np.exp(-np.linspace(0, 4, 200)) + np.random.normal(0, 0.02, 200),
#         "linear_error": np.exp(-np.linspace(0, 1.5, 200)) + np.random.normal(0, 0.08, 200),
#         "step_time": 0.05
#     }
# }
# DUMMY_PULSE_STEP = 50
# OUTPUT_PATH = Path("my_practice_plot.png")

# # ==========================================
# # 你的默写区：核心绘图逻辑
# # ==========================================
# def plot_comparison_curves(
#     node_results: dict, 
#     pulse_step: int, 
#     out_path: Path, 
#     nodes: list[int]
# ) -> None:
#     """
#     绘制 PPO 与 Linear 控制器的误差对比图。
#     """
    
#     # 1. 计算要绘制的节点数量（即子图的行数）
#     len_nodes = len(nodes)
#     # 2. 创建图表画布 (subplots)：
#     #    - 行数为节点数量，列数为 1
#     #    - 图表大小 figsize 设为 (9, 6.8)
#     #    - 开启共享 X 轴 (sharex)
#     #    - 接收返回值：fig 和 axes
#     fig,axes = plt.subplots(len_nodes,1,figsize=(9,6.8),sharex=True)

#     # 3. 防御性处理：如果节点数量为 1，axes 返回的是单个对象，请将其转换为单元素列表，以免后续 for 循环索引报错
#     if len_nodes == 1:
#         axes = [axes]
#     # 4. 初始化一个布尔变量 pulse_label_added 为 False，用于控制脉冲图例只添加一次
#     pulse_label_added = False
#     # 5. 使用 enumerate 遍历 nodes 列表（获取 索引 index 和 节点编号 node）:
#     for subplot_index, node in enumerate(nodes):
#         # 6. 根据 index 从 axes 中取出当前需要操作的坐标系 axis
#         axis = axes[subplot_index]
#         # 7. 从 node_results 字典中提取当前 node 的三个数据：
#         data=node_results[node] 
#         ppo_error = data["ppo_mean_error"]
#         linear_error = data["linear_error"]
#         step_time = data["step_time"]
#         #    - ppo_error (对应键 "ppo_mean_error")
#         #    - linear_error (对应键 "linear_error")
#         #    - step_time (对应键 "step_time")
        
#         # 8. 核心物理转换：
#         #    - 计算 time_axis：生成与 ppo_error 等长的序列，乘以 step_time
#         time_axis = np.arange(len(ppo_error))*step_time
#         #    - 计算 pulse_time：用 pulse_step 乘以 step_time
#         pulse_time = pulse_step*step_time
#         # 9. 在当前 axis 上画出 PPO 误差曲线（红色，线型自定，标签设为 "PPO"）
#         axis.plot(time_axis,ppo_error,color="red",linestyle=":",label="ppo")
#         # 10. 在当前 axis 上画出 Linear 误差曲线（蓝色，线型自定，标签设为 "Linear"）
#         axis.plot(time_axis,linear_error,color="blue",linestyle=":",label="linear")
#         # 11. 处理垂直脉冲线：如果 pulse_label_added 为 False:
#             # - 画一条垂直线 (axvline)，位置在 pulse_time，颜色设为紫色，加上图例标签 "Pulse"
#             # - 将 pulse_label_added 设为 True
#         # 12. 否则 (说明已经加过了):
#             # - 画同样的垂直线，但不加标签参数
#         if not pulse_label_added:
#             axis.axvline(pulse_time,linstyle=":",color="purple",label="pulse")
#             pulse_label_added = True
#         else:
#             axis.axvline(pulse_time,linstyle=":",color="purple",label="pulse")
#         # 13. 视觉优化：
#         #     - 给当前 Y 轴设置标签，包含节点信息，例如 f"Node {node} Error"
#         #     - 将当前 Y 轴设为对数刻度 ("log")
#         #     - 调用图例显示函数 (legend)
#         axis.set_ylabel(f"Node {node} Error")
#         axis.set_yscale("log")
#         axis.legend()
#     # 14. 循环结束后，拿到最后一个 axis（可以用 axes[-1]），为其设置 X 轴标签 "Time (s)"
#     axes[-1].set_xlabel("Time (s)")
#     # 15. 调用 tight_layout() 优化整体布局
#     fig.tight_layout()
#     # 16. 创建输出路径的父文件夹 (确保 parents 和 exist_ok 参数开启)
#     out_path.mkdir(parents=True,exist_ok=True)
#     # 17. 保存图片 fig.savefig 到 out_path
#     fig.savefig(out_path)
#     # 18. 关闭当前图表 fig，释放内存
#     print(f"图表已成功保存至：{out_path.absolute()}")

# # ==========================================
# # 运行入口
# # ==========================================
# if __name__ == "__main__":
#     plot_comparison_curves(DUMMY_RESULTS, DUMMY_PULSE_STEP, OUTPUT_PATH, DUMMY_NODES)




# import argparse
# from pathlib import Path
# import numpy as np

# # ==========================================
# # 桩代码区（Dummy Setup，导师已准备好，供你运行测试用）
# # ==========================================
# class DummyPPO:
#     @classmethod
#     def load(cls, path, device): return cls()
#     def predict(self, obs, deterministic=True): return np.array([0.5]), None

# class ContinuousHopfieldEnv:
#     def __init__(self, **kwargs):
#         self.B = np.array([0.1, 0.9, 0.2])  # 假设节点 1 的控制权重最大
#         self.statex = np.array([1.0, 2.0, 3.0])
#         self.statey = np.array([1.1, 1.8, 3.2])
#         self.scale = 10.0
#         self.dt = 0.01
#         self.rk4_steps = 5
#     def reset(self, seed=None): return np.array([0.1]), {}
#     def step(self, action): return np.array([0.1]), 0, True, False, {}
#     def close(self): pass

# def make_env(**kwargs): return ContinuousHopfieldEnv(**kwargs)

# def run_episode(policy_fn, env, seed):
#     # 模拟跑了一次 episode，返回假数据
#     return {
#         "error": np.random.rand(100, 3), # 100步，3个节点
#         "action": np.random.rand(100)    # 100步的标量动作
#     }

# # 测试用的 args
# args = argparse.Namespace(
#     mismatch=0.1, pulse_step=50, pulse_width=5, pulse_vector=[10.,10.,10.], seed=42
# )
# dummy_model_paths = [Path("model_1.zip"), Path("model_2.zip")]

# # ==========================================
# # 你的默写区 1：PPO 模型批量评估
# # ==========================================
# def evaluate_ppo_models(
#     model_paths: list[Path], args: argparse.Namespace, pinning_node: int
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    
#     # 1. 初始化两个空列表，用于收集 error_histories 和 action_curves
#     error_histories = []
#     action_curves = []
#     # 2. 初始化 step_time 为 None
#     step_time = None
#     # 3. 遍历 model_paths 中的每一个 model_path:
#     for model_path in model_paths:
#         # 4. 加载模型：调用 DummyPPO.load，传入路径的字符串格式，指定 device="cpu"
#         model = DummyPPO.load(model_path)
#         # 5. 定义内部策略函数 def ppo_policy(obs: np.ndarray, env: ContinuousHopfieldEnv) -> np.ndarray:
#             # 5.1 调用 model.predict 获取 action 和隐状态（忽略隐状态）
#             # 5.2 将 action 转换为 np.float32 类型的 numpy 数组并返回
#         def ppo_policy(obs,env):
#             action,_=model.predict(obs,deterministic=True)
#             return np.asarray(action,dtype=np.float32)
#         # 6. 调用 make_env 创建环境，传入 args 中的参数 (mismatch_scale, pulse_step, pulse_width, pulse_vector转tuple) 和 pinning_node
#         env=make_env(args,pinning_node)
#         # 7. 懒加载 step_time：如果 step_time 为 None，则计算并赋值为 env.dt * env.rk4_steps
#         if step_time == None:
#             step_time = env.dt * env.rk4_steps
#         # 8. 调用 run_episode 跑数据，传入刚刚定义的 ppo_policy, env, 以及 args.seed，拿到 result 字典
#         results=run_episode(ppo_policy,env,args.seed)
#         # 9. 关闭环境 env.close()
#         env.close()
#         # 10. 把 result 里的 "error" 和 "action" 分别追加到你初始化的列表中
#         error_histories.append(results['error'])
#         action_curves.append(results["action"])
#     # 11. 将收集到的 error_histories 列表转换为 NumPy 数组堆叠起来（提示：使用 np.stack，并在第 0 维度堆叠）
#     error_sum=np.stack(np.asarray(error_histories),0)
#     # 12. 同样地，把 action_curves 也 stack 起来
#     action_sum=np.stack(np.asarray(action_curves),0)
#     # 13. 返回一个元组：
#     #     - error 的均值（沿第 0 维度取平均，使用 np.mean）
#     #     - error 的标准差（沿第 0 维度取标准差，使用 np.std）
#     #     - action 的均值（沿第 0 维度取平均）
#     #     - step_time (转换为 float)
#     return tuple[
#         "error_mean":np.mean(error_sum),
#         "error_std":np.std(error_sum)

#     ]
# # ==========================================
# # 你的默写区 2：基线策略评估
# # ==========================================
# def evaluate_baseline(args: argparse.Namespace, mode: str, pinning_node: int) -> tuple[np.ndarray, np.ndarray]:
    
#     # 1. 直接调用 make_env 创建环境，参数同上
#     env=make_env(args,pinning_node)
#     # 2. 判断 mode 字符串：
#     # 2.1 如果 mode 是 "linear":
#         # 定义内部函数 def policy(obs: np.ndarray, env: ContinuousHopfieldEnv) -> np.ndarray:
#             # a. 找到 env.B 中最大值的索引（提示：使用 np.argmax），并转为 int 类型赋予 controlled_idx
#             # b. 计算误差 error：env.statey[controlled_idx] 减去 env.statex[controlled_idx]
#             # c. 计算动作 action：-8.0 乘以 error，再除以 env.scale
#             # d. 将 action 放入列表中转为 numpy 数组，然后使用 np.clip 裁剪到 -1.0 到 1.0 之间并返回
            
#     # 2.2 如果 mode 是 "zero":
#         # 定义内部函数 def policy(obs: np.ndarray, env: ContinuousHopfieldEnv) -> np.ndarray:
#             # 直接返回长度为 1 的全 0 numpy 数组（提示：使用 np.zeros，指定 dtype=np.float32）
            
#     # 2.3 否则（else）：
#         # 抛出 ValueError 异常，提示 "Unsupported baseline mode: " 加上传入的 mode

#     # 3. 调用 run_episode 跑数据，传入对应的 policy, env, 以及 args.seed
    
#     # 4. 关闭环境
    
#     # 5. 直接从 result 字典中提取 "error" 和 "action" 作为元组返回
#     pass # 删掉这行，开始写代码

# # ==========================================
# # 运行入口
# # ==========================================
# if __name__ == "__main__":
#     print("--- 正在测试 evaluate_ppo_models ---")
#     mean_err, std_err, mean_act, dt = evaluate_ppo_models(dummy_model_paths, args, pinning_node=1)
#     print(f"PPO Error Mean shape: {mean_err.shape}, DT: {dt}")

#     print("\n--- 正在测试 evaluate_baseline (Linear) ---")
#     base_err, base_act = evaluate_baseline(args, mode="linear", pinning_node=1)
#     print(f"Baseline Error shape: {base_err.shape}")


# from collections import defaultdict
# import numpy as np

# # ==========================================
# # 桩代码区（测试数据，无需修改）
# # ==========================================
# # 模拟 run_episode 返回的结果（50步，3个节点）
# dummy_result = {
#     "error": np.random.normal(0, 0.5, (50, 3)), # 误差会随时间衰减的假象
#     "action": np.random.normal(0, 0.2, (50, 1))
# }
# # 手动制造一个巨大的脉冲误差（第20步）和随后的收敛
# dummy_result["error"][20] = np.array([5.0, 5.0, 5.0]) 
# dummy_result["error"][25:] = np.random.normal(0, 0.05, (25, 3)) 

# # 模拟之前收集到的两行生数据记录
# dummy_records = [
#     {"method": "PPO", "mismatch": 0.5, "pulse_step": 20, "mae": 1.2, "rmse": 1.5},
#     {"method": "PPO", "mismatch": 0.5, "pulse_step": 20, "mae": 1.0, "rmse": 1.3},
#     {"method": "Linear", "mismatch": 0.5, "pulse_step": 20, "mae": 2.0, "rmse": 2.5}
# ]

# # ==========================================
# # 你的默写区 1：指标计算计算
# # ==========================================
# def compute_metrics(result: dict, step_time: float, pulse_step: int) -> dict:
#     error = result["error"]
#     action = result["action"]
    
#     # 1. 计算每一时间步的 L1 范数（绝对值之和）。
#     # 提示：使用 np.linalg.norm，指定 ord=1，并且沿着节点维度（axis=1）求和
#     # 最终 l1 是一个长度与时间步相同的一维数组
    
#     # 2. 组装基础指标字典 metrics：
#     #    - mae: error 的绝对值的全局平均值 (np.abs 然后 np.mean)
#     #    - rmse: error 平方的全局平均值，再开根号 (np.square, np.mean, np.sqrt)
#     #    - final_l1: l1 数组的最后一个元素
#     #    - energy_u: action 的平方的总和，乘以 step_time (np.square, np.sum)
#     # 注意：记得把所有的 numpy 标量用 float() 转换成纯 Python 浮点数
    
#     metrics = {} # 请用上面的逻辑覆盖这行代码

#     # 3. 截断出脉冲发生后的时间序列
#     # 提示：使用列表切片从 pulse_step 一直取到最后，赋值给 post_pulse

#     # 4. 找到脉冲后的最大误差，存入 metrics["max_post_pulse"]
    
#     # 5. 寻找系统恢复所需的时间步数
#     recovery_steps = -1
#     recovery_threshold = 0.1
#     # 遍历 post_pulse 的每一个索引 idx：
#         # 如果 post_pulse 从 idx 开始到结尾的所有元素，都小于 recovery_threshold：
#             # (提示：使用 np.all 和 切片 post_pulse[idx:] )
#             # 则将 recovery_steps 设为 idx
#             # 使用 break 跳出循环
            
#     metrics["recovery_steps"] = float(recovery_steps)
    
#     # 6. 如果 recovery_steps >= 0，计算 recovery_time (步数 * step_time)
#     # 否则，赋值为 float("nan")
    
#     return metrics

# # ==========================================
# # 你的默写区 2：数据聚合分组
# # ==========================================
# def aggregate_records(records: list[dict]) -> list[dict]:
#     # 定义需要求平均的数值型键列表
#     numeric_keys = ["mae", "rmse"] 
    
#     # 1. 初始化一个 defaultdict，它的默认值类型应该是 list
#     # group_map 的键将是 (method, mismatch, pulse_step) 的元组，值是包含对应记录的列表

#     # 2. 遍历 records 中的每一行 row:
#         # 提取 method (转为str), mismatch (转为float), pulse_step (转为int)
#         # 将这三个值打包成一个元组 key
#         # 把当前的 row 追加到 group_map[key] 这个列表中

#     summary_rows = []
#     # 3. 遍历 group_map 的 items()。为了输出稳定，请对 items() 使用 sorted() 进行排序
#     # (method, mismatch, pulse_step), rows in ...
        
#         # 4. 构建当前组的基础字典 summary，包含：
#         # "method", "mismatch", "pulse_step"，以及 "num_runs": len(rows)
        
#         # 5. 遍历 numeric_keys 中的每一个 key：
#             # 从 rows 的每一行中提取该 key 的值，收集到一个列表 values 中
#             # (提示：用列表推导式，同时需要判断 key 是否存在于 row 中)
            
#             # 如果 values 列表不为空：
#                 # 计算 values 的 np.mean，并将其转为 float，存入 summary[f"{key}_mean"] 中
                
#         # 6. 将构建好的 summary 追加到 summary_rows 列表中
        
#     return summary_rows

# # ==========================================
# # 运行入口
# # ==========================================
# if __name__ == "__main__":
#     print("--- 测试 compute_metrics ---")
#     m = compute_metrics(dummy_result, 0.05, 20)
#     print(f"MAE: {m.get('mae', '未计算'):.4f}")
#     print(f"Max Post-Pulse: {m.get('max_post_pulse', '未计算'):.4f}")
#     print(f"Recovery Steps: {m.get('recovery_steps', '未计算')}")
    
#     print("\n--- 测试 aggregate_records ---")
#     summaries = aggregate_records(dummy_records)
#     for s in summaries:
#         print(s)

# import re
# def aaa(model_path):
#     path =Path(model_path)
#     stem = path.stem
#     epoch_match=re.search("epoch_(\d+)",stem)
#     num=int(epoch_match.group(1)) if epoch_match else -1
#     st=re.search("lr_(\d+\.?\d*)",stem)
#     return {
#         "name":num,
#         "str":st.group(1) if st else -1,
#     }

# from collections import defaultdict
# import numpy as np

# # ==========================================
# # 桩代码区（导师已埋好“脏数据”地雷，用于检验你的清洗能力）
# # ==========================================
# dummy_records = [
#     # 正常数据
#     {"method": "PPO", "mismatch": 0.5, "pulse_step": 20, "mae": 1.2, "rmse": 1.5, "final_l1": 0.1},
#     # 脏数据 1：这行没有 "final_l1" 键 (缺失键)
#     {"method": "PPO", "mismatch": 0.5, "pulse_step": 20, "mae": 1.0, "rmse": 1.3},
#     # 脏数据 2：这行的 rmse 是 NaN (无效值)
#     {"method": "PPO", "mismatch": 0.5, "pulse_step": 20, "mae": 1.4, "rmse": np.nan, "final_l1": 0.2},
#     # 另一组数据
#     {"method": "Linear", "mismatch": 0.5, "pulse_step": 20, "mae": 2.0, "rmse": 2.5, "final_l1": 0.5}
# ]

# # ==========================================
# # 你的默写区：带数据清洗的聚合逻辑
# # ==========================================
# def aggregate_records(records: list[dict]) -> list[dict]:
    
#     # 1. 第一道防线：如果传入的 records 是空的，直接返回空列表 []
#     if not records:
#         return []
#     # 2. 定义你想要求均值的键列表 numeric_keys 
#     # (填入 "mae", "rmse", "final_l1")
#     numeric_keys = ["mae", "rmse","final_l1"]
#     # 3. 初始化 defaultdict，默认值为 list。将其命名为 group_map
#     group_map = defaultdict(list)
    
#     # 4. 第一遍遍历：按元组拆分数据
#     # 遍历 records 中的每一行 row:
#         # a. 提取 str 类型的 "method"，float 类型的 "mismatch"，int 类型的 "pulse_step"
#         # b. 将它们打包成一个元组 key
#         # c. 将当前的 row 字典追加到 group_map[key] 对应的列表中
#     for row in records:
#         method = str(row["method"])
#         mismatch = float(row["mismatch"])
#         pulse_step = int(row["pulse_step"])

#         key = (method,mismatch,pulse_step)
#         group_map[key].append(row)

#     # 5. 初始化用于存放结果的空列表 summary_rows
#     summary_rows = []
#     # 6. 第二遍遍历：数据清洗与求均值
#     # 遍历按照 key 排序后的 group_map.items() 
#     # (拆包为 (method, mismatch, pulse_step) 和 rows)
#     for (method,mismatch,pulse_step),rows in sorted(group_map.items()):
#         # a. 构建当前分组的初始 summary 字典：
#         #    包含 method, mismatch, pulse_step，以及 num_runs (值为 len(rows))
#         summary = {"method":method,"mismatch":mismatch,"pulse_step":pulse_step,"num_runs":len(rows)}
#         # b. 遍历 numeric_keys 中的每一个 key：
#         for metric_key in numeric_keys:
#             # c. 【核心挑战：数据清洗逻辑】
#             #    使用列表推导式，从 rows 中提取每一行 (记为 row) 的对应 key 的值。
#             #    提取条件必须同时满足两个：
#             #    1) key 必须存在于 row 中
#             #    2) 提取出来的值转换为 float 后，不能是 np.isnan
#             #    如果满足条件，将值转为 float 并收集到 values 列表中。
#             values = [float(row[key]) for row in rows if key in row and not np.isnan(float(row[key]))]
#             # d. 如果 values 列表非空 (因为如果全部是脏数据，列表可能为空)：
#                 # 计算 values 的 np.mean，转换为 float 
#                 # 存入 summary 字典中，键名为 key 加上 "_mean" 的后缀 (例如 "mae_mean")
#             if values:
#                 summary[f"{metric_key}_mean"]=float(np.mean(values))
#         # e. 将组装好的 summary 字典追加到 summary_rows 列表中
#         summary_rows.append(summary)

#     # 7. 返回 summary_rows
#     return summary_rows


# # ==========================================
# # 运行入口
# # ==========================================
# if __name__ == "__main__":
#     print("--- 测试带数据清洗的 aggregate_records ---")
#     summaries = aggregate_records(dummy_records)
#     for s in summaries:
#         print(s)
        
#     # 自测标准：
#     # PPO 组的 mae_mean 应该是 (1.2+1.0+1.4)/3 = 1.2
#     # PPO 组的 rmse_mean 应该只有两个有效值参与计算：(1.5+1.3)/2 = 1.4 (NaN必须被跳过)
#     # PPO 组的 final_l1_mean 应该只有两个有效值参与计算：(0.1+0.2)/2 = 0.15 (缺失键必须被跳过)

from collections import defaultdict
import numpy as np
nums = [2, 5, 2, 8, 5, 2, 9, 8, 5, 2]
count_dict=defaultdict(int)

for num in nums:
    count_dict[num]+=1

print(dict(count_dict))

employees = [
    ("技术部", "张三", 15000),
    ("市场部", "李四", 12000),
    ("技术部", "王五", 18000),
    ("市场部", "赵六", 10000),
    ("人事部", "钱七", 9000),
    ("技术部", "孙八", 16000)
]

department_dict=defaultdict(list)

for department,employee,salary in employees:
    department_dict[department].append(employee)

print(dict(department_dict))

salary_dict=defaultdict(set)

for department,employee,salary in employees:
    salary_dict[department].add(salary)

print(dict(salary_dict))