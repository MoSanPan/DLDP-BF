import math
import random
import hashlib
import numpy as np
import pandas as pd
from collections import Counter

def classify(num):
    if 30 < num <= 40:
        return 4
    elif 40 < num <= 50:
        return 5
    elif 50 < num <= 60:
        return 6
    elif 60 < num <= 70:
        return 7
    elif 70 < num <= 80:
        return 8
    elif 80 < num <= 90:
        return 9
    elif 90 < num <= 100:
        return 10
    elif 100 < num <= 110:
        return 11
    elif 110 < num <= 120:
        return 12
    elif 120 < num <= 130:
        return 13
    elif 130 < num <= 140:
        return 14
    else:
        return 15  # 如果超出范围，返回 -1

# 哈希函数，用于计算哈希值
def get_hash_position(value, hash_count):
    positions = []
    for i in range(hash_count):
        hash_value = int(hashlib.md5((str(value) + str(i)).encode('utf-8')).hexdigest(), 16)
        position = hash_value % m  # 取模操作确保在布隆过滤器范围内
        positions.append(position)
    return positions

# 位数组大小和元素总数
m = 10000  # 位数组大小
n = 1000  # 总的元素数量
alpha = 0.5
k_hash = math.ceil((m/n)*math.log(2))

# ===================== 读取数据 =====================
data_start = pd.read_excel('Obesity.xlsx')

# 强制将数据转换为数值型，无法转换的变为 NaN
data_start['Weight'] = pd.to_numeric(data_start['Weight'], errors='coerce')

# 处理数值类型的分类
data = data_start['Weight'].apply(lambda x: classify(x) if pd.notna(x) else -1)

S = data.iloc[1: n+1].tolist()   # 插入数据集

# ===================== 查询数据集 =====================
# Zipf分布的排名（0到8，包含9个值）
# Zipf分布的排名（0到8，包含9个值）
values = np.array([4,5,6,7,8,9,10,11,12,13,14])  # Now 9 elements
ranks = np.arange(1, len(values) + 1)  # Adjust ranks to have 9 elements

# 设置Zipf分布的参数，s为幂指数
s = 1  # 常见的Zipf参数为1
frequencies = 1 / ranks**s  # 根据Zipf分布公式计算频率

# 归一化频率（确保概率总和为1）
frequencies /= frequencies.sum()

# Check that lengths of values and frequencies are the same
assert len(values) == len(frequencies), "Mismatch between values and frequencies length!"

# 使用这些概率生成大小为100的数据集
query_data = np.random.choice(values, size=n, p=frequencies)

Q = query_data.tolist()

print(len(Q))
print(len(S))
print("插入数据集 S:", S)
print("查询数据集 Q:", Q)


# 定义 epsilon 的范围
epsilon_values = [2, 4, 6, 8, 10]

# 存储每个 epsilon 对应的平均 RMSE 和 MAE
epsilon_avg_rmse = []
epsilon_avg_mae = []

# 遍历 epsilon 的值
for epsilon in epsilon_values:
    print(f"\n当前 epsilon: {epsilon}")

    # 存储当前 epsilon 的 100 次实验 RMSE 和 MAE
    rmse_list = []
    mae_list = []

    # 重复 100 次实验
    for _ in range(100):
        # 初始化布隆过滤器
        bloom_filter = [0] * m

        # 计算概率
        prob_keep = np.exp(epsilon) / (np.exp(epsilon) + 1)

        # 对每个插入的元素，根据对应的哈希函数个数计算并更新布隆过滤器的位置
        for element in S:
            positions = get_hash_position(element, k_hash)  # 获取哈希值并计算对应位置
            for pos in positions:
                bloom_filter[pos] = 1  # 首先置为1

        # 判断bloom_filter是否需要翻转
        for i in range(len(bloom_filter)):
            if np.random.rand() > prob_keep:
                bloom_filter[i] = 1 - bloom_filter[i]  # 以概率 (1-p) 报告翻转数据


        # 计算真实值和预测值
        actual_values = [1 if item in S else 0 for item in Q]
        predicted_values = [False] * len(Q)

        # 遍历 Q 中的每个元素
        for idx, element in enumerate(Q):

            positions = get_hash_position(element, k_hash)  # 获取哈希值并计算对应位置

            # 检查每个哈希位置
            all_positions_are_one = True
            for pos in positions:
                if bloom_filter[pos] != 1:
                    all_positions_are_one = False
                    break  # 如果某个位置不为 1，直接跳出循环

            # 更新 predicted_values
            if all_positions_are_one:
                predicted_values[idx] = True  # 设置为 True

        # 计算 RMSE
        def calculate_rmse(actual, predicted):
            mse = np.mean((np.array(actual) - np.array(predicted)) ** 2)
            return np.sqrt(mse)

        rmse = calculate_rmse(actual_values, predicted_values)
        rmse_list.append(rmse)

        # 将 actual_values 和 predicted_values 转换为 NumPy 数组
        actual_values = np.array(actual_values)
        predicted_values = np.array(predicted_values)

        # 计算 MAE
        mae = np.mean(np.abs(actual_values - predicted_values))
        mae_list.append(mae)

    # 计算当前 epsilon 的平均 RMSE 和 MAE
    avg_rmse = np.mean(rmse_list)
    epsilon_avg_rmse.append(avg_rmse)

    avg_mae = np.mean(mae_list)
    epsilon_avg_mae.append(avg_mae)

    # 输出当前 epsilon 的平均 RMSE 和 MAE
    print(f"epsilon = {epsilon}: 平均 RMSE = {avg_rmse:.4f}")
    print(f"epsilon = {epsilon}: 平均 MAE = {avg_mae:.4f}")

# 输出所有 epsilon 对应的平均 RMSE 和 MAE
print("\n所有 epsilon 对应的RAPPOR平均 RMSE 和 MAE:")
for epsilon, avg_rmse, avg_mae in zip(epsilon_values, epsilon_avg_rmse, epsilon_avg_mae):
    print(f"epsilon = {epsilon}: 平均 RMSE = {avg_rmse:.4f}, 平均 MAE = {avg_mae:.4f}")