import math
import random
import hashlib
import numpy as np
import pandas as pd
from collections import Counter

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
k_hash = math.ceil((m/n)*math.log(2))
print(k_hash)
# ===================== 读取数据 =====================
data = pd.read_excel('Car.xlsx')

S = data.iloc[1: n, 0].tolist()  # 插入数据集

# ===================== 查询数据集 =====================
# Zipf分布的排名（0到8，包含9个值）
values = np.array([1, 3, 5, 2, 4, 0, 6, 8, 9, 7])  # 对应的9个值
ranks = np.arange(1, 11)  # 排名 1 到 9

# 设置Zipf分布的参数，s为幂指数
s = 1  # 常见的Zipf参数为1
frequencies = 1 / ranks**s  # 根据Zipf分布公式计算频率

# 归一化频率（确保概率总和为1）
frequencies /= frequencies.sum()

# 设置随机种子，确保每次生成相同的数据集
np.random.seed(42)  # 你可以选择任意数字作为种子

# 使用这些概率生成大小为100的数据集
query_data = np.random.choice(values, size=n, p=frequencies)

Q = query_data.tolist()

print("插入数据集 S:", S)
print("查询数据集 Q:", Q)


# 定义 epsilon 的范围
epsilon_values = [2, 4, 6, 8, 10]

# 存储每个 epsilon 对应的平均 RMSE 和 ACC
epsilon_avg_rmse = []
epsilon_avg_acc = []

# 遍历 epsilon 的值
for epsilon in epsilon_values:
    print(f"\n当前 epsilon: {epsilon}")

    # 存储当前 epsilon 的 100 次实验 RMSE
    rmse_list = []
    acc_list = []

    # 重复 100 次实验
    for _ in range(100):
        # 初始化布隆过滤器
        bloom_filter = [0] * m

        # 计算概率
        # 对每个插入的元素，根据对应的哈希函数个数计算并更新布隆过滤器的位置
        for element in S:
            positions = get_hash_position(element, k_hash)  # 获取哈希值并计算对应位置
            for pos in positions:
                bloom_filter[pos] = 1  # 首先置为1

        # # 判断bloom_filter是否需要翻转
        # prob_keep = np.exp(epsilon) / (np.exp(epsilon) + 1)
        # for i in range(len(bloom_filter)):
        #     if np.random.rand() > prob_keep:
        #         bloom_filter[i] = 1 - bloom_filter[i]  # 以概率 (1-p) 报告翻转数据

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


        def calculate_accuracy(actual, predicted):
            correct = np.sum(np.array(actual) == np.array(predicted))
            return correct / len(actual)


        rmse = calculate_rmse(actual_values, predicted_values)
        rmse_list.append(rmse)

        acc = calculate_accuracy(actual_values, predicted_values)
        acc_list.append(acc)

    # 计算当前 epsilon 的平均 RMSE 和 ACC
    avg_rmse = np.mean(rmse_list)
    epsilon_avg_rmse.append(avg_rmse)

    avg_acc = np.mean(acc_list)
    epsilon_avg_acc.append(avg_acc)

    # 输出当前 epsilon 的结果
    print(f"epsilon = {epsilon}: 平均 RMSE = {avg_rmse:.4f},  平均 ACC = {avg_acc:.4f}")

# 输出所有 epsilon 对应的平均 RMSE 和 MAE
print("\n所有 epsilon 对应的RAPPOR平均 RMSE :")
for epsilon, avg_rmse, avg_acc in zip(epsilon_values, epsilon_avg_rmse, epsilon_avg_acc):
    print(f"epsilon = {epsilon}: 平均 RMSE = {avg_rmse:.4f},  平均 ACC = {avg_acc:.4f}")