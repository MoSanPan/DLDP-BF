import random
import hashlib
import numpy as np
import math
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

# 计算 k_i
# def calculate_k_star(frequencies, expected_values, m, n):
#     k_star = {}
#     for element in frequencies:
#         xi = frequencies[element]  # 当前元素的频率
#         ri = expected_values[element]  # 当前元素的期望值
#
#         k_avg = (m / n) * math.log(2)
#         print(math.ceil(k_avg))
#
#         # 计算共享项 term3
#         term3 = 0
#         for j in frequencies:
#             if(expected_values[j] == 0 or frequencies[j] == 0):
#                 result_mid = 0
#             else:
#                 result_mid = (frequencies[j] / n) * math.log2(expected_values[j] / frequencies[j])
#             term3 += result_mid
#
#         # print(term3)
#         # 根据公式计算 k_i
#         k_i = k_avg + math.log2(ri / xi) - term3
#         print(k_i)
#         # 对 k_i 上取整
#         k_star[element] = round(k_i)
#     return k_star

def calculate_k_star(frequencies, expected_values, m, n):
    k_star = {}
    for element in frequencies:
        xi = frequencies[element]  # 当前元素的频率
        ri = expected_values.get(element, 0)  # 当前元素的期望值, 默认值为 0

        k_avg = (m / n) * math.log(2)
        print(math.ceil(k_avg))

        # 计算共享项 term3
        term3 = 0
        for j in frequencies:
            # 确保 j 同时出现在 frequencies 和 expected_values 中
            if expected_values.get(j, 0) == 0 or frequencies.get(j, 0) == 0:
                result_mid = 0
            else:
                # Check if the logarithm will be valid (both values > 0)
                if frequencies.get(j, 0) > 0 and expected_values.get(j, 0) > 0:
                    result_mid = (frequencies[j] / n) * math.log2(expected_values[j] / frequencies[j])
                else:
                    result_mid = 0
            term3 += result_mid

        # Calculate k_i only if ri and xi are greater than 0
        if ri > 0 and xi > 0:
            k_i = k_avg + math.log2(ri / xi) - term3
        else:
            k_i = k_avg - term3  # Avoid log calculation, use only k_avg and term3
        print(k_i)
        # 对 k_i 上取整
        k_star[element] = round(k_i)
    return k_star

# 计算 epsilon_i
# def calculate_epsilon_star(frequencies, expected_values, epsilon, n):
#     epsilon_star = {}
#     for element in frequencies:
#         xi = frequencies[element]  # 当前元素的频率
#         ri = expected_values[element]  # 当前元素的期望值
#
#         # 计算共享项 term3
#         term3 = 0
#         for j in frequencies:
#             if(expected_values[j] == 0 or frequencies[j] == 0):
#                 result_mid = 0
#             else:
#                 result_mid = (frequencies[j] / n) * math.log2(expected_values[j] / frequencies[j])
#             term3 += result_mid
#
#         # print(term3)
#         # 根据公式计算 k_i
#         epsilon_i = epsilon + math.log2(ri / xi) - term3
#         print(epsilon_i)
#         # 对 k_i 上取整
#         epsilon_star[element] = round(epsilon_i, 2)
#     return epsilon_star

# 计算 epsilon_i
def calculate_epsilon_star(frequencies, expected_values, epsilon, n):
    epsilon_star = {}
    for element in frequencies:
        xi = frequencies[element]  # 当前元素的频率
        ri = expected_values.get(element, 0)  # 当前元素的期望值, 默认值为 0

        # 计算共享项 term3
        term3 = 0
        for j in frequencies:
            # 确保 j 同时出现在 frequencies 和 expected_values 中
            if expected_values.get(j, 0) == 0 or frequencies.get(j, 0) == 0:
                result_mid = 0
            else:
                # Check if the logarithm will be valid (both values > 0)
                if frequencies.get(j, 0) > 0 and expected_values.get(j, 0) > 0:
                    result_mid = (frequencies[j] / n) * math.log2(expected_values[j] / frequencies[j])
                else:
                    result_mid = 0
            term3 += result_mid

        # Calculate epsilon_i only if ri and xi are greater than 0
        if ri > 0 and xi > 0:
            epsilon_i = epsilon + math.log2(ri / xi) - term3
        else:
            epsilon_i = epsilon - term3  # Avoid log calculation, use only epsilon and term3
        print(epsilon_i)
        # 对 epsilon_i 上取整
        epsilon_star[element] = round(epsilon_i, 2)
    return epsilon_star

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

# ===================== 读取数据 =====================
data_start = pd.read_excel('Obesity.xlsx')

# 强制将数据转换为数值型，无法转换的变为 NaN
data_start['Weight'] = pd.to_numeric(data_start['Weight'], errors='coerce')

# 处理数值类型的分类
data = data_start['Weight'].apply(lambda x: classify(x) if pd.notna(x) else -1)

T = 2112  # 集合大小
History_data_set = data.iloc[1: 2112].tolist()  # Corrected for Series, no need to specify the column
S = data.iloc[1: n+1].tolist()  # Corrected for Series

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

# ===================== 计算频率 =====================
# 计算每个元素的频率
frequency = Counter(History_data_set)
total_frequency = sum(frequency.values())

frequency_likehood = {element: round((count / total_frequency) * n, 4) for element, count in frequency.items()}
print("元素存在的可能性:", frequency_likehood)
number = sum(frequency_likehood.values())
print("元素个数:", number)

# 按元素编号排序
sorted_frequency_likehood = dict(sorted(frequency_likehood.items()))

# 打印排序后的结果
print("按元素编号排序后的期望值:")
for element, value in sorted_frequency_likehood.items():
    print(f"元素 {element}: {value:.3f}")

# ===================== 计算期望值 E[r_i] =====================
# 计算每个元素的频率
frequency_expect_value = Counter(Q)
total_frequency_expect_value = sum(frequency_expect_value.values())

frequency_expect = {element: round(count / total_frequency_expect_value, 4) for element, count in frequency_expect_value.items()}
print("元素对应的期望值:", frequency_expect)
# 按元素编号排序
sorted_frequency_expect = dict(sorted(frequency_expect.items()))

# 打印排序后的结果
print("按元素编号排序后的期望值:")
for element, value in sorted_frequency_expect.items():
    print(f"元素 {element}: {value:.3f}")



# 计算每个元素的哈希函数数目
k_star_values = calculate_k_star(frequency_likehood, frequency_expect, m, n)
print(k_star_values)



print("每个元素的哈希函数数目 k*:")
for element, k in k_star_values.items():
    print(f"{element}: {k}")

total_hash_functions = 0
for element in frequency_likehood:
    total_hash_functions += frequency_likehood[element] * k_star_values[element]
print("总的哈希函数数目:", total_hash_functions)



# 定义 epsilon 的范围
epsilon_values = [2, 4, 6, 8, 10]

# 存储每个 epsilon 对应的 RMSE 平均值
epsilon_rmse_avg = []

# 遍历 epsilon 的值
for epsilon in epsilon_values:
    print(f"\n当前WBF_epsilon: {epsilon}")

    epsilon_star_values = calculate_epsilon_star(frequency_likehood, frequency_expect, epsilon, n)
    # print(epsilon_star_values)

    total_epsilon = 0
    for element in frequency_likehood:
        total_epsilon += frequency_likehood[element] * epsilon_star_values[element]
    # print("总的隐私预算:", total_epsilon)

    # 存储当前 epsilon 的 100 次 RMSE
    rmse_list = []

    # 重复运行 100 次
    for run in range(100):
        # 初始化布隆过滤器
        bloom_filter = [0] * m

        # 对每个插入的元素，根据对应的哈希函数个数计算并更新布隆过滤器的位置
        for element in S:
            # 计算概率
            k_epsilon = epsilon_star_values[element]  # 获取该元素的隐私预算
            prob_keep = np.exp(k_epsilon) / (np.exp(k_epsilon) + 1)
            k_hash = k_star_values[element]  # 获取该元素的哈希函数个数
            positions = get_hash_position(element, k_hash)  # 获取哈希值并计算对应位置
            for pos in positions:
                bloom_filter[pos] = 1  # 首先置为1
                # 然后判断这个位置的结果是否需要翻转
            for pos in positions:
                p = np.random.rand()
                if p > prob_keep:
                    bloom_filter[pos] = 1 - bloom_filter[pos]  # 以概率 (1-p) 报告翻转数据

        # 计算真实值和预测值
        actual_values = [1 if item in S else 0 for item in Q]
        predicted_values = [False] * len(Q)

        # 遍历 Q 中的每个元素
        for idx, element in enumerate(Q):
            k_hash = k_star_values[element]  # 获取该元素的哈希函数个数
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

    # 计算当前 epsilon 的 RMSE 平均值
    rmse_avg = np.mean(rmse_list)
    epsilon_rmse_avg.append(rmse_avg)

    # 输出当前 epsilon 的结果
    print(f"epsilon = {epsilon}: 平均 RMSE = {rmse_avg:.4f}")

# 输出所有 epsilon 对应的 RMSE 平均值
print("\n所有 epsilon 对应的PrivMQA的RMSE平均值:")
for epsilon, rmse_avg in zip(epsilon_values, epsilon_rmse_avg):
    print(f"epsilon = {epsilon}: 平均 RMSE = {rmse_avg:.4f}")