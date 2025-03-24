import random
import hashlib
import numpy as np
import math
import pandas as pd
from collections import Counter



# 计算 k_i
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
            if expected_values.get(j, 0) == 0 or frequencies.get(j, 0) == 0:
                result_mid = 0
            else:
                if frequencies.get(j, 0) > 0 and expected_values.get(j, 0) > 0:
                    result_mid = (frequencies[j] / n) * math.log2(expected_values[j] / frequencies[j])
                else:
                    result_mid = 0
            term3 += result_mid

        if ri > 0 and xi > 0:
            k_i = k_avg + math.log2(ri / xi) - term3
        else:
            k_i = k_avg - term3
        print(k_i)
        k_star[element] = round(k_i)
    return k_star

# 计算 epsilon_i
def calculate_epsilon_star(frequencies, expected_values, epsilon, n):
    epsilon_star = {}
    for element in frequencies:
        xi = frequencies[element]  # 当前元素的频率
        ri = expected_values.get(element, 0)  # 当前元素的期望值, 默认值为 0

        term3 = 0
        for j in frequencies:
            if expected_values.get(j, 0) == 0 or frequencies.get(j, 0) == 0:
                result_mid = 0
            else:
                if frequencies.get(j, 0) > 0 and expected_values.get(j, 0) > 0:
                    result_mid = (frequencies[j] / n) * math.log2(expected_values[j] / frequencies[j])
                else:
                    result_mid = 0
            term3 += result_mid

        if ri > 0 and xi > 0:
            epsilon_i = epsilon + math.log2(ri / xi) - term3
        else:
            epsilon_i = epsilon - term3
        print(epsilon_i)
        epsilon_star[element] = round(epsilon_i, 2)
    return epsilon_star

def classify(num):
    if 0 <= num <= 10:
        return 1
    elif 10 < num <= 20:
        return 2
    elif 20 < num <= 30:
        return 3
    elif 30 < num <= 40:
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
    else:
        return 0  # 如果超出范围，返回 -1

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
data_start = pd.read_excel('acc.xlsx')

# 强制将数据转换为数值型，无法转换的变为 NaN
data_start['age_of_casualty'] = pd.to_numeric(data_start['age_of_casualty'], errors='coerce')

# 处理数值类型的分类
data = data_start['age_of_casualty'].apply(lambda x: classify(x) if pd.notna(x) else -1)

T = 13000  # 集合大小
History_data_set = data.iloc[1: 13000].tolist()  # Corrected for Series, no need to specify the column
S = data.iloc[1: n].tolist()  # Corrected for Series


# 设定 epsilon 固定为 2
epsilon = 5

# ===================== 查询数据集 =====================
# Zipf分布的参数
# Zipf分布的排名（0到8，包含9个值）
values = np.array([1, 3, 5, 2, 4, 0, 6, 8, 9, 7])  # 对应的9个值
ranks = np.arange(1, 11)  # 排名 1 到 9

s = 1
frequencies = 1 / ranks**s
frequencies /= frequencies.sum()

# 生成查询数据集
query_data = np.random.choice(values, size=n, p=frequencies)
Q = query_data.tolist()

# 计算频率
frequency = Counter(History_data_set)
total_frequency = sum(frequency.values())

frequency_likehood = {element: round((count / total_frequency) * n, 4) for element, count in frequency.items()}

# 计算期望值
frequency_expect_value = Counter(Q)
total_frequency_expect_value = sum(frequency_expect_value.values())

frequency_expect = {element: round(count / total_frequency_expect_value, 4) for element, count in frequency_expect_value.items()}




# 存储每个 m 对应的 RMSE 平均值
m_values = [10000, 20000, 30000, 40000, 50000]  # 可以调整 m 的不同值
m_rmse_avg = {}
# 遍历不同的 m 值
for m in m_values:
    print(f"\n当前位数组大小 m: {m}")
    # 计算 k_star
    k_star_values = calculate_k_star(frequency_likehood, frequency_expect, m, n)
    # print(k_star_values)

    total_hash_functions = 0
    for element in frequency_likehood:
        total_hash_functions += frequency_likehood[element] * k_star_values[element]
    # print("总的哈希函数数目:", total_hash_functions)

    # 计算 epsilon_star 值
    epsilon_star_values = calculate_epsilon_star(frequency_likehood, frequency_expect, epsilon, n)

    # 初始化 RMSE 存储
    rmse_list = []

    # 重复运行 100 次
    for run in range(100):
        bloom_filter = [0] * m

        # 更新布隆过滤器
        for element in S:
            k_epsilon = epsilon_star_values[element]
            prob_keep = np.exp(k_epsilon) / (np.exp(k_epsilon) + 1)
            k_hash = k_star_values[element]
            positions = get_hash_position(element, k_hash)
            for pos in positions:
                bloom_filter[pos] = 1
            for pos in positions:
                p = np.random.rand()
                if p > prob_keep:
                    bloom_filter[pos] = 1 - bloom_filter[pos]

        # 计算真实值和预测值
        actual_values = [1 if item in S else 0 for item in Q]
        predicted_values = [False] * len(Q)

        for idx, element in enumerate(Q):
            k_hash = k_star_values[element]
            positions = get_hash_position(element, k_hash)

            all_positions_are_one = True
            for pos in positions:
                if bloom_filter[pos] != 1:
                    all_positions_are_one = False
                    break

            if all_positions_are_one:
                predicted_values[idx] = True

        # 计算 RMSE
        def calculate_rmse(actual, predicted):
            mse = np.mean((np.array(actual) - np.array(predicted)) ** 2)
            return np.sqrt(mse)

        rmse = calculate_rmse(actual_values, predicted_values)
        rmse_list.append(rmse)

    # 计算当前 m 值的 RMSE 平均值
    rmse_avg = np.mean(rmse_list)
    m_rmse_avg[m] = rmse_avg

    # 输出当前 m 的结果
    print(f"m = {m}: 平均 RMSE = {rmse_avg:.4f}")

# 输出所有 m 对应的 RMSE 平均值
print("\n所有 m 对应的PrivMQA RMSE 平均值:")
for m, rmse_avg in m_rmse_avg.items():
    print(f"m = {m}: 平均 RMSE = {rmse_avg:.4f}")
