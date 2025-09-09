import random
import hashlib
import numpy as np
import pandas as pd
import math
from collections import Counter

# 哈希函数，用于计算哈希值
def get_hash_position(value, hash_count):
    positions = []
    for i in range(hash_count):
        hash_value = int(hashlib.md5((str(value) + str(i)).encode('utf-8')).hexdigest(), 16)
        position = hash_value % m  # 取模操作确保在布隆过滤器范围内
        positions.append(position)
    return positions


def transform_data(data, interval_size=10):
    """
    将数据按照指定区间大小进行划分，并映射为区间编号。

    :param data: 输入数据列表
    :param interval_size: 区间大小，默认为 10
    :return: 转换后的数据列表
    """
    transformed_data = []
    for value in data:
        # 确保 value 是数值类型
        if isinstance(value, str):
            try:
                value = float(value)  # 尝试将字符串转换为浮点数
            except ValueError:
                value = 0  # 如果转换失败，设置为 0
        if value < 0:
            transformed_data.append(0)  # 小于 0 记为 0
        else:
            # 计算区间编号
            interval_index = (value // interval_size) + 1
            transformed_data.append(interval_index)
    return transformed_data

# 位数组大小和元素总数
n = 1000  # 总的元素数量
epsilon = 5 # 固定 epsilon
m_values = [10000, 20000, 30000, 40000, 50000]  # 不同的位数组大小

# ===================== 读取数据 =====================
data = pd.read_excel('Car.xlsx')

T = 1000000  # 集合大小
History_data_set = data.iloc[1:2 * T, 0].tolist()  # 历史数据集
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

# 存储每个 epsilon 对应的平均 RMSE 和 ACC
m_avg_rmse = []
m_avg_acc = []

# 遍历不同的 m 值
for m in m_values:
    k_hash = math.ceil((m/n) * math.log(2))  # 计算哈希函数个数
    print(f"\n当前 m = {m}, k_hash = {k_hash}")

    # 存储当前 m 值的 100 次实验 RMSE 和 MAE
    rmse_list = []
    acc_list = []

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

        # # 判断bloom_filter是否需要翻转
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
    m_avg_rmse.append(avg_rmse)

    avg_acc = np.mean(acc_list)
    m_avg_acc.append(avg_acc)

    # 输出当前 m_values 的结果
    print(f"m = {m}: 平均 RMSE = {avg_rmse:.4f},  平均 ACC = {avg_acc:.4f}")

# 输出所有 epsilon 对应的平均 RMSE 和 MAE
print("\n所有 epsilon 对应的RAPPOR平均 RMSE :")
for m, avg_rmse, avg_acc in zip(m_values, m_avg_rmse, m_avg_acc):
    print(f"epsilon = {m}: 平均 RMSE = {avg_rmse:.4f},  平均 ACC = {avg_acc:.4f}")
