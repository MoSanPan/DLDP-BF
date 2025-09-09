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
        return 15  # If out of range, return 15

# Hash function to compute positions in the bloom filter
def get_hash_position(value, hash_count):
    positions = []
    for i in range(hash_count):
        hash_value = int(hashlib.md5((str(value) + str(i)).encode('utf-8')).hexdigest(), 16)
        position = hash_value % m  # modulo operation ensures positions are within the bloom filter size
        positions.append(position)
    return positions

# Different m values to test
m_values = [10000, 20000, 30000, 40000, 50000]
n = 1000  # Total number of elements


# ===================== Read Data =====================
data_start = pd.read_excel('Obesity.xlsx')

# Force conversion to numeric, any non-convertible values become NaN
data_start['Weight'] = pd.to_numeric(data_start['Weight'], errors='coerce')

# Classify numerical values into bins
data = data_start['Weight'].apply(lambda x: classify(x) if pd.notna(x) else -1)

S = data.iloc[1: n+1].tolist()  # Insertion data set

# ===================== Query Data Set =====================
values = np.array([4,5,6,7,8,9,10,11,12,13,14])  # Now 9 elements
ranks = np.arange(1, len(values) + 1)  # Adjust ranks for 9 elements

# Set Zipf distribution parameters, s is the exponent
s = 1  # Common Zipf parameter is 1
frequencies = 1 / ranks**s  # Calculate frequencies according to Zipf's law

# Normalize the frequencies to ensure they sum to 1
frequencies /= frequencies.sum()

# Assert lengths match
assert len(values) == len(frequencies), "Mismatch between values and frequencies length!"

# Generate a query data set using Zipf distribution
query_data = np.random.choice(values, size=n, p=frequencies)

Q = query_data.tolist()

print(len(Q))
print(len(S))
print("Insertion Data Set S:", S)
print("Query Data Set Q:", Q)

# Define epsilon
epsilon = 5

# 存储每个 epsilon 对应的平均 RMSE 和 ACC
m_avg_rmse = []
m_avg_acc = []

# Loop through different m values
for m in m_values:
    print(f"\nTesting with m = {m}")
    # Calculate the number of hash functions based on the new m
    k_hash = math.ceil((m/n) * math.log(2))

    # 存储当前 m 值的 100 次实验 RMSE 和 MAE
    rmse_list = []
    acc_list = []

    # Repeat 100 times for experiments
    for _ in range(100):
        # Initialize the bloom filter
        bloom_filter = [0] * m

        # Calculate the probability of keeping the element in the bloom filter
        prob_keep = np.exp(epsilon) / (np.exp(epsilon) + 1)

        # Insert each element in S based on its hash positions
        for element in S:
            positions = get_hash_position(element, k_hash)  # Get hash positions
            for pos in positions:
                bloom_filter[pos] = 1  # Set the position to 1

        # # Flip positions based on the probability
        # for i in range(len(bloom_filter)):
        #     if np.random.rand() > prob_keep:
        #         bloom_filter[i] = 1 - bloom_filter[i]  # Flip the value with probability (1 - p)

        # Calculate actual values and predicted values
        actual_values = [1 if item in S else 0 for item in Q]
        predicted_values = [False] * len(Q)

        # Check the bloom filter for each element in the query set Q
        for idx, element in enumerate(Q):
            positions = get_hash_position(element, k_hash)  # Get hash positions

            # Check if all positions are 1
            all_positions_are_one = True
            for pos in positions:
                if bloom_filter[pos] != 1:
                    all_positions_are_one = False
                    break  # If any position is 0, stop checking

            # Update predicted values
            if all_positions_are_one:
                predicted_values[idx] = True  # Mark as True if all positions are 1


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
