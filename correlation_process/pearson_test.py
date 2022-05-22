import math
import pandas as pd
from scipy.stats import pearsonr

def cbf():
    csv_dir = r'E:\Samples\correlation_process\pea_csv\peaAll.csv'
    pearson_dir = r'E:\Samples\correlation_process\hardwarePea1.csv'
    dataset = pd.read_csv(csv_dir)
    features = []
    f_pearson = {}
    drop_list = []
    count = 0
    print(len(dataset.columns))
#     for f in dataset.columns:
#         features.append(f)
#     for i in range(0, len(features)-2):
#         for j in range(i+1, len(features)-1):
#             f_pearson[(features[i], features[j])] = abs(pearsonr(dataset[features[i]], dataset[features[j]])[0])
#     print(f_pearson)
#     for key in f_pearson.keys():
#         if f_pearson.get(key) >= 0.6:
#             if compare_f(dataset[key[0]], dataset[key[1]]):
#                 drop_list.append(key[0])
#             else:
#                 drop_list.append(key[1])
#     dataset = dataset.drop(drop_list, axis=1)
#     dataset.to_csv(pearson_dir, index=False, encoding='utf-8')
#
# def compare_f(vector1, vector2):
#     count1 = 0
#     count2 = 0
#     for i in vector1:
#         if i == 1:
#             count1 += 1
#     for j in vector2:
#         if j == 1:
#             count2 += 1
#     frequency1 = float(count1) / len(vector1)
#     frequency2 = float(count2) / len(vector2)
#     if frequency1 >= frequency2:
#         return True
#     else:
#         return False

if __name__ == '__main__':
    cbf()