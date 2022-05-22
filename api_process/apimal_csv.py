import os
import os.path
import numpy as np
import pandas as pd
import csv

# 将列表转换为csv文件进行存储

def transition_csv(apiList):
    apiList.append("label")
    api_malware = pd.DataFrame(columns=apiList)
    # for n in range(len(api_all)):
    api_malware.to_csv(r'E:\VS\correlation_process\api_malware.csv', encoding='utf-8', index=False)

def api_csv():
    apiAll_dir = r'E:\VS\correlation_process\apiPeaList.csv'
    root_dir = r'E:\VS\malware\all_non_repeat'
    api_dir = r'E:\VS\correlation_process\api_malware.csv'
    apiList = []
    count_all = []
    count = 0
    with open(apiAll_dir, 'r', encoding='utf-8') as f:
        api = list(csv.reader(f))
    for line in api:
        for l in line:
            apiList.append(l.strip('\n'))
    transition_csv(apiList)
    apiList_length = len(apiList)
    print(apiList_length)
    for parent, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            count_list = [0]*apiList_length
            with open(root_dir+"\\"+filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip('\n')
                    for p in apiList:
                        if p == line:
                            count_list[apiList.index(p)] = 1
            count_list[-1] = 1
            count_all.append(count_list)
            count += 1
            print("已完成", count)
    with open(api_dir, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in count_all:
            writer.writerow(row)


if __name__ == '__main__':
    api_csv()