import os
import os.path
import numpy as np
import pandas as pd
import csv

# 将列表转换为csv文件进行存储

def transition_csv(permissionList):
    permissionList.append("label")
    permission_malware = pd.DataFrame(columns=permissionList)
    # for n in range(len(permission_all)):
    permission_malware.to_csv(r'E:\VS\correlation_process\permission_benign.csv', encoding='utf-8', index=False)

def permission_csv():
    permissionAll_dir = r'E:\VS\correlation_process\permissionPeaList.csv'
    root_dir = r'E:\VS\benign\all_non_repeat'
    permission_dir = r'E:\VS\correlation_process\permission_benign.csv'
    permissionList = []
    count_all = []
    count = 0
    with open(permissionAll_dir, 'r', encoding='utf-8') as f:
        permission = list(csv.reader(f))
    for line in permission:
        for l in line:
            permissionList.append(l.strip('\n'))
    transition_csv(permissionList)
    permissionList_length = len(permissionList)
    print(permissionList_length)
    for parent, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            count_list = [0]*permissionList_length
            with open(root_dir+"\\"+filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip('\n')
                    for p in permissionList:
                        if p == line:
                            count_list[permissionList.index(p)] = 1
            count_list[-1] = 0
            count_all.append(count_list)
            count += 1
            print("已完成", count)
    with open(permission_dir, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in count_all:
            writer.writerow(row)


if __name__ == '__main__':
    permission_csv()