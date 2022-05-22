import os
import os.path
import numpy as np
import pandas as pd
import csv

# 将列表转换为csv文件进行存储

def transition_csv(hardwareList):
    hardwareList.append("label")
    hardware_malware = pd.DataFrame(columns=hardwareList)
    # for n in range(len(hardware_all)):
    hardware_malware.to_csv(r'E:\VS\correlation_process\hardware_malware.csv', encoding='utf-8', index=False)

def hardware_csv():
    hardwareAll_dir = r'E:\VS\correlation_process\hardwarePeaList.csv'
    root_dir = r'E:\VS\malware\all_non_repeat'
    hardware_dir = r'E:\VS\correlation_process\hardware_malware.csv'
    hardwareList = []
    count_all = []
    count = 0
    with open(hardwareAll_dir, 'r', encoding='utf-8') as f:
        hardware = list(csv.reader(f))
    for line in hardware:
        for l in line:
            hardwareList.append(l.strip('\n'))
    transition_csv(hardwareList)
    hardwareList_length = len(hardwareList)
    print(hardwareList_length)
    for parent, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            count_list = [0]*hardwareList_length
            with open(root_dir+"\\"+filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip('\n')
                    for p in hardwareList:
                        if p == line:
                            count_list[hardwareList.index(p)] = 1
            count_list[-1] = 1
            count_all.append(count_list)
            count += 1
            print("已完成", count)
    with open(hardware_dir, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in count_all:
            writer.writerow(row)


if __name__ == '__main__':
    hardware_csv()