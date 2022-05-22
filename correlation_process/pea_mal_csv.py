import os
import os.path
import numpy as np
import pandas as pd
import csv

# 将列表转换为csv文件进行存储

def transition_csv(avgList):
    avgList.append("label")
    avg_malware = pd.DataFrame(columns=avgList)
    # for n in range(len(permission_all)):
    avg_malware.to_csv(r'E:\VS\correlation_process\pea_malware.csv', encoding='utf-8', index=False)

def avg_csv():
    avgAVG_dir = r'E:\VS\correlation_process\featurePea.csv'
    root_dir = r'E:\VS\malware\all_non_repeat'
    avg_dir = r'E:\VS\correlation_process\pea_malware.csv'
    avgList = []
    count_all = []
    count = 0
    with open(avgAVG_dir, 'r', encoding='utf-8') as f:
        avg = list(csv.reader(f))
    for line in avg:
        for l in line:
            avgList.append(l.strip('\n'))
    transition_csv(avgList)
    avgList_length = len(avgList)
    for parent, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            count_list = [0]*avgList_length
            with open(root_dir+"\\"+filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip('\n')
                    for avg in avgList:
                        if avg == line:
                            count_list[avgList.index(avg)] = 1
            count_list[-1] = 1
            count_all.append(count_list)
            count += 1
            print("已完成", count)
    with open(avg_dir, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in count_all:
            writer.writerow(row)


if __name__ == '__main__':
    avg_csv()