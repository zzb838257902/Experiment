import os
import os.path
import numpy as np
import pandas as pd
import csv

# 将列表转换为csv文件进行存储

def transition_csv(intentList):
    intentList.append("label")
    intent_malware = pd.DataFrame(columns=intentList)
    # for n in range(len(intent_all)):
    intent_malware.to_csv(r'E:\VS\correlation_process\intent_malware.csv', encoding='utf-8', index=False)

def intent_csv():
    intentAll_dir = r'E:\VS\correlation_process\intentPeaList.csv'
    root_dir = r'E:\VS\malware\all_non_repeat'
    intent_dir = r'E:\VS\correlation_process\intent_malware.csv'
    intentList = []
    count_all = []
    count = 0
    with open(intentAll_dir, 'r', encoding='utf-8') as f:
        intent = list(csv.reader(f))
    for line in intent:
        for l in line:
            intentList.append(l.strip('\n'))
    transition_csv(intentList)
    intentList_length = len(intentList)
    print(intentList_length)
    for parent, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            count_list = [0]*intentList_length
            with open(root_dir+"\\"+filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip('\n')
                    for p in intentList:
                        if p == line:
                            count_list[intentList.index(p)] = 1
            count_list[-1] = 1
            count_all.append(count_list)
            count += 1
            print("已完成", count)
    with open(intent_dir, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in count_all:
            writer.writerow(row)


if __name__ == '__main__':
    intent_csv()