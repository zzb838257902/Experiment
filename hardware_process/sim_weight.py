import os
import csv


# 相似度计算
def simWeight():
    hardwareList_dir = r'E:\VS\hardware_process\hardwareAll.csv'
    hardwareList_mal_dir = r'E:\VS\malware\hardwareList.csv'
    hardwareList_ben_dir = r'E:\VS\benign\hardwareList.csv'
    root_mal_dir = r'E:\VS\malware\hardware_non_repeat'
    root_ben_dir = r'E:\VS\benign\hardware_non_repeat'
    simWeight_dir = r'E:\VS\hardware_process\simWeight.csv'
    hardware_mal_List = []
    hardware_ben_List = []
    hardware_mal_Apk = []
    hardware_ben_Apk = []
    hardware_ben_count = {}
    hardware_mal_count = {}
    hardware_mal_frequency = {}
    hardware_ben_frequency = {}
    similar = {}
    sim_weight = {}
    count = 0
    with open(hardwareList_mal_dir, 'r', encoding='utf-8') as f:
        hardware_mal = list(csv.reader(f))
    for line in hardware_mal:
        for l in line:
            hardware_mal_List.append(l.strip('\n'))
    print(hardware_mal_List)
    for parent, dirnames, filenames in os.walk(root_mal_dir):
        for filename in filenames:
            with open(root_mal_dir + "\\" + filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip('\n')
                    if line not in hardware_mal_Apk:
                        hardware_mal_Apk.append(line)
            for h in hardware_mal_Apk:
                if h in hardware_mal_List:
                    if h in hardware_mal_count.keys():
                        hardware_mal_count[h] = hardware_mal_count.get(h) + 1
                    else:
                        hardware_mal_count[h] = 1
            hardware_mal_Apk.clear()
            count += 1
    for h in hardware_mal_count.keys():
        hardware_mal_frequency[h] = hardware_mal_count.get(h) / 4356
    hardware_mal_frequency = dict(sorted(hardware_mal_frequency.items(), key=lambda x: x[1], reverse=True))

    with open(hardwareList_ben_dir, 'r', encoding='utf-8') as f:
        hardware_ben = list(csv.reader(f))
    for line in hardware_ben:
        for l in line:
            hardware_ben_List.append(l.strip('\n'))
    print(hardware_ben_List)
    for parent, dirnames, filenames in os.walk(root_ben_dir):
        for filename in filenames:
            with open(root_ben_dir + "\\" + filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip('\n')
                    if line not in hardware_ben_Apk:
                        hardware_ben_Apk.append(line)
            for h in hardware_ben_Apk:
                if h in hardware_ben_List:
                    if h in hardware_ben_count.keys():
                        hardware_ben_count[h] = hardware_ben_count.get(h) + 1
                    else:
                        hardware_ben_count[h] = 1
            hardware_ben_Apk.clear()
            count += 1
            print("已完成", count)
    for h in hardware_ben_count.keys():
        hardware_ben_frequency[h] = hardware_ben_count.get(h) / 5475
    hardware_ben_frequency = dict(sorted(hardware_ben_frequency.items(), key=lambda x: x[1], reverse=True))

    for h in hardware_mal_count.keys():
        if h not in hardware_ben_count.keys():
            hardware_ben_count[h] = 0

    for h in hardware_ben_count.keys():
        if h not in hardware_mal_count.keys():
            hardware_mal_count[h] = 0


    for h in hardware_ben_frequency.keys():
        if h not in hardware_mal_frequency.keys():
            hardware_mal_frequency[h] = 0

    for h in hardware_mal_frequency.keys():
        if h not in hardware_ben_frequency.keys():
            hardware_ben_frequency[h] = 0

    for h in hardware_ben_frequency.keys():
        similar[h] = abs(hardware_ben_frequency.get(h) - hardware_mal_frequency.get(h)) / \
                     abs(hardware_ben_frequency.get(h) + hardware_mal_frequency.get(h))

    for h in hardware_ben_frequency.keys():
        sim_weight[h] = similar[h] * abs(hardware_ben_frequency.get(h) - hardware_mal_frequency.get(h))

    sim_weight = dict(sorted(sim_weight.items(), key=lambda x: x[1], reverse=True))
    with open(simWeight_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in sim_weight.items():
            writer.writerow(row)
    with open(hardwareList_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in sim_weight.keys():
            writer.writerow([row])


if __name__ == '__main__':
    simWeight()
