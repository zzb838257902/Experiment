import os
import csv


# 相似度计算
def simWeight():
    permissionList_dir = r'E:\VS\permission_process\permissionAll.csv'
    permissionList_mal_dir = r'E:\VS\malware\permissionList.csv'
    permissionList_ben_dir = r'E:\VS\benign\permissionList.csv'
    root_mal_dir = r'E:\VS\malware\permission_abstract'
    root_ben_dir = r'E:\VS\benign\permission_abstract'
    similarity_dir = r'E:\Samples\permission_process\similarity.csv'
    simWeight_dir = r'E:\VS\permission_process\simWeight.csv'
    permission_mal_List = []
    permission_ben_List = []
    permission_mal_Apk = []
    permission_ben_Apk = []
    permission_ben_count = {}
    permission_mal_count = {}
    permission_mal_frequency = {}
    permission_ben_frequency = {}
    similar = {}
    sim_weight = {}
    count = 0
    with open(permissionList_mal_dir, 'r', encoding='utf-8') as f:
        permission_mal = list(csv.reader(f))
    for line in permission_mal:
        for l in line:
            permission_mal_List.append(l.strip('\n'))
    print(len(permission_mal_List))
    for parent, dirnames, filenames in os.walk(root_mal_dir):
        for filename in filenames:
            with open(root_mal_dir + "\\" + filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip('\n')
                    if line not in permission_mal_Apk:
                        permission_mal_Apk.append(line)
            for p in permission_mal_Apk:
                if p in permission_mal_List:
                    if p in permission_mal_count.keys():
                        permission_mal_count[p] = permission_mal_count.get(p) + 1
                    else:
                        permission_mal_count[p] = 1
            permission_mal_Apk.clear()
            count += 1
    for p in permission_mal_count.keys():
        permission_mal_frequency[p] = permission_mal_count.get(p) / 14664
    permission_mal_frequency = dict(sorted(permission_mal_frequency.items(), key=lambda x: x[1], reverse=True))

    with open(permissionList_ben_dir, 'r', encoding='utf-8') as f:
        permission_ben = list(csv.reader(f))
    for line in permission_ben:
        for l in line:
            permission_ben_List.append(l.strip('\n'))
    print(permission_ben_List)
    for parent, dirnames, filenames in os.walk(root_ben_dir):
        for filename in filenames:
            with open(root_ben_dir + "\\" + filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip('\n')
                    if line not in permission_ben_Apk:
                        permission_ben_Apk.append(line)
            for p in permission_ben_Apk:
                if p in permission_ben_List:
                    if p in permission_ben_count.keys():
                        permission_ben_count[p] = permission_ben_count.get(p) + 1
                    else:
                        permission_ben_count[p] = 1
            permission_ben_Apk.clear()
            count += 1
            print("已完成", count)
    for p in permission_ben_count.keys():
        permission_ben_frequency[p] = permission_ben_count.get(p) / 14925
    permission_ben_frequency = dict(sorted(permission_ben_frequency.items(), key=lambda x: x[1], reverse=True))

    for p in permission_mal_count.keys():
        if p not in permission_ben_count.keys():
            permission_ben_count[p] = 0

    for p in permission_ben_count.keys():
        if p not in permission_mal_count.keys():
            permission_mal_count[p] = 0


    for p in permission_ben_frequency.keys():
        if p not in permission_mal_frequency.keys():
            permission_mal_frequency[p] = 0

    for p in permission_mal_frequency.keys():
        if p not in permission_ben_frequency.keys():
            permission_ben_frequency[p] = 0

    for p in permission_ben_frequency.keys():
        similar[p] = abs(permission_ben_frequency.get(p) - permission_mal_frequency.get(p)) / \
                     abs(permission_ben_frequency.get(p) + permission_mal_frequency.get(p))

    for p in permission_ben_frequency.keys():
        sim_weight[p] = similar[p] * abs(permission_ben_frequency.get(p) - permission_mal_frequency.get(p))

    sim_weight = dict(sorted(sim_weight.items(), key=lambda x: x[1], reverse=True))
    with open(simWeight_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in sim_weight.items():
            writer.writerow(row)
    with open(permissionList_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in sim_weight.keys():
            writer.writerow([row])


if __name__ == '__main__':
    simWeight()
