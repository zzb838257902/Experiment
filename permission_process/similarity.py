import os
import csv

#相似度计算
def similarity():
    permissionList_mal_dir = r'E:\Samples\merge_mal_permission\permissionList.csv'
    permissionList_ben_dir = r'E:\Samples\merge_ben_permission\permissionList.csv'
    root_mal_dir = r'E:\Samples\merge_mal_permission\permission_non_repeat'
    root_ben_dir = r'E:\Samples\merge_ben_permission\permission_non_repeat'
    similarity_dir = r'E:\Samples\permission_process\similarity.csv'
    permission_mal_List = []
    permission_ben_List = []
    permission_mal_Apk = []
    permission_ben_Apk = []
    permission_mal_frequency = {}
    permission_ben_frequency = {}
    similar = {}
    count = 0
    with open(permissionList_mal_dir, 'r', encoding='utf-8') as f:
        permission_mal = list(csv.reader(f))
    for line in permission_mal:
        for l in line:
            permission_mal_List.append(l.strip('\n'))
    print(permission_mal_List)
    for parent, dirnames, filenames in os.walk(root_mal_dir):
        for filename in filenames:
            with open(root_mal_dir + "\\" + filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip('\n')
                    if line not in permission_mal_Apk:
                        permission_mal_Apk.append(line)
            for p in permission_mal_Apk:
                if p in permission_mal_List:
                    if p in permission_mal_frequency.keys():
                        permission_mal_frequency[p] = permission_mal_frequency.get(p) + 1
                    else:
                        permission_mal_frequency[p] = 1
            permission_mal_Apk.clear()
            count += 1
    for p in permission_mal_frequency.keys():
        permission_mal_frequency[p] = permission_mal_frequency.get(p) / 14661
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
                    if p in permission_ben_frequency.keys():
                        permission_ben_frequency[p] = permission_ben_frequency.get(p) + 1
                    else:
                        permission_ben_frequency[p] = 1
            permission_ben_Apk.clear()
            count += 1
            print("已完成", count)
    for p in permission_ben_frequency.keys():
        permission_ben_frequency[p] = permission_ben_frequency.get(p)/14925
    permission_ben_frequency = dict(sorted(permission_ben_frequency.items(), key=lambda x: x[1], reverse=True))
    for p in permission_ben_frequency.keys():
        if p not in permission_mal_frequency.keys():
            permission_mal_frequency[p] = 0
    for p in permission_mal_frequency.keys():
        if p not in permission_ben_frequency.keys():
            permission_ben_frequency[p] = 0
    for p in permission_ben_frequency.keys():
        similar[p] = abs(permission_ben_frequency.get(p) - permission_mal_frequency.get(p)) / \
                     abs(permission_ben_frequency.get(p) + permission_mal_frequency.get(p))
    similar = dict(sorted(similar.items(), key=lambda x: x[1], reverse=True))
    with open(similarity_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in similar.items():
            writer.writerow(row)

if __name__ == '__main__':
    similarity()