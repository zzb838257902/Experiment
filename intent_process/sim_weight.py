import os
import csv


# 相似度计算
def simWeight():
    intentList_dir = r'E:\VS\intent_process\intentAll.csv'
    intentList_mal_dir = r'E:\VS\malware\intentList.csv'
    intentList_ben_dir = r'E:\VS\benign\intentList.csv'
    root_mal_dir = r'E:\VS\malware\intent_abstract'
    root_ben_dir = r'E:\VS\benign\intent_abstract'
    similarity_dir = r'E:\Samples\intent_process\similarity.csv'
    simWeight_dir = r'E:\VS\intent_process\simWeight.csv'
    intent_mal_List = []
    intent_ben_List = []
    intent_mal_Apk = []
    intent_ben_Apk = []
    intent_ben_count = {}
    intent_mal_count = {}
    intent_mal_frequency = {}
    intent_ben_frequency = {}
    similar = {}
    sim_weight = {}
    count = 0
    with open(intentList_mal_dir, 'r', encoding='utf-8') as f:
        intent_mal = list(csv.reader(f))
    for line in intent_mal:
        for l in line:
            intent_mal_List.append(l.strip('\n'))
    print(intent_mal_List)
    for parent, dirnames, filenames in os.walk(root_mal_dir):
        for filename in filenames:
            with open(root_mal_dir + "\\" + filename, 'r') as f:
                for line in f:
                    line = line.strip('\n')
                    if line not in intent_mal_Apk:
                        intent_mal_Apk.append(line)
            for i in intent_mal_Apk:
                if i in intent_mal_List:
                    if i in intent_mal_count.keys():
                        intent_mal_count[i] = intent_mal_count.get(i) + 1
                    else:
                        intent_mal_count[i] = 1
            intent_mal_Apk.clear()
            count += 1
    for i in intent_mal_count.keys():
        intent_mal_frequency[i] = intent_mal_count.get(i) / 14666
    intent_mal_frequency = dict(sorted(intent_mal_frequency.items(), key=lambda x: x[1], reverse=True))

    with open(intentList_ben_dir, 'r', encoding='utf-8') as f:
        intent_ben = list(csv.reader(f))
    for line in intent_ben:
        for l in line:
            intent_ben_List.append(l.strip('\n'))
    print(intent_ben_List)
    for parent, dirnames, filenames in os.walk(root_ben_dir):
        for filename in filenames:
            with open(root_ben_dir + "\\" + filename, 'r') as f:
                for line in f:
                    line = line.strip('\n')
                    if line not in intent_ben_Apk:
                        intent_ben_Apk.append(line)
            for i in intent_ben_Apk:
                if i in intent_ben_List:
                    if i in intent_ben_count.keys():
                        intent_ben_count[i] = intent_ben_count.get(i) + 1
                    else:
                        intent_ben_count[i] = 1
            intent_ben_Apk.clear()
            count += 1
            print("已完成", count)
    for i in intent_ben_count.keys():
        intent_ben_frequency[i] = intent_ben_count.get(i) / 14955
    intent_ben_frequency = dict(sorted(intent_ben_frequency.items(), key=lambda x: x[1], reverse=True))

    for i in intent_mal_count.keys():
        if i not in intent_ben_count.keys():
            intent_ben_count[i] = 0

    for i in intent_ben_count.keys():
        if i not in intent_mal_count.keys():
            intent_mal_count[i] = 0


    for i in intent_ben_frequency.keys():
        if i not in intent_mal_frequency.keys():
            intent_mal_frequency[i] = 0

    for i in intent_mal_frequency.keys():
        if i not in intent_ben_frequency.keys():
            intent_ben_frequency[i] = 0

    for i in intent_ben_frequency.keys():
        similar[i] = abs(intent_ben_frequency.get(i) - intent_mal_frequency.get(i)) / \
                     abs(intent_ben_frequency.get(i) + intent_mal_frequency.get(i))

    for i in intent_ben_frequency.keys():
        sim_weight[i] = similar[i] * abs(intent_ben_frequency.get(i) - intent_mal_frequency.get(i))

    sim_weight = dict(sorted(sim_weight.items(), key=lambda x: x[1], reverse=True))

    with open(simWeight_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in sim_weight.items():
            writer.writerow(row)
    with open(intentList_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in sim_weight.keys():
            writer.writerow([row])


if __name__ == '__main__':
    simWeight()
