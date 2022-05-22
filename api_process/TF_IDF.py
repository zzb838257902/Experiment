import os
import csv
import math


def TFIDF():
    root_mal_dir = r'E:\VS\malware\api_non_repeat'
    root_ben_dir = r'E:\VS\benign\api_non_repeat'
    api_mal_count = r'E:\VS\malware\apiCount.csv'
    api_ben_count = r'E:\VS\benign\apiCount.csv'
    apiWeight_dir = r'E:\VS\api_process\apiWeight.csv'
    api_all_dir = r'E:\VS\api_process\apiAll.csv'
    number_ben = 0
    number_mal = 0
    api_mal_List = []
    api_ben_List = []
    api_mal_APK = []
    api_ben_APK = []
    api_ben_frequency = {}
    api_mal_frequency = {}
    api_ben_idf = {}
    api_mal_idf = {}
    count = 0
    tfidf_difference = {}
    # 计算TF
    with open(api_ben_count, 'r', encoding='utf-8') as f:
        api_ben = list(csv.reader(f))
        for api in api_ben:
            number_ben += float(api[1])
            api_ben_List.append(api[0].strip('\n'))
        for api in api_ben:
            api_ben_frequency[api[0]] = float(api[1]) / number_ben

    with open(api_mal_count, 'r', encoding='utf-8') as f:
        api_mal = list(csv.reader(f))
        for api in api_mal:
            number_mal += float(api[1])
            api_mal_List.append(api[0].strip('\n'))
        for api in api_mal:
            api_mal_frequency[api[0]] = float(api[1]) / number_mal

    # 计算IDF
    for parent, dirnames, filenames in os.walk(root_ben_dir):
        for filename in filenames:
            with open(root_ben_dir + "\\" + filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip('\n')
                    if line not in api_ben_APK:
                        api_ben_APK.append(line)
            for api in api_ben_APK:
                if api in api_ben_List:
                    if api in api_ben_idf.keys():
                        api_ben_idf[api] = api_ben_idf.get(api) + 1
                    else:
                        api_ben_idf[api] = 1
            api_ben_APK.clear()
            count += 1
            print("已完成", count)
    for api in api_ben_idf.keys():
        api_ben_idf[api] = math.log(float(14958) / api_ben_idf.get(api))
    print(api_ben_idf)

    for parent, dirnames, filenames in os.walk(root_mal_dir):
        for filename in filenames:
            with open(root_mal_dir + "\\" + filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip('\n')
                    if line not in api_mal_APK:
                        api_mal_APK.append(line)
            for api in api_mal_APK:
                if api in api_mal_List:
                    if api in api_mal_idf.keys():
                        api_mal_idf[api] = api_mal_idf.get(api) + 1
                    else:
                        api_mal_idf[api] = 1
            api_mal_APK.clear()
            count += 1
            print("已完成", count)
    for api in api_mal_idf.keys():
        api_mal_idf[api] = math.log(float(14663) / api_mal_idf.get(api))
    print(api_mal_idf)

    # 互相补充，不存在的填0
    for api in api_mal_frequency.keys():
        if api not in api_ben_frequency.keys():
            api_ben_frequency[api] = 0

    for api in api_ben_frequency.keys():
        if api not in api_mal_frequency.keys():
            api_mal_frequency[api] = 0

    for api in api_mal_idf.keys():
        if api not in api_ben_idf.keys():
            api_ben_idf[api] = 0

    for api in api_ben_idf.keys():
        if api not in api_mal_idf.keys():
            api_mal_idf[api] = 0

    for api in api_mal_frequency.keys():
        tfidf_difference[api] = abs(api_ben_frequency.get(api) * api_ben_idf.get(api) - api_mal_frequency.get(api) * api_mal_idf.get(api))

    tfidf_difference = dict(sorted(tfidf_difference.items(), key=lambda x: x[1], reverse=True))
    with open(apiWeight_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in tfidf_difference.items():
            writer.writerow(row)
    with open(api_all_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in tfidf_difference.keys():
            writer.writerow([row])


if __name__ == '__main__':
    TFIDF()
