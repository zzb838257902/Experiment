import csv


def filter_avg():
    weight_dir = r'E:\VS\api_process\apiWeight.csv'
    weight_new_dir = r'E:\VS\api_process\apiWeight_avgtest.csv'
    weight = 0
    count = 0
    api_dictWeights = {}
    api_list = []
    with open(weight_dir, 'r', encoding='utf-8') as f:
        api_weights = list(csv.reader(f))
        for api_weight in api_weights:
            weight += float(api_weight[1])
            api_list.append(api_weight[0].strip('\n'))
            count += 1
            api_dictWeights[api_weight[0]] = api_weight[1]
    avg_weight = weight / count
    print(avg_weight)
    for api in api_list:
        if float(api_dictWeights.get(api)) < avg_weight:
            del api_dictWeights[api]
    with open(weight_new_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in api_dictWeights.items():
            writer.writerow(row)

def apiList():
    apiList_new = r'E:\VS\api_process\apiList_avg.csv'
    weight_new_dir = r'E:\VS\api_process\apiWeight_avg.csv'
    api_list = []
    with open(weight_new_dir, 'r', encoding='utf-8') as f:
        api = list(csv.reader(f))
        for a in api:
            api_list.append(a[0].strip('\n'))
    with open(apiList_new, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in api_list:
            writer.writerow([row])

if __name__ == '__main__':
    filter_avg()
    #apiList()
