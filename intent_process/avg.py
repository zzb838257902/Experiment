import csv


def filter_avg():
    weight_dir = r'E:\VS\intent_process\simWeight.csv'
    weight_new_dir = r'E:\VS\intent_process\simWeight_avgtest.csv'
    weight = 0
    count = 0
    intent_dictWeights = {}
    intent_list = []
    with open(weight_dir, 'r', encoding='utf-8') as f:
        intent_weights = list(csv.reader(f))
        for intent_weight in intent_weights:
            weight += float(intent_weight[1])
            intent_list.append(intent_weight[0].strip('\n'))
            count += 1
            intent_dictWeights[intent_weight[0]] = intent_weight[1]
    avg_weight = weight / count
    print(avg_weight)
    for intent in intent_list:
        if float(intent_dictWeights.get(intent)) < avg_weight:
            del intent_dictWeights[intent]
    with open(weight_new_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in intent_dictWeights.items():
            writer.writerow(row)

def intentList():
    intentList_new = r'E:\VS\intent_process\intentList_avg.csv'
    weight_new_dir = r'E:\VS\intent_process\simWeight_avg.csv'
    intent_list = []
    with open(weight_new_dir, 'r', encoding='utf-8') as f:
        intent = list(csv.reader(f))
        for a in intent:
            intent_list.append(a[0].strip('\n'))
    with open(intentList_new, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in intent_list:
            writer.writerow([row])

if __name__ == '__main__':
    filter_avg()
    #intentList()
