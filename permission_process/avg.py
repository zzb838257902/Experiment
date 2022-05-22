import csv


def filter_avg():
    weight_dir = r'E:\VS\permission_process\simWeight.csv'
    weight_new_dir = r'E:\VS\permission_process\simWeight_avgtest.csv'
    weight = 0
    count = 0
    permission_dictWeights = {}
    permission_list = []
    with open(weight_dir, 'r', encoding='utf-8') as f:
        permission_weights = list(csv.reader(f))
        for permission_weight in permission_weights:
            weight += float(permission_weight[1])
            permission_list.append(permission_weight[0].strip('\n'))
            count += 1
            permission_dictWeights[permission_weight[0]] = permission_weight[1]
    avg_weight = weight / count
    print(avg_weight)
    for permission in permission_list:
        if float(permission_dictWeights.get(permission)) < avg_weight:
            del permission_dictWeights[permission]
    with open(weight_new_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in permission_dictWeights.items():
            writer.writerow(row)

def permissionList():
    permissionList_new = r'E:\VS\permission_process\permissionList_avg.csv'
    weight_new_dir = r'E:\VS\permission_process\simWeight_avg.csv'
    permission_list = []
    with open(weight_new_dir, 'r', encoding='utf-8') as f:
        permission = list(csv.reader(f))
        for a in permission:
            permission_list.append(a[0].strip('\n'))
    with open(permissionList_new, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in permission_list:
            writer.writerow([row])

if __name__ == '__main__':
    filter_avg()
    #permissionList()
