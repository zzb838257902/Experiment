import csv


def filter_avg():
    weight_dir = r'E:\VS\hardware_process\simWeight.csv'
    weight_new_dir = r'E:\VS\hardware_process\hardwareWeight_avgtest.csv'
    weight = 0
    count = 0
    hardware_dictWeights = {}
    hardware_list = []
    with open(weight_dir, 'r', encoding='utf-8') as f:
        hardware_weights = list(csv.reader(f))
        for hardware_weight in hardware_weights:
            weight += float(hardware_weight[1])
            hardware_list.append(hardware_weight[0].strip('\n'))
            count += 1
            hardware_dictWeights[hardware_weight[0]] = hardware_weight[1]
    avg_weight = weight / count
    print(avg_weight)
    for hardware in hardware_list:
        if float(hardware_dictWeights.get(hardware)) < avg_weight:
            del hardware_dictWeights[hardware]
    with open(weight_new_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in hardware_dictWeights.items():
            writer.writerow(row)

def hardwareList():
    hardwareList_new = r'E:\VS\hardware_process\hardwareList_avg.csv'
    weight_new_dir = r'E:\VS\hardware_process\hardwareWeight_avg.csv'
    hardware_list = []
    with open(weight_new_dir, 'r', encoding='utf-8') as f:
        hardware = list(csv.reader(f))
        for a in hardware:
            hardware_list.append(a[0].strip('\n'))
    with open(hardwareList_new, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in hardware_list:
            writer.writerow([row])

if __name__ == '__main__':
    filter_avg()
    #hardwareList()
