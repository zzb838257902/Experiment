import csv
import pandas as pd


def peaList():
    api_dir = r'E:\VS\api_process\apiList_avg.csv'
    hardware_dir = r'E:\VS\hardware_process\hardwareList_avg.csv'
    intent_dir = r'E:\VS\intent_process\intentList_avg.csv'
    permission_dir = r'E:\VS\permission_process\permissionList_avg.csv'
    pea_dir = r'E:\VS\correlation_process\peaBetween.csv'
    api_pea_dir = r'E:\VS\correlation_process\apiPeaList.csv'
    hardware_pea_dir = r'E:\VS\correlation_process\hardwarePeaList.csv'
    intent_pea_dir = r'E:\VS\correlation_process\intentPeaList.csv'
    permission_pea_dir = r'E:\VS\correlation_process\permissionPeaList.csv'
    api_avg = []
    hardware_avg = []
    intent_avg = []
    permission_avg = []
    api_pea = []
    hardware_pea = []
    intent_pea = []
    permission_pea = []
    with open(api_dir, 'r', encoding='utf-8') as f:
        apis = list(csv.reader(f))
        for api in apis:
            for a in api:
                api_avg.append(a.strip('\n'))
    with open(hardware_dir, 'r', encoding='utf-8') as f:
        hardwares = list(csv.reader(f))
        for hardware in hardwares:
            for h in hardware:
                hardware_avg.append(h.strip('\n'))
    with open(intent_dir, 'r', encoding='utf-8') as f:
        intents = list(csv.reader(f))
        for intent in intents:
            for i in intent:
                intent_avg.append(i.strip('\n'))
    with open(permission_dir, 'r', encoding='utf-8') as f:
        permissions = list(csv.reader(f))
        for permission in permissions:
            for p in permission:
                permission_avg.append(p.strip('\n'))
    for f in pd.read_csv(pea_dir).columns:
        f = f.strip('\n')
        if f in api_avg:
            api_pea.append(f)
        if f in hardware_avg:
            hardware_pea.append(f)
        if f in intent_avg:
            intent_pea.append(f)
        if f in permission_avg:
            permission_pea.append(f)
    with open(api_pea_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in api_pea:
            writer.writerow([row])
    with open(hardware_pea_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in hardware_pea:
            writer.writerow([row])
    with open(intent_pea_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in intent_pea:
            writer.writerow([row])
    with open(permission_pea_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in permission_pea:
            writer.writerow([row])

def pealist1():
    pea_dir = r'E:\VS\correlation_process\peaBetween.csv'
    pea_list = r'E:\VS\correlation_process\peaList.csv'
    pea = []
    for f in pd.read_csv(pea_dir).columns:
        f = f.strip('\n')
        pea.append(f)
    with open(pea_list, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in pea:
            writer.writerow([row])

if __name__ == '__main__':
    pealist1()