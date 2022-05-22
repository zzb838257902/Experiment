import csv
import pandas as pd

def merge_feature():
    api_dir = r'E:\VS\correlation_process\apiPea.csv'
    hardware_dir = r'E:\VS\correlation_process\hardwarePea.csv'
    intent_dir = r'E:\VS\correlation_process\intentPea.csv'
    permission_dir = r'E:\VS\correlation_process\permissionPea.csv'
    feature_dir = r'E:\VS\correlation_process\featurePeaTest.csv'
    apiList = []
    hardwareList = []
    intentList = []
    permissionList = []
    for api in pd.read_csv(api_dir).columns:
        if api == 'label':
            continue
        else:
            apiList.append(api.strip('\n'))
    for h in pd.read_csv(hardware_dir).columns:
        if h == 'label':
            continue
        else:
            hardwareList.append(h.strip('\n'))
    for i in pd.read_csv(intent_dir).columns:
        if i == 'label':
            continue
        else:
            intentList.append(i.strip('\n'))
    for p in pd.read_csv(permission_dir).columns:
        if p == 'label':
            continue
        else:
            permissionList.append(p.strip('\n'))
    featureList = apiList+hardwareList+intentList+permissionList
    with open(feature_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in featureList:
            writer.writerow([row])


if __name__ == '__main__':
    merge_feature()