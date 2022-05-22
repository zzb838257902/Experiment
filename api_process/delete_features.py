import csv
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from RFE_process import utils

def delete_f():
    rank_dir = r'E:\Samples\api_process\apiWeight_new.csv'
    api = []
    apiList = []
    api_48 = []
    api_rank = {}
    with open(rank_dir, 'r', newline='', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            api.append(row)
        for i in range(-48, 0):
            api_48.append(api[i])
        for a in api_48:
            if a[0] == 'true':
                a[0] = 'TRUE'
            if a[0] == 'false':
                a[0] = 'FALSE'
            if a[0] == '':
                continue
            apiList.append(a[0].strip('\n'))
        for i in range(-48, 0):
            del api[i]
        for row in api:
            api_rank[row[0]] = row[1]

    with open(rank_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in api_rank.items():
            writer.writerow(row)
    print(apiList)
    drop_column(apiList)
    RFE_Logistic(len(api))

def drop_column(pList):
    csv_dir = r'E:\Samples\api_process\all.csv'
    df = pd.read_csv(csv_dir)
    df = df.drop(pList, axis=1)
    df.to_csv(csv_dir, index=False, encoding='utf-8')


# def test():
#     csv_dir = r'E:\Samples\api_process\all.csv'
#     txt_dir = r'E:\Samples\api_process\p.txt'
#     df = pd.read_csv(csv_dir)
#     print(len(df.columns))
#     f = open(txt_dir, 'a')
#     for p in df.columns:
#         f.write(p)
#         f.write('\n')
#     f.close()


def RFE_Logistic(number):
    csv_dir = r'E:\Samples\api_process\all.csv'

    x_train, x_test, y_train, y_test = utils.load_handle_data(csv_dir)
    # Feature extraction
    model = LogisticRegression(solver='lbfgs')
    rfe = RFE(model, number)
    fit = rfe.fit(x_train, y_train)
    acc = fit.score(x_test, y_test)
    # print("Num Features: %d"% fit.n_features_)
    # print("Selected Features: %s"% fit.support_)
    # print("Feature Ranking: %s"% fit.ranking_)
    print("Model accuracy: %f" % acc)

def start():
    for i in range(0, 100):
        if i == 0:
            RFE_Logistic(4825)
        else:
            delete_f()

if __name__ == '__main__':
    start()