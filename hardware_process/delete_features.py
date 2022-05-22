import csv
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from RFE_process import utils

def delete_f():
    rank_dir = r'E:\Samples\hardware_process\simWeight.csv'
    hardware = []
    hList = []
    h_3 = []
    h_rank = {}
    with open(rank_dir, 'r', newline='', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            hardware.append(row)
        for i in range(-3, 0):
            h_3.append(hardware[i])
        for p in h_3:
            if p[0] == 'true':
                p[0] = 'TRUE'
            if p[0] == 'false':
                p[0] = 'FALSE'
            if p[0] == '':
                continue
            hList.append(p[0])
        for i in range(-3, 0):
            del hardware[i]
        for row in hardware:
            h_rank[row[0]] = row[1]

    with open(rank_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in h_rank.items():
            writer.writerow(row)
    print(hList)
    drop_column(hList)
    RFE_Logistic(len(hardware))

def drop_column(hList):
    csv_dir = r'E:\Samples\hardware_process\all.csv'
    df = pd.read_csv(csv_dir)
    df = df.drop(hList, axis=1)
    df.to_csv(csv_dir, index=False, encoding='utf-8')


# def test():
#     csv_dir = r'E:\Samples\RFE_process\csv\all.csv'
#     txt_dir = r'E:\Samples\RFE_process\p.txt'
#     df = pd.read_csv(csv_dir)
#     print(len(df.columns))
#     f = open(txt_dir, 'a')
#     for p in df.columns:
#         f.write(p)
#         f.write('\n')
#     f.close()


def RFE_Logistic(number):
    csv_dir = r'E:\Samples\hardware_process\all.csv'
    acc_txt = r'E:\Samples\hardware_process\acc.txt'
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
    with open(acc_txt, 'a', newline='', encoding='utf-8') as f:
        f.write(str(acc)+'\r\n')
def start():
    for i in range(0, 49):
        if i == 0:
            RFE_Logistic(149)
        else:
            delete_f()

if __name__ == '__main__':
    start()