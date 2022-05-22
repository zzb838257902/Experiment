#
# 数据加载、处理
#

import pandas as pd
from sklearn.model_selection import train_test_split

# 加载和处理数据集


from sklearn.utils import shuffle


def load_handle_data(filename):
    '''
    @param filename : 权限数据集 csv 文件名称
    '''
    dataset = pd.read_csv(filename)
    #dataset = shuffle(dataset, random_state=1)
    # 权限
    X = dataset.drop('label', axis=1)
    # 标签数据
    Y = dataset.label
    # 拆分数据集为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    return x_train, x_test, y_train, y_test


# 更具粒子的位置，提取特征作为子特征集
def extract_data(t, x, dataset):
    index = []
    for i in range(len(x)):
        if 1 == x[i]:
            index.append(i)
    # 根据粒子位置提取权限子集
    X = dataset.iloc[:, index]
    Y = dataset.vir
    # print(t)
    # print(x)
    # print(X)
    # print(Y)
    sub_x_train, sub_x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    return sub_x_train, sub_x_test, y_train, y_test
