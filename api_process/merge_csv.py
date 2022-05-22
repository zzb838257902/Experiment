import pandas as pd
import numpy as np
import glob
import csv
import os


# my_HeBingCSV1函数目的：将path目录下的所有csv文件合并成一个文件保存在SaveCSVName里面，注意函数适用范围为列名完全一致的文件
def my_HeBingCSV1(path, SaveCsvName):
    file_list = glob.glob(path+'*.csv')   # 把path目录下的所有后缀为.csv的文件找出来，存入一个列表
    print(u'totally %s CSV files found' % len(file_list))

    # 读取第一个CSV文件（包含表头）并写入
    csvOne = pd.read_csv(file_list[0],encoding='utf-8')   # encoding='utf-8-sig'与encoding='utf-8'的区别在于前者可以解决BOM等不可见但作怪的字符
    csvOne.to_csv(SaveCsvName, encoding="utf-8", index=False)     # index = False是为了去掉索引号

    # 循环读写列表中的CSV文件（不加表头）
    for i in range(1,len(file_list)):
        csvOne = pd.read_csv(file_list[i], encoding='utf-8')
        csvOne.to_csv(SaveCsvName, encoding="utf-8", index=False, header=False, mode='a+')    # mode = 'a+'表示附件方式写入，文件原有内容不会被清楚，否则会在原基础上覆盖
    print(u'Done!')
    """
    注意：若报错：UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 383-384: invalid continuation byte
        证明某个csv文件的编码并不是utf-8
    """

# 主函数
if __name__ == '__main__':
    path = r'E:\VS\merge_features\api/'  # r代表不转义
    SaveCsvName = path+'apiAll.csv'  # 合并后的数据要保存的地方
    my_HeBingCSV1(path, SaveCsvName)