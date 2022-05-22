import os
import shutil

def merge(file1_dir, file2_dir):
    permissionList = []
    for parent, dirnames, file1_names in os.walk(file1_dir):
        for file1_name in file1_names:
            for par, dirs, file2_names in os.walk(file2_dir):
                permissionList.clear()
                if file1_name in file2_names:
                    with open(file1_dir+"\\"+file1_name, 'r', encoding='utf-8') as f:
                        for line in f:
                            permissionList.append(line.strip('\n'))
                    with open(file2_dir+"\\"+file1_name, 'r+', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip('\n')
                            if line not in permissionList:
                                permissionList.append(line)
                        f.truncate(0)
                    with open(file2_dir+"\\"+file1_name, 'a', encoding='utf-8') as f:
                        for p in permissionList:
                            f.write(p)
                            f.write('\n')
                else:
                    shutil.move(file1_dir+"\\"+file1_name, file2_dir+"\\"+file1_name)
    print("合并完成")

if __name__ == '__main__':
    # file1放文件少的路径，file2放文件多的路径
    file2 = r'E:\VS\benign\permission_all'
    file1 = r'E:\VS\benign\permission_non_repeat'
    file3 = r'E:\VS\benign\activity_non_repeat'
    file4 = r'E:\VS\benign\provider_non_repeat'
    file5 = r'E:\VS\benign\receiver_non_repeat'
    file6 = r'E:\VS\benign\service_non_repeat'
    file7 = r'E:\VS\benign\rp_non_repeat'
    merge(file7, file2)