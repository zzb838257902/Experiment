from androguard.misc import AnalyzeAPK
import os.path
import os
from bs4 import BeautifulSoup


permission_dir = r'E:\VS_2018\malware\permission_malware'
hardware_dir = r'E:\VS_2018\malware\hardware_malware'
intentfilter_action_dir = r'E:\VS_2018\malware\intentfilter_malware\action'
intentfilter_category_dir = r'E:\VS_2018\malware\intentfilter_malware\category'
component_activity_dir = r'E:\VS_2018\malware\component_malware\activity'
component_service_dir = r'E:\VS_2018\malware\component_malware\service'
component_provider_dir = r'E:\VS_2018\malware\component_malware\provider'
component_receiver_dir = r'E:\VS_2018\malware\component_malware\receiver'
api_dir = r"E:\VS_2018\malware\api_malware"
used_permission_dir = r"E:\VS_2018\malware\used_malware_permission"
dex_intent_dir = r'E:\VS_2018\malware\dex_malware_intent'

def Extract_Permission():
    rootdir = r"F:\2018malware"
    count = 0
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            try:
                a, d, dx = AnalyzeAPK(rootdir +"\\"+ filename)
            except Exception:
                continue
            print(filename)
            file_name = os.path.splitext(filename)[0]
            for p in a.get_permissions():
                f = open(permission_dir + "\\" + file_name + ".txt", 'a', encoding='utf-8')
                f.write(str(p))
                f.write('\n')
            try:
                strings = (a.get_android_manifest_axml().get_xml())
            except Exception:
                continue
            string = strings.decode('utf-8', 'ignore')
            soup = BeautifulSoup(string, "xml")
            for link in soup.find_all('uses-feature'):
                if link.get('android:name') != None:
                    f = open(hardware_dir + "\\" + file_name + ".txt", 'a', encoding='utf-8')
                    f.write(str(link.get('android:name')))
                    f.write('\n')

            for link in soup.find_all('action'):
                if link.get('android:name')!=None:
                    f = open(intentfilter_action_dir + "\\" + file_name + ".txt", 'a', encoding='utf-8')
                    f.write(str(link.get('android:name')))
                    f.write('\n')
            for link in soup.find_all('category'):
                if link.get('android:name') != None:
                    f = open(intentfilter_category_dir + "\\" + file_name + ".txt", 'a', encoding='utf-8')
                    f.write(str(link.get('android:name')))
                    f.write('\n')

            for link in soup.find_all('activity'):
                if link.get('android:permission')!=None:
                    f = open(component_activity_dir + "\\" + file_name + ".txt", 'a', encoding='utf-8')
                    f.write(str(link.get('android:permission')))
                    f.write('\n')
            for link in soup.find_all('service'):
                if link.get('android:permission') != None:
                    f = open(component_service_dir + "\\" + file_name + ".txt", 'a', encoding='utf-8')
                    f.write(str(link.get('android:permission')))
                    f.write('\n')
            for link in soup.find_all('receiver'):
                if link.get('android:permission') != None:
                    f = open(component_receiver_dir + "\\" + file_name + ".txt", 'a', encoding='utf-8')
                    f.write(str(link.get('android:permission')))
                    f.write('\n')
            for link in soup.find_all('provider'):
                if link.get('android:permission') != None:
                    f = open(component_provider_dir + "\\" + file_name + ".txt", 'a', encoding='utf-8')
                    f.write(str(link.get('android:permission')))
                    f.write('\n')

            for method_analysis in dx.get_android_api_usage():
                f = open(api_dir + "\\" + file_name + ".txt", 'a', encoding='utf-8')
                f.write(str(method_analysis.full_name))
                f.write('\n')

            for meth, perm in dx.get_permissions(a.get_effective_target_sdk_version()):
                for p in perm:
                    f = open(used_permission_dir + "\\" + file_name + ".txt", 'a', encoding='utf-8')
                    f.write(str(p))
                    f.write('\n')
            try:
                class_list = dx.get_classes()  # get all classes
            except Exception:
                continue
            # 输出当前类的数量
            print('class num:{0}'.format(len(class_list)))
            for class_item in class_list:
                class_name = class_item.name
                methods = class_item.get_methods()  # get all methods

                for m in methods:
                    raw_code_list = []
                    try:
                        for x in m.code.code.get_instructions():
                            raw_code_list.append(x)
                    except Exception as e:
                        continue
                    for line in raw_code_list:  # loop all method
                        if "intent" in line.get_output():
                            try:
                                intent = line.get_output().split(',')[1]
                            except Exception:
                                continue
                            if len((intent).split('.')) >= 4 and len((intent).split(" ")) <= 2 and ';' not in intent:
                                f = open(dex_intent_dir + "\\" + file_name + ".txt", 'a', encoding='utf-8')
                                f.write(str(intent))
                                f.write('\n')

            count += 1
            print("已完成:"+str(count))
    print("提取完成")

def Extract_HardWare():
    rootdir = r"E:\Samples\malware"
    count = 0
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            try:
                a, d, dx = AnalyzeAPK(rootdir + "\\" + filename)
            except Exception:
                continue
            strings = (a.get_android_manifest_axml().get_xml())
            string = strings.decode('utf-8', 'ignore')
            soup = BeautifulSoup(string, "xml")
            file_name = os.path.splitext(filename)[0]
            for link in soup.find_all('uses-feature'):
                if link.get('android:name') != None:
                    f = open(hardware_dir + "\\" + file_name + ".txt", 'a')
                    f.write(str(link.get('android:name')))
                    f.write('\n')
            count += 1
            print("已完成:" + str(count))
    print("hardware提取完成")

def Extract_Intent():
    rootdir = r"E:\Samples\malware"
    count = 0
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            try:
                a, d, dx = AnalyzeAPK(rootdir + "\\" + filename)
            except Exception:
                continue
            strings = (a.get_android_manifest_axml().get_xml())
            string = strings.decode('utf-8', 'ignore')
            # print(string)
            soup = BeautifulSoup(string, "xml")
            file_name = os.path.splitext(filename)[0]
            for link in soup.find_all('action'):
                if link.get('android:name')!=None:
                    f = open(intentfilter_action_dir + "\\" + file_name + ".txt", 'a')
                    f.write(str(link.get('android:name')))
                    f.write('\n')
            for link in soup.find_all('category'):
                if link.get('android:name') != None:
                    f = open(intentfilter_category_dir + "\\" + file_name + ".txt", 'a')
                    f.write(str(link.get('android:name')))
                    f.write('\n')
            count += 1
            print("已完成:" + str(count))
    print("intentfilter提取完成")

def Extract_Components():
    rootdir = r"E:\Samples\malware"
    count = 0
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            try:
                a, d, dx = AnalyzeAPK(rootdir + "\\" + filename)
            except Exception:
                continue
            strings = (a.get_android_manifest_axml().get_xml())
            string = strings.decode('utf-8', 'ignore')
            # print(string)
            soup = BeautifulSoup(string, "xml")
            file_name = os.path.splitext(filename)[0]
            for link in soup.find_all('activity'):
                if link.get('android:permission')!=None:
                    f = open(component_activity_dir + "\\" + file_name + ".txt", 'a')
                    f.write(str(link.get('android:permission')))
                    f.write('\n')
            for link in soup.find_all('service'):
                if link.get('android:permission') != None:
                    f = open(component_service_dir + "\\" + file_name + ".txt", 'a')
                    f.write(str(link.get('android:permission')))
                    f.write('\n')
            for link in soup.find_all('receiver'):
                if link.get('android:permission') != None:
                    f = open(component_receiver_dir + "\\" + file_name + ".txt", 'a')
                    f.write(str(link.get('android:permission')))
                    f.write('\n')
            for link in soup.find_all('provider'):
                if link.get('android:permission') != None:
                    f = open(component_provider_dir + "\\" + file_name + ".txt", 'a')
                    f.write(str(link.get('android:permission')))
                    f.write('\n')
            count += 1
            print("已完成:" + str(count))
    print("component提取完成")

def Extract_api():
    rootdir = r"E:\Samples\malware"
    count = 0
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            try:
                a, d, dx = AnalyzeAPK(rootdir + "\\" + filename)
            except Exception:
                continue
            file_name = os.path.splitext(filename)[0]
            for method_analysis in dx.get_android_api_usage():
                f = open(api_dir + "\\" + file_name + ".txt", 'a')
                f.write(str(method_analysis.full_name))
                f.write('\n')
            count += 1
            print("已完成:" + str(count))
    print("api提取完成")

def Extract_used_permission():
    rootdir = r"E:\Samples\malware"
    count = 0
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            try:
                a, d, dx = AnalyzeAPK(rootdir + "\\" + filename)
            except Exception:
                continue
            file_name = os.path.splitext(filename)[0]
            for meth, perm in dx.get_permissions(a.get_effective_target_sdk_version()):
                for p in perm:
                    f = open(used_permission_dir + "\\" + file_name + ".txt", 'a')
                    f.write(str(p))
                    f.write('\n')
            count += 1
            print("已完成:" + str(count))
    print("used_permission提取完成")

def Extract_dex_intent():
    rootdir = "E:\\Samples\\malware\\"
    count = 0
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            a, d, dx = AnalyzeAPK(rootdir + filename)
            # a  APK文件对象
            # d  DEX文件对象
            # dx 分析结果对象

            class_list = dx.get_classes()  # get all classes

            # 输出当前类的数量

            print('class num:{0}'.format(len(class_list)))
            file_name = os.path.splitext(filename)[0]
            for class_item in class_list:
                class_name = class_item.name
                methods = class_item.get_methods()  # get all methods

                for m in methods:
                    raw_code_list = []
                    try:
                        for x in m.code.code.get_instructions():
                            raw_code_list.append(x)
                    except Exception as e:
                        continue
                    for line in raw_code_list:  # loop all method
                        if "intent" in line.get_output():
                            intent = line.get_output().split(',')[1]
                            if len((intent).split('.')) >= 4 and len((intent).split(" ")) <= 2:
                                f = open(dex_intent_dir + "\\" + file_name + ".txt", 'a')
                                f.write(str(intent))
                                f.write('\n')
            count += 1
            print("已完成:" + str(count))
    print("dex_intent提取完成")

if __name__ == '__main__':
    Extract_Permission()
    # Extract_api()
    # Extract_Components()
    # Extract_HardWare()
    # Extract_Intent()
    # Extract_used_permission()
    # Extract_dex_intent()
