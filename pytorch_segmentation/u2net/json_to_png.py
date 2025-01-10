import os
import glob
import shutil

path = r'D:\880data\880img-seg'  # 这里是指.json文件所在文件夹的路径
mask_path = 'D:\880data\880img-seg\mask'
# 如果不存在mask文件夹则创建
os.makedirs(mask_path, exist_ok=True)

for json in os.listdir(path):
    if json.endswith('.json'):
        print(json)
        json_file = glob.glob(os.path.join(path, json))  # 返回为列表
        print(json_file)
        # os.system("activate labelme")
        for file in json_file:  # 读取列表中路径
            os.system("labelme_export_json.exe %s" % file)
            print(file)

for eachfile in os.listdir(path):
    if os.path.isdir(os.path.join(path, eachfile)):
        label_path = os.path.join(path, eachfile, 'label.png')
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(mask_path, eachfile.split('_')[0] + '.png'))
            print(eachfile)
            print(eachfile.split('_')[0])
            print(eachfile + ' successfully moved')
