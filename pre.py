# -*- coding: utf-8 -*-
# @Time : 2020/9/22 7:34 下午
# @Author : ligang
# @FileName: pre.py
# @Email   : ucasligang@163.com
# @Software: PyCharm
import os
import shutil


def merge_files():
    foldnames = os.listdir("data/DataSets/CUB_birds_2010/images")
    for foldname in foldnames:
        foldname = "data/DataSets/CUB_birds_2010/images/"+foldname
        for filename in os.listdir(foldname):
            # print("data/DataSets/CUB_birds_2010/images/"+filename)
            shutil.move(foldname+"/"+filename, "data/DataSets/CUB_birds_2010/images/"+filename)


    # foldname = os.path.exists("data/algae/变胞藻"):
    #     for filename in foldname:
    #         file.write(str(i)+" "+foldname+"\n")
    #         filenames = os.listdir(source_path+"/"+foldname)
    #         for filename in filenames:
    #             num = filename.split("_")[1].split(".")[0]
    #             temp = str(i)+"_"+str(num)   # 设置文件名
    #             if int(num) <= 90:
    #                 shutil.move(source_path+"/"+foldname+"/"+filename, "data/train"+"/"+temp+".png")
    #                 print(filename)
    #             elif int(num)>90 and int(num)<=120:
    #                 temp = str(i)+"_"+str(int(num)-90)
    #                 shutil.move(source_path+"/"+foldname+"/"+filename, "data/val"+"/"+temp+".png")
    #             else:
    #                 temp = str(i)+"_"+str(int(num)-120)
    #                 shutil.move(source_path+"/"+foldname+"/"+filename, "data/test"+"/"+temp+".png")
    #                 print(filename)
    #         i = i+1
    # file.close()

if __name__ == '__main__':
    merge_files()