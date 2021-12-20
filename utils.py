# -*- coding: utf-8 -*- 
from ctypes import addressof
import cv2
import numpy as np
import pickle
import requests
import json
import urllib
import hashlib
import urllib.parse
from hashlib import md5
import sys
from xlrd import open_workbook # xlrd用于读取xld
import xlwt  # 用于写入xls


def load_map(dir):
        
    # 读入六环内地图
    BJ_map = cv2.imread(dir)
    BJ_map = cv2.cvtColor(BJ_map,cv2.COLOR_BGR2RGB)

    # 绿色地块0,97,0
    a = (BJ_map[:,:,0]>=0) * \
        (BJ_map[:,:,0] <= 10) * \
        (BJ_map[:,:,1] >= 95) * \
        (BJ_map[:,:,1] <= 100) * \
        (BJ_map[:,:,2] >= 0) * \
        (BJ_map[:,:,2] <= 10)

    use_map = np.zeros_like(a,dtype=np.uint8)
    use_map[a] = 1
    with open('BJ.pkl','wb') as f:
        pickle.dump(use_map,f)


def get_coord(address,ak='F471GPBS4iIPbPqI5t9EUg1Fc97aakOt'):
    url = 'http://api.map.baidu.com/geocoding/v3/?address={inputAddress}&output=json&ak={myAk}'.format(inputAddress=address,myAk=ak)  
        
    # sk = 'sD6fln89DbEvL1fMW190Z7KY0iuX65X0'
    
    res = requests.get(url)
    jd = json.loads(res.text)
    return jd['result']['location']


def load_xls():
    '''
    读取租房信息的地址
    即第1列-第4列
    '''
    workbook = open_workbook(r'D:\桌面\算法收藏夹\ABMRL\20年.xls')  # 打开xls文件
    sheet_name= workbook.sheet_names()  # 打印所有sheet名称，是个列表
    sheet = workbook.sheet_by_index(0)  # 根据sheet索引读取sheet中的所有内容
    sheet1= workbook.sheet_by_name('20年')  # 根据sheet名称读取sheet中的所有内容
    print(sheet.name, sheet.nrows, sheet.ncols)  # sheet的名称、行数、列数
    
    # column_0 = sheet.col_values(0)  # 第0列内容
    column_1 = sheet.col_values(1)  # 第1列内容
    column_2 = sheet.col_values(2)  # 第2列内容
    column_3 = sheet.col_values(3)  # 第3列内容
    column_4 = sheet.col_values(4)  # 第4列内容
    
    address = []
    for i in range(sheet.nrows):
        address.append(column_1[i]+column_2[i]+column_3[i]+column_4[i])
    with open('address.pkl','wb') as f:
        pickle.dump(add,f)
    return None
    # print(content)

def get_coords(add_file):
    '''
    批量计算经纬度
    '''
    with open(add_file,'rb') as f:
        address = pickle.load(f)
    
    coord = []
    for i in range(len(address)):
        if i == 0:
            continue
        add = address[i]
        tmp = get_coord(address=add)
        coord.append([tmp['lng'],tmp['lat']])
    with open('coord.pkl','wb') as f:
        pickle.dump(coord,f)


def cal_distance(A,B):
    '''
    给定AB两点经纬度，计算△x和△y
    假设北京地区为平面，球面距离近似为平面直线距离
    R = 6378.137km
    △x = (A点经度-B点经度) * R
    △y = (A点纬度-B点纬度) * R
    '''
    R = 6378.137 # km

    A_longitude, A_latitude = A
    B_longitude, B_latitude = B

    delta_x = R * (A_longitude - B_longitude)
    delta_y = R * (A_latitude - B_latitude)

    return [delta_x,delta_y]

    

if __name__ == '__main__':
    # load_map('BJ_map_crop.jpg')
    # g.load_map()
    
    # coord = get_coord(address='北京市海淀区上地十街10号')
    
    # load_xls()

    # get_coords(add_file='address.pkl')
    
    import matplotlib.pyplot as plt
    plt.imread('BJ_')


