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

import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d #引入scipy中的一维插值库
from scipy.interpolate import griddata#引入scipy中的二维插值库

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
    workbook = open_workbook('./data/20年房租.xls')  # 打开xls文件
    sheet_name = workbook.sheet_names()  # 打印所有sheet名称，是个列表
    sheet = workbook.sheet_by_index(0)  # 根据sheet索引读取sheet中的所有内容
    sheet1 = workbook.sheet_by_name('20年')  # 根据sheet名称读取sheet中的所有内容
    print(sheet.name, sheet.nrows, sheet.ncols)  # sheet的名称、行数、列数
    
    # column_0 = sheet.col_values(0)  # 第0列内容
    column_1 = sheet.col_values(1)  # 第1列内容
    column_2 = sheet.col_values(2)  # 第2列内容
    column_3 = sheet.col_values(3)  # 第3列内容
    column_4 = sheet.col_values(4)  # 第4列内容
    
    address = []
    for i in range(sheet.nrows):
        address.append(column_1[i]+column_2[i]+column_3[i]+column_4[i])
    with open('./data/20年房租address.pkl','wb') as f:
        pickle.dump(add,f)
    return None
    # print(content)

def get_coords_from_xls(add_file):
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

    delta_long = R * (A_longitude - B_longitude) * np.pi / 180
    delta_lat = R * (A_latitude - B_latitude) * np.pi / 180

    return [delta_long,delta_lat]

def cal_coord(A):
    '''
    根據如下兩點的經緯坐標，與其在圖像中的index，計算A點在圖像中的index
    A = [A_longitude, A_latitude]

    "天安门"在BJ_map_crop.jpg中的坐标是[11750,10250],经纬度是[116.403882,39.914824]
    "双清路与荷清路交叉口"在BJ_map_crop.jpg中的坐标是[7830,8250],经纬度是[116.343885,40.004397]

    图像矩阵的
    第1维，index增大，向南移动1個格點，緯度變小，位移 -dy km
    第1维，index減小，向北移动1個格點，緯度變大，位移  dy km

    第2維，index增大，向東移動1個格點，經度變大，位移  dx km
    第2維，index減小，向西移動1個格點，經度變小，位移 -dx km
    '''
    Tiananmen = [116.403882,39.914824]
    Tam_coord = [11750,10250]
    Shuangqingheqing = [116.343885,40.004397]
    Sqhq_coord = [7830,8250]

    # Wudaokou = [116.337742,39.992894]
    # Xizhimen = [116.355426,39.940474]
    dx = 0.003339417744561774 # 經度
    dy = 0.0025436787624554245

    dx1,dy1 = cal_distance(Tiananmen,A)
    dx2,dy2 = cal_distance(Shuangqingheqing,A)
    dx1 = int(0.5*dx1 + 0.5*dx2)
    dy1 = int(0.5*dy1 + 0.5*dy2)

    A_coord = [0,0]
    A_coord[0] = Tam_coord[0] + int(np.round(dy1/dy))
    A_coord[1] = Tam_coord[1] - int(np.round(dx1/dx))
    return A_coord

def get_coord_railway_station(file=r'C:\Users\44670\Documents\GitHub\ABMRL\data\地铁站点.xls'):
    '''
    從Excel file中读取地鐵站點的經緯度坐標，再將其轉化為圖像index

    '''
    max_x = 21290
    max_y = 20890

    workbook = open_workbook(file)  # 打开xls文件
    sheet_name= workbook.sheet_names()  # 打印所有sheet名称，是个列表
    sheet = workbook.sheet_by_index(0)  # 根据sheet索引读取sheet中的所有内容
    sheet1= workbook.sheet_by_name('Sheet1')  # 根据sheet名称读取sheet中的所有内容
    # print(sheet.name, sheet.nrows, sheet.ncols)  # sheet的名称、行数、列数
    
    # column_0 = sheet.col_values(0)  # 第0列内容
    longitude = sheet.col_values(4)  # 第4列内容
    latitude = sheet.col_values(5)  # 第5列内容
    longitude.pop(0) # pop表头
    latitude.pop(0) # pop表头
    
    railway_station_coord = []
    for i in range(len(longitude)):
        coord = cal_coord([longitude[i],latitude[i]])
        if coord[0]<0 or coord[0]>max_x or coord[1]<0 or coord[1]>max_y:
            continue
        railway_station_coord.append(coord)
    with open('railway_station_coord.pkl','wb') as f:
        pickle.dump(railway_station_coord,f)
    return railway_station_coord
        
    '''
    address = []
    for i in range(sheet.nrows):
        address.append(column_1[i]+column_2[i]+column_3[i]+column_4[i])
    with open('address.pkl','wb') as f:
        pickle.dump(add,f)
    return None
    '''
    
def plot_railway_station(r_coord_list):
    BJ = plt.imread('BJ_map_crop.jpg')
    for x,y in r_coord_list:
        BJ[x:x+100,y:y+100,:] = 0
    plt.imshow(BJ[0:20000,0:20000,:])
    plt.pause(5000)

def rent_index_price(file=r'./data/20年房租coord.pkl'):
    '''
    计算20年房租数据在图像矩阵上的index
    计算20年房租数据的单套均价
    根据(index,price)插值，得到size=(21290,20890)的房价分布图
    '''
    max_x = 21290
    max_y = 20890
    # 读取经纬坐标
    with open(file,'rb') as f:
        rent_2020 = pickle.load(f) # 65535条数据,经纬坐标
    
    # 读取房租价格
    file = r'./data/20年房租.xls'
    workbook = open_workbook(file)  # 打开xls文件
    sheet = workbook.sheet_by_index(0)  # 根据sheet索引读取sheet中的所有内容
    c16 = sheet.col_values(16)  # 成交套数
    c17 = sheet.col_values(17)  # 成交总面积
    c18 = sheet.col_values(18)  # 每平米均价
    c16.pop(0) # pop表头
    c17.pop(0) # pop表头
    c18.pop(0) # pop表头

    # 计算单房均价
    rent_price = np.array(c17) * np.array(c18) / np.array(c16) # 单套房成交均价，65535条数据
    
    # 整合数据，剔除无效数据
    rent_index_price = []
    for i in range(len(rent_2020)):
        # 计算index
        x,y = rent_2020[i]
        coord = cal_coord([x,y])
        # 判定有无超出仿真边界
        if coord[0]<0 or coord[0]>max_x or coord[1]<0 or coord[1]>max_y:
            continue
        # 记录合法数据
        rent_index_price.append([coord[0],coord[1],rent_price[i]])
    with open('./data/20年房租index+price.pkl','wb') as f:
        pickle.dump(rent_index_price,f)
    
    r_i_p = np.array(rent_index_price)
    xy = r_i_p[:,0:2]
    p = r_i_p[:,2]
    
    grid_x, grid_y = np.mgrid[0:max_x, 0:max_y]
    val_map_origin = griddata(xy, p, (grid_x, grid_y), method='cubic',fill_value=5)
    with open('./data/val_map_origin.pkl','wb') as f:
        pickle.dump(val_map_origin,f)
    
    print(val_map_origin.shape)




if __name__ == '__main__':
    
    # load_map('BJ_map_crop.jpg')
    # g.load_map()
    
    # coord = get_coord(address='北京市海淀区上地十街10号')
    
    # load_xls()

    # get_coords_from_xls(add_file='address.pkl')
    

    # r_coord_list = get_coord_railway_station()
    # print(len(r_coord_list))
    
    # plot_railway_station(r_coord_list)

    # rent_index_price()
    rent_index_price()

