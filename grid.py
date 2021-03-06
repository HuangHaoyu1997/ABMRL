import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

from utils import rent_index_price

class Grid:
    def __init__(self, scale_factor=25) -> None:
        # self.inf_num = 20 # 基础设施数量
        # self.edu_num = 20 # 学校数量
        # self.init_value = 1000 # 初始地价在[0,1000]随机分布
        # self.env_num = 5 # 自然资源[河流，山脉]的数量
        self.work_num = 20 # 企业的数量
        # self.tra_num = 50 # 地铁站位置
        
        self.scale = scale_factor # 对原始地图的缩放系数
        self.scale_map()
        self.init_map()
        

    def scale_map(self):
        # 导入原尺寸use_map onehot版本，能住=1，不能住=0
        with open('./data/BJ_use_map_onehot.pkl','rb') as f:
            self.use_map_onehot = pickle.load(f)
        
        # 原尺寸
        self.origin_width = self.use_map_onehot.shape[0]
        self.origin_height = self.use_map_onehot.shape[1]

        # 缩小后的尺寸
        self.scale_width = int(np.round(self.origin_width / self.scale))
        self.scale_height = int(np.round(self.origin_height / self.scale))

        # resize use_map
        self.use_map = cv2.resize(self.use_map_onehot,(self.scale_height, self.scale_width))
        self.use_map = np.array(self.use_map,dtype=np.int32)
        # print(self.use_map.shape,self.scale_height,self.scale_width)
        self.use_map -= 1
        plt.imsave('./img/BJ_scale_use_map.png',self.use_map)
        self.map_size = list(self.use_map.shape)
        # print(self.use_map.min(), self.use_map.max(), type(self.use_map[0,0]))


    def init_map(self):
        # 初始化各图层

        # 基础设施
        # self.inf_map, self.inf_xy = self.init_inf_map()
        
        # 教育
        # self.edu_map, self.edu_xy = self.init_edu_map() 
        
        # 地价
        self.val_map = self.init_val_map()
        
        # 交通地图
        self.tra_map, self.tra_xy = self.init_tra_map()
        
        # 工作地图
        self.work_map, self.work_xy = self.init_work_map()
        
        # 环境
        # self.env_map, self.env_xy = self.init_env_map()

        # 土地利用
        self.use_map = self.init_use_map()
        
    def init_work_map(self,):
        # 初始化工作地点的地图
        work_map = np.zeros(self.map_size)
        xy = np.random.randint(np.ones((self.work_num,2))*[self.map_size[0]-5,self.map_size[1]-5]) # 取480保证企业不在地图边缘
        for xxyy in xy:
            x,y = xxyy
            work_map[x,y] = 1
        return work_map, xy
    '''
    def init_inf_map(self):
        # 初始化基础设施地图
        inf_map = np.zeros(self.map_size)
        xy = np.random.randint(np.ones((self.inf_num,2))*[480,480]) # 取480保证基础设施不在地图边缘
        for xxyy in xy:
            x,y = xxyy
            inf_map[x,y] = 1
        return inf_map, xy

    def init_edu_map(self):
        # 初始化教育地图, 教育资源伴随基础设施附近生成
        edu_map = np.zeros(self.map_size)
        xy = self.inf_xy + np.random.randint(np.ones((self.edu_num,2))*[10,10])-5
        xy[xy<0] = 0
        for xxyy in xy:
            x,y = xxyy
            edu_map[x,y] = 1
        return edu_map, xy
    '''
    def init_val_map(self):
        # 读取最大尺寸的地价分布图
        with open('./data/val_f32.pkl','rb') as f:
            self.val_map_origin = pickle.load(f)
        
        val_map = cv2.resize(self.val_map_origin,(self.scale_height ,self.scale_width))

        
        '''
        # val_map = np.zeros_like(self.use_map,dtype=np.float64)
        
        # 地图尺度缩小之后，原坐标会重合，故一个格点会叠加多个价格，需要都收集起来，取均值
        price_index = {}
        for x,y,p in rent_index_price:
            x = int(x / self.scale)
            y = int(y / self.scale)
            if (x,y) not in list(price_index.keys()):
                price_index[x,y] = []
            price_index[x,y].append(p)
        price_item = list(price_index.items())
        price = []
        location = []
        for i in price_item: 
            x,y = i[0]
            p = np.array(i[1]).mean()
            price.append(p)
            location.append([x,y])
            val_map[x,y] = p
        '''
        return val_map.tolist()
    
    def init_tra_map(self):
        
        tra_map = np.zeros_like(self.use_map)
        # 导入地铁站的index
        with open('./data/railway_station_coord.pkl','rb') as f:
            xy = pickle.load(f)
        # print('x,',len(xy))
        tra_xy = []
        for xxyy in xy:
            x,y = xxyy
            # x = int(self.scale_width * x / self.origin_width)
            # y = int(self.scale_height * y / self.origin_height)
            x = int(x / self.scale)
            y = int(y / self.scale)
            # 在经过缩放之后，距离过近的地铁站会重合为1个点，需要判断是否重复append
            if [x,y] not in tra_xy:
                tra_xy.append([x,y])
            tra_map[x,y] = 1
        # a = tra_map ==1
        # print('tra:',a.sum())
        '''
        for xy in self.inf_xy:
            x,y = xy
            tra_map[x,y] = 3   # 市中心拥有最高交通等级
            tra_map[x-1,y] = 2 # 周边区域次之
            tra_map[x,y-1] = 2
            tra_map[x+1,y] = 2
            tra_map[x,y+1] = 2
            tra_map[x-1,y-1] = 2
            tra_map[x+1,y+1] = 2
            tra_map[x-1,y+1] = 2
            tra_map[x+1,y-1] = 2
        '''

        '''
        xy = np.zeros((self.map_size[0],2)) # 解析曲线表示一条道路
        for x in range(self.map_size[0]):
            y = np.sin(0.1*x)
            xy[x,0] = x, xy[x,1] = y
            tra_map[x,y] = 1
        '''
        
        return tra_map, np.array(tra_xy)

    '''
    def init_env_map(self):
        # 初始化自然资源地图
        env_map = np.zeros(self.map_size)
        env_xy = []
        for _ in range(self.env_num):
            xy = np.random.randint(self.map_size)
            while xy in self.inf_xy or xy in self.edu_xy:
                xy = np.random.randint(self.map_size)
            env_xy.append(xy)
            x,y = xy
            env_map[x,y] = 1
        env_xy = np.array(env_xy)
        return env_map, env_xy
    '''
    
    def init_use_map(self):
        # 2:智能体
        # use_map = np.zeros(self.map_size).tolist()
        use_map = self.use_map.tolist()
        for xy in self.tra_xy:
            x,y = xy
            use_map[x][y] = -100 # 地铁站点不可占用
        for xy in self.work_xy:
            x,y = xy
            use_map[x][y] = -200 # 工作地点不可占用
        return use_map

if __name__ == '__main__':
    g = Grid()
    # g.scale_map()
    # g.load_map()
    