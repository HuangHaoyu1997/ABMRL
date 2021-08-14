import numpy as np

class Grid:
    def __init__(self) -> None:
        self.inf_num = 20 # 基础设施数量
        self.edu_num = 30 # 学校数量
        self.init_value = 1000 # 初始地价在[0,1000]随机分布
        self.env_num = [20,30] # 自然资源[河流，山脉]的数量
        
        self.map_size = [500,500]
        self.init_map()
    
    def init_map(self):
        # 初始化各图层

        # 基础设施
        self.inf_map, self.inf_xy = self.init_inf_map()
        
        # 教育
        self.edu_map, self.edu_xy = self.init_edu_map() 
        
        # 地价
        self.val_map = self.init_val_map()

        # 交通地图
        self.tra_map = self.init_tra_map()
        
        self.env_map = 
        self.val_map = None # 地价
        self.use_map = None # 土地利用
        
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
    
    def init_val_map(self):
        return np.random.randint(np.ones(self.map_size)*self.init_value)
    
    def init_tra_map(self):
        # 基础设施周围具有较好的交通等级
        tra_map = np.ones(self.map_size)
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
        return tra_map

    def init_env_map(self):
        # 初始化自然资源地图
        edu_map = np.zeros(self.map_size)
        xy = self.inf_xy + np.random.randint(np.ones((self.edu_num,2))*[10,10])-5
        xy[xy<0] = 0
        for xxyy in xy:
            x,y = xxyy
            edu_map[x,y] = 1
        return edu_map, xy