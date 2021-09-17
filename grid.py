import numpy as np

class Grid:
    def __init__(self, map_size=[500,500]) -> None:
        self.inf_num = 20 # 基础设施数量
        self.edu_num = 20 # 学校数量
        self.init_value = 1000 # 初始地价在[0,1000]随机分布
        self.env_num = 5 # 自然资源[河流，山脉]的数量
        self.work_num = 20 # 企业的数量
        
        self.map_size = map_size
        self.init_map()
    
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
        xy = np.random.randint(np.ones((self.work_num,2))*[480,480]) # 取480保证基础设施不在地图边缘
        for xxyy in xy:
            x,y = xxyy
            work_map[x,y] = 1
        return work_map, xy

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
        tra_map = np.zeros(self.map_size)
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
        xy = np.zeros((self.map_size[0],2))
        for x in range(self.map_size[0]):
            y = np.sin(0.1*x)
            xy[x,0] = x, xy[x,1] = y
            tra_map[x,y] = 1
        return tra_map, xy

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
    
    def init_use_map(self):
        # 1:自然资源
        # 2:智能体
        use_map = np.zeros(self.map_size)
        for xy in self.env_xy:
            x,y = xy
            use_map[x,y] = -1
        return use_map
    
    