import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import count
from numpy.testing._private.utils import rand
from agent import *
from grid import *

class env:
    def __init__(self) -> None:
        self.map_size = [200,200]
        self.init_pop = 500 # 初始人口
        self.max_pop = 6000 # 人口上限
        self.max_income = 5000 # 最高收入
        self.r = 0.005 # 收入增速
        self.R = 0.005 # 地价增速
        self.D = 0.1 # 地价折旧率
        self.c1 = 1.0 # 内部经济压力权重
        self.c2 = 1.0 # 内部社会压力权重
        self.ws = 1.0 # 外部压力权重
        self.wg = 1.0 # 内部压力权重
        self.a = 0.5 # 更新地价的权重
        self.class_ratio = np.array([0.1,0.2,0.4,0.2,0.1]) # 低,中低,中,中高,高
        # 各个阶层的初始收入上下限，需要实时更新
        self.income = np.array([[100,175],   # 低
                                [175,350],   # 中低
                                [350,500],   # 中
                                [500,750],   # 中高
                                [750,1000]]) # 高
        self.WT = 0.15 # 迁居阈值

        self.grid = Grid()
        self.agent_pool = {}
        self.pop_size = len(self.agent_pool)
        
        self.gen_agent(N=500)

    def step(self):
        # 单步执行函数
        # 改变收入，更新地价，执行每个智能体的迁居判断
        
        pass

    def change_income(self):
        # 更新智能体的收入
        self.income_list = []
        for a in self.agent_pool:
            income = a.update_income(self.r, self.max_income)
            self.income_list.append(income)
        max_income = np.max(self.income_list)
        # 更新智能体的阶层、视域和权重
        for a in self.agent_pool:
            a.def_class(max_income)


    def cal_in_pressure(self, ID, xy):
        '''
        计算内部社会经济压力
        S_h^t = c1*|I_h^t - V_h^t| + c2*|I_h^t - P_h^t|
        S是社会经济压力
        I是个人收入
        V是所占据土地的价值
        P是邻居平均经济状况
        c1、c2是系数
        '''
        x,y = xy
        income = self.agent_pool[ID].income
        price = self.grid.val_map[x,y] # 所占土地地价
        IorV = self.neighbor(xy) # 计算ID智能体的周围的价值
        P = np.mean(IorV)
        S = self.c1 * np.abs(income-price) + self.c2 * np.abs(income-P)
        return S

    def is_agent(self,xy):
        '''
        判断当前地块是空地还是被智能体占据
        ID > 999是ID号
        ID = 0是空地
        '''
        x,y = xy
        ID = self.grid.use_map[x,y]
        if ID > 999:
            return ID
        elif ID <= 999:
            return 0

    def neighbor(self,xy,offset=10):
        '''
        计算邻域内被占据的格点数（已出租房屋）占比
        总地块数量要排除不能访问的地块
        input：中心点坐标
        output：已占据格点数/总格点数
        '''
        x,y = xy
        dir = self.meshgrid(offset=[offset,offset])
        n_occupied, n_total = 0, 0
        for off_x,off_y in dir:
            if ((x+off_x) >= 0) and ((x+off_x)<self.map_size[0]) and ((y+off_y) >= 0) and ((y+off_y)<self.map_size[1]): # 不越界
                if self.grid.use_map[x+off_x,y+off_y] >= 0: # 能访问
                    n_total += 1
                    if self.grid.use_map[x+off_x,y+off_y] >= 1000: # 1000以上的id代表agent
                        n_occupied += 1
        return n_occupied / n_total

        
    
    def meshgrid(self,offset=[10,10],pop=True):
        '''
        生成类似于如下的矩阵
        ((-1,-1),(-1,0),(-1,1),
        (0,-1),         (0,1),
        (1,-1),(1,0),(1,1))
        '''
        x_offset,y_offset = offset
        dir = []
        for x in range(-x_offset,x_offset):
            for y in range(-y_offset,y_offset):
                dir.append([x,y])
        if pop:
            dir.pop(int(0.5*2*x_offset*2*y_offset+y_offset)) # 刨除(0,0)点

        return dir

    def neighbor_value(self,xy,offset=10):
        '''
        计算周围土地的价值
        input：xy坐标
        output：地价list和均值
        '''
        x,y = xy
        dir = self.meshgrid(offset=[offset,offset])
        n_value = []
        for off_x,off_y in dir:
            if ((x+off_x) >= 0) and ((x+off_x)<self.map_size[0]) and ((y+off_y) >= 0) and ((y+off_y)<self.map_size[1]): # 不越界
                if self.grid.use_map[x+off_x,y+off_y] >= 0: # 能访问
                    n_value.append(self.grid.val_map[x+off_x,y+off_y])
        return n_value, np.mean(n_value)

    def cal_out_pressure(self, xy, weight):
        '''
        计算给定坐标的外部居住环境吸引力
        G_h^t = w_env*E_env + w_edu*E_edu + w_tra*E_tra + w_pri*E_pri + w_con*E_con
        Agent权重的排序: 交通，地价，公共设施，环境，教育
        '''
        
        '''环境、教育、基础设施
        E_env = np.min(np.sqrt(np.sum((self.grid.env_xy-xy)**2,1)))
        E_env = np.exp(1-0.001*E_env) # 指数距离衰减函数

        E_edu = np.min(np.sqrt(np.sum((self.grid.edu_xy-xy)**2,1)))
        E_edu = np.exp(1-0.001*E_edu) # 指数距离衰减函数
        
        E_inf = np.min(np.sqrt(np.sum((self.grid.inf_xy-xy)**2,1)))
        E_inf = np.exp(1-0.001*E_inf) # 指数距离衰减函数
        '''
        x,y = xy
        # 交通        
        E_tra = np.min(np.sqrt(np.sum((self.grid.tra_xy-xy)**2,1)))
        E_tra = np.exp(1-0.001*E_tra) # 指数距离衰减函数
        
        # 通勤
        E_work = 

        # 房价
        # 实际上外部吸引力与房价并非正比关系，而是对不同人群有不同影响，人们可能更偏好比当前收入稍好一点的房屋，但不喜欢房价高出很多的房屋
        E_price = self.grid.val_map[x,y] 
        
        return weight * np.array([E_tra,0,E_inf,E_env,E_edu])

    def location_effect(self,ID,xy):
        '''
        计算区位效应
        U_t^h = w_g * G_h^t + w_s * (1 - S_h^t) + e_h^t
        e_h^t是随机变量
        xy可以是ID的坐标，也可以不是
        因此可以计算某一位置上的Agent与某一块土地的假设区位效应
        
        '''
        
        weight = self.agent_pool[ID].weight # 智能体的权重
        G = self.cal_out_pressure(xy, weight) # 外部居住环境吸引力
        S = self.cal_in_pressure(ID, xy)
        LE = self.wg*G + self.ws*(1-S) + 0.1*np.random.rand()
        return LE 

    def move(self,ID):
        '''
        判断迁居阈值，选择迁居地

        计算迁居意愿
        AW_h^t = U_b^t - U_h^t
        AW_h^t >= WT，迁居
        AW_h^t < WT，不迁居
        U_b^t是Agent视域内最佳居住点的区位效应
        '''
        # 为了方便计算，规定视域=3x3区域
        # 计算视域内的最优居住点
        xy = self.agent_pool[ID].coord
        is_occupied = []
        if np.min(xy+[0,1])>=0 and (np.max(xy+[0,1])<=(self.map_size[0]-1)):
            is_occupied.append(self.is_agent(xy+[0,1]))
        if np.min(xy+[0,-1])>=0 and (np.max(xy+[0,-1])<=(self.map_size[0]-1)):
            is_occupied.append(self.is_agent(xy+[0,-1]))
        if np.min(xy+[1,1])>=0 and (np.max(xy+[1,1])<=(self.map_size[0]-1)):
            is_occupied.append(self.is_agent(xy+[1,1]))
        if np.min(xy+[1,-1])>=0 and (np.max(xy+[1,-1])<=(self.map_size[0]-1)):
            is_occupied.append(self.is_agent(xy+[1,-1]))
        if np.min(xy+[1,0])>=0 and (np.max(xy+[1,0])<=(self.map_size[0]-1)):
            is_occupied.append(self.is_agent(xy+[1,0]))
        if np.min(xy+[-1,0])>=0 and (np.max(xy+[-1,0])<=(self.map_size[0]-1)):
            is_occupied.append(self.is_agent(xy+[-1,0]))
        if np.min(xy+[-1,-1])>=0 and (np.max(xy+[-1,-1])<=(self.map_size[0]-1)):
            is_occupied.append(self.is_agent(xy+[-1,-1]))
        if np.min(xy+[-1,1])>=0 and (np.max(xy+[-1,1])<=(self.map_size[0]-1)):
            is_occupied.append(self.is_agent(xy+[-1,1]))

        for ID in is_occupied:
            if ID>999:
                xy = self.agent_pool[ID].coord
                LE = self.location_effect(ID,xy)
            elif ID<999:
                pass

        

    def change_value(self):
        '''
        改变土地价值
        土地价值与上一时刻的地价、Agent所持有的财富、周边地价有关
        若土地被Agent占据：
        V_{ij}^{t+1} = a*(1+R)*V_{ij}^t 
                        + (1-a)*[(I_h^t+\sum_{m \in \Omega(\hat{ij})} V_m^t)/9]
        a是权重
        R是地价增速
        \sum_{m \in \Omega(\hat{ij})} V_m^t是智能体周围8个地块的地价之和

        若土地未被Agent占据，则价值随时间折旧：
        V_{ij}^{t+1} = a*(1-D)*V_{ij}^t 
                        + (1-a)*[(\sum_{m \in \Omega(ij)} V_m^t)/9]
        D是折旧率
        '''
        for x in range(self.map_size[0]):
            for y in range(self.map_size[1]):
                ID = self.is_agent(np.array([x,y]))
                V = self.grid.val_map[x,y] # 当前地价
                V_n = self.neighbor_value([x,y]) # 邻居的地价
                if ID: # 被智能体占据
                    income = self.agent_pool[ID].income # 当前收入
                    
                    V_n.append(income)
                    V_n = np.mean(V_n)
                    self.grid.val_map[x,y] = self.a*(1+self.R)*V + (1-self.a)*V_n
                elif not ID: # 空地
                    V_n.append(V)
                    V_n = np.mean(V_n)
                    self.grid.val_map[x,y] = self.a*(1-self.D)*V + (1-self.a)*V_n


    def gen_agent(self, N):
        # 生成新智能体
        # N:新增总人口
        number = N * self.class_ratio # 将新增总人口按收入阶层比例划分
        
        for i in range(5): # 为每个阶层产生新人口
            for _ in range(number[i]):
                xy = np.random.randint(self.map_size) # 随机生成位置
                x,y = xy
                while self.grid.use_map[x,y] != 0: # 如果该位置被占据或不能放置，则重新生成
                    xy = np.random.randint(self.map_size)
                    x,y = xy
                ID = self.pop_size + 1000 # 用ID为智能体建立索引
                self.grid.use_map[x,y] = ID # 在use_map中更新智能体的ID
                l,h = self.income[i] # 各个阶层的初始收入上限
                income = np.random.randint(low=l, high=h) # 随机收入
                self.agent_pool[ID] = Agent(ID,xy,income,self.WT)
                self.pop_size += 1

    def render(self):
        
        pass