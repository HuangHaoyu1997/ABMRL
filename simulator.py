import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import count
from agent import *
from grid import *

class env:
    def __init__(self) -> None:
        self.map_size = [500,500]
        self.init_pop = 500 # 初始人口
        self.max_pop = 6000 # 人口上限
        self.max_income = 5000 # 最高收入
        self.r = 0.005 # 收入增速
        self.R = 0.005 # 地价增速
        self.D = 0.1 # 地价折旧率
        self.class_ratio = np.array([0.1,0.2,0.4,0.2,0.1]) # 低,中低,中,中高,高
        # 各个阶层的初始收入上下限，需要实时更新
        self.income = np.array([[100,280],   # 低
                                [280,460],   # 中低
                                [460,640],   # 中
                                [640,820],   # 中高
                                [820,1000]]) # 高
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
        for a in self.agent_pool:
            income = a.update_income(self.r, self.max_income)

    def cal_in_pressure(self, ID):
        '''
        计算内部社会经济压力
        S_h^t = c1*|I_h^t - V_h^t| + c2*|I_h^t - P_h^t|
        S是社会经济压力
        I是个人收入
        V是所占据土地的价值
        P是邻居平均经济状况
        c1、c2是系数
        '''
        x,y = self.agent_pool[ID].coord
        price = self.grid.val_map[x,y] # 所占土地地价
        self.grid.use_map
        pass

    def is_agent(self,xy):
        # 判断当前地块是空地还是被智能体占据
        x,y = xy
        ID = self.grid.use_map[x,y]
        if ID > 999:
            return ID
        elif ID <= 999:
            return 0

    def neighbor(self,ID):
        # 计算智能体的邻居的价值，或相邻土地的价值
        x,y = self.agent_pool[ID].coord
        x_max, y_max = self.map_size
        # 考虑智能体处于地图上的四个角
        if x == 0 and y == y_max-1:
            tmp = []
            if self.is_agent(x+1,y): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x+1,y)].income)
            elif not self.is_agent(x+1,y): # 空地
                tmp.append(self.grid.val_map[x+1,y])

            if self.is_agent(x+1,y-1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x+1,y-1)].income)
            elif not self.is_agent(x+1,y-1): # 空地
                tmp.append(self.grid.val_map[x+1,y-1])
            
            if self.is_agent(x,y-1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x,y-1)].income)
            elif not self.is_agent(x,y-1): # 空地
                tmp.append(self.grid.val_map[x,y-1])
            return tmp

        elif x == x_max-1 and y == 0:
            tmp = []
            if self.is_agent(x-1,y): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x-1,y)].income)
            elif not self.is_agent(x-1,y): # 空地
                tmp.append(self.grid.val_map[x-1,y])

            if self.is_agent(x-1,y+1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x-1,y+1)].income)
            elif not self.is_agent(x-1,y+1): # 空地
                tmp.append(self.grid.val_map[x-1,y+1])
            
            if self.is_agent(x,y+1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x,y+1)].income)
            elif not self.is_agent(x,y+1): # 空地
                tmp.append(self.grid.val_map[x,y+1])
            return tmp

        elif x == x_max-1 and y == y_max-1:
            tmp = []
            if self.is_agent(x-1,y): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x-1,y)].income)
            elif not self.is_agent(x-1,y): # 空地
                tmp.append(self.grid.val_map[x-1,y])

            if self.is_agent(x-1,y-1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x-1,y-1)].income)
            elif not self.is_agent(x-1,y-1): # 空地
                tmp.append(self.grid.val_map[x-1,y-1])
            
            if self.is_agent(x,y-1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x,y-1)].income)
            elif not self.is_agent(x,y-1): # 空地
                tmp.append(self.grid.val_map[x,y-1])
            return tmp

        elif x == 0 and y == 0:
            tmp = []
            if self.is_agent(x+1,y): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x+1,y)].income)
            elif not self.is_agent(x+1,y): # 空地
                tmp.append(self.grid.val_map[x+1,y])

            if self.is_agent(x+1,y+1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x+1,y+1)].income)
            elif not self.is_agent(x+1,y+1): # 空地
                tmp.append(self.grid.val_map[x+1,y+1])
            
            if self.is_agent(x,y+1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x,y+1)].income)
            elif not self.is_agent(x,y+1): # 空地
                tmp.append(self.grid.val_map[x,y+1])
            return tmp
        
        # 考虑智能体处于地图上的四条边，不包含四角
        elif x == 0 and (y>0 and y<y_max-1):
            tmp = []
            if self.is_agent(x+1,y): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x+1,y)].income)
            elif not self.is_agent(x+1,y): # 空地
                tmp.append(self.grid.val_map[x+1,y])

            if self.is_agent(x+1,y+1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x+1,y+1)].income)
            elif not self.is_agent(x+1,y+1): # 空地
                tmp.append(self.grid.val_map[x+1,y+1])
            
            if self.is_agent(x+1,y-1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x+1,y-1)].income)
            elif not self.is_agent(x+1,y-1): # 空地
                tmp.append(self.grid.val_map[x+1,y-1])

            if self.is_agent(x,y+1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x,y+1)].income)
            elif not self.is_agent(x,y+1): # 空地
                tmp.append(self.grid.val_map[x,y+1])
            
            if self.is_agent(x,y-1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x,y-1)].income)
            elif not self.is_agent(x,y-1): # 空地
                tmp.append(self.grid.val_map[x,y-1])
            return tmp
        elif x == x_max-1 and (y>0 and y<y_max-1):
            tmp = []
            if self.is_agent(x-1,y): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x-1,y)].income)
            elif not self.is_agent(x-1,y): # 空地
                tmp.append(self.grid.val_map[x-1,y])

            if self.is_agent(x-1,y+1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x-1,y+1)].income)
            elif not self.is_agent(x-1,y+1): # 空地
                tmp.append(self.grid.val_map[x-1,y+1])
            
            if self.is_agent(x-1,y-1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x-1,y-1)].income)
            elif not self.is_agent(x-1,y-1): # 空地
                tmp.append(self.grid.val_map[x-1,y-1])

            if self.is_agent(x,y+1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x,y+1)].income)
            elif not self.is_agent(x,y+1): # 空地
                tmp.append(self.grid.val_map[x,y+1])
            
            if self.is_agent(x,y-1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x,y-1)].income)
            elif not self.is_agent(x,y-1): # 空地
                tmp.append(self.grid.val_map[x,y-1])
            return tmp
        elif (x>0 and x<x_max-1) and y == 0:
            tmp = []
            if self.is_agent(x-1,y): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x-1,y)].income)
            elif not self.is_agent(x-1,y): # 空地
                tmp.append(self.grid.val_map[x-1,y])

            if self.is_agent(x+1,y): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x+1,y)].income)
            elif not self.is_agent(x+1,y): # 空地
                tmp.append(self.grid.val_map[x+1,y])
            
            if self.is_agent(x-1,y+1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x-1,y+1)].income)
            elif not self.is_agent(x-1,y+1): # 空地
                tmp.append(self.grid.val_map[x-1,y+1])

            if self.is_agent(x,y+1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x,y+1)].income)
            elif not self.is_agent(x,y+1): # 空地
                tmp.append(self.grid.val_map[x,y+1])
            
            if self.is_agent(x+1,y+1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x+1,y+1)].income)
            elif not self.is_agent(x+1,y+1): # 空地
                tmp.append(self.grid.val_map[x+1,y+1])
            return tmp
        elif (x>0 and x<x_max-1) and y == y_max-1:
            tmp = []
            if self.is_agent(x-1,y): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x-1,y)].income)
            elif not self.is_agent(x-1,y): # 空地
                tmp.append(self.grid.val_map[x-1,y])

            if self.is_agent(x+1,y): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x+1,y)].income)
            elif not self.is_agent(x+1,y): # 空地
                tmp.append(self.grid.val_map[x+1,y])
            
            if self.is_agent(x-1,y-1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x-1,y-1)].income)
            elif not self.is_agent(x-1,y-1): # 空地
                tmp.append(self.grid.val_map[x-1,y-1])

            if self.is_agent(x,y-1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x,y-1)].income)
            elif not self.is_agent(x,y-1): # 空地
                tmp.append(self.grid.val_map[x,y-1])
            
            if self.is_agent(x+1,y-1): # 被智能体占据
                tmp.append(self.agent_pool[self.is_agent(x+1,y-1)].income)
            elif not self.is_agent(x+1,y-1): # 空地
                tmp.append(self.grid.val_map[x+1,y-1])
            return tmp

        else:
            

        use_map[x,y]

    def cal_out_pressure(self):
        '''
        计算外部居住环境吸引力
        G_h^t = w_env*E_env + w_edu*E_edu + w_tra*E_tra + w_pri*E_pri + w_con*E_con
        
        '''
        pass

    def move(self):
        '''
        判断迁居阈值，选择迁居地

        计算区位效应
        U_t^h = w_g * G_h^t + w_s * (1 - S_h^t) + e_h^t
        e_h^t是随机变量
        计算迁居意愿
        AW_h^t = U_b^t - U_h^t
        AW_h^t >= WT，迁居
        AW_h^t < WT，不迁居
        U_b^t是Agent视域内最佳居住点的区位效应
        '''
        

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
        
        
        pass

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