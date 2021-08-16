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
        self.income = np.array([[100,280],   # 低
                                [280,460],   # 中低
                                [460,640],   # 中
                                [640,820],   # 中高
                                [820,1000]]) # 高
        self.WT = 0.15 # 迁居阈值

        self.grid = Grid()
        self.agent_pool = []
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
                xy = np.random.randint(self.map_size)
                x,y = xy
                while self.grid.use_map[x,y] != 0:
                    xy = np.random.randint(self.map_size)
                    x,y = xy
                self.grid.use_map[x,y] = 2
                l,h = self.income[0]
                income = np.random.randint(low=l, high=h)
                self.agent_pool.append(Agent(self.pop_size,xy,income,self.WT))
                self.pop_size += 1

    def render(self):
        
        pass