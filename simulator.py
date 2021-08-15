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
        self.class_ratio = np.array([0.1,0.2,0.4,0.2,0.1]) # 低,中低,中,中高,高
        self.income = np.array([[100,280],   # 低
                                [280,460],   # 中低
                                [460,640],   # 中
                                [640,820],   # 中高
                                [820,1000]]) # 高
        self.WT = 0.15 # 迁居阈值

        self.grid = Grid()
        self.agent_pool = []

    def step(self):
        # 单步执行函数
        # 改变收入，更新地价，执行每个智能体的迁居判断
        
        pass

    def change_income(self):
        # 更新智能体的收入
        
        pass

    def change_value(self):
        # 改变土地价值
        
        pass

    def gen_agent(self, N):
        # 生成新智能体
        # N:新增总人口
        number = N * self.class_ratio # 将新增总人口按收入阶层比例划分
        count = 0
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
                self.agent_pool.append(Agent(count,xy,income,self.WT))
                count += 1

    def render(self):
        
        pass