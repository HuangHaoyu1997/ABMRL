import numpy as np
import matplotlib.pyplot as plt

class env:
    def __init__(self) -> None:
        self.map_size = [500,500]
        self.init_pop = 500 # 初始人口
        self.max_pop = 6000 # 人口上限
        self.max_income = 5000 # 最高收入
        

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

    def gen_agent(self):
        # 生成新智能体
        pass