import numpy as np
from numpy.core.defchararray import index

class Agent:
    def __init__(self, index, coord, income, WT, work_id) -> None:
        self.index = index
        self.coord = coord
        self.income = income # 初始income[100,1000]
        self.WT = WT # 迁居阈值
        self.work_id = work_id # 所属企业
        self.clas, self.RF, self.weight = self.def_class(max_income=1000) # 确定阶层,视域和权重
        
    def update_income(self, r , max_income):
        '''
        更新收入
        r：增长率
        max_income：收入上限
        '''
        r1, r2 = np.random.randint([20,20])-10 # r1，r2都是[-10,10]内的随机整数
        tmp = self.income + r * self.income * (1 - self.income/max_income)
        self.income = tmp + (r1 - r2)/2
        return self.income

    def def_class(self,max_income):
        
        '''
        根据相对收入的取值范围确定,相对收入=self.income/max_income
        [0,0.175)    低收入
        [0.175,0.35) 中低收入
        [0.35,0.5)   中等收入
        [0.5,0.75)   中高收入
        [0.75,1.0]   高收入
        权重的排序: 交通，地价，公共设施，环境，教育
        新权重排序：交通，通勤，地价
        '''
        IR = self.income / max_income
        if IR>=0 and IR<0.175:
            return 'Low', 2, np.array([0.4,0.5,0.1]) # np.array([0.3,0.6,0.05,0,0.05])
        elif IR>=0.175 and IR<0.35:
            return 'MediumLow', 3, np.array([0.4,0.4,0.2]) # np.array([0.25,0.45,0.15,0.05,0.1])
        elif IR>=0.35 and IR<0.5:
            return 'Medium', 4, np.array([0.3,0.4,0.3]) # np.array([0.1,0.1,0.4,0.3,0.1])
        elif IR>=0.5 and IR<0.75:
            return 'MediumHigh', 5, np.array([0.2,0.4,0.4]) # np.array([0.05,0.05,0.4,0.4,0.1])
        elif IR>=0.75 and IR<=1:
            return 'High', 6, np.array([0.2,0.2,0.6]) # np.array([0.05,0,0.45,0.45,0.05])
    