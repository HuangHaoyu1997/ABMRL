import numpy as np
from numpy.core.defchararray import index

class Agent:
    def __init__(self, index, coord, income, WT) -> None:
        self.index = index
        self.coord = coord
        self.income = income # 初始income[100,1000]
        self.WT = WT # 迁居阈值
        self.clas = self.def_class() 
        self.RF = self.receptive_field() # RF是receptive field
        self.weight = self.def_weight()
        self.out_pressure = None # 外部压力
        self.inner_pressure = None # 内部压力

    def cal_out_pressure(self):
        '''
        计算内部社会经济压力
        S_h^t = c1*|I_h^t - V_h^t| + c2*|I_h^t - P_h^t|
        S是社会经济压力
        I是个人收入
        V是所占据土地的价值
        P是邻居平均经济状况
        c1、c2是系数
        '''

        pass

    def cal_inner_pressure(self):
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
        # 

        pass

    def update_income(self, r , max_income):
        r1, r2 = np.random.randint([20,20])-10
        tmp = self.income + r * self.income * (1 - self.income/max_income)
        self.income = tmp + (r1 - r2)/2
        return self.income

    def def_class(self):
        
        '''
        根据相对收入的取值范围确定
        [0,0.175)    低收入
        [0.175,0.35) 中低收入
        [0.35,0.5)   中等收入
        [0.5,0.75)   中高收入
        [0.75,1.0]   高收入
        '''
        return 'High', 6
        

    def receptive_field(self):
        if self.clas == 'High':
            return 6
        elif self.clas == 'MediumHigh':
            return 5
        elif self.clas == 'Medium':
            return 4
        elif self.clas == 'MediumLow':
            return 3
        elif self.clas == 'Low':
            return 2
    
    def def_weight(self):
        # 交通，地价，公共设施，环境，教育
        if self.clas == 'High':
            return np.array([0.05,0,0.45,0.45,0.05])
        elif self.clas == 'MediumHigh':
            return np.array([0.05,0.05,0.4,0.4,0.1])
        elif self.clas == 'Medium':
            return np.array([0.1,0.1,0.4,0.3,0.1])
        elif self.clas == 'MediumLow':
            return np.array([0.25,0.45,0.15,0.05,0.1])
        elif self.clas == 'Low':
            return np.array([0.3,0.6,0.05,0,0.05])
    
    
    