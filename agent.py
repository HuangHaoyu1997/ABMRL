import numpy as np

class agent:
    def __init__(self,income, WT) -> None:
        
        self.income = income # 初始income[100,1000]
        self.WT = WT # 迁居阈值
        self.class = self.def_class() 
        self.RF = self.receptive_field() # RF是receptive field
        self.weight = self.def_weight()
        self.out_pressure = None # 外部压力
        self.inner_pressure = None # 内部压力
        self.R = None # 收入增速

    def cal_out_pressure(self):
        pass

    def cal_inner_pressure(self):
        pass

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
        if self.class == 'High':
            return 6
        elif self.class == 'MediumHigh':
            return 5
        elif self.class == 'Medium':
            return 4
        elif self.class == 'MediumLow':
            return 3
        elif self.class == 'Low':
            return 2
    
    def def_weight(self):
        # 交通，地价，公共设施，环境，教育
        if self.class == 'High':
            return np.array([0.05,0,0.45,0.45,0.05])
        elif self.class == 'MediumHigh':
            return np.array([0.05,0.05,0.4,0.4,0.1])
        elif self.class == 'Medium':
            return np.array([0.1,0.1,0.4,0.3,0.1])
        elif self.class == 'MediumLow':
            return np.array([0.25,0.45,0.15,0.05,0.1])
        elif self.class == 'Low':
            return np.array([0.3,0.6,0.05,0,0.05])
    
    def move(self):
        # 判断迁居阈值，选择迁居地
        pass
    