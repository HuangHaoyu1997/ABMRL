import numpy as np

def neighbor(xy,map_size,val_map,use_map,agent_pool,offset=5):
    '''
    计算邻里平均经济状况
    也就是xy周围一定范围内的土地/智能体的价值之平均
    若地块空闲，则计算地价，若地块有人居住，则计算智能体收入
    
    '''

    def is_agent(xy,use_map):
        '''
        判断当前地块是空地还是被智能体占据
        ID > 999是ID号
        ID = 0是空地
        '''
        x,y = xy
        ID = use_map[x,y]
        if ID > 999:   return ID
        elif ID < 999: return 0

    x,y = xy

    x_offset, y_offset = offset, offset
    dir = []
    for x in range(-x_offset,x_offset):
        for y in range(-y_offset,y_offset):
            dir.append([x,y])
    dir.pop(int(0.5*2*x_offset*2*y_offset+y_offset)) # 刨除(0,0)点
    
    
    sum = []
    for off_x, off_y in dir:
        if ((x+off_x) >= 0) and ((x+off_x)<map_size[0]) and ((y+off_y) >= 0) and ((y+off_y)<map_size[1]): # 不越界
            if use_map[x+off_x,y+off_y] >= 0: # 能访问
                id = is_agent([x+off_x,y+off_y],use_map) # 查找该地块上有没有人居住
                if id >= 1000: # 1000以上的id代表agent
                    sum.append(agent_pool[id].income) # 收入
                elif id < 1000: # 1000以下的id代表空地
                    sum.append(val_map[x+off_x,y+off_y]) # 地价
    return np.mean(sum)