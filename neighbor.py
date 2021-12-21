import numpy as np

def neighbor(xy,map_size,val_map,use_map,agent_pool,offset=5):
    '''
    计算邻里平均经济状况
    也就是xy周围一定范围内的土地/智能体的价值之平均
    若地块空闲，则计算地价，若地块有人居住，则计算智能体收入
    
    '''
    # print(type(xy),type(map_size),type(val_map),type(use_map),type(agent_pool),type(offset))
    

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
            if use_map[x+off_x][y+off_y] >= 0: # 能访问
                
                id = use_map[x+off_x][y+off_y] if use_map[x+off_x][y+off_y]>999 else 0# 查找该地块上有没有人居住
                
                if id >= 1000: # 1000以上的id代表agent
                    sum.append(agent_pool[id].income) # 收入
                elif id < 1000: # 1000以下的id代表空地
                    sum.append(val_map[x+off_x][y+off_y]) # 地价
    return np.mean(sum)