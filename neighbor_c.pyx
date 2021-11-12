
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)

cdef float _neighbor(list xy, list map_size, list val_map, list use_map, dict agent_pool, int offset):
    '''
    计算邻里平均经济状况
    也就是xy周围一定范围内的土地/智能体的价值之平均
    若地块空闲，则计算地价，若地块有人居住，则计算智能体收入
    
    '''
    cdef int x, y, x_offset, y_offset, count
    cdef float id, sum
    cdef list dir

    x,y = xy
    x_offset = offset
    y_offset = offset
    sum = 0.
    count = 0
    dir = []

    for x in range(-x_offset,x_offset):
        for y in range(-y_offset,y_offset):
            dir.append([x,y])
    dir.pop(int(0.5*2*x_offset*2*y_offset+y_offset)) # 刨除(0,0)点
    
    for off_x, off_y in dir:
        if ((x+off_x) >= 0) and ((x+off_x)<map_size[0]) and ((y+off_y) >= 0) and ((y+off_y)<map_size[1]): # 不越界
            if use_map[x+off_x][y+off_y] >= 0: # 能访问
                id = use_map[x+off_x][y+off_y] if use_map[x+off_x][y+off_y]>999 else 0 # 查找该地块上有没有人居住
                count += 1
                if id >= 1000: # 1000以上的id代表agent
                    sum += agent_pool[id].income # 收入
                elif id < 1000: # 1000以下的id代表空地
                    sum += val_map[x+off_x][y+off_y] # 地价
    
    return sum/count

def neighbor(xy,map_size,val_map,use_map,agent_pool,offset):
    return _neighbor(xy,map_size,val_map,use_map,agent_pool,offset)