cimport cython
# import numpy as np
# cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)

cdef float _neighbor_value(list xy,list use_map,list val_map,list map_size,int offset):
    '''
    计算周围土地的价值
    input：xy坐标
    output：地价list和均值
    '''
    cdef int x, y, x_offset, y_offset, count
    cdef float value
    cdef list dir

    x,y = xy
    x_offset = offset
    y_offset = offset
    dir = []

    for x in range(-x_offset,x_offset):
        for y in range(-y_offset,y_offset):
            dir.append([x,y])
    dir.pop(int(0.5*2*x_offset*2*y_offset+y_offset)) # 刨除(0,0)点

    value = 0.
    count = 0
    # n_value = []
    for off_x,off_y in dir:
        if  ((x+off_x) >= 0)        and \
            ((x+off_x)<map_size[0]) and \
            ((y+off_y) >= 0)        and \
            ((y+off_y)<map_size[1]) and \
            (use_map[x+off_x][y+off_y] >= 0): # 不越界,能访问
            
            value += val_map[x+off_x][y+off_y]
            count += 1
            # n_value.append(self.grid.val_map[x+off_x,y+off_y])
    # print('count:',count)
    return value/count # n_value, np.mean(n_value)

def neighbor_value(xy,use_map,val_map,map_size,offset=25):
    return _neighbor_value(xy,use_map,val_map,map_size,offset)