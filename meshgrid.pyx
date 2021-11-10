cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list _meshgrid(list offset,int pop):
    '''
    生成类似于如下的矩阵
    ((-1,-1),(-1,0),(-1,1),
    (0,-1),         (0,1),
    (1,-1),(1,0),(1,1))
    '''
    cdef int x_offset, y_offset
    cdef list dir

    x_offset, y_offset = offset
    dir = []
    for x in range(-x_offset,x_offset):
        for y in range(-y_offset,y_offset):
            dir.append([x,y])
    if pop==1:
        dir.pop(int(0.5*2*x_offset*2*y_offset+y_offset)) # 刨除(0,0)点
    return dir

def meshgrid(offset=[10,10], pop=1):
    return _meshgrid(offset,pop)