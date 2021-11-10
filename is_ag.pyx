import numpy as np
# cimport numpy as np
# cimport cython

# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef float _is_a(list xy, list use_map):
    '''
    判断当前地块是空地还是被智能体占据
    ID > 999是ID号
    ID = 0是空地
    '''
    cdef int x, y
    cdef float ID
    x,y = xy
    ID = use_map[x][y]
    if ID > 999:   return ID
    elif ID < 999: return 0.

def is_a(xy,use_map):
    return _is_a(xy,use_map)
