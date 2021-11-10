# setup.py
from distutils.core import setup, Extension
from os import name
from Cython.Build import cythonize
# import numpy

setup(
    name='neighbor_value',
    ext_modules = cythonize(Extension(
    name='neighbor_value',
    sources=['neighbor_value.pyx'],
    language='c++',
    include_dirs=[],# numpy.get_include()
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))

'''
setup(
    name='is_agent',
    ext_modules = cythonize("is_agent.pyx"))

'''
