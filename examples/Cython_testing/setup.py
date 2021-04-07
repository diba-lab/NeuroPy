from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize(["ccg_py.py"]), 
      include_dirs=[numpy.get_include()])
