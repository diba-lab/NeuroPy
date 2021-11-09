from setuptools import setup
from Cython.Build import cythonize
import numpy

## Build using:  python setup.py build_ext --inplace

# examples/Cython_testing/convolve_py.py
setup(ext_modules = cythonize(["../ccg_py.py", "convolve_py.py", "primes_python.py"]), 
      include_dirs=[numpy.get_include()])
# setup(ext_modules = cythonize(["ccg_py.py"]), 
#       include_dirs=[numpy.get_include()])
