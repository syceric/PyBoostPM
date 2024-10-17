from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import os

cython_directives = {"language_level": 3, "boundscheck": False, "wraparound": False, "initializedcheck": False, \
                       "profile": True}

extra_compile_args = ['-O3', '-march=native', '-ffast-math', '-fopenmp']
extra_link_args = ['-fopenmp']

extensions = [
    Extension("boostPM.model", ["boostPM/model.pyx"], include_dirs=[np.get_include()], language="c++", \
              extra_compile_args=extra_compile_args, extra_link_args=extra_link_args),
    Extension("boostPM.helpers", ["boostPM/helpers.pyx", "boostPM/rand.cpp"], include_dirs=[np.get_include(), "./boostPM"], \
              language="c++", extra_compile_args=extra_compile_args, extra_link_args=extra_link_args)
]

setup(name="boostPM",
      packages=find_packages(), 
      ext_modules=cythonize(extensions, compiler_directives=cython_directives))