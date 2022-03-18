import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy

ext = Extension('fillnamedian_cpu',
        sources=['dev_bucket_sort.cpp', 'fillnamedian_cpu.pyx'],
        library_dirs = [],
        libraries = [],
        include_dirs = [numpy.get_include()],
        language = 'c++',
        runtime_library_dirs = [],
        extra_compile_args=['-fopenmp']
    )

setup(name = 'fillnamedian_cpu',
      ext_modules = [ext],
      cmdclass={'build_ext': build_ext},
      zip_safe = False)
      

