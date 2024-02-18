from setuptools import Extension, setup
import numpy

ext_modules = [
    Extension('libphoebe',
      sources = ['phoebe/lib/libphoebe.cpp'],
      language='c++',
      extra_compile_args = ["-std=c++11"],
      include_dirs=[numpy.get_include()]
      ),

    Extension(
      'cndpolator',
      sources=[
          'phoebe/dependencies/ndpolator/ndpolator.c',
          'phoebe/dependencies/ndpolator/ndp_types.c',
      ],
      language='c',
      extra_compile_args=["-Werror", "-O0", "-g"],
      include_dirs=[numpy.get_include()],
    ),

    Extension('phoebe.algorithms.ceclipse',
      language='c++',
      sources = ['phoebe/algorithms/ceclipse.cpp'],
      include_dirs=[numpy.get_include()]
      ),
]

setup(
    ext_modules=ext_modules,
)
