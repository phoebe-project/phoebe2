from setuptools import Extension, setup
import numpy

ext_modules = [
    Extension(
        'libphoebe',
        sources=['phoebe/lib/libphoebe.cpp'],
        language='c++',
        extra_compile_args=["-std=c++14"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        'phoebe.algorithms.ceclipse',
        sources=['phoebe/algorithms/ceclipse.cpp'],
        language='c++',
        include_dirs=[numpy.get_include()]
    ),
]

setup(
    ext_modules=ext_modules,
)
