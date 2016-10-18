from numpy.distutils.core import setup, Extension

# Set to true if you want to link against electric fence:
CDEBUG = False

libraries = []
if CDEBUG:
    libraries += ['efence']

ext_modules = [
    Extension('libphoebe',
      sources = ['./phoebe/lib/libphoebe.cpp'],
      extra_compile_args = ["-std=c++11"]),

    Extension('phoebe.algorithms.ceclipse',
              sources = ['phoebe/algorithms/ceclipse.cpp']),
]

setup (name = 'phoebe',
       version = 'devel',
       description = 'PHOEBE 2.0 devel',
       packages = ['phoebe', 'phoebe.constants', 'phoebe.parameters', 'phoebe.frontend', 'phoebe.constraints', 'phoebe.dynamics', 'phoebe.distortions', 'phoebe.algorithms', 'phoebe.atmospheres', 'phoebe.backend', 'phoebe.utils'],
       install_requires=['numpy','scipy','astropy'],
       package_data={'phoebe.atmospheres':['tables/wd/*', 'tables/passbands/*'],
                    },
       ext_modules = ext_modules)
