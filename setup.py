from numpy.distutils.core import setup, Extension
import platform

platform = platform.system()

if platform == 'Windows':
  import os
  os.environ['VS90COMNTOOLS'] = os.environ['VS140COMNTOOLS']

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
       description = 'PHOEBE devel',
       author = 'PHOEBE development team',
       author_email = 'phoebe-devel@lists.sourceforge.net',
       url = 'http://github.com/phoebe-project/phoebe2',
       download_url = 'https://github.com/phoebe-project/phoebe2/tarball/2.0.3',
       packages = ['phoebe', 'phoebe.constants', 'phoebe.parameters', 'phoebe.frontend', 'phoebe.constraints', 'phoebe.dynamics', 'phoebe.distortions', 'phoebe.algorithms', 'phoebe.atmospheres', 'phoebe.backend', 'phoebe.utils', 'phoebe.dependencies'],
       install_requires=['numpy>=1.10','scipy>=0.18','astropy>=1.0'],
       package_data={'phoebe.atmospheres':['tables/wd/*', 'tables/passbands/*'],
                    },
       ext_modules = ext_modules)
