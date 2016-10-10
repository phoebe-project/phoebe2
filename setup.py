from numpy.distutils.core import setup, Extension

import os
#os.environ["CXX"] = "g++"

# Set to true if you want to link against electric fence:
CDEBUG = False

libraries = []
if CDEBUG:
    libraries += ['efence']

ext_modules = [

    Extension('phoebe_burlishstoer',
        sources = ['./phoebe/algorithms/burlishstoer/phoebe_BS_nbody.cpp',
                    './phoebe/algorithms/burlishstoer/n_body.cpp',
                    './phoebe/algorithms/burlishstoer/n_body_state.cpp',
                    './phoebe/algorithms/burlishstoer/kepcart.h'
                    ]
              ),

    Extension('libphoebe',
      sources = ['./phoebe/lib/libphoebe.cpp'],
      extra_compile_args = ["-std=c++11"]),

    Extension('phoebe.algorithms.ceclipse',
              sources = ['phoebe/algorithms/ceclipse.cpp']),

    Extension('phoebe.algorithms.interp',
             sources = ['phoebe/algorithms/interp.cpp']),

    Extension('phoebe.atmospheres.atmcof',
              sources = ['./phoebe/atmospheres/atmcof.f']),
]

setup (name = 'phoebe',
       version = 'devel',
       description = 'PHOEBE 2.0 devel',
       packages = ['phoebe', 'phoebe.constants', 'phoebe.parameters', 'phoebe.frontend', 'phoebe.constraints', 'phoebe.dynamics', 'phoebe.distortions', 'phoebe.algorithms', 'phoebe.atmospheres', 'phoebe.backend', 'phoebe.utils'],
       install_requires=['numpy','scipy','astropy'],
       package_data={'phoebe.atmospheres':['tables/wd/*', 'tables/ptf/*.*','redlaws/*.*','tables/ld_coeffs/README',
                                          'tables/ld_coeffs/blackbody_uniform_none_teff.fits',
                                          'tables/spectra/README','tables/spec_intens/README',
                                          'tables/gravb/claret.dat', 'tables/gravb/espinosa.dat',
                                          'tables/passbands/*'],
                    },
       ext_modules = ext_modules)
