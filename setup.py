from numpy.distutils.core import setup, Extension

import os
#os.environ["CC"] = "gcc" 
os.environ["CXX"] = "g++"
os.environ["CFLAGS"] = "-std=c++11"  # numpy mixes CXXFLAGS and CFLAGS

# Set to true if you want to link against electric fence:
CDEBUG = False

libraries = []
if CDEBUG:
    libraries += ['efence']

ext_modules = [
    # Extension('phoebe.algorithms.burlishstoer',
    #     sources = ['./phoebe/algorithms/burlishstoer/phoebe_BS_nbody.cpp', 
    #                 './phoebe/algorithms/burlishstoer/n_body.cpp', 
    #                 './phoebe/algorithms/burlishstoer/n_body_state.cpp', 
    #                 './phoebe/algorithms/burlishstoer/kepcart.c'
    #                 ]
    #           ),
    Extension('phoebe_burlishstoer',
        sources = ['./phoebe/algorithms/burlishstoer/phoebe_BS_nbody.cpp', 
                    './phoebe/algorithms/burlishstoer/n_body.cpp', 
                    './phoebe/algorithms/burlishstoer/n_body_state.cpp', 
                    './phoebe/algorithms/burlishstoer/kepcart.c'
                    ]
              ),
    Extension('phoebe_roche',
      sources = ['./phoebe/algorithms/roche/critical_potential.cpp',
                  ], 
              ),

    Extension('phoebe.algorithms.interp',
             sources = ['phoebe/algorithms/interp.c']),
    Extension('phoebe.algorithms.cmarching',
              sources=['./phoebe/algorithms/mrch.c'],
              libraries = libraries + ['m']
              ),
    Extension('phoebe.utils.cgeometry',
              sources = ['phoebe/utils/cgeometry.c']),
    Extension('phoebe.algorithms.fraytracing',
              sources = ['phoebe/algorithms/fraytracing_double.f']),
    Extension('phoebe.algorithms.fsubdivision',
              sources = ['phoebe/algorithms/fsubdivision.f']),
    Extension('phoebe.algorithms.ceclipse',
              sources = ['phoebe/algorithms/ceclipse.cpp']),
    Extension('phoebe.dynamics.ctrans',
              sources = ['./phoebe/dynamics/ctrans.cpp']),

    Extension('phoebe.utils.fgeometry',
              sources = ['./phoebe/utils/fgeometry.f']),

    Extension('phoebe.atmospheres.atmcof',
              sources = ['./phoebe/atmospheres/atmcof.f']),
]

setup (name = 'phoebe',
       version = '2.0b',
       description = 'PHOEBE 2.0 beta',
       packages = ['phoebe', 'phoebe.constants', 'phoebe.parameters', 'phoebe.frontend', 'phoebe.constraints', 'phoebe.dynamics', 'phoebe.distortions', 'phoebe.algorithms', 'phoebe.atmospheres', 'phoebe.backend', 'phoebe.utils'],
       install_requires=['numpy','scipy','astropy'],
       package_data={'phoebe.atmospheres':['tables/wd/*', 'tables/ptf/*.*','redlaws/*.*','tables/ld_coeffs/README',
                                          'tables/ld_coeffs/blackbody_uniform_none_teff.fits',
                                          'tables/spectra/README','tables/spec_intens/README',
                                          'tables/gravb/claret.dat', 'tables/gravb/espinosa.dat',
                                          'tables/passbands/*'],
                    },
       ext_modules = ext_modules)
