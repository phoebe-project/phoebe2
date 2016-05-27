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
    # Extension('phoebe2.algorithms.burlishstoer',
    #     sources = ['./phoebe2/algorithms/burlishstoer/phoebe_BS_nbody.cpp', 
    #                 './phoebe2/algorithms/burlishstoer/n_body.cpp', 
    #                 './phoebe2/algorithms/burlishstoer/n_body_state.cpp', 
    #                 './phoebe2/algorithms/burlishstoer/kepcart.c'
    #                 ]
    #           ),
    Extension('phoebe_burlishstoer',
        sources = ['./phoebe2/algorithms/burlishstoer/phoebe_BS_nbody.cpp', 
                    './phoebe2/algorithms/burlishstoer/n_body.cpp', 
                    './phoebe2/algorithms/burlishstoer/n_body_state.cpp', 
                    './phoebe2/algorithms/burlishstoer/kepcart.c'
                    ]
              ),
    Extension('phoebe_roche',
      sources = ['./phoebe2/algorithms/roche/critical_potential.cpp',
                  ], 
              ),

    Extension('phoebe2.algorithms.interp',
             sources = ['phoebe2/algorithms/interp.c']),
    Extension('phoebe2.algorithms.cmarching',
              sources=['./phoebe2/algorithms/mrch.c'],
              libraries = libraries + ['m']
              ),
    Extension('phoebe2.utils.cgeometry',
              sources = ['phoebe2/utils/cgeometry.c']),
    Extension('phoebe2.algorithms.fraytracing',
              sources = ['phoebe2/algorithms/fraytracing_double.f']),
    Extension('phoebe2.algorithms.fsubdivision',
              sources = ['phoebe2/algorithms/fsubdivision.f']),
    Extension('phoebe2.algorithms.ceclipse',
              sources = ['phoebe2/algorithms/ceclipse.cpp']),
    Extension('phoebe2.dynamics.ctrans',
              sources = ['./phoebe2/dynamics/ctrans.cpp']),

    Extension('phoebe2.utils.fgeometry',
              sources = ['./phoebe2/utils/fgeometry.f']),

    Extension('phoebe2.atmospheres.atmcof',
              sources = ['./phoebe2/atmospheres/atmcof.f']),
]

setup (name = 'phoebe2',
       version = '2.0b',
       description = 'PHOEBE 2.0 beta',
       packages = ['phoebe2', 'phoebe2.constants', 'phoebe2.parameters', 'phoebe2.frontend', 'phoebe2.constraints', 'phoebe2.dynamics', 'phoebe2.distortions', 'phoebe2.algorithms', 'phoebe2.atmospheres', 'phoebe2.backend', 'phoebe2.utils'],
       install_requires=['numpy','scipy','astropy'],
       package_data={'phoebe2.atmospheres':['tables/wd/*', 'tables/ptf/*.*','redlaws/*.*','tables/ld_coeffs/README',
                                          'tables/ld_coeffs/blackbody_uniform_none_teff.fits',
                                          'tables/spectra/README','tables/spec_intens/README',
                                          'tables/gravb/claret.dat', 'tables/gravb/espinosa.dat',
                                          'tables/passbands/*'],
                    },
       ext_modules = ext_modules)