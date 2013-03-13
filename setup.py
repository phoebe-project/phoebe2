"""
Install Phoebe.

Example usage: without pip:

$:> python setup.py build
$:> sudo python setup.py install

With pip:

$:> python setup.py sdist
$:> sudo pip install dist/phoebe-2.0.tar.gz

And with pip you can uninstall:

$:> sudo pip uninstall phoebe

On *buntu systems, the installation directory is

/usr/local/lib/python2.7/dist-packages/phoebe

"""
from numpy.distutils.core import setup, Extension
import glob
import sys
from numpy.distutils.command.build import build as _build

class Build(_build):
    user_options = _build.user_options
    
    user_options.append(('wd', None,'build fpywd'))
    for o in user_options:
        print o
    
    def initialize_options(self): 
        self.wd = False
        _build.initialize_options(self)
        
    def finalize_options(self):
        _build.finalize_options(self)    
        
ext_modules = [
        Extension('phoebe.utils.cgeometry',
                  sources = ['phoebe/utils/cgeometry.c']),
        Extension('phoebe.atmospheres.cspots',
                  sources = ['phoebe/atmospheres/cspots.c']),
        #Extension('phoebe.algorithms.marching2FLib',
        #          sources = ['phoebe/algorithms/marching2FLib.c'],
        #          libraries = ['meschach'],
        #          extra_compile_args=['-fPIC']),
        Extension('phoebe.algorithms.fsubdivision',
                  sources = ['phoebe/algorithms/fsubdivision.f']),
        Extension('phoebe.algorithms.fraytracing',
                  sources = ['phoebe/algorithms/fraytracing_double.f']),
        Extension('phoebe.wd.fwd',
                  include_dirs=['phoebe/wd'],
                  sources = glob.glob('phoebe/wd/*.f')),
        ]
setup(
    
    cmdclass = {'build' : Build},    
    
    name="phoebe",
    version="2.0",
    description="It does everything(TM).",
    long_description="reStructuredText format",
    author="Some dudes, a girl",
    author_email="dude@girl.com",
    url="http://www.phoebe-project.org",
    classifiers=["Programming Language :: Python :: 2.7",
                 "Programming Language :: C",
                 "Programming Language :: Fortran",
                 "Development Status :: Pre-Alpha",
                 "Intended Audience :: Science/Research",
                 "Topic :: Scientific/Engineering :: Astronomy",
                 "License :: OSI Approved :: GNU General Public License (GPL)",
                 "Operating System :: OS Independent"],
    
    packages=['phoebe','phoebe.wd','phoebe.units','phoebe.backend',
              'phoebe.units.uncertainties','phoebe.units.uncertainties.unumpy',
              'phoebe.utils','phoebe.atmospheres','phoebe.algorithms',
              'phoebe.parameters','phoebe.io','phoebe.dynamics'],
    
    package_data={'phoebe.atmospheres':['ptf/*.*','redlaws/*.*','tables/ld_coeffs/*.fits'],
                  'phoebe.parameters':['catalogs/*.dat','library/*.par'],},

    ext_modules = ext_modules,
)


