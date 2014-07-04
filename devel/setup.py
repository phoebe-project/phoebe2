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
import re
from numpy.distutils.command.build import build as _build

# Get version number
import re

VERSIONFILE = "phoebe/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


class Build(_build):
    user_options = _build.user_options
    
    user_options.append(('wd', None,'build fpywd'))
    for o in user_options:
        print(o)
    
    def initialize_options(self): 
        self.wd = False
        _build.initialize_options(self)
        
    def finalize_options(self):
        _build.finalize_options(self)    
        
ext_modules = [
        Extension('phoebe.utils.cgeometry',
                  sources = ['phoebe/utils/cgeometry.c']),
        Extension('phoebe.utils.fgeometry',
                  sources = ['phoebe/utils/fgeometry.f']),
        Extension('phoebe.atmospheres.cspots',
                  sources = ['phoebe/atmospheres/cspots.c']),
        Extension('phoebe.algorithms.fsubdivision',
                  sources = ['phoebe/algorithms/fsubdivision.f']),
        Extension('phoebe.algorithms.freflection',
                  sources = ['phoebe/algorithms/freflection.f']),
        Extension('phoebe.algorithms.fraytracing',
                  sources = ['phoebe/algorithms/fraytracing_double.f']),
        Extension('phoebe.algorithms.ceclipse',
                  sources = ['phoebe/algorithms/ceclipse.cpp']),
        #Extension('phoebe.algorithms.interp',
        #          sources = ['phoebe/algorithms/interp.c']),
        Extension('phoebe.dynamics.ftrans',
                  sources = ['phoebe/dynamics/ftrans.f']),
        Extension('phoebe.dynamics.ctrans',
                  sources = ['phoebe/dynamics/ctrans.cpp']),
        Extension('phoebe.wd.fwd',
                  include_dirs=['phoebe/wd'],
                  sources = glob.glob('phoebe/wd/*.f')),
        Extension('phoebe.algorithms.cmarching',
                  sources=['phoebe/algorithms/mrch.c'],
                  libraries = ['m']),
        Extension('phoebe.atmospheres.froche',
                  sources=['phoebe/atmospheres/froche.f']),
        Extension('phoebe.atmospheres.fpulsations',
                  sources=['phoebe/atmospheres/pulsations-devel-optim.f90']),
                  #sources=['phoebe/atmospheres/pulsations-devel.f']),
        ]
setup(
    
    cmdclass = {'build' : Build},    
    
    name="phoebe",
    version=verstr,
    description="Physics of stars and stellar and planetary systems",
    long_description="Physics of stars and stellar and planetary systems",
    author="Pieter Degroote, Kyle Conroy, Steven Bloemen, Bert Pablo, Kelly Hambleton, Joe Giammarco, Andrej Prsa",
    author_email="phoebe-devel@lists.sourceforge.net",
    url="http://www.phoebe-project.org",
    classifiers=["Programming Language :: Python :: 2.7",
                 "Programming Language :: C",
                 "Programming Language :: C++",
                 "Programming Language :: Fortran",
                 "Development Status :: 3 - Alpha",
                 "Intended Audience :: Science/Research",
                 "Topic :: Scientific/Engineering :: Astronomy",
                 "License :: OSI Approved :: GNU General Public License (GPL)",
                 "Operating System :: Unix"],
    
    packages=['phoebe','phoebe.wd','phoebe.units','phoebe.backend',
              'phoebe.utils','phoebe.atmospheres','phoebe.algorithms',
              'phoebe.parameters','phoebe.io','phoebe.dynamics',
              'phoebe.frontend','phoebe.frontend.gui'],
    
    package_data={'phoebe.atmospheres':['ptf/*.*','redlaws/*.*','tables/ld_coeffs/README',
                                        'ptf/phoebe1set/*', 'ptf/phoebe2set/*',
                                        'tables/ld_coeffs/blackbody_uniform_none_teff.fits',
                                        'tables/spectra/README','tables/spec_intens/README',],
                  'phoebe.parameters':['catalogs/*.dat','library/*.par','library/*.phoebe'],
                  'phoebe.wd':['*.dat'],
                  'phoebe.frontend.gui':['icons/*','html/*']},
    scripts=['phoebe/frontend/gui/phoebe_gui.py'],
    install_requires=['numpy','scipy','matplotlib','pyfits','uncertainties'],
    #~ entry_points = {
        #~ "gui_scripts": ["phoebe_gui = phoebe.frontend.phoebe_gui"]},

    ext_modules = ext_modules,
)
