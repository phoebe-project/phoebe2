import sys
import os

#-- these are the packages that need to be tested
version_IPython = '0.11.0'
version_numpy = '1.7.1'
version_matplotlib = '1.2.0'
version_scipy = '0.9.0'
version_pyfits = '3.1.0'
version_emcee = '1.2.0'
version_pymc = '2.2.0'
version_lmfit = '0.7.0'
version_mayavi = '4.1.0'
version_iminuit = '1.0.6'

#-- it's easier to test numbers than strings, so these small functions
#   convert a string to a list of floats, and test whether or not the version
#   is at least the required one.
def string_to_triple(current):
    # remember major, minor and micro if available
    version = [0,0,0]
    current = current.split('.')
    for i,value in enumerate(current):
        try:        
            version[i] = float(value)
        except:
            version[i] = 0
    return version

def check_version_info(current,required):
    # check if all of major, minor and micro are greater or equal to the required
    # version
    current = string_to_triple(current)
    required = string_to_triple(required)
    if all([(i>=j) for i,j in zip(current,required)]):
        return True

#-- check Python version
if sys.version_info[0]<2 or sys.version_info[1]<7:
    print("You need to have Python 2.7 at least, it makes the transition to Python 3 easier")
else:
    print("Your Python version is up-to-date")

#-- check all other packages: I was lazy in naming variables, so I can just
#   cycle over all the variables that are defined here. Whenever it says "version_*"
#   the variable hold the name of the package and the value is the required version
for version_package in locals().keys():
    # of course we have other global variables as well, we want to skip those
    if not 'version'==version_package.split('_')[0]: continue
    # derive the package name and required version
    package = "_".join(version_package.split('_')[1:])
    required_version = locals()[version_package]
    # try to import, and check the version
    try:
        mypackage = __import__(package)
        if not check_version_info(mypackage.__version__,required_version):
            print("Package {:s} is old (version {:s} instead of {:s}). Please upgrade.".format(package,mypackage.__version__,required_version))
        else:
            print("Package {:s} is installed and up-to-date ({:s})".format(package,mypackage.__version__))
    except ImportError:
        print("Package {:s} is not installed".format(package))

try:
    from phoebe.algorithms import cmarching
    print("C-marching installed")
except ImportError:
    print("Non-fatal warning: C-marching not available")

try:
    from phoebe.algorithms import dc
    print("DC installed")
except ImportError:
    print("Non-fatal warning: DC not available")

try:
    from phoebe.utils import transit
    print("Analytical transit computations installed")
except ImportError:
    print("Non-fatal warning: Analytical transit computations from Mandel & Agol 2002 are not available")

try:
    from phoebe.wd import fwd
    print("Wilson-Devinney interface available")
except ImportError:
    print("Non-fatal warning: Wilson-Devinney code not available")

        
