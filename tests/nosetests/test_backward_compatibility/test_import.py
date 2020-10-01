"""
"""

import phoebe
import numpy as np

from distutils.version import LooseVersion, StrictVersion

import os
dir = os.path.dirname(os.path.realpath(__file__))


def _export_21(filename, plot=False):
    """
    this isn't run during testing, but should be edited to run on certain versions
    of phoebe to then store the .phoebe file in this directory and ensure it can
    be imported later
    """


    if LooseVersion(phoebe.__version__) >= LooseVersion("2.2"):
       raise ImportError("script runs on PHOEBE 2.1.x")
       exit()

    b = phoebe.default_binary()
    # TESS:default was renamed to TESS:T in 2.2
    b.add_dataset('lc', times=np.linspace(0,1,11), passband='TESS:default')
    b.add_dataset('rv', times=phoebe.linspace(0,1,4))
    b.add_dataset('lp', times=phoebe.linspace(0,1,4), wavelengths=np.linspace(500,510,51))
    b.add_dataset('mesh', times=[0])

    b.run_compute()

    if plot:
        b.plot(show=True, time=0)

    b.save(os.path.join(dir, filename))

def _export_22(filename, plot=False):
    """
    this isn't run during testing, but should be edited to run on certain versions
    of phoebe to then store the .phoebe file in this directory and ensure it can
    be imported later
    """


    if LooseVersion(phoebe.__version__) >= LooseVersion("2.3"):
       raise ImportError("script runs on PHOEBE 2.2.x")
       exit()

    b = phoebe.default_binary()
    # TESS:default was renamed to TESS:T in 2.2
    b.add_dataset('lc', times=np.linspace(0,1,11), passband='Johnson:V', Av=0.1)
    b.add_dataset('lc', times=np.linspace(0,1,11), passband='Johnson:R', Av=0.2)
    b.add_dataset('rv', times=phoebe.linspace(0,1,4))
    b.add_dataset('lp', times=phoebe.linspace(0,1,4), wavelengths=np.linspace(500,510,51))
    b.add_dataset('mesh', times=[0])

    b.run_compute()

    if plot:
        b.plot(show=True, time=0)

    b.save(os.path.join(dir, filename))

def _export_23(filename, plot=False):
    """
    this isn't run during testing, but should be edited to run on certain versions
    of phoebe to then store the .phoebe file in this directory and ensure it can
    be imported later
    """


    if LooseVersion(phoebe.__version__) >= LooseVersion("2.4"):
       raise ImportError("script runs on PHOEBE 2.3.x")
       exit()

    b = phoebe.default_binary()
    # TESS:default was renamed to TESS:T in 2.2
    b.add_dataset('lc', times=np.linspace(0,1,11), passband='Johnson:V', Av=0.1)
    b.add_dataset('lc', times=np.linspace(0,1,11), passband='Johnson:R', Av=0.2)
    b.add_dataset('rv', times=phoebe.linspace(0,1,4))
    b.add_dataset('lp', times=phoebe.linspace(0,1,4), wavelengths=np.linspace(500,510,51))
    b.add_dataset('mesh', times=[0])

    b.run_compute()

    if plot:
        b.plot(show=True, time=0)

    b.save(os.path.join(dir, filename))


def test_21(verbose=False, plot=False):
    b = phoebe.load(os.path.join(dir, '21_export.phoebe'))
    b.run_compute()

    if plot:
        b.plot(show=True, time=0)

    return b

def test_22(verbose=False, plot=False):
    b = phoebe.load(os.path.join(dir, '22_export.phoebe'))
    b.run_compute()

    if plot:
        b.plot(show=True, time=0)

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='WARNING')

    # if LooseVersion(phoebe.__version__) >= LooseVersion("2.1.0") and LooseVersion(phoebe.__version__) < LooseVersion("2.2.0"):
        # _export_21('21_export.phoebe')
        # exit()
    # if LooseVersion(phoebe.__version__) >= LooseVersion("2.2.0") and LooseVersion(phoebe.__version__) < LooseVersion("2.3.0"):
    #     _export_22('22_export.phoebe')
    #     exit()
    # if LooseVersion(phoebe.__version__) >= LooseVersion("2.3.0") and LooseVersion(phoebe.__version__) < LooseVersion("2.4.0"):
    #     _export_23('23_export.phoebe')
    #     exit()

    b = test_21(verbose=True, plot=True)
    b = test_22(verbose=True, plot=True)
