"""
"""

import phoebe
import numpy as np

from packaging.version import parse

import os
dir = os.path.dirname(os.path.realpath(__file__))


def _export_21(filename, plot=False):
    """
    this isn't run during testing, but should be edited to run on certain versions
    of phoebe to then store the .phoebe file in this directory and ensure it can
    be imported later
    """

    if parse(phoebe.__version__) >= parse("2.2"):
        raise ImportError("script runs on PHOEBE 2.1.x")

    b = phoebe.default_binary()
    # TESS:default was renamed to TESS:T in 2.2
    b.add_dataset('lc', times=np.linspace(0, 1, 11), passband='TESS:default')
    b.add_dataset('rv', times=phoebe.linspace(0, 1, 4))
    b.add_dataset('lp', times=phoebe.linspace(0, 1, 4), wavelengths=np.linspace(500, 510, 51))
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

    if parse(phoebe.__version__) >= parse("2.3"):
        raise ImportError("script runs on PHOEBE 2.2.x")

    b = phoebe.default_binary()
    b.add_dataset('lc', times=np.linspace(0, 1, 11), passband='Johnson:V', Av=0.1)
    b.add_dataset('lc', times=np.linspace(0, 1, 11), passband='Johnson:R', Av=0.2)
    b.add_dataset('rv', times=phoebe.linspace(0, 1, 4))
    b.add_dataset('lp', times=phoebe.linspace(0, 1, 4), wavelengths=np.linspace(500, 510, 51))
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

    if parse(phoebe.__version__) >= parse("2.4"):
        raise ImportError("script runs on PHOEBE 2.3.x")

    b = phoebe.default_binary()
    b.add_dataset('lc', times=np.linspace(0, 1, 11), passband='Johnson:V', Av=0.1)
    b.add_dataset('lc', times=np.linspace(0, 1, 11), passband='Johnson:R', Av=0.2)
    b.add_dataset('rv', times=phoebe.linspace(0, 1, 4))
    b.add_dataset('lp', times=phoebe.linspace(0, 1, 4), wavelengths=np.linspace(500, 510, 51))
    b.add_dataset('mesh', times=[0])

    b.run_compute()

    # migrations needed for GPs and ebai solver
    b.add_feature('gaussian_process', dataset='lc01', kernel='sho')
    b.add_feature('gaussian_process', dataset='lc01', kernel='matern32')
    b.add_solver('estimator.ebai')

    if plot:
        b.plot(show=True, time=0)

    b.save(os.path.join(dir, filename))


def _export_24(filename, plot=False):
    """
    this isn't run during testing, but should be edited to run on certain versions
    of phoebe to then store the .phoebe file in this directory and ensure it can
    be imported later
    """

    if parse(phoebe.__version__) >= parse("2.5"):
        raise ImportError("script runs on PHOEBE 2.4.x")

    b = phoebe.default_binary()
    b.add_dataset('lc', times=np.linspace(0, 1, 11), passband='Johnson:V', Av=0.1)
    b.add_dataset('lc', times=np.linspace(0, 1, 11), passband='Johnson:R', Av=0.2)
    b.add_dataset('rv', times=phoebe.linspace(0, 1, 4))
    b.add_dataset('lp', times=phoebe.linspace(0, 1, 4), wavelengths=np.linspace(500, 510, 51))
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


def test_22(verbose=False, plot=False):
    b = phoebe.load(os.path.join(dir, '22_export.phoebe'))
    b.run_compute()

    if plot:
        b.plot(show=True, time=0)


def test_23(verbose=False, plot=False):
    b = phoebe.load(os.path.join(dir, '23_export.phoebe'))
    # can't run forward model with GPs without data (and will trip error on CI
    # if dependency isn't installed)
    for gp in b.filter(context='feature', kind='gp*').features:
        b.remove_feature(feature=gp)
    b.run_compute()

    if plot:
        b.plot(show=True, time=0)


def test_24(verbose=False, plot=False):
    b = phoebe.load(os.path.join(dir, '24_export.phoebe'))
    b.run_compute()

    if plot:
        b.plot(show=True, time=0)


if __name__ == '__main__':
    logger = phoebe.logger(clevel='WARNING')

    if False:
        if parse(phoebe.__version__) >= parse("2.1.0") and parse(phoebe.__version__) < parse("2.2.0"):
            _export_21('21_export.phoebe')
            exit()
        if parse(phoebe.__version__) >= parse("2.2.0") and parse(phoebe.__version__) < parse("2.3.0"):
            _export_22('22_export.phoebe')
            exit()
        if parse(phoebe.__version__) >= parse("2.3.0") and parse(phoebe.__version__) < parse("2.4.0"):
            _export_23('23_export.phoebe')
            exit()
        if parse(phoebe.__version__) >= parse("2.4.0") and parse(phoebe.__version__) < parse("2.5.0"):
            _export_24('24_export.phoebe')
            exit()

    b = test_21(verbose=True, plot=True)
    b = test_22(verbose=True, plot=True)
    b = test_23(verbose=True, plot=True)
    b = test_24(verbose=True, plot=True)
