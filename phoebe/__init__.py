"""import phoebe"""

__version__ = 'devel'

import os
import sys

# People shouldn't import Phoebe from the installation directory (inspired upon
# pymc warning message).
if os.getcwd().find(os.path.abspath(os.path.split(os.path.split(__file__)[0])[0]))>-1:
    # We have a clash of package name with the standard library: we implement an
    # "io" module and also they do. This means that you can import Phoebe from its
    # main source tree; then there is no difference between io from here and io
    # from the standard library. Thus, if the user loads the package from here
    # it will never work. Instead of letting Python raise the io clash (which
    # is uniformative to the unexperienced user), we raise the importError here
    # with a helpful error message
    raise ImportError('\n\tYou cannot import Phoebe from inside its main source tree.\n')

import logging
_logger = logging.getLogger("PHOEBE")
_logger.addHandler(logging.NullHandler())

class Settings(object):
    def __init__(self):
        # Check to see whether in interactive mode
        import __main__
        # hasattr(__main__, '__file__') will be True if running a python script, but
        # false if in a python or ipython interpreter.
        # sys.flags.interactive will be 1 if the -i flag is sent to python

        # For now we'll set interactive_constraints to True by default, requiring it to
        # explicitly be disabled.
        # See #154 (https://github.com/phoebe-project/phoebe2/issues/154)
        self._interactive_constraints = True

        # We'll set interactive system checks to be on if running within a Python
        # console, but False if running from within a script
        # See #255 (https://github.com/phoebe-project/phoebe2/issues/255)
        self._interactive_checks = not hasattr(__main__, '__file__') or bool(sys.flags.interactive)

        # And we'll require explicitly setting developer mode on
        self._devel = False

        def _to_bool(value):
            if isinstance(value, bool):
                return value
            elif value.upper()=='TRUE':
                return True
            else:
                return False

        self._do_mpirun = _to_bool(os.getenv('PHOEBE_ENABLE_MPI', False))
        self._mpi_np = int(os.getenv('PHOEBE_MPI_NP', 2))
        self._force_serial = False

    def reset(self):
        self.__init__()

    def interactive_on(self):
        self.interactive_checks_on()
        self.interactive_constraints_on()

    def interactive_off(self):
        self.interactive_checks_off()
        self.interactive_constraints_off()

    def interactive_checks_on(self):
        self._interactive_checks = True

    def interactive_checks_off(self):
        self._interactive_checks = False

    def interactive_constraints_on(self):
        self._interactive_constraints = True

    def interactive_constraints_off(self):
        _logger.warning("constraints will not be run until 'run_delayed_constraints' or 'run_compute' is called.  This may result in inconsistent parameters if printing values before calling either of these methods.")
        self._interactive_constraints = False

    @property
    def interactive_checks(self):
        return self._interactive_checks

    @property
    def interactive_constraints(self):
        return self._interactive_constraints

    def devel_on(self):
        self._devel = True

    def devel_off(self):
        self._devel = False

    @property
    def devel(self):
        return self._devel

    @property
    def force_serial(self):
        return self._force_serial

    @property
    def mpi(self):
        return self._do_mpirun

    @property
    def detach_cmd(self):
        if self._do_mpirun:
            return 'mpirun -np %d python {} &>/dev/null &' % self._mpi_np
        else:
            return 'python {} &>/dev/null &'


conf = Settings()





# make packages available at top-level
from .dependencies.unitsiau2015 import *
from .dependencies.nparray import array, linspace, arange, logspace, geomspace
from .atmospheres.passbands import install_passband, download_passband, list_online_passbands, list_installed_passbands, list_passbands, list_passband_directories, get_passband
from .parameters import *
from .parameters import hierarchy, component, compute, constraint, dataset
from .frontend.bundle import Bundle
from .backend import *
import utils as utils

import dynamics as dynamics
import distortions as distortions
import algorithms as algorithms
import libphoebe

# Shortcut to building logger
def logger(*args, **kwargs):
    """
    shortcut to :func:`utils.get_basic_logger`
    """
    return utils.get_basic_logger(*args, **kwargs)

# Shortcuts to bundle classmethods
def open(*args, **kwargs):
    return Bundle.open(*args, **kwargs)

def load(*args, **kwargs):
    return Bundle.open(*args, **kwargs)

def from_legacy(*args, **kwargs):
    return Bundle.from_legacy(*args, **kwargs)

def default_star(*args, **kwargs):
    return Bundle.default_star(*args, **kwargs)

def default_binary(*args, **kwargs):
    return Bundle.default_binary(*args, **kwargs)

def default_triple(*args, **kwargs):
    return Bundle.default_triple(*args, **kwargs)


def reset_settings():
    conf.reset()

def is_interactive():
    return conf.interactive

def interactive_on():
    conf.interactive_on()

def interactive_off():
    conf.interactive_off()

def interactive_constraints_on():
    conf.interactive_constraints_on()

def interactive_constraints_off():
    conf.interactive_constraints_off()

def interactive_checks_on():
    conf.interactive_checks_on()

def interactive_checks_off():
    conf.interactive_checks_off()

def is_devel():
    return conf.devel

def devel_on():
    conf.devel_on()

def devel_off():
    conf.devel_off()

def force_serial():
    """
    force serial mode when called from within mpirun
    """
    conf._force_serial = True

def mpi_on(np=None):
    conf._do_mpirun = True
    if np is not None:
        conf._mpi_np = np

def mpi_off():
    conf._do_mpirun = False

def set_np(np):
    conf._mpi_np = np
