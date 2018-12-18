"""
>>> import phoebe

Available environment variables:
* PHOEBE_ENABLE_PLOTTING=TRUE/FALSE (whether to import plotting libraries with phoebe: defaults to True)
* PHOEBE_ENABLE_SYMPY=TRUE/FALSE (whether to attempt to import sympy for constraint algebra: defaults to True if sympy installed, otherwise False)
* PHOEBE_ENABLE_ONLINE_PASSBANDS=TRUE/FALSE (whether to query for online passbands and download on-the-fly: defaults to True)
* PHOEBE_ENABLE_MPI=TRUE/FALSE (whether to use internal parallelization: defaults to True if within mpirun, otherwise False, can override in python with phoebe.mpi.on() and phoebe.mpi.off())
* PHOEBE_MPI_NPROCS=INT (number of procs to spawn in mpi is enabled but not running within mpirun: defaults to 4, only applicable if not within mpirun and PHOEBE_ENABLE_MPI=TRUE or phoebe.mpi.on() called, can override in python by passing nprocs to phoebe.mpi.on() or by setting phoebe.mpi.nprocs)

"""

__version__ = '2.1.1'

import os
import sys as _sys
import atexit

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

def _env_variable_int(key, default):
    value = os.getenv(key, default)
    return int(value)

def _env_variable_bool(key, default):
    value = os.getenv(key, default)
    if isinstance(value, bool):
        return value
    elif value.upper()=='TRUE':
        return True
    else:
        return False

# If we try to load matplotlib.pyplot on a non-X system, it will fail
# unless 'Agg' is used before the import. All X-systems define the
# 'DISPLAY' environment variable, and all non-X-systems do not. We do make a
# distinction between windows and unix based system. Hence:
if _env_variable_bool('PHOEBE_ENABLE_PLOTTING', True):
    try:
        import matplotlib
    except ImportError:
        pass
        # we'll catch this later in plotting and throw warnings as necessary
    else:
        if 'DISPLAY' not in os.environ.keys() and _sys.platform not in ['win32','cygwin']:
            matplotlib.use('Agg')
        elif hasattr(_sys, 'real_prefix'):
            # then we're likely in a virtualenv.  Our best bet is to use the 'TkAgg'
            # backend, but this will require python-tk to be installed on the system
            matplotlib.use('TkAgg')



import logging
_logger = logging.getLogger("PHOEBE")
_logger.addHandler(logging.NullHandler())

###############################################################################
#########################         BEGIN MPI          ##########################
###############################################################################

# detect if we're within mpirun and if so, place all non-zero-rank
# processors into a wait loop.  This must happen before we start importing from
# phoebe so that those can have access to the _mpi object.

class MPI(object):
    def __init__(self):
        # this is a bit of a hack and will only work with openmpi, but environment
        # variables seem to be the only way to detect whether the script was run
        # via mpirun or not
        evars = os.environ.keys()
        if 'OMPI_COMM_WORLD_SIZE' in evars or 'MV2_COMM_WORLD_SIZE' in evars or 'PMI_SIZE' in evars:
            from mpi4py import MPI as mpi4py
            self._within_mpirun = True
            self._internal_mpi = True

            self._comm   = mpi4py.COMM_WORLD
            self._myrank = self.comm.Get_rank()
            self._nprocs = self.comm.Get_size()

            if self._nprocs==1:
                raise ImportError("need more than 1 processor to run with mpi")

            self._enabled = _env_variable_bool("PHOEBE_ENABLE_MPI", True)

        else:
            self._within_mpirun = False
            self._internal_mpi = False
            self._comm = None
            self._myrank = 0
            self._nprocs = _env_variable_int("PHOEBE_MPI_NPROCS", 4)

            self._enabled = _env_variable_bool("PHOEBE_ENABLE_MPI", False)


    def __repr__(self):
        return "<MPI internal_mpi={} myrank={} nprocs={}>".format(self.internal_mpi, self.myrank, self.nprocs)

    @property
    def mode(self):
        if self.within_mpirun:
            if self.enabled:
                return "internal handling of mpi within mpirun"
            else:
                return "external handling of mpi by the user within mpirun"
        else:
            if self.enabled:
                return "internal handling of mpi in spawned separate threads during run_compute"
            else:
                return "serial mode"

    @property
    def enabled(self):
        return self._enabled

    def on(self, nprocs=None):
        if self.within_mpirun and not self.enabled:
            raise ValueError("cannot enable mpi after disabling within mpirun.")

        self._enabled = True

        if nprocs is not None:
            self.nprocs = nprocs

    def off(self):
        if self.within_mpirun and self.myrank == 0:
            self.comm.bcast({'worker_command': 'release'}, root=0)

        self._enabled = False

    @property
    def myrank(self):
        return self._myrank

    @property
    def nprocs(self):
        if not self.enabled and not self.within_mpirun:
            return 1
        else:
            return self._nprocs

    @nprocs.setter
    def nprocs(self, nprocs):
        if self.within_mpirun:
            _logger.warning("ignoring setting nprocs while within mpirun, nprocs={}".format(self.nprocs))
        else:
            self._nprocs = nprocs

    @property
    def comm(self):
        return self._comm

    @property
    def within_mpirun(self):
        return self._within_mpirun

    @property
    def detach_cmd(self):
        if self.within_mpirun:
            raise ValueError("detach not available within mpirun")

        if self.enabled:
            return 'mpirun -np %d python {} &>/dev/null &' % self.nprocs
        else:
            return 'python {} &>/dev/null &'

    def shutdown_workers(self):
        if self.within_mpirun and self.myrank == 0:
            self.comm.bcast({'worker_command': 'shutdown'}, root=0)
            self._enabled = False
            # even though technically not true, we're now strictly serial and have no way of regaining the workers
            self._within_mpirun = False


mpi = MPI()

# NOTE: logic for worker waiting for tasks below after phoebe imports

###############################################################################
##########################         END MPI          ###########################
###############################################################################


###############################################################################
#########################        BEGIN SETTINGS        ########################
###############################################################################

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
        self._interactive_checks = not hasattr(__main__, '__file__') or bool(_sys.flags.interactive)

        # And we'll require explicitly setting developer mode on
        self._devel = False

    def __repr__(self):
        return "<Settings interactive_checks={} interactive_constraints={}>".format(self.interactive_checks, self.interactive_constraints)

    def reset(self):
        self.__init__()

    def interactive_on(self):
        self.interactive_checks_on()
        self.interactive_constraints_on()

    def interactive_off(self, suppress_warning=False):
        self.interactive_checks_off(suppress_warning=suppress_warning)
        self.interactive_constraints_off(suppress_warning=suppress_warning)

    def interactive_checks_on(self):
        self._interactive_checks = True

    def interactive_checks_off(self, suppress_warning=False):
        if not suppress_warning:
            _logger.warning("checks will not be run until 'run_checks' or 'run_compute' is called.")
        self._interactive_checks = False

    def interactive_constraints_on(self):
        self._interactive_constraints = True

    def interactive_constraints_off(self, suppress_warning=False):
        if not suppress_warning:
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

conf = Settings()

###############################################################################
##########################        END SETTINGS        #########################
###############################################################################



# make packages available at top-level
from .dependencies.unitsiau2015 import u,c
from .dependencies.nparray import array, linspace, arange, logspace, geomspace
from .atmospheres.passbands import install_passband, download_passband, list_online_passbands, list_installed_passbands, list_passbands, list_passband_directories, get_passband
# from .parameters import *
from .parameters import hierarchy, component, compute, constraint, dataset
from .frontend.bundle import Bundle
# from .backend import *
from .backend import backends as _backends
import utils as _utils

import dynamics as dynamics
import distortions as distortions
import algorithms as algorithms
import libphoebe

# Shortcut to building logger
def logger(*args, **kwargs):
    """
    shortcut to :func:`utils.get_basic_logger`
    """
    if mpi.within_mpirun and mpi.myrank == 0:
        # tell the workers to invoke the same logger
        mpi.comm.bcast({'worker_command': 'logger', 'args': args, 'kwargs': kwargs}, root=0)

    return _utils.get_basic_logger(*args, **kwargs)


if mpi.within_mpirun and mpi.enabled and mpi.myrank != 0:
    while True:
        packet = mpi.comm.bcast(None, root=0)

        if packet.get('worker_command', False) == 'shutdown':
            _logger.debug("rank:{}/{} message to shutdown".format(mpi.myrank, mpi.nprocs))
            exit()

        if packet.get('worker_command', False) == 'release':
            _logger.debug("rank:{}/{} message to release".format(mpi.myrank, mpi.nprocs))
            break

        elif packet.get('worker_command', False) == 'logger':
            _logger.debug("rank:{}/{} message to invoke logger".format(mpi.myrank, mpi.nprocs))
            logger(*packet['args'], **packet['kwargs'])

        elif hasattr(_backends, packet.get('backend', False)):
            backend = getattr(_backends, packet.pop('backend'))()
            backend._run_worker(packet)

        else:
            raise ValueError("could not recognize packet: {}".format(packet))





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

# Shortcuts to settings
def reset_settings():
    conf.reset()

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

def devel_on():
    conf.devel_on()

def devel_off():
    conf.devel_off()

# Shortcuts to MPI options
def mpi_on(nprocs=None):
    mpi.on(nprocs=nprocs)

def mpi_off():
    mpi.off()

# let's use magic to shutdown the workers when the user-script is complete
atexit.register(mpi.shutdown_workers)

# delete things we don't want exposed to the user at the top-level
# NOTE: we need _sys for reset_settings
del os
del atexit
try:
    del matplotlib
except:
    pass
try:
    del mpi4py
except:
    pass

del logging
del Settings
del MPI
