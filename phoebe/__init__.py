"""
>>> import phoebe

Available environment variables:
* PHOEBE_ENABLE_PLOTTING=TRUE/FALSE (whether to import plotting libraries with phoebe: defaults to True)
* PHOEBE_ENABLE_SYMPY=TRUE/FALSE (whether to attempt to import sympy for constraint algebra: defaults to True if sympy installed, otherwise False)
* PHOEBE_ENABLE_ONLINE_PASSBANDS=TRUE/FALSE (whether to query for online passbands and download on-the-fly: defaults to True)
* PHOEBE_PBDIR (directory to search for passbands, in addition to phoebe.list_passband_directories())
* PHOEBE_DOWNLOAD_PASSBAND_DEFAULTS_GZIPPED=TRUE/FALSE (whether to download gzipped version of passbands by default.  Defaults to False.  Note that gzipped files take longer to load and will increase time for import, but take significantly less disk-space.)
* PHOEBE_DOWNLOAD_PASSBAND_DEFAULTS_CONTENT (default content, comma separated for list.  Defaults to 'all')
* PHOEBE_UPDATE_PASSBAND_IGNORE_VERSION=TRUE/FALSE (update passbands that need new content even if the online version is newer than the installed version.  Defaults to False.)
* PHOEBE_ENABLE_MPI=TRUE/FALSE (whether to use internal parallelization: defaults to True if within mpirun, otherwise False, can override in python with phoebe.mpi.on() and phoebe.mpi.off())
* PHOEBE_MPI_NPROCS=INT (number of procs to spawn in mpi is enabled but not running within mpirun: defaults to 4, only applicable if not within mpirun and PHOEBE_ENABLE_MPI=TRUE or phoebe.mpi.on() called, can override in python by passing nprocs to phoebe.mpi.on() or by setting phoebe.mpi.nprocs)
* PHOEBE_MULTIPROC_NPROCS=INT (number of proces to use within multiprocessing.  Multiprocessing is used for solver that support it and when sampling over a distribution in run_compute if MPI is not in use.  Set to 0 to disable multiprocessing and force serial.  Defaults to number of CPUs available.)
* PHOEBE_PBDIR (directory to search for passbands, in addition to phoebe.list_passband_directories())
* PHOEBE_DEVEL=TRUE/FALSE enable developer mode by default

"""

__version__ = '2.3.49'

import os as _os
import sys as _sys
import inspect as _inspect
import multiprocessing as _multiprocessing
import atexit
import re

# People shouldn't import Phoebe from the installation directory (inspired upon
# pymc warning message).
if _os.getcwd().find(_os.path.abspath(_os.path.split(_os.path.split(__file__)[0])[0]))>-1:
    # We have a clash of package name with the standard library: we implement an
    # "io" module and also they do. This means that you can import Phoebe from its
    # main source tree; then there is no difference between io from here and io
    # from the standard library. Thus, if the user loads the package from here
    # it will never work. Instead of letting Python raise the io clash (which
    # is uniformative to the unexperienced user), we raise the importError here
    # with a helpful error message
    raise ImportError('\n\tYou cannot import Phoebe from inside its main source tree.\n')

def _env_variable_string_or_list(key, default):
    value = _os.getenv(key, default)
    if "," in value:
        return value.split(",")
    else:
        return value

def _env_variable_int(key, default):
    value = _os.getenv(key, default)

    return int(value)

def _env_variable_int_or_none(key, default):
    value = _os.getenv(key, default)
    if value is None or isinstance(value, str) and value.lower()=='none':
        return None
    if isinstance(value, str):
        value = float(value)
    return int(value)

def _env_variable_bool(key, default):
    value = _os.getenv(key, default)
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
        if 'DISPLAY' not in _os.environ.keys() and _sys.platform not in ['win32','cygwin']:
            matplotlib.use('Agg')
        elif hasattr(_sys, 'real_prefix'):
            # then we're likely in a virtualenv.  Our best bet is to use the 'TkAgg'
            # backend, but this will require python-tk to be installed on the system
            try:
                matplotlib.use('Agg')
            except:
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
        evars = _os.environ.keys()
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
        return "<MPI mode={} myrank={} nprocs={}>".format(self.mode, self.myrank, self.nprocs)

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

        # TODO: allow this as an option in the settings?
        python = 'python3'

        if self.enabled:
            return 'mpiexec -np %d %s {}' % (self.nprocs, python)
        else:
            return '%s {}' % python

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
        # _sys.flags.interactive will be 1 if the -i flag is sent to python

        # For now we'll set interactive_constraints to True by default, requiring it to
        # explicitly be disabled.
        # See #154 (https://github.com/phoebe-project/phoebe2/issues/154)
        self._interactive_constraints = True

        # We'll set interactive system checks to be on if running within a Python
        # console, but False if running from within a script
        # See #255 (https://github.com/phoebe-project/phoebe2/issues/255)
        self._interactive_checks = not hasattr(__main__, '__file__') or bool(_sys.flags.interactive)

        self._download_passband_defaults = {'content': _env_variable_string_or_list('PHOEBE_DOWNLOAD_PASSBAND_DEFAULTS_CONTENT', 'all'),
                                            'gzipped': _env_variable_bool('PHOEBE_DOWNLOAD_PASSBAND_DEFAULTS_GZIPPED', False)}

        self._update_passband_ignore_version = _env_variable_bool('PHOEBE_UPDATE_PASSBAND_IGNORE_VERSION', False)

        self._multiprocessing_nprocs = _env_variable_int_or_none('PHOEBE_MULTIPROC_NPROCS', None)
        self._progressbars = True

        # And we'll require explicitly setting developer mode on
        self._devel = _env_variable_bool('PHOEBE_DEVEL', False)

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

    def set_download_passband_defaults(self, **kwargs):
        """
        """
        for k,v in kwargs.items():
            if k not in self._download_passband_defaults.keys():
                raise KeyError("{} must be one of {}".format(self._download_passband_defaults.keys()))
            if k=='content' and not (isinstance(v, str) or isinstance(v, list)):
                raise TypeError("content must be of type string or list")
            if k=='gzipped' and not (isinstance(v, bool)):
                raise TypeError("gzipped must be of type bool")
            self._download_passband_defaults[k] = v

    def get_download_passband_defaults(self):
        return self._download_passband_defaults

    @property
    def download_passband_defaults(self):
        return self._download_passband_defaults

    @property
    def update_passband_ignore_version(self):
        return self._update_passband_ignore_version

    def update_passband_ignore_version_on(self):
        self._update_passband_ignore_version = True

    def update_passband_ignore_version_on(self):
        self._update_passband_ignore_version = False

    def multiprocessing_off(self):
        self._multiprocessing_nprocs = 0

    def multiprocessing_on(self):
        self._multiprocessing_nprocs = None

    def multiprocessing_set_nprocs(self, value):
        if not isinstance(value, int):
            return TypeError("must be integer")
        if value > _multiprocessing.cpu_count():
            return ValueError("only {} CPUs available".format(value))
        elif value < 0:
            return ValueError("nprocs must be >= 0")
        self._multiprocessing_nprocs = value

    @property
    def multiprocessing_nprocs(self):
        if self._multiprocessing_nprocs is None:
            return _multiprocessing.cpu_count()
        return self._multiprocessing_nprocs

    def progressbars_on(self):
        self._progressbars = True

    def progressbars_off(self):
        self._progressbars = False

    @property
    def progressbars(self):
        return self._progressbars

conf = Settings()

###############################################################################
##########################        END SETTINGS        #########################
###############################################################################



# make packages available at top-level
from .dependencies.unitsiau2015 import u,c
from .dependencies.nparray import array, linspace, arange, logspace, geomspace, invspace
from .dependencies.distl import gaussian, gaussian_around, normal, boxcar, uniform, uniform_around, histogram_from_bins, histogram_from_data, mvgaussian, mvhistogram_from_data
from .atmospheres.passbands import install_passband, uninstall_passband, uninstall_all_passbands, download_passband, list_passband_online_history, update_passband_available, update_passband, update_all_passbands, list_all_update_passbands_available, list_online_passbands, list_installed_passbands, list_passbands, list_passband_directories, get_passband
from .parameters import hierarchy, component, compute, constraint, dataset, feature, figure, solver
from .frontend.bundle import Bundle
from .backend import backends as _backends
from .solverbackends import solverbackends as _solverbackends
from . import utils as _utils

from . import dynamics as dynamics
from . import distortions as distortions
from . import algorithms as algorithms
import libphoebe

# Shortcut to building logger
def logger(*args, **kwargs):
    """
    Return a basic logger via a log file and/or terminal.

    Example 1: log only to the console, accepting levels "INFO" and above
    ```py
    logger = logger()
    ```

    Example 2: log only to the console, accepting levels "DEBUG" and above
    ```py
    logger(clevel='DEBUG')
    ```

    Example 3: log only to a file, accepting levels "DEBUG" and above
    ```py
    logger(clevel=None,filename='mylog.log')
    ```

    Example 4: log only to a file, accepting levels "INFO" and above
    ```py
    logger(clevel=None,flevel='INFO',filename='mylog.log')
    ```

    Example 5: log to the terminal (INFO and above) and file (DEBUG and above)
    ```py
    logger(filename='mylog.log')
    ```

    Arguments
    ----------
    * `clevel` (string, optional): level to be logged to the console.
        One of: "ERROR", "WARNING", "INFO", "DEBUG".
    * `flevel` (string, optional): level to be logged to the file.
        Must also provide `filename`.  One of: "ERROR", "WARNING", "INFO", "DEBUG".
    * `filename` (string, optional): path to the file to log at the `flevel` level.
    * `style` (string, optional, default='default'): style to use for logging.
        One of: "default", "minimal", "grandpa".
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

        elif hasattr(_solverbackends, packet.get('backend', False)):
            backend = getattr(_solverbackends, packet.pop('backend'))()
            backend._run_worker(packet)

        else:
            raise ValueError("could not recognize packet: {}".format(packet))





# Shortcuts to bundle classmethods
def open(*args, **kwargs):
    return Bundle.open(*args, **kwargs)

open.__doc__ = Bundle.open.__doc__

def load(*args, **kwargs):
    return Bundle.open(*args, **kwargs)

load.__doc__ = Bundle.open.__doc__

def from_legacy(*args, **kwargs):
    return Bundle.from_legacy(*args, **kwargs)

from_legacy.__doc__ = Bundle.from_legacy.__doc__

def from_server(*args, **kwargs):
    return Bundle.from_server(*args, **kwargs)

from_server.__doc__ = Bundle.from_server.__doc__

def default_star(*args, **kwargs):
    return Bundle.default_star(*args, **kwargs)

default_star.__doc__ = Bundle.default_star.__doc__

def default_binary(*args, **kwargs):
    return Bundle.default_binary(*args, **kwargs)

default_binary.__doc__ = Bundle.default_binary.__doc__

def default_contact_binary(*args, **kwargs):
    return Bundle.default_contact_binary(*args, **kwargs)

default_contact_binary.__doc__ = Bundle.default_contact_binary.__doc__

def default_triple(*args, **kwargs):
    return Bundle.default_triple(*args, **kwargs)

default_triple.__doc__ = Bundle.default_triple.__doc__

# Shortcuts to settings
def reset_settings():
    """
    Reset all configuration settings (interactivity, etc) but NOT MPI settings.

    See also:
    * <phoebe.interactive_on>
    * <phoebe.interactive_off>
    * <phoebe.interactive_constraints_on>
    * <phoebe.interactive_constraints_off>
    * <phoebe.interactive_checks_on>
    * <phoebe.interactive_checks_off>
    """
    conf.reset()

def interactive_on():
    """
    Turn on both interactive constraints and interactive checks

    See also:
    * <phoebe.interactive_off>
    * <phoebe.interactive_constraints_on>
    * <phoebe.interactive_checks_on>
    """
    conf.interactive_on()

def interactive_off():
    """
    **USE WITH CAUTION**

    Turn off both interactive constraints and interactive checks

    See also:
    * <phoebe.interactive_on>
    * <phoebe.interactive_constraints_off>
    * <phoebe.interactive_checks_off>
    """
    conf.interactive_off()

def interactive_constraints_on():
    """
    Turn interactive constraints on.  When enabled, PHOEBE will update all
    constraints whenever a <phoebe.parameters.Parameter> value is changed.
    Although this adds to the run-time, it ensures that all values are updated
    when accessed.

    By default, interactive constraints are always on unless disabled.

    See also:
    * <phoebe.interactive_constraints_off>
    """
    conf.interactive_constraints_on()

def interactive_constraints_off():
    """
    **USE WITH CAUTION**

    Turn interactive constraints off.  When disabled, PHOEBE will **NOT** update
    constraints whenever a <phoebe.parameters.Parameter> value is changed, but
    will instead wait until needed (for example, by
    <phoebe.frontend.bundle.Bundle.run_compute>).  Accessing/printing the value
    of a constrained Parameter, may be out-of-date when interactive constraints
    is off.

    By default, interactive constraints are always on unless disabled.

    To update constraints manually, you can call
    <phoebe.frontend.bundle.Bundle.run_delayed_constraints>.

    See also:
    * <phoebe.interactive_constraints_on>
    """
    conf.interactive_constraints_off()

def interactive_checks_on():
    """
    Turn interactive checks on.  When enabled, PHOEBE will run system checks
    (<phoebe.frontend.bundle.Bundle.run_checks>) after any
    <phoebe.parameters.Parameter> value is changed and will log any issues
    to the logger as a warning.  In order to see these messages, you must
    have a logger enabled with at least the "WARNING" level (see <phoebe.logger>).

    Whether interactive checks is on or off, system checks will be run when
    calling <phoebe.frontend.bundle.Bundle.run_compute> and will raise
    an error if failing.

    By default, interactive checks is ON if running PHOEBE in an interactive
    console (or Jupyter notebook), but OFF if running in a script (to save
    time but also save confusing logger messages).

    See also:
    * <phoebe.interactive_checks_off>
    """
    conf.interactive_checks_on()

def interactive_checks_off():
    """
    Turn interactive checks off.  When disabled, PHOEBE will **NOT** run system checks
    (<phoebe.frontend.bundle.Bundle.run_checks>) after any
    <phoebe.parameters.Parameter> value is changed and will **NOT** log any issues
    to the logger as a warning.

    Whether interactive checks is on or off, system checks will be run when
    calling <phoebe.frontend.bundle.Bundle.run_compute> and will raise
    an error if failing.

    To manually run system checks at any time, you can call
    <phoebe.frontend.bundle.Bundle.run_checks>.

    By default, interactive checks is ON if running PHOEBE in an interactive
    console (or Jupyter notebook), but OFF if running in a script (to save
    time but also save confusing logger messages).

    See also:
    * <phoebe.interactive_checks_on>
    """
    conf.interactive_checks_off()

def devel_on():
    conf.devel_on()

def devel_off():
    conf.devel_off()

def set_download_passband_defaults(**kwargs):
    """
    Set default options to use for <phoebe.atmospheres.passbands.download_passband>.

    These can also be set at import time via the following environment variables:
    * PHOEBE_DOWNLOAD_PASSBAND_DEFAULTS_CONTENT (defaults to 'all')
    * PHOEBE_DOWNLOAD_PASSBAND_DEFAULTS_GZIPPED (defaults to FALSE)

    See also:
    * <phoebe.get_download_passband_defaults>
    * <phoebe.atmospheres.passbands.download_passband>
    * <phoebe.atmospheres.passbands.update_passband>
    * <phoebe.atmospheres.passbands.get_passband>

    Arguments
    ------------
    * `content` (string or list, optional): override the current value for
        `content` in <phoebe.get_download_passband_defaults>.
    * `gzipped` (bool, optional): override the current value for `gzipped`
        in <phoebe.get_download_passband_defaults>.

    """
    conf.set_download_passband_defaults(**kwargs)

def get_download_passband_defaults():
    """
    Access default options to use for <phoebe.atmospheres.passbands.download_passband>.

    See also:
    * <phoebe.set_download_passband_defaults>
    * <phoebe.atmospheres.passbands.download_passband>
    * <phoebe.atmospheres.passbands.update_passband>
    * <phoebe.atmospheres.passbands.get_passband>

    Returns
    ---------
    * dictionary of defaults, including `content` and `gzipped`.
    """
    return conf.get_download_passband_defaults()

def update_passband_ignore_version_on():
    """
    Turn ingoring passband versions when checking for necessary updates on.
    <phoebe.frontend.bundle.Bundle.run_checks_compute> checks to see if any
    additional content is required from the used passbands.  If so, these will
    be queried from the online tables if the timestamps match.  Otherwise, an
    error will be raised requiring manually calling <phoebe.atmospheres.passbands.update_passband>.
    By enabling this, this version conflict will be ignored, preventing the need
    to manually update the passbands.

    This can also be set at import time via the following environment variables:
    * PHOEBE_UPDATE_PASSBAND_IGNORE_VERSION (defaults to FALSE)

    See also:
    * <phoebe.update_passband_ignore_version_off>

    """
    conf.update_passband_ignore_version_on()

def update_passband_ignore_version_off():
    """
    Turn ingoring passband versions when checking for necessary updates off.
    <phoebe.frontend.bundle.Bundle.run_checks_compute> checks to see if any
    additional content is required from the used passbands.  If so, these will
    be queried from the online tables if the timestamps match.  Otherwise, an
    error will be raised requiring manually calling <phoebe.atmospheres.passbands.update_passband>.

    This can also be set at import time via the following environment variables:
    * PHOEBE_UPDATE_PASSBAND_IGNORE_VERSION (defaults to FALSE)

    See also:
    * <phoebe.update_passband_ignore_version_on>
    """
    conf.update_passband_ignore_version_on()

# Shortcuts to MPI options
def mpi_on(nprocs=None):
    """
    ENABLE PHOEBE to use MPI (parallelization).

    Default case:
    * If PHOEBE is run within an mpirun environment, MPI is ENABLED by default.
    * If PHOEBE is not run within an mpirun environment, MPI is DISABLED by default.

    When MPI is enabled, PHOEBE will do the following:
    * if within mpirun: uses PHOEBE's built-in per-dataset or per-time
        parallelization for <phoebe.frontend.bundle.Bundle.run_compute>
        and per-model parallelization when possible for
        <phoebe.frontend.bundle.Bundle.run_solver>.
    * if not within mpirun (ie. in a serial python environment): will spawn a
        separate thread at <phoebe.frontend.bundle.Bundle.run_compute>
        and <phoebe.frontend.bundle.Bundle.run_solver>,
        using `nprocs` processors.  This separate thread will be detached
        from the main thread if sending `detach=True` to
        <phoebe.frontend.bundle.Bundle.run_compute> or
        <phoebe.frontend.bundle.Bundle.run_solver>.

    See also:
    * <phoebe.mpi_off>

    Arguments
    ----------
    * `nprocs` (int, optional): number of processors.  Only applicable if **NOT**
        within mpirun (see above).
    """
    mpi.on(nprocs=nprocs)

def mpi_off():
    """
    Run PHOEBE in Serial Mode.

    Default case:
    * If PHOEBE is run within an mpirun environment, MPI is ENABLED by default.
    * If PHOEBE is not run within an mpirun environment, MPI is DISABLED by default.

    When MPI is disabled, PHOEBE will do the following:
    * if within mpirun: PHOEBE will run equally on all processors.  The user can
        customize parallelization with access to `phoebe.mpi.nprocs`,
        `phoebe.mpi.myrank`.
    * if not within mpirun (ie. in a serial python environment): PHOEBE will
        run on a single processor in serial-mode.  Compute jobs can still
        be detached from the main thread by sending `detach=True` to
        <phoebe.frontend.bundle.Bundle.run_compute> or
        <phoebe.frontend.bundle.Bundle.run_solver> but will still run
        on a single processor.

    See also:
    * <phoebe.mpi_on>
    """
    mpi.off()

def multiprocessing_on():
    """
    **NEW IN 2.3.26**

    Enable multiprocessing to use all CPUs available (this is the state by default).
    MPI will always take preference over multiprocessing.  See <phoebe.mpi_on>
    and <phoebe.mpi_off>.

    Multiprocessing is used by
    <phoebe.frontend.bundle.Bundle.run_solver> (for some solvers) and
    <phoebe.frontend.bundle.Bundle.run_compute> when `sample_from` is used.

    See also:
    * <phoebe.multiprocessing_off>
    * <phoebe.multiprocessing_get_nprocs>
    * <phoebe.multiprocessing_set_nprocs>
    """
    conf.multiprocessing_on()

def multiprocessing_off():
    """
    **NEW IN 2.3.26**

    Disable multiprocessing and force serial mode (if MPI is also off: see
    <phoebe.mpi_on> and <phoebe.mpi_off>).

    See also:
    * <phoebe.multiprocessing_on>
    * <phoebe.multiprocessing_get_nprocs>
    * <phoebe.multiprocessing_set_nprocs>
    """
    conf.multiprocessing_off()

def multiprocessing_get_nprocs():
    """
    **NEW IN 2.3.26**

    Get the number of processors used within multiprocessing.

    MPI will always take preference over multiprocessing.  See <phoebe.mpi_on>
    and <phoebe.mpi_off>.

    Multiprocessing is used by
    <phoebe.frontend.bundle.Bundle.run_solver> (for some solvers) and
    <phoebe.frontend.bundle.Bundle.run_compute> when `sample_from` is used.

    See also:
    * <phoebe.multiprocessing_on>
    * <phoebe.multiprocessing_off>
    * <phoebe.multiprocessing_set_nprocs>
    """
    return conf.multiprocessing_nprocs

def multiprocessing_set_nprocs(nprocs):
    """
    **NEW IN 2.3.26**

    Set a custom number of processors to use within multiprocessing.
    MPI will always take preference over multiprocessing.  See <phoebe.mpi_on>
    and <phoebe.mpi_off>.

    Multiprocessing is used by
    <phoebe.frontend.bundle.Bundle.run_solver> (for some solvers) and
    <phoebe.frontend.bundle.Bundle.run_compute> when `sample_from` is used.

    See also:
    * <phoebe.multiprocessing_on>
    * <phoebe.multiprocessing_off>
    * <phoebe.multiprocessing_get_nprocs>
    """
    conf.multiprocessing_set_nprocs(nprocs)

def progressbars_on():
    """
    Enable progressbars. Progressbars require `tqdm` to be installed
    (will silently ignore if not installed).

    See also:
    * <phoebe.progressbars_off>
    """
    conf.progressbars_on()

def progressbars_off():
    """
    Disable progressbars. Progressbars require `tqdm` to be installed
    (will silently ignore if not installed).

    See also:
    * <phoebe.progressbars_on>
    """
    conf.progressbars_off()

# let's use magic to shutdown the workers when the user-script is complete
atexit.register(mpi.shutdown_workers)

# edit API docs for imported functions

def strip_docstring_refs(matchobj):
    text = matchobj.group(0)
    path = text[1:-1]
    return path

def add_nparray_docstring(obj):

    docsprefix = """This is an included dependency from [nparray 1.2.0](https://nparray.readthedocs.io/en/1.2.0/).\n\n===============================================================\n\n"""

    docstring = docsprefix + "\n".join([l.lstrip() for l in obj.__doc__.split("\n")])
    docstring = re.sub(r"(?P<name>\<[0-9a-zA-Z_\.]*\>)", strip_docstring_refs, docstring)

    obj.__doc__ = docstring

add_nparray_docstring(array)
add_nparray_docstring(linspace)
add_nparray_docstring(arange)
add_nparray_docstring(logspace)
add_nparray_docstring(geomspace)
add_nparray_docstring(invspace)


def add_distl_docstring(obj):
    docsprefix = """This is an included dependency from [distl](https://distl.readthedocs.io).\n\n===============================================================\n\n"""

    docstring = docsprefix + "\n".join([l.lstrip() for l in obj.__doc__.split("\n")])
    docstring = re.sub(r"(?P<name>\<[0-9a-zA-Z_\.]*\>)", strip_docstring_refs, docstring)

    obj.__doc__ = docstring

add_distl_docstring(uniform)
add_distl_docstring(boxcar)
# add_distl_docstring(delta)
add_distl_docstring(gaussian)
add_distl_docstring(normal)
add_distl_docstring(histogram_from_bins)
add_distl_docstring(histogram_from_data)
add_distl_docstring(mvgaussian)
add_distl_docstring(mvhistogram_from_data)
add_distl_docstring(uniform_around)
add_distl_docstring(gaussian_around)


# expose available "kinds" per-context
def _get_phoebe_funcs(module, devel=False):
    ignore = ['_empty_array', 'deepcopy', 'fnmatch',
              'download_passband', 'list_installed_passbands', 'list_online_passbands', 'list_passbands', 'parameter_from_json', 'parse_json',
              'send_if_client', 'update_if_client',
              '_add_component', '_add_dataset', '_label_units_lims', '_run_compute',
              'phase_mask_inds']

    if not devel:
        ignore += ['pulsation']
        ignore += ['photodynam']

    mod_split = module.__name__.split('.')
    if mod_split[-1] in ['figure'] or (mod_split[-1] in ['solver'] and 'figure' not in mod_split):
        ret = []
        for sub_module in _inspect.getmembers(module):
            if _inspect.ismodule(sub_module[1]):
                ret += [sub_module[0]+"."+o for o in _get_phoebe_funcs(sub_module[1], devel=devel)]
        return ret

    return [o[0] for o in _inspect.getmembers(module) if _inspect.isfunction(o[1]) and o[0] not in ignore and o[0][0] != '_']


def list_available_components(devel=False):
    """
    List all available 'kinds' for component from <phoebe.parameters.component>.

    See also:
    * <phoebe.list_available_features>
    * <phoebe.list_available_datasets>
    * <phoebe.list_available_computes>
    * <phoebe.list_available_solvers>

    Arguments
    -----------
    * `devel` (bool, default, optional=False): whether to include development-only
        kinds.  See <phoebe.devel_on>.

    Returns
    ---------
    * (list of strings)
    """
    return _get_phoebe_funcs(component, devel=devel)

def list_available_features(devel=False):
    """
    List all available 'kinds' for feature from <phoebe.parameters.feature>.

    See also:
    * <phoebe.list_available_components>
    * <phoebe.list_available_datasets>
    * <phoebe.list_available_computes>
    * <phoebe.list_available_solvers>

    Arguments
    -----------
    * `devel` (bool, default, optional=False): whether to include development-only
        kinds.  See <phoebe.devel_on>.

    Returns
    ---------
    * (list of strings)
    """
    return _get_phoebe_funcs(feature, devel=devel)

def list_available_datasets(devel=False):
    """
    List all available 'kinds' for dataset from <phoebe.parameters.dataset>.

    See also:
    * <phoebe.list_available_components>
    * <phoebe.list_available_features>
    * <phoebe.list_available_computes>
    * <phoebe.list_available_solvers>

    Arguments
    -----------
    * `devel` (bool, default, optional=False): whether to include development-only
        kinds.  See <phoebe.devel_on>.

    Returns
    ---------
    * (list of strings)
    """
    return  _get_phoebe_funcs(dataset, devel=devel)

def list_available_figures(devel=False):
    """
    List all available 'kinds' for figure from <phoebe.parameters.figure>.

    See also:
    * <phoebe.list_available_components>
    * <phoebe.list_available_features>
    * <phoebe.list_available_computes>
    * <phoebe.list_available_solvers>

    Arguments
    -----------
    * `devel` (bool, default, optional=False): whether to include development-only
        kinds.  See <phoebe.devel_on>.

    Returns
    ---------
    * (list of strings)
    """
    return  _get_phoebe_funcs(figure, devel=devel)

def list_available_computes(devel=False):
    """
    List all available 'kinds' for compute from <phoebe.parameters.compute>.

    See also:
    * <phoebe.list_available_components>
    * <phoebe.list_available_features>
    * <phoebe.list_available_datasets>
    * <phoebe.list_available_solvers>

    Arguments
    -----------
    * `devel` (bool, default, optional=False): whether to include development-only
        kinds.  See <phoebe.devel_on>.

    Returns
    ---------
    * (list of strings)
    """
    return _get_phoebe_funcs(compute, devel=devel)

def list_available_solvers(devel=False):
    """
    List all available 'kinds' for solver from <phoebe.parameters.solver>.

    See also:
    * <phoebe.list_available_components>
    * <phoebe.list_available_features>
    * <phoebe.list_available_datasets>
    * <phoebe.list_available_computes>

    Arguments
    -----------
    * `devel` (bool, default, optional=False): whether to include development-only
        kinds.  See <phoebe.devel_on>.

    Returns
    ---------
    * (list of strings)
    """
    return _get_phoebe_funcs(solver, devel=devel)

if _env_variable_bool('PHOEBE_ENABLE_ONLINE_PASSBANDS', 'TRUE'):
    for pb in list_all_update_passbands_available():
        msg = 'passband "{}" has a newer version available.  Run phoebe.list_passband_online_history("{}") to get a list of available changes and phoebe.update_passband("{}") or phoebe.update_all_passbands() to update.'.format(pb, pb, pb)
        # NOTE: we'll print since the logger hasn't been initialized yet.
        print('PHOEBE: {}'.format(msg))

# delete things we don't want exposed to the user at the top-level
# NOTE: we need _sys for reset_settings, that's why its __sys
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

del re
del strip_docstring_refs
del add_nparray_docstring
del add_distl_docstring
