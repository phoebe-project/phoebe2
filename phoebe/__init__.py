"""Import PHOEBE 2.0."""

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

# If we try to load matplotlib.pyplot on a non-X system, it will fail
# unless 'Agg' is used before the import. All X-systems define the
# 'DISPLAY' environment variable, and all non-X-systems do not. We do make a
# distinction between windows and unix based system. Hence:
import matplotlib
if 'DISPLAY' not in os.environ.keys() and sys.platform not in ['win32','cygwin']:
    matplotlib.use('Agg')

class Settings(object):
    def __init__(self):
        # Check to see whether in interactive mode
        import __main__
        # hasattr(__main__, '__file__') will be True if running a python script, but
        # false if in a python or ipython interpreter.
        # sys.flags.interactive will be 1 if the -i flag is sent to python
        self._interactive = not hasattr(__main__, '__file__') or bool(sys.flags.interactive)

        # Check to see whether developer/testing mode is enabled.
        self._devel = os.path.isfile(os.path.expanduser('~/.phoebe_devel_enabled'))
        if self._devel:
            print("WARNING: developer mode enabled, to disable 'rm ~/.phoebe_devel_enabled' and restart phoebe")

    def interactive_on(self):
        self._interactive = True

    def interactive_off(self):
        self._interactive = False

    @property
    def interactive(self):
        return self._interactive

    def devel_on(self):
        self._devel = True

    def devel_off(self):
        self._devel = False

    @property
    def devel(self):
        return self._devel



conf = Settings()





# make packages available at top-level
from .atmospheres.passbands import install_passband, download_passband, list_online_passbands, list_installed_passbands, list_passbands, get_passband
from .constants import *
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

def is_interactive():
    return conf.interactive

def interactive_on():
    conf.interactive_on()

def interactive_off():
    conf.interactive_off()

def is_devel():
    return conf.devel

def devel_on():
    conf.devel_on()

def devel_off():
    conf.devel_off()



