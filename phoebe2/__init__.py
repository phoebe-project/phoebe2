"""Import PHOEBE 2.0."""

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
    if os.getcwd() == os.path.abspath(os.path.dirname(__file__)):
        raise ImportError('\n\tYou cannot import Phoebe from inside its main source tree.\n')
    # Anywhere else the source tree, it should be possible to import Phoebe.
    # However, it is still not advised to do that.
    else:
        print('\n\tWarning: you are importing Phoebe from inside its source tree.\n')

# If we try to load matplotlib.pyplot on a non-X system, it will fail
# unless 'Agg' is used before the import. All X-systems define the
# 'DISPLAY' environment variable, and all non-X-systems do not. We do make a
# distinction between windows and unix based system. Hence:
import matplotlib
if 'DISPLAY' not in os.environ.keys() and sys.platform not in ['win32','cygwin']:
    matplotlib.use('Agg')


# make packages available at top-level
from .constants import *
from .parameters import *
from .parameters import hierarchy, component, compute, constraint, dataset
from .frontend.bundle import Bundle
from .backend import *
import utils as utils

from phoebe2.atmospheres.limbdark import download_atm

import dynamics as dynamics
import distortions as distortions
import algorithms as algorithms


def logger(*args, **kwargs):
    """
    shortcut to :func:`utils.get_basic_logger`
    """
    return utils.get_basic_logger(*args, **kwargs)