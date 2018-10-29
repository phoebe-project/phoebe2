__version__ = '1.0.0'

import os as _os
import sys as _sys
import matplotlib as _matplotlib
# If we try to load matplotlib.pyplot on a non-X system, it will fail
# unless 'Agg' is used before the import. All X-systems define the
# 'DISPLAY' environment variable, and all non-X-systems do not. We do make a
# distinction between windows and unix based system. Hence:
if 'DISPLAY' not in _os.environ.keys() and _sys.platform not in ['win32','cygwin']:
    _matplotlib.use('Agg')
elif hasattr(_sys, 'real_prefix'):
    # then we're likely in a virtualenv.  Our best bet is to use the 'TkAgg'
    # backend, but this will require python-tk to be installed on the system
    _matplotlib.use('TkAgg')

from .call import Plot, Mesh
from .axes import Axes
from .figure import Figure

global _figure
_figure = None

def reset():
    global _figure
    _figure = None

def gcf():
    global _figure

    if _figure is None:
        _figure = Figure()

    return _figure

def add_axes(*args):
    return gcf().add_axes(*args)

def plot(*args, **kwargs):
    return gcf().plot(*args, **kwargs)

def mesh(*args, **kwargs):
    return gcf().mesh(*args, **kwargs)

def draw(*args, **kwargs):
    return gcf().draw(*args, **kwargs)

def animate(*args, **kwargs):
    return gcf().animate(*args, **kwargs)

def inline(inline=True):
    if not isinstance(inline, bool):
        raise TypeError("inline must be of type bool")
    common._inline = inline
