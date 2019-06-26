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
    """
    Reset the current <autofig.figure.Figure> object.

    See also:

    * <autofig.gcf>
    """
    global _figure
    _figure = None

def gcf():
    """
    Access the current <autofig.figure.Figure> object.

    See also:

    * <autofig.reset>

    Returns
    ----------
    * <Figure>
    """
    global _figure

    if _figure is None:
        _figure = Figure()

    return _figure

def add_axes(*args):
    """
    Add a new <autofig.axes.Axes> to the current <autofig.figure.Figure>.

    See also:

    * <autofig.gcf>
    * <autofig.figure.Figure.add_axes>

    Arguments
    ------------
    * `*args`: all arguments are passed on to <autofig.figure.Figure.add_axes>

    Returns
    ----------
    * the return from <autofig.figure.Figure.add_axes>
    """
    return gcf().add_axes(*args)

def plot(*args, **kwargs):
    """
    Add a new <autofig.call.Plot> call to the current <autofig.figure.Figure>.

    See also:

    * <autofig.gcf>
    * <autofig.figure.Figure.plot>
    * <autofig.call.Plot.__init__>

    Arguments
    -----------
    * `*args`: all arguments are passed on to <autofig.figure.Figure.plot>,
        most of which are then passed on to <autofig.call.Plot.__init__>.
    * `**kwargs`: all keyword arguments are passed on to <autofig.figure.Figure.plot>,
        most of which are then passed on to <autofig.call.Plot.__init__>.

    Returns
    ---------
    * the return from <Figure.plot>
    """
    return gcf().plot(*args, **kwargs)

def mesh(*args, **kwargs):
    """
    Add a new <autofig.plot.Mesh> call to the current <autofig.figure.Figure>.

    See also:

    * <autofig.gcf>
    * <autofig.figure.Figure.mesh>
    * <autofig.call.Mesh.__init__>

    Arguments
    ----------
    * `*args`: all arguments are passed on to <autofig.figure.Figure.mesh>,
        most of which are then passed on to <autofig.call.Mesh.__init__>.
    * `**kwargs`: all keyword arguments are passed on to <autofig.figure.Figure.mesh>,
        most of which are then passed on to <autofig.call.Mesh.__init__>.

    Returns
    -----------
    * the return from <Figure.mesh>
    """
    return gcf().mesh(*args, **kwargs)

def draw(*args, **kwargs):
    """
    Draw the current <autofig.figure.Figure>.

    See also:

    * <autofig.gcf>
    * <autofig.figure.Figure.draw>
    * <autofig.axes.Axes.draw>
    * <autofig.call.Plot.draw>
    * <autofig.call.Mesh.draw>

    Arguments
    ----------
    * `*args`: all arguments are passed on to <autofig.figure.Figure.draw>
    * `**kwargs`: all keyword arguments are passed on to <autofig.figure.Figure.draw>

    Returns
    ----------
    * the return from <Figure.draw>
    """
    return gcf().draw(*args, **kwargs)

def animate(*args, **kwargs):
    """
    Animate the current <autofig.figure.Figure>.

    See also:

    * <autofig.gcf>
    * <autofig.figure.Figure.animate>

    Arguments
    ----------
    * `*args`: all arguments are passed on to <autofig.figure.Figure.animate>
    * `**kwargs`: all keyword arguments are passed on to <autofig.figure.Figure.animate>

    Returns
    ----------
    * the return from <autofig.figure.Figure.animate>
    """
    return gcf().animate(*args, **kwargs)

def inline(inline=True):
    """
    Enable/disable inline mode.

    Arguments
    ------------
    * `inline` (bool, optional, default=True): whether inline mode should be
        enabled
    """
    if not isinstance(inline, bool):
        raise TypeError("inline must be of type bool")
    common._inline = inline
