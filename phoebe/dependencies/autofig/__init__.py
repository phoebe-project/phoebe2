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
