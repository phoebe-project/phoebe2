import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from . import common
from . import callbacks

def _map_none(value):
    if isinstance(value, str):
        if value.lower() == 'none':
            return 'None'
        else:
            return value
    else:
        # NOTE: including None - we want this to fallback on the cycler
        return value

def _to_linebreak_list(thing, N=1):
    if isinstance(thing, list):
        return thing
    else:
        return [thing]*N

class CallGroup(common.Group):
    def __init__(self, items):
        super(CallGroup, self).__init__(Call, [], items)

    @property
    def callbacks(self):
        """
        Returns
        ---------
        * (list) a list of  <autofig.call.Call.callbacks> for each child
            <autofig.call.Call>
        """
        return self._get_attrs('callbacks')

    def connect_callback(self, callback):
        for call in self._items:
            call.connect_callback(callback)

    @property
    def i(self):
        """
        Returns
        ---------
        * (list) a list of  <autofig.call.Call.i> for each child
            <autofig.call.Call>
        """
        return CallDimensionGroup(self._get_attrs('i'))

    @property
    def x(self):
        """
        Returns
        ---------
        * (list) a list of  <autofig.call.Call.x> for each child
            <autofig.call.Call>
        """
        return CallDimensionGroup(self._get_attrs('x'))

    @property
    def y(self):
        """
        Returns
        ---------
        * (list) a list of  <autofig.call.Call.y> for each child
            <autofig.call.Call>
        """
        return CallDimensionGroup(self._get_attrs('y'))

    @property
    def z(self):
        """
        Returns
        ---------
        * (list) a list of  <autofig.call.Call.z> for each child
            <autofig.call.Call>
        """
        return CallDimensionGroup(self._get_attrs('z'))

    @property
    def consider_for_limits(self):
        """
        Returns
        ---------
        * (list) a list of  <autofig.call.Call.consider_for_limits> for each child
            <autofig.call.Call>
        """
        return self._get_attrs('consider_for_limits')

    @consider_for_limits.setter
    def consider_for_limits(self, consider_for_limits):
        return self._set_attrs('consider_for_limits', consider_for_limits)

    def draw(self, *args, **kwargs):
        """
        Calls <autofig.call.Plot.draw> or <autofig.call.Mesh.draw> for each
        <autofig.call.Call> in the <autofig.call.CallGroup>.

        See also:

        * <autofig.draw>
        * <autofig.figure.Figure.draw>
        * <autofig.axes.Axes.draw>
        * <autofig.call.Plot.draw>
        * <autofig.call.Mesh.draw>

        Arguments
        ------------
        * `*args`: all arguments are passed on to each <autofig.call.Call>.
        * `**kwargs`: all keyword arguments are passed on to each
            <autofig.call.Call>.

        Returns
        -----------
        * (list): list of all created matplotlib artists
        """
        # CallGroup.draw
        return_artists = []
        for call in self._items:
            artists = call.draw(*args, **kwargs)
            return_artists += artists

        return return_artists

class PlotGroup(CallGroup):
    @property
    def s(self):
        """
        Returns
        ---------
        * (list) a list of  <autofig.call.Plot.s> for each child
            <autofig.call.Plot>
        """
        return CallDimensionSGroup(self._get_attrs('s'))

    @property
    def c(self):
        """
        Returns
        ---------
        * (list) a list of  <autofig.call.Plot.c> for each child
            <autofig.call.Plot>
        """
        return CallDimensionCGroup(self._get_attrs('c'))

    @property
    def size_scale(self):
        """
        Returns
        ---------
        * (list) a list of  <autofig.call.Plot.size_scale> for each child
            <autofig.call.Plot>
        """
        return self._get_attrs('size_scale')

    @size_scale.setter
    def size_scale(self, size_scale):
        return self._set_attrs('size_scale', size_scale)

class MeshGroup(CallGroup):
    @property
    def fc(self):
        """
        Returns
        ---------
        * (list) a list of  <autofig.call.Mesh.fc> for each child
            <autofig.call.Mesh>
        """
        return CallDimensionCGroup(self._get_attrs('fc'))

    @property
    def ec(self):
        """
        Returns
        ---------
        * (list) a list of  <autofig.call.Mesh.ec> for each child
            <autofig.call.Mesh>
        """
        return CallDimensionCGroup(self._get_attrs('ec'))

def make_callgroup(items):
    if np.all([isinstance(item, Plot) for item in items]):
        return PlotGroup(items)
    elif np.all([isinstance(item, Mesh) for item in items]):
        return MeshGroup(items)
    else:
        return CallGroup(items)

class Call(object):
    def __init__(self, x=None, y=None, z=None, i=None,
                 xerror=None, xunit=None, xlabel=None, xnormals=None,
                 yerror=None, yunit=None, ylabel=None, ynormals=None,
                 zerror=None, zunit=None, zlabel=None, znormals=None,
                 iunit=None, itol=0.0,
                 axorder=None, axpos=None,
                 title=None,
                 label=None,
                 consider_for_limits=True,
                 uncover=False,
                 trail=False,
                 **kwargs):
        """
        Create a <autofig.call.Call> object which defines a single call to
        matplotlib.

        Arguments
        -------------
        * `x` (list/array, optional, default=None): array of values for the x-axes.
            Access via <autofig.call.Call.x>.
        * `y` (list/array, optional, default=None): array of values for the y-axes.
            Access via <autofig.call.Call.y>.
        * `z` (list/array, optional, default=None): array of values for the z-axes.
            Access via <autofig.call.Call.z>
        * `i` (list/array or string, optional, default=None): array of values for
            the independent-variable.  If a string, can be one of: 'x', 'y', 'z'
            to reference an existing array.  Access via <autofig.call.Call.i>.
        * `xerror` (float or list/array, optional, default=None): errors for `x`.
            See <autofig.call.Call.x> and <autofig.call.CallDimensionX.error>.
        * `xunit` (string or astropy unit, optional, default=None): units for `x`.
            See <autofig.call.Call.x> and <autofig.call.CallDimensionX.unit>.
        * `xlabel` (strong, optional, default=None): label for `x`.
            See <autofig.call.Call.x> and <autofig.call.CallDimensionX.label>.
        * `xnormals` (list/array, optional, default=None): normals for `x`.
            Currently ignored.
        * `yerror` (float or list/array, optional, default=None): errors for `y`.
            See <autofig.call.Call.y> and <autofig.call.CallDimensionY.error>.
        * `yunit` (string or astropy unit, optional, default=None): units for `y`.
            See <autofig.call.Call.y> and <autofig.call.CallDimensionY.unit>.
        * `ylabel` (strong, optional, default=None): label for `y`.
            See <autofig.call.Call.y> and <autofig.call.CallDimensionY.label>.
        * `ynormals` (list/array, optional, default=None): normals for `y`.
            Currently ignored.
        * `zerror` (float or list/array, optional, default=None): errors for `z`.
            See <autofig.call.Call.z> and <autofig.call.CallDimensionZ.error>.
        * `zunit` (string or astropy unit, optional, default=None): units for `z`.
            See <autofig.call.Call.z> and <autofig.call.CallDimensionZ.unit>.
        * `zlabel` (strong, optional, default=None): label for `x`.
            See <autofig.call.Call.z> and <autofig.call.CallDimensionZ.label>.
        * `znormals` (list/array, optional, default=None): normals for `z`.
            Currently only used for <autofig.call.Mesh>.
        * `iunit` (string or astropy unit, optional, default=None): units for `i`.
            See <autofig.call.Call.i> and <autofig.call.CallDimensionI.unit>.
        * `itol` (float, optional, default=0.0): see <autofig.call.DimensionI.tol>.
        * `axorder` (int, optional, default=None): see <autofig.call.Call.axorder>.
        * `axpos` (tuple, optional, default=None): see <autofig.call.Call.axpos>.
        * `title` (string, optional, default=None): see <autofig.call.Call.title>.
        * `label` (string, optional, default=None): see <autofig.call.Call.label>.
        * `consider_for_limits` (bool, optional, default=True): see
            <autofig.call.Call.consider_for_limits>.
        * `uncover` (bool, optional, default=False): see <autofig.call.Call.uncover>.
        * `trail` (bool or Float, optional, default=False): see
            <autofig.call.Call.trail>.
        * `**kwargs`: additional keyword arguments are stored and passed on when
            attaching to a parent axes.  See <autofig.axes.Axes.add_call>.

        Returns
        ---------
        * the instantiated <autofig.call.Call> object.
        """
        self._class = 'Call' # just to avoid circular import in order to use isinstance

        self._axes = None
        self._backend_objects = []
        self._callbacks = []

        self._x = CallDimensionX(self, x, xerror, xunit, xlabel, xnormals)
        self._y = CallDimensionY(self, y, yerror, yunit, ylabel, ynormals)
        self._z = CallDimensionZ(self, z, zerror, zunit, zlabel, znormals)

        # defined last so all other dimensions are in place in case indep
        # is a reference and needs to access units, etc
        self._i = CallDimensionI(self, i, iunit, itol)

        self.consider_for_limits = consider_for_limits
        self.uncover = uncover
        self.trail = trail

        self.axorder = axorder
        self.axpos = axpos
        self.title = title
        self.label = label

        self.kwargs = kwargs

        # TODO: add style

    def _get_backend_object():
        return self._backend_artists

    @property
    def callbacks(self):
        return self._callbacks

    def connect_callback(self, callback):
        if not isinstance(callback, str):
            callback = callback.__name__

        if callback not in self.callbacks:
            self._callbacks.append(callback)

    @property
    def axes(self):
        """
        Returns
        --------
        * (<autofig.axes.Axes> or None): the parent axes, if applicable.
        """
        # no setter as this can only be set internally when attaching to an axes
        return self._axes

    @property
    def figure(self):
        """
        Returns
        --------
        * (<autofig.figure.Figure> or None): the parent figure, if applicable.
        """
        # no setter as this can only be set internally when attaching to an axes
        if self.axes is None:
            return None
        return self.axes.figure

    @property
    def i(self):
        """
        Returns
        ----------
        * <autofig.call.CallDimensionI>
        """
        return self._i

    @property
    def indep(self):
        """
        Shortcut to <autofig.call.Call.i>

        Returns
        ----------
        * <autofig.call.CallDimensionI>
        """
        return self.i

    @property
    def x(self):
        """
        Returns
        ----------
        * <autofig.call.CallDimensionX>
        """
        return self._x

    @property
    def y(self):
        """
        Returns
        ----------
        * <autofig.call.CallDimensionY>
        """
        return self._y

    @property
    def z(self):
        """
        Returns
        ----------
        * <autofig.call.CallDimensionZ>
        """
        return self._z

    @property
    def consider_for_limits(self):
        """
        Returns
        -----------
        * (bool): whether the data in this <autofig.call.Call> should be considered
            when determining axes limits.
        """
        return self._consider_for_limits

    @consider_for_limits.setter
    def consider_for_limits(self, consider):
        if not isinstance(consider, bool):
            raise TypeError("consider_for_limits must be of type bool")

        self._consider_for_limits = consider

    @property
    def uncover(self):
        """
        Returns
        ---------
        * (bool): whether uncover is enabled
        """
        return self._uncover

    @uncover.setter
    def uncover(self, uncover):
        if not isinstance(uncover, bool):
            raise TypeError("uncover must be of type bool")

        self._uncover = uncover

    @property
    def trail(self):
        """
        Returns
        ---------
        * (bool or Float): whether trail is enabled.  If a float, then a value
            between 0 and 1 indicating the length of the trail.
        """
        return self._trail

    @trail.setter
    def trail(self, trail):
        if not (isinstance(trail, bool) or isinstance(trail, float)):
            if isinstance(trail, int):
                trail = float(trail)
            else:
                raise TypeError("trail must be of type bool or float")

        if trail < 0 or trail > 1:
            raise ValueError("trail must be between 0 and 1")

        self._trail = trail

    @property
    def axorder(self):
        """
        See tutorial:

        * [Subplot/Axes Positioning](../../tutorials/subplot_positioning.md)

        Returns
        --------
        * (int or None)
        """
        return self._axorder

    @axorder.setter
    def axorder(self, axorder):
        if axorder is None:
            self._axorder = None
            return

        if not isinstance(axorder, int):
            raise TypeError("axorder must be of type int")

        self._axorder = axorder

    @property
    def axpos(self):
        """
        See tutorial:

        * [Subplot/Axes Positioning](../../tutorials/subplot_positioning.md)

        Returns
        --------
        * (tuple or None)
        """
        return self._axpos

    @axpos.setter
    def axpos(self, axpos):
        if axpos is None:
            self._axpos = axpos

            return

        if isinstance(axpos, list) or isinstance(axpos, np.ndarray):
            axpos = tuple(axpos)

        if isinstance(axpos, tuple) and (len(axpos) == 3 or len(axpos) == 6) and np.all(isinstance(ap, int) for ap in axpos):
            self._axpos = axpos

        elif isinstance(axpos, int) and axpos >= 100 and axpos < 1000:
            self._axpos = (int(axpos/100), int(axpos/10 % 10), int(axpos % 10))

        elif isinstance(axpos, int) and axpos >= 110011 and axpos < 999999:
            self._axpos = tuple([int(ap) for ap in str(axpos)])

        else:
            raise ValueError("axpos must be of type int or tuple between 100 and 999 (subplot syntax: ncols, nrows, ind) or 110011 and 999999 (gridspec syntax: ncols, nrows, indx, indy, widthx, widthy)")

    @property
    def title(self):
        """
        Returns
        -----------
        * (str): title used for axes title
        """
        return self._title

    @title.setter
    def title(self, title):
        if title is None:
            self._title = title
            return

        if not isinstance(title, str):
            raise TypeError("title must be of type str")

        self._title = title

    @property
    def label(self):
        """
        Returns
        -----------
        * (str): label used for legends
        """
        return self._label

    @label.setter
    def label(self, label):
        if label is None:
            self._label = label
            return

        if not isinstance(label, str):
            raise TypeError("label must be of type str")

        self._label = label


class Plot(Call):
    def __init__(self, x=None, y=None, z=None, c=None, s=None, i=None,
                       xerror=None, xunit=None, xlabel=None,
                       yerror=None, yunit=None, ylabel=None,
                       zerror=None, zunit=None, zlabel=None,
                       cunit=None, clabel=None, cmap=None,
                       sunit=None, slabel=None, smap=None, smode=None,
                       iunit=None, itol=0.0,
                       axorder=None, axpos=None,
                       title=None,
                       label=None,
                       marker=None,
                       linestyle=None, linebreak=None,
                       highlight=True, uncover=False, trail=False,
                       consider_for_limits=True,
                       **kwargs):
        """
        Create a <autofig.call.Plot> object which defines a single call to
        matplotlib.

        See also:

        * <autofig.call.Mesh>

        Note that the following keyword arguments are not allowed and will raise
        an error suggesting the appropriate autofig argument:

        * `markersize` or `ms`: use `size` or `s`
        * `linewidth` or `lw`: use `size` or `s`


        Arguments
        -------------
        * `x` (list/array, optional, default=None): array of values for the x-axes.
            Access via <autofig.call.Plot.x>.
        * `y` (list/array, optional, default=None): array of values for the y-axes.
            Access via <autofig.call.Plot.y>.
        * `z` (list/array, optional, default=None): array of values for the z-axes.
            Access via <autofig.call.Plot.z>
        * `c` or `color` (list/array, optional, default=None): array of values for the
            color-direction.  Access via <autofig.call.Plot.c>.  Note: `color`
            takes precedence over `c` if both are provided.
        * `s` or `size` (list/array, optional, default=None): array of values for the
            size-direction.  Access via <autofig.call.Plot.s>.  Note: `size` takes
            precedence over `s` if both are provided.
        * `i` (list/array or string, optional, default=None): array of values for
            the independent-variable.  If a string, can be one of: 'x', 'y', 'z',
            'c', 's' to reference an existing array.  Access via
            <autofig.call.Plot.i>.
        * `xerror` (float or list/array, optional, default=None): errors for `x`.
            See <autofig.call.Plot.x> and <autofig.call.CallDimensionX.error>.
        * `xunit` (string or astropy unit, optional, default=None): units for `x`.
            See <autofig.call.Plot.x> and <autofig.call.CallDimensionX.unit>.
        * `xlabel` (strong, optional, default=None): label for `x`.
            See <autofig.call.Plot.x> and <autofig.call.CallDimensionX.label>.
        * `yerror` (float or list/array, optional, default=None): errors for `y`.
            See <autofig.call.Plot.y> and <autofig.call.CallDimensionY.error>.
        * `yunit` (string or astropy unit, optional, default=None): units for `y`.
            See <autofig.call.Plot.y> and <autofig.call.CallDimensionY.unit>.
        * `ylabel` (strong, optional, default=None): label for `y`.
            See <autofig.call.Plot.y> and <autofig.call.CallDimensionY.label>.
        * `zerror` (float or list/array, optional, default=None): errors for `z`.
            See <autofig.call.Plot.z> and <autofig.call.CallDimensionZ.error>.
        * `zunit` (string or astropy unit, optional, default=None): units for `z`.
            See <autofig.call.Plot.z> and <autofig.call.CallDimensionZ.unit>.
        * `zlabel` (strong, optional, default=None): label for `x`.
            See <autofig.call.Plot.z> and <autofig.call.CallDimensionZ.label>.
        * `cerror` (float or list/array, optional, default=None): errors for `c`.
            See <autofig.call.Plot.c> and <autofig.call.CallDimensionC.error>.
        * `cunit` (string or astropy unit, optional, default=None): units for `c`.
            See <autofig.call.Plot.c> and <autofig.call.CallDimensionC.unit>.
        * `clabel` (strong, optional, default=None): label for `c`.
            See <autofig.call.Plot.c> and <autofig.call.CallDimensionC.label>.
        * `serror` (float or list/array, optional, default=None): errors for `s`.
            See <autofig.call.Plot.s> and <autofig.call.CallDimensionS.error>.
        * `sunit` (string or astropy unit, optional, default=None): units for `s`.
            See <autofig.call.Plot.s> and <autofig.call.CallDimensionS.unit>.
        * `slabel` (strong, optional, default=None): label for `s`.
            See <autofig.call.Plot.s> and <autofig.call.CallDimensionS.label>.
        * `iunit` (string or astropy unit, optional, default=None): units for `i`.
            See <autofig.call.Plot.i> and <autofig.call.CallDimensionI.unit>.
        * `itol` (float, optional, default=0.0): see <autofig.call.DimensionI.tol>.
        * `axorder` (int, optional, default=None): see <autofig.call.Plot.axorder>.
        * `axpos` (tuple, optional, default=None): see <autofig.call.Plot.axpos>.
        * `title` (string, optional, default=None): see <autofig.call.Plot.title>.
        * `label` (string, optional, default=None): see <autofig.call.Plot.label>.
        * `marker` or `m` (string, optional, default=None): see <autofig.call.Plot.marker>.
            Note: `marker` takes precedence over `m` if both are provided.
        * `linestyle` or `ls` (string, optional, default=None): see
            <autofig.call.Plot.linestyle>. Note: `linestyle` takes precedence
            over `ls` if both are provided.
        * `linebreak` (string, optional, default=None): see <autofig.call.Plot.linebreak>.
        * `highlight` (bool, optional, default=False): see <autofig.call.Plot.highlight>.
        * `highlight_marker` (string, optional, default=None)
        * `highlight_linestyle` or `highlight_ls` (string, optional, default=None):
            Note: `highlight_linestyle` takes precedence over `highlight_ls` if
            both are provided.
        * `highlight_size` or `highlight_s` (float, optional, default=None):
            Note: `highlight_size` takes precedence over `highlight_s` if both
            are provided.
        * `highlight_color` or `highlight_c` (string, optional, default=None):
            Note: `highlight_color` takes precedence over `highlight_c` if both
            are provided.
        * `consider_for_limits` (bool, optional, default=True): see
            <autofig.call.Call.consider_for_limits>.
        * `uncover` (bool, optional, default=False): see <autofig.call.Call.uncover>.
        * `trail` (bool or Float, optional, default=False): see
            <autofig.call.Call.trail>.
        * `**kwargs`: additional keyword arguments are stored and passed on when
            attaching to a parent axes.  See <autofig.axes.Axes.add_call>.

        Returns
        ---------
        * the instantiated <autofig.call.Plot> object.
        """
        if 'markersize' in kwargs.keys():
            raise ValueError("use 'size' or 's' instead of 'markersize'")
        if 'ms' in kwargs.keys():
            raise ValueError("use 'size' or 's' instead of 'ms'")
        if 'linewidth' in kwargs.keys():
            raise ValueError("use 'size' or 's' instead of 'linewidth'")
        if 'lw' in kwargs.keys():
            raise ValueError("use 'size' or 's' instead of 'lw'")
        size = kwargs.pop('size', None)
        s = size if size is not None else s
        smap = kwargs.pop('sizemap', smap)
        self._s = CallDimensionS(self, s, None, sunit, slabel,
                                 smap=smap, mode=smode)

        color = kwargs.pop('color', None)
        c = color if color is not None else c
        cmap = kwargs.pop('colormap', cmap)
        self._c = CallDimensionC(self, c, None, cunit, clabel, cmap=cmap)

        self._axes = None # super will do this again, but we need it for setting marker, etc
        self._axes_c = None
        self._axes_s = None

        self.highlight = highlight

        highlight_marker = kwargs.pop('highlight_marker', None)
        self.highlight_marker = highlight_marker

        highlight_s = kwargs.pop('highlight_s', None)
        highlight_size = kwargs.pop('highlight_size', highlight_s)
        self.highlight_size = highlight_size

        highlight_c = kwargs.pop('highlight_c', None)
        highlight_color = kwargs.pop('highlight_color', highlight_c)
        self.highlight_color = highlight_color

        highlight_ls = kwargs.pop('highlight_ls', None)
        highlight_linestyle = kwargs.pop('highlight_linestyle', highlight_ls)
        self.highlight_linestyle = highlight_linestyle

        m = kwargs.pop('m', None)
        self.marker = marker if marker is not None else m

        ls = kwargs.pop('ls', None)
        self.linestyle = linestyle if linestyle is not None else ls

        self.linebreak = linebreak

        super(Plot, self).__init__(i=i, iunit=iunit, itol=itol,
                                   x=x, xerror=xerror, xunit=xunit, xlabel=xlabel,
                                   y=y, yerror=yerror, yunit=yunit, ylabel=ylabel,
                                   z=z, zerror=zerror, zunit=zunit, zlabel=zlabel,
                                   consider_for_limits=consider_for_limits,
                                   uncover=uncover, trail=trail,
                                   axorder=axorder, axpos=axpos,
                                   title=title, label=label,
                                   **kwargs
                                   )

        self.connect_callback(callbacks.update_sizes)

    def __repr__(self):
        dirs = []
        for direction in ['i', 'x', 'y', 'z', 's', 'c']:
            if getattr(self, direction).value is not None:
                dirs.append(direction)

        return "<Call:Plot | dims: {}>".format(", ".join(dirs))

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return {'classname': self.__class__.__name__,
                'x': self.x.to_dict(),
                'y': self.y.to_dict(),
                'z': self.z.to_dict(),
                'c': self.c.to_dict(),
                's': self.s.to_dict(),
                'i': self.i.to_dict(),
                'axorder': self._axorder,
                'axpos': self._axpos,
                'title': self._title,
                'label': self._label,
                'marker': self._marker,
                'linestyle': self._linestyle,
                'linebreak': self._linebreak,
                'highlight': self._highlight,
                'highlight_linestyle': self._highlight_linestyle,
                'highlight_size': self._highlight_size,
                'highlight_color': self._highlight_color,
                'highlight_marker': self._highlight_marker,
                'uncover': self._uncover,
                'trail': self._trail,
                'consider_for_limits': self._consider_for_limits}

    @property
    def axes_c(self):
        # currently no setter as this really should be handle by axes.add_call
        return self._axes_c

    @property
    def axes_s(self):
        # currently no setter as this really should be handle by axes.add_call
        return self._axes_s

    @property
    def do_sizescale(self):
        x = self.x.get_value()
        y = self.y.get_value()
        z = self.z.get_value()
        s = self.s.get_value()

        # DETERMINE WHICH SCALINGS WE NEED TO USE
        if x is not None and y is not None:
            return s is not None and not (isinstance(s, float) or isinstance(s, int))
        else:
            return False

    @property
    def do_colorscale(self):
        x = self.x.get_value()
        y = self.y.get_value()
        z = self.z.get_value()
        c = self.c.get_value()

        # DETERMINE WHICH SCALINGS WE NEED TO USE
        if x is not None and y is not None:
            return c is not None and not isinstance(c, str)
        else:
            return False

    @property
    def highlight(self):
        return self._highlight

    @highlight.setter
    def highlight(self, highlight):
        if not isinstance(highlight, bool):
            raise TypeError("highlight must be of type bool")

        self._highlight = highlight

    @property
    def highlight_size(self):
        if self._highlight_size is None:
            # then default to twice the non-highlight size plus an offset
            # so that small markers still have a considerably larger marker

            # TODO: can we make this dependent on i?
            if self.s.mode == 'pt':
                return np.mean(self.get_sizes())*2
            else:
                return np.mean(self.get_sizes())*2

        return self._highlight_size

    @highlight_size.setter
    def highlight_size(self, highlight_size):
        if highlight_size is None:
            self._highlight_size = None
            return

        if not (isinstance(highlight_size, float) or isinstance(highlight_size, int)):
            raise TypeError("highlight_size must be of type float or int")
        if highlight_size <= 0:
            raise ValueError("highlight_size must be > 0")

        self._highlight_size = highlight_size

    @property
    def highlight_marker(self):
        if self._highlight_marker is None:
            return 'o'

        return self._highlight_marker

    @highlight_marker.setter
    def highlight_marker(self, highlight_marker):
        if highlight_marker is None:
            self._highlight_marker = None
            return

        if not isinstance(highlight_marker, str):
            raise TypeError("highlight_marker must be of type str")

        # TODO: make sure valid marker?
        self._highlight_marker = highlight_marker

    @property
    def highlight_color(self):
        # if self._highlight_color is None:
            # return self.get_color()

        return self._highlight_color

    @highlight_color.setter
    def highlight_color(self, highlight_color):
        if highlight_color is None:
            self._highlight_color = None
            return

        if not isinstance(highlight_color, str):
            raise TypeError("highlight_color must be of type str")

        self._highlight_color = common.coloralias.map(highlight_color)

    @property
    def highlight_linestyle(self):
        if self._highlight_linestyle is None:
            return 'None'

        return self._highlight_linestyle

    @highlight_linestyle.setter
    def highlight_linestyle(self, highlight_linestyle):
        if highlight_linestyle is None:
            self._highlight_linestyle = None
            return

        if not isinstance(highlight_linestyle, str):
            raise TypeError("highlight_linestyle must be of type str")

        # TODO: make sure value ls?
        self._highlight_linestyle = highlight_linestyle

    def get_sizes(self, i=None):

        s = self.s.get_value(i=i, unit=self.axes_s.unit if self.axes_s is not None else None)

        if self.do_sizescale:
            if self.axes_s is not None:
                sizes = self.axes_s.normalize(s, i=i)
            else:
                # fallback on 0.01-0.05 mapping for just this call
                sall = self.s.get_value(unit=self.axes_s.unit if self.axes_s is not None else None)
                norm = plt.Normalize(np.nanmin(sall), np.nanmax(sall))
                sizes = norm(s) * 0.04+0.01

        else:
            if s is not None:
                sizes = s
            elif self.s.mode == 'pt':
                sizes = 1
            else:
                sizes = 0.02

        return sizes

    @property
    def s(self):
        return self._s

    @property
    def c(self):
        return self._c

    def get_color(self, colorcycler=None):
        if isinstance(self.c.value, str):
            color = self.c.value
        else:
            # then we'll defer to the cycler.  If we want to color by
            # the dimension, we should call self.c directly
            color = None
        if color is None and colorcycler is not None:
            color = colorcycler.next_tmp
        return color

    @property
    def color(self):
        return self.get_color()

    @color.setter
    def color(self, color):
        # TODO: type and cycler checks
        color = common.coloralias.map(_map_none(color))
        if self.axes is not None:
            self.axes._colorcycler.replace_used(self.get_color(), color)
        self._c.value = color

    def get_cmap(self, cmapcycler=None):
        if isinstance(self.c.value, str):
            return None
        if self.c.value is None:
            return None

        cmap = self.c.cmap
        if cmap is None and cmapcycler is not None:
            cmap = cmapcycler.next_tmp

        return cmap

    def get_marker(self, markercycler=None):
        marker = self._marker
        if marker is None:
            if markercycler is not None:
                marker = markercycler.next_tmp
            else:
                marker = '.'
        return marker

    @property
    def marker(self):
        return self.get_marker()

    @marker.setter
    def marker(self, marker):
        # TODO: type and cycler checks
        marker = _map_none(marker)
        if self.axes is not None:
            self.axes._markercycler.replace_used(self.get_marker(), marker)
        self._marker = marker

    def get_linestyle(self, linestylecycler=None):
        ls = self._linestyle
        if ls is None and linestylecycler is not None:
            ls = linestylecycler.next_tmp
        return ls

    @property
    def linestyle(self):
        return self.get_linestyle()

    @linestyle.setter
    def linestyle(self, linestyle):
        # TODO: type and cycler checks
        linestyle = common.linestylealias.map(_map_none(linestyle))
        if self.axes is not None:
            self.axes._linestylecycler.replace_used(self.get_linestyle(), linestyle)
        self._linestyle = linestyle

    @property
    def linebreak(self):
        if self._linebreak is None:
            return False

        return self._linebreak

    @linebreak.setter
    def linebreak(self, linebreak):
        if linebreak is None:
            self._linebreak = linebreak
            return

        if not isinstance(linebreak, str):
            raise TypeError("linebreak must be of type str, found {} {}".format(type(linebreak), linebreak))

        if not len(linebreak)==2:
            raise ValueError("linebreak must be of length 2")

        if linebreak[0] not in common.dimensions:
            raise ValueError("linebreak must start with one of {}".format(common.dimensions))

        acceptable_ends = ['+', '-']
        if linebreak[1] not in acceptable_ends:
            raise ValueError("linebreak must end with one of {}".format(acceptable_ends))

        self._linebreak = linebreak

    def draw(self, ax=None, i=None,
             colorcycler=None, markercycler=None, linestylecycler=None):
        """
        See also:

        * <autofig.draw>
        * <autofig.figure.Figure.draw>
        * <autofig.axes.Axes.draw>
        * <autofig.call.Mesh.draw>

        Arguments
        -----------
        * `ax`
        * `i`
        * `colorcycler`
        * `markercycler`
        * `linestylecycler`
        """
        # Plot.draw
        if ax is None:
            ax = plt.gca()
        else:
            if not isinstance(ax, plt.Axes):
                raise TypeError("ax must be of type plt.Axes")

        if not (i is None or isinstance(i, float) or isinstance(i, int) or isinstance(i, u.Quantity) or isinstance(i, list) or isinstance(i, np.ndarray)):
            raise TypeError("i must be of type float/int/list/None")

        kwargs = self.kwargs.copy()

        # determine 2D or 3D
        axes_3d = isinstance(ax, Axes3D)
        if (axes_3d and self.axes.projection=='2d') or (not axes_3d and self.axes.projection=='3d'):
            raise ValueError("axes and projection do not agree")

        # marker
        marker = self.get_marker(markercycler=markercycler)

        # linestyle - 'linestyle' has priority over 'ls'
        ls = self.get_linestyle(linestylecycler=linestylecycler)

        # color (NOTE: not necessarily the dimension c)
        color = self.get_color(colorcycler=colorcycler)

        # PREPARE FOR PLOTTING AND GET DATA
        return_artists = []
        # TODO: handle getting in correct units (possibly passed from axes?)
        x = self.x.get_value(i=i, unit=self.axes.x.unit)
        xerr = self.x.get_error(i=i, unit=self.axes.x.unit)
        y = self.y.get_value(i=i, unit=self.axes.y.unit)
        yerr = self.y.get_error(i=i, unit=self.axes.y.unit)
        z = self.z.get_value(i=i, unit=self.axes.z.unit)
        # zerr is handled below, only if axes_3ds
        c = self.c.get_value(i=i, unit=self.axes_c.unit if self.axes_c is not None else None)
        s = self.s.get_value(i=i, unit=self.axes_s.unit if self.axes_s is not None else None)

        # bail on cases where we can't plot.  This could possibly be due to
        # sending Nones or Nans
        # if x is None and y is None:
        #     return []
        # if x is None and len(y) > 1:
        #     return []
        # if y is None and len(x) > 1:
        #     return []

        if axes_3d:
            zerr = self.z.get_error(i=i, unit=self.axes.z.unit)
        else:
            zerr = None

        # then we need to loop over the linebreaks
        if isinstance(x, list) or isinstance(y, list):
            linebreak_n = len(x) if isinstance(x, list) else len(y)
        else:
            linebreak_n = 1

        xs = _to_linebreak_list(x, linebreak_n)
        xerrs = _to_linebreak_list(xerr, linebreak_n)
        ys = _to_linebreak_list(y, linebreak_n)
        yerrs = _to_linebreak_list(yerr, linebreak_n)
        zs = _to_linebreak_list(z, linebreak_n)
        # zerrs = _to_linebreak_list(zerr, linebreak_n)
        cs = _to_linebreak_list(c, linebreak_n)
        ss = _to_linebreak_list(s, linebreak_n)

        for loop1,(x,xerr,y,yerr,z,c,s) in enumerate(zip(xs, xerrs, ys, yerrs, zs, cs, ss)):
            if axes_3d:
                data = np.array([x, y, z])
                points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            else:
                data = np.array([x, y])
                points = np.array([x, y]).T.reshape(-1, 1, 2)

            # segments are used for LineCollection
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # DETERMINE WHICH SCALINGS WE NEED TO USE
            do_colorscale = self.do_colorscale
            do_sizescale = self.do_sizescale
            if x is not None and y is not None:
                do_colorscale = c is not None and not isinstance(c, str)
                do_sizescale = s is not None and not (isinstance(s, float) or isinstance(s, int))
            else:
                do_colorscale = False
                do_sizescale = False

            # DETERMINE PER-DATAPOINT Z-ORDERS
            zorders, do_zorder = self.axes.z.get_zorders(z, i=i)
            if axes_3d:
                # TODO: we probably want to re-implement zorder, but then we need to
                # sort in the *projected* z rather than data-z.  We'll also need to
                # figure out why LineCollection is complaining about the input shape
                do_zorder = False

            # ALLOW ACCESS TO COLOR FOR I OR LOOP
            # TODO: in theory these could be exposed (maybe not the loop, but i)
            def get_color_i(i, default=color):
                if do_colorscale and self.axes_c is not None:
                    cmap = self.axes_c.cmap
                    norm = self.axes_c.get_norm(i=i)
                    ci = self.axes.c.get_value(i=i)
                    return plt.get_cmap(cmap)(norm(ci))
                else:
                    return default

            def get_color_loop(loop, do_zorder, default=color):
                if do_colorscale and self.axes_c is not None:
                    cmap = self.axes_c.cmap
                    norm = self.axes_c.get_norm(i=i)
                    if do_zorder:
                        cloop = c[loop]
                    else:
                        cloop = c
                    return plt.get_cmap(cmap)(norm(cloop))
                else:
                    return default

            # BUILD KWARGS NEEDED FOR EACH CALL TO ERRORBAR
            def error_kwargs_loop(xerr, yerr, zerr, loop, do_zorder):
                def _get_error(errorarray, loop, do_zorder):
                    if errorarray is None:
                        return None
                    elif do_zorder:
                        return errorarray[loop]
                    else:
                        return errorarray

                error_kwargs = {'xerr': _get_error(xerr, loop, do_zorder),
                                'yerr': _get_error(yerr, loop, do_zorder)}

                if axes_3d:
                    error_kwargs['zerr'] = _get_error(zerr, loop, do_zorder)

                error_kwargs['ecolor'] = get_color_loop(loop, do_zorder)

                # not so sure that we want the errorbar linewidth to adjust based
                # on size-scaling... but in theory we could do something like this:
                # error_kwargs['elinewidth'] = sizes[loop]

                return error_kwargs

            # BUILD KWARGS NEEDED FOR EACH CALL TO LINECOLLECTION
            lc_kwargs_const = {}
            lc_kwargs_const['linestyle'] = ls
            if do_colorscale:
                lc_kwargs_const['norm'] = self.axes_c.get_norm(i=i) if self.axes_c is not None else None
                lc_kwargs_const['cmap'] = self.axes_c.cmap if self.axes_c is not None else None
            else:
                lc_kwargs_const['color'] = color

            # also set self._sizes so its accessible from the callback which
            # will actually handle setting the sizes
            sizes = self.get_sizes(i)
            self._sizes = sizes

            def sizes_loop(loop, do_zorder):
                if do_zorder:
                    if isinstance(sizes, float):
                        return sizes
                    return sizes[loop]
                else:
                    return sizes

            def lc_kwargs_loop(lc_kwargs, loop, do_zorder):
                if do_colorscale:
                    # nothing to do here, the norm and map are passed rather than values
                    pass
                if do_sizescale:
                    # linewidth is handled by the callback
                    pass

                return lc_kwargs

            # BUILD KWARGS NEEDED FOR EACH CALL TO SCATTER
            sc_kwargs_const = {}
            sc_kwargs_const['marker'] = marker
            sc_kwargs_const['linewidths'] = 0 # linewidths = 0 removes the black edge
            sc_kwargs_const['edgecolors'] = 'none'
            if do_colorscale:
                sc_kwargs_const['norm'] = self.axes_c.get_norm(i=i) if self.axes_c is not None else None
                sc_kwargs_const['cmap'] = self.axes_c.cmap if self.axes_c is not None else None
                # we'll set sc_kwargs['cmap'] per-loop in the function below
            else:
                sc_kwargs_const['color'] = color


            def sc_kwargs_loop(sc_kwargs, loop, do_zorder):
                if do_colorscale:
                    if do_zorder:
                        sc_kwargs['color'] = c[loop]
                    else:
                        sc_kwargs['color'] = c
                # if do_sizescale:
                    # if do_zorder:
                        # sc_kwargs['s'] = self.get_markersize(sizes[loop], scatter=True)
                    # else:
                        # sc_kwargs['s'] = self.get_markersize(sizes, scatter=True)

                return sc_kwargs

            # DRAW IF X AND Y ARE ARRAYS
            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                # LOOP OVER DATAPOINTS so that each can be drawn with its own zorder
                if do_zorder:
                    datas = data.T
                    segments = segments
                    zorders = zorders
                else:
                    datas = [data]
                    zorders = [zorders]
                    segments = [segments]

                for loop2, (datapoint, segment, zorder) in enumerate(zip(datas, segments, zorders)):
                    return_artists_this_loop = []
                    # DRAW ERRORBARS, if applicable
                    # NOTE: we never pass a label here to avoid duplicate entries
                    # the actual datapoints are handled and labeled separately.
                    # Unfortunately this means the error bar will not be included
                    # in the styling of the legend.
                    if xerr is not None or yerr is not None or zerr is not None:
                        artists = ax.errorbar(*datapoint,
                                               fmt='', linestyle='None',
                                               zorder=zorder,
                                               label=None,
                                               **error_kwargs_loop(xerr, yerr, zerr, loop2, do_zorder))

                        # NOTE: these are currently not included in return_artists
                        # so they don't scale according to per-element sizes.
                        # But we may want to return them for completeness and may
                        # want some way of setting the size of the errobars,
                        # maybe similar to how highlight_size is handled
                        # errorbar actually returns a Container object of artists,
                        # so we need to cast to a list
                        # for artist_list in list(artists):
                            # if isinstance(artist_list, tuple):
                                # return_artists += list(artist_list)
                            # else:
                                # return_artists += [artist_list]

                    if do_colorscale or do_sizescale or do_zorder or marker in ['x', '+']:
                        # DRAW LINECOLLECTION, if applicable
                        if ls.lower() != 'none':
                            # TODO: color and zorder are assigned from the LEFT point in
                            # the segment.  It may be nice to interpolate from LEFT-RIGHT
                            # by accessing zorder[loop+1] and c[loop+1]
                            if do_zorder:
                                segments = (segment,)
                            else:
                                segments = segment

                            if axes_3d:
                                lccall = Line3DCollection
                            else:
                                lccall = LineCollection

                            # we'll only include this in the legend for the first loop
                            # and if the marker isn't going to get its own entry.
                            # Unfortunately this means in these cases the
                            # marker will get precedent in the legend if both
                            # marker and linestyle are present
                            lc = lccall(segments,
                                        zorder=zorder,
                                        label=self.label if loop1==0 and loop2==0 and marker.lower()=='none' else None,
                                        **lc_kwargs_loop(lc_kwargs_const, loop2, do_zorder))

                            if do_colorscale:
                                if do_zorder:
                                    lc.set_array(np.array([c[loop2]]))
                                else:
                                    lc.set_array(c)


                            return_artists_this_loop.append(lc)
                            ax.add_collection(lc)


                        # DRAW SCATTER, if applicable
                        if marker.lower() != 'none':
                            artist = ax.scatter(*datapoint,
                                                zorder=zorder,
                                                label=self.label if loop1==0 and loop2==0 else None,
                                                **sc_kwargs_loop(sc_kwargs_const, loop2, do_zorder))

                            return_artists_this_loop.append(artist)


                    else:
                        # let's use plot whenever possible... it'll be faster
                        # and will guarantee that the linestyle looks correct
                        artists = ax.plot(*datapoint,
                                          marker=marker,
                                          ls=ls,
                                          mec='none',
                                          color=color,
                                          label=self.label if loop1==0 and loop2==0 else None)

                        return_artists_this_loop += artists

                    size_this_loop = sizes_loop(loop2, do_zorder)
                    for artist in return_artists_this_loop:
                        # store the sizes so they can be rescaled appropriately by
                        # the callback
                        artist._af_sizes = size_this_loop

                    return_artists += return_artists_this_loop



            # DRAW IF X OR Y ARE NOT ARRAYS
            if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
                # TODO: can we do anything in 3D?
                if x is not None:
                    artist = ax.axvline(x, ls=ls, color=color, label=self.label)
                    return_artists += [artist]

                if y is not None:
                    artist = ax.axhline(y, ls=ls, color=color, label=self.label)
                    return_artists += [artist]

        # DRAW HIGHLIGHT, if applicable (outside per-datapoint loop)
        if self.highlight and i is not None:
            if self.highlight_linestyle != 'None' and self.i.is_reference:
                i_direction = self.i.reference
                if i_direction == 'x':
                    linefunc = 'axvline'
                elif i_direction == 'y':
                    linefunc = 'axhline'
                else:
                    # TODO: can we do anything if in z?
                    linefunc = None

                if linefunc is not None:
                    artist = getattr(ax, linefunc)(i,
                                                   ls=self.highlight_linestyle,
                                                   color=self.highlight_color if self.highlight_color is not None else color)

                    artist._af_highlight = True
                    return_artists += [artist]


            if axes_3d:
                # I do not understand why, but matplotlib requires these to be
                # iterable when in 3d projection
                highlight_data = ([self.x.highlight_at_i(i)],
                                  [self.y.highlight_at_i(i)],
                                  [self.z.highlight_at_i(i)])
            else:
                highlight_data = (self.x.highlight_at_i(i),
                                  self.y.highlight_at_i(i))

            artists = ax.plot(*highlight_data,
                              marker=self.highlight_marker,
                              ls=self.highlight_linestyle,
                              color=self.highlight_color if self.highlight_color is not None else color)

            for artist in artists:
                artist._af_highlight=True
            return_artists += artists

        self._backend_objects = return_artists

        for artist in return_artists:
            callbacks._connect_to_autofig(self, artist)

            for callback in self.callbacks:
                callback_callable = getattr(callbacks, callback)
                callback_callable(artist, self)

        return return_artists

class FillBetween(Call):
    def __init__(self, x=None, y=None, c=None, i=None,
                   xunit=None, xlabel=None,
                   yunit=None, ylabel=None,
                   cunit=None, clabel=None, cmap=None,
                   iunit=None, itol=0.0,
                   axorder=None, axpos=None,
                   title=None,
                   label=None,
                   linebreak=None,
                   uncover=False, trail=False,
                   consider_for_limits=True,
                   **kwargs):
        """
        Create a <autofig.call.FillBetween> object which defines a single call to
        matplotlib.

        See also:

        * <autofig.call.Plot>
        * <autofig.call.Mesh>

        Arguments
        -------------
        * `x` (list/array, optional, default=None): array of values for the x-axes.
            Access via <autofig.call.FillBetween.x>.
        * `y` (list/array, optional, default=None): array of values for the y-axes.
            Must have shape (len(x), 2)
            Access via <autofig.call.FillBetween.y>.
        * `c` or `color` (list/array, optional, default=None): array of values for the
            color-direction.  Access via <autofig.call.FillBetween.c>.  Note: `color`
            takes precedence over `c` if both are provided.
        * `i` (list/array or string, optional, default=None): array of values for
            the independent-variable.  If a string, can be one of: 'x', 'y', 'z',
            'c', 's' to reference an existing array.  Access via
            <autofig.call.FillBetween.i>.
        * `xunit` (string or astropy unit, optional, default=None): units for `x`.
            See <autofig.call.FillBetween.x> and <autofig.call.CallDimensionX.unit>.
        * `xlabel` (strong, optional, default=None): label for `x`.
            See <autofig.call.FillBetween.x> and <autofig.call.CallDimensionX.label>.
        * `yunit` (string or astropy unit, optional, default=None): units for `y`.
            See <autofig.call.FillBetween.y> and <autofig.call.CallDimensionY.unit>.
        * `ylabel` (strong, optional, default=None): label for `y`.
            See <autofig.call.FillBetween.y> and <autofig.call.CallDimensionY.label>.
        * `iunit` (string or astropy unit, optional, default=None): units for `i`.
            See <autofig.call.FillBetween.i> and <autofig.call.CallDimensionI.unit>.
        * `itol` (float, optional, default=0.0): see <autofig.call.DimensionI.tol>.
        * `axorder` (int, optional, default=None): see <autofig.call.FillBetween.axorder>.
        * `axpos` (tuple, optional, default=None): see <autofig.call.FillBetween.axpos>.
        * `title` (string, optional, default=None): see <autofig.call.FillBetween.title>.
        * `label` (string, optional, default=None): see <autofig.call.FillBetween.label>.
        * `linebreak` (string, optional, default=None): see <autofig.call.FillBetween.linebreak>.
        * `consider_for_limits` (bool, optional, default=True): see
            <autofig.call.Call.consider_for_limits>.
        * `uncover` (bool, optional, default=False): see <autofig.call.Call.uncover>.
        * `trail` (bool or Float, optional, default=False): see
            <autofig.call.Call.trail>.
        * `**kwargs`: additional keyword arguments are stored and passed on when
            attaching to a parent axes.  See <autofig.axes.Axes.add_call>.

        Returns
        ---------
        * the instantiated <autofig.call.FillBetween> object.
        """
        color = kwargs.pop('color', None)
        c = color if color is not None else c
        cmap = kwargs.pop('colormap', cmap)
        self._c = CallDimensionC(self, c, None, cunit, clabel, cmap=cmap)

        self._axes = None # super will do this again, but we need it for setting marker, etc
        self._axes_c = None

        color = kwargs.pop('color', None)
        c = color if color is not None else c
        cmap = kwargs.pop('colormap', cmap)
        self._c = CallDimensionC(self, c, None, cunit, clabel, cmap=cmap)

        self.alpha = kwargs.pop('alpha', 0.6)

        self.linebreak = linebreak

        if x is None:
            raise TypeError("x cannot be None for FillBetween")
        x = np.asarray(x)

        if y is None:
            raise TypeError("y cannot be None for FillBetween")
        y = np.asarray(y)

        if y.shape not in [(len(x), 2), (len(x), 3)]:
            raise ValueError("y must be of shape ({}, 2) or ({}, 3), not {}".format(len(x), len(x), y.shape))

        super(FillBetween, self).__init__(i=i, iunit=iunit, itol=itol,
                                          x=x, xunit=xunit, xlabel=xlabel,
                                          y=y, yunit=yunit, ylabel=ylabel,
                                          consider_for_limits=consider_for_limits,
                                          uncover=uncover, trail=trail,
                                          axorder=axorder, axpos=axpos,
                                          title=title, label=label,
                                          **kwargs
                                          )


        # self.connect_callback(callbacks.update_sizes)

    def __repr__(self):
        dirs = []
        for direction in ['i', 'x', 'y', 'c']:
            if getattr(self, direction).value is not None:
                dirs.append(direction)

        return "<Call:FillBetween | dims: {}>".format(", ".join(dirs))

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return {'classname': self.__class__.__name__,
                'x': self.x.to_dict(),
                'y': self.y.to_dict(),
                'c': self.c.to_dict(),
                'i': self.i.to_dict(),
                'axorder': self._axorder,
                'axpos': self._axpos,
                'title': self._title,
                'label': self._label,
                'uncover': self._uncover,
                'trail': self._trail,
                'consider_for_limits': self._consider_for_limits}

    @property
    def axes_c(self):
        # currently no setter as this really should be handle by axes.add_call
        return self._axes_c

    @property
    def do_colorscale(self):
        x = self.x.get_value()
        y = self.y.get_value()
        c = self.c.get_value()

        # DETERMINE WHICH SCALINGS WE NEED TO USE
        if x is not None and y is not None:
            return c is not None and not isinstance(c, str)
        else:
            return False

    @property
    def c(self):
        return self._c

    def get_color(self, colorcycler=None):
        if isinstance(self.c.value, str):
            color = self.c.value
        else:
            # then we'll defer to the cycler.  If we want to color by
            # the dimension, we should call self.c directly
            color = None
        if color is None and colorcycler is not None:
            color = colorcycler.next_tmp
        return color

    @property
    def color(self):
        return self.get_color()

    @color.setter
    def color(self, color):
        # TODO: type and cycler checks
        color = common.coloralias.map(_map_none(color))
        if self.axes is not None:
            self.axes._colorcycler.replace_used(self.get_color(), color)
        self._c.value = color

    def get_cmap(self, cmapcycler=None):
        if isinstance(self.c.value, str):
            return None
        if self.c.value is None:
            return None

        cmap = self.c.cmap
        if cmap is None and cmapcycler is not None:
            cmap = cmapcycler.next_tmp

        return cmap

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, float):
            raise TypeError("alpha must be of type float")
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1")

        self._alpha = alpha

    @property
    def linebreak(self):
        if self._linebreak is None:
            return False

        return self._linebreak

    @linebreak.setter
    def linebreak(self, linebreak):
        if linebreak is None:
            self._linebreak = linebreak
            return

        if not isinstance(linebreak, str):
            raise TypeError("linebreak must be of type str, found {} {}".format(type(linebreak), linebreak))

        if not len(linebreak)==2:
            raise ValueError("linebreak must be of length 2")

        if linebreak[0] not in common.dimensions:
            raise ValueError("linebreak must start with one of {}".format(common.dimensions))

        acceptable_ends = ['+', '-']
        if linebreak[1] not in acceptable_ends:
            raise ValueError("linebreak must end with one of {}".format(acceptable_ends))

        self._linebreak = linebreak

    def draw(self, ax=None, i=None,
             colorcycler=None, markercycler=None, linestylecycler=None):
        """
        See also:

        * <autofig.draw>
        * <autofig.figure.Figure.draw>
        * <autofig.axes.Axes.draw>

        Arguments
        -----------
        * `ax`
        * `i`
        * `colorcycler`
        * `markercycler`: ignored
        * `linestylecycler`: ignored
        """

        # Plot.draw
        if ax is None:
            ax = plt.gca()
        else:
            if not isinstance(ax, plt.Axes):
                raise TypeError("ax must be of type plt.Axes")

        if not (i is None or isinstance(i, float) or isinstance(i, int) or isinstance(i, u.Quantity) or isinstance(i, list) or isinstance(i, np.ndarray)):
            raise TypeError("i must be of type float/int/list/None")

        kwargs = self.kwargs.copy()

        # determine 2D or 3D
        axes_3d = isinstance(ax, Axes3D)
        if (axes_3d and self.axes.projection=='2d') or (not axes_3d and self.axes.projection=='3d'):
            raise ValueError("axes and projection do not agree")

        # color (NOTE: not necessarily the dimension c)
        color = self.get_color(colorcycler=colorcycler)

        # PREPARE FOR PLOTTING AND GET DATA
        return_artists = []
        # TODO: handle getting in correct units (possibly passed from axes?)
        x = self.x.get_value(i=i, unit=self.axes.x.unit)
        y = self.y.get_value(i=i, unit=self.axes.y.unit)
        if isinstance(y, list):
            y = [yi.T for yi in y]
        else:
            y = y.T

        c = self.c.get_value(i=i, unit=self.axes_c.unit if self.axes_c is not None else None)

        # then we need to loop over the linebreaks
        if isinstance(x, list) or isinstance(y, list):
            linebreak_n = len(x) if isinstance(x, list) else len(y)
        else:
            linebreak_n = 1

        xs = _to_linebreak_list(x, linebreak_n)
        ys = _to_linebreak_list(y, linebreak_n)
        cs = _to_linebreak_list(c, linebreak_n)

        for loop1,(x,y,c) in enumerate(zip(xs, ys, cs)):
            data = np.array([x, y[0], y[-1]])
            if len(y) == 3:
                data_middle = np.array([x, y[1]])
            else:
                data_middle = None
            # points = np.array([x, y1, y2]).T.reshape(-1, 1, 3)

            # segments are used for LineCollection
            # segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # DETERMINE WHICH SCALINGS WE NEED TO USE
            do_colorscale = self.do_colorscale
            if x is not None and y is not None:
                do_colorscale = c is not None and not isinstance(c, str)
            else:
                do_colorscale = False

            # ALLOW ACCESS TO COLOR FOR I OR LOOP
            # TODO: in theory these could be exposed (maybe not the loop, but i)
            # def get_color_i(i, default=color):
            #     if do_colorscale and self.axes_c is not None:
            #         cmap = self.axes_c.cmap
            #         norm = self.axes_c.get_norm(i=i)
            #         ci = self.axes.c.get_value(i=i)
            #         return plt.get_cmap(cmap)(norm(ci))
            #     else:
            #         return default
            #
            # def get_color_loop(loop, do_zorder, default=color):
            #     if do_colorscale and self.axes_c is not None:
            #         cmap = self.axes_c.cmap
            #         norm = self.axes_c.get_norm(i=i)
            #         if do_zorder:
            #             cloop = c[loop]
            #         else:
            #             cloop = c
            #         return plt.get_cmap(cmap)(norm(cloop))
            #     else:
            #         return default

            fb_kwargs = {}
            fb_kwargs['color'] = color
            fb_kwargs['alpha'] = self.alpha # defaults to 0.6
            if do_colorscale:
                fb_kwargs['norm'] = self.axes_c.get_norm(i=i) if self.axes_c is not None else None
                fb_kwargs['cmap'] = self.axes_c.cmap if self.axes_c is not None else None

            artist = ax.fill_between(*data, **fb_kwargs)
            return_artists += [artist]

            if data_middle is not None:
                _ = fb_kwargs.pop('alpha')
                fb_kwargs['linestyle'] = 'solid'
                fb_kwargs['marker'] = 'None'
                artists = ax.plot(*data_middle, **fb_kwargs)
                return_artists += artists

        self._backend_objects = return_artists

        for artist in return_artists:
            callbacks._connect_to_autofig(self, artist)

            for callback in self.callbacks:
                callback_callable = getattr(callbacks, callback)
                callback_callable(artist, self)

        return return_artists


class Mesh(Call):
    def __init__(self, x=None, y=None, z=None, fc=None, ec=None, i=None,
                       xerror=None, xunit=None, xlabel=None, xnormals=None,
                       yerror=None, yunit=None, ylabel=None, ynormals=None,
                       zerror=None, zunit=None, zlabel=None, znormals=None,
                       fcunit=None, fclabel=None, fcmap=None,
                       ecunit=None, eclabel=None, ecmap=None,
                       iunit=None, itol=0.0,
                       axorder=None, axpos=None,
                       title=None, label=None,
                       linestyle=None,
                       consider_for_limits=True,
                       uncover=True,
                       trail=0,
                       exclude_back=False,
                       **kwargs):
        """
        Create a <autofig.call.Mesh> object which defines a single call to
        matplotlib.

        See also:

        * <autofig.call.Plot>


        Arguments
        -------------
        * `x` (list/array, optional, default=None): array of values for the x-axes.
            Access via <autofig.call.Mesh.x>.
        * `y` (list/array, optional, default=None): array of values for the y-axes.
            Access via <autofig.call.Mesh.y>.
        * `z` (list/array, optional, default=None): array of values for the z-axes.
            Access via <autofig.call.Mesh.z>
        * `fc` or `facecolor` (list/array, optional, default=None): array of values for the
            facecolor-direction.  Access via <autofig.call.Mesh.fc>.  Note: `facecolor`
            takes precedence over `fc` if both are provided.
        * `ec` or `edgecolor` (list/array, optional, default=None): array of values for the
            edgecolor-direction.  Access via <autofig.call.Mesh.ec>.  Note: `edgecolor`
            takes precedence over `ec` if both are provided.
        * `i` (list/array or string, optional, default=None): array of values for
            the independent-variable.  If a string, can be one of: 'x', 'y', 'z',
            'fc', 'ec' to reference an existing array.  Access via
            <autofig.call.Mesh.i>.
        * `xerror` (float or list/array, optional, default=None): errors for `x`.
            See <autofig.call.Mesh.x> and <autofig.call.CallDimensionX.error>.
        * `xunit` (string or astropy unit, optional, default=None): units for `x`.
            See <autofig.call.Mesh.x> and <autofig.call.CallDimensionX.unit>.
        * `xlabel` (strong, optional, default=None): label for `x`.
            See <autofig.call.Mesh.x> and <autofig.call.CallDimensionX.label>.
        * `xnormals` (list/array, optional, default=None): normals for `x`.
            Currently ignored.
            See <autofig.call.Mesh.x> and <autofig.call.CallDimensionX.normals>.
        * `yerror` (float or list/array, optional, default=None): errors for `y`.
            See <autofig.call.Mesh.y> and <autofig.call.CallDimensionY.error>.
        * `yunit` (string or astropy unit, optional, default=None): units for `y`.
            See <autofig.call.Mesh.y> and <autofig.call.CallDimensionY.unit>.
        * `ylabel` (strong, optional, default=None): label for `y`.
            See <autofig.call.Mesh.y> and <autofig.call.CallDimensionY.label>.
        * `ynormals` (list/array, optional, default=None): normals for `y`.
            Currently ignored.
            See <autofig.call.Mesh.y> and <autofig.call.CallDimensionY.normals>.
        * `zerror` (float or list/array, optional, default=None): errors for `z`.
            See <autofig.call.Mesh.z> and <autofig.call.CallDimensionZ.error>.
        * `zunit` (string or astropy unit, optional, default=None): units for `z`.
            See <autofig.call.Mesh.z> and <autofig.call.CallDimensionZ.unit>.
        * `zlabel` (strong, optional, default=None): label for `x`.
            See <autofig.call.Mesh.z> and <autofig.call.CallDimensionZ.label>.
        * `znormals` (list/array, optional, default=None): normals for `z`.
            If provided then the back of the mesh can be ignored by setting
            `exclude_back=True`.
            See <autofig.call.Mesh.z> and <autofig.call.CallDimensionZ.normals>.
        * `fcerror` (float or list/array, optional, default=None): errors for `fc`.
            See <autofig.call.Mesh.fc> and <autofig.call.CallDimensionC.error>.
        * `fcunit` (string or astropy unit, optional, default=None): units for `fc`.
            See <autofig.call.Mesh.fc> and <autofig.call.CallDimensionC.unit>.
        * `fclabel` (strong, optional, default=None): label for `fc`.
            See <autofig.call.Mesh.fc> and <autofig.call.CallDimensionC.label>.
        * `ecerror` (float or list/array, optional, default=None): errors for `ec`.
            See <autofig.call.Mesh.ec> and <autofig.call.CallDimensionC.error>.
        * `ecunit` (string or astropy unit, optional, default=None): units for `ec`.
            See <autofig.call.Mesh.ec> and <autofig.call.CallDimensionC.unit>.
        * `eclabel` (strong, optional, default=None): label for `ec`.
            See <autofig.call.Mesh.ec> and <autofig.call.CallDimensionC.label>.
        * `iunit` (string or astropy unit, optional, default=None): units for `i`.
            See <autofig.call.Mesh.i> and <autofig.call.CallDimensionI.unit>.
        * `itol` (float, optional, default=0.0): see <autofig.call.DimensionI.tol>.
        * `axorder` (int, optional, default=None): see <autofig.call.Mesh.axorder>.
        * `axpos` (tuple, optional, default=None): see <autofig.call.Mesh.axpos>.
        * `title` (string, optional, default=None): see <autofig.call.Mesh.title>.
        * `label` (string, optional, default=None): see <autofig.call.Mesh.label>.
        * `linestyle` or `ls` (string, optional, default='solid'): see
            <autofig.call.Mesh.linestyle>. Note: `linestyle` takes precedence
            over `ls` if both are provided.  So technically `ls` defaults
            to 'solid' and `linestyle` defaults to None.
        * `consider_for_limits` (bool, optional, default=True): see
            <autofig.call.Call.consider_for_limits>.
        * `exclude_back` (bool, optional, default=False): whether to exclude
            any elements pointing away from the screen.  This will be ignored
            for 3d projections or if `znormals` is not provided.  Setting this
            to True can save significant time in drawing the mesh in matplotlib,
            and is especially useful for closed surfaces if `fc` is not 'none'.
        * `**kwargs`: additional keyword arguments are stored and passed on when
            attaching to a parent axes.  See <autofig.axes.Axes.add_call>.

        Returns
        ---------
        * the instantiated <autofig.call.Mesh> object.
        """
        self._axes_fc = None
        self._axes_ec = None

        facecolor = kwargs.pop('facecolor', None)
        fc = facecolor if facecolor is not None else fc
        self._fc = CallDimensionC(self, fc, None, fcunit, fclabel, cmap=fcmap)

        edgecolor = kwargs.pop('edgecolor', None)
        ec = edgecolor if edgecolor is not None else ec
        self._ec = CallDimensionC(self, ec, None, ecunit, eclabel, cmap=ecmap)

        ls = kwargs.pop('ls', 'solid')
        self.linestyle = linestyle if linestyle is not None else ls

        self.linebreak = False

        self.exclude_back = exclude_back

        if hasattr(i, '__iter__') and not isinstance(i, u.Quantity):
            raise ValueError("i as an iterable not supported for Meshes, make separate calls for each value of i")

        super(Mesh, self).__init__(i=i, iunit=iunit, itol=itol,
                                   x=x, xerror=xerror, xunit=xunit, xlabel=xlabel, xnormals=xnormals,
                                   y=y, yerror=yerror, yunit=yunit, ylabel=ylabel, ynormals=ynormals,
                                   z=z, zerror=zerror, zunit=zunit, zlabel=zlabel, znormals=znormals,
                                   consider_for_limits=consider_for_limits,
                                   uncover=uncover, trail=trail,
                                   axorder=axorder, axpos=axpos,
                                   title=title, label=label,
                                   **kwargs
                                   )

    def __repr__(self):
        dirs = []
        for direction in ['i', 'x', 'y', 'z', 'fc', 'ec']:
            if getattr(self, direction).value is not None:
                dirs.append(direction)

        return "<Call:Mesh | dims: {}>".format(", ".join(dirs))


    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return {'classname': self.__class__.__name__,
                'x': self.x.to_dict(),
                'y': self.y.to_dict(),
                'z': self.z.to_dict(),
                'fc': self.fc.to_dict(),
                'ec': self.ec.to_dict(),
                'i': self.i.to_dict(),
                'axorder': self._axorder,
                'axpos': self._axpos,
                'title': self._title,
                'label': self._label,
                'uncover': self._uncover,
                'trail': self._trail,
                'consider_for_limits': self._consider_for_limits,
                'exclude_back': self._exclude_back}

    @property
    def axes_fc(self):
        # currently no setter as this really should be handle by axes.add_call
        return self._axes_fc

    @property
    def axes_ec(self):
        # currently no setter as this really should be handle by axes.add_call
        return self._axes_ec

    @property
    def c(self):
        """
        Returns
        ---------
        * <autofig.call.CallDimensionCGroup> of <autofig.call.Mesh.fc> and
            <autofig.call.Mesh.ec>
        """
        return CallDimensionCGroup([self.fc, self.ec])

    @property
    def fc(self):
        """
        See also:

        * <autofig.call.Mesh.get_facecolor>

        Returns
        ----------
        * <autofig.call.CallDimensionC>
        """
        return self._fc

    def get_facecolor(self, colorcycler=None):
        """
        See also:

        * <autofig.call.Mesh.fc>

        Arguments
        -----------
        * `colorcycler` (optional, default=None): **IGNORED** (only included
            to have a similar calling signature as other methods that do
            account for color cyclers)

        Returns
        ----------
        * (string): 'none' if <autofig.call.Mesh.fc> is not a string.
        """
        if isinstance(self.fc.value, str):
            color = self.fc.value
        else:
            # then we'll default to 'none'.  If we want to color by
            # the dimension, we should call self.c directly
            color = 'none'

        # we won't use the colorcycler for facecolor

        return color

    @property
    def facecolor(self):
        """
        Shortcut to <autofig.call.Mesh.get_facecolor>.

        See also:

        * <autofig.call.Mesh.fc>

        Returns
        ----------
        * (string)
        """
        return self.get_facecolor()

    @facecolor.setter
    def facecolor(self, facecolor):
        # TODO: type and cycler checks
        facecolor = common.coloralias.map(_map_none(facecolor))
        if self.axes is not None:
            self.axes._colorcycler.replace_used(self.get_facecolor(), facecolor)
        self._fc.value = facecolor

    def get_fcmap(self, cmapcycler=None):
        if isinstance(self.fc.value, str):
            return None
        if self.fc.value is None:
            return None

        cmap = self.fc.cmap
        if cmap is None and cmapcycler is not None:
            cmap = cmapcycler.next_tmp

        return cmap

    @property
    def ec(self):
        """
        See also:

        * <autofig.call.Mesh.get_edgecolor>

        Returns
        ----------
        * <autofig.call.CallDimensionC>
        """
        return self._ec

    def get_edgecolor(self, colorcycler=None):
        """
        See also:

        * <autofig.call.Mesh.ec>

        Arguments
        -----------
        * `colorcycler` (optional, default=None): **IGNORED** (only included
            to have a similar calling signature as other methods that do
            account for color cyclers)

        Returns
        ----------
        * (string): 'black' if <autofig.call.Mesh.ec> is not a string.
        """
        if isinstance(self.ec.value, str):
            color = self.ec.value
        else:
            # then we'll default to black.  If we want to color by
            # the dimension, we should call self.c directly
            color = 'black'

        # we won't use the colorcycler for edgecolor

        return color

    @property
    def edgecolor(self):
        """
        Shortcut to <autofig.call.Mesh.get_edgecolor>.

        See also:

        * <autofig.call.Mesh.ec>

        Returns
        ----------
        * (string)
        """
        return self.get_edgecolor()

    @edgecolor.setter
    def edgecolor(self, edgecolor):
        # TODO: type and cycler checks
        if edgecolor in ['face']:
            self._ec.value = edgecolor
            return

        edgecolor = common.coloralias.map(_map_none(edgecolor))
        if self.axes is not None:
            self.axes._colorcycler.replace_used(self.get_edgecolor(), edgecolor)
        self._ec.value = edgecolor

    def get_ecmap(self, cmapcycler=None):
        if isinstance(self.ec.value, str):
            return None
        if self.ec.value is None:
            return None

        cmap = self.ec.cmap
        if cmap is None and cmapcycler is not None:
            cmap = cmapcycler.next_tmp

        return cmap

    @property
    def exclude_back(self):
        return self._exclude_back

    @exclude_back.setter
    def exclude_back(self, exclude_back):
        if not isinstance(exclude_back, bool):
            raise TypeError("exclude back must be of type bool")

        self._exclude_back = exclude_back

    def draw(self, ax=None, i=None,
             colorcycler=None, markercycler=None, linestylecycler=None):
        """

        See also:

        * <autofig.draw>
        * <autofig.figure.Figure.draw>
        * <autofig.axes.Axes.draw>
        * <autofig.call.Plot.draw>

        Arguments
        ----------
        * `ax`
        * `i`
        * `colorcycler`
        * `markercycler`
        * `linestylecycler`
        """
        # Mesh.draw
        if ax is None:
            ax = plt.gca()
        else:
            if not isinstance(ax, plt.Axes):
                raise TypeError("ax must be of type plt.Axes")

        if not (i is None or isinstance(i, float) or isinstance(i, int) or isinstance(i, u.Quantity)):
            raise TypeError("i must be of type float/int/None")

        # determine 2D or 3D
        axes_3d = isinstance(ax, Axes3D)

        kwargs = self.kwargs.copy()

        # PLOTTING
        return_artists = []
        x = self.x.get_value(i=i, sort_by_indep=False, exclude_back=self.exclude_back, unit=self.axes.x.unit)
        y = self.y.get_value(i=i, sort_by_indep=False, exclude_back=self.exclude_back, unit=self.axes.y.unit)
        z = self.z.get_value(i=i, sort_by_indep=False, exclude_back=self.exclude_back, unit=self.axes.z.unit)
        fc = self.fc.get_value(i=i, sort_by_indep=False, exclude_back=self.exclude_back, unit=self.axes_fc.unit if self.axes_fc is not None else None)
        ec = self.ec.get_value(i=i, sort_by_indep=False, exclude_back=self.exclude_back, unit=self.axes_ec.unit if self.axes_ec is not None else None)

        # DETERMINE PER-DATAPOINT Z-ORDERS
        zorders, do_zorder = self.axes.z.get_zorders(z, i=i)

        if do_zorder:
            # we can perhaps skip doing the zorder loop if there are no other
            # calls within the axes
            if len(self.axes.calls) == 1:
                do_zorder = False
                zorders = np.mean(zorders)

        if axes_3d:
            if x is not None and y is not None and z is not None:
                polygons = np.concatenate((x[:,:,np.newaxis], y[:,:,np.newaxis], z[:,:,np.newaxis]), axis=2)
            else:
                # there isn't anything to plot here, the current i probably
                # filtered this call out
                return []

            pccall = Poly3DCollection
        else:
            if x is not None and y is not None:
                polygons = np.concatenate((x[:,:,np.newaxis], y[:,:,np.newaxis]), axis=2)

                if not do_zorder and z is not None:
                    # then we'll handle zorder within this Mesh call by
                    # sorting instead of looping.  This is MUCH quicking
                    # and less memory instensive
                    sortinds = np.mean(z, axis=1).argsort()
                    polygons = polygons[sortinds, :, :]
                    if isinstance(fc, np.ndarray):
                        fc = fc[sortinds]
                    if isinstance(ec, np.ndarray):
                        ec = ec[sortinds]
            else:
                # there isn't anything to plot here, the current i probably
                # filtered this call out
                return []

            pccall = PolyCollection


        do_facecolorscale = fc is not None and not isinstance(fc, str)
        do_edgecolorscale = ec is not None and not isinstance(ec, str)

        if do_edgecolorscale:
            if self.axes_ec is None:
                raise NotImplementedError("currently only support edgecolor once attached to axes")
            else:
                edgenorm = self.axes_ec.get_norm(i=i)
                edgecmap = self.axes_ec.cmap
                edgecolors = plt.get_cmap(edgecmap)(edgenorm(ec))
        else:
            edgecolors = self.get_edgecolor(colorcycler=colorcycler)

        if do_facecolorscale:
            if self.axes_fc is None:
                raise NotImplementedError("currently only support facecolor once attached to axes")
            else:
                facenorm = self.axes_fc.get_norm(i=i)
                facecmap = self.axes_fc.cmap
                facecolors = plt.get_cmap(facecmap)(facenorm(fc))

        else:
            facecolors = self.get_facecolor(colorcycler=colorcycler)

        if do_zorder:
            # LOOP THROUGH POLYGONS so each can be assigned its own zorder
            if isinstance(edgecolors, str):
                edgecolors = [edgecolors] * len(zorders)
            if isinstance(facecolors, str):
                facecolors = [facecolors] * len(zorders)

            for loop, (polygon, zorder, edgecolor, facecolor) in enumerate(zip(polygons, zorders, edgecolors, facecolors)):
                pc = pccall((polygon,),
                            linestyle=self.linestyle,
                            edgecolors=edgecolor,
                            facecolors=facecolor,
                            zorder=zorder,
                            label=self.label if loop==0 else None)
                ax.add_collection(pc)

                return_artists += [pc]

        else:
            # DON'T LOOP as all have the same zorder, this should be faster
            pc = pccall(polygons,
                        linestyle=self.linestyle,
                        edgecolors=edgecolors,
                        facecolors=facecolors,
                        zorder=zorders,
                        label=self.label)

            ax.add_collection(pc)

            return_artists += [pc]

        self._backend_objects = return_artists

        for artist in return_artists:
            callbacks._connect_to_autofig(self, artist)

        return return_artists


class CallDimensionGroup(common.Group):
    def __init__(self, items):
        super(CallDimensionGroup, self).__init__(CallDimension, [], items)

    @property
    def value(self):
        """
        Returns
        ---------
        * (list) a list of  <autofig.call.CallDimension.value> for each child
            <autofig.call.CallDimension>
        """
        return np.array([c.value for c in self._items]).flatten()

    @property
    def units(self):
        """
        """
        return [c.unit for c in self._items]

    @property
    def unit(self):
        units = list(set(self.units))
        if len(units) > 1:
            raise ValueError("more than 1 units, see units")
        else:
            return units[0]

    @property
    def labels(self):
        """
        """
        return [c.label for c in self._items]

    @property
    def label(self):
        labels = list(set(self.labels))
        if len(labels) > 1:
            raise ValueError("more than 1 labels, see labels")
        else:
            return labels[0]

class CallDimensionCGroup(CallDimensionGroup):
    @property
    def cmap(self):
        """
        Returns
        ---------
        * (list) a list of  <autofig.call.CallDimensionC.cmap> for each child
            <autofig.call.CallDimensionC>
        """
        return self._get_attrs('cmap')

    @cmap.setter
    def cmap(self, smap):
        return self._set_attrs('cmap', cmap)

class CallDimensionSGroup(CallDimensionGroup):
    @property
    def smap(self):
        """
        Returns
        ---------
        * (list) a list of  <autofig.call.CallDimensionS.smap> for each child
            <autofig.call.CallDimensionS>
        """
        return self._get_attrs('smap')

    @smap.setter
    def smap(self, smap):
        return self._set_attrs('smap', smap)

    @property
    def mode(self):
        """
        Returns
        ---------
        * (list) a list of  <autofig.call.CallDimensionS.mode> for each child
            <autofig.call.CallDimensionS>
        """
        return self._get_attrs('mode')

    @mode.setter
    def mode(self, mode):
        return self._set_attrs('mode', mode)

def make_calldimensiongroup(items):
    if np.all([isinstance(item, CallDimensionC) for item in items]):
        return CallDimensionCGroup(items)
    elif np.all([isinstance(item, CallDimensionS) for item in items]):
        return CallDimensionSGroup(items)
    else:
        return CallDimensionGroup(items)

class CallDimension(object):
    def __init__(self, direction, call, value, error=None, unit=None, label=None, normals=None):
        if isinstance(value, dict):
            error = value.get('error', error)
            unit = value.get('unit', unit)
            label = value.get('label', label)
            normals = value.get('normals', normals)
            value = value.get('value')

        self._call = call
        self.direction = direction
        # unit must be set before value as setting value pulls the appropriate
        # unit for CallDimensionI
        self.unit = unit
        self.value = value
        self.error = error
        self.label = label
        self.normals = normals
        # self.lim = lim

    def __repr__(self):


        if isinstance(self.value, np.ndarray):
            info = "len: {}".format(len(self.value))
        else:
            info = "value: {}".format(self.value)


        return "<{} | {} | type: {} | label: {}>".format(self.direction,
                                       info,
                                       self.unit.physical_type,
                                       self.label)

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return {'direction': self.direction,
                'unit': self.unit.to_string(),
                'value': common.arraytolistrecursive(self._value),
                'error': common.arraytolistrecursive(self._error),
                'label': self._label,
                'normals': common.arraytolistrecursive(self._normals)}

    @property
    def call(self):
        """
        Returns
        ---------
        * <autofig.call.Call> (<autofig.call.Plot> or <autofig.call.Mesh>): the
            parent call object.
        """
        return self._call

    @property
    def direction(self):
        """
        Returns
        -------------
        * (str) one of 'i', 'x', 'y', 'z', 's', 'c'
        """
        return self._direction

    @direction.setter
    def direction(self, direction):
        """
        set the direction
        """
        if not isinstance(direction, str):
            raise TypeError("direction must be of type str")

        accepted_values = ['i', 'x', 'y', 'z', 's', 'c']
        if direction not in accepted_values:
            raise ValueError("must be one of: {}".format(accepted_values))

        self._direction = direction

    def _to_unit(self, value, unit=None):
        if isinstance(value, str):
            return value
        if value is None:
            return value

        if unit is not None and unit!=u.dimensionless_unscaled:
            unit = common._convert_unit(unit)
            value = value*self.unit.to(unit)

        return value

    def interpolate_at_i(self, i, unit=None):
        """
        Access the interpolated value at a given value of `i` (independent-variable).

        Arguments
        -----------
        `i`
        `unit` (unit or string, optional, default=None)

        Returns
        -------------
        * (float): the interpolated value

        Raises
        ------------
        * ValueError: if there is a lenght mismatch
        """
        if isinstance(self.call.i._value, float):
            if self.call.i._value==i:
                return self._to_unit(self._value, unit)
            else:
                return None



        # we can't call i._value here because that may point to a string, and
        # we want this to resolve the array
        i_value = self.call.i.get_value(linebreak=False, sort_by_indep=False)
        if len(i_value) != len(self._value):
            raise ValueError("length mismatch with independent-variable")

        sort_inds = i_value.argsort()
        indep_value = i_value[sort_inds]
        this_value = self._value[sort_inds]
        if len(self._value.shape) > 1:
            return np.asarray([self._to_unit(np.interp(i, indep_value, this_value_col, left=np.nan, right=np.nan), unit) for this_value_col in this_value.T]).T

        return self._to_unit(np.interp(i, indep_value, this_value, left=np.nan, right=np.nan), unit)

    def highlight_at_i(self, i, unit=None):
        """
        """
        if len(self._value.shape)==1 and isinstance(self.call.i.value, np.ndarray):
            return self.interpolate_at_i(i, unit=unit)
        else:
            return self._to_unit(self._value[self._filter_at_i(i,
                                                               uncover=True,
                                                               trail=0)].T,
                                 unit)

    def _do_linebreak(self, func='get_value', i=None, unit=None,
                      uncover=None, trail=None, linebreak=None,
                      sort_by_indep=None):
        """
        """
        if linebreak is None:
            linebreak = self.linebreak

        this_array = getattr(self, func)(i=i,
                                    unit=unit,
                                    uncover=uncover,
                                    trail=trail,
                                    linebreak=False)

        if linebreak is False:
            return this_array

        break_direction = linebreak[0]
        # NOTE: we don't need the unit here since we just use it to find
        # breakpoints
        break_array = getattr(self.call, break_direction).get_value(i=i,
                                                                    unit=None,
                                                                    uncover=uncover,
                                                                    trail=trail,
                                                                    linebreak=False,
                                                                    sort_by_indep=sort_by_indep)

        if linebreak[1] == '+':
            split_inds = np.where(break_array[1:]-break_array[:-1]>0)[0]
        elif linebreak[1] == '-':
            split_inds = np.where(break_array[1:]-break_array[:-1]<0)[0]
        else:
            raise NotImplementedError("linebreak='{}' not supported".format(linebreak))

        return np.split(this_array, split_inds+1)


    def _sort_by_indep(self, func='get_value', i=None, iunit=None, unit=None,
                       uncover=None, trail=None, linebreak=None,
                       sort_by_indep=None):

        """
        must be called before (or within) _do_linebreak
        """

        if sort_by_indep is None:
            # TODO: add property of the call?
            sort_by_indep = True

        indep_array = self.call.i.get_value(i=i,
                                            unit=iunit,
                                            uncover=uncover,
                                            trail=trail,
                                            linebreak=False,
                                            sort_by_indep=False)

        this_array = getattr(self, func)(i=i,
                                         unit=unit,
                                         uncover=uncover,
                                         trail=trail,
                                         linebreak=False,
                                         sort_by_indep=False)

        if not (isinstance(indep_array, np.ndarray) and len(indep_array)==len(this_array)):
            sort_by_indep = False

        if sort_by_indep:
            # TODO: it might be nice to buffer this at the call level, so making
            # multiple get_value calls doesn't have to recompute the sort-order
            sort_inds = indep_array.argsort()
            return this_array[sort_inds]
        else:
            return this_array


    def _get_trail_min(self, i, trail=None):
        trail = self.call.trail if trail is None else trail

        # determine length of the trail (if applicable)
        if trail is not False:
            if trail is True:
                # then fallback on 10% default
                trail_perc = 0.1
            else:
                trail_perc = float(trail)

            if trail_perc == 0.0:
                trail_i = i
            else:
                all_i = np.hstack(self.call.axes.calls.i.value)
                trail_i = i - trail_perc*(np.nanmax(all_i) - np.nanmin(all_i))
                if trail_i < np.nanmin(self.call.i.get_value(linebreak=False, sort_by_indep=False)):
                    # don't allow extraploating below the lower range
                    trail_i = np.nanmin(self.call.i.get_value(linebreak=False, sort_by_indep=False))

        else:
            trail_i = None

        return trail_i


    def _filter_at_i(self, i, uncover=None, trail=None):
        uncover = self.call.uncover if uncover is None else uncover
        trail = self.call.trail if trail is None else trail

        # we can't call i._value here because that may point to a string, and
        # we want this to resolve the array
        i_value = self.call.i.get_value(linebreak=False, sort_by_indep=False)

        if isinstance(i_value, np.ndarray):
            trues = np.ones(i_value.shape, dtype=bool)
        else:
            trues = True

        if trail is not False:
            trail_i = self._get_trail_min(i=i, trail=trail)

            left_filter = i_value >= trail_i - self.call.i.tol

        else:
            left_filter = trues


        if uncover is not False:
            right_filter = i_value <= i + self.call.i.tol

        else:
            right_filter = trues

        return (left_filter & right_filter)

    def get_value(self, i=None, unit=None,
                  uncover=None, trail=None,
                  linebreak=None, sort_by_indep=None,
                  exclude_back=False,
                  attr='_value'):
        """
        Access the value for a given value of `i` (independent-variable) depending
        on which effects (i.e. uncover) are enabled.

        If `uncover`, `trail`, or `linebreak` are None (default), then the value from
        the parent <autofig.call.Call> from <autofig.call.CallDimension.call>
        (probably (<autofig.call.Plot>) will be used.  See <autofig.call.Plot.uncover>,
        <autofig.call.Plot.trail>, <autofig.call.Plot.linebreak>.

        Arguments
        -----------
        * `i`
        * `unit`
        * `uncover`
        * `trail`
        * `linebreak`
        * `sort_by_indep`
        * `exclude_back`
        * `attr`

        Returns
        ----------
        * (array or None)
        """

        value = getattr(self, attr)  # could be self._value or self._error
        if value is None:
            return None

        if uncover is None:
            uncover = self.call.uncover

        if trail is None:
            trail = self.call.trail

        if linebreak is None:
            linebreak = self.call.linebreak

        if sort_by_indep is None:
            # TODO: make this a property of the call?
            sort_by_indep = True

        if isinstance(value, str) or isinstance(value, float):
            if i is None:
                return self._to_unit(value, unit)
            elif isinstance(self.call.i.value, float):
                # then we still want to "select" based on the value of i
                if self._filter_at_i(i):
                    return value
                else:
                    return None
            else:
                # then we should show either way.  For example - a color or
                # axhline even with i given won't change in i
                return self._to_unit(value, unit)

        if isinstance(value, list) or isinstance(value, tuple):
            value = np.asarray(value)

        # from here on we're assuming the value is an array, so let's just check
        # to be sure
        if not isinstance(value, np.ndarray):
            raise NotImplementedError("value/error must be a numpy array")

        if exclude_back and self.call.z.normals is not None and self.call.axes.projection == '2d':
            value = value[self.call.z.normals >= 0]

        if linebreak is not False:
            return self._do_linebreak(func='get{}'.format(attr),
                                      i=i,
                                      unit=unit,
                                      uncover=uncover,
                                      trail=trail,
                                      linebreak=linebreak,
                                      sort_by_indep=sort_by_indep)

        if sort_by_indep is not False:
            # if we've made it here, linebreak should already be False (if
            # linebreak was True, then we'd be within _do_linebreak and those
            # get_value calls pass linebreak=False)
            return self._sort_by_indep(func='get{}'.format(attr),
                                       i=i,
                                       unit=unit,
                                       uncover=uncover,
                                       trail=trail,
                                       linebreak=False,
                                       sort_by_indep=sort_by_indep)

        # from here on, linebreak==False and sort_by_indep==False (if either
        # were True, then we're within those functions and asking for the original
        # array)
        if i is None:
            if len(value.shape)==1:
                return self._to_unit(value, unit)
            else:
                if isinstance(self.call, Plot):
                    return self._to_unit(value.T, unit)
                else:
                    return self._to_unit(value, unit)

        # filter the data as necessary
        filter_ = self._filter_at_i(i, uncover=uncover, trail=trail)

        if isinstance(self.call.i.value, float):
            if filter_:
                return self._to_unit(value, unit)
            else:
                return None

        if len(value.shape)==1 or isinstance(self.call, FillBetween):
            # then we're dealing with a flat 1D array
            if attr == '_value':
                if trail is not False:
                    trail_i = self._get_trail_min(i)
                    first_point = self.interpolate_at_i(trail_i)


                if uncover:
                    last_point = self.interpolate_at_i(i)
            else:
                first_point = np.nan
                last_point = np.nan

            if uncover and trail is not False:
                concat = (np.array([first_point]),
                          value[filter_],
                          np.array([last_point]))
            elif uncover:
                concat = (value[filter_],
                          np.array([last_point]))

            elif trail:
                concat = (np.array([first_point]),
                          value[filter_])
            else:
                return self._to_unit(value[filter_], unit)

            return self._to_unit(np.concatenate(concat), unit)

        else:
            # then we need to "select" based on the indep and the value
            if isinstance(self.call, Plot):
                return self._to_unit(value[filter_].T, unit)
            else:
                return self._to_unit(value[filter_], unit)


    # for value we need to define the property without decorators because of
    # this: https://stackoverflow.com/questions/13595607/using-super-in-a-propertys-setter-method-when-using-the-property-decorator-r
    # and the need to override these in the CallDimensionI class
    def _get_value(self):
        """
        access the value
        """
        return self.get_value(i=None, unit=None)

    def _set_value(self, value):
        """
        set the value
        """
        if value is None:
            self._value = value
            return

        # handle casting to acceptable types
        if isinstance(value, list) or isinstance(value, tuple):
            value = np.asarray(value)
        elif isinstance(value, int):
            value = float(value)

        if isinstance(value, u.Quantity):
            if self.unit == u.dimensionless_unscaled:
                # then take the unit from quantity and apply it
                self.unit = value.unit
                value = value.value
            else:
                # then convert to the requested unit
                value = value.to(self.unit).value

        # handle setting based on type
        if isinstance(value, np.ndarray):
            # if len(value.shape) != 1:
                # raise ValueError("value must be a flat array")

            self._value = value
        elif isinstance(value, float):
            # TODO: do we want to cast to np.array([value])??
            # this will most likely be used for axhline/axvline
            self._value = value
        elif self.direction=='c' and isinstance(value, str):
            self._value = common.coloralias.map(value)
        else:
            raise TypeError("value must be of type array (or similar), found {} {}".format(type(value), value))

    value = property(_get_value, _set_value)

    def get_error(self, i=None, unit=None,
                  uncover=None, trail=None,
                  linebreak=None, sort_by_indep=None):
        """
        access the error for a given value of i (independent-variable) depending
        on which effects (i.e. uncover) are enabled.
        """
        return self.get_value(i=i, unit=unit,
                              uncover=uncover, trail=trail,
                              linebreak=linebreak, sort_by_indep=sort_by_indep,
                              attr='_error')


    @property
    def error(self):
        """
        access the error
        """
        return self._error

    @error.setter
    def error(self, error):
        """
        set the error
        """
        # TODO: check length with value?
        # TODO: type checks (similar to value)
        if self.direction not in ['x', 'y', 'z'] and error is not None:
            raise ValueError("error only accepted for x, y, z dimensions")

        if isinstance(error, u.Quantity):
            error = error.to(self.unit).value

        if isinstance(error, list) or isinstance(error, tuple):
            error = np.asarray(error)

        self._error = error

    @property
    def unit(self):
        """
        access the unit
        """
        return self._unit

    @unit.setter
    def unit(self, unit):
        """
        set the unit
        """
        unit = common._convert_unit(unit)
        self._unit = unit

    @property
    def label(self):
        """
        access the label
        """
        return self._label

    @label.setter
    def label(self, label):
        """
        set the label
        """
        if self.direction in ['i'] and label is not None:
            raise ValueError("label not accepted for indep dimension")


        if label is None:
            self._label = label
            return

        if not isinstance(label, str):
            try:
                label = str(label)
            except:
                raise TypeError("label must be of type str")

        self._label = label

    @property
    def normals(self):
        """
        access the normals
        """
        return self._normals

    @normals.setter
    def normals(self, normals):
        """
        set the normals
        """
        if self.direction not in ['x', 'y', 'z'] and normals is not None:
            raise ValueError("normals only accepted for x, y, z dimensions")

        if normals is None:
            self._normals = None
            return

        if not (isinstance(normals, list) or isinstance(normals, np.ndarray)):
            raise TypeError("normals must be of type list or array")

        self._normals = normals


class CallDimensionI(CallDimension):
    def __init__(self, call, value, unit, tol):
        if isinstance(value, dict):
            tol = value.get('tol', tol)

        self.tol = tol
        super(CallDimensionI, self).__init__('i', call, value, unit)

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return {'direction': self.direction,
                'unit': self.unit.to_string(),
                'value': common.arraytolistrecursive(self._value),
                'tol': self._tol}

    @property
    def tol(self):
        """
        Returns
        -----------
        * (float) tolerance to use when selecting/uncover/trail
        """
        if self._tol is None:
            return 0.0

        return self._tol

    @tol.setter
    def tol(self, tol):
        if not isinstance(tol, float):
            raise TypeError("tol must be of type float")

        # TODO: handle units?
        self._tol = tol

    @property
    def value(self):
        """
        access the value
        """
        if isinstance(self._value, str):
            dimension = self._value
            return getattr(self.call, dimension).value

        return super(CallDimensionI, self)._get_value()

    @value.setter
    def value(self, value):
        """
        set the value
        """
        # for the indep direction we also allow a string which points to one
        # of the other available dimensions
        # TODO: support c, fc, ec?
        if isinstance(value, common.basestring) and value in ['x', 'y', 'z']:
            # we'll cast just to get rid of any python2 unicodes
            self._value  = str(value)
            dimension = value
            self._unit = getattr(self.call, dimension).unit
            return

        # NOTE: cannot do super on setter directly, see this python
        # bug: https://bugs.python.org/issue14965 and discussion:
        # https://mail.python.org/pipermail/python-dev/2010-April/099672.html
        super(CallDimensionI, self)._set_value(value)

    def get_value(self, *args, **kwargs):
        if isinstance(self._value, str):
            dimension = self._value
            return getattr(self.call, dimension).get_value(*args, **kwargs)

        return super(CallDimensionI, self).get_value(*args, **kwargs)

    @property
    def is_reference(self):
        """
        whether referencing another dimension or its own
        """
        return isinstance(self._value, str)

    @property
    def reference(self):
        """
        reference (will return None if not is_reference)
        """
        if self.is_reference:
            return self._value

        else:
            return None

class CallDimensionX(CallDimension):
    def __init__(self, *args):
        super(CallDimensionX, self).__init__('x', *args)

class CallDimensionY(CallDimension):
    def __init__(self, *args):
        super(CallDimensionY, self).__init__('y', *args)

class CallDimensionZ(CallDimension):
    def __init__(self, *args):
        super(CallDimensionZ, self).__init__('z', *args)

class CallDimensionS(CallDimension):
    def __init__(self, call, value, error=None, unit=None, label=None,
                 smap=None, mode=None):

        if isinstance(value, dict):
            error = value.get('error', error)
            smap = value.get('smap', smap)
            mode = value.get('mode', mode)

        if error is not None:
            raise ValueError("error not supported for 's' dimension")

        self.smap = smap
        self.mode = mode

        super(CallDimensionS, self).__init__('s', call, value, error, unit,
                                             label)

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return {'direction': self.direction,
                'unit': self.unit.to_string(),
                'value': common.arraytolistrecursive(self._value),
                'error': common.arraytolistrecursive(self._error),
                'label': self._label,
                'smap': self._smap,
                'mode': self._mode}

    @property
    def smap(self):
        return self._smap

    @smap.setter
    def smap(self, smap):
        if smap is None:
            self._smap = smap
            return

        if not isinstance(smap, tuple):
            try:
                smap = tuple(smap)
            except:
                raise TypeError('smap must be of type tuple')

        if not len(smap)==2:
            raise ValueError('smap must have length 2')

        self._smap = smap

    def _mode_split(self, mode=None):
        if mode is None:
            mode = self.mode

        split = mode.split(':')
        mode_dims = split[0]
        mode_obj = split[1] if len(split) > 1 else 'axes'
        mode_mode = split[2] if len(split) > 2 else 'fixed'

        return mode_dims, mode_obj, mode_mode


    @property
    def mode(self):
        if self._mode is None:
            return 'xy:figure:fixed'

        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode is None:
            self._mode = None
            return

        if not isinstance(mode, str):
            raise TypeError("mode must be of type str")

        split = mode.split(':')
        mode_dims, mode_obj, mode_mode = self._mode_split(mode)

        if len(split) > 3:
            raise ValueError("mode not recognized")

        if mode_dims == 'pt' and len(split) > 1:
            raise ValueError("mode not recognized")


        if mode_dims not in ['x', 'y', 'xy', 'pt']:
            raise ValueError("mode not recognized")

        if mode_obj not in ['axes', 'figure']:
            raise ValueError("mode not recognized")

        if mode_mode not in ['fixed', 'current', 'original']:
            raise ValueError("mode not recognized")

        if mode_dims == 'pt':
            self._mode = mode
        else:
            self._mode = '{}:{}:{}'.format(mode_dims, mode_obj, mode_mode)

class CallDimensionC(CallDimension):
    def __init__(self, call, value, error=None, unit=None, label=None, cmap=None):
        if isinstance(value, dict):
            error = value.get('error', error)
            cmap = value.get('cmap', cmap)

        if error is not None:
            raise ValueError("error not supported for 'c' dimension")

        self.cmap = cmap
        super(CallDimensionC, self).__init__('c', call, value, error, unit,
                                             label)

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return {'direction': self.direction,
                'unit': self.unit.to_string(),
                'value': common.arraytolistrecursive(self._value),
                'error': common.arraytolistrecursive(self._error),
                'label': self._label,
                'cmap': self._cmap}

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        # print("setting call cmap: {}".format(cmap))
        try:
            cmap_ = plt.get_cmap(cmap)
        except:
            raise TypeError("could not find cmap")

        self._cmap = cmap
