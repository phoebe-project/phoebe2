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

class CallGroup(common.Group):
    def __init__(self, items):
        super(CallGroup, self).__init__(Call, [], items)

    @property
    def callbacks(self):
        return self._get_attrs('callbacks')

    def connect_callback(self, callback):
        for call in self._items:
            call.connect_callback(callback)

    @property
    def i(self):
        return CallDimensionGroup(self._get_attrs('i'))

    @property
    def x(self):
        return CallDimensionGroup(self._get_attrs('x'))

    @property
    def y(self):
        return CallDimensionGroup(self._get_attrs('y'))

    @property
    def z(self):
        return CallDimensionGroup(self._get_attrs('z'))

    @property
    def consider_for_limits(self):
        return self._get_attrs('consider_for_limits')

    @consider_for_limits.setter
    def consider_for_limits(self, consider_for_limits):
        return self._set_attrs('consider_for_limits', consider_for_limits)

    def draw(self, *args, **kwargs):
        """
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
        return CallDimensionSGroup(self._get_attrs('s'))

    @property
    def c(self):
        return CallDimensionCGroup(self._get_attrs('c'))

    @property
    def size_scale(self):
        return self._get_attrs('size_scale')

    @size_scale.setter
    def size_scale(self, size_scale):
        return self._set_attrs('size_scale', size_scale)

class MeshGroup(CallGroup):
    @property
    def fc(self):
        return CallDimensionCGroup(self._get_attrs('fc'))

    @property
    def ec(self):
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
                 xerror=None, xunit=None, xlabel=None,
                 yerror=None, yunit=None, ylabel=None,
                 zerror=None, zunit=None, zlabel=None,
                 iunit=None,
                 consider_for_limits=True,
                 uncover=False,
                 trail=False,
                 **kwargs):
        """
        """
        self._class = 'Call' # just to avoid circular import in order to use isinstance

        self._axes = None
        self._backend_objects = []
        self._callbacks = []

        self._x = CallDimensionX(self, x, xerror, xunit, xlabel)
        self._y = CallDimensionY(self, y, yerror, yunit, ylabel)
        self._z = CallDimensionZ(self, z, zerror, zunit, zlabel)

        # defined last so all other dimensions are in place in case indep
        # is a reference and needs to access units, etc
        self._i = CallDimensionI(self, i, iunit)

        self.consider_for_limits = consider_for_limits
        self.uncover = uncover
        self.trail = trail

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
        # no setter as this can only be set internally when attaching to an axes
        return self._axes

    @property
    def figure(self):
        # no setter as this can only be set internally when attaching to an axes
        if self.axes is None:
            return None
        return self.axes.figure

    @property
    def i(self):
        return self._i

    @property
    def indep(self):
        return self.i

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def consider_for_limits(self):
        return self._consider_for_limits

    @consider_for_limits.setter
    def consider_for_limits(self, consider):
        if not isinstance(consider, bool):
            raise TypeError("consider_for_limits must be of type bool")

        self._consider_for_limits = consider

    @property
    def uncover(self):
        return self._uncover

    @uncover.setter
    def uncover(self, uncover):
        if not isinstance(uncover, bool):
            raise TypeError("uncover must be of type bool")

        self._uncover = uncover

    @property
    def trail(self):
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


class Plot(Call):
    def __init__(self, x=None, y=None, z=None, c=None, s=None, i=None,
                       xerror=None, xunit=None, xlabel=None,
                       yerror=None, yunit=None, ylabel=None,
                       zerror=None, zunit=None, zlabel=None,
                       cunit=None, clabel=None, cmap=None,
                       sunit=None, slabel=None, smap=None, smode=None,
                       iunit=None,
                       marker=None, linestyle=None, linewidth=None,
                       highlight=True, uncover=False, trail=False,
                       consider_for_limits=True,
                       **kwargs):
        """
        marker
        size (takes precedence over s)
        color (takes precedence over c)

        highlight_marker
        highlight_size / highlight_s
        highlight_color /highlight_c
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

        super(Plot, self).__init__(i=i, iunit=iunit,
                                   x=x, xerror=xerror, xunit=xunit, xlabel=xlabel,
                                   y=y, yerror=yerror, yunit=yunit, ylabel=ylabel,
                                   z=z, zerror=zerror, zunit=zunit, zlabel=zlabel,
                                   consider_for_limits=consider_for_limits,
                                   uncover=uncover, trail=trail,
                                   **kwargs
                                   )

        self.connect_callback(callbacks.update_sizes)

    def __repr__(self):
        dirs = []
        for direction in ['i', 'x', 'y', 'z', 's', 'c']:
            if getattr(self, direction).value is not None:
                dirs.append(direction)

        return "<Call:Plot | dims: {}>".format(", ".join(dirs))

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
        if self._highlight_color is None:
            return self.get_color()

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

    def draw(self, ax=None, i=None,
             colorcycler=None, markercycler=None, linestylecycler=None):
        """
        """
        # Plot.draw
        if ax is None:
            ax = plt.gca()
        else:
            if not isinstance(ax, plt.Axes):
                raise TypeError("ax must be of type plt.Axes")

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
        c = self.c.get_value(i=i, unit=self.axes_c.unit if self.axes_c is not None else None)
        s = self.s.get_value(i=i, unit=self.axes_s.unit if self.axes_s is not None else None)

        if axes_3d:
            zerr = self.z.get_error(i=i, unit=self.axes.z.unit)

            data = np.array([x, y, z])
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        else:
            zerr = None

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
        def error_kwargs_loop(loop, do_zorder):
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
            sc_kwargs_const['c'] = color


        def sc_kwargs_loop(sc_kwargs, loop, do_zorder):
            if do_colorscale:
                if do_zorder:
                    sc_kwargs['c'] = c[loop]
                else:
                    sc_kwargs['c'] = c
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

            for loop, (datapoint, segment, zorder) in enumerate(zip(datas, segments, zorders)):
                return_artists_this_loop = []
                # DRAW ERRORBARS, if applicable
                if xerr is not None or yerr is not None or zerr is not None:
                    artists = ax.errorbar(*datapoint,
                                           fmt='', linestyle='None',
                                           zorder=zorder,
                                           **error_kwargs_loop(loop, do_zorder))

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

                if do_colorscale or do_sizescale or do_zorder:
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

                        lc = lccall(segments,
                                    zorder=zorder,
                                    **lc_kwargs_loop(lc_kwargs_const, loop, do_zorder))

                        if do_colorscale:
                            if do_zorder:
                                lc.set_array(np.array([c[loop]]))
                            else:
                                lc.set_array(c)


                        return_artists_this_loop.append(lc)
                        ax.add_collection(lc)


                    # DRAW SCATTER, if applicable
                    if marker.lower() != 'none':
                        artist = ax.scatter(*datapoint,
                                            zorder=zorder,
                                            **sc_kwargs_loop(sc_kwargs_const, loop, do_zorder))

                        return_artists_this_loop.append(artist)


                else:
                    # let's use plot whenever possible... it'll be faster
                    # and will guarantee that the linestyle looks correct
                    artists = ax.plot(*datapoint,
                                      marker=marker,
                                      ls=ls,
                                      mec='none',
                                      color=color)

                    return_artists_this_loop += artists

                size_this_loop = sizes_loop(loop, do_zorder)
                for artist in return_artists_this_loop:
                    # store the sizes so they can be rescaled appropriately by
                    # the callback
                    artist._af_sizes = size_this_loop

                return_artists += return_artists_this_loop



        # DRAW IF X OR Y ARE NOT ARRAYS
        if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            # TODO: can we do anything in 3D?
            if x is not None:
                artist = ax.axvline(x, ls=ls, color=color)
                return_artists += [artist]

            if y is not None:
                artist = ax.axhline(y, ls=ls, color=color)
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
                                                   color=self.highlight_color)

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
                              ls='None', color=self.highlight_color)

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


class Mesh(Call):
    def __init__(self, x=None, y=None, z=None, fc=None, ec=None, i=None,
                       xerror=None, xunit=None, xlabel=None,
                       yerror=None, yunit=None, ylabel=None,
                       zerror=None, zunit=None, zlabel=None,
                       fcunit=None, fclabel=None, fcmap=None,
                       ecunit=None, eclabel=None, ecmap=None,
                       iunit=None,
                       consider_for_limits=True,
                       uncover=True,
                       trail=0,
                       **kwargs):
        """
        """

        self._axes_fc = None
        self._axes_ec = None

        facecolor = kwargs.pop('facecolor', None)
        fc = facecolor if facecolor is not None else fc
        self._fc = CallDimensionC(self, fc, None, fcunit, fclabel, cmap=fcmap)

        edgecolor = kwargs.pop('edgecolor', None)
        ec = edgecolor if edgecolor is not None else ec
        self._ec = CallDimensionC(self, ec, None, ecunit, eclabel, cmap=ecmap)

        if hasattr(i, '__iter__'):
            raise ValueError("i as an iterable not supported for Meshes, make separate calls for each value of i")

        super(Mesh, self).__init__(i=i, iunit=iunit,
                                   x=x, xerror=xerror, xunit=xunit, xlabel=xlabel,
                                   y=y, yerror=yerror, yunit=yunit, ylabel=ylabel,
                                   z=z, zerror=zerror, zunit=zunit, zlabel=zlabel,
                                   consider_for_limits=consider_for_limits,
                                   uncover=uncover, trail=trail,
                                   **kwargs
                                   )

    def __repr__(self):
        dirs = []
        for direction in ['i', 'x', 'y', 'z', 'fc', 'ec']:
            if getattr(self, direction).value is not None:
                dirs.append(direction)

        return "<Call:Mesh | dims: {}>".format(", ".join(dirs))

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
        return CallDimensionGroup([self.fc, self.ec])

    @property
    def fc(self):
        return self._fc

    def get_facecolor(self, colorcycler=None):
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
        return self._ec

    def get_edgecolor(self, colorcycler=None):
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
        return self.get_edgecolor()

    @edgecolor.setter
    def edgecolor(self, edgecolor):
        # TODO: type and cycler checks
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

    def draw(self, ax=None, i=None,
             colorcycler=None, markercycler=None, linestylecycler=None):
        """
        """
        # Mesh.draw
        if ax is None:
            ax = plt.gca()
        else:
            if not isinstance(ax, plt.Axes):
                raise TypeError("ax must be of type plt.Axes")

        # determine 2D or 3D
        axes_3d = isinstance(ax, Axes3D)

        kwargs = self.kwargs.copy()

        # PLOTTING
        return_artists = []
        # TODO: handle getting in correct units (possibly passed from axes?)
        x = self.x.get_value(i=i, unit=self.axes.x.unit)
        y = self.y.get_value(i=i, unit=self.axes.y.unit)
        z = self.z.get_value(i=i, unit=self.axes.z.unit)
        fc = self.fc.get_value(i=i, unit=self.axes_fc.unit if self.axes_fc is not None else None)
        ec = self.ec.get_value(i=i, unit=self.axes_ec.unit if self.axes_ec is not None else None)

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

                z = self.z.get_value(i=i)
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

            for polygon, zorder, edgecolor, facecolor in zip(polygons, zorders, edgecolors, facecolors):
                pc = pccall((polygon,),
                            edgecolors=edgecolor,
                            facecolors=facecolor,
                            zorder=zorder)
                ax.add_collection(pc)

                return_artists += [pc]

        else:
            # DON'T LOOP as all have the same zorder, this should be faster
            pc = pccall(polygons,
                        edgecolors=edgecolors,
                        facecolors=facecolors,
                        zorder=zorders)

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
        return np.array([c.value for c in self._items]).flatten()

class CallDimensionCGroup(CallDimensionGroup):
    @property
    def cmap(self):
        return self._get_attrs('cmap')

    @cmap.setter
    def cmap(self, smap):
        return self._set_attrs('cmap', cmap)

class CallDimensionSGroup(CallDimensionGroup):
    @property
    def smap(self):
        return self._get_attrs('smap')

    @smap.setter
    def smap(self, smap):
        return self._set_attrs('smap', smap)

    @property
    def mode(self):
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
    def __init__(self, direction, call, value, error=None, unit=None, label=None):
        self._call = call
        self.direction = direction
        # unit must be set before value as setting value pulls the appropriate
        # unit for CallDimensionI
        self.unit = unit
        self.value = value
        self.error = error
        self.label = label
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

    @property
    def call(self):
        return self._call

    @property
    def direction(self):
        """
        access the direction
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

        if unit is not None:
            unit = common._convert_unit(unit)
            value = value*self.unit.to(unit)

        return value

    def interpolate_at_i(self, i, unit=None):
        """
        access the interpolated value at a give value of i (independent-variable)
        """
        if isinstance(self.call.i.value, float):
            if self.call.i.value==i:
                return self.value
            else:
                return None

        if len(self.call.i.value) != len(self._value):
            raise ValueError("length mismatch with independent-variable")

        return self._to_unit(np.interp(i, self.call.i.value, self._value), unit)

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

    def _get_trail_min(self, i, trail=None):
        trail = self.call.trail if trail is None else trail

        # determine length of the trail (if applicable)
        if trail is not False:
            if trail is True:
                # then fallback on 10% default
                trail_perc = 0.1
            else:
                trail_perc = float(trail)

            all_i = np.hstack(self.call.axes.calls.i.value)
            trail_i = i - trail_perc*(np.nanmax(all_i) - np.nanmin(all_i))
            if trail_i < np.nanmin(self.call.i.value):
                # don't allow extraploating below the lower range
                trail_i = np.nanmin(self.call.i.value)

        else:
            trail_i = None

        return trail_i


    def _filter_at_i(self, i, uncover=None, trail=None):
        uncover = self.call.uncover if uncover is None else uncover
        trail = self.call.trail if trail is None else trail

        if isinstance(self.call.i.value, np.ndarray):
            trues = np.ones(self.call.i.value.shape, dtype=bool)
        else:
            trues = True

        if trail is not False:
            trail_i = self._get_trail_min(i=i, trail=trail)

            left_filter = self.call.i.value >= trail_i

        else:
            left_filter = trues


        if uncover is not False:
            right_filter = self.call.i.value <= i

        else:
            right_filter = trues

        return (left_filter & right_filter)

    def get_value(self, i=None, unit=None):
        """
        access the value for a given value of i (independent-variable) depending
        on which effects (i.e. uncover) are enabled.
        """
        if self._value is None:
            return None

        if isinstance(self._value, str) or isinstance(self._value, float):
            if i is None:
                return self._to_unit(self._value, unit)
            elif isinstance(self.call.i.value, float):
                # then we still want to "select" based on the value of i
                if self._filter_at_i(i):
                    return self._value
                else:
                    return None
            else:
                # then we should show either way.  For example - a color or
                # axhline even with i given won't change in i
                return self._to_unit(self._value, unit)

        # from here on we're assuming the value is an array, so let's just check
        # to be sure
        if not isinstance(self._value, np.ndarray):
            raise NotImplementedError


        if i is None:
            if len(self._value.shape)==1:
                return self._to_unit(self._value, unit)
            else:
                if isinstance(self.call, Plot):
                    return self._to_unit(self._value.T, unit)
                else:
                    return self._to_unit(self._value, unit)

        # filter the data as necessary
        filter_ = self._filter_at_i(i)

        if isinstance(self.call.i.value, float):
            if filter_:
                return self._to_unit(self._value, unit)
            else:
                return None

        if len(self._value.shape)==1:
            # then we're dealing with a flat 1D array
            if self.call.trail is not False:
                trail_i = self._get_trail_min(i)
                first_point = self.interpolate_at_i(trail_i)
            else:
                first_point = np.nan

            if self.call.uncover:
                last_point = self.interpolate_at_i(i)
            else:
                last_point = np.nan

            return self._to_unit(np.concatenate((np.array([first_point]),
                                                 self._value[filter_],
                                                 np.array([last_point]))),
                                 unit)

        else:
            # then we need to "select" based on the indep and the value
            if isinstance(self.call, Plot):
                return self._to_unit(self._value[filter_].T, unit)
            else:
                return self._to_unit(self._value[filter_], unit)


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
            value = np.array(value)
        if isinstance(value, int):
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
        # elif isinstance(value, str):
            # TODO: then need to pull from the bundle??? Or will this happen
            # at a higher level
        elif self.direction=='c' and isinstance(value, str):
            self._value = common.coloralias.map(value)
        else:
            raise TypeError("value must be of type array (or similar)")

    value = property(_get_value, _set_value)

    def get_error(self, i=None, unit=None):
        """
        access the error for a given value of i (independent-variable) depending
        on which effects (i.e. uncover) are enabled.
        """
        if i is None:
            return self._to_unit(self._error, unit)

        if self._error is None:
            return None


        # filter the data as necessary
        filter_ = self._filter_at_i(i)

        if isinstance(self.call.i.value, float):
            if filter_:
                return self._to_unit(self._error, unit)
            else:
                return None


        if len(self._error.shape)==1:
            # then we're dealing with a flat 1D array
            first_point = np.nan
            last_point = np.nan

            return self._to_unit(np.concatenate((np.array([first_point]),
                                                 self._error[filter_],
                                                 np.array([last_point]))),
                                 unit)

        else:
            # then we need to "select" based on the indep and the value
            if isinstance(self.call, Plot):
                return self._to_unit(self._error[filter_].T, unit)
            else:
                return self._to_unit(self._error[filter_], unit)


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


class CallDimensionI(CallDimension):
    def __init__(self, *args):
        super(CallDimensionI, self).__init__('i', *args)

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
        if error is not None:
            raise ValueError("error not supported for 's' dimension")

        self.smap = smap
        self.mode = mode

        super(CallDimensionS, self).__init__('s', call, value, error, unit,
                                             label)

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
        if error is not None:
            raise ValueError("error not supported for 'c' dimension")

        self.cmap = cmap
        super(CallDimensionC, self).__init__('c', call, value, error, unit,
                                             label)

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
