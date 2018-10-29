import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib import colorbar as mplcolorbar

from . import common
from . import callbacks
from . import cyclers
from . import call as _call

def _consistent_allow_none(thing1, thing2):
    if thing1 is None or thing2 is None:
        return True
    else:
        return thing1 == thing2

class AxesGroup(common.Group):
    def __init__(self, items):
        super(AxesGroup, self).__init__(Axes, [], items)

    @property
    def i(self):
        return AxDimensionGroup(self._get_attrs('i'))

    @property
    def x(self):
        return AxDimensionGroup(self._get_attrs('x'))

    @property
    def y(self):
        return AxDimensionGroup(self._get_attrs('y'))

    @property
    def z(self):
        return AxDimensionGroup(self._get_attrs('z'))

    @property
    def ss(self):
        return AxDimensionSGroup(self._get_attrs('ss'))

    @property
    def cs(self):
        return AxDimensionCGroup(self._get_attrs('cs'))

    @property
    def equal_aspect(self):
        return self._get_attrs('equal_aspect')

    @equal_aspect.setter
    def equal_aspect(self):
        return self._set_attrs('equal_aspect', equal_aspect)

    @property
    def pad_aspect(self):
        return self._get_attrs('pad_aspect')

    @pad_aspect.setter
    def pad_aspect(self, pad_aspect):
        return self._set_attrs('pad_aspect', pad_aspect)

    @property
    def projection(self):
        return self._get_attrs('projection')

    @projection.setter
    def projection(self, projection):
        return self._set_attrs('projection', projection)

    @property
    def elev(self):
        return AxViewGroup(self._get_attrs('elev'))

    @property
    def azim(self):
        return AxViewGroup(self._get_attrs('azim'))


class Axes(object):
    def __init__(self, *calls, **kwargs):
        self._class = 'Axes' # just to avoid circular import in order to use isinstance

        self._figure = None
        self.projection = kwargs.pop('projection', None)
        self.legend = kwargs.pop('legend', False)
        self.legend_kwargs = kwargs.pop('legend_kwargs', {})

        self._backend_object = None
        self._backend_artists = []

        self._colorcycler = cyclers.MPLColorCycler()
        self._cmapcycler = cyclers.MPLCmapCycler()
        self._markercycler = cyclers.MPLMarkerCycler()
        self._linestylecycler = cyclers.MPLLinestyleCycler()

        self._calls = []

        self.title = kwargs.pop('title', None)
        self.axorder = kwargs.pop('axorder', None)
        self.axpos = kwargs.pop('axpos', None)

        self._i = AxDimensionI(self, **kwargs)
        self._x = AxDimensionX(self, **kwargs)
        self._y = AxDimensionY(self, **kwargs)
        self._z = AxDimensionZ(self, **kwargs)

        self._elev = AxViewElev(self, **kwargs)
        self._azim = AxViewAzim(self, **kwargs)

        # set default padding
        self.xyz.pad = 0.1

        self._ss = []
        self._cs = []

        self.equal_aspect = None
        self.pad_aspect = None

        self.add_call(*calls)

    def __repr__(self):
        dirs = []
        for direction in common.dimensions:
            if direction in ['c', 's']:
                if len(getattr(self, '{}s'.format(direction))):
                    dirs.append(direction)
            elif getattr(self, direction).lim != (None, None):
                dirs.append(direction)

        ncalls = len(self.calls)
        return "<Axes | {} call(s) | dims: {}>".format(ncalls, ", ".join(dirs))

    @property
    def figure(self):
        # no setter as this can only be set internally when attaching to a figure
        return self._figure

    @property
    def calls(self):
        return _call.make_callgroup(self._calls)

    @property
    def calls_sorted(self):
        """
        calls sorted in z
        """
        def _z(call):
            if isinstance(call.z.value, np.ndarray):
                return np.mean(call.z.value.flatten())
            elif isinstance(call.z.value, float) or isinstance(call.z.value, int):
                return call.z.value
            else:
                # put it at the back
                return -np.inf

        calls = self._calls
        zs = np.array([_z(c) for c in calls])
        sorted_inds = zs.argsort()
        # TODO: ugh, this is ugly.  Test to find the optimal way to sort
        # while still ending up with a list
        return _call.make_callgroup(np.array(calls)[sorted_inds].tolist())

    @property
    def colorcycler(self):
        return self._colorcycler

    @property
    def markercycler(self):
        return self._markercycler

    @property
    def linestylecycler(self):
        return self._linestylecycler

    @property
    def cmapcycler(self):
        return self._cmapcycler

    @property
    def projection(self):
        if self._projection is None:
            return '2d'

        return self._projection

    @projection.setter
    def projection(self, projection):
        if projection not in [None, '3d', '2d']:
            raise ValueError("projection must be None or '3d'")

        if projection == '2d':
            projection = None

        self._projection = projection

    @property
    def legend(self):
        return self._legend

    @legend.setter
    def legend(self, legend):
        if not isinstance(legend, bool):
            raise TypeError("legend must be of type bool (send kwargs to legend_kwargs)")

        self._legend = legend

    @property
    def legend_kwargs(self):
        return self._legend_kwargs

    @legend_kwargs.setter
    def legend_kwargs(self, legend_kwargs):
        if not isinstance(legend_kwargs, dict):
            raise TypeError("legend_kwargs must by of type dict")

        self._legend_kwargs = legend_kwargs

    @property
    def axorder(self):
        if self._axorder is None:
            if self._figure is not None:
                axorders = [ax._axorder for ax in self._figure._axes if ax._axorder is not None]
                if len(axorders):
                    return max(axorders)+1
                else:
                    return 0
            else:
                return 0

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
        return self._axpos

    @axpos.setter
    def axpos(self, axpos):
        if axpos is None:
            self._axpos = axpos

            return

        if isinstance(axpos, tuple) and len(axpos) == 3 and np.all(isinstance(ap, int) for ap in axpos):
            self._axpos = axpos

        elif isinstance(axpos, int) and axpos >= 100 and axpos < 1000:
            self._axpos = (int(axpos/100), int(axpos/10 % 10), int(axpos % 10))

        else:
            raise ValueError("axpos must be of type int or tuple between 100 and 999")

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        if title is None:
            self._title = None
            return

        if not isinstance(title, str):
            raise TypeError("title must be of type str or None")

        self._title = title

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
    def xy(self):
        return AxDimensionGroup([self.x, self.y])

    @property
    def xyz(self):
        return AxDimensionGroup([self.x, self.y, self.z])

    @property
    def ss(self):
        return AxDimensionGroup(self._ss)

    @property
    def sizes(self):
        return self.ss

    @property
    def cs(self):
        return AxDimensionGroup(self._cs)

    @property
    def colors(self):
        return self.cs

    @property
    def elev(self):
        if self.projection != '3d':
            raise ValueError("elev only applicable for 3D projection")

        return self._elev

    @property
    def azim(self):
        if self.projection != '3d':
            raise ValueError("azim only applicable for 3D projection")

        return self._azim

    @property
    def equal_aspect(self):
        if self._equal_aspect is None:
            # TODO: logic for 3D
            if self.x.unit.physical_type == self.y.unit.physical_type:
                if self.x.unit.physical_type == 'dimensionless':
                    return False
                else:
                    return True
            else:
                return False

        return self._equal_aspect

    @equal_aspect.setter
    def equal_aspect(self, equal_aspect):
        if equal_aspect is None:
            self._equal_aspect = None
            return

        if not isinstance(equal_aspect, bool):
            raise TypeError("equal_aspect must be of type bool")

        self._equal_aspect = equal_aspect

    @property
    def pad_aspect(self):
        if self._pad_aspect is None:
            if self.equal_aspect:
                if self.x._lim in [None, (None, None)] or isinstance(self.x._lim, str):
                    return True
                else:
                    return False

            return False

        return self._pad_aspect

    @pad_aspect.setter
    def pad_aspect(self, pad_aspect):
        if pad_aspect is None:
            self._pad_aspect = None
            return

        if not isinstance(pad_aspect, bool):
            raise TypeError("pad_aspect must be of type bool")

        self._pad_aspect = pad_aspect

    def consistent_with_call(self, call):
        """
        check to see if a new call would be consistent to add to this Axes instance

        checks include:
        * compatible units in all directions
        * compatible independent-variable (if applicable)
        """
        if len(self.calls) == 0:
            return True, ''

        msg = []

        if not _consistent_allow_none(call._axorder, self._axorder):
            msg.append('inconsistent axorder, {} != {}'.format(call.axorder, self.axorder))

        if not _consistent_allow_none(call._axpos, self._axpos):
            msg.append('inconsistent axpos, {} != {}'.format(call.axpos, self.axpos))

        if call._axorder == self._axorder and call._axorder is not None:
            # then despite other conflicts, attempt to put on same axes
            return True, ''

        if call._axpos == self._axpos and call._axpos is not None:
            # then despite other conflicts, attempt to put on same axes
            return True, ''

        # TODO: include s, c, fc, ec, etc and make these checks into loops
        if call.x.unit.physical_type != self.x.unit.physical_type:
            msg.append('inconsitent xunit, {} != {}'.format(call.x.unit, self.x.unit))
        if call.y.unit.physical_type != self.y.unit.physical_type:
            msg.append('inconsitent yunit, {} != {}'.format(call.y.unit, self.y.unit))
        if call.z.unit.physical_type != self.z.unit.physical_type:
            msg.append('inconsitent zunit, {} != {}'.format(call.z.unit, self.z.unit))
        if call.i.unit.physical_type != self.i.unit.physical_type:
            msg.append('inconsistent iunit, {} != {}'.format(call.i.unit, self.i.unit))
        if call.i.is_reference or self.i.is_reference:
            if call.i.reference != self.i.reference:
                msg.append('inconsistent i reference, {} != {}'.format(call.i.reference, self.i.reference))

        if not _consistent_allow_none(call.title, self.title):
            msg.append('inconsistent axes title, {} != {}'.format(call.title, self.title))

        # here we send the protected _label so that we get None instead of empty string
        if not _consistent_allow_none(call.x._label, self.x._label):
            msg.append('inconsitent xlabel, {} != {}'.format(call.x.label, self.x.label))
        if not _consistent_allow_none(call.y._label, self.y._label):
            msg.append('inconsitent ylabel, {} != {}'.format(call.y.label, self.y.label))
        if not _consistent_allow_none(call.z._label, self.z._label):
            msg.append('inconsitent zlabel, {} != {}'.format(call.z.label, self.z.label))


        if len(msg):
            return False, ', '.join(msg)
        else:
            return True, ''

    def _match_color(self, call, call_c_attr):
        # handle axes-level colorscale(s)
        c_match = None
        call_c = getattr(call, call_c_attr)
        if call_c.value is not None and not isinstance(call_c.value, str):
            # now check to see whether we're consistent with any of the existing
            # colorscales - in reverse priority (i.e. the first most-recently
            # added match will be applied)
            used_cmaps = []
            for c in reversed(self.cs):
                if c.consistent_with_calldimension(call_c):
                    c_match = c
                    break
                used_cmaps.append(c.cmap)
            else:
                # then we haven't found any matches so we need to add a new
                # color dimension.  But first we want to make sure the cmap
                # isn't in use by an existing colordimension.
                if call_c.cmap is None:
                    # then add a new one from the cycler
                    call_c.cmap = self._cmapcycler.next_tmp

                if call_c.cmap in used_cmaps:
                    raise ValueError("cmap already in use in this axes, but could not attach to same colorscale")

                c_match = AxDimensionC(self, unit=call_c.unit,
                                       label=call_c.label,
                                       cmap=call_c.cmap)

                self._cs.append(c_match)

            # when the Call is in its draw method, it needs to know which
            # of the colorscales to obey.
            setattr(call, '_axes_{}'.format(call_c_attr), c_match)
            c_match._calls.append(call)
            c_match._calldimensions.append(call_c)

        return c_match

    def _match_size(self, call, call_s_attr):
        # handle axes-level sizescale(s)
        s_match = None
        call_s = getattr(call, call_s_attr)
        if call_s.value is not None and not (isinstance(call_s.value, float) or isinstance(call_s.value, int)):
            # now check to see whether we're consistent with any of the existing
            # sizescales - in reverse priority (i.e. the first most-recently
            # added match will be applied)
            for s in reversed(self.ss):
                if s.consistent_with_calldimension(call_s):
                    s_match = s
                    break
            else:
                # unlike colors, we don't really care if the cmap is in
                # use by an existing sizedimension
                s_match = AxDimensionS(self, unit=call_s.unit,
                                       label=call_s.label,
                                       smap=call_s.smap,
                                       mode=call_s.mode)

                self._ss.append(s_match)

            # when the Call is in its draw method, it needs to know which of
            # the sizescales to obey
            setattr(call, '_axes_{}'.format(call_s_attr), s_match)
            s_match._calls.append(call)
            s_match._calldimensions.append(call_s)

        return s_match

    def add_call(self, *calls):
        if len(calls) > 1:
            for c in calls:
                self.add_call(c)
            return

        elif len(calls) == 1:
            call = calls[0]
            if not isinstance(call, _call.Call):
                raise TypeError("call must be of type Call")

            consistent, reason = self.consistent_with_call(call)
            if not consistent:
                raise ValueError("call is not consistent with Axes: {}".format(reason))

            call._axes = self
            self._calls.append(call)

            if len(self.calls) == 1:
                # then this was the first, so set default units
                self.x.unit = call.x.unit
                self.y.unit = call.y.unit
                self.z.unit = call.z.unit
                self.i.unit = call.i.unit

                self.i.reference = call.i.reference

            if self._axorder is None:
                self.axorder = call.axorder

            if self._axpos is None:
                self.axpos = call.axpos

            # either way, fill in any missing labels - first set instance
            # will stick.  We check the protected underscored version to have
            # access to None instead of the empty string.
            if self.x._label is None:
                self.x.label = call.x._label
            if self.y._label is None:
                self.y.label = call.y._label
            if self.z._label is None:
                self.z.label = call.z._label

            # also set the title, setting the first instance
            if self.title is None:
                self.title = call.title

            # append the set props to the prop cycler.  Any prop that is None
            # will then request a temporary unused value from the prop cycler
            # at draw-time but will remain None in the object
            if isinstance(call, _call.Plot):
                self._colorcycler.add_to_used(call.get_color())
                self._linestylecycler.add_to_used(call.get_linestyle())
                self._markercycler.add_to_used(call.get_marker())
                self._cmapcycler.add_to_used(call.get_cmap())

                c_match = self._match_color(call, 'c')
                s_match = self._match_size(call, 's')

            elif isinstance(call, _call.Mesh):
                self._colorcycler.add_to_used(call.get_facecolor())
                self._colorcycler.add_to_used(call.get_edgecolor())

                self._linestylecycler.check_validity(call.linestyle)

                fc_match = self._match_color(call, 'fc')
                ec_match = self._match_color(call, 'ec')

            # lastly, especially if coming from a top-down call, let's try
            # to steal any remaining kwargs that may belong to the axes-level
            # (e.g. xylim)
            if 'equal_aspect' in call.kwargs.keys():
                self.equal_aspect = call.kwargs.pop('equal_aspect')
            if 'pad_aspect' in call.kwargs.keys():
                self.pad_aspect = call.kwargs.pop('pad_aspect')
            if 'projection' in call.kwargs.keys():
                self.projection = call.kwargs.pop('projection')
            if 'legend' in call.kwargs.keys():
                self.legend = call.kwargs.pop('legend')
            if 'legend_kwargs' in call.kwargs.keys():
                self.legend_kwargs = call.kwargs.pop('legend_kwargs')
            if 'elev' in call.kwargs.keys():
                self.elev.value = call.kwargs.pop('elev')
            if 'azim' in call.kwargs.keys():
                self.azim.value = call.kwargs.pop('azim')

            # now try attributes that belong to AxDimensions
            directions = ['xyz', 'xy', 'x', 'y', 'z', 'cs', 'ss', 'c', 's', 'ec', 'fc']
            for direction in directions:
                dkwargs = _process_dimension_kwargs(direction, call.kwargs)
                for k,v in dkwargs.items():
                    original_k = "{}{}".format(direction, k)

                    if direction=='c':
                        # then only apply to c_match
                        if c_match is None:
                            # I hate to raise an error here since stuff has already been done
                            raise ValueError("could not set {}, call still added".format(original_k))
                        setattr(c_match, k, v)
                    elif direction=='fc':
                        if fc_match is None:
                            # I hate to raise an error here since stuff has already been done
                            raise ValueError("could not set {}, call still added".format(original_k))
                        setattr(fc_match, k, v)
                    elif direction=='ec':
                        if ec_match is None:
                            # I hate to raise an error here since stuff has already been done
                            raise ValueError("could not set {}, call still added".format(original_k))
                        setattr(ec_match, k, v)
                    elif direction=='s':
                        # then only apply to s_match
                        if s_match is not None:
                            setattr(s_match, k, v)
                        # else:
                            # I hate to raise an error here since stuff has already been done
                            # this case could happen under normal circumstances for smode when CallDimensionS is a float instead of an array
                            # raise ValueError("could not set {}, call still added".format(original_k))
                    else:
                        setattr(getattr(self, direction), k, v)

                    # remove from the call.kwargs so it isn't passed on to MPL
                    del call.kwargs[original_k]

    def append_subplot(self, fig=None, subplot_grid=None):
        def determine_grid(N):
            cols = np.floor(np.sqrt(N))
            rows = np.ceil(float(N)/cols) if cols > 0 else 1
            return int(rows), int(cols)

        if fig is None:
            fig = plt.gcf()

        N = len(fig.axes)

        if subplot_grid is not None:
            # do type checks
            if not isinstance(subplot_grid, tuple):
                raise TypeError("subplot_grid must be tuple of length 2 (nrows [int], ncols [int])")
            if len(subplot_grid) != 2:
                raise ValueError("subplot_grid must be tuple of length 2 (nrows [int], ncols [int])")
            if not np.all([isinstance(s, int) for s in subplot_grid]):
                raise ValueError("subplot_grid must be tuple of length 2 (nrows [int], ncols [int])")

        # we'll reset the layout later anyways
        ax_new = fig.add_subplot(1,N+1,N+1, projection=self._projection)

        axes = fig.axes
        N = len(axes)

        ind = None
        if self.axpos is not None:
            rows, cols, ind = self.axpos
        elif subplot_grid is None:
            rows, cols = determine_grid(N)
        elif (isinstance(subplot_grid, list) or isinstance(subplot_grid, tuple)) and len(subplot_grid)==2:
            rows, cols = subplot_grid
        else:
            raise TypeError("subplot_grid must be None or tuple/list of length 2 (rows/cols)")

        if ind is None:
            for i,ax in enumerate(axes):
                try:
                    ax.change_geometry(rows, cols, i+1)
                except AttributeError:
                    # colorbars and sizebars won't be able to change geometry
                    pass
        else:
            ax_new.change_geometry(rows, cols, ind)

        ax = self._get_backend_object(ax_new)
        self._backend_artists = []

        return ax

    def _get_backend_object(self, ax=None):
        if ax is None:
            if self._backend_object:
                ax = self._backend_object
            else:
                ax = plt.gca()
        else:
            if not isinstance(ax, plt.Axes):
                raise TypeError("ax must be of type plt.Axes")

        self._backend_object = ax

        callbacks._connect_to_autofig(self, ax)
        return ax

    def _get_backend_artists(self):
        return self._backend_artists

    def draw_sidebars(self, ax=None, i=None):

        ax = self._get_backend_object(ax)

        for c in self.cs:
            # then make axes for the colorbar(s) to sit in
            cbax, cbkwargs = mplcolorbar.make_axes((ax,), location='right', fraction=0.15, shrink=1.0, aspect=20, panchor=False)
            callbacks._connect_to_autofig(self, cbax)

            cbartist = mplcolorbar.ColorbarBase(cbax, cmap=c.cmap, norm=c.get_norm(i=i), **cbkwargs)
            cbartist.set_label(c.label_with_units)

            callbacks._connect_to_autofig(c, cbartist)

        for s in self.ss:
            if s.mode in ['pt']:
                fraction = 0.15
            else:
                fraction = 1.1*abs(s.smap[1] - s.smap[0])
                if fraction < 0.05:
                    fraction = 0.05

            sbax, sbkwargs = mplcolorbar.make_axes((ax,), location='right', fraction=fraction, shrink=1.0)
            sbax.set_aspect(aspect='auto', adjustable='datalim')
            callbacks._connect_to_autofig(self, sbax)

            ys, sizes = s.get_sizebar_samples(i=i)
            # TODO: how to handle marker/color???
            sbax_done_markers = ['None']
            sbax_needs_line = False
            x = 0
            for n,call in enumerate(s.calls):
                # still not sure how to access the USED marker without
                # re-envoking the cycler...
                marker = call.get_marker()
                if marker not in sbax_done_markers:
                    x += 10
                    xs = [x]*len(ys)
                    sbax_done_markers.append(marker)
                    artist = sbax.scatter(xs, ys, s=sizes,
                                          marker=call.get_marker(),
                                          color='black',
                                          linewidths=0)

                    callbacks._connect_to_autofig(s, artist)
                    callbacks.update_sizes(artist, s, run_callback=True)

                linestyle = call.get_linestyle()
                if linestyle != 'None':
                    sbax_needs_line = True

            if sbax_needs_line:
                x += 10 # for counter for xlim
                # we'll sample linewidths at a higher rate
                # NOTE: if changing nsamples here, also need to change in
                # in callbacks.update_sizes.
                ys, sizes = s.get_sizebar_samples(nsamples=100, i=i)
                xs = [x]*len(ys)
                points = np.array([xs, ys]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, color='k', linewidth=sizes)
                sbax.add_collection(lc)

                callbacks._connect_to_autofig(s, lc)
                callbacks.update_sizes(lc, s, run_callback=True)


            sbax.yaxis.set_ticks_position('right')
            sbax.yaxis.set_label_position('right')
            sbax.set_xlim(0, x+10)
            sbax.set_xticks([])
            sbax.set_ylim(s.get_lim(i=i))
            sbax.set_ylabel(s.label_with_units)

    @property
    def plots(self):
        calls = [c for c in self._calls if isinstance(c, _call.Plot)]
        return _call.PlotGroup(calls)

    @property
    def meshes(self):
        calls = [c for c in self._calls if isinstance(c, _call.Mesh)]
        return _call.MeshGroup(calls)

    def draw(self, ax=None, i=None, calls=None,
             draw_sidebars=True,
             draw_title=True,
             show=False, save=False,
             in_animation=False):

        ax = self._get_backend_object(ax)

        # handle aspect ratio
        if self.equal_aspect:
            aspect = 'equal'
            if self.pad_aspect:
                if in_animation:
                    print("WARNING: pad_aspect not supported for animations, ignoring")
                    adjustable = 'box'
                else:
                    adjustable = 'datalim'
            else:
                adjustable = 'box'

        else:
            aspect = 'auto'
            adjustable = 'box'

        ax.set_aspect(aspect=aspect, adjustable=adjustable)

        # return_calls = []
        self._colorcycler.clear_tmp()
        self._linestylecycler.clear_tmp()
        self._markercycler.clear_tmp()
        for call in self.calls_sorted:
            if calls is None or call in calls:
                artists = call.draw(ax=ax, i=i,
                                    colorcycler=self._colorcycler,
                                    markercycler=self._markercycler,
                                    linestylecycler=self._linestylecycler)
                # return_calls.append(call)
                self._backend_artists += artists

        if draw_sidebars:
            self.draw_sidebars(ax=ax, i=i)

        if draw_title and self.title is not None:
            ax.set_title(self.title)

        axes_3d = isinstance(ax, Axes3D)

        ax.set_xlabel(self.x.label_with_units)
        ax.set_ylabel(self.y.label_with_units)
        if axes_3d:
            ax.set_zlabel(self.z.label_with_units)

        xlim = self.x.get_lim(i=i)
        if not np.any(np.isnan(xlim)):
            ax.set_xlim(xlim)
        ylim = self.y.get_lim(i=i)
        if not np.any(np.isnan(ylim)):
            ax.set_ylim(ylim)

        if axes_3d:
            zlim = self.z.get_lim(i=i)
            if not np.any(np.isnan(zlim)):
                ax.set_zlim(zlim)

            elev_current = self.elev.get_value(i=i)
            azim_current = self.azim.get_value(i=i)
            ax.view_init(elev_current, azim_current)

        if self.legend:
            plt.legend(**self.legend_kwargs)

        if show:
            plt.show()

        if save:
            plt.savefig(save)

        # return return_calls


class AxDimensionGroup(common.Group):
    def __init__(self, items):
        super(AxDimensionGroup, self).__init__(AxDimension, ['direction', 'label'], items)

    @property
    def direction(self):
        return self._get_attrs('direction')

    @property
    def unit(self):
        return self._get_attrs('unit')

    @unit.setter
    def unit(self, unit):
        return self._set_attrs('unit', unit)

    @property
    def pad(self):
        return self._get_attrs('pad')

    @pad.setter
    def pad(self, pad):
        return self._set_attrs('pad', pad)

    @property
    def lim(self):
        return self._get_attrs('lim')

    @lim.setter
    def lim(self, lim):
        return self._set_attrs('lim', lim)

    @property
    def label(self):
        return self._get_attrs('label')

    @label.setter
    def label(self, label):
        return self._set_attrs('label', label)

class AxDimensionCGroup(AxDimensionGroup):
    @property
    def cmap(self):
        return self._get_attrs('cmap')

    @cmap.setter
    def cmap(self, smap):
        return self._set_attrs('cmap', cmap)

class AxDimensionSGroup(AxDimensionGroup):
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

class AxArray(object):
    def __init__(self, direction, axes):
        # just to avoid circular import in order to use isinstance
        self._class = self.__class__.__name__

        self._axes = axes

        if not isinstance(direction, str):
            raise TypeError("direction must be of type str")

        accepted_values = ['i', 'x', 'y', 'z', 's', 'c', 'fc', 'ec', 'elev', 'azim']
        if direction not in accepted_values:
            raise ValueError("must be one of: {}".format(accepted_values))

        self._direction = direction

    def __repr__(self):

        return "<{}>".format(self.direction)

    @property
    def axes(self):
        return self._axes

    @property
    def direction(self):
        return self._direction

class AxDimension(AxArray):
    def __init__(self, direction, axes, unit=None, pad=None, lim=[None, None], label=None):
        self.unit = unit
        self.pad = pad
        self.lim = lim
        self.label = label

        super(AxDimension, self).__init__(direction, axes)

    def __repr__(self):

        return "<{} | limits: {} | type: {} | label: {}>".format(self.direction,
                                                                 self.lim,
                                                                 self.unit.physical_type,
                                                                 self.label)

    @property
    def unit(self):
        return self._unit

    @property
    def unit_string(self):
        return self.unit.to_string()

    @property
    def unit_latex(self):
        return self.unit._repr_latex_()

    @unit.setter
    def unit(self, unit):
        unit = common._convert_unit(unit)
        self._unit = unit

    @property
    def pad(self):
        return self._pad

    @pad.setter
    def pad(self, pad):
        # TODO type checking
        self._pad = pad

    def get_lim(self, pad=None, i=None):

        def _central_values(indep):
            central_values = []
            for call in self.axes.calls:
                if not call.consider_for_limits:
                    continue

                try:
                    interp_in_direction = getattr(call, self.direction).interpolate_at_i(indep, self.unit)
                except ValueError:
                    pass
                else:
                    if interp_in_direction is None:
                        continue
                    elif isinstance(call, _call.Mesh):
                        # then interp_in_direction should be [polygon, vertex]
                        interp_in_direction_flat = interp_in_direction.flatten()
                        central_values.append(np.nanmin(interp_in_direction_flat))
                        central_values.append(np.nanmax(interp_in_direction_flat))
                    else:
                        central_values.append(interp_in_direction)

            return central_values

        if pad is None:
            pad = self.pad

        if self.direction == 'i':
            # then this doesn't really make sense
            return (None, None)

        lim_orig = self._lim

        if isinstance(lim_orig, tuple):
            # we'll need to edit the entries, so cast to list
            lim = list(self._lim)
        else:
            lim = [None, None]

        fixed_min = lim[0] is not None
        fixed_max = lim[1] is not None

        if lim_orig == 'fixed':
            # then fixed with automatic limits, ignoring i
            kind = 'fixed'
        elif isinstance(lim_orig, tuple):
            # then fixed with set limits, we'll still get the array in
            # case fixed_min==False or fixed_max==False
            kind = 'fixed'
        elif lim_orig == 'symmetric':
            kind = 'fixed'
        elif lim_orig == 'frame':
            # then per-frame limits
            kind = 'frame'
        elif lim_orig == 'sliding':
            # then sliding with automatic range
            kind = 'sliding' if i is not None else 'fixed'
        elif lim_orig is None:
            kind = 'sliding' if i is not None else 'fixed'
        elif isinstance(lim_orig, float):
            # then sliding with fixed range
            # let's also disable padding
            if i is not None:
                fixed_min = True
                fixed_max = True
                kind = 'sliding'
            else:
                kind = 'fixed'

        else:
            raise NotImplementedError

        if kind == 'sliding':
            central_value = np.mean(_central_values(i))

            if lim_orig in [None, 'sliding']:
                # then automatically try to determine the range
                rang = 0

                # try to set based on the maximum spread of the central values
                # through all available indeps
                # TODO: please make the following line less hideous
                # TODO: its not really fair to pass self.unit on to call.i.get_value()... but will falling back on its default units cause issues?
                i_all = list(set(np.concatenate([common.tolist(call.i.get_value()) for call in self.axes.calls])))
                for i_this in i_all:
                    central_values = _central_values(i_this)

                    if not len(central_values):
                        continue

                    rang_at_indep = np.nanmax(central_values) - np.nanmin(central_values)
                    if rang_at_indep > rang:
                        rang = rang_at_indep

                if rang == 0:
                    # TODO: we should be able to predict this and avoid wasting time above
                    # if call.i.get_value()==self.direction for all calls in self.axes.calls?

                    # then fallback on 10% of the array(s)
                    for call in self.axes.calls:
                        array = getattr(call, self.direction).get_value(None, unit=self.unit).flatten()
                        rang_this_call = 0.1 * (np.nanmax(array) - np.nanmin(array))

                        if rang_this_call > rang:
                            rang = rang_this_call
                # else:
                    # raneg = rang * (1+pad)
                    # print "rang after padding", rang
            else:
                rang = float(lim_orig)

            # TODO: how will this handle flipped axes?
            lim = [central_value-rang/2, central_value+rang/2]


        elif kind in ['fixed', 'frame']:
            if hasattr(self, 'calldimensions'):
                # particularly color where we need to link to c, fc, or ec
                cds = self.calldimensions
            else:
                cds = [getattr(c, self.direction) for c in self.axes.calls]

            for cd in cds:
                call = cd.call
                if not call.consider_for_limits:
                    continue
                if not hasattr(call, self.direction):
                    continue

                if kind=='fixed':
                    error = cd.get_error(None, unit=self.unit, linebreak=False, sort_by_indep=False)
                    array = cd.get_value(None, unit=self.unit, linebreak=False, sort_by_indep=False)
                elif kind=='frame':
                    error = cd.get_error(i, unit=self.unit, linebreak=False, sort_by_indep=False)
                    array = cd.get_value(i, unit=self.unit, linebreak=False, sort_by_indep=False)
                else:
                    raise NotImplementedError

                if array is None:
                    # i.e. for axvline/axhline
                    continue

                array_flat = array.flatten() if isinstance(array, np.ndarray) else array

                if error is None:
                    error = np.zeros_like(array_flat)

                if not fixed_min and (lim[0] is None or np.nanmin(array_flat-error) < lim[0]):
                    lim[0] = np.nanmin(array_flat-error)
                if not fixed_max and (lim[1] is None or np.nanmax(array_flat+error) > lim[1]):
                    lim[1] = np.nanmax(array_flat+error)

        else:
            raise NotImplementedError

        if lim_orig == 'symmetric':
            limabs = np.nanmax(abs(np.array(lim)))
            # TODO: how will this work with inverting?
            lim = [-limabs, limabs]

        # now handle padding
        if pad is not None and lim != [None, None]:
            rang = abs(lim[1] - lim[0])
            if not fixed_min:
                lim[0] -= rang*pad
            if not fixed_max:
                lim[1] += rang*pad

        if np.nan in lim:
            return (None, None)

        return tuple(lim)

    @property
    def lim(self):
        return self._lim

    @lim.setter
    def lim(self, lim):
        """
        set lim (limits)
        """
        if lim is None:
            self._lim = lim
            return

        typeerror_msg = "lim must be of type tuple, float, None, or in ['fixed', 'symmetric', 'frame', 'sliding']"

        if isinstance(lim, str):
            if lim in ['fixed', 'symmetric', 'frame', 'sliding']:
                self._lim = lim
                return
            else:
                raise ValueError(typeerror_msg)

        if isinstance(lim, int):
            lim = float(lim)

        if isinstance(lim, float):
            if lim <= 0.0:
                raise ValueError("lim cannot be <= 0")
            self._lim = lim
            return

        if not isinstance(lim, tuple):
            try:
                lim = tuple(lim)
            except:
                raise TypeError(typeerror_msg)

        if not len(lim)==2:
            raise ValueError('lim must have length 2')

        for l in lim:
            if not (isinstance(l, float) or isinstance(l, int) or l is None):
                raise ValueError("each item in limit must be of type float, int, or None")

        self._lim = lim

    def get_norm(self, pad=None, i=None):
        return plt.Normalize(*self.get_lim(pad=pad, i=i))

    @property
    def norm(self):
        return self.get_norm(pad=self.pad)

    @property
    def label(self):
        return '' if self._label is None else self._label

    @property
    def label_with_units(self):
        if self.unit.physical_type != 'dimensionless':
            return r"{} [{}]".format(self.label, self.unit_latex)
        else:
            return r"{}".format(self.label)

    @label.setter
    def label(self, label):
        self._label = label


def _process_dimension_kwargs(direction, kwargs):
    """
    process kwargs for AxDimension instances by stripping off the prefix
    for the appropriate direction
    """
    acceptable_keys = ['unit', 'pad', 'lim', 'label']
    # if direction in ['s']:
        # acceptable_keys += ['mode']
    processed_kwargs = {}
    for k,v in kwargs.items():
        if k.startswith(direction):
            processed_key = k.lstrip(direction)
        else:
            processed_key = k

        if processed_key in acceptable_keys:
            processed_kwargs[processed_key] = v

    return processed_kwargs

class AxDimensionI(AxDimension):
    def __init__(self, *args, **kwargs):
        processed_kwargs = _process_dimension_kwargs('i', kwargs)
        self._reference = None
        super(AxDimensionI, self).__init__('i', *args, **processed_kwargs)

    @property
    def is_reference(self):
        return self.reference is not None

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, reference):
        if not isinstance(reference, str) and reference is not None:
            raise TypeError("reference must be of type str")

        self._reference = reference

class AxDimensionX(AxDimension):
    def __init__(self, *args, **kwargs):
        processed_kwargs = _process_dimension_kwargs('x', kwargs)
        super(AxDimensionX, self).__init__('x', *args, **processed_kwargs)

class AxDimensionY(AxDimension):
    def __init__(self, *args, **kwargs):
        processed_kwargs = _process_dimension_kwargs('y', kwargs)
        super(AxDimensionY, self).__init__('y', *args, **processed_kwargs)

class AxDimensionZ(AxDimension):
    def __init__(self, *args, **kwargs):
        processed_kwargs = _process_dimension_kwargs('z', kwargs)
        super(AxDimensionZ, self).__init__('z', *args, **processed_kwargs)

    def get_zorders(self, z, i=None):
        if z is None:
            zorders = -np.inf
            do_zorder = False
        elif isinstance(z, np.ndarray):
            # make a deepcopy here so when we exagerate later it doesn't
            # affect the original z
            znorm = self.get_norm(i=i)
            # map zorders from 0-1000 depending on zlim
            if len(z.shape)==1:
                zorders = znorm(z)*1e4
            else:
                zorders = znorm(np.mean(z, axis=1))*1e4
            do_zorder = True
        else:
            znorm = self.axes.z.get_norm(i=i)
            # map zorders from 0-1000 depending on zlim
            zorders = znorm(z)*1e4
            do_zorder = False

        return zorders, do_zorder

class AxDimensionScale(AxDimension):
    def __init__(self, direction, *args, **kwargs):
        self._calls = []
        self._calldimensions = []
        super(AxDimensionScale, self).__init__(direction, *args, **kwargs)

    @property
    def calls(self):
        return self._calls

    @property
    def calldimensions(self):
        return self._calldimensions

    def consistent_with_calldimension(self, calldimension):
        cd = calldimension
        if cd.direction != self.direction:
            raise TypeError("can only compare with another '{}' dimension".format(self.direction))

        if cd.unit.physical_type != self.unit.physical_type:
            return False

        if not _consistent_allow_none(cd._label, self._label):
            return False

        if self.direction=='c' and not _consistent_allow_none(cd.cmap, self.cmap):
            return False

        if self.direction=='s' and not _consistent_allow_none(cd.smap, self.smap):
            return False

        if self.direction=='s' and not _consistent_allow_none(cd.mode, self.mode):
            return False

        return True


class AxDimensionS(AxDimensionScale):
    def __init__(self, *args, **kwargs):
        processed_kwargs = _process_dimension_kwargs('s', kwargs)
        smap_ = kwargs.pop('smap', None)
        smap = kwargs.pop('sizemap', smap_)
        self.smap = smap

        self.mode = kwargs.pop('mode', None)

        self.nsamples = 20
        super(AxDimensionS, self).__init__('s', *args, **processed_kwargs)

    @property
    def nsamples(self):
        return self._nsamples

    @nsamples.setter
    def nsamples(self, nsamples):
        if not isinstance(nsamples, int):
            raise TypeError("nsamples must be of type int")
        if nsamples < 2:
            raise ValueError("nsamples must be >= 2")

        self._nsamples = nsamples

    @property
    def smap(self):
        smap = self._smap
        if smap is None:
            return (0.01,0.05)
        return smap

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


    def normalize(self, values, pad=None, i=None):
        norm = self.get_norm(pad=pad, i=None)
        values_normed = norm(values)
        smap = self.smap
        srang = smap[1] - smap[0]
        values_mapped = values_normed*srang+smap[0]
        return values_mapped

    def get_sizebar_samples(self, nsamples=None, pad=None, i=None):
        if nsamples is None:
            nsamples = self.nsamples
        lim = self.get_lim(pad=pad, i=i)
        smap = self.smap
        rang = float(lim[1] - lim[0])  # TODO: not sure how this will react with flipped limits
        srang = float(smap[1] - smap[0])
        samples = []
        sizes = []
        for i in range(nsamples):
            samples.append(lim[0]+i*rang/(nsamples-1))
            sizes.append(smap[0]+i*srang/(nsamples-1))
        return np.array(samples), np.array(sizes)

class AxDimensionC(AxDimensionScale):
    def __init__(self, *args, **kwargs):
        cmap_ = kwargs.pop('cmap', None)
        cmap = kwargs.pop('colormap', cmap_)
        self.cmap = cmap

        processed_kwargs = _process_dimension_kwargs('c', kwargs)
        super(AxDimensionC, self).__init__('c', *args, **processed_kwargs)

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        # print("setting axes cmap: {}".format(cmap))
        try:
            cmap_ = plt.get_cmap(cmap)
        except:
            raise TypeError("could not find cmap")

        self._cmap = cmap

class AxViewGroup(common.Group):
    def __init__(self, items):
        super(AxViewGroup, self).__init__(AxView, ['direction'], items)

    @property
    def direction(self):
        return self._get_attrs('direction')

    @property
    def value(self):
        return self._get_attrs('value')

    @value.setter
    def value(self, value):
        return self._set_attrs('value', value)

class AxView(AxArray):
    def __init__(self, direction, axes, value):
        self._value = value

        super(AxView, self).__init__(direction, axes)

    def __repr__(self):

        return "<{} | >".format(self.direction)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value is None:
            self._value = value
            return

        if isinstance(value, float) or isinstance(value, int):
            value = [value]

        if isinstance(value, list) or isinstance(value, tuple):
            value = np.array(value)

        if isinstance(value, np.ndarray):
            self._value = value
            return

        raise TypeError("{} value must be a numpy array or float or None".format(self.direction))

    def get_value(self, i, indeps=None):
        """
        access the interpolated value at a give value of i (independent-variable)

        if indeps is not passed, then the entire range of indeps over all calls is assumed
        """
        if self.value is None:
            return None

        if i is None:
            return np.median(self.value)

        if indeps is None:
            indeps_all_calls = list(set(np.concatenate([common.tolist(call.i.get_value(unit=self.axes.i.unit)) for call in self.axes.calls])))
            indeps = np.linspace(np.nanmin(indeps_all_calls),
                                 np.nanmax(indeps_all_calls),
                                 len(self.value))

        if len(indeps) != len(self.value):
            raise ValueError("indeps and value must have same length")

        return np.interp(i, indeps, self.value)

class AxViewElev(AxView):
    def __init__(self, axes, value=None):
        super(AxViewElev, self).__init__('elev', axes, value)

class AxViewAzim(AxView):
    def __init__(self, axes, value=None):
        super(AxViewAzim, self).__init__('azim', axes, value)
