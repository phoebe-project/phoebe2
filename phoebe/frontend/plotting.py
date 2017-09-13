
import numpy as np
import os
import webbrowser
from scipy import interpolate
from time import sleep

# import phoebe.parameters as parameters


try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib import colors
    from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
    from mpl_toolkits.mplot3d import Axes3D
except (ImportError, TypeError):
    _use_mpl = False
else:
    _use_mpl = True
    _mplcolors = colors.cnames.keys() + colors.ColorConverter.colors.keys() + ['None', 'none']

try:
    from bokeh import plotting as bkh
    from bokeh import mpl as bkhmpl
except ImportError:
    _use_bkh = False
else:
    _use_bkh = True

try:
    import mpld3 as _mpld3
except ImportError:
    _use_mpld3 = False
else:
    _use_mpld3 = True

from phoebe import conf

import logging
logger = logging.getLogger("PLOTTING")
logger.addHandler(logging.NullHandler())


# For other less-common plotting backends, we'll wait to attempt importing until
# they're requested.  This could cause some redundancy if those functions are called
# multiple times, but at least we won't be importing all these large plotting packages
# always.


# TODO: this is redundant with parameters.py - figure out a way to import without circular imports (perhaps put these in their own module)
_meta_fields_twig = ['time', 'qualifier', 'history', 'component',
                'dataset', 'constraint', 'compute', 'model', 'fitting',
                'feedback', 'plugin', 'kind',
                'context']

_meta_fields_all = _meta_fields_twig + ['twig', 'uniquetwig', 'uniqueid']

def _is_none(value):
    if isinstance(value, list) or isinstance(value, np.ndarray):
        return False

    if value is None:
        return True

    if isinstance(value, str) and value.lower()=='none':
        return True

    return False

def _mpl_append_axes(fig, **kwargs):
    def determine_grid(N):
        cols = np.floor(np.sqrt(N))
        rows = np.ceil(float(N)/cols) if cols > 0 else 1
        return int(rows), int(cols)

    N = len(fig.axes)

    # we'll reset the layout later anyways
    ax_new = fig.add_subplot(1,N+1,N+1, **kwargs)

    axes = fig.axes
    N = len(axes)

    rows, cols = determine_grid(N)

    for i,ax in enumerate(axes):
        ax.change_geometry(rows, cols, i+1)

    return ax_new


def mpl(ps, data, plot_inds, do_plot=True, **kwargs):
    if not _use_mpl:
        raise ImportError("failed to import matplotlib: try 'sudo pip install matplotlib' or choose another plotting backend")

    def _value(obj):
        if hasattr(obj, 'value'):
            return obj.value
        return obj

    def _process_colorarray(ps, kwargs, colorkey, make_array=None, array_inds=None):
        # If color is a recognized mpl color then we can just pass it through
        # mplkwargs to the plot call.
        # If color is a twig pointing to an array, then we need to remove
        # color from mplkwargs, make the plotting call without it, and then
        # overplot a scatter call

        _symmetric_colorkeys = ['rvs', 'vxs', 'vys', 'vzs', 'nxs', 'nys', 'nzs']

        if isinstance(kwargs[colorkey], float):
            kwargs[colorkey] = str(kwargs[colorkey])
            is_float = True
        elif isinstance(kwargs[colorkey], list) or isinstance(kwargs[colorkey], np.ndarray):
            colorarray = np.asarray(kwargs[colorkey])
            is_float = False
        else:
            try:
                float(kwargs[colorkey])
            except ValueError:
                is_float = False
                colorarray = None
            except TypeError:
                is_float = False
                colorarray = None
            else:
                # matplotlib also accepts stringed floats which will be converted to grayscale (ie '0.8')
                is_float = True

        if _is_none(kwargs[colorkey]) or is_float or (kwargs[colorkey] in _mplcolors and kwargs[colorkey].split('@')[0] not in _symmetric_colorkeys):
            colorarray = None
        else:
            color = kwargs.pop(colorkey)
            if not isinstance(color, str):
                color = ''
            # print "***", color, ps.qualifiers
            # colorarray = ps.get_value(twig=color)
            # metawargs = {k:v for k,v in ps.meta.items() if k not in ['qualifier', ]

            # TODO: can we handle looping over times here?  Or does that belong in animate only?
            #colorarray = np.concatenate([_value(ps.get_value(twig=color, component=c, time=t)) for c in ps.components for t in ps.times])
            if colorarray is None:
                if len(ps.times)>1:
                    colorarray = np.concatenate([_value(ps.get_value(twig=color, component=c, time=t)) for c in ps.components for t in ps.times])
                else:
                    colorarray = np.concatenate([_value(ps.get_value(twig=color, component=c)) for c in ps.components]) if len(ps.components) else _value(ps.get_value(twig=color))

            if make_array is not None:
                # then we need to alter the value in the kwargs dictionary to be a
                # normalized array that can be sent to the cmap

                if colorarray.dtype != np.float:
                    colorarray = colorarray.astype('float')

                # let's make the colorarray not just set by the visible elements, but the elements on all surfaces across all components
                # we'll do this by setting based on the whole array and then only return the values at array_inds
                # NOTE: this will not handle constant cmap with time, but will make it fixed with zooming/perspective for a given time

                if color.split('@')[0] in _symmetric_colorkeys:
                    logger.info("forcing color range to be centered around 0 for '{}'".format(color))
                    # then we want to make sure 0 velocity is in the middle of the range
                    # imagine RVs from -10 ... 0 ... 20
                    valmin = np.nanmin(colorarray)
                    valmax = np.nanmax(colorarray)
                    # now valmin is -10, valmax is 20
                    half_range = np.max([abs(valmin), abs(valmax)])
                    # now half_range is 20

                    # actual color-range needs to go from -half_range to +half_range
                    colorarray += half_range
                    # now colorarray is 10 ... 20 ... 40
                    colorarray /= half_range*2
                    # now colorarray is 0.25 ... 0.5 ... 1.0
                    # perfect!

                else:
                    colorarray -= np.nanmin(colorarray)
                    colorarray /= np.nanmax(colorarray)

                kwargs[colorkey] = plt.get_cmap(make_array)(colorarray)[array_inds]


        return kwargs, colorarray

    def _default_cmap(ps, colorkey):

        #colorparam = ps.filter(twig=color, component=c)
        if isinstance(colorkey, str):
            colorkey = colorkey.split('@')[0]
        elif colorkey is None:
            return None
        else:
            return 'rainbow'

        # TODO: add support for plt.blackbody_cmap()

        if colorkey in ['rvs']:
            cmap = 'RdBu_r'
        elif colorkey in ['vxs', 'vys', 'vzs']:
            cmap = 'RdBu'
        elif colorkey in ['teffs']:
            cmap = 'afmhot'
        elif colorkey in ['loggs']:
            cmap = 'gnuplot'
        elif colorkey in ['visibilities']:
            cmap = 'RdYlGn'
        elif len(colorkey)<=1 or colorkey in _mplcolors:
            # then we don't need a cmap, this is a single color
            return None
        else:
            cmap = 'rainbow'

        logger.info("defaulting to '{}' colormap for '{}'".format(cmap, colorkey))
        return cmap

    return_artists = []
    return_data_per_artist = []

    xunit = kwargs.pop('xunit', None)
    yunit = kwargs.pop('yunit', None)
    zunit = kwargs.pop('zunit', None)
    xlabel = kwargs.pop('xlabel', '')
    ylabel = kwargs.pop('ylabel', '')
    zlabel = kwargs.pop('zlabel', '')

    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()
        if hasattr(ax, '_phoebe_kind') and ps.kind != ax._phoebe_kind:
            if ps.kind in ['orb', 'mesh'] and xunit==yunit:
                ax = _mpl_append_axes(plt.gcf(), aspect='equal')
            else:
                ax = _mpl_append_axes(plt.gcf())
        else:
            # not sure if we want this - the user may have set the aspect ratio already
            if ps.kind in ['orb', 'mesh'] and xunit==yunit:
                # TODO: for aspect ratio (here and above) can we be smarter and
                # check for same units?
                ax.set_aspect('equal')

    ax._phoebe_kind = ps.kind

    # let's check to see if we're dealing with a 3d axes or not
    axes_3d = isinstance(ax, Axes3D)





    mplkwargs = {k:v for k,v in kwargs.items() if k not in
        _meta_fields_all+['x', 'y', 'z', 'xerrors', 'yerrors', 'zerrors',
            'xlabel', 'ylabel', 'zlabel',
            'xlim', 'ylim', 'zlim',
            'xunit', 'yunit', 'zunit', 'time', 't0', 'highlight', 'highlight_ms',
            'highlight_marker', 'highlight_color', 'uncover',
            'facecolor', 'edgecolor', 'facecmap', 'edgecmap', 'correct', 'plotting_backend']}

    if ps.kind in ['mesh', 'mesh_syn'] and kwargs.get('polycollection', False):
        # here plot_inds is used to sort in the z-direction
        pckwargs = {}
        pckwargs['facecolors'] = kwargs.get('facecolor', 'w')  # note change from singular -> plural
        pckwargs['edgecolors'] = kwargs.get('edgecolor', 'k')  # note change from singular -> plural
        pckwargs['zorder'] = mplkwargs.get('zorder', 1)

        pckwargs, facecolorarray = _process_colorarray(ps, pckwargs, 'facecolors', kwargs.get('facecmap', _default_cmap(ps, pckwargs['facecolors'])), plot_inds)
        pckwargs, edgecolorarray = _process_colorarray(ps, pckwargs, 'edgecolors', kwargs.get('edgecmap', _default_cmap(ps, pckwargs['facecolors'])), plot_inds)

        if _is_none(pckwargs['edgecolors']) and not _is_none(pckwargs['facecolors']):

            # we should set linewidths to 0 so that colors from background triangles
            # don't show through the gaps
            pckwargs['edgecolors'] = pckwargs['facecolors']


        # pckwargs will now have facecolors and edgecolors - either as strings
        # or as an array of colors ready for matplotlib.  We do that this way
        # here because edge and face may have different cmaps.  For individual
        # xy plots we leave the normalizing and cmaps up to matplotlib.

        if do_plot:
            if axes_3d:
                #coordinate_inds = [['x', 'y', 'z'].index(q) for q in [xqualifier, yqualifier,zqualifier]]
                pc = Poly3DCollection(data[plot_inds], **pckwargs)
            else:
                #coordinate_inds = [['x', 'y', 'z'].index(q) for q in [xqualifier, yqualifier]]
                pc = PolyCollection(data[plot_inds], **pckwargs)

            return_artists.append(pc)
            ax.add_collection(pc)
        else:
            # collections don't support set_data, so in order to animate we'll need
            # to delete and recreate the collection.  We'll get the data if do_plot=False,
            # but we need to remember the kwargs to send when we recreate the collection
            datakwargs = pckwargs.copy()
            datakwargs['data'] = data[plot_inds]
            return_data_per_artist.append(datakwargs)

        # matplotlib doesn't set smart limits when polycollections are added, so we'll extend
        # the axes limits if necessary.
        if do_plot:
            # first we'll get what the limits are now
            curr_xlim = ax.get_xlim()
            curr_ylim = ax.get_ylim()
            if axes_3d:
                curr_zlim = ax.get_zlim()

            # then we'll extend each limit only if the data requires
            data_xs = data[plot_inds,:,0].flatten()
            data_ys = data[plot_inds,:,1].flatten()
            if axes_3d:
                data_zs = data[plot_inds,:,2].flatten()

            # if the user (or a previous plotting call) already set the limits beyond
            # the data, we definitely don't want to shrink the axes.  However, we do
            # want to ignore the default blank-axes limits of (0, 1)
            xlim = (curr_xlim[0] if curr_xlim[0] < min(data_xs) and curr_xlim[0]!=0 else min(data_xs), curr_xlim[1] if curr_xlim[1] > max(data_xs) and curr_xlim[1]!=1 else max(data_xs))
            ylim = (curr_ylim[0] if curr_ylim[0] < min(data_ys) and curr_ylim[0]!=0 else min(data_ys), curr_ylim[1] if curr_ylim[1] > max(data_ys) and curr_ylim[1]!=1 else max(data_ys))
            if axes_3d:
                zlim = (curr_zlim[0] if curr_zlim[0] < min(data_zs) and curr_zlim[0]!=0 else min(data_zs), curr_zlim[1] if curr_zlim[1] > max(data_zs) and curr_zlim[1]!=1 else max(data_zs))


            # now, if we changed a limit, we want to pad it by 10%, but if it was its original value we don't want to add extra padding
            xrange = xlim[1]-xlim[0]
            yrange = ylim[1]-ylim[0]
            if axes_3d:
                zrange = zlim[1]-zlim[0]

            pad = 0.1
            xlim = (xlim[0] if xlim[0]==curr_xlim[0] else xlim[0]-pad*xrange, xlim[1] if xlim[1]==curr_xlim[1] else xlim[1]+pad*xrange)
            ylim = (ylim[0] if ylim[0]==curr_ylim[0] else ylim[0]-pad*yrange, ylim[1] if ylim[1]==curr_ylim[1] else ylim[1]+pad*yrange)
            if axes_3d:
                zlim = (zlim[0] if zlim[0]==curr_zlim[0] else zlim[0]-pad*zrange, zlim[1] if zlim[1]==curr_zlim[1] else zlim[1]+pad*zrange)

            # and lastly, we'll apply these limits to the axes - unless the user sent values for the limits
            if axes_3d:
                ax.set_xlim3d(kwargs.get('xlim', xlim))
            else:
                ax.set_xlim(kwargs.get('xlim', xlim))
            if axes_3d:
                ax.set_ylim3d(kwargs.get('ylim', ylim))
            else:
                ax.set_ylim(kwargs.get('ylim', ylim))
            if axes_3d:
                ax.set_zlim3d(kwargs.get('zlim', zlim))


    else:
        # Then we're just a general x-y (or x-y-z) plot
        xarray, yarray, zarray, tarray = data

        # Handle color
        mplkwargs, colorarray = _process_colorarray(ps, mplkwargs, 'color')

        # Handle errorbars by making a separate plot call (with no marker)
        if kwargs['xerrors'] not in [None, False] or kwargs['yerrors'] not in [None, False]:
            xerr = ps.get_value(kwargs['xerrors'], xunit) if kwargs['xerrors'] not in [None, False] else None
            yerr = ps.get_value(kwargs['yerrors'], yunit) if kwargs['yerrors'] not in [None, False] else None
            if axes_3d:
                zerr = ps.get_value(kwargs['zerrors'], zunit) if kwargs['zerrors'] not in [None, False] else None

            # TODO: make color match the plot call
            # TODO: allow passing kwargs (ie. errorevery, ecolor, elinewidth, capsize) - will need to remove these from mplkwargs and pull just them here, defaulting ecolor to color or the color from the plot command if not provided

            if do_plot:
                if axes_3d:
                    # TODO: test this with animations
                    artist = ax.errorbar(xarray[plot_inds], yarray[plot_inds], zarray[plot_inds], fmt='', linestyle='None',
                        zerr=zerr, yerr=yerr , xerr=xerr, ecolor='k')
                else:
                    artist = ax.errorbar(xarray[plot_inds], yarray[plot_inds], fmt='', linestyle='None',
                        yerr=yerr , xerr=xerr, ecolor='k')

                return_artists.append(artist)
            else:
                if axes_3d:
                    return_data_per_artist.append((xarray[plot_inds], yarray[plot_inds], zarray[plot_inds]))
                else:
                    return_data_per_artist.append((xarray[plot_inds], yarray[plot_inds]))


        # Make the main plotting call
        if colorarray is None:
            if do_plot:
                # then we can do this the easy way with a single call to plot
                if axes_3d:
                    artists = ax.plot(xarray[plot_inds], yarray[plot_inds], zarray[plot_inds], **mplkwargs)
                else:
                    artists = ax.plot(xarray[plot_inds], yarray[plot_inds], **mplkwargs)

                return_artists += artists
            else:
                if axes_3d:
                    return_data_per_artist.append((xarray[plot_inds], yarray[plot_inds], zarray[plot_inds]))
                else:
                    return_data_per_artist.append((xarray[plot_inds], yarray[plot_inds]))

        # NOTE that these are separate ifs (rather than elifs) because they may
        # both need to be called if asking for a linestyle and marker
        if colorarray is not None and not _is_none(kwargs['linestyle']):
            # then we have to do this the hard way
            # see: http://matplotlib.org/examples/pylab_examples/multicolored_line.html
            if axes_3d:
                # TODO: test this!
                points = np.array([xarray[plot_inds], yarray[plot_inds], zarray[plot_inds]]).T.reshape(-1, 1, 3)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

            else:
                points = np.array([xarray[plot_inds], yarray[plot_inds]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # TODO: add support for linewidths (but what keyword would do this - data would need to be markersizes)

            # Now we handle the color map for a bunch of line-segments.  Note
            # that we normalize the colormap based on the entire colorarray,
            # so that the map will stay in place if uncover=True
            if do_plot:
                logger.debug("kwargs passed to LineCollection: {}".format(mplkwargs))

                mplkwargs.setdefault('cmap', _default_cmap(ps, mplkwargs.get('color', None)))
                lc = LineCollection(segments,
                    norm=plt.Normalize(min(colorarray), max(colorarray)),
                    **{k:v for k,v in mplkwargs.items() if k not in ['markersize', 'marker']})
                lc.set_array(colorarray[plot_inds])
                return_artists.append(lc)
                ax.add_collection(lc)
            else:
                return_data_per_artist.append(segments)
                # TODO: test this in animation - what about colorarray[plot_inds]???


        if colorarray is not None and not _is_none(kwargs['marker']):
            # we could loop through ax.plot calls... but let's just call ax.scatter
            # TODO: take s for sizes
            if do_plot:
                if axes_3d:
                    artist = ax.scatter(xarray[plot_inds], yarray[plot_inds], zarray[plot_inds], c=colorarray[plot_inds],
                        cmap=plt.get_cmap(kwargs.get('cmap', _default_cmap(ps, mplkwargs.get('color', None)))),
                        norm=plt.Normalize(min(colorarray), max(colorarray)),
                        marker=kwargs['marker'],
                        linewidths=0) # linewidths=0 removes the black edge
                else:
                    artist = ax.scatter(xarray[plot_inds], yarray[plot_inds], c=colorarray[plot_inds],
                        cmap=plt.get_cmap(kwargs.get('cmap', _default_cmap(ps, mplkwargs.get('color', None)))),
                        norm=plt.Normalize(min(colorarray), max(colorarray)),
                        marker=kwargs['marker'],
                        linewidths=0) # linewidths=0 removes the black edge

                return_artists.append(artist)
            else:
                if axes_3d:
                    return_data_per_artist.append((xarray[plot_inds], yarray[plot_inds], zarray[plot_inds]))
                else:
                    return_data_per_artist.append((xarray[plot_inds], yarray[plot_inds]))


        # Now handle higlighting, if applicable.  We need to do this last to get
        # it on top of the line plot
        if kwargs['highlight'] and kwargs['time'] is not None and len(tarray):
            logger.info("highlight at time={}".format(kwargs['time']))

            # TODO: can we changed to use FloatArrayParameter.interp_value?
            interp_x = interpolate.interp1d(tarray, xarray, bounds_error=False)
            interp_y = interpolate.interp1d(tarray, yarray, bounds_error=False)
            h_xarray = [interp_x(kwargs['time'])]
            h_yarray = [interp_y(kwargs['time'])]

            if axes_3d:
                interp_z = interpolate.interp1d(tarray, zarray, bounds_error=False)
                h_zarray = [interp_z(kwargs['time'])]


            highlight_kwargs = {}
            highlight_kwargs['zorder'] = 99 # try to place on top
            highlight_kwargs['marker'] = kwargs['highlight_marker']
            if kwargs['highlight_color'] is not None:
                highlight_kwargs['color'] = kwargs['highlight_color']
            else:
                # we want the same color as the last plot call
                # TODO: make this work
                highlight_kwargs['color'] = 'b'

            if kwargs['highlight_ms'] is not None:
                highlight_kwargs['ms'] = kwargs['highlight_ms']
            else:
                # mpl will handle automatically
                pass

            if do_plot:
                if axes_3d:
                    artists = ax.plot(h_xarray, h_yarray, h_zarray, **highlight_kwargs)
                else:
                    artists = ax.plot(h_xarray, h_yarray, **highlight_kwargs)

                return_artists += artists
            else:
                if axes_3d:
                    return_data_per_artist.append((h_xarray, h_yarray, h_zarray))
                else:
                    return_data_per_artist.append((h_xarray, h_yarray))

    # Lastly, let's handle smart axes labels if they're not already set.
    # If they are already set but not as expected, let's just log a warning
    # but leave them as they are
    if do_plot:
        if len(ax.get_xlabel()) and ax.get_xlabel()!=xlabel:
            logger.warning("xlabel is already set but is not '{}' - ignoring".format(xlabel))
        else:
            ax.set_xlabel(xlabel)

        if len(ax.get_ylabel()) and ax.get_ylabel()!=ylabel:
            logger.warning("ylabel is already set but is not '{}' - ignoring".format(ylabel))
        else:
            ax.set_ylabel(ylabel)

        if axes_3d:
            if len(ax.get_zlabel()) and ax.get_zlabel()!=zlabel:
                logger.warning("zlabel is already set but is not '{}' - ignoring".format(zlabel))
            else:
                ax.set_zlabel(zlabel)

        # now let's handle setting the axes limits, but only if the user sent values
        # TODO: should we handle ax._phoebe_xlim and friends here?
        if kwargs.get('xlim', False):
            if axes_3d:
                ax.set_xlim3d(kwargs['xlim'])
            else:
                ax.set_xlim(kwargs['xlim'])
        if kwargs.get('ylim', False):
            if axes_3d:
                ax.set_ylim3d(kwargs['ylim'])
            else:
                ax.set_ylim(kwargs['ylim'])
        if axes_3d and kwargs.get('zlim', False):
            ax.set_zlim3d(kwargs['zlim'])



    if do_plot:
        if kwargs.get('correct', False):
            # Easter Egg for "calculations correct"
            # inspired by The Martian
            at = AnchoredText("CALCULATIONS\n     CORRECT",
                              prop=dict(size=15), frameon=True,
                              loc=10,
                              )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            # this is not animatable
            #return_artists.append(at)
            ax.add_artist(at)

    if do_plot:
        return ax, return_artists
    else:
        return return_data_per_artist

def show_mpl(**kwargs):
    plt.show()

def save_mpl(fname, **kwargs):
    return plt.savefig(fname)

def mpld3(ps, data, plot_inds, **kwargs):
    """
    """
    if not conf.devel:
        raise NotImplementedError("'mpld3' plotting backend not officially supported for this release.  Enable developer mode to test.")


    if not _use_mpld3:
        raise ImportError("failed to import mpld3: try 'sudo pip install mpld3' or choose another plotting backend")
    if not _use_mpl:
        raise ImportError("failed to import matplotlib: try 'sudo pip install matplotlib' or choose another plotting backend")

    return mpl(ps, data, plot_inds, **kwargs)


def show_mpld3(**kwargs):
    if not _use_mpld3:
        raise ImportError("failed to import mpld3: try 'sudo pip install mpld3' or choose another plotting backend")

    # _mpld3.show(plt.gcf())
    _mpld3.save_html(kwargs.get('ax', plt.gcf()), '_mpld3.html')

    logger.info("opening mpld3 plot in browser")
    webbrowser.open('_mpld3.html')

    # TODO: needs to just be input for python3
    # raw_input("press enter to continue...")
    sleep(2)

    os.remove('_mpld3.html')


def save_mpld3(fname, **kwargs):
    if not _use_mpld3:
        raise ImportError("failed to import mpld3: try 'sudo pip install mpld3' or choose another plotting backend")

    _mpld3.save_html(kwargs.get('ax', plt.gcf()), fname)
    return fname

def mpl2bokeh(ps, data, plot_inds, **kwargs):
    """
    """
    if not conf.devel:
        raise NotImplementedError("'mpld3' plotting backend not officially supported for this release.  Enable developer mode to test.")

    if not _use_bkh:
        raise ImportError("failed to import bokeh: try 'sudo pip install bokeh' or choose another plotting backend")
    if not _use_mpl:
        raise ImportError("failed to import matplotlib: try 'sudo pip install matplotlib' or choose another plotting backend")

    # mpl2bokeh compiles everything through matplotlib and tries to convert
    # only when calling show
    return mpl(ps, data, plot_inds, **kwargs)

def show_mpl2bokeh(**kwargs):
    """
    """
    if not _use_bkh:
        raise ImportError("failed to import bokeh: try 'sudo pip install bokeh' or choose another plotting backend")

    bkh.output_file('_bkh.html', title='')
    show_bokeh(ax=bkhmpl.to_bokeh())

def save_mpl2bokeh(fname, **kwargs):
    if not _use_bkh:
        raise ImportError("failed to import bokeh: try 'sudo pip install bokeh' or choose another plotting backend")

    raise NotImplementedError


def bokeh(ps, data, plot_inds, **kwargs):
    """

    Args:
        ps
        data
        plot_inds
        xunit, yunit, zunit
        xlabel, ylabel, zlabel
        label
        title
        linewidth
    """
    if not conf.devel:
        raise NotImplementedError("'mpld3' plotting backend not officially supported for this release.  Enable developer mode to test.")

    if not _use_bkh:
        raise ImportError("failed to import bokeh: try 'sudo pip install bokeh' or choose another plotting backend")


    # TODO: look into streaming option to stream fits as fitting or as sampling???

    logger.warning("bokeh plotting backend is VERY EXPERIMENTAL and most features are missing")

    xarray, yarray, zarray, tarray = data
    xunit = kwargs.pop('xunit', None)
    yunit = kwargs.pop('yunit', None)
    zunit = kwargs.pop('zunit', None)
    xlabel = kwargs.pop('xlabel', '')
    ylabel = kwargs.pop('ylabel', '')
    zlabel = kwargs.pop('zlabel', '')
    title = kwargs.pop('title', 'PHOEBE 2.0 Bokeh Plot')

    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = bkh.figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel)
        bkh.output_file('_bkh.html', title=title)


    if not is_none(kwargs['linestyle']):
        # TODO: handle color
        ax.line(xarray[plot_inds], yarray[plot_inds], legend=kwargs.get('label', ''), line_width=kwargs.get('linewidth', 2))

    if not is_none(kwargs['marker']):
        # Then let's try to get the closest match
        if kwargs['marker'] in ['o', '.', ',']:
            marker = 'circle'
        elif kwargs['marker'] in ['s']:
            marker = 'square'
        elif kwargs['marker'] in ['^', '>', '<',  '2', '3', '4']:
            marker = 'triangle'
        elif kwargs['marker'] in ['v', '1']:
            marker = 'inverted_triangle'
        elif kwargs['marker'] in ['*']:
            marker = 'asterisk'
        elif kwargs['marker'] in ['+']:
            marker = 'cross'
        elif kwargs['marker'] in ['x']:
            marker = 'x'
        elif kwargs['marker'] in ['d', 'D']:
            marker = 'diamond'
        else:
            logger.warning("could not convert marker to one supported by bokeh - defaulting to circle")
            marker = 'circle'

        # TODO: handle markersize (markersize -> size)
        # TODO: handle color
        # edgecolor -> line_color
        # color -> fill_color
        ax.scatter(xarray[plot_inds], yarray[plot_inds], marker=marker)

    # TODO: handle highlighting
    # TODO: handle color and colorarrays (see matplotlib)
    # TODO: handle errorbars

    return ax, []


def show_bokeh(**kwargs):

    if not _use_bkh:
        raise ImportError("failed to import bokeh: try 'sudo pip install bokeh' or choose another plotting backend")

    bkh.show(kwargs['ax'])

    # TODO: needs to just be input for python3
    # raw_input("press enter to contisnue...")
    sleep(2)

    os.remove('_bkh.html')

def save_bokeh(fname, **kwargs):
    bkh.save(kwargs['ax'])
    os.rename('_bkh.html', fname)
    return fname



def lightning(ps, data, plot_inds, **kwargs):
    """
    """
    if not conf.devel:
        raise NotImplementedError("'mpld3' plotting backend not officially supported for this release.  Enable developer mode to test.")

    try:
        from lightning import Lightning
    except ImportError:
        raise ImportError("failed to import lightning: try 'sudo pip install lightning-python' or choose another plotting backend")
        _use_lgn = False
    else:
        _use_lgn = True
        # TODO: option to use server instead of local (for sharing links, etc)
        lgn = Lightning(local=True)

    logger.warning("lightning plotting backend is VERY EXPERIMENTAL (read: non-existant)")

    # it looks like all calls for a single figure need to be made at once by giving lists
    # TODO: look into streaming option to stream fits as fitting or as sampling???

    series = np.random.randn(5, 50)

    viz = lgn.line(series)

    # TODO: return and accept viz and then write a .show() function?
    viz.save_html('_lgn.html', overwrite=True)

    return None, []

def show_lightning(**kwargs):

    logger.info("opening lightning plot in browser")
    webbrowser.open('_lgn.html')

    # TODO: needs to just be input for python3
    raw_input("press enter to continue...")

    os.remove('_lgn.html')

def save_lightning(fname, **kwargs):

    os.rename('_lgn.html', fname)
    return fname


def plotly(ps, data, plot_inds, **kwargs):
    raise NotImplementedError

    # https://plot.ly/python/getting-started/
    # NOTE that unless paid/offline plan, plots may be publicly available
    # could create an mpl2plotly that would be fairly easy


def gnuplot(ps, data, plot_inds, **kwargs):
    raise NotImplementedError

    # https://sourceforge.net/projects/gnuplot-py/
    # http://gnuplot-py.sourceforge.net/


def _template_plotting_function(ps, data, plot_inds, **kwargs):
    """
    """

    xarray, yarray, zarray, tarray = data
    xunit = kwargs.pop('xunit', None)
    yunit = kwargs.pop('yunit', None)
    zunit = kwargs.pop('zunit', None)
    xlabel = kwargs.pop('xlabel', '')
    ylabel = kwargs.pop('ylabel', '')
    zlabel = kwargs.pop('zlabel', '')

    ps.kind

    plot(xarray[plot_inds], yarray[plot_inds])


# OTHER BACKENDS TO CONSIDER:
# - mpl2plotly
# - chaco (https://github.com/enthought/chaco)
# - mayavi (***)
# - opengl
# - visvis
