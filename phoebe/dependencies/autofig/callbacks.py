from matplotlib import backend_bases, collections, lines
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

class Callbacks(object):
    def __init__(self, parent):
        # parent should be Axes or Figure instance
        self.parent = parent
        self._artists = {}

    def connect(self, afevent, artist, mplevent, callback):
        self.parent.callbacks.connect(mplevent, callback)
        # if callback in self.parent.callbacks.callbacks.get(event, {}):

        if afevent not in self._artists.keys():
            self._artists[afevent] = []

        self._artists[afevent].append(artist)

    def get_artists(self, callback):
        if not isinstance(callback, str):
            # then maybe we were based the callable itself
            callback = callback.__name__

        return self._artists.get(callback, [])


def _connect_to_autofig(afobj, mplobj):
    if not hasattr(mplobj, '_af'):
        mplobj._af = afobj
        mplobj._af_callbacks = Callbacks(mplobj)


##################### BEGIN AVAILABLE CALLBACKS ################################

def update_indep(artist, call):
    def callback_key_event(event):
        if event.key == 'i':
            callback_event(event)

    def callback_event(event):
        axes = event.inaxes
        if axes is None:
            return

        if isinstance(axes, Axes3D):
            # xdata and ydata will likely be meaningless
            return

        afaxes = axes._af
        if afaxes.i.reference == 'x':
            i = event.xdata
        elif afaxes.i.reference == 'y':
            i = event.ydata

        for artist in axes._af_callbacks.get_artists('update_indep'):
            if not hasattr(artist, '_af'):
                continue

            call = artist._af

            # TODO: need to clear artists from call and redraw with i=i

    raise NotImplementedError
    artist.axes.figure.canvas._af_callbacks.connect('update_indep', artist, 'button_press_event', callback_event)
    artist.axes.figure.canvas._af_callbacks.connect('update_indep', artist, 'key_press_event', callback_key_event)

def update_sizes(artist, call, run_callback=False):
    def callback_canvas(event):
        for axes in event.canvas.figure.axes:
            callback_axes(axes)

        return

    def callback_axes(ax):
        if not hasattr(ax, '_af_callbacks'):
            return


        for artist in ax._af_callbacks.get_artists('update_sizes'):
            if not hasattr(artist, '_af'):
                continue

            afobj = artist._af

            if afobj._class == 'Call':
                # then we are an artist in THIS axes, so let's check the
                # necessary mode for resizing and set the size based on THIS ax
                call = afobj
                axes = call.axes

                if not hasattr(call, 's'):
                    continue

                mode_dims, mode_obj, mode_mode = call.s._mode_split()

                ax = ax

                if hasattr(artist, '_af_highlight'):
                    sizes_orig = call.highlight_size
                elif hasattr(artist, '_af_sizes'):
                    sizes_orig = artist._af_sizes
                else:
                    # TODO: need to get sizes with current i
                    sizes_orig = call._sizes

            elif afobj._class == 'AxDimensionS':
                # then we should be an artist in a sidebar, so we actually
                # care about the parent axes limits rather than our own
                axdimensions = afobj
                axes = axdimensions.axes

                # ax will be the MPL object of the "parent" axes
                ax = axdimensions.axes._backend_object

                mode_dims, mode_obj, mode_mode = axdimensions._mode_split()

                # TODO: pass i
                if isinstance(artist, collections.LineCollection):
                    nsamples = 100
                else:
                    nsamples = None

                ys, sizes_orig = afobj.get_sizebar_samples(i=None, nsamples=nsamples)
            else:
                raise NotImplementedError

            if mode_dims == 'pt':
                a_disp = 1

            else:
                if mode_mode == 'fixed':
                    if hasattr(artist, '_af_update_size_draw_complete'):
                        continue
                    else:
                        artist._af_update_size_draw_complete = True
                        mode_mode = 'current'


                if mode_mode == 'current':
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                elif mode_mode == 'original':
                    # TODO: need to pass i
                    xlim = axes.x.get_lim()
                    ylim = axes.y.get_lim()
                elif mode_mode == 'figure':
                    pass
                else:
                    raise NotImplementedError


                if mode_obj == 'figure':
                    w_disp, h_disp = ax.figure.get_size_inches() * ax.figure.dpi

                    if mode_mode == 'original':
                        # then we need to fake w_disp and h_disp to get the
                        # correct factor when zooming
                        w_disp *= abs((xlim[1]-xlim[0])/(ax.get_xlim()[1]-ax.get_xlim()[0]))
                        h_disp *= abs((ylim[1]-ylim[0])/(ax.get_ylim()[1]-ax.get_ylim()[0]))

                elif mode_obj == 'axes':
                    xr_disp = ax.transData.transform([float(max(xlim)), 0])
                    xl_disp = ax.transData.transform([float(min(xlim)), 0])

                    yt_disp = ax.transData.transform([0, float(max(ylim))])
                    yb_disp = ax.transData.transform([0, float(min(ylim))])

                    w_disp = abs(xr_disp[0] - xl_disp[0])
                    h_disp = abs(yt_disp[1] - yb_disp[1])

                else:
                    raise NotImplementedError


                if mode_dims == 'x':
                    a_disp = w_disp**2
                elif mode_dims == 'y':
                    a_disp = h_disp**2
                elif mode_dims == 'xy':
                    a_disp = w_disp * h_disp
                else:
                    raise NotImplementedError

            # TODO: need to pass i, need to handle z-order loop
            if sizes_orig is None:
                continue

            ms = sizes_orig * np.sqrt(a_disp) / 1.11
            lw = 0.25*ms
            scatter_sizes = sizes_orig**2 * a_disp / 1.23

            if isinstance(artist, collections.PathCollection):
                if isinstance(scatter_sizes, float):
                    scatter_sizes = [scatter_sizes]
                artist.set_sizes(scatter_sizes)
            elif isinstance(artist, lines.Line2D):
                artist.set_markersize(ms)
                artist.set_linewidth(lw)
            elif isinstance(artist, collections.LineCollection):
                artist.set_linewidths(lw)
            else:
                raise NotImplementedError("rescale_sizes not implemented for artist-type: {}".format(type(artist)))

        return

    artist.axes.figure.canvas._af_callbacks.connect('update_sizes', artist, 'resize_event', callback_canvas)

    # we want to link to a zoom event on the axes.  In the case of a sidebar,
    # this should be the "parent" axes rather than the sidebar axes
    call.axes._backend_object._af_callbacks.connect('update_sizes', artist, 'xlim_changed', callback_axes)
    call.axes._backend_object._af_callbacks.connect('update_sizes', artist, 'ylim_changed', callback_axes)

    # make sure the sizes are updated to start.  This feels a bit like overkill,
    # but when saving to an image, we can't be sure that any of the above
    # signals have fired before saving.
    if run_callback:
        callback_axes(call.axes._backend_object)
