import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.collections import LineCollection, PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from matplotlib import animation

except (ImportError, TypeError):
    _use_mpl = False
else:
    _use_mpl = True

from tempfile import NamedTemporaryFile


_latest_frame = None

VIDEO_TAG = """<video controls loop>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
    """
    adapted from: http://jakevdp.github.io/blog/2013/05/12/embedding-matplotlib-animations/

    This function converts and animation object from matplotlib into HTML which can then
    be embedded in an IPython notebook.

    This requires ffmpeg to be installed in order to build the intermediate mp4 file

    To get these to display automatically, you need to set animation.Animation._repr_html_ = plotlib.anim_to_html
    (this is done on your behalf by PHOEBE)
    """
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    
    return VIDEO_TAG.format(anim._encoded_video)


def reset_limits(ax, reset=True):
    """
    """
    if reset or not hasattr(ax, '_phoebe_xlim'):
        ax._phoebe_xlim = [np.inf, -np.inf]
    if reset or not hasattr(ax, '_phoebe_ylim'):
        ax._phoebe_ylim = [np.inf, -np.inf]
    if reset or not hasattr(ax, '_phoebe_zlim'):
        ax._phoebe_zlim = [np.inf, -np.inf]    

    return ax

def handle_limits(ax, xarray, yarray, zarray=None, reset=False, apply=False):
    """
    """

    ax = reset_limits(ax, reset=reset)

    if xarray is not None and len(xarray):
        if xarray.min() < ax._phoebe_xlim[0]:
            ax._phoebe_xlim[0] = xarray.min()
        if xarray.max() > ax._phoebe_xlim[1]:
            ax._phoebe_xlim[1] = xarray.max()
    if yarray is not None and len(yarray):
        if yarray.min() < ax._phoebe_ylim[0]:
            ax._phoebe_ylim[0] = yarray.min()
        if yarray.max() > ax._phoebe_ylim[1]:
            ax._phoebe_ylim[1] = yarray.max()
    if zarray is not None and len(zarray):
        if zarray.min() < ax._phoebe_zlim[0]:
            ax._phoebe_zlim[0] = zarray.min()
        if zarray.max() > ax._phoebe_zlim[1]:
            ax._phoebe_zlim[1] = zarray.max()

    if apply:
        ax = apply_limits(ax)

    return ax


def apply_limits(ax, pad=0.1):
    """
    apply the stored phoebe_limits to an axes, applying an additional padding

    :parameter ax:
    :parameter float pad: ratio of the range to apply as a padding (default: 0.1)
    """

    #try:
    if True:
        xlim = ax._phoebe_xlim
        ylim = ax._phoebe_ylim
        zlim = ax._phoebe_zlim
    #except AttributeError:
    #    return ax

    # initialize new lists for the padded limits.  We don't want to directly
    # edit xlim, ylim, zlim because we need padding based off the originals
    # and we don't want to have to worry about deepcopying issues
    xlim_pad = xlim[:]
    ylim_pad = ylim[:]
    zlim_pad = zlim[:]

    xlim_pad[0] = xlim[0] - pad*(xlim[1]-xlim[0])
    xlim_pad[1] = xlim[1] + pad*(xlim[1]-xlim[0])
    ylim_pad[0] = ylim[0] - pad*(ylim[1]-ylim[0])
    ylim_pad[1] = ylim[1] + pad*(ylim[1]-ylim[0])
    zlim_pad[0] = zlim[0] - pad*(zlim[1]-zlim[0])
    zlim_pad[1] = zlim[1] + pad*(zlim[1]-zlim[0])

    if isinstance(ax, Axes3D):
        ax.set_xlim3d(xlim_pad)
        ax.set_ylim3d(ylim_pad)
        ax.set_zlim3d(zlim_pad)
    else:
        ax.set_xlim(xlim_pad)
        ax.set_ylim(ylim_pad)

    return ax

def anim_set_data(artist, data, fixed_limits=True):
    """
    """
    # of course mpl has to be difficult and not allow set_data
    # on polycollections...
    ax = artist.axes
    if isinstance(artist, PolyCollection) or isinstance(artist, Poly3DCollection):
        ax.collections.remove(artist)

        if data is not None:
            pckwargs = {k:v for k,v in data.items() if k!='data'}
            data = data['data']
        else:
            data = []
            pckwargs = {}
          
        if len(data):
            xarray = np.array(data[:, :, 0])
            yarray = np.array(data[:, :, 1])
        else:
            xarray = []
            yarray = []

        if isinstance(artist, Poly3DCollection):
            if len(data):
                zarray = np.array(data[:, :, 2])
            else:
                zarray = []
            artist = Poly3DCollection(data, **pckwargs)
        else:
            zarray = None
            artist = PolyCollection(data, **pckwargs)

        ax.add_collection(artist)

        created = True
    else:             
        if data is None:
            # TODO: may need to be smart here to send the right shape,
            # especially for 3d axes
            data = ([], []) 
        artist.set_data(*data)

        created = False
        xarray = np.array(data[0])
        yarray = np.array(data[1])
        zarray = None # TODO: add support for 3d


    # TODO: need to be smarter about this - the user may have provided limits
    # in one of the plot_argss
    if not fixed_limits:
        ax = handle_limits(ax, xarray, yarray, zarray, apply=True)

    return artist, created


class Animation(object):
    def __init__(self, ps, time, fixed_limits, pa, metawargs={}, **kwargs):

        self.fixed_limits = fixed_limits
        self.latest_frame = None
        self.metawargs = metawargs

        # before we go in to the animation, we need to build
        # each of the artists and take care of all bookkeeping 
        self._mpl_artists = []
        self._mpl_artists_per_plotcall = []
        self.plot_argss = pa
        for pi,plot_args_ in enumerate(self.plot_argss):
            plot_args = plot_args_.copy()
            plot_args.setdefault('time', time)
            plot_args.setdefault('highlight', True)
            plot_args['do_plot'] = True
            ax, artists = ps.plot(**plot_args)
            
            # since we're doing the loop here, each call
            # should only be drawing to a single ax instance
            # (ax will still be a list, but all entries should
            # be identical, so we'll just take the first)
            ax = ax[0]

            # let's make sure future updates go to the same axes
            plot_args_['ax'] = ax

            if fixed_limits:
                # then set the initial limits to be the fixed limits
                # that should work over all frames of the animation
                ax = apply_limits(ax)

            for artist in artists:
                # we need a flattened list since each frame
                # needs to return a single list
                self._mpl_artists.append(artist)

            # we also need a list per plotcall so that we know
            # which artists are updated with which data arrays
            self._mpl_artists_per_plotcall.append(artists)

        self.fig = ax.figure

        try:
            self.fig.tight_layout()
        except ValueError:
            pass

    def anim_init(self):

        for plot_args, artists in zip(self.plot_argss, self._mpl_artists_per_plotcall):

            for j,artist in enumerate(artists):
                newartist, created = anim_set_data(artist, data=None, fixed_limits=self.fixed_limits)
                if created:
                    artists[j] = newartist # should update in _mpl_artists_per_plotcall
                    self._mpl_artists[self._mpl_artists.index(artist)] = newartist

        return self._mpl_artists

    def __call__(self, frame, base_ps):

        self.latest_frame = frame
        for plot_args, artists in zip(self.plot_argss, self._mpl_artists_per_plotcall):
            if isinstance(frame, float) or isinstance(frame, int):
                # then we're probably coming from the animate function, in
                # which case we're looping over times and want to pass that 
                # on the plotting call
                plot_args['time'] = frame
                ps = base_ps
            else:
                #print "*** frame", frame, type(frame)
                # then we're probably coming from plotting, in which case
                # "frame" is the current state of the ParameterSet of the 
                # synthetic model.  We want this ParameterSet to be included
                # when searching for arrays.
                ps_tmp, time = frame
                ps_tmp.set_meta(**self.metawargs)
                ps = base_ps + ps_tmp
                plot_args['time'] = time

            plot_args['do_plot'] = False
            data_per_artist = ps.plot(**plot_args)
            # TODO: need to get things like xerr, yerr??

            for j, (artist, data) in enumerate(zip(artists, data_per_artist)):  
                newartist, created = anim_set_data(artist, data, self.fixed_limits)
                if created:
                    artists[j] = newartist # should update in _mpl_artists_per_plotcall
                    self._mpl_artists[self._mpl_artists.index(artist)] = newartist

        return self._mpl_artists

def animate(base_ps, init_ps, init_time, frames, plotting_args, fixed_limits=True,\
             interval=100, blit=False, metawargs={}, **kwargs):


    ao = Animation(init_ps, init_time, fixed_limits, plotting_args, metawargs)
    anim = animation.FuncAnimation(ao.fig, ao, fargs=(base_ps,),\
            init_func=ao.anim_init, frames=frames, interval=interval,\
            blit=blit, **kwargs)

    return anim, ao


if _use_mpl:
    # setup hooks for inline animations in IPython notebooks
    try:
        from JSAnimation import IPython_display
    except:
        # built-in mp4 support
        animation.Animation._repr_html_ = anim_to_html
