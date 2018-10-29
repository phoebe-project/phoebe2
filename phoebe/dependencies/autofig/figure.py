import traceback
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from matplotlib import animation

from . import common
from . import callbacks
from . import call as _call
from . import axes as _axes
from . import mpl_animate as _mpl_animate

class Figure(object):
    def __init__(self, *args, **kwargs):
        self._class = 'Figure' # just to avoid circular import in order to use isinstance

        self._backend_object = None
        self._backend_artists = []
        self._inline = kwargs.pop('inline', False)

        self._axes = []
        self._calls = []

        if len(kwargs.keys()):
            raise ValueError("kwargs: {} not recognized".format(kwargs.keys()))

        for ca in args:
            if isinstance(ca, _axes.Axes):
                self.add_axes(ca)
            elif isinstance(ca, _call.Call):
                self.add_call(ca)
            else:
                raise TypeError("all arguments must be of type Call or Axes")

    def __repr__(self):
        naxes = len(self.axes)
        ncalls = len(self.calls)
        return "<Figure | {} axes | {} call(s)>".format(naxes, ncalls)

    @property
    def axes(self):
        axorders = [ax.axorder for ax in self._axes]
        axes = [self._axes[i] for i in np.argsort(axorders)]
        return _axes.AxesGroup(axes)

    def add_axes(self, *axes):
        if len(axes) == 0:
            axes = [_axes.Axes()]

        if len(axes) > 1:
            for a in axes:
                self.add_axes(a)
            return

        elif len(axes) == 1:
            axes = axes[0]
            if not isinstance(axes, _axes.Axes):
                raise TypeError("axes must be of type Axes")

            axes._figure = self
            self._axes.append(axes)
            for call in axes.calls:
                self._calls.append(call)

    @property
    def calls(self):
        return _call.make_callgroup(self._calls)

    def add_call(self, *calls):
        if len(calls) > 1:
            for c in calls:
                self.add_call(c)
            return

        elif len(calls) == 1:
            call = calls[0]
            if not isinstance(call, _call.Call):
                raise TypeError("call must be of type Call")

            # try to add to existing axes in reverse order before making a new one
            for ax in reversed(self.axes):
                consistent, reason = ax.consistent_with_call(call)
                if consistent:
                    break
            else:
                # then no axes were consistent so we must add a new one
                ax = _axes.Axes()
                self.add_axes(ax)

            ax.add_call(call)
            self._calls.append(call)

    def _get_backend_object(self, fig=None):
        if fig is None:
            if self._backend_object:
                fig = self._backend_object
            else:
                fig = plt.gcf()
                fig.clf()
                self._backend_artists = []

        self._backend_object = fig
        return fig

    def _get_backend_artists(self):
        return self._backend_artists

    @property
    def plots(self):
        calls = [c for c in self._calls if isinstance(c, _call.Plot)]
        return _call.PlotGroup(calls)

    def plot(self, *args, **kwargs):
        """
        """

        tight_layout = kwargs.pop('tight_layout', True)
        draw_sidebars = kwargs.pop('draw_sidebars', True)
        draw_title = kwargs.pop('draw_title', True)
        subplot_grid = kwargs.pop('subplot_grid', None)

        show = kwargs.pop('show', False)
        save = kwargs.pop('save', False)

        call = _call.Plot(*args, **kwargs)
        self.add_call(call)

        if show or save:
            self.reset_draw()
            return self.draw(tight_layout=tight_layout,
                             draw_sidebars=draw_sidebars,
                             draw_title=draw_title,
                             subplot_grid=subplot_grid,
                             show=show, save=save)

    @property
    def meshes(self):
        calls = [c for c in self._calls if isinstance(c, _call.Mesh)]
        return _call.MeshGroup(calls)

    def mesh(self, *args, **kwargs):
        """
        """

        tight_layout = kwargs.pop('tight_layout', True)
        draw_sidebars = kwargs.pop('draw_sidebars', True)
        draw_title = kwargs.pop('draw_title', True)
        subplot_grid = kwargs.pop('subplot_grid', None)
        show = kwargs.pop('show', False)
        save = kwargs.pop('save', False)

        call = _call.Mesh(*args, **kwargs)
        self.add_call(call)
        if show or save:

            self.reset_draw()
            return self.draw(tight_layout=tight_layout,
                             draw_sidebars=draw_sidebars,
                             draw_title=draw_title,
                             subplot_grid=None,
                             show=show, save=save)

    # def show(self):
    #     plt.show()

    def reset_draw(self):
        # TODO: figure options like figsize, etc

        fig = self._get_backend_object()
        fig.clf()

    def draw(self, fig=None, i=None, calls=None,
             tight_layout=True,
             draw_sidebars=True,
             draw_title=True,
             subplot_grid=None,
             show=False, save=False,
             in_animation=False):

        fig = self._get_backend_object(fig)
        callbacks._connect_to_autofig(self, fig)
        callbacks._connect_to_autofig(self, fig.canvas)

        if calls is None:
            # then we need to reset the backend figure.  This is especially
            # important when passing draw(i=something)
            fig.clf()

        for axesi in self.axes:
            if axesi._backend_object not in fig.axes:
                # then axes doesn't have a subplot yet.  Adding one will also
                # shift the location of all axes already drawn/created.
                ax = axesi.append_subplot(fig=fig, subplot_grid=subplot_grid)
                # if axesi._backend_object already existed (but maybe on a
                # different figure) it will be reset on the draw call below.
            else:
                # then this axes already has a subplot on the figure, so we'll
                # allow it to default to that instance
                ax = None

            axesi.draw(ax=ax, i=i, calls=calls,
                       draw_sidebars=False,
                       draw_title=draw_title,
                       show=False, save=False, in_animation=in_animation)

            self._backend_artists += axesi._get_backend_artists()

        # must call tight_layout BEFORE adding any sidebars
        if tight_layout:
            fig.tight_layout()

        if draw_sidebars:
            for axesi in self.axes:
                axesi.draw_sidebars(i=i)

        if save:
            fig.savefig(save)

        if show:
            # TODO: allow top-level option for whether to block or not?
            if not common._inline:
                plt.show()  # <-- blocking
                # fig.show()  #<-- not blocking

        return fig

    def animate(self, fig=None, i=None,
                tight_layout=False,
                draw_sidebars=True,
                draw_title=True,
                subplot_grid=None,
                show=False, save=False, save_kwargs={}):

        if tight_layout:
            print("WARNING: tight_layout with fixed limits may cause jittering in the animation")

        if i is None:
            # TODO: can we get i from the underlying Axes/Calls?
            raise NotImplementedError("must pass a list/array for i")

        if not hasattr(i, '__iter__'):
            raise ValueError("i must be iterable for animations")


        interval = 100 # time interval in ms between each frame
        blit = False # TODO: set this to True if no Mesh calls?

        ao = _mpl_animate.Animation(self,
                                    tight_layout=tight_layout,
                                    draw_sidebars=draw_sidebars,
                                    draw_title=draw_title,
                                    subplot_grid=subplot_grid)

        anim = animation.FuncAnimation(ao.mplfig, ao, fargs=(),\
                init_func=ao.anim_init, frames=i, interval=interval,\
                blit=blit)

        if save:
            try:
                anim.save(save, **save_kwargs)
            except ValueError as err:
                if err.message=='I/O operation on closed file':
                    raise ValueError("saving animation failed (with message: {}). Try passing a valid option to 'write' via save_kwargs.  For example: save_kwargs={{'writer': 'imagemagick'}}".format(err.message))
                else:
                    traceback.print_exc()

        if show:
            # TODO: allow top-level option for whether to block or not?
            if not common._inline:
                plt.show()  # <-- blocking
                # fig.show()  #<-- not blocking

        return anim
