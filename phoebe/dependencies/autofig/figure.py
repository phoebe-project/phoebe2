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
    """
    """
    def __init__(self, *args, **kwargs):
        """
        Create a new <autofig.figure.Figure> object.

        Arguments
        -----------
        * `*args` (<autofig.call.Call> or <autofig.axes.Axes>, optional):
            positional arguments can be <autofig.call.Call> (<autofig.call.Plot>
            or <autofig.call.Mesh>) or <autofig.axes.Axes> objects to attach to
            the <autofig.figure.Figure>
        * `inline` (bool, optional, default=False): whether in inline mode

        Returns
        -----------
        * <autofig.figure.Figure>: the instantiated <autofig.figure.Figure> object.
        """
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
                raise TypeError("all arguments must be of type Call or Axes, found {}".format(type(ca)))

    def __repr__(self):
        naxes = len(self.axes)
        ncalls = len(self.calls)
        return "<autofig.figure.Figure | {} axes | {} call(s)>".format(naxes, ncalls)

    @classmethod
    def from_dict(cls, dict):
        args = []
        for axd in dict.pop('axes', []):
            args.append(_axes.Axes.from_dict(axd))
        for calld in dict.pop('calls', []):
            args.append(getattr(_call, calld.pop('classname', 'Plot')).from_dict(calld))

        if len(dict.items()):
            raise ValueError("could not recognize remaining content {}".format(dict))

        return cls(*args)

    def to_dict(self, renders=[]):
        """
        Export the current <autofig.figure.Figure> to a json-safe dictionary.

        See also:
        * <autofig.figure.Figure.save>

        Arguments
        -----------
        * `filename` (string): path to save the figure instance.
        * `renders` (list of dictionaries, default=[]): commands to execute
            for rendering when opened by the command-line tool or by passing
            `do_renders` to <autofig.figure.Figure.open>.  The format must
            be a list of dictionaries, where each dictionary must at least have
            'render': 'draw' or 'render': 'animate'.  Any additional key-value
            pairs will be passed as keyword arguments to the respective
            rendering method.


        Returns
        -----------
        * (str) the path of the saved figure instance.
        """
        renders_json_safe = []
        if len(renders):
            for render in renders:
                if render.get('render', None) not in ['draw', 'animate']:
                    raise ValueError("invalid format for render: {}.  Must include render='draw' or render='animate'".format(render))

                renders_json_safe.append({k: common._json_safe(v) for k,v in render.items()})

        return {'axes': [ax.to_dict() for ax in self.axes], 'calls': [c.to_dict() for c in self.calls], 'renders': renders_json_safe}

    @classmethod
    def open(cls, filename, do_renders=False, allow_renders_save=False):
        """
        Open a <autofig.figure.Figure> from a saved file.

        See also:
        * <autofig.figure.Figure.save>

        Arguments
        -----------
        * `filename` (string): path to the saved figure instance
        * `do_renders` (bool, default=False): whether to execute any render
            (ie. draw/animate) statements included in the file.
        * `allow_renders_save` (bool, default=False): whether to allow render
            statements to save images/animations to disk.  Be careful if setting
            this to True from an untrusted source.

        Returns
        ---------
        * the loaded <autofig.figure.Figure> instance.

        Raises
        ----------
        * ValueError: if `do_render` is True but the render statements are invalid.
        """
        dict = common.load(filename)
        renders = dict.pop('renders', [])
        fig = cls.from_dict(dict)

        if do_renders:
            for render in renders:
                render_cmd = render.pop('render', None)
                if render_cmd not in  ['draw', 'animate']:
                    raise ValueError("invalid format for renders, only accepts draw or animate.  Try passing do_renders=False to skip.")

                if not allow_renders_save:
                    save_dump = render.pop('save', None)

                if 'show' not in render.keys() and 'save' not in render.keys():
                    render['show'] = True

                print("calling {} with kwargs {}".format(render_cmd, render))
                getattr(fig, render_cmd)(**render)

        return fig

    def save(self, filename, renders=[]):
        """
        Save the current <autofig.figure.Figure>.  Note: this saves the autofig
        figure object itself, not the image.  To save the image, call
        <autofig.figure.Figure.draw> and pass `save`.

        See also:
        * <autofig.figure.Figure.open>
        * <autofig.figure.Figure.to_dict>

        Arguments
        -----------
        * `filename` (string): path to save the figure instance.
        * `renders` (list of dictionaries, default=[]): commands to execute
            for rendering when opened by the command-line tool or by passing
            `do_renders` to <autofig.figure.Figure.open>.  The format must
            be a list of dictionaries, where each dictionary must at least have
            'render': 'draw' or 'render': 'animate'.  Any additional key-value
            pairs will be passed as keyword arguments to the respective
            rendering method.


        Returns
        -----------
        * (str) the path of the saved figure instance.
        """
        common.save(self.to_dict(renders=renders), filename)


    @property
    def axes(self):
        """
        Access the children <autofig.axes.Axes> of the <autofig.figure.Figure>

        Returns
        ---------
        * <autofig.axes.AxesGroup> of <autofig.axes.Axes> ordered according to
            <autofig.axes.Axes.axorder>
        """
        axorders = [ax.axorder for ax in self._axes]
        axes = [self._axes[i] for i in np.argsort(axorders)]
        return _axes.AxesGroup(axes)

    def add_axes(self, *axes):
        """
        Add one or multiple <autofig.axes.Axes> to the <autofig.figure.Figure>

        Arguments
        -----------
        * `*args` (<autofig.axes.Axes>, optional): each positional argument must
            be an <autofig.axes.Axes> object.

        Returns
        -----------
        * None
        """
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
        """
        Access all children <autofig.call.Call>s of the <autofig.figure.Figure>.

        See also:

        * <autofig.figure.Figure.plots>
        * <autofig.figure.Figure.meshes>

        Returns
        -----------
        * <autofig.call.CallGroup> of <autofig.call.Call> objects
        """
        return _call.make_callgroup(self._calls)

    def add_call(self, *calls):
        """
        Add a <autofig.call.Call> to the <autofig.figure.Figure>

        Arguments
        ----------
        * `*calls`: positional arguments must each be of type <autofig.call.Call>

        Returns
        ---------
        * None
        """
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

    def _get_backend_object(self, fig=None, naxes=1):
        if fig is None:
            if self._backend_object:
                fig = self._backend_object
            else:
                fig = plt.gcf()
                fig.clf()

                rows, cols = _axes._determine_grid(naxes)
                fig.set_figwidth(8*cols)
                fig.set_figheight(6*rows)

                self._backend_artists = []

        self._backend_object = fig
        return fig

    def _get_backend_artists(self):
        return self._backend_artists

    @property
    def plots(self):
        """
        Access all children <autofig.call.Plot>s of the <autofig.figure.Figure>.

        See also:

        * <autofig.figure.Figure.calls>
        * <autofig.figure.Figure.fill_betweens>
        * <autofig.figure.Figure.meshes>

        Returns
        -------------
        * <autofig.call.PlotGroup> of all <autofig.call.Plot> objects
        """
        calls = [c for c in self._calls if isinstance(c, _call.Plot)]
        return _call.PlotGroup(calls)

    def plot(self, *args, **kwargs):
        """
        Add a new <autofig.call.Plot> to the <autofig.figure.Figure>.

        See also:

        * <autofig.call.Plot.__init__>

        Arguments
        ----------
        * `*args`: all positional arguments are passed on to
            <autofig.call.Plot.__init__> to initialize the new
            <autofig.call.Plot>.
        * `tight_layout` (bool, optional, default=True): passed to
            <autofig.figure.Figure.draw> if `show` or `save`.  Whether to draw
            with the `tight_layout` option.
        * `draw_title` (bool, optional, default=True): passed to
            <autofig.figure.Figure.draw> if `show` or `save`.  Whether to draw
            the title on the matplotlib axes.
        * `subplot_grid` (None or tuple, optional, default=None): passed to
            <autofig.figure.Figure.draw> if `show` or `save`.  Override the
            subplot locations.
        * `show` (bool, optional, default=False): whether to immediately
            draw and show the resulting matplotlib figure.  If True,
            <autofig.figure.Figure.draw> will be called.
        * `save` (False or string, optional, default=False): the filename
            to save the resulting matplotlib figure, or False to not save.
            If not False, <autofig.figure.Figure.draw> will be called.
        * `**kwargs`: additional keyword arguments are passed on to
            <autofig.call.Plot.__init__> to initialize the new
            <autofig.call.Plot>.
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
    def fill_betweens(self):
        """
        Access all children <autofig.call.FillBetween>s of the <autofig.figure.Figure>.

        See also:

        * <autofig.figure.Figure.calls>
        * <autofig.figure.Figure.plots>
        * <autofig.figure.Figure.meshes>

        Returns
        -------------
        * <autofig.call.CallGroup> of all <autofig.call.FillBetween> objects
        """
        calls = [c for c in self._calls if isinstance(c, _call.FillBetween)]
        return _call.CallGroup(calls)

    def fill_between(self, *args, **kwargs):
        """
        Add a new <autofig.call.FillBetween> to the <autofig.figure.Figure>.

        See also:

        * <autofig.call.FillBetween.__init__>

        Arguments
        ----------
        * `*args`: all positional arguments are passed on to
            <autofig.call.FillBetween.__init__> to initialize the new
            <autofig.call.FillBetween>.
        * `tight_layout` (bool, optional, default=True): passed to
            <autofig.figure.Figure.draw> if `show` or `save`.  Whether to draw
            with the `tight_layout` option.
        * `draw_title` (bool, optional, default=True): passed to
            <autofig.figure.Figure.draw> if `show` or `save`.  Whether to draw
            the title on the matplotlib axes.
        * `subplot_grid` (None or tuple, optional, default=None): passed to
            <autofig.figure.Figure.draw> if `show` or `save`.  Override the
            subplot locations.
        * `show` (bool, optional, default=False): whether to immediately
            draw and show the resulting matplotlib figure.  If True,
            <autofig.figure.Figure.draw> will be called.
        * `save` (False or string, optional, default=False): the filename
            to save the resulting matplotlib figure, or False to not save.
            If not False, <autofig.figure.Figure.draw> will be called.
        * `**kwargs`: additional keyword arguments are passed on to
            <autofig.call.FillBetween.__init__> to initialize the new
            <autofig.call.FillBetween>.
        """

        tight_layout = kwargs.pop('tight_layout', True)
        draw_sidebars = kwargs.pop('draw_sidebars', True)
        draw_title = kwargs.pop('draw_title', True)
        subplot_grid = kwargs.pop('subplot_grid', None)

        show = kwargs.pop('show', False)
        save = kwargs.pop('save', False)

        call = _call.FillBetween(*args, **kwargs)
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
        """
        Access all children <autofig.call.Mesh>es of the <autofig.figure.Figure>.

        See also:

        * <autofig.figure.Figure.calls>
        * <autofig.figure.Figure.plots>
        * <autofig.figure.Figure.fill_betweens>

        Returns
        -------------
        * <autofig.call.MeshGroup> of all <autofig.call.Mesh> objects.
        """
        calls = [c for c in self._calls if isinstance(c, _call.Mesh)]
        return _call.MeshGroup(calls)

    def mesh(self, *args, **kwargs):
        """
        Add a new <autofig.call.Mesh> to the <autofig.figure.Figure>.

        See also:

        * <autofig.call.Mesh.__init__>

        Arguments
        ----------
        * `*args`: all positional arguments are passed on to
            <autofig.call.Mesh.__init__> to initialize the new
            <autofig.call.Mesh>.
        * `tight_layout` (bool, optional, default=True): passed to
            <autofig.figure.Figure.draw> if `show` or `save`.  Whether to draw
            with the `tight_layout` option.
        * `draw_title` (bool, optional, default=True): passed to
            <autofig.figure.Figure.draw> if `show` or `save`.  Whether to draw
            the title on the matplotlib axes.
        * `subplot_grid` (None or tuple, optional, default=None): passed to
            <autofig.figure.Figure.draw> if `show` or `save`.  Override the
            subplot locations.
        * `show` (bool, optional, default=False): whether to immediately
            draw and show the resulting matplotlib figure.  If True,
            <autofig.figure.Figure.draw> will be called.
        * `save` (False or string, optional, default=False): the filename
            to save the resulting matplotlib figure, or False to not save.
            If not False, <autofig.figure.Figure.draw> will be called.
        * `**kwargs`: additional keyword arguments are passed on to
            <autofig.call.Mesh.__init__> to initialize the new
            <autofig.call.Plot>.
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
        """
        Clear the underlying matplotlib figure object.
        """
        # TODO: figure options like figsize, etc

        fig = self._get_backend_object(naxes=len(self.axes))
        fig.clf()

    def draw(self, fig=None, i=None, calls=None,
             tight_layout=True,
             draw_sidebars=True,
             draw_title=True,
             subplot_grid=None,
             show=False, save=False,
             save_afig=False,
             in_animation=False):
        """
        Draw the contents of the <autofig.figure.Figure> to a matplotlib figure
        object.

        See also:

        * <autofig.figure.Figure.animate>
        * <autofig.draw>
        * <autofig.axes.Axes.draw>
        * <autofig.call.Plot.draw>
        * <autofig.call.Mesh.draw>

        Arguments
        ------------
        * `fig` (matplotlib figure or None, optional, default=None): matplotlib
            figure instances to use during drawing.
        * `i` (float or None, optional, default=None): passed on to
            <autofig.axes.Axes.draw> for all <autofig.axes.Axes> in
            <autofig.figure.Figure.axes>.
        * `calls` (list of <autofig.call.Call> objects or None, optional, default=None):
            passed on to <autofig.axes.Axes.draw> for all <autofig.axes.Axes> in
            <autofig.figure.Figure.axes>.
        * `tight_layout` (bool, optional, default=True): whether to call
            `fig.tight_layout` after positioning all subplots, but before
            drawing any applicable sidebars.
        * `draw_sidebars` (bool, optional, default=True): whether to draw
            any applicable sidebars.
        * `draw_title` (bool, optional, default=True): passed on to
            <autofig.axes.Axes.draw> for all <autofig.axes.Axes> in
            <autofig.figure.Figure.axes>.  Whether to draw the title on the
            matplotlib axes.
        * `subplot_grid` (None or tuple, optional, default=None): Override the
            subplot locations.  Passed on to <autofig.axes.Axes.append_subplot>
            for each <autofig.axes.Axes> in <autofig.figure.Figure.axes>.
        * `show` (bool, optional, default=False): whether to immediately
            draw and show the resulting matplotlib figure.
        * `save` (False or string, optional, default=False): the filename
            to save the resulting matplotlib figure, or False to not save.
        * `save_afig` (False or string, optional, default=False): the filename
            to save the autofig object, along with the options for this
            draw call.  See also <autofig.figure.Figure.save>.
        * `in_animation` (bool, optional, default=False): whether the current
            call to `draw` is a single frame in an animation.  Usually this
            should not be changed by the user.  See <autofig.figure.Figure.animate>
            for creating animations.

        Returns
        ----------
        * ([matplotlib Figure](https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure)): the matplotlib figure object.
        """

        if save_afig:
            render = {'render': 'draw'}
            render['i'] = i
            render['tight_layout'] = tight_layout
            render['draw_sidebars'] = draw_sidebars
            render['draw_title'] = draw_title
            render['subplot_grid'] = subplot_grid

            self.save(save_afig, renders=[render])

        fig = self._get_backend_object(fig, naxes=len(self.axes))
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
                interval=100,
                animate_callback=None,
                show=False, save=False, save_kwargs={},
                save_afig=False):
        """
        Draw the contents of the <autofig.figure.Figure> to a matplotlib animation.

        See also:

        * <autofig.figure.Figure.plot>

        Arguments
        ------------
        * `fig` (matplotlib figure or None, optional, default=None): matplotlib
            figure instances to use during drawing.
        * `i` (list/array, **required**): iterable values for `i`.  Each item
            in the list/array will become a single frame in the resulting
            animation.
        * `tight_layout` (bool, optional, default=True): whether to call
            `fig.tight_layout` after positioning all subplots, but before
            drawing any applicable sidebars.
        * `draw_sidebars` (bool, optional, default=True): whether to draw
            any applicable sidebars.
        * `draw_title` (bool, optional, default=True): passed on to
            <autofig.axes.Axes.draw> for all <autofig.axes.Axes> in
            <autofig.figure.Figure.axes>.  Whether to draw the title on the
            matplotlib axes.
        * `subplot_grid` (None or tuple, optional, default=None): Override the
            subplot locations.  Passed on to <autofig.axes.Axes.append_subplot>
            for each <autofig.axes.Axes> in <autofig.figure.Figure.axes>.
        * `interval` (int, optional, default=100): time in ms between each
            frame in the animation.
        * `animate_callback` (callable, optional, default=None): Function which
            takes the matplotlib figure object and will be called at each frame
            within the animation.
        * `show` (bool, optional, default=False): whether to immediately
            draw and show the resulting matplotlib animation.
        * `save` (False or string, optional, default=False): the filename
            to save the resulting matplotlib animation, or False to not save.
        * `save_kwargs` (dict, optional, default={}): dictionary of keyword
            arguments to be passed on to [anim.save](https://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html#matplotlib.animation.FuncAnimation.save)
        * `save_afig` (False or string, optional, default=False): the filename
            to save the autofig object, along with the options for this
            animate call.  See also <autofig.figure.Figure.save>.

        Returns
        ----------
        * ([matplotlib FuncAnimation](https://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html#matplotlib-animation-funcanimation)): the matplotlib animation object.
        """

        if save_afig:
            render = {'render': 'animate'}
            render['i'] = i
            render['tight_layout'] = tight_layout
            render['draw_sidebars'] = draw_sidebars
            render['draw_title'] = draw_title
            render['subplot_grid'] = subplot_grid
            render['interval'] = interval

            self.save(save_afig, renders=[render])

        if tight_layout:
            print("WARNING: tight_layout with fixed limits may cause jittering in the animation")

        if i is None:
            # TODO: can we get i from the underlying Axes/Calls?
            raise NotImplementedError("must pass a list/array for i")

        if not hasattr(i, '__iter__'):
            raise ValueError("i must be iterable for animations")


        # time interval in ms between each frame
        interval = int(interval)

        # TODO: use blitting (probably only if no mesh calls)
        # https://matplotlib.org/3.1.0/api/animation_api.html#funcanimation
        blit = False

        ao = _mpl_animate.Animation(self,
                                    tight_layout=tight_layout,
                                    draw_sidebars=draw_sidebars,
                                    draw_title=draw_title,
                                    subplot_grid=subplot_grid,
                                    animate_callback=animate_callback)

        anim = animation.FuncAnimation(ao.mplfig, ao, fargs=(),\
                init_func=ao.anim_init, frames=i, interval=interval,\
                blit=blit)

        if save:
            try:
                anim.save(save, **save_kwargs)
            except ValueError as err:
                if str(err)=='I/O operation on closed file':
                    raise ValueError("saving animation failed (with message: {}). Try passing a valid option to 'write' via save_kwargs.  For example: save_kwargs={{'writer': 'imagemagick'}}".format(str(err)))
                else:
                    traceback.print_exc()

        if show:
            # TODO: allow top-level option for whether to block or not?
            if not common._inline:
                plt.show()  # <-- blocking
                # fig.show()  #<-- not blocking

        return anim
