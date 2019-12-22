from phoebe.parameters import *
from phoebe.parameters.unit_choices import unit_choices as _unit_choices
import numpy as np

if os.getenv('PHOEBE_ENABLE_PLOTTING', 'TRUE').upper() == 'TRUE':
    try:
        from phoebe.dependencies import autofig
    except (ImportError, TypeError):
        _use_autofig = False
    else:
        _use_autofig = True

        _mplcmaps = autofig.cyclers._mplcmaps
        _mplcolors = autofig.cyclers._mplcolors
        _mplmarkers = autofig.cyclers._mplmarkers
        _mpllinestyles = autofig.cyclers._mpllinestyles

else:
    _use_autofig = False


if not _use_autofig:
    # we still want some choices for the parameters

    # TODO: we need to have a way to update these choices if the parameters
    # were created when plotting disabled (or an older version of MPL) and then
    # run with plotting enabled (or a different version of MPL)

    # https://matplotlib.org/examples/color/colormaps_reference.html
    # unfortunately some colormaps have been added, but we want the nice ones
    # first.  So for defaults we'll use those defined in MPL 1.5.
    _mplcmaps = ['viridis', 'plasma', 'inferno', 'magma',
                     'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                        'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                        'hot', 'afmhot', 'gist_heat', 'copper',
                 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                        'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
                 'Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c',
                 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                        'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
                        'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

    _mplcolors = ['black', 'blue', 'red', 'green', 'purple']
    _mplmarkers = ['.', 'o', '+', 's', '*', 'v', '^', '<', '>', 'p', 'h', 'o', 'D']
    _mpllinestyles = ['solid', 'dashed', 'dotted', 'dashdot', 'None']

class MPLPropCycler(object):
    def __init__(self, prop, options=[]):
        self._prop = prop
        self._options_orig = options
        self._options = options
        self._used = []
        self._used_tmp = []

    def __repr__(self):
        return '<{}cycler | cycle: {} | used: {}>'.format(self._prop,
                                                          self.cycle,
                                                          self.used)


    @property
    def cycle(self):
        return self._options

    @cycle.setter
    def cycle(self, cycle):
        for option in cycle:
            if option not in self._options_orig:
                raise ValueError("invalid option: {}".format(option))
        self._options = cycle

    @property
    def used(self):
        return list(set(self._used + self._used_tmp))

    @property
    def next(self):
        for option in self.cycle:
            if option not in self.used:
                self.add_to_used(option)
                return option
        else:
            return self.cycle[-1]

    @property
    def next_tmp(self):
        for option in self.cycle:
            if option not in self.used:
                self.add_to_used_tmp(option)
                return option
        else:
            return self.cycle[-1]

    def get(self, option=None):
        if option is not None:
            self.add_to_used(option)
            return option
        else:
            return self.next

    def get_tmp(self, option=None):
        if option is not None:
            self.add_to_used_tmp(option)
            return option
        else:
            return self.next_tmp

    def check_validity(self, option):
        if option not in self._options_orig:
            raise ValueError("{} not one of {}".format(option, self._options_orig))

    def add_to_used(self, option):
        if option in [None, 'None', 'none', 'face']:
            return
        if option not in self._options_orig:
            return
            # raise ValueError("{} not one of {}".format(option, self._options_orig))
        if option not in self._used:
            self._used.append(option)

    def replace_used(self, oldoption, newoption):
        if newoption in [None, 'None', 'none']:
            return
        if newoption not in self._options_orig:
            raise ValueError("{} not one of {}".format(newoption, self._options_orig))

        if oldoption in self._used:
            ind = self._used.index(oldoption)
            self._used[ind] = newoption
        elif oldoption in self._used_tmp:
            # in many cases the old option may actually be in _used_tmp but
            # None will be passed because we don't have access to the previous
            # state of the color cycler.  But _used_tmp will be reset on the
            # next draw anyways, so this doesn't really hurt anything.
            self._used_tmp.remove(oldoption)
            self.add_to_used(newoption)
        else:
            self.add_to_used(newoption)


    def add_to_used_tmp(self, option):
        if option in [None, 'None', 'none']:
            return
        if option not in self._options_orig:
            raise ValueError("{} not one of {}".format(option, self._options_orig))
        if option not in self._used_tmp:
            self._used_tmp.append(option)

    def clear_tmp(self):
        self._used_tmp = []
        return



def _label_units_lims(axis, default_unit, visible_if=None, is_default=True, **kwargs):
    params = []

    if visible_if:
        # only apply kwargs if we match the choice
        kwargs = kwargs if (is_default or kwargs.get(visible_if.split(':')[0], None) == visible_if.split(':')[1]) else {}

    if is_default:
        params += [ChoiceParameter(qualifier='{}label_source'.format(axis), value=kwargs.get('{}label_source'.format(axis), 'auto'), choices=['auto', 'manual'], advanced=True, description='Whether to automatically or manually provide label for the {}-axis'.format(axis))]
        params += [StringParameter(qualifier='{}label'.format(axis), visible_if='{}label_source:manual'.format(axis), value=kwargs.get('{}label'.format(axis), '{}label'.format(axis)), advanced=True, description='Custom label for the {}-axis'.format(axis))]

        params += [ChoiceParameter(qualifier='{}unit_source'.format(axis), value=kwargs.get('{}unit_source'.format(axis), 'auto'), choices=['auto', 'manual'], advanced=True, description='Whether to automatically or manually set the {}-units.'.format(axis))]
        params += [ChoiceParameter(qualifier='{}lim_source'.format(axis), value=kwargs.get('{}lim_source'.format(axis), 'auto'), choices=['auto', 'manual'], advanced=True, description='Whether to automatically or manually set the {}-limits.'.format(axis))]


    visible_if_unit = '{}unit_source:manual'.format(axis)
    if visible_if is not None:
        visible_if_unit = visible_if_unit+','+visible_if
    params += [UnitParameter(qualifier='{}unit'.format(axis), visible_if=visible_if_unit, value=kwargs.get('{}unit'.format(axis), default_unit), choices=_unit_choices(default_unit), description='Unit for {}-axis'.format(axis))]

    visible_if_lim = '{}lim_source:manual'.format(axis)
    if visible_if is not None:
        visible_if_lim = visible_if_lim+','+visible_if
    params += [FloatArrayParameter(qualifier='{}lim'.format(axis), visible_if=visible_if_lim, value=kwargs.get('{}lim'.format(axis), []), default_unit=default_unit, description='Limit for the {}-axis'.format(axis))]

    return params

def _figure_style_sources(b, default_color='component', default_linestyle='dataset', default_marker='dataset', **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='color_source', value=kwargs.get('color_source', default_color), choices=['dataset', 'model', 'component', 'manual'], description='Source to use for color.  For manual, see the c parameter in the figure context.  Otherwise, see the c parameters tagged with the corresponding dataset/model/component.')]
    params += [ChoiceParameter(qualifier='color', visible_if='color_source:manual', value=b._mplcolorcycler.get(kwargs.get('color', None)), choices=b._mplcolorcycler.cycle, description='Default color when plotted via run_figure')]

    params += [ChoiceParameter(qualifier='marker_source', value=kwargs.get('marker_source', default_marker), choices=['dataset', 'component', 'manual'], description='Source to use for marker (datasets only, not models).  For manual, see the marker parameter in the figure context.  Otherwise, see the marker parameters tagged with the corresponding dataset/model/component.')]
    params += [ChoiceParameter(qualifier='marker', visible_if='marker_source:manual', value=b._mplmarkercycler.get(kwargs.get('marker', None)), choices=b._mplmarkercycler.cycle, description='Default marker (datasets only, not models) when plotted via run_figure')]

    params += [ChoiceParameter(qualifier='linestyle_source', value=kwargs.get('linestyle_source', default_linestyle), choices=['dataset', 'model', 'component', 'manual'], description='Source to use for linestyle.  For manual, see the linestyle parameter in the figure context.  Otherwise, see the linestyle parameters tagged with the corresponding dataset/model/component.')]
    params += [ChoiceParameter(qualifier='linestyle', visible_if='linestyle_source:manual', value=b._mpllinestylecycler.get(kwargs.get('linestyle', None)), choices=b._mpllinestylecycler.cycle, description='Default linestyle when plotted via run_figure')]

    return params

def _figure_uncover_highlight_animate(b, uncover=True, highlight=True, **kwargs):
    params = []

    if uncover:
        params += [BoolParameter(qualifier='uncover', value=kwargs.get('uncover', False), advanced=True, description='Whether to uncover up to the current time(s) (see times_source and times parameters).')]
    if highlight:
        params += [BoolParameter(qualifier='highlight', value=kwargs.get('highlight', True), advanced=True, description='Whether to higlight the current time(s) (see times_source and times parameters).')]
    params += [ChoiceParameter(qualifier='time_source', value=kwargs.get('time_source', 'default'), choices=['None', 'default', 'manual'], description='Source to use for highlight/uncover time for this individual figure (or set to default to respect the default_time_source parameter).')]
    params += [FloatParameter(qualifier='time', visible_if='time_source:manual', value=kwargs.get('time', 0.0), default_unit=u.d, description='Times to use for highlighting/uncovering if time_source=manual.')]

    return params

def _new_bundle(**kwargs):
    params = []

    params += [ChoiceParameter(qualifier='default_time_source', value=kwargs.get('default_time_source', 'None'), choices=['None', 'manual'], description='Source to use for highlight/uncover time for any figure in which time_source is set to default.')]
    params += [FloatParameter(qualifier='default_time', visible_if='default_time_source:manual', value=kwargs.get('default_time', 0.0), default_unit=u.d, description='Times to use for highlighting/uncovering if default_time_source=manual.')]

    return params

def _add_component(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='color', value=b._mplcolorcycler.get(kwargs.get('color', None)), choices=b._mplcolorcycler.cycle, description='Color to use for figures in which color_source is set to component')]
    params += [ChoiceParameter(qualifier='marker', value=b._mplmarkercycler.get(kwargs.get('marker', None)), choices=b._mplmarkercycler.cycle, description='Marker (datasets only, not models) to use for figures in which marker_source is set to component')]
    params += [ChoiceParameter(qualifier='linestyle', value=b._mpllinestylecycler.get(kwargs.get('linestyle', None)), choices=b._mpllinestylecycler.cycle, description='Linestyle to use for figures in which linestyle_source is set to component')]

    return ParameterSet(params)

def _add_dataset(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='color', value=b._mplcolorcycler.get(kwargs.get('color', None)), choices=b._mplcolorcycler.cycle, description='Color to use for figures in which color_source is set to dataset')]
    params += [ChoiceParameter(qualifier='marker', value=b._mplmarkercycler.get(kwargs.get('marker', None)), choices=b._mplmarkercycler.cycle, description='Marker (datasets only, not models) to use for figures in which marker_source is set to dataset')]
    params += [ChoiceParameter(qualifier='linestyle', value=b._mpllinestylecycler.get(kwargs.get('linestyle', 'solid' if _use_autofig else None)), choices=b._mpllinestylecycler.cycle, description='Linestyle to use for figures in which linestyle_source is set to dataset')]

    return ParameterSet(params)

def _run_compute(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='color', value=b._mplcolorcycler.get(kwargs.get('color', None)), choices=b._mplcolorcycler.cycle, description='Color to use for figures in which color_source is set to model')]
    # params += [ChoiceParameter(qualifier='marker', value=kwargs.get('marker', None), choices=b._mpl, description='Default marker when plotted, overrides dataset value unless set to <dataset>')]
    params += [ChoiceParameter(qualifier='linestyle', value=b._mpllinestylecycler.get(kwargs.get('linestyle', None)), choices=b._mpllinestylecycler.cycle, description='Linestyle to use for figures in which linestyle_source is set to model')]

    return ParameterSet(params)

# def subplots(b, **kwargs):
#     # enabling this would require moving all the exiting figure stuff into a 'subplot' context so that we can copy each of these for each subplot (instead of figure)
#     params = []
#
#     params += [BoolParameter(qualfier='include_subplot', copy_for={'figure', '*'}, figure='_default', value=kwargs.get('include_subplot', True))]
#
#     params += [ChoiceParameter(qualifier='axorder_source', visible_if='include_subplot:True', copy_for={'figure', '*'}, figure='_default', value=kwargs.get('axorder_source', 'auto'), choices=['auto', 'manual'])]
#     params += [IntParameter(qualifier='axorder', visible_if='include_subplot:True,axorder_source:manual', copy_for={'figure', '*'}, figure='_default', value=kwargs.get('axorder', 0), limits=[0,None])]
#
#     params += [ChoiceParameter(qualifier='axpos_source', visible_if='include_subplot:True', copy_for={'figure', '*'}, figure='_default', value=kwargs.get('axpos_source', 'auto'), choices=['auto', 'manual'])]
#     params += [IntParameter(qualifier='axpos', visible_if='include_subplot:True,axpos_source:manual', copy_for={'figure', '*'}, figure='_default', value=kwargs.get('axpos', 111), limits=[111,999])]


def lc(b, **kwargs):
    params = []

    # TODO: set hierarchy needs to update choices of any x,y,z in context='figure' to include phases:* a la pblum_ref

    params += [SelectParameter(qualifier='contexts', value=kwargs.get('contexts', '*'), choices=['dataset', 'model'], description='Contexts to include in the plot')]
    params += [SelectParameter(qualifier='datasets', value=kwargs.get('datasets', '*'), choices=[''], description='Datasets to include in the plot')]
    params += [SelectParameter(qualifier='models', value=kwargs.get('models', '*'), choices=[''], description='Models to include in the plot')]

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'times'), choices=['times', 'phases'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'fluxes'), choices=['fluxes', 'residuals'], description='Array to plot along y-axis')]

    params += _label_units_lims('x', visible_if='x:times', default_unit=u.d, is_default=True, **kwargs)
    params += _label_units_lims('x', visible_if='x:!times', default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)

    params += _label_units_lims('y', default_unit=u.W/u.m**2, is_default=True, **kwargs)

    kwargs.setdefault('color', 'black' if _use_autofig else None)
    kwargs.setdefault('marker', '.' if _use_autofig else None)
    kwargs.setdefault('linestyle', 'solid' if _use_autofig else None)
    params += _figure_style_sources(b, default_color='model', default_marker='manual', default_linestyle='manual', **kwargs)

    params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', True), advanced=True, description='Whether to draw the legend')]

    params += _figure_uncover_highlight_animate(b, **kwargs)

    return ParameterSet(params)


def rv(b, **kwargs):
    params = []

    params += [SelectParameter(qualifier='contexts', value=kwargs.get('contexts', '*'), choices=['dataset', 'model'], description='Contexts to include in the plot')]
    params += [SelectParameter(qualifier='datasets', value=kwargs.get('datasets', '*'), choices=[''], description='Datasets to include in the plot')]
    params += [SelectParameter(qualifier='models', value=kwargs.get('models', '*'), choices=[''], description='Models to include in the plot')]
    params += [SelectParameter(qualifier='components', value=kwargs.get('components', '*'), choices=[''], description='Components to include in the plot')]

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'times'), choices=['times', 'phases'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'rvs'), choices=['rvs', 'residuals'], description='Array to plot along y-axis')]

    params += _label_units_lims('x', visible_if='x:times', default_unit=u.d, is_default=True, **kwargs)
    params += _label_units_lims('x', visible_if='x:!times', default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)

    params += _label_units_lims('y', default_unit=u.km/u.s, is_default=True, **kwargs)

    kwargs.setdefault('color', 'black') if _use_autofig else None
    kwargs.setdefault('marker', '.' if _use_autofig else None)
    kwargs.setdefault('linestyle', 'solid' if _use_autofig else None)
    params += _figure_style_sources(b, default_color='component', default_marker='manual', default_linestyle='manual', **kwargs)

    params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', True), advanced=True, description='Whether to draw the legend')]

    params += _figure_uncover_highlight_animate(b, **kwargs)

    return ParameterSet(params)


# def etv(b, **kwargs):
#     params = []
#
#     params += [SelectParameter(qualifier='dataset', value=kwargs.get('dataset', '*'), choices=[''], description='Datasets to include in the plot')]
#     params += [SelectParameter(qualifier='model', value=kwargs.get('model', '*'), choices=[''], description='Models to include in the plot')]
#
#     params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'time_ephems'), choices=['time_ephems', 'Ns'], description='Array to plot along x-axis')]
#     params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'etvs'), choices=['etvs', 'time_ecls'], description='Array to plot along y-axis')]
#
#     params += _label_units_lims('x', visible_if='x:time_ephems', default_unit=u.d, is_default=True, **kwargs)
#     params += _label_units_lims('x', visible_if='x:Ns', default_unit=u.dimensionless_unscaled, **kwargs)
#
#     params += _label_units_lims('y', visible_if='y:etvs', default_unit=u.d, is_default=True, **kwargs)
#     params += _label_units_lims('y', visible_if='y:time_ecls', default_unit=u.d, **kwargs)
#
#     return ParameterSet(params)


def orb(b, **kwargs):
    params = []

    params += [SelectParameter(qualifier='datasets', value=kwargs.get('datasets', '*'), choices=[''], description='Datasets to include in the plot')]
    params += [SelectParameter(qualifier='models', value=kwargs.get('models', '*'), choices=[''], description='Models to include in the plot')]
    params += [SelectParameter(qualifier='components', value=kwargs.get('components', '*'), choices=[''], description='Components to include in the plot')]

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'us'), choices=['times', 'phases', 'us', 'vs', 'ws', 'vus', 'vvs', 'vws'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'ws'), choices=['us', 'vs', 'ws', 'vus', 'vvs', 'vws'], description='Array to plot along y-axis')]

    params += _label_units_lims('x', visible_if='x:times', default_unit=u.d, is_default=False, **kwargs)
    # TODO: this visible_if will likely fail if/once we implement phases:* for the choices for x
    params += _label_units_lims('x', visible_if='x:phases', default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)
    params += _label_units_lims('x', visible_if='x:us|vs|ws', default_unit=u.solRad, is_default=True, **kwargs)
    params += _label_units_lims('x', visible_if='x:vus|vvs|vws', default_unit=u.km/u.s, is_default=False, **kwargs)

    params += _label_units_lims('y', visible_if='y:us|vs|ws', default_unit=u.solRad, is_default=True, **kwargs)
    params += _label_units_lims('y', visible_if='y:vus|vvs|vws', default_unit=u.km/u.s, is_default=False, **kwargs)

    kwargs.setdefault('color', 'black' if _use_autofig else None)
    kwargs.setdefault('marker', '.' if _use_autofig else None)
    kwargs.setdefault('linestyle', 'solid' if _use_autofig else None)
    params += _figure_style_sources(b, default_color='component', default_marker='manual', default_linestyle='manual', **kwargs)

    params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', True), advanced=True, description='Whether to draw the legend')]

    params += _figure_uncover_highlight_animate(b, **kwargs)

    return ParameterSet(params)

def lp(b, **kwargs):
    params = []

    params += [SelectParameter(qualifier='contexts', value=kwargs.get('contexts', '*'), choices=['dataset', 'model'], description='Contexts to include in the plot')]
    params += [SelectParameter(qualifier='datasets', value=kwargs.get('datasets', '*'), choices=[''], description='Datasets to include in the plot')]
    params += [SelectParameter(qualifier='models', value=kwargs.get('models', '*'), choices=[''], description='Models to include in the plot')]
    # params += [SelectParameter(qualifier='times', value=kwargs.get('times', '*'), choices=[''], description='Times to include in the plot')]
    params += [SelectParameter(qualifier='components', value=kwargs.get('components', '*'), choices=[''], description='Components to include in the plot')]

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'wavelengths'), choices=['wavelengths'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'flux_densities'), choices=['flux_densities'], description='Array to plot along y-axis')]

    params += _label_units_lims('x', default_unit=u.nm, is_default=True, **kwargs)
    params += _label_units_lims('y', default_unit=u.W/(u.m**2*u.nm), is_default=True, **kwargs)

    kwargs.setdefault('color', 'black' if _use_autofig else None)
    kwargs.setdefault('marker', '.' if _use_autofig else None)
    kwargs.setdefault('linestyle', 'solid' if _use_autofig else None)
    params += _figure_style_sources(b, default_color='component', default_marker='manual', default_linestyle='manual', **kwargs)

    params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', True), advanced=True, description='Whether to draw the legend')]

    params += _figure_uncover_highlight_animate(b, **kwargs)

    return ParameterSet(params)

def mesh(b, **kwargs):
    params = []

    params += [SelectParameter(qualifier='datasets', value=kwargs.get('datasets', '*'), choices=[''], description='Datasets to include in the plot')]
    params += [SelectParameter(qualifier='models', value=kwargs.get('models', '*'), choices=[''], description='Models to include in the plot')]
    # params += [SelectParameter(qualifier='times', value=kwargs.get('times', '*'), choices=[''], description='Times to include in the plot')]
    params += [SelectParameter(qualifier='components', value=kwargs.get('components', '*'), choices=[''], description='Components to include in the plot')]

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'us'), choices=['us', 'vs', 'ws', 'xs', 'ys', 'zs'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'vs'), choices=['us', 'vs', 'ws', 'xs', 'ys', 'zs'], description='Array to plot along y-axis')]


    params += _label_units_lims('x', visible_if='x:us|vs|ws', default_unit=u.solRad, is_default=True, **kwargs)
    params += _label_units_lims('x', visible_if='x:xs|ys|zs', default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)

    params += _label_units_lims('y', visible_if='y:us|vs|ws', default_unit=u.solRad, is_default=True, **kwargs)
    params += _label_units_lims('y', visible_if='y:xs|ys|zs', default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)

    params += [ChoiceParameter(qualifier='fc_source', value=kwargs.get('fc_source', 'column'), choices=['column', 'manual', 'component', 'model'], description='Source to use for facecolor.  For column, see the fc_column parameter.  For manual, see the fc parameter.  Otherwise, see the color parameter tagged with the corresponding component/model')]
    params += [ChoiceParameter(qualifier='fc_column', visible_if='fc_source:column', value=kwargs.get('fc_column', 'None'), choices=['None'], description='Column from the mesh to plot as facecolor if fc_source is column.')]
    params += [ChoiceParameter(qualifier='fc', visible_if='fc_source:manual', value=kwargs.get('fc', 'None'), choices=['None']+_mplcolors, description='Color to use as facecolor if fc_source is manual.')]
    params += [ChoiceParameter(qualifier='fcmap_source', visible_if='fc_source:column', value=kwargs.get('fcmap_source', 'auto'), choices=['auto', 'manual'], description='Source to use for facecolor colormap.  To have fcmap adjust based on the value of fc, use auto.  For manual, see the fcmap parameter.')]
    params += [ChoiceParameter(qualifier='fcmap', visible_if='fc_source:column,fcmap_source:manual', value=kwargs.get('fcmap', 'viridis'), choices=_mplcmaps, description='Colormap to use for facecolor if fcmap_source=\'manual\'.')]

    params += [ChoiceParameter(qualifier='ec_source', value=kwargs.get('ec_source', 'manual'), choices=['column', 'manual', 'component', 'model', 'face'], description='Source to use for edgecolor.  For column, see the ec_column parameter.  For manual, see the ec parameter.  Otherwise, see the color parameter tagged with the corresponding component/model')]
    params += [ChoiceParameter(qualifier='ec_column', visible_if='ec_source:column', value=kwargs.get('ec_column', 'None'), choices=['None'], description='Column from the mesh to plot as edgecolor if ec_source is column.')]
    params += [ChoiceParameter(qualifier='ec', visible_if='ec_source:manual', value=kwargs.get('ec', 'black' if _use_autofig else None), choices=['face']+_mplcolors, description='Color to use as edgecolor if ec_source is manual.')]
    params += [ChoiceParameter(qualifier='ecmap_source', visible_if='ec_source:column', value=kwargs.get('ecmap_source', 'auto'), choices=['auto', 'manual'], description='Source to use for edgecolor colormap.  To have ecmap adjust based on the value of ec, use auto.  For manual, see the ecmap parameter.')]
    params += [ChoiceParameter(qualifier='ecmap', visible_if='ec_source:column,ecmap_source:manual', value=kwargs.get('fcmap', 'viridis'), choices=_mplcmaps, description='Colormap to use for edgecolor if ecmap_source=\'manual\'.')]




    for q in ['fc', 'ec']:
        # see dataset._mesh_columns
        # even though this isn't really the default case, we'll pass is_default=True
        # so that we get fc/eclim_source, etc, parameters
        params += _label_units_lims(q, visible_if='{}_source:column,{}_column:vxs|vys|vzs|vus|vws|vvs|rvs'.format(q,q), default_unit=u.km/u.s, is_default=True, **kwargs)
        params += _label_units_lims(q, visible_if='{}_source:column,{}_column:rs|rprojs'.format(q,q), default_unit=u.solRad, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if='{}_source:column,{}_column:areas'.format(q,q), default_unit=u.solRad**2, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if='{}_source:column,{}_column:teffs'.format(q,q), default_unit=u.K, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if='{}_source:column,{}_column:pblum_ext|abs_pblum_ext'.format(q,q), default_unit=u.W, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if='{}_source:column,{}_column:ptfarea'.format(q,q), default_unit=u.m, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if='{}_source:column,{}_column:intensities|normal_intensities|abs_intensities|abs_normal_intensities'.format(q,q), default_unit=u.W/u.m**3, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if='{}_source:column,{}_column:visibilities|mus|loggs|boost_factors|ldint'.format(q,q), default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)

    # TODO: legend=True currently fails
    params += [BoolParameter(qualifier='draw_sidebars', value=kwargs.get('draw_sidebars', True), advanced=True, description='Whether to draw the sidebars')]
    params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', False), advanced=True, description='Whether to draw the legend')]

    params += _figure_uncover_highlight_animate(b, uncover=False, highlight=False, **kwargs)

    return ParameterSet(params)

# del deepcopy
# del _add_component, _add_dataset, _label_units_lims, _run_compute
# del download_passband, list_installed_passbands, list_online_passbands, list_passbands, parameter_from_json, parse_json, send_if_client, update_if_client
# del fnmatch
