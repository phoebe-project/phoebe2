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


    _mplmarkers = ['.', 'o', '+', 's', '*', 'v', '^', '<', '>', 'p', 'h', 'o', 'D']
    _mpllinestyles = ['solid', 'dashed', 'dotted', 'dashdot']

# we don't want to use all the custom hex-codes within phoebe, let's just use a simple set of colors
_mplcolors = ['black', 'blue', 'orange', 'green', 'red', 'purple', 'ping', 'pink', 'yellow']

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

def _figure_style_sources(b, default_color='component', default_linestyle='dataset', default_marker='dataset', cycler='default', **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='color_source', value=kwargs.get('color_source', default_color), choices=['dataset', 'model', 'component', 'manual'], description='Source to use for color.  For manual, see the c parameter in the figure context.  Otherwise, see the c parameters tagged with the corresponding dataset/model/component.')]
    params += [ChoiceParameter(qualifier='color', visible_if='color_source:manual', value=b._mplcolorcyclers.get(cycler).get(kwargs.get('color', None)), choices=_mplcolors, description='Default color when plotted via run_figure')]

    params += [ChoiceParameter(qualifier='marker_source', value=kwargs.get('marker_source', default_marker), choices=['dataset', 'component', 'manual'], description='Source to use for marker (datasets only, not models).  For manual, see the marker parameter in the figure context.  Otherwise, see the marker parameters tagged with the corresponding dataset/model/component.')]
    params += [ChoiceParameter(qualifier='marker', visible_if='marker_source:manual', value=b._mplmarkercyclers.get(cycler).get(kwargs.get('marker', None)), choices=['None']+_mplmarkers, description='Default marker (datasets only, not models) when plotted via run_figure')]

    params += [ChoiceParameter(qualifier='linestyle_source', value=kwargs.get('linestyle_source', default_linestyle), choices=['dataset', 'model', 'component', 'manual'], description='Source to use for linestyle.  For manual, see the linestyle parameter in the figure context.  Otherwise, see the linestyle parameters tagged with the corresponding dataset/model/component.')]
    params += [ChoiceParameter(qualifier='linestyle', visible_if='linestyle_source:manual', value=b._mpllinestylecyclers.get(cycler).get(kwargs.get('linestyle', None)), choices=['None']+_mpllinestyles, description='Default linestyle when plotted via run_figure')]

    return params

def _figure_style_nosources(b, cycler='default', **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='color', value=b._mplcolorcyclers.get(cycler).get(kwargs.get('color', None)), choices=_mplcolors, description='Default color when plotted via run_figure')]

    params += [ChoiceParameter(qualifier='marker', value=b._mplmarkercyclers.get(cycler).get(kwargs.get('marker', None)) if kwargs.get('marker', None) != "None" else "None", choices=["None"] + _mplmarkers, description='Default marker when plotted via run_figure')]

    params += [ChoiceParameter(qualifier='linestyle', value=b._mpllinestylecyclers.get(cycler).get(kwargs.get('linestyle', None)) if kwargs.get('linestyle', None) != "None" else "None", choices=["None"] + _mpllinestyles, description='Default linestyle when plotted via run_figure')]

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

def _add_component(b, kind, **kwargs):
    params = []

    params += [StringParameter(qualifier='latex_repr', value='', description='Representation to use in place of the component label when rendering latex representations of parameters.  If blank, will use the labels directly.')]

    if kind in ['star']:
        params += [ChoiceParameter(qualifier='color', value=b._mplcolorcyclers.get('component').get(kwargs.get('color', None)), choices=_mplcolors, description='Color to use for figures in which color_source is set to component')]
        params += [ChoiceParameter(qualifier='marker', value=b._mplmarkercyclers.get('component').get(kwargs.get('marker', None)), choices=['None']+_mplmarkers, description='Marker (datasets only, not models) to use for figures in which marker_source is set to component')]
        params += [ChoiceParameter(qualifier='linestyle', value=b._mpllinestylecyclers.get('component').get(kwargs.get('linestyle', None)), choices=['None']+_mpllinestyles, description='Linestyle to use for figures in which linestyle_source is set to component')]

    return ParameterSet(params)

def _add_dataset(b, **kwargs):
    params = []

    params += [StringParameter(qualifier='latex_repr', value='', description='Representation to use in place of the dataset label when rendering latex representations of parameters.  If blank, will use the labels directly.')]

    params += [ChoiceParameter(qualifier='color', value=b._mplcolorcyclers.get('dataset').get(kwargs.get('color', None)), choices=_mplcolors, description='Color to use for figures in which color_source is set to dataset')]
    params += [ChoiceParameter(qualifier='marker', value=b._mplmarkercyclers.get('dataset').get(kwargs.get('marker', None)), choices=['None']+_mplmarkers, description='Marker (datasets only, not models) to use for figures in which marker_source is set to dataset')]
    params += [ChoiceParameter(qualifier='linestyle', value=b._mpllinestylecyclers.get('dataset').get(kwargs.get('linestyle', 'solid' if _use_autofig else None)), choices=['None']+_mpllinestyles, description='Linestyle to use for figures in which linestyle_source is set to dataset')]

    return ParameterSet(params)

def _run_compute(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='color', value=b._mplcolorcyclers.get('model').get(kwargs.get('color', None)), choices=_mplcolors, description='Color to use for figures in which color_source is set to model')]
    # params += [ChoiceParameter(qualifier='marker', value=kwargs.get('marker', None), choices=b._mpl, description='Default marker when plotted, overrides dataset value unless set to <dataset>')]
    params += [ChoiceParameter(qualifier='linestyle', value=b._mpllinestylecyclers.get('model').get(kwargs.get('linestyle', None)), choices=['None']+_mpllinestyles, description='Linestyle to use for figures in which linestyle_source is set to model')]

    return ParameterSet(params)
