from phoebe.parameters import *
import numpy as np

try:
    from matplotlib import colors, markers
except (ImportError, TypeError):
    _use_mpl = False
    _mplcolors = []
    _mplmarkers = []
    _mpllinestyles = []
else:
    _use_mpl = True
    _mplcolors = ['k', 'b', 'r', 'g', 'p']
    _mplcolors = _mplcolors + [c for c in list(colors.ColorConverter.colors.keys()) + list(colors.cnames.keys()) if c not in _mplcolors and 'xkcd' not in c]
    _mplmarkers = ['.', 'o', '+', 's', '*', 'v', '^', '<', '>', 'p', 'h', 'o', 'D']
    # could do matplotlib.markers.MarkerStyle.markers.keys()
    _mpllinestyles = ['solid', 'dashed', 'dotted', 'dashdot', 'None']
    # could also do matplotlib.lines.lineStyles.keys()


class MPLPropCycler(object):
    def __init__(self, options=[]):
        self._options = options
        self._used = []
        self._index = 0

    @property
    def options(self):
        return self._options

    @property
    def next(self):
        for option in self._options:
            if option not in self._used:
                self.add_to_used(option)
                return option

    def get(self, option=None):
        if option is not None:
            self.add_to_used(option)
            return option
        else:
            return self.next


    def add_to_used(self, option):
        if option not in self._used and option not in ['<dataset>', '<component>']:
            self._used.append(option)

        if len(self._used) == len(self._options):
            # then start cycle over
            self._used = []



def _label_units_lims(axis, default_unit, visible_if=None, is_default=True, **kwargs):
    params = []

    if visible_if:
        # only apply kwargs if we match the choice
        kwargs = kwargs if (is_default or kwargs.get(visible_if.split(':')[0], None) == visible_if.split(':')[1]) else {}

    if is_default:
        params += [ChoiceParameter(qualifier='{}label_mode'.format(axis), value=kwargs.get('{}label_mode'.format(axis), 'auto'), choices=['auto', 'manual'], advanced=True, description='Whether to automatically or manually provide label for the {}-axis'.format(axis))]
        params += [StringParameter(qualifier='{}label'.format(axis), visible_if='{}label_mode:manual'.format(axis), value=kwargs.get('{}label'.format(axis), '{}label'.format(axis)), advanced=True, description='Custom label for the {}-axis'.format(axis))]

        params += [ChoiceParameter(qualifier='{}unit_mode'.format(axis), value=kwargs.get('{}unit_mode'.format(axis), 'auto'), choices=['auto', 'manual'], advanced=True, description='Whether to automatically or manually set the {}-units.'.format(axis))]
        params += [ChoiceParameter(qualifier='{}lim_mode'.format(axis), value=kwargs.get('{}lim_mode'.format(axis), 'auto'), choices=['auto', 'manual'], advanced=True, description='Whether to automatically or manually set the {}-limits.'.format(axis))]


    # TODO: change this to a ChoiceParameter and move logic of available units from phoebe-server into phoebe
    visible_if_unit = '{}unit_mode:manual'.format(axis)
    if visible_if is not None:
        visible_if_unit = visible_if_unit+','+visible_if
    params += [UnitParameter(qualifier='{}unit'.format(axis), visible_if=visible_if_unit, value=kwargs.get('{}unit'.format(axis), default_unit), description='Unit for {}-axis'.format(axis))]

    visible_if_lim = '{}lim_mode:manual'.format(axis)
    if visible_if is not None:
        visible_if_lim = visible_if_lim+','+visible_if
    params += [FloatArrayParameter(qualifier='{}lim'.format(axis), visible_if=visible_if_lim, value=kwargs.get('{}lim'.format(axis), []), default_unit=default_unit, description='Limit for the {}-axis'.format(axis))]

    return params

def _figure_style_modes(b, default_color='component', default_linestyle='dataset', default_marker='dataset', **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='color_mode', value=kwargs.get('color_mode', default_color), choices=['dataset', 'model', 'component', 'manual'], description='Source to use for color.  For manual, see the color parameter in the figure context.  Otherwise, see the color parameters tagged with the corresponding dataset/model/component.')]
    params += [ChoiceParameter(qualifier='color', visible_if='color_mode:manual', value=b._mplcolorcycler.get(kwargs.get('color', None)), choices=b._mplcolorcycler.options, description='Default color when plotted via run_figure')]

    params += [ChoiceParameter(qualifier='marker_mode', value=kwargs.get('marker_mode', default_marker), choices=['dataset', 'component', 'manual'], description='Source to use for marker (datasets only, not models).  For manual, see the marker parameter in the figure context.  Otherwise, see the marker parameters tagged with the corresponding dataset/model/component.')]
    params += [ChoiceParameter(qualifier='marker', visible_if='marker_mode:manual', value=b._mplmarkercycler.get(kwargs.get('marker', None)), choices=b._mplmarkercycler.options, description='Default marker (datasets only, not models) when plotted via run_figure')]

    params += [ChoiceParameter(qualifier='linestyle_mode', value=kwargs.get('linestyle_mode', default_linestyle), choices=['dataset', 'model', 'component', 'manual'], description='Source to use for linestyle.  For manual, see the linestyle parameter in the figure context.  Otherwise, see the linestyle parameters tagged with the corresponding dataset/model/component.')]
    params += [ChoiceParameter(qualifier='linestyle', visible_if='linestyle_mode:manual', value=b._mpllinestylecycler.get(kwargs.get('linestyle', None)), choices=b._mpllinestylecycler.options, description='Default linestyle when plotted via run_figure')]

    return params

def _figure_uncover_highlight_animate(b, **kwargs):
    params = []

    params += [BoolParameter(qualifier='uncover', value=kwargs.get('uncover', False), advanced=True, description='Whether to uncover up to the current time(s) (see times_mode and times parameters).')]
    params += [BoolParameter(qualifier='highlight', value=kwargs.get('highlight', True), advanced=True, description='Whether to higlight the current time(s) (see times_mode and times parameters).')]
    params += [ChoiceParameter(qualifier='time_source', value=kwargs.get('time_source', 'setting'), choices=['None', 'setting', 'manual'], description='Source to use for highlight/uncover time for this individual figure (or set to setting respect the figure_time_source parameter in the setting context).')]
    params += [FloatParameter(qualifier='time', visible_if='time_source:manual', value=kwargs.get('time', 0.0), default_unit=u.d, description='Times to use for highlighting/uncovering if time_source=manual.')]

    return params

def _add_component(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='color', value=b._mplcolorcycler.get(kwargs.get('color', None)), choices=b._mplcolorcycler.options, description='Color to use for figures in which color_mode is set to component')]
    params += [ChoiceParameter(qualifier='marker', value=b._mplmarkercycler.get(kwargs.get('marker', None)), choices=b._mplmarkercycler.options, description='Marker (datasets only, not models) to use for figures in which marker_mode is set to component')]
    params += [ChoiceParameter(qualifier='linestyle', value=b._mpllinestylecycler.get(kwargs.get('linestyle', None)), choices=b._mpllinestylecycler.options, description='Linestyle to use for figures in which linestyle_mode is set to component')]

    return ParameterSet(params)

def _add_dataset(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='color', value=b._mplcolorcycler.get(kwargs.get('color', None)), choices=b._mplcolorcycler.options, description='Color to use for figures in which linestyle_mode is set to dataset')]
    params += [ChoiceParameter(qualifier='marker', value=b._mplmarkercycler.get(kwargs.get('marker', None)), choices=b._mplmarkercycler.options, description='Marker (datasets only, not models) to use for figures in which marker_mode is set to dataset')]
    params += [ChoiceParameter(qualifier='linestyle', value=b._mpllinestylecycler.get(kwargs.get('linestyle', 'solid')), choices=b._mpllinestylecycler.options, description='Linestyle to use for figures in which linestyle_mode is set to dataset')]

    return ParameterSet(params)

def _run_compute(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='color', value=b._mplcolorcycler.get(kwargs.get('color', None)), choices=b._mplcolorcycler.options, description='Color to use for figures in which color_mode is set to model')]
    # params += [ChoiceParameter(qualifier='marker', value=kwargs.get('marker', None), choices=b._mpl, description='Default marker when plotted, overrides dataset value unless set to <dataset>')]
    params += [ChoiceParameter(qualifier='linestyle', value=b._mpllinestylecycler.get(kwargs.get('linestyle', None)), choices=b._mpllinestylecycler.options, description='Linestyle to use for figures in which linestyle_mode is set to model')]

    return ParameterSet(params)

# def subplots(b, **kwargs):
#     # enabling this would require moving all the exiting figure stuff into a 'subplot' context so that we can copy each of these for each subplot (instead of figure)
#     params = []
#
#     params += [BoolParameter(qualfier='include_subplot', copy_for={'figure', '*'}, figure='_default', value=kwargs.get('include_subplot', True))]
#
#     params += [ChoiceParameter(qualifier='axorder_mode', visible_if='include_subplot:True', copy_for={'figure', '*'}, figure='_default', value=kwargs.get('axorder_mode', 'auto'), choices=['auto', 'manual'])]
#     params += [IntParameter(qualifier='axorder', visible_if='include_subplot:True,axorder_mode:manual', copy_for={'figure', '*'}, figure='_default', value=kwargs.get('axorder', 0), limits=[0,None])]
#
#     params += [ChoiceParameter(qualifier='axpos_mode', visible_if='include_subplot:True', copy_for={'figure', '*'}, figure='_default', value=kwargs.get('axpos_mode', 'auto'), choices=['auto', 'manual'])]
#     params += [IntParameter(qualifier='axpos', visible_if='include_subplot:True,axpos_mode:manual', copy_for={'figure', '*'}, figure='_default', value=kwargs.get('axpos', 111), limits=[111,999])]


def lc(b, **kwargs):
    params = []

    # TODO: set hierarchy needs to update choices of any x,y,z in context='figure' to include phases:* a la pblum_ref

    params += [SelectParameter(qualifier='contexts', value=kwargs.get('contexts', '*'), choices=['dataset', 'model'], description='Contexts to include in the plot')]
    params += [SelectParameter(qualifier='datasets', value=kwargs.get('datasets', '*'), choices=[''], description='Datasets to include in the plot')]
    params += [SelectParameter(qualifier='models', value=kwargs.get('models', '*'), choices=[''], description='Models to include in the plot')]

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'times'), choices=['times', 'phases'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'fluxes'), choices=['fluxes', 'residuals'], description='Array to plot along y-axis')]

    params += _label_units_lims('x', visible_if='x:times', default_unit=u.d, is_default=True, **kwargs)
    params += _label_units_lims('x', visible_if='x:!times', default_unit=u.cycle, is_default=False, **kwargs)

    params += _label_units_lims('y', default_unit=u.W/u.m**2, is_default=True, **kwargs)

    kwargs.setdefault('color', 'k')
    kwargs.setdefault('marker', '.')
    kwargs.setdefault('linestyle', 'solid')
    params += _figure_style_modes(b, default_color='model', default_marker='manual', default_linestyle='manual', **kwargs)

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
    params += _label_units_lims('x', visible_if='x:!times', default_unit=u.cycle, is_default=False, **kwargs)

    params += _label_units_lims('y', default_unit=u.km/u.s, is_default=True, **kwargs)

    kwargs.setdefault('color', 'k')
    kwargs.setdefault('marker', '.')
    kwargs.setdefault('linestyle', 'solid')
    params += _figure_style_modes(b, default_color='component', default_marker='manual', default_linestyle='manual', **kwargs)

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
    params += _label_units_lims('x', visible_if='x:phases', default_unit=u.cycle, is_default=False, **kwargs)
    params += _label_units_lims('x', visible_if='x:us|vs|ws', default_unit=u.solRad, is_default=True, **kwargs)
    params += _label_units_lims('x', visible_if='x:vus|vvs|vws', default_unit=u.km/u.s, is_default=False, **kwargs)

    params += _label_units_lims('y', visible_if='y:us|vs|ws', default_unit=u.solRad, is_default=True, **kwargs)
    params += _label_units_lims('y', visible_if='y:vus|vvs|vws', default_unit=u.km/u.s, is_default=False, **kwargs)

    kwargs.setdefault('color', 'k')
    kwargs.setdefault('marker', '.')
    kwargs.setdefault('linestyle', 'solid')
    params += _figure_style_modes(b, default_color='component', default_marker='manual', default_linestyle='manual', **kwargs)

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

    kwargs.setdefault('color', 'k')
    kwargs.setdefault('marker', '.')
    kwargs.setdefault('linestyle', 'solid')
    params += _figure_style_modes(b, default_color='component', default_marker='manual', default_linestyle='manual', **kwargs)

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


    params += [ChoiceParameter(qualifier='fc', value=kwargs.get('fc', 'None'), choices=['None'], description='Array to plot as facecolor')]
    params += [ChoiceParameter(qualifier='ec', value=kwargs.get('ec', 'None'), choices=['None'], description='Array to plot as edgecolor')]


    for q in ['fc', 'ec']:
        # see dataset._mesh_columns
        # even though this isn't really the default case, we'll pass is_default=True
        # so that we get fc/eclim_mode, etc, parameters
        params += _label_units_lims(q, visible_if=q+':vxs|vys|vzs|vus|vws|vvs|rvs', default_unit=u.km/u.s, is_default=True, **kwargs)
        params += _label_units_lims(q, visible_if=q+':rs|rprojs', default_unit=u.solRad, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if=q+':areas', default_unit=u.solRad**2, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if=q+':teffs', default_unit=u.K, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if=q+':pblum_ext|abs_pblum_ext', default_unit=u.W, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if=q+':ptfarea', default_unit=u.m, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if=q+':intensities|normal_intensities|abs_intensities|abs_normal_intensities', default_unit=u.W/u.m**3, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if=q+':visibilities', default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if=q+':mus', default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if=q+':loggs', default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if=q+':boost_factors', default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if=q+':ldint', default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)

    # TODO: legend=True currently fails
    params += [BoolParameter(qualifier='draw_sidebars', value=kwargs.get('draw_sidebars', True), advanced=True, description='Whether to draw the sidebars')]
    params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', False), advanced=True, description='Whether to draw the legend')]

    params += _figure_uncover_highlight_animate(b, **kwargs)

    return ParameterSet(params)

# del deepcopy
# del _add_component, _add_dataset, _label_units_lims, _run_compute
# del download_passband, list_installed_passbands, list_online_passbands, list_passbands, parameter_from_json, parse_json, send_if_client, update_if_client
# del fnmatch
