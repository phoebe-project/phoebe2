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
    _mplcolors = _mplcolors + [c for c in colors.ColorConverter.colors.keys() + colors.cnames.keys() if c not in _mplcolors]
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

mplcolorcycler = MPLPropCycler(_mplcolors)
mplmarkercycler = MPLPropCycler(_mplmarkers)
mpllinestylecycler = MPLPropCycler(_mpllinestyles)

def _label_units_lims(axis, default_unit, visible_if=None, is_default=False, **kwargs):
    params = []

    if visible_if:
        # only apply kwargs if we match the choice
        kwargs = kwargs if (is_default or kwargs.get(visible_if.split(':')[0], None) == visible_if.split(':')[1]) else {}

    params += [StringParameter(qualifier='{}label'.format(axis), visible_if=visible_if, value=kwargs.get('{}label'.format(axis), '<auto>'), description='Label for the {}-axis (or <auto> to base on {} and {}unit)'.format(axis, axis, axis))]
    params += [UnitParameter(qualifier='{}unit'.format(axis), visible_if=visible_if, value=kwargs.get('{}unit'.format(axis), default_unit), description='Unit for {}-axis'.format(axis))]
    params += [FloatArrayParameter(qualifier='{}lim'.format(axis), visible_if=visible_if, value=kwargs.get('{}lim'.format(axis), []), default_unit=default_unit, description='Limit for the {}-axis'.format(axis))]

    return params

def _add_component(**kwargs):
    params = []

    params += [ChoiceParameter(qualifier='color', value=mplcolorcycler.get(kwargs.get('color', None)), choices=mplcolorcycler.options, description='Default color when plotted via run_figure')]
    params += [ChoiceParameter(qualifier='marker', value=mplmarkercycler.get(kwargs.get('marker', None)), choices=mplmarkercycler.options, description='Default marker when plotted via run_figure')]
    params += [ChoiceParameter(qualifier='linestyle', value=mpllinestylecycler.get(kwargs.get('linestyle', None)), choices=mpllinestylecycler.options, description='Default linestyle when plotted via run_figure')]

    return ParameterSet(params)

def _add_dataset(allow_per_component=False, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='color', value=mplcolorcycler.get(kwargs.get('color', '<component>' if allow_per_component else None)), choices=['<component>']+mplcolorcycler.options if allow_per_component else mplcolorcycler.options, description='Default color when plotted, overrides component value unless set to <component>')]
    params += [ChoiceParameter(qualifier='marker', value=mplmarkercycler.get(kwargs.get('marker', '<component>' if allow_per_component else None)), choices=['<component>']+mplmarkercycler.options if allow_per_component else mplmarkercycler.options, description='Default marker when plotted, overrides component value unless set to <component>')]
    params += [ChoiceParameter(qualifier='linestyle', value=mpllinestylecycler.get(kwargs.get('linestyle', 'solid')), choices=['<component>']+mpllinestylecycler.options if allow_per_component else mpllinestylecycler.options, description='Default linestyle when plotted, overrides component value unless set to <component>')]

    return ParameterSet(params)

def _run_compute(**kwargs):
    params = []

    params += [ChoiceParameter(qualifier='color', value=mplcolorcycler.get(kwargs.get('color', '<dataset>')), choices=['<dataset>']+mplcolorcycler.options, description='Default color when plotted, overrides dataset value unless set to <dataset>')]
    # params += [ChoiceParameter(qualifier='marker', value=kwargs.get('marker', '.'), choices=['.', 'o', '+'], description='Default marker when plotted, overrides dataset value unless set to <dataset>')]
    params += [ChoiceParameter(qualifier='linestyle', value=mpllinestylecycler.get(kwargs.get('linestyle', '<dataset>')), choices=['<dataset>']+mpllinestylecycler.options, description='Default linestyle when plotted, overrides dataset value unless set to <dataset>')]

    return ParameterSet(params)

def lc(**kwargs):
    params = []

    # TODO: set hierarchy needs to update choices of any x,y,z in context='figure' to include phases:* a la pblum_ref

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'times'), choices=['times', 'phases'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'fluxes'), choices=['fluxes'], description='Array to plot along y-axis')]

    params += _label_units_lims('x', visible_if='x:times', default_unit=u.d, is_default=True, **kwargs)
    params += _label_units_lims('x', visible_if='x:!times', default_unit=u.cycle, **kwargs)

    params += _label_units_lims('y', default_unit=u.W/u.m**2, is_default=True, **kwargs)

    return ParameterSet(params)


def rv(**kwargs):
    params = []

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'times'), choices=['times', 'phases'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'rvs'), choices=['rvs'], description='Array to plot along y-axis')]

    params += _label_units_lims('x', visible_if='x:times', default_unit=u.d, is_default=True, **kwargs)
    params += _label_units_lims('x', visible_if='x:!times', default_unit=u.cycle, **kwargs)

    params += _label_units_lims('y', default_unit=u.km/u.s, is_default=True, **kwargs)

    return ParameterSet(params)


def etv(**kwargs):
    params = []

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'time_ephems'), choices=['time_ephems', 'Ns'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'etvs'), choices=['etvs', 'time_ecls'], description='Array to plot along y-axis')]

    params += _label_units_lims('x', visible_if='x:time_ephems', default_unit=u.d, is_default=True, **kwargs)
    params += _label_units_lims('x', visible_if='x:Ns', default_unit=u.dimensionless_unscaled, **kwargs)

    params += _label_units_lims('y', visible_if='y:etvs', default_unit=u.d, is_default=True, **kwargs)
    params += _label_units_lims('y', visible_if='y:time_ecls', default_unit=u.d, **kwargs)

    return ParameterSet(params)


def ifm(**kwargs):
    raise NotImplementedError

def orb(**kwargs):
    params = []

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'xs'), choices=['times', 'phases', 'xs', 'ys', 'zs', 'vxs', 'vys', 'vzs'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'zs'), choices=['xs', 'ys', 'zs', 'vxs', 'vys', 'vzs'], description='Array to plot along y-axis')]

    params += _label_units_lims('x', visible_if='x:times', default_unit=u.d, **kwargs)
    params += _label_units_lims('x', visible_if='x:phases', default_unit=u.cycle, **kwargs)  # TODO: this will fail once we implement phases:*
    params += _label_units_lims('x', visible_if='x:xs', default_unit=u.solRad, is_default=True, **kwargs)
    params += _label_units_lims('x', visible_if='x:ys', default_unit=u.solRad, **kwargs)
    params += _label_units_lims('x', visible_if='x:zs', default_unit=u.solRad, **kwargs)
    params += _label_units_lims('x', visible_if='x:vxs', default_unit=u.km/u.s, **kwargs)
    params += _label_units_lims('x', visible_if='x:vys', default_unit=u.km/u.s, **kwargs)
    params += _label_units_lims('x', visible_if='x:vzs', default_unit=u.km/u.s, **kwargs)

    params += _label_units_lims('y', visible_if='y:xs', default_unit=u.solRad, **kwargs)
    params += _label_units_lims('y', visible_if='y:ys', default_unit=u.solRad, **kwargs)
    params += _label_units_lims('y', visible_if='y:zs', default_unit=u.solRad, is_default=True, **kwargs)
    params += _label_units_lims('y', visible_if='y:vxs', default_unit=u.km/u.s, **kwargs)
    params += _label_units_lims('y', visible_if='y:vys', default_unit=u.km/u.s, **kwargs)
    params += _label_units_lims('y', visible_if='y:vzs', default_unit=u.km/u.s, **kwargs)


    return ParameterSet(params)

def mesh(**kwargs):
    raise NotImplementedError

    params = []

    # ChoiceParameter for dataset that isn't changeable???
    # how do we know which columns are available for facecolor/edgecolor unless we already know the dataset - but then mesh figures will work differently than all others

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'xs'), choices=['xs', 'ys', 'zs'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'ys'), choices=['xs', 'ys', 'zs'], description='Array to plot along y-axis')]
    params += [ChoiceParameter(qualifier='facecolor', value=kwargs.get('facecolor', 'none'), choices=['none'], description='Array to plot as facecolor')]
    params += [ChoiceParameter(qualifier='edgecolor', value=kwargs.get('edgecolor', 'none'), choices=['none'], description='Array to plot as edgecolor')]

    params += _labels('x', **kwargs)
    params += _labels('y', **kwargs)

    params += _units_lims('x', visible_if='x:xs', default_unit=u.solRad, is_default=True, **kwargs)
    params += _units_lims('x', visible_if='x:ys', default_unit=u.solRad, **kwargs)
    params += _units_lims('x', visible_if='x:zs', default_unit=u.solRad, **kwargs)

    params += _units_lims('y', visible_if='y:xs', default_unit=u.solRad, **kwargs)
    params += _units_lims('y', visible_if='y:ys', default_unit=u.solRad, is_default=True, **kwargs)
    params += _units_lims('y', visible_if='y:zs', default_unit=u.solRad, **kwargs)

    params += _units_lim('facecolor', visible_if='facecolor:teffs', default_unit=u.K, **kwargs)


    return ParameterSet(params)
