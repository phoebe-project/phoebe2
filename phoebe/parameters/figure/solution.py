from phoebe.parameters import *
from phoebe.parameters.unit_choices import unit_choices as _unit_choices
import numpy as np

from .common import _use_autofig, _mplcmaps, _mplcolors, _mplmarkers, _mpllinestyles, MPLPropCycler, _label_units_lims, _figure_style_sources, _figure_style_nosources, _figure_uncover_highlight_animate

def lc_periodogram(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='solution', value=kwargs.get('solution', ''), choices=[''], description='Solution to include in the plot')]

    kwargs.setdefault('color', 'black' if _use_autofig else None)
    kwargs.setdefault('marker', 'None' if _use_autofig else None)
    kwargs.setdefault('linestyle', 'solid' if _use_autofig else None)
    params += _figure_style_nosources(b, **kwargs)

    # params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', True), advanced=True, description='Whether to draw the legend')]

    return ParameterSet(params)

def rv_periodogram(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='solution', value=kwargs.get('solution', ''), choices=[''], description='Solution to include in the plot')]

    kwargs.setdefault('color', 'black' if _use_autofig else None)
    kwargs.setdefault('marker', 'None' if _use_autofig else None)
    kwargs.setdefault('linestyle', 'solid' if _use_autofig else None)
    params += _figure_style_nosources(b, **kwargs)

    # params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', True), advanced=True, description='Whether to draw the legend')]

    return ParameterSet(params)

def lc_geometry(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='solution', value=kwargs.get('solution', ''), choices=[''], description='Solution to include in the plot')]

    # kwargs.setdefault('color', 'black' if _use_autofig else None)
    # kwargs.setdefault('marker', 'None' if _use_autofig else None)
    # kwargs.setdefault('linestyle', 'solid' if _use_autofig else None)
    # params += _figure_style_nosources(b, **kwargs)

    # params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', True), advanced=True, description='Whether to draw the legend')]

    return ParameterSet(params)

def rv_geometry(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='solution', value=kwargs.get('solution', ''), choices=[''], description='Solution to include in the plot')]

    # kwargs.setdefault('color', 'black' if _use_autofig else None)
    # kwargs.setdefault('marker', 'None' if _use_autofig else None)
    # kwargs.setdefault('linestyle', 'solid' if _use_autofig else None)
    # params += _figure_style_nosources(b, **kwargs)

    # params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', True), advanced=True, description='Whether to draw the legend')]

    return ParameterSet(params)

def ebai(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='solution', value=kwargs.get('solution', ''), choices=[''], description='Solution to include in the plot')]

    # kwargs.setdefault('color', 'black' if _use_autofig else None)
    # kwargs.setdefault('marker', 'None' if _use_autofig else None)
    # kwargs.setdefault('linestyle', 'solid' if _use_autofig else None)
    # params += _figure_style_nosources(b, **kwargs)

    # params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', True), advanced=True, description='Whether to draw the legend')]

    return ParameterSet(params)


def emcee(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='solution', value=kwargs.get('solution', ''), choices=[''], description='Solution to include in the plot')]

    params += [ChoiceParameter(qualifier='style', value=kwargs.get('style', 'lnprobability'), choices=['corner', 'failed', 'lnprobability', 'walks'], description='style of plot')]

    # TODO: implement y for walks (need to set choices based on the solution fitted_twigs)
    # params += [ChoiceParameter(visible_if='style:walks', qualifier='y', value=kwargs.get('y', ''), choices=[''], description='Parameter samples to plot along y-axis')]

    # params += _label_units_lims('y', visible_if='style:lnprobability', default_unit=u.dimensionless_unscaled, is_default=True, **kwargs)

    kwargs.setdefault('color', 'black' if _use_autofig else None)
    kwargs.setdefault('marker', 'None' if _use_autofig else None)
    kwargs.setdefault('linestyle', 'solid' if _use_autofig else None)
    params += _figure_style_nosources(b, **kwargs)

    # params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', True), advanced=True, description='Whether to draw the legend')]

    return ParameterSet(params)

def dynesty(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='solution', value=kwargs.get('solution', ''), choices=[''], description='Solution to include in the plot')]

    params += [ChoiceParameter(qualifier='style', value=kwargs.get('style', 'run'), choices=['corner', 'run', 'trace'], description='style of plot')]

    # params += _label_units_lims('y', visible_if='style:lnprobability', default_unit=u.dimensionless_unscaled, is_default=True, **kwargs)

    # kwargs.setdefault('color', 'black' if _use_autofig else None)
    # kwargs.setdefault('marker', 'None' if _use_autofig else None)
    # kwargs.setdefault('linestyle', 'solid' if _use_autofig else None)
    # params += _figure_style_nosources(b, **kwargs)

    # params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', True), advanced=True, description='Whether to draw the legend')]

    return ParameterSet(params)

# del deepcopy
# del _add_component, _add_dataset, _label_units_lims, _run_compute
# del download_passband, list_installed_passbands, list_online_passbands, list_passbands, parameter_from_json, parse_json, send_if_client, update_if_client
# del fnmatch
