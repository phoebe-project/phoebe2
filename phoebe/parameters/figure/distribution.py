from phoebe.parameters import *
#from phoebe.parameters.unit_choices import unit_choices as _unit_choices
#import numpy as np

#from .common import _use_autofig, _mplcmaps, _mplcolors, _mplmarkers, _mpllinestyles, MPLPropCycler, _label_units_lims, _figure_style_sources, _figure_style_nosources, _figure_uncover_highlight_animate

def distribution_collection(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='distribution_set', value=kwargs.get('distribution_set', 'manual'), choices=['manual'], description='Set of distributions to include in the plot.  Use \'manual\' to explicitly select distributions and combining logic')]
    params += [SelectParameter(visible_if='distribution_set:manual', qualifier='distributions', value=kwargs.get('distributions', []), choices=[''], description='Distributions to include in the plot')]
    params += [ChoiceParameter(visible_if='distribution_set:manual', qualifier='combine', value=kwargs.get('combine', 'and'), choices=['and', 'or', 'first'], description='How to combine multiple distributions for the same parameter.')]
    params += [BoolParameter(visible_if='distribution_set:manual', qualifier='include_constrained', value=kwargs.get('include_constrained', False), description='Whether to include constrained parameters')]
    params += [BoolParameter(visible_if='distribution_set:manual', qualifier='to_uniforms', value=kwargs.get('to_uniforms', False), description='Wheter to convert all distribution to univariate uniforms')]
    params += [IntParameter(visible_if='distribution_set:manual,to_uniforms:True', qualifier='to_uniforms_sigma', value=kwargs.get('to_uniforms_sigma', 1), description='Sigma to adopt for non-uniform distributions')]
    params += [BoolParameter(visible_if='distribution_set:manual,to_uniforms:False', qualifier='to_univariates', value=kwargs.get('to_univariates', False), description='Whether to convert any multivariate distribution to univaritates before adding to the collection')]

    params += [IntParameter(qualifier='titles_sigma', value=kwargs.get('titles_sigma', 1), limits=[0, None], description='Sigma to adopt for uncertainties in axes titles (or 0 to disable).')]

    return ParameterSet(params)
