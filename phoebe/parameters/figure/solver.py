from phoebe.parameters import *
#from phoebe.parameters.unit_choices import unit_choices as _unit_choices
#import numpy as np

#from .common import _use_autofig, _mplcmaps, _mplcolors, _mplmarkers, _mpllinestyles, MPLPropCycler, _label_units_lims, _figure_style_sources, _figure_style_nosources, _figure_uncover_highlight_animate

def emcee(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='solver', value=kwargs.get('solver', ''), choices=[kwargs.get('solver', '')], description='Solver to include in the plot')]
    params += [ChoiceParameter(qualifier='distribution', value=kwargs.get('distribution', 'init_from'), choices=['init_from', 'priors'], description='Distribution collection to plot')]

    return ParameterSet(params)

def dynesty(b, **kwargs):
    params = []

    params += [ChoiceParameter(qualifier='solver', value=kwargs.get('solver', ''), choices=[kwargs.get('solver', '')], description='Solver to include in the plot')]
    params += [ChoiceParameter(qualifier='distribution', value=kwargs.get('distribution', 'priors'), choices=['priors'], description='Distribution collection to plot')]

    return ParameterSet(params)
