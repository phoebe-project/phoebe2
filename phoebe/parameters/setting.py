
from phoebe.parameters import *
from phoebe import conf, __version__

def settings(**kwargs):
    """
    Generally, this will automatically be added to a newly initialized
    :class:`phoebe.frontend.bundle.Bundle`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """

    params = []

    params += [StringParameter(qualifier='phoebe_version', value=kwargs.get('phoebe_version', __version__), description='Version of PHOEBE - change with caution')]
    params += [BoolParameter(qualifier='log_history', value=kwargs.get('log_history', False), description='Whether to log history (undo/redo)')]
    params += [DictParameter(qualifier='dict_filter', value=kwargs.get('dict_filter', {}), description='Filters to use when using dictionary access')]
    params += [BoolParameter(qualifier='dict_set_all', value=kwargs.get('dict_set_all', False), description='Whether to set all values for dictionary access that returns more than 1 result')]

    params += [ChoiceParameter(qualifier='plotting_backend', value=kwargs.get('plotting_backend', 'mpl'), choices=['mpl', 'mpld3', 'mpl2bokeh', 'bokeh'] if conf.devel else ['mpl'], description='Default backend to use for plotting')]

    # problem with try_sympy parameter: it can't be used during initialization... so this may need to be a phoebe-level setting
    # params += [BoolParameter(qualifier='try_sympy', value=kwargs.get('try_sympy', True), description='Whether to use sympy if installed for constraints')]

    # This could be complicated - because then we'll have to specifically watch to see when its enabled and then run all constraints - not sure if that is worth the time savings
    # params += [BoolParameter(qualifier='run_constraints', value=kwargs.get('run_constraints', True), description='Whether to run_constraints whenever a parameter changes (warning: turning off will disable constraints until enabled at which point all constraints will be run)')]

    return ParameterSet(params)
