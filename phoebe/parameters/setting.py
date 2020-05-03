
from phoebe.parameters import *
from phoebe import conf, __version__

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

def settings(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for bundle-level settings.

    Generally, this will automatically be added to a newly initialized
    <phoebe.frontend.bundle.Bundle>

    Arguments
    ----------
    * `log_history` (bool, optional, default=False): whether to log history (undo/redo)
        NOTE: this is experimental and not fully-functional.
    * `dict_filter` (dictionary, optional, default={}): filter to use when using
        dictionary access in the bundle.
    * `dict_set_all` (bool, optional, default=False): whether to set all values
        for dictionary access that returns more than 1 results.
    * `run_checks_compute` (list or string, optional, default='*'): Compute
        options to use when calling run_checks/run_checks_compute or within
        interactive checks.
    * `run_checks_solver` (list or string, optional, default='*'): Solver
        options to use when calling run_checks/run_checks_solver or within
        interactive checks.
    * `run_checks_solution` (list or string, optional, default='*'): Solutions
        to use when calling run_checks/run_checks_solution or within
        interactive checks.
    * `run_checks_figure` (list or string, optional, default='*'): Figures
        to use when calling run_checks/run_checks_figure or within
        interactive checks.
    * `auto_add_figure` (bool, optional, default=False): Whether to automatically
        add figure parameters when a dataset is added with a new dataset type,
        or a solution is added.
    * `auto_remove_figure` (bool, optional, default=False): Whether to
        automatically remove figure parameters when the referenced
        dataset/solution are removed.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """

    params = []

    params += [StringParameter(qualifier='phoebe_version', value=kwargs.get('phoebe_version', __version__), advanced=True, readonly=True, description='Version of PHOEBE')]
    params += [BoolParameter(qualifier='log_history', value=kwargs.get('log_history', False), advanced=True, description='Whether to log history (undo/redo)')]
    params += [DictParameter(qualifier='dict_filter', value=kwargs.get('dict_filter', {}), advanced=True, description='Filters to use when using dictionary access')]
    params += [BoolParameter(qualifier='dict_set_all', value=kwargs.get('dict_set_all', False), advanced=True, description='Whether to set all values for dictionary access that returns more than 1 result')]

    params += [SelectParameter(qualifier='run_checks_compute', value=kwargs.get('run_checks_compute', '*'), choices=[], advanced=False, description='Compute options to use when calling run_checks/run_checks_compute or within interactive checks.')]
    params += [SelectParameter(qualifier='run_checks_solver', value=kwargs.get('run_checks_solver', '*'), choices=[], advanced=False, description='Solver options to use when calling run_checks/run_checks_solver or within interactive checks.')]
    params += [SelectParameter(qualifier='run_checks_solution', value=kwargs.get('run_checks_solution', []), choices=[], advanced=False, description='Solutions to use when calling run_checks/run_checks_solution or within interactive checks.')]
    params += [SelectParameter(qualifier='run_checks_figure', value=kwargs.get('run_checks_figure', []), choices=[], advanced=False, description='Figures to use when calling run_checks/run_checks_figure or within interactive checks.')]

    params += [BoolParameter(qualifier='auto_add_figure', value=kwargs.get('auto_add_figure', False), description='Whether to automatically add figure parameters when a dataset is added with a new dataset type, or a solution is added.')]
    params += [BoolParameter(qualifier='auto_remove_figure', value=kwargs.get('auto_remove_figure', False), description='Whether to automatically remove figure parameters when the referenced dataset/solution are removed.')]

    return ParameterSet(params)
