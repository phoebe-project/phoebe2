
from phoebe.parameters import *
from phoebe import conf

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

def emcee(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    emcee backend.  To use this backend, you must have emcee 3.0+ installed.

    To install emcee, see https://emcee.readthedocs.io

    If using this backend for solver, consider citing:
    * https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F

    See also:
    * <phoebe.frontend.bundle.Bundle.references>

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_solver>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_solver>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_solver('sampler.emcee')
    b.run_solver(kind='emcee')
    ```

    Arguments
    ----------

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = []

    params += [ChoiceParameter(qualifier='compute', value=kwargs.get('compute', 'None'), choices=['None'], description='compute options to use for forward model')]

    params += [SelectParameter(qualifier='init_from', value=kwargs.get('init_from', []), choices=[], description='distribution(s) to initialize samples from (all UNCONSTRAINED parameters with attached distributions will be sampled/fitted)')]
    params += [ChoiceParameter(qualifier='init_from_combine', value=kwargs.get('init_from_combine', 'first'), choices=['first'], description='Method to use to combine multiple distributions from init_from for the same parameter.  first: ignore duplicate entries and take the first in the priors parameter.')]

    params += [SelectParameter(qualifier='priors', value=kwargs.get('priors', []), choices=[], description='distribution(s) to use for priors')]
    params += [ChoiceParameter(qualifier='priors_combine', value=kwargs.get('priors_combine', 'and'), choices=['and'], description='Method to use to combine multiple distributions from priors for the same parameter.  first: ignore duplicate entries and take the first in the priors parameter.')]

    params += [IntParameter(qualifier='nwalkers', value=kwargs.get('nwalkers', 16), limits=(1,1e5), description='number of walkers')]
    params += [IntParameter(qualifier='niters', value=kwargs.get('niters', 100), limits=(1,1e12), description='number of iterations')]

    params += [StringParameter(qualifier='filename', value=kwargs.get('filename', 'emcee_progress.hd5'), description='filename to use for storing progress and continuing from previous run')]
    params += [BoolParameter(qualifier='continue_previous_run', value=kwargs.get('continue_previous_run', False), description='continue previous run by reading contents in the file defined by filename')]

    return ParameterSet(params)