
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

    params += [SelectParameter(qualifier='init_from', value=kwargs.get('init_from', []), choices=[], description='distribution(s) to initialize samples from (all unconstrained parameters with attached distributions will be sampled/fitted, constrained parameters will be ignored, covariances will be respected)')]
    params += [ChoiceParameter(visible_if='init_from:<notempty>', qualifier='init_from_combine', value=kwargs.get('init_from_combine', 'first'), choices=['first', 'and', 'or'], description='Method to use to combine multiple distributions from init_from for the same parameter.  first: ignore duplicate entries and take the first in the init_from parameter. and: combine duplicate entries via AND logic, dropping covariances.  or: combine duplicate entries via OR logic, dropping covariances.')]

    params += [SelectParameter(qualifier='priors', value=kwargs.get('priors', []), choices=[], description='distribution(s) to use for priors (constrained and unconstrained parameters will be included, covariances will be respected except for distributions merge via priors_combine)')]
    params += [ChoiceParameter(visible_if='priors:<notempty>', qualifier='priors_combine', value=kwargs.get('priors_combine', 'and'), choices=['first', 'and', 'or'], description='Method to use to combine multiple distributions from priors for the same parameter.  irst: ignore duplicate entries and take the first in the priors parameter. and: combine duplicate entries via AND logic, dropping covariances.  or: combine duplicate entries via OR logic, dropping covariances.')]

    params += [IntParameter(qualifier='nwalkers', value=kwargs.get('nwalkers', 16), limits=(1,1e5), description='number of walkers')]
    params += [IntParameter(qualifier='niters', value=kwargs.get('niters', 100), limits=(1,1e12), description='number of iterations')]

    params += [StringParameter(qualifier='filename', value=kwargs.get('filename', 'emcee_progress.hd5'), description='filename to use for storing progress and continuing from previous run')]
    params += [BoolParameter(qualifier='continue_previous_run', value=kwargs.get('continue_previous_run', False), description='continue previous run by reading contents in the file defined by filename')]

    return ParameterSet(params)

def dynesty(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    dynesty backend.  To use this backend, you must have dynesty installed.

    To install dynesty, see https://dynesty.readthedocs.io

    If using this backend for solver, consider citing:
    * https://ui.adsabs.harvard.edu/abs/2019arXiv190402180S
    * https://ui.adsabs.harvard.edu/abs/2004AIPC..735..395S
    * https://projecteuclid.org/euclid.ba/1340370944

    and see:
    * https://dynesty.readthedocs.io/en/latest/#citations

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
    b.add_solver('sampler.dynesty')
    b.run_solver(kind='dynesty')
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

    params += [SelectParameter(qualifier='priors', value=kwargs.get('priors', []), choices=[], description='distribution(s) to use for priors (as dynesty samples directly from the prior, constrained parameters will be ignored, covariances will be dropped)')]
    params += [ChoiceParameter(visible_if='priors:<notempty>', qualifier='priors_combine', value=kwargs.get('priors_combine', 'and'), choices=['first', 'and', 'or'], description='Method to use to combine multiple distributions from priors for the same parameter.irst: ignore duplicate entries and take the first in the priors parameter. and: combine duplicate entries via AND logic, dropping covariances.  or: combine duplicate entries via OR logic, dropping covariances.')]

    params += [IntParameter(qualifier='nlive', value=kwargs.get('nlive', 100), limits=(1,1e12), description='number of live points.   Larger numbers result in a more finely sampled posterior (more accurate evidence), but also a larger number of iterations required to converge.')]
    params += [IntParameter(qualifier='maxiter', value=kwargs.get('maxiter', 100), limits=(1,1e12), description='maximum number of iterations')]
    params += [IntParameter(qualifier='maxcall', value=kwargs.get('maxcall', 1000), limits=(1,1e12), description='maximum number of calls (forward models)')]

    params += [StringParameter(qualifier='filename', value=kwargs.get('filename', 'dynesty_progress.pkl'), description='filename to use for storing progress and continuing from previous run')]
    # params += [BoolParameter(qualifier='continue_previous_run', value=kwargs.get('continue_previous_run', False), description='continue previous run by reading contents in the file defined by filename')]

    return ParameterSet(params)
