
from phoebe.parameters import *
from phoebe import conf

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

def _comments_params(**kwargs):
    """
    """
    params = []

    params += [StringParameter(qualifier='comments', value=kwargs.get('comments', ''), description='User-provided comments for these solver-options.  Feel free to place any notes here - if not overridden, they will be copied to any resulting solutions.')]
    return params

def emcee(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    emcee backend.  To use this backend, you must have emcee 3.0+ installed.

    To install emcee, see https://emcee.readthedocs.io

    If using this backend for solver, consider citing:
    * https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F

    See also:
    * <phoebe.frontend.bundle.Bundle.references>
    * <phoebe.helpers.get_emcee_object_from_solution>
    * <phoebe.helpers.process_mcmc_chains_from_solution>

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

    Parallelization support: emcee supports both MPI and multiprocessing.  If
    using MPI and nprocs > nwalkers and `compute` is a phoebe backend, then MPI
    will be handled at the compute-level (per-time).  In all other cases,
    parallelization is handled at the solver-level (per-model).

    The resulting solution (from <phoebe.frontend.bundle.Bundle.run_solver> or
    <phoebe.frontend.bundle.Bundle.export_solver> and <phoebe.frontend.bundle.Bundle.import_solution>)
    then exposes the raw-products from `emcee`, after which the following
    actions can be taken:

    * <phoebe.parameters.ParameterSet.plot> with `style` as one of:
        'corner', 'failed', 'lnprobabilities', 'trace'/'walks'.
    * <phoebe.frontend.bundle.Bundle.adopt_solution> to adopt the resulting
        posteriors in a distribution.  Use `adopt_values=True` (defaults to False)
        to adopt the face-values.  Use `trial_run=True` to see the adopted
        distributions and/or values without applying to the Bundle.
    * <phoebe.frontend.bundle.Bundle.get_distribution_collection> to access
        the multivariate distribution representation of the posteriors.
    * <phoebe.helpers.process_mcmc_chains> or <phoebe.helpers.process_mcmc_chains_from_solution>
        to manually access the `lnprobabilities` and `samples` after applying
        `burnin`, `thin`, `lnprob_cutoff`, and `adopt_parameters`/`adopt_inds`.

    Arguments
    ----------
    * `compute` (string, optional): compute options to use for forward model
    * `continue_from` (string, optional, default='None'): continue the MCMC run
        from an existing emcee solution.  Chains will be appended to existing
        chains (so it is safe to overwrite the existing solution).  If 'None',
        will start a new run using `init_from`.
    * `init_from` (list, optional, default=[]): only applicable if `continue_from`
        is 'None'.  distribution(s) to initialize samples from (all unconstrained
        parameters with attached distributions will be sampled/fitted, constrained
        parameters will be ignored, covariances will be respected)
    * `init_from_combine` (string, optional, default='first'): only applicable
        if `continue_from` is 'None' and `init_from` is not empty.  Method to use
        to combine multiple distributions from `init_from` for the same parameter.
        first: ignore duplicate entries and take the first in the init_from parameter.
        and: combine duplicate entries via AND logic, dropping covariances.
         or: combine duplicate entries via OR logic, dropping covariances.
    * `priors` (list, optional, default=[]): distribution(s) to use for priors
        (constrained and unconstrained parameters will be included, covariances
        will be respected except for distributions merge via `priors_combine`)
    * `priors_combine` (string, optional, default='and'): only applicable
        if `priors` is not empty.  Method to use to combine multiple distributions
        from `priors` for the same parameter.
        first: ignore duplicate entries and take the first in the priors parameter.
        and: combine duplicate entries via AND logic, dropping covariances.
        or: combine duplicate entries via OR logic, dropping covariances.
    * `nwalkers` (int, optional, default=16): only appicable if `continue_from`
        is 'None'.  Number of walkers.
    * `niters` (int, optional, default=100): Number of iterations.
    * `burnin_factor` (float, optional, default=2): factor of max(autocorr_time)
        to apply for burnin (burnin not applied until adopting the solution)
    * `thin_factor` (float, optional, default=0.5): factor of min(autocorr_time)
        to apply for thinning (thinning not applied until adopting the solution)
    * `progress_every_niters` (int, optional, default=0): Save the progress of
        the solution every n iterations.  The solution can only be recovered
        from an early termination by loading the bundle from a saved file and
        then calling <phoebe.frontend.bundle.Bundle.import_solution>(filename).
        The filename of the saved file will default to solution.ps.progress within
        <phoebe.frontend.bundle.Bundle.run_solver>, or the output filename provided
        to <phoebe.frontend.bundle.Bundle.export_solver> suffixed with .progress.
        If using detach=True within run_solver, attach job will load the progress
        and allow re-attaching until the job is completed.  If 0 will not save
        and will only return after completion.
    * `expose_failed` (bool, optional, default=True): only applicable if
        `continue_from` is 'None'. whether to expose dictionary of failed samples
        and their error messages.  Note: depending on the number of failed
        samples, this could add overhead.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _comments_params(**kwargs)

    params += [ChoiceParameter(qualifier='compute', value=kwargs.get('compute', 'None'), choices=['None'], description='compute options to use for forward model')]

    params += [ChoiceParameter(qualifier='continue_from', value=kwargs.get('continue_from', 'None'), choices=['None'], description='continue the MCMC run from an existing emcee solution.  Chains will be appended to existing chains (so it is safe to overwrite the existing solution).  If None, will start a new run using init_from.')]
    params += [SelectParameter(visible_if='continue_from:None', qualifier='init_from', value=kwargs.get('init_from', []), choices=[], description='distribution(s) to initialize samples from (all unconstrained parameters with attached distributions will be sampled/fitted, constrained parameters will be ignored, covariances will be respected)')]
    params += [ChoiceParameter(visible_if='continue_from:None,init_from:<notempty>', qualifier='init_from_combine', value=kwargs.get('init_from_combine', 'first'), choices=['first', 'and', 'or'], description='Method to use to combine multiple distributions from init_from for the same parameter.  first: ignore duplicate entries and take the first in the init_from parameter. and: combine duplicate entries via AND logic, dropping covariances.  or: combine duplicate entries via OR logic, dropping covariances.')]

    params += [SelectParameter(qualifier='priors', value=kwargs.get('priors', []), choices=[], description='distribution(s) to use for priors (constrained and unconstrained parameters will be included, covariances will be respected except for distributions merge via priors_combine)')]
    params += [ChoiceParameter(visible_if='priors:<notempty>', qualifier='priors_combine', value=kwargs.get('priors_combine', 'and'), choices=['first', 'and', 'or'], description='Method to use to combine multiple distributions from priors for the same parameter.  first: ignore duplicate entries and take the first in the priors parameter. and: combine duplicate entries via AND logic, dropping covariances.  or: combine duplicate entries via OR logic, dropping covariances.')]

    params += [IntParameter(visible_if='continue_from:None', qualifier='nwalkers', value=kwargs.get('nwalkers', 16), limits=(1,1e5), description='Number of walkers')]
    params += [IntParameter(qualifier='niters', value=kwargs.get('niters', 100), limits=(1,1e12), description='Number of iterations')]

    params += [FloatParameter(qualifier='burnin_factor', value=kwargs.get('burnin_factor', 2), default_unit=u.dimensionless_unscaled, limits=(1, 1000), description='factor of max(autocorr_time) to apply for burnin (burnin not applied until adopting the solution)')]
    params += [FloatParameter(qualifier='thin_factor', value=kwargs.get('thin_factor', 0.5), default_unit=u.dimensionless_unscaled, limits=(0.001, 1000), description='factor of min(autocorr_time) to apply for thinning (thinning not applied until adopting the solution)')]

    params += [IntParameter(qualifier='progress_every_niters', value=kwargs.get('progress_every_niters', 0), limits=(0,1e6), description='save the progress of the solution every n iterations.  The solution can only be recovered from an early termination by loading the bundle from a saved file and then calling b.import_solution(filename).  The filename of the saved file will default to solution.ps.progress within run_solver, or the output filename provided to export_solver suffixed with .progress.  If using detach=True within run_solver, attach job will load the progress and allow re-attaching until the job is completed.  If 0 will not save and will only return after completion.')]

    params += [BoolParameter(visible_if='continue_from:None', qualifier='expose_failed', value=kwargs.get('expose_failed', True), description='whether to expose dictionary of failed samples and their error messages.  Note: depending on the number of failed samples, this could add overhead.')]

    return ParameterSet(params)

def dynesty(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    dynesty backend.  To use this backend, you must have dynesty installed.

    To install dynesty, see https://dynesty.readthedocs.io

    If using this backend for solver, consider citing:
    * https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S
    * https://ui.adsabs.harvard.edu/abs/2004AIPC..735..395S
    * https://projecteuclid.org/euclid.ba/1340370944

    and see:
    * https://dynesty.readthedocs.io/en/latest/#citations

    See also:
    * <phoebe.frontend.bundle.Bundle.references>
    * <phoebe.helpers.get_dynesty_object_from_solution>

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

    Parallelization support: dynesty supports both MPI and multiprocessing, always
    at the solver-level (per-model).

    The resulting solution (from <phoebe.frontend.bundle.Bundle.run_solver> or
    <phoebe.frontend.bundle.Bundle.export_solver> and <phoebe.frontend.bundle.Bundle.import_solution>)
    then exposes the raw-products from `dynesty`, after which the following
    actions can be taken:

    * <phoebe.parameters.ParameterSet.plot> with `style` as one of:
        'corner', 'failed', 'trace', 'run'.
    * <phoebe.frontend.bundle.Bundle.adopt_solution> to adopt the resulting
        posteriors in a distribution.  Use `adopt_values=True` (defaults to False)
        to adopt the face-values.  Use `trial_run=True` to see the adopted
        distributions and/or values without applying to the Bundle.
    * <phoebe.frontend.bundle.Bundle.get_distribution_collection> to access
        the multivariate distribution representation of the posteriors.
    * <phoebe.helpers.get_dynesty_object> or <phoebe.helpers.get_dynesty_object_from_solution>
        to manually access the dynesty object which can then be passed to any
        [dynesty helper function](https://dynesty.readthedocs.io/en/latest/api.html#module-dynesty.results)
        which accepts the `results` object.


    Arguments
    ----------
    * `compute` (string, optional): compute options to use for forward model
    * `priors` (list, optional, default=[]): distribution(s) to use for priors
        (as dynesty samples directly from the prior, constrained parameters will
        be ignored, covariances will be dropped)
    * `priors_combine` (string, optional, default='and'): only applicable
        if `priors` is not empty.  Method to use to combine multiple distributions
        from `priors` for the same parameter.
        first: ignore duplicate entries and take the first in the priors parameter.
        and: combine duplicate entries via AND logic, dropping covariances.
        or: combine duplicate entries via OR logic, dropping covariances.
    * `nlive` (int, optional, default=100): number of live points.   Larger
        numbers result in a more finely sampled posterior (more accurate evidence),
        but also a larger number of iterations required to converge.
    * `maxiter` (int, optional, default=100): maximum number of iterations
    * `maxcall` (int, optional, default=1000): maximum number of calls (forward models)
    * `progress_every_niters` (int, optional, default=0): Save the progress of
        the solution every n iterations.  The solution can only be recovered
        from an early termination by loading the bundle from a saved file and
        then calling <phoebe.frontend.bundle.Bundle.import_solution>(filename).
        The filename of the saved file will default to solution.ps.progress within
        <phoebe.frontend.bundle.Bundle.run_solver>, or the output filename provided
        to <phoebe.frontend.bundle.Bundle.export_solver> suffixed with .progress.
        If using detach=True within run_solver, attach job will load the progress
        and allow re-attaching until the job is completed.  If 0 will not save
        and will only return after completion.
    * `expose_failed` (bool, optional, default=True): only applicable if
        `continue_from` is 'None'. whether to expose dictionary of failed samples
        and their error messages.  Note: depending on the number of failed
        samples, this could add overhead.


    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _comments_params(**kwargs)

    params += [ChoiceParameter(qualifier='compute', value=kwargs.get('compute', 'None'), choices=['None'], description='compute options to use for forward model')]

    params += [SelectParameter(qualifier='priors', value=kwargs.get('priors', []), choices=[], description='distribution(s) to use for priors (as dynesty samples directly from the prior, constrained parameters will be ignored, covariances will be dropped)')]
    params += [ChoiceParameter(visible_if='priors:<notempty>', qualifier='priors_combine', value=kwargs.get('priors_combine', 'and'), choices=['first', 'and', 'or'], description='Method to use to combine multiple distributions from priors for the same parameter. first: ignore duplicate entries and take the first in the priors parameter. and: combine duplicate entries via AND logic, dropping covariances.  or: combine duplicate entries via OR logic, dropping covariances.')]

    params += [IntParameter(qualifier='nlive', value=kwargs.get('nlive', 100), limits=(1,1e12), description='number of live points.   Larger numbers result in a more finely sampled posterior (more accurate evidence), but also a larger number of iterations required to converge.')]
    params += [IntParameter(qualifier='maxiter', value=kwargs.get('maxiter', 100), limits=(1,1e12), description='maximum number of iterations')]
    params += [IntParameter(qualifier='maxcall', value=kwargs.get('maxcall', 1000), limits=(1,1e12), description='maximum number of calls (forward models)')]

    params += [IntParameter(qualifier='progress_every_niters', value=kwargs.get('progress_every_niters', 0), limits=(0,1e6), description='save the progress of the solution every n iterations.  The solution can only be recovered from an early termination by loading the bundle from a saved file and then calling b.import_solution(filename).  The filename of the saved file will default to solution.ps.progress within run_solver, or the output filename provided to export_solver suffixed with .progress.  If using detach=True within run_solver, attach job will load the progress and allow re-attaching until the job is completed.  If 0 will not save and will only return after completion.')]

    params += [BoolParameter(qualifier='expose_failed', value=kwargs.get('expose_failed', True), description='whether to expose dictionary of failed samples and their error messages.  Note: depending on the number of failed samples, this could add overhead.')]


    return ParameterSet(params)
