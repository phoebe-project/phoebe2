import numpy as np
from copy import deepcopy as _deepcopy

try:
    import dynesty as _dynesty
except ImportError:
    _use_dynesty = False
else:
    _use_dynesty = True

try:
    import emcee
except ImportError:
    _use_emcee = False
else:
    _use_emcee = True

_skip_filter_checks = {'check_default': False, 'check_visible': False}

def process_mcmc_chains_from_solution(b, solution, burnin=None, thin=None, lnprob_cutoff=None, adopt_parameters=None, flatten=True):
    """
    Process the full MCMC chain and expose the `lnprobabilities` and `samples`
    after applying `burnin`, `thin`, `lnprob_cutoff`.

    See also:
    * <phoebe.helpers.process_mcmc_chains>
    * <phoebe.parameters.solver.sampler.emcee>

    Arguments
    ---------------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `solution` (string): solution label
    * `burnin` (int, optional, default=None): If not None, will override
        the value in the solution.
    * `thin` (int, optional, default=None): If not None, will override
        the value in the solution.
    * `lnprob_cutoff` (float, optional, default=None): If not None, will override
        the value in the solution.
    * `adopt_parameters` (list, optional, default=None): If not None, will
        override the value in the solution.
    * `flatten` (bool, optional, default=True): whether to flatten to remove
        the "walkers" dimension.  If False, `lnprob_cutoff` will replace entries
        with nans rather than excluding them.

    Returns
    -----------
    * (array, array): `lnprobablities` (flattened to 1D if `flatten`) and `samples`
        (flattened to 2D if `flatten`) after processing
    """
    solution_ps = b.get_solution(solution=solution, **_skip_filter_checks)
    adopt_inds, adopt_uniqueids = b._get_adopt_inds_uniqueids(solution_ps, adopt_parameters=adopt_parameters)

    return process_mcmc_chains(solution_ps.get_value(qualifier='lnprobabilities', **_skip_filter_checks),
                               solution_ps.get_value(qualifier='samples', **_skip_filter_checks),
                               solution_ps.get_value(qualifier='burnin', burnin=burnin, **_skip_filter_checks),
                               solution_ps.get_value(qualifier='thin', thin=thin, **_skip_filter_checks),
                               solution_ps.get_value(qualifier='lnprob_cutoff', lnprob_cutoff=lnprob_cutoff, **_skip_filter_checks),
                               adopt_inds=adopt_inds, flatten=flatten)

def process_mcmc_chains(lnprobabilities, samples, burnin=0, thin=1, lnprob_cutoff=-np.inf, adopt_inds=None, flatten=True):
    """
    Process the full MCMC chain and expose the `lnprobabilities` and `samples`
    after applying `burnin`, `thin`, `lnprob_cutoff`.

    See also:
    * <phoebe.helpers.process_mcmc_chains_from_solution>

    Arguments
    -------------
    * `lnprobabilities` (array): array of all lnprobabilites as returned by MCMC.
        Should have shape: (`niters`, `nwalkers`).
    * `samples` (array): array of all samples from the chains as returned by MCMC.
        Should have shape: (`niters`, `nwalkers`, `nparams`).
    * `burnin` (int, optional, default=0): number of iterations to exclude from
        the beginning of the chains.  Must be between `0` and `niters`.
    * `thin` (int, optional, default=1): after applying `burnin`, take every
        `thin` iterations.  Must be between `1` and (`niters`-`burnin`)
    * `lnprob_cutoff` (float, optiona, default=-inf): after applying `burnin`
        and `thin`, reject any entries (in both `lnprobabilities` and `samples`)
        in which `lnprobabilities` is below `lnprob_cutoff`.
    * `adopt_inds` (list, optional, default=None): if not None, only expose
        the parameters in `samples` with `adopt_inds` indices.  Each entry
        in the list should be between `0` and `nparams`.
    * `flatten` (bool, optional, default=True): whether to flatten to remove
        the "walkers" dimension.  If False, `lnprob_cutoff` will replace entries
        with nans rather than excluding them.

    Returns
    -----------
    * (array, array): `lnprobablities` (flattened to 1D if `flatten`) and `samples`
        (flattened to 2D if `flatten`) after processing

    Raises
    ------------
    * TypeError: if `burnin` or `thin` are not integers
    * ValueError: if `lnprobabilities` or `samples` have the wrong shape
    * ValueError: if `burnin`, `thin`, or `adopt_inds` are not valid given
        the shapes of `lnprobabilities` and `samples`.
    """
    if not isinstance(burnin, int):
        raise TypeError("burnin must be of type int")
    if not isinstance(thin, int):
        raise TypeError("thin must be of type int")

    if len(lnprobabilities.shape) != 2:
        raise ValueError("lnprobablities must have shape (niters, nwalkers), not {}".format(lnprobabilities.shape))
    if len(samples.shape) != 3:
        raise ValueError("samples must have shape (niters, nwalkers, nparams), not {}".format(samples.shape))

    if lnprobabilities.shape[0] != samples.shape[0] or lnprobabilities.shape[1] != samples.shape[1]:
        raise ValueError("lnprobablities and samples must have same size for first 2 dimensions lnprobabilities.shape=(niters, nwalkers), samples.shape=(niters, nwalkers, nparams)")

    niters, nwalkers, nparams = samples.shape

    if adopt_inds is not None:
        if not isinstance(adopt_inds, list):
            raise TypeError("adopt_inds must be of type list or None")
        if not np.all([ai < nparams and ai >=0 for ai in adopt_inds]):
            raise ValueError("each item in adopt_inds must be between 0 and {}".format(nparams))

    if burnin < 0:
        raise ValueError("burnin must be >=0")
    if burnin >= niters:
        raise ValueError("burnin must be < {}".format(niters))
    if thin < 1:
        raise ValueError("thin must be >= 1")
    if thin >= niters-burnin:
        raise ValueError("thin must be < {}".format(niters-burnin))


    # lnprobabilities[iteration, walker]
    lnprobabilities = lnprobabilities[burnin:, :][::thin, :]
    # samples[iteration, walker, parameter]
    samples = samples[burnin:, :, :][::thin, : :]
    if adopt_inds is not None:
        samples = samples[:, :, adopt_inds]

    if flatten:
        lnprob_inds = np.where(lnprobabilities > lnprob_cutoff)
        samples = samples[lnprob_inds]
        lnprobabilities = lnprobabilities[lnprob_inds]
        # samples 2D (n, nparameters)
        # lnprobabilities 1D (n)
    else:
        lnprobabilities = _deepcopy(lnprobabilities)
        samples = _deepcopy(samples)

        samples[lnprobabilities < lnprob_cutoff] = np.nan
        lnprobabilities[lnprobabilities < lnprob_cutoff] = np.nan

        # lnprobabilities 2D (niters (after burnin, thin), nwalkers), replaced with nans where lnprob_cutoff applies
        # samples 3D (niters (after burning,thin), nwalkers, nparameters)

    return lnprobabilities, samples

def get_emcee_object_from_solution(b, solution, adopt_parameters=None):
    """
    Expose the `EnsembleSampler` object in `emcee` from the solution <phoebe.parameters.ParameterSet>.

    See also:
    * <phoebe.helpers.get_emcee_object>
    * <phoebe.parameters.solver.sampler.emcee>

    Arguments
    ------------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `solution` (string): solution label with `kind=='dynesty'`
    * `adopt_parameters` (list, optional, default=None): If not None, will
        override the value of `adopt_parameters` in the solution.

    Returns
    -----------
    * [emcee.EnsembleSampler](https://emcee.readthedocs.io/en/stable/user/sampler/#emcee.EnsembleSampler) object
    """
    solution_ps = b.get_solution(solution=solution, **_skip_filter_checks)
    if solution_ps.kind != 'emcee':
        raise ValueError("solution_ps must have kind 'emcee'")

    adopt_inds, adopt_uniqueids = b._get_adopt_inds_uniqueids(solution_ps, adopt_parameters=adopt_parameters)

    samples = solution_ps.get_value(qualifier='samples', **_skip_filter_checks) # shape: (niters, nwalkers, nparams)
    lnprobabilites = solution_ps.get_value(qualifier='lnprobabilities', **_skip_filter_checks) # shape: (niters, nwalkers)
    acceptance_fractions = solution_ps.get_value(qualifier='acceptance_fractions', **_skip_filter_checks) # shape: (nwalkers)

    return get_emcee_object(samples[:,:,adopt_inds], lnprobabilites, acceptance_fractions)

def get_emcee_object(samples, lnprobabilities, acceptance_fractions):
    """
    Expose the `EnsembleSampler` object in `emcee`.

    See also:
    * <phoebe.helpers.get_emcee_object_from_solution>

    Arguments
    ------------
    * `samples` (array): samples with shape (niters, nwalkers, nparams)
    * `lnprobabilities` (array): log-probablities with shape (niters, nwalkers)
    * `acceptance_fractions` (array): acceptance fractions with shape (nwalkers)

    Returns
    -----------
    * [emcee.EnsembleSampler](https://emcee.readthedocs.io/en/stable/user/sampler/#emcee.EnsembleSampler) object
    """
    if not _use_emcee:
        raise ImportError("emcee is not installed")

    backend = emcee.backends.Backend()
    backend.nwalkers = samples.shape[1]
    backend.ndim = samples.shape[2]
    backend.iteration = samples.shape[0]
    backend.accepted = acceptance_fractions
    backend.chain = samples
    backend.log_prob = lnprobabilities
    backend.initialized = True
    backend.random_state = None
    if not hasattr(backend, 'blobs'):
        # some versions of emcee seem to have a bug where it tries
        # to access backend.blobs but that does not exist.  Since
        # we don't use blobs, we'll get around that by faking it to
        # be None
        backend.blobs = None

    return backend


def get_dynesty_object_from_solution(b, solution, adopt_parameters=None):
    """
    Expose the `results` object in `dynesty` from the solution <phoebe.parameters.ParameterSet>.

    See also:
    * <phoebe.helpers.get_dynesty_object>
    * <phoebe.parameters.solver.sampler.dynesty>

    Arguments
    ------------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `solution` (string): solution label with `kind=='dynesty'`
    * `adopt_parameters` (list, optional, default=None): If not None, will
        override the value of `adopt_parameters` in the solution.

    Returns
    -----------
    * [dynesty.results.Results](https://dynesty.readthedocs.io/en/latest/api.html#module-dynesty.results) object
    """
    solution_ps = b.get_solution(solution=solution, **_skip_filter_checks)
    if solution_ps.kind != 'dynesty':
        raise ValueError("solution_ps must have kind 'dynesty'")

    adopt_inds, adopt_uniqueids = b._get_adopt_inds_uniqueids(solution_ps, adopt_parameters=adopt_parameters)

    return get_dynesty_object(adopt_inds=adopt_inds, **{p.qualifier: p.get_value() for p in solution_ps.to_list() if p.qualifier in ['nlive', 'niter', 'ncall', 'eff', 'samples', 'samples_id', 'samples_it', 'samples_u', 'logwt', 'logl', 'logvol', 'logz', 'logzerr', 'information', 'bound_iter', 'samples_bound', 'scale']})

def get_dynesty_object(nlive, niter, ncall, eff,
                       samples, samples_id, samples_it, samples_u,
                       logwt, logl, logvol, logz, logzerr,
                       information, bound_iter,
                       samples_bound, scale,
                       adopt_inds=None):
    """
    Expose the `results` object in `dynesty`.

    This can then be passed to any [dynesty helper function](https://dynesty.readthedocs.io/en/latest/api.html#module-dynesty.results)
    that takes `results`.

    See also:
    * <phoebe.helpers.get_dynesty_object_from_solution>

    Arguments
    ------------
    * positional arguments: all positional arguments are passed directly to dynesty.
    * `adopt_inds` (list, optional, default=None): If not None, only include
        the parameters with `adopt_inds` indices from `samples` and `samples_u`.

    """
    if not _use_dynesty:
        raise ImportError("dynesty is not installed")

    def _filter_by_adopt_inds(p, adopt_inds):
        if adopt_inds is not None and p.qualifier in ['samples', 'samples_u']:
            return p.value[:, adopt_inds]
        return p.value

    return _dynesty.results.Results(nlive=nlive, niter=niter, ncall=ncall, eff=eff,
                                    samples=samples, samples_id=samples_id,
                                    samples_it=samples_it, samples_u=samples_u,
                                    logwt=logwt, logl=logl, logvol=logvol, logz=logz, logzerr=logzerr,
                                    information=information, bound_iter=bound_iter,
                                    samples_bound=samples_bound, scale=scale)
