import os
import numpy as np

# try:
#   import commands
# except:
#   import subprocess as commands

# import tempfile
from phoebe.parameters import ParameterSet
import phoebe.parameters as _parameters
import phoebe.frontend.bundle
from phoebe import conf
from phoebe.dependencies import distl as _distl


try:
    import emcee
    import h5py
except ImportError:
    _use_emcee = False
else:
    _use_emcee = True

try:
    import dynesty
    import pickle
except ImportError:
    _use_dynesty = False
else:
    _use_dynesty = True

import logging
logger = logging.getLogger("SOLUTION")
logger.addHandler(logging.NullHandler())

_skip_filter_checks = {'check_default': False, 'check_visible': False}


class BaseSolutionBackend(object):
    def __init__(self, bundle, solution, solution_kwargs):
        self._process_cache = {}
        self._bundle = bundle
        self._solution = solution
        self._solution_kwargs = solution_kwargs
        self._needs_process = True  # set to False once processed or if a backend doesn't need to be processed to adopt
        return

    def __repr__(self):
        return "<{} keys={}>".format(self.__class__.__name__, list(self.keys()))

    def __str__(self):
        return str(self._cache)

    def keys(self):
        return self._cache.keys()

    def items(self):
        return self._cache.items()

    def values(self):
        return self._cache.values()

    def __getitem__(self, k):
        return self._cache.get(k)

    @property
    def bundle(self):
        return self._bundle

    @property
    def solution(self):
        return self._solution

    @property
    def solution_kwargs(self):
        return self._solution_kwargs

    @property
    def _solver_kind(self):
        return self.__class__.__name__.strip('Solution').lower()

    @property
    def _cache(self):
        ret = self.solution_kwargs
        for k,v in self._process_cache.items():
            ret[k] = v
        return ret

    def get(self, key, kwargs={}, default=None):
        # kwargs -> solution_kwargs -> default
        return kwargs.get(key, self.solution_kwargs.get(key, default))

    def _process(self, **kwargs):
        """
        """
        raise NotImplementedError("_process not subclassed by {}".format(self.__class__.__name__))

    def process(self, **kwargs):
        """
        """
        ret = self._process(self, **kwargs)
        self._process_cache = ret
        # _cache will also automatically include solution_kwargs
        return self._cache

    def adopt(self, **kwargs):
        """
        """
        raise NotImplementedError("adopt not subclassed by {}".format(self.__class__.__name__))

class BaseDistributionSolutionBackend(BaseSolutionBackend):
    def adopt(self, distribution=None):
        """
        """
        raise NotImplementedError()

class BaseValueSolutionBackend(BaseSolutionBackend):
    def __init__(self, *args, **kwargs):
        super(BaseValueSolutionBackend, self).__init__(*args, **kwargs)
        self._needs_process = False

    def _process(self, filename, **kwargs):
        """
        """
        uniqueids = self.solution_kwargs.get('fitted_parameters')
        fitted_values = self.solution_kwargs.get('fitted_values')
        fitted_units = self.solution_kwargs.get('fitted_units')

        quantities = {self.bundle.get_parameter(uniqueid=uniqueid, **_skip_filter_checks).get_uniquetwig(): value*unit for uniqueid, value, unit in zip(uniqueids, fitted_values, fitted_units)}
        values = {k: v.value for k,v in quantities.items()}
        return {'values': values, 'quantities': quantities}

    def _preadopt_check(self):
        if not self.get('success'):
            logger.warning('success=False was returned by the {} backend.'.format(self._solver_kind))

    def adopt(self, distribution=None):
        """
        """
        if distribution is not None:
            raise ValueError("{} adopts single values, not distributions.  distribution must be None".format(self.__class__.__name__))
        if self._needs_process and not len(self._process_cache.keys()):
            self.process()

        self._preadopt_check()


        user_interactive_constraints = conf.interactive_constraints
        conf.interactive_constraints_off(suppress_warning=True)

        uniqueids = self.solution_kwargs.get('fitted_parameters')
        values = self.solution_kwargs.get('fitted_values')
        units = self.solution_kwargs.get('fitted_units')

        changed_params = []
        for uniqueid, value, unit in zip(uniqueids, values, units):
            param = self.bundle.get_parameter(uniqueid=uniqueid, **_skip_filter_checks)

            param.set_value(value, unit=unit)
            changed_params.append(param)

        changed_params += self.bundle.run_delayed_constraints()
        if user_interactive_constraints:
            conf.interactive_constraints_on()

        return ParameterSet(changed_params)


class EmceeSolution(BaseDistributionSolutionBackend):
    """
    See <phoebe.parameters.solver.sampler.emcee>.

    Generally this class will be instantiated by
    * <phoebe.frontend.bundle.Bundle.process_solution>

    after calling
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
    def _process(self, filename, **kwargs):
        """
        """
        if not _use_emcee:
            raise ImportError("could not import emcee")

        filename = self.get('filename', kwargs)
        # TODO: do we need to be careful about full-path?

        reader = emcee.backends.HDFBackend(filename)
        # TODO: remove quiet or re-implement logic as warning
        autocorr_time = reader.get_autocorr_time(quiet=True)
        try:
            burnin = self.get('burnin', kwargs, int(self.get('burnin_factor', kwargs, 2) * np.max(autocorr_time)))
        except:
            logger.warning("could not compute burnin, falling back on 0")
            burnin = 0
        try:
            thin = int(self.get('thin', kwargs, int(self.get('thin_factor', kwargs, 0.5) * np.min(autocorr_time))))
        except:
            logger.warning("could not compute thin, falling back on 1")
            thin = 1
        try:
            samples = reader.get_chain(discard=burnin, thin=thin, flat=False)
        except:
            logger.warning("could not get samples within burnin={}, thin={}".format(burnin, thin))
            raise

        log_prob_samples = reader.get_log_prob(discard=burnin, thin=thin, flat=False)

        ps = self.bundle.filter(context=['component', 'dataset'], **_skip_filter_checks)
        fitted_params = [ps.get_parameter(uniqueid=uniqueid, **_skip_filter_checks) for uniqueid in self.get('fitted_parameters', kwargs)]
        fitted_twigs = [param.get_uniquetwig(ps, exclude_levels=['context']) for param in fitted_params]

        # TODO: this assumes the unit hasn't changed since the solver run... alternatively we could store units in the solution as 'fitted_units'
        # TODO: distl multivariate support needs to accept list of units
        fitted_units = None
        #fitted_units = [param.default_unit for param in fitted_params]

        dist = _distl.mvhistogram_from_data(samples.reshape((-1,len(fitted_twigs))), bins=kwargs.get('bins', 10), range=None, weights=None, units=fitted_units, labels=fitted_twigs, wrap_ats=None)


        return {'autocorr_time': autocorr_time,
                'burnin': burnin,
                'thin': thin,
                'samples': samples,
                'fitted_twigs': fitted_twigs,
                'lnp': log_prob_samples,
                'distribution': dist,
                'object': reader}

    def adopt(self, distribution):
        """
        """
        if not isinstance(distribution, str):
            # TODO: check for validity in bundle or leave that for later?
            raise ValueError("distribution must be a valid string")

        if self._needs_process and not len(self._process_cache.keys()):
            self.process()

        dc = self.solution_kwargs.get('distribution')
        uniqueids = self.solution_kwargs.get('fitted_parameters')

        for i, uniqueid in enumerate(uniqueids):
            self.bundle.add_distribution(uniqueid=uniqueid, value=dc.slice(i), distribution=distribution)

        # TODO: do we want to only return newly added distributions?
        return self.bundle.get_distribution(distribution=distribution)

class DynestySolution(BaseDistributionSolutionBackend):
    """
    See <phoebe.parameters.solver.sampler.dynesty>.

    Generally this class will be instantiated by
    * <phoebe.frontend.bundle.Bundle.process_solution>

    after calling
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
    def _process(self, filename, **kwargs):
        """
        """
        if not _use_dynesty:
            raise ImportError("could not import dynesty and pickle")

        filename = self.get('filename', kwargs)
        # TODO: do we need to be careful about full-path?

        with open(filename, 'rb') as pfile:
            dynesty_results = pickle.load(pfile)

        ret = {k:v for k,v in dynesty_results.items() if k not in ['bound']}
        ret['object'] = dynesty_results
        # TODO: how do we get distributions out of dynesty???
        ret['distribution'] = None
        return ret

class Lc_Eclipse_GeometrySolution(BaseValueSolutionBackend):
    """
    See <phoebe.parameters.solver.optimizier.nelder_mead>.

    Generally this class will be instantiated by
    * <phoebe.frontend.bundle.Bundle.process_solution>

    after calling
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """

class Nelder_MeadSolution(BaseValueSolutionBackend):
    """
    See <phoebe.parameters.solver.optimizier.nelder_mead>.

    Generally this class will be instantiated by
    * <phoebe.frontend.bundle.Bundle.process_solution>

    after calling
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """


class Differential_EvolutionSolution(BaseValueSolutionBackend):
    """
    See <phoebe.parameters.solver.optimizier.differential_evolution>.

    Generally this class will be instantiated by
    * <phoebe.frontend.bundle.Bundle.process_solution>

    after calling
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
