import os
import numpy as np

# try:
#   import commands
# except:
#   import subprocess as commands

# import tempfile
# from phoebe.parameters import ParameterSet
import phoebe.parameters as _parameters
import phoebe.frontend.bundle
from phoebe import u, c
from phoebe import conf, mpi
from phoebe.backend.backends import _simplify_error_message
from phoebe.utils import phase_mask_inds

from distutils.version import LooseVersion, StrictVersion
from copy import deepcopy as _deepcopy

from . import lc_eclipse_geometry

try:
    import emcee
    import schwimmbad
except ImportError:
    _use_emcee = False
else:
    _use_emcee = True

try:
    import dynesty
    import schwimmbad
    import pickle
except ImportError:
    _use_dynesty = False
else:
    _use_dynesty = True

try:
    from astropy.timeseries import BoxLeastSquares as _BoxLeastSquares
    from astropy.timeseries import LombScargle as _LombScargle
except ImportError:
    _use_astropy_timeseries = False
else:
    _use_astropy_timeseries = True

from scipy import optimize

import logging
logger = logging.getLogger("SOLVER")
logger.addHandler(logging.NullHandler())

_skip_filter_checks = {'check_default': False, 'check_visible': False}


def _bsolver(b, solver, compute, distributions):
    # TODO: OPTIMIZE exclude disabled datasets?
    # TODO: re-enable removing unused compute options - currently causes some constraints to fail
    # TODO: is it quicker to initialize a new bundle around b.exclude?  Or just leave everything?
    bexcl = b.copy()
    bexcl.remove_parameters_all(context=['model', 'solution', 'figure'], **_skip_filter_checks)
    if len(b.solvers) > 1:
        bexcl.remove_parameters_all(solver=[f for f in b.solvers if f!=solver and solver is not None], **_skip_filter_checks)
    bexcl.remove_parameters_all(distribution=[d for d in b.distributions if d not in distributions], **_skip_filter_checks)

    # handle solver_times
    for param in bexcl.filter(qualifier='solver_times', **_skip_filter_checks).to_list():
        solver_times = param.get_value()
        ds_ps = bexcl.get_dataset(dataset=param.dataset, **_skip_filter_checks)
        mask_enabled = ds_ps.get_value(qualifier='mask_enabled', default=False)
        if mask_enabled:
            mask_phases = ds_ps.get_value(qualifier='mask_phases', **_skip_filter_checks)
            mask_t0 = ds_ps.get_value(qualifier='phases_t0', **_skip_filter_checks)
        else:
            mask_phases = None
            mask_t0 = 't0_supconj'

        new_compute_times = None

        if solver_times == 'auto':
            compute_times = bexcl.get_value(qualifier='compute_times', dataset=param.dataset, context='dataset', **_skip_filter_checks)
            compute_phases = bexcl.to_phases(compute_times, t0=mask_t0)
            masked_compute_times = compute_times[phase_mask_inds(compute_phases, mask_phases)]

            times = np.unique(np.concatenate([time_param.get_value() for time_param in bexcl.filter(qualifier='times', dataset=param.dataset, **_skip_filter_checks).to_list()]))
            phases = bexcl.to_phases(times, t0=mask_t0)
            masked_times = times[phase_mask_inds(phases, mask_phases)]

            if len(masked_times) < len(masked_compute_times):
                logger.debug("solver_times=auto: using times instead of compute_times")
                if mask_enabled and len(mask_phases):
                    new_compute_times = masked_times
                else:
                    new_compute_times = []
            else:
                logger.debug("solver_times=auto: using compute_times")
                if mask_enabled and len(mask_phases):
                    new_compute_times = masked_compute_times
                else:
                    new_compute_times = None  # leave at user-set compute_times


        elif solver_times == 'times':
            logger.debug("solver_times=times: resetting compute_times in copied bundle")
            if mask_enabled and len(mask_phases):
                times = np.unique(np.concatenate([time_param.get_value() for time_param in bexcl.filter(qualifier='times', dataset=param.dataset, **_skip_filter_checks).to_list()]))
                phases = bexcl.to_phases(times, t0=mask_t0)
                new_compute_times = times[phase_mask_inds(phases, mask_phases)]
            else:
                new_compute_times = []

        elif solver_times == 'compute_times':
            logger.debug("solver_times=compute_times")
            if mask_enabled and len(mask_phases):
                compute_times = bexcl.get_value(qualifier='compute_times', dataset=param.dataset, context='dataset', **_skip_filter_checks)
                compute_phases = bexcl.to_phases(compute_times, t0=mask_t0)
                new_compute_times = compute_times[phase_mask_inds(compute_phases, mask_phases)]

            else:
                new_compute_times = None  # leave at user-set compute_times
        else:
            raise NotImplementedError("solver_times='{}' not implemented".format(solver_times))

        if new_compute_times is not None:
            ds_ps.set_value_all(qualifier='compute_times', value=new_compute_times, **_skip_filter_checks)

    return bexcl


def _lnprobability(sampled_values, b, params_uniqueids, compute,
                  priors, priors_combine,
                  solution,
                  compute_kwargs={},
                  custom_lnprobability_callable=None,
                  return_blob=False):


    def _return(lnprob, msg):
        if return_blob:
            return lnprob, (_simplify_error_message(msg), sampled_values) if msg != 'success' else (None, None)
        else:
            return lnprob

    # copy the bundle to make sure any changes by setting values/running models
    # doesn't affect the user-copy (or in other processors)
    b = b.copy()
    # prevent any *_around distributions from adjusting to the changes in
    # face-values
    b._within_sampling = True
    if sampled_values is not False:
        for uniqueid, value in zip(params_uniqueids, sampled_values):
            try:
                b.set_value(uniqueid=uniqueid, value=value, run_checks=False, run_constraints=False, **_skip_filter_checks)
            except ValueError as err:
                logger.warning("received error while setting values: {}. lnprobability=-inf".format(err))
                return _return(-np.inf, str(err))

    lnpriors = b.calculate_lnp(distribution=priors, combine=priors_combine)
    if not np.isfinite(lnpriors):
        # no point in calculating the model then
        return _return(-np.inf, 'lnpriors = -inf')

    # print("*** _lnprobability run_compute from rank: {}".format(mpi.myrank))
    try:
        # override sample_from that may be set in the compute options
        compute_kwargs['sample_from'] = []
        b.run_compute(compute=compute, model=solution, do_create_fig_params=False, **compute_kwargs)
    except Exception as err:
        logger.warning("received error from run_compute: {}.  lnprobability=-inf".format(err))
        return _return(-np.inf, str(err))

    # print("*** _lnprobability returning from rank: {}".format(mpi.myrank))
    if custom_lnprobability_callable is None:
        lnprob = lnpriors + b.calculate_lnlikelihood(model=solution)
    else:
        lnprob = custom_lnprobability_callable(b, model=solution, lnpriors=lnpriors, priors=priors, priors_combine=priors_combine)

    return _return(lnprob, 'success')

def _lnprobability_negative(sampled_values, b, params_uniqueids, compute,
                           priors, priors_combine,
                           solution,
                           compute_kwargs={},
                           custom_lnprobability_callable=None,
                           return_blob=False):

    return -1 * _lnprobability(sampled_values, b, params_uniqueids, compute,
                              priors, priors_combine,
                              solution,
                              compute_kwargs,
                              custom_lnprobability_callable,
                              return_blob)

def _sample_ppf(ppf_values, distributions_list):
    # NOTE: this will treat each item in the collection independently, ignoring any covariances

    x = np.empty_like(ppf_values)

    # TODO: replace with distl.sample_ppf_from_dists(distributions, values)
    # once implemented to support multivariate?

    for i,dist in enumerate(distributions_list):
        x[i] = dist.ppf(ppf_values[i])

    return x

class BaseSolverBackend(object):
    def __init__(self):
        return

    @property
    def workers_need_solution_ps(self):
        return False

    def run_checks(self, b, compute, times=[], **kwargs):
        """
        run any sanity checks to make sure the parameters and options are legal
        for this backend.  If they are not, raise an error here to avoid errors
        within the workers.

        Any physics-checks that are backend-independent should be in
        Bundle.run_checks, and don't need to be repeated here.

        This should be subclassed by all backends, otherwise will throw a
        NotImplementedError
        """
        raise NotImplementedError("run_checks is not implemented by the {} backend".format(self.__class__.__name__))

    def _get_packet_and_solution(self, b, solver, **kwargs):
        """
        see get_packet_and_solution.  _get_packet_and_solution provides the custom parts
        of the packet that are Backend-dependent.

        This should return the packet to send to all workers and the new_syns to
        be sent to the master.

        return packet, solution_ps
        """
        raise NotImplementedError("_get_packet_and_solution is not implemented by the {} backend".format(self.__class__.__name__))

    def get_packet_and_solution(self, b, solver, **kwargs):
        """
        get_packet_and_solution is called by the master and must get all information necessary
        to send to all workers.  The returned packet will be passed on as
        _run_chunk(**packet) with the following exceptions:

        * b: the bundle will be included in the packet serialized
        * solver: the label of the solver options will be included in the packet
        * compute: the label of the compute options will be included in the packet
        * backend: the class name will be passed on in the packet so the worker can call the correct backend
        * all kwargs will be passed on verbatim
        """
        solver_ps = b.get_solver(solver=solver, **_skip_filter_checks)
        for param in solver_ps.to_list():
            # we need to make sure SelectParameters are expanded correctly if sent through kwargs
            kwargs[param.qualifier] = param.get_value(expand=True, **{param.qualifier: kwargs.get(param.qualifier, None)})

        packet, solution_ps = self._get_packet_and_solution(b, solver, **kwargs)

        for k,v in kwargs.items():
            packet[k] = v

        # if kwargs.get('max_computations', None) is not None:
        #     if len(packet.get('infolists', packet.get('infolist', []))) > kwargs.get('max_computations'):
        #         raise ValueError("more than {} computations detected ({} estimated).".format(kwargs.get('max_computations'), len(packet['infolists'])))

        # packet['b'] = b.to_json() if mpi.enabled else b
        packet['b'] = b
        packet['solver'] = solver
        # packet['compute'] = compute  # should have been set by kwargs, when applicable
        packet['backend'] = self.__class__.__name__

        if self.workers_need_solution_ps:
            packet['solution_ps'] = solution_ps.copy()

        return packet, solution_ps

    def _fill_solution(self, solution_ps, rpacketlists_per_worker, metawargs={}):
        """
        rpacket_per_worker is a list of packetlists as returned by _run_chunk
        """
        # TODO: move to BaseBackendByDataset or BaseBackend?
        logger.debug("rank:{}/{} {}._fill_solution".format(mpi.myrank, mpi.nprocs, self.__class__.__name__))

        for packetlists in rpacketlists_per_worker:
            # single worker
            for packetlist in packetlists:
                # single time/dataset
                for packet in packetlist:
                    # single parameter
                    if 'choices' in packet.keys():
                        choices = packet.pop('choices')
                        param = solution_ps.get_parameter(check_visible=False, **{k:v for k,v in packet.items() if k not in ['choices', 'value']})
                        param._choices = choices
                    try:
                        solution_ps.set_value(check_visible=False, check_default=False, ignore_readonly=True, **packet)
                    except Exception as err:
                        raise ValueError("failed to set value from packet: {}.  Original error: {}".format(packet, str(err)))

        if metawargs:
            for param in solution_ps.to_list():
                for k,v in metawargs.items():
                    setattr(param, '_{}'.format(k), v)

        return solution_ps

    def run_worker(self):
        """
        run_worker will receive the packet (with the bundle deserialized if necessary)
        and is responsible for any work done by a worker within MPI
        """
        raise NotImplementedError("run_worker is not implemented by the {} backend".format(self.__class__.__name__))

    def _run_worker(self, packet):
        # the worker receives the bundle serialized, so we need to unpack it
        logger.debug("rank:{}/{} _run_worker".format(mpi.myrank, mpi.nprocs))
        # do the computations requested for this worker
        rpacketlists = self.run_worker(**packet)
        # send the results back to the master (root=0)
        mpi.comm.gather(rpacketlists, root=0)

    def run(self, b, solver, compute, **kwargs):
        """
        if within mpirun, workers should call _run_worker instead of run
        """
        self.run_checks(b, solver,  compute, **kwargs)

        logger.debug("rank:{}/{} calling get_packet_and_solution".format(mpi.myrank, mpi.nprocs))
        packet, solution_ps = self.get_packet_and_solution(b, solver, **kwargs)

        if mpi.enabled:
            # broadcast the packet to ALL workers
            logger.debug("rank:{}/{} broadcasting to all workers".format(mpi.myrank, mpi.nprocs))
            mpi.comm.bcast(packet, root=0)

            # now even the master can become a worker and take on a chunk
            rpacketlists_per_worker = [self.run_worker(**packet)]

        else:
            rpacketlists_per_worker = [self.run_worker(**packet)]

        logger.debug("rank:{}/{} calling _fill_solution".format(mpi.myrank, mpi.nprocs))
        return self._fill_solution(solution_ps, rpacketlists_per_worker)


class Lc_Eclipse_GeometryBackend(BaseSolverBackend):
    """
    See <phoebe.parameters.solver.estimator.lc_eclipse_geometry>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_solver>
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
    def run_checks(self, b, solver, compute, **kwargs):
        solver_ps = b.get_solver(solver)
        if not len(solver_ps.get_value(qualifier='lc', lc=kwargs.get('lc', None))):
            raise ValueError("cannot run lc_eclipse_geometry without any dataset in lc")

        # TODO: check to make sure fluxes exist, etc


    def _get_packet_and_solution(self, b, solver, **kwargs):
        # NOTE: b, solver, compute, backend will be added by get_packet_and_solution
        solution_params = []

        solution_params += [_parameters.StringParameter(qualifier='lc', value='', readonly=True, description='dataset used to detect eclipses')]
        solution_params += [_parameters.StringParameter(qualifier='orbit', value='', readonly=True, description='orbit used for phasing the light curve')]

        solution_params += [_parameters.FloatParameter(qualifier='primary_width', value=0, readonly=True, unit=u.dimensionless_unscaled, description='phase-width of primary eclipse')]
        solution_params += [_parameters.FloatParameter(qualifier='secondary_width', value=0, readonly=True, unit=u.dimensionless_unscaled, description='phase-width of secondary eclipse')]
        solution_params += [_parameters.FloatParameter(qualifier='primary_phase', value=0, readonly=True, unit=u.dimensionless_unscaled, description='phase of primary eclipse')]
        solution_params += [_parameters.FloatParameter(qualifier='secondary_phase', value=0, readonly=True, unit=u.dimensionless_unscaled, description='phase of secondary eclipse')]
        solution_params += [_parameters.FloatParameter(qualifier='primary_depth', value=0, readonly=True, unit=u.dimensionless_unscaled, description='depth of primary eclipse')]
        solution_params += [_parameters.FloatParameter(qualifier='secondary_depth', value=0, readonly=True, unit=u.dimensionless_unscaled, description='depth of secondary eclipse')]

        solution_params += [_parameters.FloatArrayParameter(qualifier='eclipse_edges', value=[], readonly=True, unit=u.dimensionless_unscaled, description='detected phases of eclipse edges')]

        solution_params += [_parameters.ArrayParameter(qualifier='fitted_uniqueids', value=[], advanced=True, readonly=True, description='uniqueids of parameters fitted by the minimizer')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_twigs', value=[], readonly=True, description='twigs of parameters fitted by the minimizer')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_values', value=[], readonly=True, description='final values returned by the minimizer (in current default units of each parameter)')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_units', value=[], advanced=True, readonly=True, description='units of the fitted_values')]
        solution_params += [_parameters.SelectTwigParameter(qualifier='adopt_parameters', value=[], description='which of the parameters should be included when adopting the solution')]
        solution_params += [_parameters.BoolParameter(qualifier='adopt_distributions', value=False, description='whether to create a distribution (of delta functions of all parameters in adopt_parameters) when calling adopt_solution.')]
        solution_params += [_parameters.BoolParameter(qualifier='adopt_values', value=True, description='whether to update the parameter face-values (of all parameters in adopt_parameters) when calling adopt_solution.')]

        return kwargs, _parameters.ParameterSet(solution_params)

    def run_worker(self, b, solver, compute=None, **kwargs):
        if mpi.within_mpirun:
            raise NotImplementedError("mpi support for scipy.optimize not yet implemented")
            # TODO: we need to tell the workers to join the pool for time-parallelization?

        lc = kwargs.get('lc')
        orbit = kwargs.get('orbit')


        lc_ps = b.get_dataset(dataset=lc, **_skip_filter_checks)
        times = lc_ps.get_value(qualifier='times', unit='d', **_skip_filter_checks)
        phases = b.to_phase(times, component=orbit, t0='t0_supconj')
        fluxes = lc_ps.get_value(qualifier='fluxes', **_skip_filter_checks)
        sigmas = lc_ps.get_value(qualifier='sigmas', **_skip_filter_checks)
        if len(sigmas) == 0:
            sigmas = 0.001*fluxes.mean()*np.ones(len(fluxes))

        if not len(times) or len(times) != len(fluxes):
            raise ValueError("times and fluxes must exist and be filled in the '{}' dataset".format(lc))

        mask_enabled = lc_ps.get_value(qualifier='mask_enabled', default=False, **_skip_filter_checks)
        if mask_enabled:
            mask_phases = lc_ps.get_value(qualifier='mask_phases', **_skip_filter_checks)
            if len(mask_phases):
                logger.warning("applying mask_phases (may not be desired for finding eclipse edges - set mask_enabled=False to disable)")
                mask_t0 = lc_ps.get_value(qualifier='phases_t0', **_skip_filter_checks)
                phases_for_mask = b.to_phase(times, component=orbit, t0=mask_t0)

                inds = phase_mask_inds(phases_for_mask, mask_phases)

                phases = phases[inds]
                fluxes = fluxes[inds]
                sigmas = sigmas[inds]

        s = phases.argsort()
        phases = phases[s]
        fluxes = fluxes[s]
        sigmas = sigmas[s]

        orbit_ps = b.get_component(component=orbit, **_skip_filter_checks)
        ecc_param = orbit_ps.get_parameter(qualifier='ecc', **_skip_filter_checks)
        per0_param = orbit_ps.get_parameter(qualifier='per0', **_skip_filter_checks)
        t0_supconj_param = orbit_ps.get_parameter(qualifier='t0_supconj', **_skip_filter_checks)
        mask_phases_param = lc_ps.get_parameter(qualifier='mask_phases', **_skip_filter_checks)

        period = orbit_ps.get_value(qualifier='period', **_skip_filter_checks)
        t0_supconj_old = orbit_ps.get_value(qualifier='t0_supconj', **_skip_filter_checks)

        diagnose = kwargs.get('diagnose', False)
        eclipse_dict = lc_eclipse_geometry.compute_eclipse_params(phases, fluxes, sigmas, diagnose=diagnose)

        edges = eclipse_dict.get('eclipse_edges')
        mask_phases = [(edges[0], edges[1]), (edges[2], edges[3])]

        # TODO: update to use widths as well (or alternate based on ecc?)
        ecc, per0 = lc_eclipse_geometry.ecc_w_from_geometry(eclipse_dict.get('secondary_position') - eclipse_dict.get('primary_position'), eclipse_dict.get('primary_width'), eclipse_dict.get('secondary_width'))

        # TODO: create parameters in the solver options if we want to expose these options to the user
        # if t0_near_times == True the computed t0 is adjusted to fall in time times array range
        t0_near_times = kwargs.get('t0_near_times', True)

        t0_supconj_new = lc_eclipse_geometry.t0_from_geometry(eclipse_dict.get('primary_position'), times,
                                period=period, t0_supconj=t0_supconj_old, t0_near_times=t0_near_times)
        # TODO: correct t0_supconj?

        params_twigs = [t0_supconj_param.twig, ecc_param.twig, per0_param.twig, mask_phases_param.twig]

        return [[{'qualifier': 'primary_width', 'value': eclipse_dict.get('primary_width')},
                 {'qualifier': 'secondary_width', 'value': eclipse_dict.get('secondary_width')},
                 {'qualifier': 'primary_phase', 'value': eclipse_dict.get('primary_position')},
                 {'qualifier': 'secondary_phase', 'value': eclipse_dict.get('secondary_position')},
                 {'qualifier': 'primary_depth', 'value': eclipse_dict.get('primary_depth')},
                 {'qualifier': 'secondary_depth', 'value': eclipse_dict.get('secondary_depth')},
                 {'qualifier': 'eclipse_edges', 'value': eclipse_dict.get('eclipse_edges')},
                 {'qualifier': 'lc', 'value': lc},
                 {'qualifier': 'orbit', 'value': orbit},
                 {'qualifier': 'fitted_uniqueids', 'value': [t0_supconj_param.uniqueid, ecc_param.uniqueid, per0_param.uniqueid, mask_phases_param.uniqueid]},
                 {'qualifier': 'fitted_twigs', 'value': params_twigs},
                 {'qualifier': 'fitted_values', 'value': [t0_supconj_new, ecc, per0, mask_phases]},
                 {'qualifier': 'fitted_units', 'value': [u.d.to_string(), u.dimensionless_unscaled.to_string(), u.rad.to_string(), u.dimensionless_unscaled.to_string()]},
                 {'qualifier': 'adopt_parameters', 'value': params_twigs[:-1], 'choices': params_twigs},
                ]]



class PeriodogramBackend(BaseSolverBackend):
    """
    See <phoebe.parameters.solver.estimator.periodogram>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_solver>
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
    def run_checks(self, b, solver, compute, **kwargs):
        if not _use_astropy_timeseries:
            raise ImportError("astropy.timeseries not installed (requires astropy 3.2+)")

        solver_ps = b.get_solver(solver)
        if not len(solver_ps.get_value(qualifier='lc', lc=kwargs.get('lc', None))):
            raise ValueError("cannot run periodogram without any dataset in lc")

        # TODO: check to make sure fluxes exist, etc


    def _get_packet_and_solution(self, b, solver, **kwargs):
        # NOTE: b, solver, compute, backend will be added by get_packet_and_solution
        solution_params = []

        solution_params += [_parameters.FloatArrayParameter(qualifier='period', value=[], readonly=True, default_unit=u.d, description='periodogram test periods')]
        solution_params += [_parameters.FloatArrayParameter(qualifier='power', value=[], readonly=True, default_unit=u.dimensionless_unscaled, description='periodogram power')]

        solution_params += [_parameters.FloatParameter(qualifier='period_factor', value=1.0, default_unit=u.dimensionless_unscaled, description='factor to apply to the max peak period when adopting or plotting the solution')]

        solution_params += [_parameters.ArrayParameter(qualifier='fitted_uniqueids', value=[], advanced=True, readonly=True, description='uniqueids of parameters fitted by the minimizer')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_twigs', value=[], readonly=True, description='twigs of parameters fitted by the minimizer')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_values', value=[], readonly=True, description='final values returned by the minimizer (in current default units of each parameter)')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_units', value=[], advanced=True, readonly=True, description='units of the fitted_values')]
        solution_params += [_parameters.SelectTwigParameter(qualifier='adopt_parameters', value=[], description='which of the parameters should be included when adopting the solution')]
        solution_params += [_parameters.BoolParameter(qualifier='adopt_distributions', value=False, description='whether to create a distribution (of delta functions of all parameters in adopt_parameters) when calling adopt_solution.')]
        solution_params += [_parameters.BoolParameter(qualifier='adopt_values', value=True, description='whether to update the parameter face-values (of all parameters in adopt_parameters) when calling adopt_solution.')]

        return kwargs, _parameters.ParameterSet(solution_params)

    def run_worker(self, b, solver, compute=None, **kwargs):
        if mpi.within_mpirun:
            raise NotImplementedError("mpi support for periodogram not yet implemented")
            # TODO: we need to tell the workers to join the pool for time-parallelization?

        algorithm = kwargs.get('algorithm')
        lc = kwargs.get('lc')
        component = kwargs.get('component')

        lc_ps = b.get_dataset(dataset=lc, **_skip_filter_checks)
        times = lc_ps.get_value(qualifier='times', unit='d', **_skip_filter_checks)
        fluxes = lc_ps.get_value(qualifier='fluxes', **_skip_filter_checks)
        sigmas = lc_ps.get_value(qualifier='sigmas', **_skip_filter_checks)
        if not len(sigmas):
            sigmas = None

        sample_mode = kwargs.get('sample_mode')

        # TODO: options for duration to autopower/power (not sure what it does from the astropy docs)
        if algorithm == 'bls':
            model = _BoxLeastSquares(times, fluxes, dy=sigmas)
            # TODO: expose duration to solver options
            # https://docs.astropy.org/en/stable/api/astropy.timeseries.BoxLeastSquares.html#astropy.timeseries.BoxLeastSquares.autoperiod
            # https://docs.astropy.org/en/stable/api/astropy.timeseries.BoxLeastSquares.html#astropy.timeseries.BoxLeastSquares.period
            power_kwargs = {'duration': kwargs.get('duration'), 'objective': kwargs.get('objective')}
            autopower_kwargs = _deepcopy(power_kwargs)
            autopower_kwargs['minimum_n_transit'] = kwargs.get('minimum_n_cycles')
            sample = kwargs.get('sample_periods')
        elif algorithm == 'ls':
            model = _LombScargle(times, fluxes, dy=sigmas)
            # https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.autopower
            # https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.power
            power_kwargs = {}
            autopower_kwargs = _deepcopy(power_kwargs)
            autopower_kwargs['samples_per_peak'] = kwargs.get('samples_per_peak')
            sample = 1./kwargs.get('sample_periods')
            sample_sort = sample.argsort()
            sample = sample[sample_sort]
        else:
            raise NotImplementedError("algorithm='{}' not supported".format(algorithm))

        if sample_mode == 'auto':
            logger.info("calling {}.autopower({})".format(algorithm, autopower_kwargs))
            out = model.autopower(**autopower_kwargs)
            if algorithm in ['bls']:
                periods = out.period
                powers = out.power
            elif algorithm in ['ls']:
                periods = out[0]
                powers = out[1]
            else:
                raise NotImplementedError("algorithm='{}' not supported".format(algorithm))
        elif sample_mode == 'manual':
            logger.info("calling {}.power({}, {})".format(algorithm, sample, power_kwargs))
            # periodogram = model.power(kwargs.get('sample_periods' if algorithm in ['bls'] else 'sample_frequencies'), **power_kwargs)
            out = model.power(sample, **power_kwargs)
            if algorithm in ['bls']:
                periods = out.period
                powers = out.power
            elif algorithm in ['ls']:
                periods = kwargs.get('sample_periods')[sample_sort]
                powers = out
            else:
                raise NotImplementedError("algorithm='{}' not supported".format(algorithm))
        else:
            raise ValueError("sample_mode='{}' not supported".format(sample_mode))

        peak_ind = np.argmax(powers)
        # stats = model.compute_stats(periodogram.period[max_power],
        #                             periodogram.duration[max_power],
        #                             periodogram.transit_time[max_power])

        period = periods[peak_ind]

        period_param = b.get_parameter(qualifier='period', component=component, context='component', **_skip_filter_checks)

        params_twigs = [period_param.twig]

        return [[{'qualifier': 'period', 'value': periods},
                 {'qualifier': 'power', 'value': powers},
                 {'qualifier': 'fitted_uniqueids', 'value': [period_param.uniqueid]},
                 {'qualifier': 'fitted_twigs', 'value': params_twigs},
                 {'qualifier': 'fitted_values', 'value': [period]},
                 {'qualifier': 'fitted_units', 'value': ['d']},
                 {'qualifier': 'adopt_parameters', 'value': params_twigs, 'choices': params_twigs}
                  ]]



class EmceeBackend(BaseSolverBackend):
    """
    See <phoebe.parameters.solver.sampler.emcee>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_solver>
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
    @property
    def workers_need_solution_ps(self):
        return True

    def run_checks(self, b, solver, compute, **kwargs):
        # check whether emcee is installed

        if not _use_emcee:
            raise ImportError("could not import emcee, schwimmbad")

        if LooseVersion(emcee.__version__) < LooseVersion("3.0.0"):
            raise ImportError("emcee backend requires emcee 3.0+, {} found".format(emcee.__version__))

        solver_ps = b.get_solver(solver)
        if not len(solver_ps.get_value(qualifier='init_from', init_from=kwargs.get('init_from', None))):
            raise ValueError("cannot run emcee without any distributions in init_from")

        # require sigmas for all enabled datasets
        computes = solver_ps.get_value(qualifier='compute', compute=kwargs.get('compute', None), **_skip_filter_checks)
        datasets = b.filter(compute=computes, qualifier='enabled', value=True).datasets
        for sigma_param in b.filter(qualifier='sigmas', dataset=datasets, check_visible=True, check_default=True).to_list():
            if not len(sigma_param.get_value()):
                times = b.get_value(qualifier='times', dataset=sigma_param.dataset, component=sigma_param.component, **_skip_filter_checks)
                if len(times):
                    raise ValueError("emcee requires sigmas for all datasets where times exist (not found for {})".format(sigma_param.twig))



    def _get_packet_and_solution(self, b, solver, **kwargs):
        # NOTE: b, solver, compute, backend will be added by get_packet_and_solution

        solution_params = []
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_uniqueids', value=[], advanced=True, readonly=True, description='uniqueids of parameters fitted by the sampler')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_twigs', value=[], readonly=True, description='twigs of parameters fitted by the sampler')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_units', value=[], advanced=True, readonly=True, description='units of parameters fitted by the sampler')]
        solution_params += [_parameters.SelectTwigParameter(qualifier='adopt_parameters', value=[], description='which of the parameters should be included when adopting (and plotting) the solution')]
        solution_params += [_parameters.BoolParameter(qualifier='adopt_distributions', value=True, description='whether to create a distribution (of all parameters in adopt_parameters according to distributions_convert) when calling adopt_solution.')]
        solution_params += [_parameters.ChoiceParameter(qualifier='distributions_convert', value='mvsamples', choices=['mvsamples', 'mvhistogram', 'mvgaussian', 'samples', 'histogram', 'gaussian'], description='type of distribution to use when calling adopt_solution, get_distribution_collection, or plot. mvsamples: chains are stored directly and used for sampling with a KDE generated on-the-fly to compute probabilities.  mvhistogram: chains are binned according to distributions_bins and stored as an n-dimensional histogram.  mvgaussian: a multivariate gaussian is fitted to the samples, use only if distribution is sufficiently represented by gaussians.  samples: a univariate representation of mvsamples.  histogram: a univariate representation of mvhistogram.  gaussian: a univariate representation of mvgaussian.')]
        solution_params += [_parameters.IntParameter(visible_if='distributions_convert:mvhistogram|histogram', qualifier='distributions_bins', value=20, limits=(5,1000), description='number of bins to use for the distribution when calling adopt_solution, get_distribution_collection, or plot.')]
        solution_params += [_parameters.BoolParameter(qualifier='adopt_values', value=True, description='whether to update the parameter face-values (of the means of all parameters in adopt_parameters) when calling adopt_solution.')]


        solution_params += [_parameters.ArrayParameter(qualifier='samples', value=[], readonly=True, description='MCMC samples with shape (niters, nwalkers, len(fitted_twigs))')]
        if kwargs.get('expose_failed', True):
            solution_params += [_parameters.DictParameter(qualifier='failed_samples', value={}, readonly=True, description='MCMC samples that returned lnprobability=-inf.  Dictionary keys are the messages with values being an array with shape (N, len(fitted_twigs))')]
        solution_params += [_parameters.ArrayParameter(qualifier='lnprobabilities', value=[], readonly=True, description='log probabilities with shape (niters, nwalkers)')]

        # solution_params += [_parameters.ArrayParameter(qualifier='accepteds', value=[], description='whether each iteration was an accepted move with shape (niters)')]
        solution_params += [_parameters.ArrayParameter(qualifier='acceptance_fractions', value=[], readonly=True, description='fraction of proposed steps that were accepted with shape (nwalkers)')]

        solution_params += [_parameters.IntParameter(qualifier='autocorr_time', value=0, readonly=True, description='measured autocorrelation time')]
        solution_params += [_parameters.IntParameter(qualifier='burnin', value=0, limits=(0,1e6), description='burnin to use when adopting/plotting the solution')]
        solution_params += [_parameters.IntParameter(qualifier='thin', value=1, limits=(1,1e6), description='thin to use when adopting/plotting the solution')]
        solution_params += [_parameters.FloatParameter(qualifier='lnprob_cutoff', value=-np.inf, default_unit=u.dimensionless_unscaled, description='lower limit cuttoff on lnproabilities to use when adopting/plotting the solution')]

        solution_params += [_parameters.FloatParameter(qualifier='progress', value=0, limits=(0,100), default_unit=u.dimensionless_unscaled, advanced=True, readonly=True, descrition='percentage of requested iterations completed')]

        return kwargs, _parameters.ParameterSet(solution_params)

    def _run_worker(self, packet):
        # here we'll override loading the bundle since it is not needed
        # in run_worker (for the workers.... note that the master
        # will enter run_worker through run, not here)

        return self.run_worker(**packet)

    def run_worker(self, b, solver, compute, **kwargs):

        def _get_packetlist():
            return_ = [{'qualifier': 'fitted_uniqueids', 'value': params_uniqueids},
                     {'qualifier': 'fitted_twigs', 'value': params_twigs},
                     {'qualifier': 'fitted_units', 'value': params_units},
                     {'qualifier': 'adopt_parameters', 'value': params_twigs, 'choices': params_twigs},
                     {'qualifier': 'samples', 'value': samples},
                     {'qualifier': 'lnprobabilities', 'value': lnprobabilities},
                     {'qualifier': 'acceptance_fractions', 'value': acceptance_fractions},
                     {'qualifier': 'burnin', 'value': burnin},
                     {'qualifier': 'thin', 'value': thin},
                     {'qualifier': 'progress', 'value': progress}]

            if kwargs.get('expose_failed'):
                return_ += [{'qualifier': 'failed_samples', 'value': failed_samples}]

            return [return_]

        # emcee handles workers itself.  So here we'll just take the workers
        # from our own waiting loop in phoebe's __init__.py and subscribe them
        # to emcee's pool.
        if mpi.within_mpirun:
            pool = schwimmbad.MPIPool()
            is_master = pool.is_master()
        else:
            pool = schwimmbad.MultiPool()
            is_master = True

        # temporarily disable MPI within run_compute to disabled parallelizing
        # per-time.
        within_mpirun = mpi.within_mpirun
        mpi_enabled = mpi.enabled
        mpi._within_mpirun = False
        mpi._enabled = False

        if is_master:
            niters = kwargs.get('niters')
            nwalkers = kwargs.get('nwalkers')
            continue_from = kwargs.get('continue_from')

            init_from = kwargs.get('init_from')
            init_from_combine = kwargs.get('init_from_combine')
            priors = kwargs.get('priors')
            priors_combine = kwargs.get('priors_combine')

            progress_every_niters = kwargs.get('progress_every_niters')

            burnin_factor = kwargs.get('burnin_factor')
            thin_factor = kwargs.get('thin_factor')

            solution_ps = kwargs.get('solution_ps')
            solution = kwargs.get('solution')
            metawargs = {'context': 'solution',
                         'solver': solver,
                         'compute': compute,
                         'kind': 'emcee',
                         'solution': solution}

            # EnsembleSampler kwargs
            esargs = {}

            if continue_from == 'None':

                expose_failed = kwargs.get('expose_failed')

                dc, params_uniqueids = b.get_distribution_collection(distribution=init_from,
                                                                     combine=init_from_combine,
                                                                     include_constrained=False,
                                                                     keys='uniqueid')

                p0 = dc.sample(size=nwalkers).T
                params_units = [dist.unit.to_string() for dist in dc.dists]

                continued_failed_samples = {}
                start_iteration = 0
            else:
                # ignore the value from init_from (hidden parameter)
                init_from = []
                continue_from_ps = kwargs.get('continue_from_ps', b.filter(context='solution', solution=continue_from, **_skip_filter_checks))
                params_uniqueids = continue_from_ps.get_value(qualifier='fitted_uniqueids', **_skip_filter_checks)
                params_units = continue_from_ps.get_value(qualifier='fitted_units', **_skip_filter_checks)
                continued_samples = continue_from_ps.get_value(qualifier='samples', **_skip_filter_checks)
                expose_failed = 'failed_samples' in continue_from_ps.qualifiers
                if expose_failed:
                    continued_failed_samples = continue_from_ps.get_value(qualifier='failed_samples', **_skip_filter_checks)
                else:
                    continued_failed_samples = {}

                # continued_samples [iterations, walkers, parameter]
                # continued_accepteds = continue_from_ps.get_value(qualifier='accepteds', **_skip_filter_checks)
                # # continued_accepted [iterations, walkers]
                continued_acceptance_fractions = continue_from_ps.get_value(qualifier='acceptance_fractions', **_skip_filter_checks)
                # continued_acceptance_fractions [iterations, walkers]
                continued_lnprobabilities = continue_from_ps.get_value(qualifier='lnprobabilities', **_skip_filter_checks)
                # continued_lnprobabilities [iterations, walkers]
                p0 = continued_samples[-1].T
                # p0 [parameter, walkers]
                nwalkers = int(p0.shape[-1])

                start_iteration = continued_lnprobabilities.shape[0]

                # fake a backend object from the previous solution so that emcee
                # can continue from where it left off and still compute
                # autocorrelation times, etc.
                backend = emcee.backends.Backend()
                backend.nwalkers = int(nwalkers)
                backend.ndim = int(len(params_uniqueids))
                backend.iteration = start_iteration
                backend.accepted = np.asarray(continued_acceptance_fractions * start_iteration, dtype='int')
                backend.chain = continued_samples
                backend.log_prob = continued_lnprobabilities
                backend.initialized = True
                backend.random_state = None
                # reconstructing blobs will be messy, so we'll just get the correct
                # shape with nones, but add to the existing dictionary later
                backend.blobs = np.full(tuple(list(continued_samples.shape[:-1])+[2]), fill_value=None)

                esargs['backend'] = backend

            params_twigs = [b.get_parameter(uniqueid=uniqueid, **_skip_filter_checks).twig for uniqueid in params_uniqueids]

            esargs['pool'] = pool
            esargs['nwalkers'] = nwalkers
            esargs['ndim'] = len(params_uniqueids)
            esargs['log_prob_fn'] = _lnprobability
            # esargs['a'] = kwargs.pop('a', None),
            # esargs['moves'] = kwargs.pop('moves', None)
            # esargs['args'] = None

            esargs['kwargs'] = {'b': _bsolver(b, solver, compute, init_from+priors),
                                'params_uniqueids': params_uniqueids,
                                'compute': compute,
                                'priors': priors,
                                'priors_combine': priors_combine,
                                'solution': kwargs.get('solution', None),
                                'compute_kwargs': {k:v for k,v in kwargs.items() if k in b.get_compute(compute=compute, **_skip_filter_checks).qualifiers},
                                'custom_lnprobability_callable': kwargs.pop('custom_lnprobability_callable', None),
                                'return_blob': kwargs.get('expose_failed')}

            # esargs['live_dangerously'] = kwargs.pop('live_dangerously', None)
            # esargs['runtime_sortingfn'] = kwargs.pop('runtime_sortingfn', None)

            logger.debug("EnsembleSampler({})".format(esargs))
            sampler = emcee.EnsembleSampler(**esargs)


            sargs = {}
            sargs['iterations'] = niters
            sargs['progress'] = True
            # TODO: remove this?  Or can we reproduce the logic in a warning?
            sargs['skip_initial_state_check'] = False


            logger.debug("sampler.sample(p0, {})".format(sargs))
            for sample in sampler.sample(p0.T, **sargs):
                # TODO: parameters and options for checking convergence

                # check for kill signal
                if kwargs.get('out_fname', False) and os.path.isfile(kwargs.get('out_fname')+'.kill'):
                    logger.warning("received kill signal, exiting sampler loop")
                    break

                progress = float(sampler.iteration - start_iteration) / niters * 100

                if progress_every_niters == 0 and 'out_fname' in kwargs.keys():
                    fname = kwargs.get('out_fname') + '.progress'
                    f = open(fname, 'w')
                    f.write(str(progress))
                    f.close()

                # export progress/final results
                if (progress_every_niters > 0 and (sampler.iteration ==0 or (sampler.iteration - start_iteration) % progress_every_niters == 0)) or sampler.iteration - start_iteration == niters:
                    samples = sampler.backend.get_chain()
                    lnprobabilities = sampler.backend.get_log_prob()
                    # accepteds = sampler.backend.accepted
                    acceptance_fractions = sampler.backend.accepted / float(sampler.iteration)
                    autocorr_time = sampler.backend.get_autocorr_time(quiet=True)
                    if np.any(~np.isnan(autocorr_time)):
                        burnin = int(burnin_factor * np.nanmax(autocorr_time))
                        thin = int(thin_factor * np.nanmin(autocorr_time))
                        if thin==0:
                            thin = 1
                    else:
                        burnin =0
                        thin = 1

                    failed_samples = _deepcopy(continued_failed_samples)
                    if kwargs.get('expose_failed'):
                        for blob in sampler.get_blobs(flat=True):
                            if blob is None or blob[0] is None:
                                continue
                            failed_samples[blob[0]] = failed_samples.get(blob[0], []) + [blob[1].tolist()]

                    if progress_every_niters > 0:
                        logger.info("emcee: saving output from iteration {}".format(sampler.iteration))

                        solution_ps = self._fill_solution(solution_ps, [_get_packetlist()], metawargs)

                        if 'out_fname' in kwargs.keys():
                            if sampler.iteration - start_iteration == niters:
                                fname = kwargs.get('out_fname')
                            else:
                                fname = kwargs.get('out_fname') + '.progress'
                        else:
                            if sampler.iteration - start_iteration == niters:
                                fname = '{}.ps'.format(solution)
                            else:
                                fname = '{}.progress.ps'.format(solution)

                        solution_ps.save(fname, compact=True, sort_by_context=False)


        else:
            pool.wait()

        if pool is not None:
            pool.close()

        # restore previous MPI state
        mpi._within_mpirun = within_mpirun
        mpi._enabled = mpi_enabled

        if is_master:
            return _get_packetlist()
        return


class DynestyBackend(BaseSolverBackend):
    """
    See <phoebe.parameters.solver.sampler.dynesty>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_solver>
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
    @property
    def workers_need_solution_ps(self):
        return True

    def run_checks(self, b, solver, compute, **kwargs):
        # check whether emcee is installed

        if not _use_dynesty:
            raise ImportError("could not import dynesty, pickle, and schwimmbad")

        solver_ps = b.get_solver(solver)
        if not len(solver_ps.get_value(qualifier='priors', init_from=kwargs.get('priors', None))):
            raise ValueError("cannot run dynesty without any distributions in priors")

        # filename = solver_ps.get_value(qualifier='filename', filename=kwargs.get('filename', None))
        # continue_previous_run = solver_ps.get_value(qualifier='continue_previous_run', continue_previous_run=kwargs.get('continue_previous_run', None))
        # if continue_previous_run and not os.path.exists(filename):
            # raise ValueError("cannot file filename='{}', cannot use continue_previous_run=True".format(filename))


    def _get_packet_and_solution(self, b, solver, **kwargs):
        # NOTE: b, solver, compute, backend will be added by get_packet_and_solution

        solution_params = []
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_uniqueids', value=[], advanced=True, readonly=True, description='uniqueids of parameters fitted by the sampler')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_twigs', value=[], readonly=True, description='twigs of parameters fitted by the sampler')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_units', value=[], advanced=True, readonly=True, description='units of parameters fitted by the sampler')]
        solution_params += [_parameters.SelectTwigParameter(qualifier='adopt_parameters', value=[], description='which of the parameters should be included when adopting (and plotting) the solution')]
        solution_params += [_parameters.BoolParameter(qualifier='adopt_distributions', value=True, description='whether to create a distribution (of all parameters in adopt_parameters according to distributions_convert)  when calling adopt_solution.')]
        solution_params += [_parameters.ChoiceParameter(qualifier='distributions_convert', value='mvsamples', choices=['mvsamples', 'mvhistogram', 'mvgaussian', 'samples', 'histogram', 'gaussian'], description='type of distribution to use when calling adopt_solution, get_distribution_collection, or plot. mvsamples: chains are stored directly and used for sampling with a KDE generated on-the-fly to compute probabilities.  mvhistogram: chains are binned according to distributions_bins and stored as an n-dimensional histogram.  mvgaussian: a multivariate gaussian is fitted to the samples, use only if distribution is sufficiently represented by gaussians.  samples: a univariate representation of mvsamples.  histogram: a univariate representation of mvhistogram.  gaussian: a univariate representation of mvgaussian.')]
        solution_params += [_parameters.IntParameter(visible_if='distributions_convert:mvhistogram|histogram', qualifier='distributions_bins', value=20, limits=(5,1000), description='number of bins to use for the distribution when calling adopt_solution, get_distribution_collection, or plot.')]
        solution_params += [_parameters.BoolParameter(qualifier='adopt_values', value=True, description='whether to update the parameter face-values (of the means of all parameters in adopt_parameters) when calling adopt_solution.')]

        solution_params += [_parameters.IntParameter(qualifier='nlive', value=0, readonly=True, description='')]
        solution_params += [_parameters.IntParameter(qualifier='niter', value=0, readonly=True, description='')]
        solution_params += [_parameters.ArrayParameter(qualifier='ncall', value=0, readonly=True, description='')]
        solution_params += [_parameters.IntParameter(qualifier='eff', value=0, readonly=True, description='')]
        solution_params += [_parameters.ArrayParameter(qualifier='samples', value=0, readonly=True, description='')]
        solution_params += [_parameters.ArrayParameter(qualifier='samples_id', value=0, readonly=True, description='')]
        solution_params += [_parameters.ArrayParameter(qualifier='samples_it', value=0, readonly=True, description='')]
        solution_params += [_parameters.ArrayParameter(qualifier='samples_u', value=0, readonly=True, description='')]
        solution_params += [_parameters.ArrayParameter(qualifier='logwt', value=0, readonly=True, description='')]
        solution_params += [_parameters.ArrayParameter(qualifier='logl', value=0, readonly=True, description='')]
        solution_params += [_parameters.ArrayParameter(qualifier='logvol', value=0, readonly=True, description='')]
        solution_params += [_parameters.ArrayParameter(qualifier='logz', value=0, readonly=True, description='')]
        solution_params += [_parameters.ArrayParameter(qualifier='logzerr', value=0, readonly=True, description='')]
        solution_params += [_parameters.ArrayParameter(qualifier='information', value=0, readonly=True, description='')]
        # solution_params += [_parameters.ArrayParameter(qualifier='bound', value=0, readonly=True, description='')]
        solution_params += [_parameters.ArrayParameter(qualifier='bound_iter', value=0, readonly=True, description='')]
        solution_params += [_parameters.ArrayParameter(qualifier='samples_bound', value=0, readonly=True, description='')]
        solution_params += [_parameters.ArrayParameter(qualifier='scale', value=0, readonly=True, description='')]

        solution_params += [_parameters.FloatParameter(qualifier='progress', value=0, limits=(0,100), default_unit=u.dimensionless_unscaled, advanced=True, readonly=True, descrition='percentage of requested iterations completed')]

        return kwargs, _parameters.ParameterSet(solution_params)

    def _run_worker(self, packet):
        # here we'll override loading the bundle since it is not needed
        # in run_worker (for the workers.... note that the master
        # will enter run_worker through run, not here)
        return self.run_worker(**packet)

    def run_worker(self, b, solver, compute, **kwargs):

        def _get_packetlist(results):
            return [[{'qualifier': 'fitted_uniqueids', 'value': params_uniqueids},
                     {'qualifier': 'fitted_twigs', 'value': params_twigs},
                     {'qualifier': 'fitted_units', 'value': params_units},
                     {'qualifier': 'adopt_parameters', 'value': params_twigs, 'choices': params_twigs},
                     {'qualifier': 'nlive', 'value': results.nlive},
                     {'qualifier': 'niter', 'value': results.niter},
                     {'qualifier': 'ncall', 'value': results.ncall},
                     {'qualifier': 'eff', 'value': results.eff},
                     {'qualifier': 'samples', 'value': results.samples},
                     {'qualifier': 'samples_id', 'value': results.samples_id},
                     {'qualifier': 'samples_it', 'value': results.samples_it},
                     {'qualifier': 'samples_u', 'value': results.samples_u},
                     {'qualifier': 'logwt', 'value': results.logwt},
                     {'qualifier': 'logl', 'value': results.logl},
                     {'qualifier': 'logvol', 'value': results.logvol},
                     {'qualifier': 'logz', 'value': results.logz},
                     {'qualifier': 'logzerr', 'value': results.logzerr},
                     {'qualifier': 'information', 'value': results.information},
                     # {'qualifier': 'bound', 'value': results.bound},
                     {'qualifier': 'bound_iter', 'value': results.bound_iter},
                     {'qualifier': 'samples_bound', 'value': results.samples_bound},
                     {'qualifier': 'scale', 'value': results.scale},
                     {'qualifier': 'progress', 'value': progress}
                    ]]

        if mpi.within_mpirun:
            pool = schwimmbad.MPIPool()
            is_master = pool.is_master()
        else:
            pool = schwimmbad.MultiPool()
            is_master = True


        # temporarily disable MPI within run_compute to disabled parallelizing
        # per-time.
        within_mpirun = mpi.within_mpirun
        mpi_enabled = mpi.enabled
        mpi._within_mpirun = False
        mpi._enabled = False

        if is_master:
            priors = kwargs.get('priors')
            priors_combine = kwargs.get('priors_combine')

            maxiter = kwargs.get('maxiter')
            progress_every_niters = kwargs.get('progress_every_niters')

            solution_ps = kwargs.get('solution_ps')
            solution = kwargs.get('solution')
            metawargs = {'context': 'solution',
                         'solver': solver,
                         'compute': compute,
                         'kind': 'dynesty',
                         'solution': solution}

            # NOTE: here it is important that _sample_ppf sees the parameters in the
            # same order as _lnprobability (that is in the order of params_uniqueids)
            priors_dc, params_uniqueids = b.get_distribution_collection(distribution=priors,
                                                                        combine=priors_combine,
                                                                        include_constrained=False,
                                                                        keys='uniqueid',
                                                                        set_labels=False)

            params_units = [dist.unit.to_string() for dist in priors_dc.dists]
            params_twigs = [b.get_parameter(uniqueid=uniqueid, **_skip_filter_checks).twig for uniqueid in params_uniqueids]

            # NOTE: in dynesty we draw from the priors and pass the prior-transforms,
            # but do NOT include the lnprior term in lnlikelihood, so we pass
            # priors as []
            lnlikelihood_kwargs = {'b': _bsolver(b, solver, compute, []),
                                   'params_uniqueids': params_uniqueids,
                                   'compute': compute,
                                   'priors': [],
                                   'priors_combine': 'and',
                                   'solution': kwargs.get('solution', None),
                                   'compute_kwargs': {k:v for k,v in kwargs.items() if k in b.get_compute(compute=compute, **_skip_filter_checks).qualifiers},
                                   'custom_lnprobability_callable': kwargs.pop('custom_lnprobability_callable', None)}




            logger.debug("dynesty.NestedSampler(_lnprobability, _sample_ppf, log_kwargs, ptform_kwargs, ndim, nlive)")
            sampler = dynesty.NestedSampler(_lnprobability, _sample_ppf,
                                        logl_kwargs=lnlikelihood_kwargs, ptform_kwargs={'distributions_list': priors_dc.dists},
                                        ndim=len(params_uniqueids), nlive=int(kwargs.get('nlive')), pool=pool)

            sargs = {}
            sargs['maxiter'] = maxiter
            sargs['maxcall'] = kwargs.get('maxcall')

            # TODO: expose these via parameters?
            sargs['dlogz'] = kwargs.get('dlogz', 0.01)
            sargs['logl_max'] = kwargs.get('logl_max', np.inf)
            sargs['n_effective'] = kwargs.get('n_effective',np.inf)
            sargs['add_live'] = kwargs.get('add_live', True)
            # sargs['save_bounds'] = kwargs.get('save_bounds', True)
            # sargs['save_samples'] = kwargs.get('save_samples', True)


            sampler.run_nested(**sargs)
            for iter,result in enumerate(sampler.sample(**sargs)):
                # check for kill signal
                if kwargs.get('out_fname', False) and os.path.isfile(kwargs.get('out_fname')+'.kill'):
                    logger.warning("received kill signal, exiting sampler loop")
                    break

                progress = float(iter) / maxiter * 100

                if progress_every_niters == 0 and 'out_fname' in kwargs.keys():
                    fname = kwargs.get('out_fname') + '.progress'
                    f = open(fname, 'w')
                    f.write(str(progress))
                    f.close()


                if (progress_every_niters > 0 and (iter == 0 or iter % progress_every_niters == 0)) or iter == maxiter:
                    logger.info("dynesty: saving output from iteration {}".format(iter))

                    solution_ps = self._fill_solution(solution_ps, [_get_packetlist(sampler.results)], metawargs)

                    if 'out_fname' in kwargs.keys():
                        if iter == maxiter:
                            fname = kwargs.get('out_fname')
                        else:
                            fname = kwargs.get('out_fname') + '.progress'
                    else:
                        if iter == maxiter:
                            fname = '{}.ps'.format(solution)
                        else:
                            fname = '{}.progress.ps'.format(solution)

                    solution_ps.save(fname, compact=True, sort_by_context=False)


        else:
            # NOTE: because we overrode self._run_worker to skip loading the
            # bundle, b is just a json string here.  If we ever need the
            # bundle in here, just remove the override for self._run_worker.
            pool.wait()

        if pool is not None:
            pool.close()

        # restore previous MPI state
        mpi._within_mpirun = within_mpirun
        mpi._enabled = mpi_enabled

        if is_master:
            return _get_packetlist(sampler.results)
        return




class _ScipyOptimizeBaseBackend(BaseSolverBackend):
    """
    See <phoebe.parameters.solver.optimizer.nelder_mead>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_solver>
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
    def run_checks(self, b, solver, compute, **kwargs):
        solver_ps = b.get_solver(solver)
        if not len(solver_ps.get_value(qualifier='fit_parameters', fit_parameters=kwargs.get('fit_parameters', None), expand=True)):
            raise ValueError("cannot run scipy.optimize.minimize(method='nelder-mead') without any parameters in fit_parameters")


    def _get_packet_and_solution(self, b, solver, **kwargs):
        # NOTE: b, solver, compute, backend will be added by get_packet_and_solution
        solution_params = []

        solution_params += [_parameters.ArrayParameter(qualifier='fitted_uniqueids', value=[], advanced=True, readonly=True, description='uniqueids of parameters fitted by the minimizer')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_twigs', value=[], readonly=True, description='twigs of parameters fitted by the minimizer')]
        solution_params += [_parameters.SelectTwigParameter(qualifier='adopt_parameters', value=[], description='which of the parameters should be included when adopting the solution')]
        solution_params += [_parameters.BoolParameter(qualifier='adopt_distributions', value=False, description='whether to create a distribution (of delta functions of all parameters in adopt_parameters) when calling adopt_solution.')]
        solution_params += [_parameters.BoolParameter(qualifier='adopt_values', value=True, description='whether to update the parameter face-values (of all parameters in adopt_parameters) when calling adopt_solution.')]


        solution_params += [_parameters.StringParameter(qualifier='message', value='', readonly=True, description='message from the minimizer')]
        solution_params += [_parameters.IntParameter(qualifier='nfev', value=0, readonly=True, limits=(0,None), description='number of completed function evaluations (forward models)')]
        solution_params += [_parameters.IntParameter(qualifier='niter', value=0, readonly=True, limits=(0,None), description='number of completed iterations')]
        solution_params += [_parameters.BoolParameter(qualifier='success', value=False, readonly=True, description='whether the minimizer returned a success message')]
        solution_params += [_parameters.ArrayParameter(qualifier='initial_values', value=[], readonly=True, description='initial values before running the minimizer (in current default units of each parameter)')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_values', value=[], readonly=True, description='final values returned by the minimizer (in current default units of each parameter)')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_units', value=[], advanced=True, readonly=True, description='units of the fitted_values')]
        if kwargs.get('expose_lnlikelihoods', False):
            solution_params += [_parameters.FloatParameter(qualifier='initial_lnlikelihood', value=0.0, readonly=True, default_unit=u.dimensionless_unscaled, description='lnlikelihood of the initial_values')]
            solution_params += [_parameters.FloatParameter(qualifier='fitted_lnlikelihood', value=0.0, readonly=True, default_unit=u.dimensionless_unscaled, description='lnlikelihood of the fitted_values')]

        return kwargs, _parameters.ParameterSet(solution_params)

    def run_worker(self, b, solver, compute, **kwargs):
        if mpi.within_mpirun:
            raise NotImplementedError("mpi support for scipy.optimize not yet implemented")
            # TODO: we need to tell the workers to join the pool for time-parallelization?

        fit_parameters = kwargs.get('fit_parameters') # list of twigs
        initial_values = kwargs.get('initial_values') # dictionary
        priors = kwargs.get('priors')
        priors_combine = kwargs.get('priors_combine')

        params_uniqueids = []
        params_twigs = []
        p0 = []
        fitted_units = []
        for twig in fit_parameters:
            p = b.get_parameter(twig=twig, context=['component', 'dataset', 'feature', 'system'], **_skip_filter_checks)
            params_uniqueids.append(p.uniqueid)
            params_twigs.append(p.twig)
            p0.append(p.get_value())
            fitted_units.append(p.get_default_unit())

        # now override from initial values
        fitted_params_ps = b.filter(uniqueid=params_uniqueids, **_skip_filter_checks)
        for twig,v in initial_values.items():
            p = fitted_params_ps.get_parameter(twig=twig, **_skip_filter_checks)
            if p.uniqueid in params_uniqueids:
                index = params_uniqueids.index(p.uniqueid)
                if hasattr(v, 'unit'):
                    v = v.to(fitted_units[index]).value
                p0[index] = v
            else:
                logger.warning("ignoring {}={} in initial_values as was not found in fit_parameters".format(twig, value))

        compute_kwargs = {k:v for k,v in kwargs.items() if k in b.get_compute(compute=compute, **_skip_filter_checks).qualifiers}

        options = {k:v for k,v in kwargs.items() if k in self.valid_options}

        logger.debug("calling scipy.optimize.minimize(_lnprobability_negative, p0, method='{}', args=(b, {}, {}, {}, {}, {}), options={})".format(self.method, params_uniqueids, compute, priors, kwargs.get('solution', None), compute_kwargs, options))
        # TODO: would it be cheaper to pass the whole bundle (or just make one copy originally so we restore original values) than copying for each iteration?
        args = (_bsolver(b, solver, compute, priors), params_uniqueids, compute, priors, priors_combine, kwargs.get('solution', None), compute_kwargs, kwargs.pop('custom_lnprobability_callable', None))
        res = optimize.minimize(_lnprobability_negative, p0,
                                method=self.method,
                                args=args,
                                options=options)

        return_ = [{'qualifier': 'message', 'value': res.message},
                {'qualifier': 'nfev', 'value': res.nfev},
                {'qualifier': 'niter', 'value': res.nit},
                {'qualifier': 'success', 'value': res.success},
                {'qualifier': 'fitted_uniqueids', 'value': params_uniqueids},
                {'qualifier': 'fitted_twigs', 'value': params_twigs},
                {'qualifier': 'initial_values', 'value': p0},
                {'qualifier': 'fitted_values', 'value': res.x},
                {'qualifier': 'fitted_units', 'value': [u.to_string() for u in fitted_units]},
                {'qualifier': 'adopt_parameters', 'value': params_twigs, 'choices': params_twigs}]



        if kwargs.get('expose_lnlikelihoods', False):
            initial_lnlikelihood = _lnprobability(p0, *args)
            final_likelihood = _lnprobability(res.x, *args)

            return_ += [{'qualifier': 'initial_lnlikelihood', 'value': initial_lnlikelihood},
                         {'qualifier': 'fitted_lnlikelihood', 'value': final_likelihood}]



        return [return_]

class Nelder_MeadBackend(_ScipyOptimizeBaseBackend):
    @property
    def method(self):
        return 'nelder-mead'

    @property
    def valid_options(self):
        return ['maxiter', 'maxfev', 'xatol', 'fatol', 'adaptive']

class PowellBackend(_ScipyOptimizeBaseBackend):
    @property
    def method(self):
        return 'powell'

    @property
    def valid_options(self):
        return ['maxiter', 'maxfev', 'xtol', 'ftol']

class CgBackend(_ScipyOptimizeBaseBackend):
    @property
    def method(self):
        return 'cg'

    @property
    def valid_options(self):
        return ['maxiter', 'gtol', 'norm']



class Differential_EvolutionBackend(BaseSolverBackend):
    """
    See <phoebe.parameters.solver.optimizer.differential_evolution>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_solver>
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
    def run_checks(self, b, solver, compute, **kwargs):
        solver_ps = b.get_solver(solver)
        if not len(solver_ps.get_value(qualifier='fit_parameters', fit_parameters=kwargs.get('fit_parameters', None), expand=True)):
            raise ValueError("cannot run scipy.optimize.differential_evolution without any parameters in fit_parameters")


    def _get_packet_and_solution(self, b, solver, **kwargs):
        # NOTE: b, solver, compute, backend will be added by get_packet_and_solution
        solution_params = []

        solution_params += [_parameters.ArrayParameter(qualifier='fitted_uniqueids', value=[], advanced=True, readonly=True, description='uniqueids of parameters fitted by the minimizer')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_twigs', value=[], readonly=True, description='twigs of parameters fitted by the minimizer')]
        solution_params += [_parameters.SelectTwigParameter(qualifier='adopt_parameters', value=[], description='which of the parameters should be included when adopting the solution')]
        solution_params += [_parameters.BoolParameter(qualifier='adopt_distributions', value=False, description='whether to create a distribution (of delta functions of all parameters in adopt_parameters) when calling adopt_solution.')]
        solution_params += [_parameters.BoolParameter(qualifier='adopt_values', value=True, description='whether to update the parameter face-values (of all parameters in adopt_parameters) when calling adopt_solution.')]

        solution_params += [_parameters.StringParameter(qualifier='message', value='', readonly=True, description='message from the minimizer')]
        solution_params += [_parameters.IntParameter(qualifier='nfev', value=0, readonly=True, limits=(0,None), description='number of completed function evaluations (forward models)')]
        solution_params += [_parameters.IntParameter(qualifier='niter', value=0, readonly=True, limits=(0,None), description='number of completed iterations')]
        solution_params += [_parameters.BoolParameter(qualifier='success', value=False, readonly=True, description='whether the minimizer returned a success message')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_values', value=[],readonly=True,  description='final values returned by the minimizer (in current default units of each parameter)')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_units', value=[], advanced=True, readonly=True, description='units of the fitted_values')]
        solution_params += [_parameters.ArrayParameter(qualifier='bounds', value=kwargs.get('bounds', []), readonly=True, description='bound limits adopted and used internally.')]


        if kwargs.get('expose_lnlikelihoods', False):
            solution_params += [_parameters.FloatParameter(qualifier='initial_lnlikelihood', value=0.0, readonly=True, default_unit=u.dimensionless_unscaled, description='lnlikelihood of the initial_values')]
            solution_params += [_parameters.FloatParameter(qualifier='fitted_lnlikelihood', value=0.0, readonly=True, default_unit=u.dimensionless_unscaled, description='lnlikelihood of the fitted_values')]

        return kwargs, _parameters.ParameterSet(solution_params)

    # def _run_worker(self, packet):
    #     # here we'll override loading the bundle since it is not needed
    #     # in run_worker (for the workers.... note that the master
    #     # will enter run_worker through run, not here)
    #     return self.run_worker(**packet)

    def run_worker(self, b, solver, compute, **kwargs):
        def _get_bounds(param, dist, bounds_sigma):
            if dist is None:
                return param.limits

            return (dist.low, dist.high)

        if mpi.within_mpirun:
            pool = schwimmbad.MPIPool()
            is_master = pool.is_master()
        else:
            pool = schwimmbad.MultiPool()
            is_master = True

        # temporarily disable MPI within run_compute to disabled parallelizing
        # per-time.
        within_mpirun = mpi.within_mpirun
        mpi_enabled = mpi.enabled
        mpi._within_mpirun = False
        mpi._enabled = False

        if is_master:
            fit_parameters = kwargs.get('fit_parameters')

            params_uniqueids = []
            params_twigs = []
            params = []
            fitted_units = []
            for twig in fit_parameters:
                p = b.get_parameter(twig=twig, context=['component', 'dataset'], **_skip_filter_checks)
                params.append(p)
                params_uniqueids.append(p.uniqueid)
                params_twigs.append(p.twig)
                fitted_units.append(p.get_default_unit())

            bounds = kwargs.get('bounds')
            bounds_combine = kwargs.get('bounds_combine')
            bounds_sigma = kwargs.get('bounds_sigma')


            bounds_dc, uniqueids = b.get_distribution_collection(distribution=bounds,
                                                                 keys='uniqueid',
                                                                 combine=bounds_combine,
                                                                 include_constrained=False,
                                                                 to_univariates=True,
                                                                 to_uniforms=bounds_sigma,
                                                                 set_labels=False)

            # for each parameter, if a distribution is found in bounds_dict (from
            # the bounds parameter), then the bounds are adopted from that (taking
            # bounds_combine and bounds_sigma into account).  Otherwise, the limits
            # of the parameter itself are adopted.
            bounds = [_get_bounds(param, bounds_dc.dists[uniqueids.index(param.uniqueid)] if param.uniqueid in uniqueids else None, bounds_sigma) for param in params]

            compute_kwargs = {k:v for k,v in kwargs.items() if k in b.get_compute(compute=compute, **_skip_filter_checks).qualifiers}

            options = {k:v for k,v in kwargs.items() if k in ['strategy', 'maxiter', 'popsize']}

            logger.debug("calling scipy.optimize.differential_evolution(_lnprobability_negative, bounds={}, args=(b, {}, {}, {}, {}, {}), options={})".format(bounds, params_uniqueids, compute, [], kwargs.get('solution', None), compute_kwargs, options))
            # TODO: would it be cheaper to pass the whole bundle (or just make one copy originally so we restore original values) than copying for each iteration?
            args = (_bsolver(b, solver, compute, []), params_uniqueids, compute, [], 'first', kwargs.get('solution', None), compute_kwargs)
            res = optimize.differential_evolution(_lnprobability_negative, bounds,
                                    args=args,
                                    workers=pool.map, updating='deferred',
                                    **options)
        else:
            # NOTE: because we overrode self._run_worker to skip loading the
            # bundle, b is just a json string here.  If we ever need the
            # bundle in here, just remove the override for self._run_worker.
            pool.wait()

        if pool is not None:
            pool.close()

        # restore previous MPI state
        mpi._within_mpirun = within_mpirun
        mpi._enabled = mpi_enabled

        if is_master:
            # TODO: expose the adopted bounds?

            return_ = [{'qualifier': 'message', 'value': res.message},
                       {'qualifier': 'nfev', 'value': res.nfev},
                       {'qualifier': 'niter', 'value': res.nit},
                       {'qualifier': 'success', 'value': res.success},
                       {'qualifier': 'fitted_uniqueids', 'value': params_uniqueids},
                       {'qualifier': 'fitted_twigs', 'value': params_twigs},
                       {'qualifier': 'fitted_values', 'value': res.x},
                       {'qualifier': 'fitted_units', 'value': [u.to_string() for u in fitted_units]},
                       {'qualifier': 'adopt_parameters', 'value': params_twigs, 'choices': params_twigs},
                       {'qualifier': 'bounds', 'value': bounds}]

            if kwargs.get('expose_lnlikelihoods', False):
                initial_lnlikelihood = _lnprobability(False, *args)
                final_likelihood = _lnprobability(res.x, *args)

                return_ += [{'qualifier': 'initial_lnlikelihood', 'value': initial_lnlikelihood},
                            {'qualifier': 'fitted_lnlikelihood', 'value': final_likelihood}]



            return [return_]

        return