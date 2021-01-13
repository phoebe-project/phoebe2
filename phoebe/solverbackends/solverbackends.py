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
from phoebe.dependencies import nparray
from phoebe.helpers import get_emcee_object as _get_emcee_object
from phoebe import pool as _pool

from distutils.version import LooseVersion, StrictVersion
from copy import deepcopy as _deepcopy
import multiprocessing

from . import lc_geometry, rv_geometry
from .ebai import ebai_forward

try:
    import emcee
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

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _has_tqdm = False
else:
    _has_tqdm = True

try:
    from astropy.timeseries import BoxLeastSquares as _BoxLeastSquares
    from astropy.timeseries import LombScargle as _LombScargle
except ImportError:
    _use_astropy_timeseries = False
else:
    _use_astropy_timeseries = True

from scipy import optimize
from scipy.stats import binned_statistic

import logging
logger = logging.getLogger("SOLVER")
logger.addHandler(logging.NullHandler())

_skip_filter_checks = {'check_default': False, 'check_visible': False}

def _wrap_central_values(b, dc, uniqueids):
    ret = {}
    for dist, uniqueid in zip(dc.dists, uniqueids):
        param = b.get_parameter(uniqueid=uniqueid, **_skip_filter_checks)
        if param.default_unit.physical_type == 'angle':
            ret[uniqueid] = dist.median()
    return ret

def _bsolver(b, solver, compute, distributions, wrap_central_values={}):
    # TODO: OPTIMIZE exclude disabled datasets?
    # TODO: re-enable removing unused compute options - currently causes some constraints to fail
    # TODO: is it quicker to initialize a new bundle around b.exclude?  Or just leave everything?
    bexcl = b.copy()
    bexcl.remove_parameters_all(context=['model', 'solution', 'figure'], **_skip_filter_checks)
    if len(b.solvers) > 1:
        bexcl.remove_parameters_all(solver=[f for f in b.solvers if f!=solver and solver is not None], **_skip_filter_checks)
    bexcl.remove_parameters_all(distribution=[d for d in b.distributions if d not in distributions], **_skip_filter_checks)

    # set face-values to be central values for any angle parameters in init_from
    for uniqueid, value in wrap_central_values.items():
        # TODO: what to do if continue_from was used but constraint has since been flipped?
        bexcl.set_value(uniqueid=uniqueid, value=value, **_skip_filter_checks)

    bexcl.parse_solver_times(return_as_dict=False, set_compute_times=True)

    return bexcl


def _lnprobability(sampled_values, b, params_uniqueids, compute,
                  priors, priors_combine,
                  solution,
                  compute_kwargs={},
                  custom_lnprobability_callable=None,
                  failed_samples_buffer=False):


    def _return(lnprob, msg):
        if msg != 'success' and failed_samples_buffer is not False:
            msg_tuple = (_simplify_error_message(msg), sampled_values.tolist())
            if failed_samples_buffer.__class__.__name__ == 'ListProxy':
                failed_samples_buffer.append(msg_tuple)
            elif mpi._within_mpirun:
                # then emcee is in serial mode, run_compute is within mpi
                failed_samples_buffer.append(msg_tuple)
            else:
                # then emcee is using MPI so we need to pass the messages
                # through the MPI pool
                comm = _MPI.COMM_WORLD
                comm.ssend(msg_tuple, 0, tag=99999999)

        return lnprob

    # copy the bundle to make sure any changes by setting values/running models
    # doesn't affect the user-copy (or in other processors)
    b = b.copy()
    # prevent any *_around distributions from adjusting to the changes in
    # face-values
    b._within_solver = True
    if sampled_values is not False:
        for uniqueid, value in zip(params_uniqueids, sampled_values):
            try:
                b.set_value(uniqueid=uniqueid, value=value, run_checks=False, run_constraints=False, **_skip_filter_checks)
            except ValueError as err:
                logger.warning("received error while setting values: {}. lnprobability=-inf".format(err))
                return _return(-np.inf, str(err))

    # run delayed constraints and failed constraints would be run within calculate_lnp or run_compute,
    # but here we can catch the error in advance and return it appropriately
    try:
        b.run_delayed_constraints()
    except Exception as err:
        logger.warning("received error while running constraints: {}. lnprobability=-inf".format(err))
        return _return(-np.inf, str(err))

    try:
        b.run_failed_constraints()
    except Exception as err:
        logger.warning("received error while running constraints: {}. lnprobability=-inf".format(err))
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
        lnprob = lnpriors + b.calculate_lnlikelihood(model=solution, consider_gaussian_process=False)
    else:
        lnprob = custom_lnprobability_callable(b, model=solution, lnpriors=lnpriors, priors=priors, priors_combine=priors_combine)

    if np.isnan(lnprob):
        return _return(-np.inf, 'lnprobability returned nan')

    return _return(lnprob, 'success')

def _lnprobability_negative(sampled_values, b, params_uniqueids, compute,
                           priors, priors_combine,
                           solution,
                           compute_kwargs={},
                           custom_lnprobability_callable=None,
                           failed_samples_buffer=False):

    return -1 * _lnprobability(sampled_values, b, params_uniqueids, compute,
                              priors, priors_combine,
                              solution,
                              compute_kwargs,
                              custom_lnprobability_callable,
                              failed_samples_buffer)

def _sample_ppf(ppf_values, distributions_list):
    # NOTE: this will treat each item in the collection independently, ignoring any covariances

    x = np.empty_like(ppf_values)

    # TODO: replace with distl.sample_ppf_from_dists(distributions, values)
    # once implemented to support multivariate?

    for i,dist in enumerate(distributions_list):
        x[i] = dist.ppf(ppf_values[i])

    return x

def _get_combined_lc(b, datasets, combine, phase_component=None, mask=True, normalize=True, phase_sorted=False, phase_bin=False, warn_mask=False):
    times = np.array([])
    fluxes = np.array([])
    sigmas = np.array([])

    for dataset in datasets:
        lc_ps = b.get_dataset(dataset=dataset, **_skip_filter_checks)
        ds_fluxes = lc_ps.get_value(qualifier='fluxes', unit=u.W/u.m**2, **_skip_filter_checks)

        if not len(ds_fluxes):
            # then no observations here
            continue

        ds_times = lc_ps.get_value(qualifier='times', unit=u.d, **_skip_filter_checks)
        if len(ds_times) != len(ds_fluxes):
            raise ValueError("times and fluxes in dataset '{}' do not have same length".format(dataset))

        ds_sigmas = lc_ps.get_value(qualifier='sigmas', unit=u.W/u.m**2, **_skip_filter_checks)

        if len(ds_sigmas) == 0:
            # TODO: option for this???
            ds_sigmas = 0.001*fluxes.mean()*np.ones(len(fluxes))

        if normalize:
            if combine == 'max':
                flux_norm = np.nanmax(ds_fluxes)
            elif combine == 'median':
                flux_norm = np.nanmedian(ds_fluxes)
            else:
                raise NotImplementedError()

            ds_fluxes /= flux_norm
            ds_sigmas /= flux_norm

        mask_enabled = lc_ps.get_value(qualifier='mask_enabled', default=False, **_skip_filter_checks)
        if mask and mask_enabled:
            mask_phases = lc_ps.get_value(qualifier='mask_phases', **_skip_filter_checks)
            if len(mask_phases):
                if warn_mask:
                    logger.warning("applying mask_phases (may not be desired for finding eclipse edges - set mask_enabled=False to disable)")
                mask_t0 = lc_ps.get_value(qualifier='phases_t0', **_skip_filter_checks)
                # TODO:
                phases_for_mask = b.to_phase(ds_times, component=None, t0=mask_t0)

                inds = phase_mask_inds(phases_for_mask, mask_phases)

                ds_times = ds_times[inds]
                ds_fluxes = ds_fluxes[inds]
                ds_sigmas = ds_sigmas[inds]

        times = np.append(times, ds_times)
        fluxes = np.append(fluxes, ds_fluxes)
        sigmas = np.append(sigmas, ds_sigmas)


    phases = b.to_phase(times, component=phase_component, t0='t0_supconj')

    if phase_bin and phase_bin < len(times):
        logger.warning("binning input observations (len: {}) with {} bins (ignores sigmas)".format(len(times), phase_bin))
        fluxes_binned, phase_edges, binnumber = binned_statistic(phases, fluxes, statistic='median', bins=phase_bin)
        # NOTE: input sigmas are ignored
        sigmas_binned, phase_edges, binnumber = binned_statistic(phases, fluxes, statistic='std', bins=phase_bin)
        counts_binned, phase_edges, binnumber = binned_statistic(phases, fluxes, statistic='count', bins=phase_bin)
        counts_single_inds = np.where(counts_binned==0)[0]
        for i in np.where(counts_binned==0)[0]:
            # need to replace the sigma entry with the original observational sigma
            sigmas_binned[i] = sigmas[np.argmin(abs(phases-phase_edges[i]))]

        phases_binned = (phase_edges[1:] + phase_edges[:-1]) / 2.

        nans_inds = np.isnan(fluxes_binned)

        # NOTE: times array won't be the same size! (but we want the original
        # times array for t0_near_times in lc_geometry)
        return times, phases_binned[~nans_inds], fluxes_binned[~nans_inds], sigmas_binned[~nans_inds]

    elif phase_sorted:
        # binning would phase-sort anyways
        s = phases.argsort()
        times = times[s]
        phases = phases[s]
        fluxes = fluxes[s]
        sigmas = sigmas[s]

    return times, phases, fluxes, sigmas

def _get_combined_rv(b, datasets, components, phase_component=None, mask=True, normalize=False, mirror_secondary=False, phase_sorted=False, phase_bin=False):
    times = np.array([])
    rvs = np.array([])
    sigmas = np.array([])

    hier = b.hierarchy

    for i,comp in enumerate(components):
        c_times = np.array([])
        c_rvs = np.array([])
        c_sigmas = np.array([])

        for dataset in datasets:
            rvc_ps = b.get_dataset(dataset=dataset, component=comp, **_skip_filter_checks)
            rvc_rvs = rvc_ps.get_value(qualifier='rvs', unit=u.km/u.s, **_skip_filter_checks)
            if not len(rvc_rvs):
                # then no observations here
                continue

            rvc_times = rvc_ps.get_value(qualifier='times', unit=u.d, **_skip_filter_checks)
            if len(rvc_rvs) != len(rvc_times):
                raise ValueError("rv@{}@{} does not match length of times@{}@{}".format(comp, dataset, comp, dataset))

            rvc_sigmas = rvc_ps.get_value(qualifier='sigmas', unit=u.km/u.s, **_skip_filter_checks)
            if not len(rvc_sigmas):
                rvc_sigmas = np.full_like(rvc_rvs, fill_value=np.nan)

            mask_enabled = b.get_value(qualifier='mask_enabled', dataset=dataset, default=False, **_skip_filter_checks)
            if mask and mask_enabled:
                mask_phases = b.get_value(qualifier='mask_phases', dataset=dataset, **_skip_filter_checks)
                if len(mask_phases):
                    logger.warning("applying mask_phases - set mask_enabled=False to disable")
                    mask_t0 = b.get_value(qualifier='phases_t0', dataset=dataset, **_skip_filter_checks)
                    phases_for_mask = b.to_phase(rvc_times, component=None, t0=mask_t0)

                    inds = phase_mask_inds(phases_for_mask, mask_phases)

                    rvc_times = rvc_times[inds]
                    rvc_rvs = rvc_rvs[inds]
                    rvc_sigmas = rvc_sigmas[inds]

            c_times = np.append(c_times, rvc_times)
            c_rvs = np.append(c_rvs, rvc_rvs)
            c_sigmas = np.append(c_sigmas, rvc_sigmas)

        if normalize:
            c_rvs_max = abs(c_rvs).max()
            c_rvs /= c_rvs_max
            c_sigmas /= c_rvs_max

        if mirror_secondary and i==1:
            c_rvs *= -1

        times = np.append(times, c_times)
        rvs = np.append(rvs, c_rvs)
        sigmas = np.append(sigmas, c_sigmas)

    phases = b.to_phase(times, component=phase_component, t0='t0_supconj')



    if phase_bin and phase_bin < len(times):
        logger.warning("binning input observations (len: {}) with {} bins (ignores sigmas)".format(len(times), phase_bin))
        rvs_binned, phase_edges, binnumber = binned_statistic(phases, rvs, statistic='median', bins=phase_bin)
        # NOTE: input sigmas are ignored
        sigmas_binned, phase_edges, binnumber = binned_statistic(phases, rvs, statistic='std', bins=phase_bin)
        counts_binned, phase_edges, binnumber = binned_statistic(phases, fluxes, statistic='count', bins=phase_bin)
        counts_single_inds = np.where(counts_binned==0)[0]
        for i in np.where(counts_binned==0)[0]:
            # need to replace the sigma entry with the original observational sigma
            sigmas_binned[i] = sigmas[np.argmin(abs(phases-phase_edges[i]))]

        phases_binned = (phase_edges[1:] + phase_edges[:-1]) / 2.


        nans_inds = np.isnan(rvs_binned)

        # NOTE: times array won't be the same size! (but we want the original
        # times array for t0_near_times in lc_geometry)
        return times, phases_binned[~nans_inds], rv_binned[~nans_inds], sigmas_binned[~nans_inds]

    elif phase_sorted:
        # binning would phase-sort anyways
        s = phases.argsort()
        times = times[s]
        phases = phases[s]
        rvs = rvs[s]
        sigmas = sigmas[s]


    return times, phases, rvs, sigmas

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
            kwargs[param.qualifier] = param.get_value(expand=True, unit='solar', **{param.qualifier: kwargs.get(param.qualifier, None)})

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
        packet, solution_ps = self.get_packet_and_solution(b, solver, compute=compute, **kwargs)

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


class Lc_GeometryBackend(BaseSolverBackend):
    """
    See <phoebe.parameters.solver.estimator.lc_geometry>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_solver>
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
    def run_checks(self, b, solver, compute, **kwargs):
        solver_ps = b.get_solver(solver, **_skip_filter_checks)
        if not len(solver_ps.get_value(qualifier='lc_datasets', expand=True, lc_datasets=kwargs.get('lc_datasets', None))):
            raise ValueError("cannot run lc_geometry without any dataset in lc_datasets")

        # TODO: check to make sure fluxes exist, etc


    def _get_packet_and_solution(self, b, solver, **kwargs):
        # NOTE: b, solver, compute, backend will be added by get_packet_and_solution
        solution_params = []

        solution_params += [_parameters.StringParameter(qualifier='orbit', value='', readonly=True, description='orbit used for phasing the input light curve(s)')]
        solution_params += [_parameters.FloatArrayParameter(qualifier='input_phases', value=[], readonly=True, default_unit=u.dimensionless_unscaled, description='input phases (after binning, if applicable) used for geometry estimate')]
        solution_params += [_parameters.FloatArrayParameter(qualifier='input_fluxes', value=[], readonly=True, default_unit=u.dimensionless_unscaled, description='input fluxes (normalized per-dataset, after binning, if applicable) used for geometry estimate')]
        solution_params += [_parameters.FloatArrayParameter(qualifier='input_sigmas', value=[], readonly=True, default_unit=u.dimensionless_unscaled, description='input sigmas (after binning, if applicable) used for geometry estimate')]

        if kwargs.get('expose_model', True):
            solution_params += [_parameters.FloatArrayParameter(qualifier='analytic_phases', value=[], readonly=True, default_unit=u.dimensionless_unscaled, description='phases for analystic_fluxes')]
            solution_params += [_parameters.DictParameter(qualifier='analytic_fluxes', value={}, readonly=True, description='all GP models used to determine geometry estimate')]
            solution_params += [_parameters.StringParameter(qualifier='analytic_best_model', value='', readonly=True, description='which GP model was determined to be represent input_phases/fluxes')]

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
            raise NotImplementedError("mpi support for lc_geometry not yet implemented")
            # TODO: we need to tell the workers to join the pool for time-parallelization?

        lc_datasets = kwargs.get('lc_datasets') # NOTE: already expanded
        lc_combine = kwargs.get('lc_combine')
        orbit = kwargs.get('orbit')
        phase_bin = kwargs.get('phase_bin', False)
        if phase_bin:
            phase_bin = kwargs.get('phase_nbins')

        times, phases, fluxes, sigmas = _get_combined_lc(b, lc_datasets, lc_combine, phase_component=orbit, mask=True, normalize=True, phase_sorted=True, phase_bin=phase_bin, warn_mask=True)

        orbit_ps = b.get_component(component=orbit, **_skip_filter_checks)
        ecc_param = orbit_ps.get_parameter(qualifier='ecc', **_skip_filter_checks)
        per0_param = orbit_ps.get_parameter(qualifier='per0', **_skip_filter_checks)
        t0_supconj_param = orbit_ps.get_parameter(qualifier='t0_supconj', **_skip_filter_checks)

        period = orbit_ps.get_value(qualifier='period', **_skip_filter_checks)
        t0_supconj_old = orbit_ps.get_value(qualifier='t0_supconj', **_skip_filter_checks)

        diagnose = kwargs.get('diagnose', False)
        fit_result = lc_geometry.fit_lc(phases, fluxes, sigmas)
        eclipse_dict = lc_geometry.compute_eclipse_params(phases, fluxes, sigmas, fit_result=fit_result, diagnose=diagnose)

        edges = eclipse_dict.get('eclipse_edges')
        mask_phases = [(edges[0]-eclipse_dict.get('primary_width')*0.3, edges[1]+eclipse_dict.get('primary_width')*0.3), (edges[2]-eclipse_dict.get('secondary_width')*0.3, edges[3]+eclipse_dict.get('secondary_width')*0.3)]

        # TODO: update to use widths as well (or alternate based on ecc?)
        ecc, per0 = lc_geometry.ecc_w_from_geometry(eclipse_dict.get('secondary_position') - eclipse_dict.get('primary_position'), eclipse_dict.get('primary_width'), eclipse_dict.get('secondary_width'))

        # TODO: create parameters in the solver options if we want to expose these options to the user
        # if t0_near_times == True the computed t0 is adjusted to fall in time times array range
        t0_near_times = kwargs.get('t0_near_times', True)

        t0_supconj_new = lc_geometry.t0_from_geometry(eclipse_dict.get('primary_position'), times,
                                period=period, t0_supconj=t0_supconj_old, t0_near_times=t0_near_times)

        fitted_params = [t0_supconj_param, ecc_param, per0_param]
        fitted_params += b.filter(qualifier='mask_phases', dataset=lc_datasets, **_skip_filter_checks).to_list()

        fitted_uniqueids = [p.uniqueid for p in fitted_params]
        fitted_twigs = [p.twig for p in fitted_params]
        fitted_values = [t0_supconj_new, ecc, per0]
        fitted_values += [mask_phases for ds in lc_datasets]
        fitted_units = [u.d.to_string(), u.dimensionless_unscaled.to_string(), u.rad.to_string()]
        fitted_units += [u.dimensionless_unscaled.to_string() for ds in lc_datasets]

        return_ = [{'qualifier': 'primary_width', 'value': eclipse_dict.get('primary_width')},
                   {'qualifier': 'secondary_width', 'value': eclipse_dict.get('secondary_width')},
                   {'qualifier': 'primary_phase', 'value': eclipse_dict.get('primary_position')},
                   {'qualifier': 'secondary_phase', 'value': eclipse_dict.get('secondary_position')},
                   {'qualifier': 'primary_depth', 'value': eclipse_dict.get('primary_depth')},
                   {'qualifier': 'secondary_depth', 'value': eclipse_dict.get('secondary_depth')},
                   {'qualifier': 'eclipse_edges', 'value': eclipse_dict.get('eclipse_edges')},
                   {'qualifier': 'orbit', 'value': orbit},
                   {'qualifier': 'input_phases', 'value': phases},
                   {'qualifier': 'input_fluxes', 'value': fluxes},
                   {'qualifier': 'input_sigmas', 'value': sigmas},
                   {'qualifier': 'fitted_uniqueids', 'value': fitted_uniqueids},
                   {'qualifier': 'fitted_twigs', 'value': fitted_twigs},
                   {'qualifier': 'fitted_values', 'value': fitted_values},
                   {'qualifier': 'fitted_units', 'value': fitted_units},
                   {'qualifier': 'adopt_parameters', 'value': fitted_twigs[:3], 'choices': fitted_twigs},
                   ]

        if kwargs.get('expose_model', True):
            # then resample the models and store in the solution
            analytic_phases = np.linspace(-0.5, 0.5, 201)
            analytic_fluxes = {}
            for model, params in fit_result['fits'].items():
                analytic_fluxes[model] = getattr(lc_geometry, 'const' if model=='C' else model.lower())(analytic_phases, *params[0])

            return_ += [{'qualifier': 'analytic_phases', 'value': analytic_phases},
                        {'qualifier': 'analytic_fluxes', 'value': analytic_fluxes},
                        {'qualifier': 'analytic_best_model', 'value': fit_result['best_fit']}
                        ]


        return [return_]


class Rv_GeometryBackend(BaseSolverBackend):
    """
    See <phoebe.parameters.solver.estimator.rvc_geometry>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_solver>
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
    def run_checks(self, b, solver, compute, **kwargs):
        solver_ps = b.get_solver(solver=solver, **_skip_filter_checks)
        if not len(solver_ps.get_value(qualifier='rv_datasets', expand=True, rv_datasets=kwargs.get('rv_datasets', None))):
            raise ValueError("cannot run rv_geometry without any dataset in rv_datasets")

        # TODO: check to make sure rvs exist, etc


    def _get_packet_and_solution(self, b, solver, **kwargs):
        # NOTE: b, solver, compute, backend will be added by get_packet_and_solution
        solution_params = []

        # solution_params += [_parameters.StringParameter(qualifier='rv', value='', readonly=True, description='dataset used for RV estimation')]
        solution_params += [_parameters.StringParameter(qualifier='orbit', value='', readonly=True, description='orbit used for RV estimation')]

        # TODO: one for each component
        orbit = kwargs.get('orbit')
        starrefs = b.hierarchy.get_children_of(orbit)
        for starref in starrefs:
            solution_params += [_parameters.FloatArrayParameter(qualifier='input_phases', component=starref, value=[], readonly=True, default_unit=u.dimensionless_unscaled, description='input phases (after binning, if applicable) for geometry estimate')]
            solution_params += [_parameters.FloatArrayParameter(qualifier='input_rvs', component=starref, value=[], readonly=True, default_unit=u.km/u.s, description='input RVs (after binning, if applicable) used for geometry estimate')]
            solution_params += [_parameters.FloatArrayParameter(qualifier='input_sigmas', component=starref, value=[], readonly=True, default_unit=u.km/u.s, description='input sigmas (after binning, if applicable) used for geometry estimate')]

            if kwargs.get('expose_model', True):
                solution_params += [_parameters.FloatArrayParameter(qualifier='analytic_rvs', component=starref, value=[], readonly=True, default_unit=u.km/u.s, description='analytic RVs determined by geometry estimate')]

        if kwargs.get('expose_model', True):
            solution_params += [_parameters.FloatArrayParameter(qualifier='analytic_phases', value=[], readonly=True, default_unit=u.dimensionless_unscaled, description='phases for analytic_rvs')]

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
            raise NotImplementedError("mpi support for rv_geometry not yet implemented")
            # TODO: we need to tell the workers to join the pool for time-parallelization?

        orbit = kwargs.get('orbit')
        starrefs = b.hierarchy.get_children_of(orbit)

        phase_bin = kwargs.get('phase_bin', False)
        if phase_bin:
            phase_bin = kwargs.get('phase_nbins')

        for i,starref in enumerate(starrefs):
            times, phases, rvs, sigmas = _get_combined_rv(b, kwargs.get('rv_datasets'), components=[starref], phase_component=kwargs.get('orbit'), mask=True, normalize=False, mirror_secondary=False, phase_sorted=False, phase_bin=phase_bin)

            s = np.argsort(phases)
            if i==0:
                if len(phases):
                    rv1data = np.vstack((phases[s], rvs[s], sigmas[s])).T
                else:
                    rv1data = None
            else:
                if len(phases):
                    rv2data = np.vstack((phases[s], rvs[s], sigmas[s])).T
                else:
                    rv2data = None

        if rv1data is None and rv2data is None:
            raise ValueError("no rv data found, cannot run rv_geometry")

        period = b.get_value(qualifier='period', component=orbit, context='component', unit=u.d, **_skip_filter_checks)

        est_dict = rv_geometry.estimate_rv_parameters(rv1data, rv2data)
        est_dict['t0_supconj'] = b.to_time(est_dict['ph_supconj'], component=orbit, t0='t0_supconj')

        # est_dict['period']
        # est_dict['t0_supconj']
        # est_dict['q']
        # est_dict['asini']  (list of a1sini, a2sini)
        # est_dict['vgamma']
        # est_dict['ecc']
        # est_dict['per0']

        orbit_ps = b.get_component(component=orbit, **_skip_filter_checks)

        t0_supconj_param = orbit_ps.get_parameter(qualifier='t0_supconj', **_skip_filter_checks)
        ecc_param = orbit_ps.get_parameter(qualifier='ecc', **_skip_filter_checks)
        per0_param = orbit_ps.get_parameter(qualifier='per0', **_skip_filter_checks)

        vgamma_param = b.get_parameter(qualifier='vgamma', context='system', **_skip_filter_checks)

        if rv2data is None:
            # then we have an SB1 system with only primary RVs
            asini_param = b.get_parameter(qualifier='asini', component=starrefs[0], context='component', **_skip_filter_checks)

            fitted_params = [t0_supconj_param, asini_param, ecc_param, per0_param, vgamma_param]
            fitted_values = [est_dict.get(p.qualifier) if p.qualifier != 'asini' else est_dict.get('asini')[0]*period for p in fitted_params]
            fitted_units = [u.d.to_string(), u.km.to_string(), u.dimensionless_unscaled.to_string(), u.rad.to_string(), (u.km/u.s).to_string()]

        elif rv1data is None:
            # then we have an SB1 system with only secondary RVs
            asini_param = b.get_parameter(qualifier='asini', component=starrefs[1], context='component', **_skip_filter_checks)

            fitted_params = [t0_supconj_param, asini_param, ecc_param, per0_param, vgamma_param]
            fitted_values = [est_dict.get(p.qualifier) if p.qualifier != 'asini' else est_dict.get('asini')[1]*period for p in fitted_params]
            fitted_units = [u.d.to_string(), u.km.to_string(), u.dimensionless_unscaled.to_string(), u.rad.to_string(), (u.km/u.s).to_string()]

        else:
            # then we have an SB2 system
            asini_param = orbit_ps.get_parameter(qualifier='asini', **_skip_filter_checks)
            q_param = orbit_ps.get_parameter(qualifier='q', **_skip_filter_checks)

            fitted_params = [t0_supconj_param, q_param, asini_param, ecc_param, per0_param, vgamma_param]
            fitted_values = [est_dict.get(p.qualifier) if p.qualifier != 'asini' else np.nansum(est_dict.get('asini'))*period for p in fitted_params]
            fitted_units = [u.d.to_string(), u.dimensionless_unscaled.to_string(), u.km.to_string(), u.dimensionless_unscaled.to_string(), u.rad.to_string(), (u.km/u.s).to_string()]


        fitted_uniqueids = [p.uniqueid for p in fitted_params]
        fitted_twigs = [p.twig for p in fitted_params]

        return_ = [
                     {'qualifier': 'input_phases', 'component': starrefs[0], 'value': rv1data[:,0] if rv1data is not None else []},
                     {'qualifier': 'input_rvs', 'component': starrefs[0], 'value': rv1data[:,1] if rv1data is not None else []},
                     {'qualifier': 'input_sigmas', 'component': starrefs[0], 'value': rv1data[:,2] if rv1data is not None else []},
                     {'qualifier': 'input_phases', 'component': starrefs[1], 'value': rv2data[:,0] if rv2data is not None else []},
                     {'qualifier': 'input_rvs', 'component': starrefs[1], 'value': rv2data[:,1] if rv2data is not None else []},
                     {'qualifier': 'input_sigmas', 'component': starrefs[1], 'value': rv2data[:,2] if rv2data is not None else []},
                     {'qualifier': 'orbit', 'value': orbit},
                     {'qualifier': 'fitted_uniqueids', 'value': fitted_uniqueids},
                     {'qualifier': 'fitted_twigs', 'value': fitted_twigs},
                     {'qualifier': 'fitted_values', 'value': fitted_values},
                     {'qualifier': 'fitted_units', 'value': fitted_units},
                     {'qualifier': 'adopt_parameters', 'value': '*', 'choices': fitted_twigs},
                    ]

        if kwargs.get('expose_model', True):
            analytic_phases = np.linspace(-0.5, 0.5, 201)
            ph_supconj = b.to_phase(est_dict['t0_supconj'])
            if rv1data is not None:
                analytic_rv1 = rv_geometry.rv_model(analytic_phases, 1., est_dict['per0'], est_dict['ecc'], est_dict['asini'], est_dict['vgamma'], est_dict['ph_supconj'], component=1)
            else:
                analytic_rv1 = []
            if rv2data is not None:
                analytic_rv2 = rv_geometry.rv_model(analytic_phases, 1., est_dict['per0'], est_dict['ecc'], est_dict['asini'], est_dict['vgamma'], est_dict['ph_supconj'], component=2)
            else:
                analytic_rv2 = []

            return_ += [
                         {'qualifier': 'analytic_phases', 'value': analytic_phases},
                         {'qualifier': 'analytic_rvs', 'component': starrefs[0], 'value': analytic_rv1},
                         {'qualifier': 'analytic_rvs', 'component': starrefs[1], 'value': analytic_rv2},
                        ]

        return [return_]



class _PeriodogramBaseBackend(BaseSolverBackend):
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

    def get_observations(self, b, **kwargs):
        raise NotImplementedError("get_observations not implemented for {}".format(self.__class__.__name__))
        # return times, y, sigmas

    def run_worker(self, b, solver, compute=None, **kwargs):
        if mpi.within_mpirun:
            raise NotImplementedError("mpi support for periodograms not yet implemented")
            # TODO: we need to tell the workers to join the pool for time-parallelization?

        algorithm = kwargs.get('algorithm')
        component = kwargs.get('component')

        times, y, sigmas = self.get_observations(b, **kwargs)

        sample_mode = kwargs.get('sample_mode')

        if algorithm == 'bls':
            model = _BoxLeastSquares(times, y, dy=sigmas)
            # https://docs.astropy.org/en/stable/api/astropy.timeseries.BoxLeastSquares.html#astropy.timeseries.BoxLeastSquares.autoperiod
            # https://docs.astropy.org/en/stable/api/astropy.timeseries.BoxLeastSquares.html#astropy.timeseries.BoxLeastSquares.period
            # NOTE: duration will be in days (solar units)
            power_kwargs = {'duration': kwargs.get('duration'), 'objective': kwargs.get('objective')}
            autopower_kwargs = _deepcopy(power_kwargs)
            autopower_kwargs['minimum_n_transit'] = kwargs.get('minimum_n_cycles')
            sample = kwargs.get('sample_periods')
            if isinstance(sample, nparray.ndarray):
                sample = sample.array
        elif algorithm == 'ls':
            model = _LombScargle(times, y, dy=sigmas)
            # https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.autopower
            # https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.power
            power_kwargs = {}
            autopower_kwargs = _deepcopy(power_kwargs)
            autopower_kwargs['samples_per_peak'] = kwargs.get('samples_per_peak')
            autopower_kwargs['nyquist_factor'] = kwargs.get('nyquist_factor')
            # require at least 2 full cycles (assuming 2 eclipses or 2 RV crossings per cycle)
            autopower_kwargs['minimum_frequency'] = 1./((times.max()-times.min())/4)
            autopower_kwargs['maximum_frequency'] = 1./((times.max()-times.min())/len(times))
            sample_periods = kwargs.get('sample_periods')
            if isinstance(sample_periods, nparray.ndarray):
                sample_periods = sample_periods.array
            sample = 1./sample_periods
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

class Lc_PeriodogramBackend(_PeriodogramBaseBackend):
    """
    See <phoebe.parameters.solver.estimator.lc_periodogram>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_solver>
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
    def run_checks(self, b, solver, compute, **kwargs):
        if not _use_astropy_timeseries:
            raise ImportError("astropy.timeseries not installed (requires astropy 3.2+)")

        solver_ps = b.get_solver(solver=solver, **_skip_filter_checks)
        if not len(solver_ps.get_value(qualifier='lc_datasets', expand=True, lc_datasets=kwargs.get('lc_datasets', None))):
            raise ValueError("cannot run lc_periodogram without any dataset in lc_datasets")

        # TODO: check to make sure fluxes exist, etc

    def get_observations(self, b, **kwargs):
        times, phases, fluxes, sigmas = _get_combined_lc(b, kwargs.get('lc_datasets'), kwargs.get('lc_combine'), mask=False, normalize=True, phase_sorted=False)
        return times, fluxes, sigmas

class Rv_PeriodogramBackend(_PeriodogramBaseBackend):
    """
    See <phoebe.parameters.solver.estimator.rv_periodogram>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_solver>
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
    def run_checks(self, b, solver, compute, **kwargs):
        if not _use_astropy_timeseries:
            raise ImportError("astropy.timeseries not installed (requires astropy 3.2+)")

        solver_ps = b.get_solver(solver=solver, **_skip_filter_checks)
        if not len(solver_ps.get_value(qualifier='rv_datasets', expand=True, rv_datasets=kwargs.get('rv_datasets', None))):
            raise ValueError("cannot run rv_periodogram without any dataset in rv_datasets")

        # TODO: check to make sure rvs exist, etc

    def get_observations(self, b, **kwargs):
        times, phases, rvs, sigmas = _get_combined_rv(b, kwargs.get('rv_datasets'), components=b.hierarchy.get_children_of(kwargs.get('component', None)), phase_component=kwargs.get('component'), mask=False, normalize=True, mirror_secondary=True, phase_sorted=False)

        # print("***", times.shape, rvs.shape)
        # import matplotlib.pyplot as plt
        # plt.plot(times, rvs, '.')
        # plt.show()

        return times, rvs, sigmas

class EbaiBackend(BaseSolverBackend):
    """
    See <phoebe.parameters.solver.estimator.ebai>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_solver>
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
    def run_checks(self, b, solver, compute, **kwargs):
        solver_ps = b.get_solver(solver=solver, **_skip_filter_checks)
        if not len(solver_ps.get_value(qualifier='lc_datasets', expand=True, lc_datasets=kwargs.get('lc_datasets', None))):
            raise ValueError("cannot run ebai without any dataset in lc_datasets")

        # TODO: check to make sure fluxes exist, etc


    def _get_packet_and_solution(self, b, solver, **kwargs):
        # NOTE: b, solver, compute, backend will be added by get_packet_and_solution
        solution_params = []

        solution_params += [_parameters.StringParameter(qualifier='orbit', value='', readonly=True, description='orbit used for phasing the input light curve(s)')]
        solution_params += [_parameters.FloatArrayParameter(qualifier='input_phases', value=[], readonly=True, default_unit=u.dimensionless_unscaled, description='input phases (after binning, if applicable) used for determining ebai_phases/ebai_fluxes')]
        solution_params += [_parameters.FloatArrayParameter(qualifier='input_fluxes', value=[], readonly=True, default_unit=u.dimensionless_unscaled, description='input fluxes (after binning, if applicable) used for determining ebai_phases/ebai_fluxes')]
        solution_params += [_parameters.FloatArrayParameter(qualifier='input_sigmas', value=[], readonly=True, default_unit=u.dimensionless_unscaled, description='input sigmas (after binning, if applicable) used for determining ebai_phases/ebai_fluxes')]

        solution_params += [_parameters.FloatArrayParameter(qualifier='ebai_phases', value=[], readonly=True, default_unit=u.dimensionless_unscaled, description='input phases to ebai')]
        solution_params += [_parameters.FloatArrayParameter(qualifier='ebai_fluxes', value=[], readonly=True, default_unit=u.dimensionless_unscaled, description='input fluxes to ebai')]

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
            raise NotImplementedError("mpi support for ebai not yet implemented")
            # TODO: we need to tell the workers to join the pool for time-parallelization?

        lc_datasets = kwargs.get('lc_datasets') # NOTE: already expanded
        lc_combine = kwargs.get('lc_combine')
        orbit = kwargs.get('orbit')

        orbit_ps = b.get_component(component=orbit, **_skip_filter_checks)

        phase_bin = kwargs.get('phase_bin', False)
        if phase_bin:
            phase_bin = kwargs.get('phase_nbins')

        times, phases, fluxes, sigmas = _get_combined_lc(b, lc_datasets, lc_combine, phase_component=orbit, mask=True, normalize=True, phase_sorted=True, phase_bin=phase_bin)

        teffratio_param = orbit_ps.get_parameter(qualifier='teffratio', **_skip_filter_checks)
        requivsumfrac_param = orbit_ps.get_parameter(qualifier='requivsumfrac', **_skip_filter_checks)
        esinw_param = orbit_ps.get_parameter(qualifier='esinw', **_skip_filter_checks)
        ecosw_param = orbit_ps.get_parameter(qualifier='ecosw', **_skip_filter_checks)
        incl_param = orbit_ps.get_parameter(qualifier='incl', **_skip_filter_checks)
        t0_supconj_param = orbit_ps.get_parameter(qualifier='t0_supconj', **_skip_filter_checks)

        # TODO: cleanup this logic a bit
        lc_geom_dict = lc_geometry.estimate_eclipse_positions_widths(phases, fluxes)
        if np.max(lc_geom_dict.get('ecl_widths', [])) > 0.25:
            logger.warning("ebai: eclipse width over 0.25 detected.  Returning all nans")
            t0_supconj = np.nan
            teffratio = np.nan
            requivsumfrac = np.nan
            esinw = np.nan
            ecosw = np.nan
            sini = np.nan
        else:
            ecl_positions = lc_geom_dict.get('ecl_positions')
            # assume primary is close to zero?
            pshift = ecl_positions[np.argmin(abs(np.array(ecl_positions)))]
            fit_result = lc_geometry.fit_lc(phases-pshift, fluxes, sigmas)
            best_fit = fit_result['best_fit']
            ebai_phases = np.linspace(-0.5,0.5,201)
            ebai_fluxes = getattr(lc_geometry, 'const' if best_fit=='C' else best_fit.lower())(ebai_phases, *fit_result['fits'][best_fit][0])
            fluxes /= ebai_fluxes.max()
            ebai_fluxes /= ebai_fluxes.max()

            # update to t0_supconj based on pshift
            t0_supconj = t0_supconj_param.get_value(unit=u.d) + (pshift * orbit_ps.get_value(qualifier='period', unit=u.d, **_skip_filter_checks))

            # run ebai on polyfit sampled fluxes
            teffratio, requivsumfrac, esinw, ecosw, sini = ebai_forward(ebai_fluxes)

        fitted_params = [t0_supconj_param, teffratio_param, requivsumfrac_param, esinw_param, ecosw_param, incl_param]
        fitted_uniqueids = [p.uniqueid for p in fitted_params]
        fitted_twigs = [p.twig for p in fitted_params]
        fitted_values = [t0_supconj, teffratio, requivsumfrac, esinw, ecosw, np.arcsin(sini)]
        fitted_units = [u.d.to_string(), u.dimensionless_unscaled.to_string(), u.dimensionless_unscaled.to_string(), u.dimensionless_unscaled.to_string(), u.dimensionless_unscaled.to_string(), u.rad.to_string()]

        return [[{'qualifier': 'orbit', 'value': orbit},
                 {'qualifier': 'input_phases', 'value': ((phases-pshift+0.5) % 1) - 0.5},
                 {'qualifier': 'input_fluxes', 'value': fluxes},
                 {'qualifier': 'input_sigmas', 'value': sigmas},
                 {'qualifier': 'ebai_phases', 'value': ebai_phases},
                 {'qualifier': 'ebai_fluxes', 'value': ebai_fluxes},
                 {'qualifier': 'fitted_uniqueids', 'value': fitted_uniqueids},
                 {'qualifier': 'fitted_twigs', 'value': fitted_twigs},
                 {'qualifier': 'fitted_values', 'value': fitted_values},
                 {'qualifier': 'fitted_units', 'value': fitted_units},
                 {'qualifier': 'adopt_parameters', 'value': fitted_twigs, 'choices': fitted_twigs},
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
            raise ImportError("could not import emcee")

        try:
            if LooseVersion(emcee.__version__) < LooseVersion("3.0.0"):
                raise ImportError("emcee backend requires emcee 3.0+, {} found".format(emcee.__version__))
        except ValueError:
            # see https://github.com/phoebe-project/phoebe2/issues/378
            raise ImportError("emcee backend requires a stable release of emcee 3.0+, {} found".format(emcee.__version__))

        solver_ps = b.get_solver(solver=solver, **_skip_filter_checks)
        if not len(solver_ps.get_value(qualifier='init_from', init_from=kwargs.get('init_from', None), **_skip_filter_checks)) and solver_ps.get_value(qualifier='continue_from', continue_from=kwargs.get('continue_from', None), **_skip_filter_checks)=='None':
            raise ValueError("cannot run emcee without any distributions in init_from")

        # require sigmas for all enabled datasets
        datasets = b.filter(compute=compute, qualifier='enabled', value=True).datasets
        for sigma_param in b.filter(qualifier='sigmas', dataset=datasets, context='dataset', check_visible=True, check_default=True).to_list():
            if not len(sigma_param.get_value()):
                times = b.get_value(qualifier='times', dataset=sigma_param.dataset, component=sigma_param.component, context='dataset', **_skip_filter_checks)
                if len(times):
                    raise ValueError("emcee requires sigmas for all datasets where times exist (not found for {})".format(sigma_param.twig))



    def _get_packet_and_solution(self, b, solver, **kwargs):
        # NOTE: b, solver, compute, backend will be added by get_packet_and_solution

        solution_params = []
        solution_params += [_parameters.DictParameter(qualifier='wrap_central_values', value={}, advanced=True, readonly=True, description='Central values adopted for all parameters in init_from that allow angle-wrapping.  Sampled values are not allowed beyond +/- pi of the central value.')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_uniqueids', value=[], advanced=True, readonly=True, description='uniqueids of parameters fitted by the sampler')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_twigs', value=[], readonly=True, description='twigs of parameters fitted by the sampler')]
        solution_params += [_parameters.ArrayParameter(qualifier='fitted_units', value=[], advanced=True, readonly=True, description='units of parameters fitted by the sampler')]
        solution_params += [_parameters.SelectTwigParameter(qualifier='adopt_parameters', value=[], description='which of the parameters should be included when adopting (and plotting) the solution')]
        solution_params += [_parameters.BoolParameter(qualifier='adopt_distributions', value=True, description='whether to create a distribution (of all parameters in adopt_parameters according to distributions_convert) when calling adopt_solution.')]
        solution_params += [_parameters.ChoiceParameter(qualifier='distributions_convert', value='mvsamples', choices=['mvsamples', 'mvhistogram', 'mvgaussian', 'samples', 'histogram', 'gaussian'], description='type of distribution to use when calling adopt_solution, get_distribution_collection, or plot. mvsamples: chains are stored directly and used for sampling with a KDE generated on-the-fly to compute probabilities.  mvhistogram: chains are binned according to distributions_bins and stored as an n-dimensional histogram.  mvgaussian: a multivariate gaussian is fitted to the samples, use only if distribution is sufficiently represented by gaussians.  samples: a univariate representation of mvsamples.  histogram: a univariate representation of mvhistogram.  gaussian: a univariate representation of mvgaussian.')]
        solution_params += [_parameters.IntParameter(visible_if='distributions_convert:mvhistogram|histogram', qualifier='distributions_bins', value=20, limits=(5,1000), description='number of bins to use for the distribution when calling adopt_solution, get_distribution_collection, or plot.')]
        solution_params += [_parameters.BoolParameter(qualifier='adopt_values', value=True, description='whether to update the parameter face-values (of the means of all parameters in adopt_parameters) when calling adopt_solution.')]

        solution_params += [_parameters.IntParameter(qualifier='niters', value=0, readonly=True, description='Completed number of iterations')]
        solution_params += [_parameters.IntParameter(qualifier='nwalkers', value=0, readonly=True, description='Number of walkers in samples')]

        solution_params += [_parameters.ArrayParameter(qualifier='samples', value=[], readonly=True, description='MCMC samples with shape (niters, nwalkers, len(fitted_twigs))')]
        if kwargs.get('expose_failed', True):
            solution_params += [_parameters.DictParameter(qualifier='failed_samples', value={}, readonly=True, description='MCMC samples that returned lnprobability=-inf.  Dictionary keys are the messages with values being an array with shape (N, len(fitted_twigs))')]
        solution_params += [_parameters.ArrayParameter(qualifier='lnprobabilities', value=[], readonly=True, description='log probabilities with shape (niters, nwalkers)')]

        # solution_params += [_parameters.ArrayParameter(qualifier='accepteds', value=[], description='whether each iteration was an accepted move with shape (niters)')]
        solution_params += [_parameters.ArrayParameter(qualifier='acceptance_fractions', value=[], readonly=True, description='fraction of proposed steps that were accepted with shape (nwalkers)')]

        solution_params += [_parameters.ArrayParameter(qualifier='autocorr_times', value=[], readonly=True, description='measured autocorrelation time with shape (len(fitted_twigs)) before applying burnin/thin.  To access with a custom burnin/thin, see phoebe.helpers.get_emcee_object_from_solution')]
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
            return_ = [{'qualifier': 'wrap_central_values', 'value': wrap_central_values},
                     {'qualifier': 'fitted_uniqueids', 'value': params_uniqueids},
                     {'qualifier': 'fitted_twigs', 'value': params_twigs},
                     {'qualifier': 'fitted_units', 'value': params_units},
                     {'qualifier': 'adopt_parameters', 'value': params_twigs, 'choices': params_twigs},
                     {'qualifier': 'niters', 'value': samples.shape[0]},
                     {'qualifier': 'nwalkers', 'value': samples.shape[1]},
                     {'qualifier': 'samples', 'value': samples},
                     {'qualifier': 'lnprobabilities', 'value': lnprobabilities},
                     {'qualifier': 'acceptance_fractions', 'value': acceptance_fractions},
                     {'qualifier': 'autocorr_times', 'value': autocorr_times},
                     {'qualifier': 'burnin', 'value': burnin},
                     {'qualifier': 'thin', 'value': thin},
                     {'qualifier': 'progress', 'value': progress}]

            if expose_failed:
                failed_samples = _deepcopy(continued_failed_samples)
                if expose_failed:
                    for msg, fsamples in failed_samples_buffer:
                        failed_samples[msg] = failed_samples.get(msg, []) + [fsamples]

                return_ += [{'qualifier': 'failed_samples', 'value': failed_samples}]

            return [return_]

        within_mpirun = mpi.within_mpirun
        mpi_enabled = mpi.enabled

        # emcee handles workers itself.  So here we'll just take the workers
        # from our own waiting loop in phoebe's __init__.py and subscribe them
        # to emcee's pool.
        if mpi.within_mpirun:
            # TODO: decide whether to use MPI for emcee (via pool) or pass None
            # to allow per-model parallelization
            global failed_samples_buffer
            failed_samples_buffer = []

            if mpi.nprocs > kwargs.get('nwalkers') and b.get_compute(compute=compute, **_skip_filter_checks).kind == 'phoebe':
                logger.info("nprocs > nwalkers: using per-time parallelization and emcee in serial")

                # we'll keep MPI at the per-compute level, so we'll pass
                # pool=None to emcee and immediately release all other processors
                # to await compute jobs
                pool = None
                is_master = mpi.myrank == 0

            else:
                logger.info("nprocs <= nwalkers: handling MPI within emcee, disabling per-time parallelization")

                global _MPI
                from mpi4py import MPI as _MPI # needed for the cost-function to send failed samples

                def mpi_failed_samples_callback(result):
                    global failed_samples_buffer
                    if isinstance(result, tuple):
                        failed_samples_buffer.append(result)
                        return False
                    else:
                        return True

                pool = _pool.MPIPool(callback=mpi_failed_samples_callback)
                is_master = pool.is_master()

                # temporarily disable MPI within run_compute to disabled parallelizing
                # per-time.
                mpi._within_mpirun = False
                mpi._enabled = False


        else:
            logger.info("using multiprocessing pool for emcee")

            pool = _pool.MultiPool()
            failed_samples_buffer = multiprocessing.Manager().list()
            is_master = True



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

                wrap_central_values = _wrap_central_values(b, dc, params_uniqueids)

                p0 = dc.sample(size=nwalkers).T
                params_units = [dist.unit.to_string() for dist in dc.dists]

                continued_failed_samples = {}
                start_iteration = 0
            else:
                # ignore the value from init_from (hidden parameter)
                init_from = []
                continue_from_ps = kwargs.get('continue_from_ps', b.filter(context='solution', solution=continue_from, **_skip_filter_checks))
                wrap_central_values = continue_from_ps.get_value(qualifier='wrap_central_values', **_skip_filter_checks)
                params_uniqueids = continue_from_ps.get_value(qualifier='fitted_uniqueids', **_skip_filter_checks)
                params_units = continue_from_ps.get_value(qualifier='fitted_units', **_skip_filter_checks)
                continued_samples = continue_from_ps.get_value(qualifier='samples', **_skip_filter_checks)
                expose_failed = 'failed_samples' in continue_from_ps.qualifiers
                kwargs['expose_failed'] = expose_failed # needed for _get_packet_and_solution
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

                # fake a backend object from the previous solution so that emcee
                # can continue from where it left off and still compute
                # autocorrelation times, etc.
                esargs['backend'] = _get_emcee_object(continued_samples, continued_lnprobabilities, continued_acceptance_fractions)
                p0 = continued_samples[-1].T
                # p0 [parameter, walkers]
                nwalkers = int(p0.shape[-1])

                start_iteration = continued_lnprobabilities.shape[0]

            params_twigs = [b.get_parameter(uniqueid=uniqueid, **_skip_filter_checks).twig for uniqueid in params_uniqueids]

            esargs['pool'] = pool
            esargs['nwalkers'] = nwalkers
            esargs['ndim'] = len(params_uniqueids)
            esargs['log_prob_fn'] = _lnprobability
            # esargs['a'] = kwargs.pop('a', None),
            # esargs['moves'] = kwargs.pop('moves', None)
            # esargs['args'] = None

            esargs['kwargs'] = {'b': _bsolver(b, solver, compute, init_from+priors, wrap_central_values),
                                'params_uniqueids': params_uniqueids,
                                'compute': compute,
                                'priors': priors,
                                'priors_combine': priors_combine,
                                'solution': kwargs.get('solution', None),
                                'compute_kwargs': {k:v for k,v in kwargs.items() if k in b.get_compute(compute=compute, **_skip_filter_checks).qualifiers},
                                'custom_lnprobability_callable': kwargs.pop('custom_lnprobability_callable', None),
                                'failed_samples_buffer': False if not expose_failed else failed_samples_buffer}

            # esargs['live_dangerously'] = kwargs.pop('live_dangerously', None)
            # esargs['runtime_sortingfn'] = kwargs.pop('runtime_sortingfn', None)

            logger.debug("EnsembleSampler({})".format(esargs))
            sampler = emcee.EnsembleSampler(**esargs)


            sargs = {}
            sargs['iterations'] = niters
            sargs['progress'] = kwargs.get('progressbar', False)
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
                    autocorr_times = sampler.backend.get_autocorr_time(quiet=True)
                    if np.any(~np.isnan(autocorr_times)):
                        burnin = int(burnin_factor * np.nanmax(autocorr_times))
                        thin = int(thin_factor * np.nanmin(autocorr_times))
                        if thin==0:
                            thin = 1
                    else:
                        burnin =0
                        thin = 1

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
            if pool is not None:
                pool.wait()
            else:
                return

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
            raise ImportError("could not import dynesty, pickle")

        solver_ps = b.get_solver(solver=solver, **_skip_filter_checks)
        if not len(solver_ps.get_value(qualifier='priors', init_from=kwargs.get('priors', None))):
            raise ValueError("cannot run dynesty without any distributions in priors")

        # filename = solver_ps.get_value(qualifier='filename', filename=kwargs.get('filename', None))
        # continue_previous_run = solver_ps.get_value(qualifier='continue_previous_run', continue_previous_run=kwargs.get('continue_previous_run', None))
        # if continue_previous_run and not os.path.exists(filename):
            # raise ValueError("cannot file filename='{}', cannot use continue_previous_run=True".format(filename))


    def _get_packet_and_solution(self, b, solver, **kwargs):
        # NOTE: b, solver, compute, backend will be added by get_packet_and_solution

        solution_params = []
        solution_params += [_parameters.DictParameter(qualifier='wrap_central_values', value={}, advanced=True, readonly=True, description='Central values adopted for all parameters in init_from that allow angle-wrapping.  Sampled values are not allowed beyond +/- pi of the central value.')]
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

        if kwargs.get('expose_failed', True):
            solution_params += [_parameters.DictParameter(qualifier='failed_samples', value={}, readonly=True, description='Samples that returned lnprobability=-inf.  Dictionary keys are the messages with values being an array with shape (N, len(fitted_twigs))')]


        return kwargs, _parameters.ParameterSet(solution_params)

    def _run_worker(self, packet):
        # here we'll override loading the bundle since it is not needed
        # in run_worker (for the workers.... note that the master
        # will enter run_worker through run, not here)
        return self.run_worker(**packet)

    def run_worker(self, b, solver, compute, **kwargs):

        def _get_packetlist(results, progress):
            return_ = [{'qualifier': 'wrap_central_values', 'value': wrap_central_values},
                     {'qualifier': 'fitted_uniqueids', 'value': params_uniqueids},
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
                    ]

            if expose_failed:
                failed_samples = {}
                if expose_failed:
                    for msg, fsamples in failed_samples_buffer:
                        failed_samples[msg] = failed_samples.get(msg, []) + [fsamples]

                return_ += [{'qualifier': 'failed_samples', 'value': failed_samples}]

            return [return_]

        within_mpirun = mpi.within_mpirun
        mpi_enabled = mpi.enabled

        # emcee handles workers itself.  So here we'll just take the workers
        # from our own waiting loop in phoebe's __init__.py and subscribe them
        # to emcee's pool.
        if mpi.within_mpirun:
            logger.info("using MPI pool for dynesty")

            global failed_samples_buffer
            failed_samples_buffer = []

            global _MPI
            from mpi4py import MPI as _MPI # needed for the cost-function to send failed samples

            def mpi_failed_samples_callback(result):
                global failed_samples_buffer
                if isinstance(result, tuple):
                    failed_samples_buffer.append(result)
                    return False
                else:
                    return True

            pool = _pool.MPIPool(callback=mpi_failed_samples_callback)
            is_master = pool.is_master()

            # temporarily disable MPI within run_compute to disabled parallelizing
            # per-time.
            mpi._within_mpirun = False
            mpi._enabled = False

        else:
            logger.info("using multiprocessing pool for dynesty")

            pool = _pool.MultiPool()
            failed_samples_buffer = multiprocessing.Manager().list()
            is_master = True

        if is_master:
            priors = kwargs.get('priors')
            priors_combine = kwargs.get('priors_combine')

            maxiter = kwargs.get('maxiter')
            progress_every_niters = kwargs.get('progress_every_niters')
            expose_failed = kwargs.get('expose_failed')

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

            wrap_central_values = _wrap_central_values(b, priors_dc, params_uniqueids)

            params_units = [dist.unit.to_string() for dist in priors_dc.dists]
            params_twigs = [b.get_parameter(uniqueid=uniqueid, **_skip_filter_checks).twig for uniqueid in params_uniqueids]

            # NOTE: in dynesty we draw from the priors and pass the prior-transforms,
            # but do NOT include the lnprior term in lnlikelihood, so we pass
            # priors as []
            lnlikelihood_kwargs = {'b': _bsolver(b, solver, compute, [], wrap_central_values),
                                   'params_uniqueids': params_uniqueids,
                                   'compute': compute,
                                   'priors': [],
                                   'priors_combine': 'and',
                                   'solution': kwargs.get('solution', None),
                                   'compute_kwargs': {k:v for k,v in kwargs.items() if k in b.get_compute(compute=compute, **_skip_filter_checks).qualifiers},
                                   'custom_lnprobability_callable': kwargs.pop('custom_lnprobability_callable', None),
                                   'failed_samples_buffer': False if not expose_failed else failed_samples_buffer}




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

                    solution_ps = self._fill_solution(solution_ps, [_get_packetlist(sampler.results, progress)], metawargs)

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

                    if progress_every_niters > 0:
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
            return _get_packetlist(sampler.results, progress=100)
        return




class _ScipyOptimizeBaseBackend(BaseSolverBackend):
    """
    See <phoebe.parameters.solver.optimizer.nelder_mead>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_solver>
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
    def run_checks(self, b, solver, compute, **kwargs):
        solver_ps = b.get_solver(solver=solver, **_skip_filter_checks)
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
        if kwargs.get('expose_lnprobabilities', False):
            solution_params += [_parameters.FloatParameter(qualifier='initial_lnprobability', value=0.0, readonly=True, default_unit=u.dimensionless_unscaled, description='lnprobability of the initial_values')]
            solution_params += [_parameters.FloatParameter(qualifier='fitted_lnprobability', value=0.0, readonly=True, default_unit=u.dimensionless_unscaled, description='lnprobability of the fitted_values')]

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
            fitted_units.append(p.get_default_unit().to_string())

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

        def _progressbar(xi):
            global _minimize_iter
            _minimize_iter += 1
            global _minimize_pbar
            _minimize_pbar.update(_minimize_iter)

        logger.debug("calling scipy.optimize.minimize(_lnprobability_negative, p0, method='{}', args=(b, {}, {}, {}, {}, {}), options={})".format(self.method, params_uniqueids, compute, priors, kwargs.get('solution', None), compute_kwargs, options))
        # TODO: would it be cheaper to pass the whole bundle (or just make one copy originally so we restore original values) than copying for each iteration?
        args = (_bsolver(b, solver, compute, priors), params_uniqueids, compute, priors, priors_combine, kwargs.get('solution', None), compute_kwargs, kwargs.pop('custom_lnprobability_callable', None))

        # set _within solver to prevent run_compute progressbars
        b._within_solver = True

        if _has_tqdm and kwargs.get('progressbar', False):
            global _minimize_iter
            _minimize_iter = 0
            global _minimize_pbar
            _minimize_pbar = _tqdm(total=options.get('maxiter'))

        res = optimize.minimize(_lnprobability_negative, p0,
                                method=self.method,
                                args=args,
                                options=options,
                                callback=_progressbar if _has_tqdm and kwargs.get('progressbar', False) else None)
        b._within_solver = False

        return_ = [{'qualifier': 'message', 'value': res.message},
                {'qualifier': 'nfev', 'value': res.nfev},
                {'qualifier': 'niter', 'value': res.nit},
                {'qualifier': 'success', 'value': res.success},
                {'qualifier': 'fitted_uniqueids', 'value': params_uniqueids},
                {'qualifier': 'fitted_twigs', 'value': params_twigs},
                {'qualifier': 'initial_values', 'value': p0},
                {'qualifier': 'fitted_values', 'value': res.x},
                {'qualifier': 'fitted_units', 'value': [u if isinstance(u, str) else u.to_string() for u in fitted_units]},
                {'qualifier': 'adopt_parameters', 'value': params_twigs, 'choices': params_twigs}]



        if kwargs.get('expose_lnprobabilities', False):
            initial_lnprobability = _lnprobability(p0, *args)
            fitted_lnprobability = _lnprobability(res.x, *args)

            return_ += [{'qualifier': 'initial_lnprobability', 'value': initial_lnprobability},
                         {'qualifier': 'fitted_lnprobability', 'value': fitted_lnprobability}]



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
        solver_ps = b.get_solver(solver=solver, **_skip_filter_checks)
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


        if kwargs.get('expose_lnprobabilities', False):
            solution_params += [_parameters.FloatParameter(qualifier='initial_lnprobability', value=0.0, readonly=True, default_unit=u.dimensionless_unscaled, description='lnprobability of the initial_values')]
            solution_params += [_parameters.FloatParameter(qualifier='fitted_lnprobability', value=0.0, readonly=True, default_unit=u.dimensionless_unscaled, description='lnprobability of the fitted_values')]

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
            pool = _pool.MPIPool()
            is_master = pool.is_master()
        else:
            pool = _pool.MultiPool()
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
                fitted_units.append(p.get_default_unit().to_string())

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
                       {'qualifier': 'fitted_units', 'value': [u if isinstance(u, str) else u.to_string() for u in fitted_units]},
                       {'qualifier': 'adopt_parameters', 'value': params_twigs, 'choices': params_twigs},
                       {'qualifier': 'bounds', 'value': bounds}]

            if kwargs.get('expose_lnprobabilities', False):
                initial_lnprobability = _lnprobability(False, *args)
                fitted_lnprobability = _lnprobability(res.x, *args)

                return_ += [{'qualifier': 'initial_lnprobability', 'value': initial_lnprobability},
                            {'qualifier': 'fitted_lnprobability', 'value': fitted_lnprobability}]



            return [return_]

        return
