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
# from phoebe import u, c
from phoebe import conf, mpi

from distutils.version import LooseVersion, StrictVersion
from copy import deepcopy

try:
    import emcee
    import h5py
    import schwimmbad
except ImportError:
    _use_emcee = False
else:
    _use_emcee = True

from scipy import optimize

import logging
logger = logging.getLogger("SOLVER")
logger.addHandler(logging.NullHandler())

_skip_filter_checks = {'check_default': False, 'check_visible': False}

def _bjson(b, solver, compute, distributions):
    # TODO: OPTIMIZE exclude disabled datasets?
    # TODO: re-enable removing unused compute options - currently causes some constraints to fail
    return b.exclude(context=['model', 'feedback', 'figure'], **_skip_filter_checks).exclude(
                      solver=[f for f in b.solvers if f!=solver and solver is not None], **_skip_filter_checks).exclude(
                      # compute=[c for c in b.computes if c!=compute and compute is not None], **_skip_filter_checks).exclude(
                      distribution=[d for d in b.distributions if d not in distributions], **_skip_filter_checks).to_json(incl_uniqueid=True, exclude=['description', 'advanced', 'copy_for'])


def _lnlikelihood(sampled_values, bjson, params_uniqueids, compute, priors, priors_combine, feedback, compute_kwargs={}):
    # print("*** _lnlikelihood from rank: {}".format(mpi.myrank))
    # TODO: [OPTIMIZE] make sure that run_checks=False, run_constraints=False is
    # deferring constraints/checks until run_compute.

    # TODO: [OPTIMIZE] try to remove this deepcopy - for some reason distribution objects
    # are being stripped of their units without it
    b = phoebe.frontend.bundle.Bundle(deepcopy(bjson))

    for uniqueid, value in zip(params_uniqueids, sampled_values):
        try:
            b.set_value(uniqueid=uniqueid, value=value, run_checks=False, run_constraints=False, **_skip_filter_checks)
        except ValueError as err:
            logger.warning("received error while setting values: {}. lnlikelihood=-inf".format(err))
            return -np.inf

    # print("*** _lnlikelihood run_compute from rank: {}".format(mpi.myrank))
    try:
        b.run_compute(compute=compute, model=feedback, do_create_fig_params=False, **compute_kwargs)
    except Exception as err:
        logger.warning("received error from run_compute: {}.  lnlikelihood=-inf".format(err))
        return -np.inf

    # print("*** _lnlikelihood returning from rank: {}".format(mpi.myrank))
    return b.calculate_lnp(distribution=priors, combine=priors_combine) + b.calculate_lnlikelihood(model=feedback)

def _lnlikelihood_negative(sampled_values, bjson, params_uniqueids, compute, priors, priors_combine, feedback, compute_kwargs={}):
    return -1 * _lnlikelihood(sampled_values, bjson, params_uniqueids, compute, priors, priors_combine, feedback, compute_kwargs)

class BaseSolverBackend(object):
    def __init__(self):
        return

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

    def _get_packet_and_feedback(self, b, solver, **kwargs):
        """
        see get_packet_and_feedback.  _get_packet_and_feedback provides the custom parts
        of the packet that are Backend-dependent.

        This should return the packet to send to all workers and the new_syns to
        be sent to the master.

        return packet, feedback_ps
        """
        raise NotImplementedError("_get_packet_and_feedback is not implemented by the {} backend".format(self.__class__.__name__))

    def get_packet_and_feedback(self, b, solver, **kwargs):
        """
        get_packet_and_feedback is called by the master and must get all information necessary
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
            kwargs.setdefault(param.qualifier, param.get_value(expand=True))

        packet, feedback_ps = self._get_packet_and_feedback(b, solver, **kwargs)

        for k,v in kwargs.items():
            packet[k] = v

        # if kwargs.get('max_computations', None) is not None:
        #     if len(packet.get('infolists', packet.get('infolist', []))) > kwargs.get('max_computations'):
        #         raise ValueError("more than {} computations detected ({} estimated).".format(kwargs.get('max_computations'), len(packet['infolists'])))

        packet['b'] = b.to_json() if mpi.enabled else b
        packet['solver'] = solver
        # packet['compute'] = compute  # should have been set by kwargs, when applicable
        packet['backend'] = self.__class__.__name__

        return packet, feedback_ps

    def _fill_feedback(self, feedback_ps, rpacketlists_per_worker):
        """
        rpacket_per_worker is a list of packetlists as returned by _run_chunk
        """
        # TODO: move to BaseBackendByDataset or BaseBackend?
        logger.debug("rank:{}/{} {}._fill_feedback".format(mpi.myrank, mpi.nprocs, self.__class__.__name__))

        for packetlists in rpacketlists_per_worker:
            # single worker
            for packetlist in packetlists:
                # single time/dataset
                for packet in packetlist:
                    # single parameter
                    try:
                        feedback_ps.set_value(check_visible=False, check_default=False, **packet)
                    except Exception as err:
                        raise ValueError("failed to set value from packet: {}.  Original error: {}".format(packet, str(err)))

        return feedback_ps

    def run_worker(self):
        """
        run_worker will receive the packet (with the bundle deserialized if necessary)
        and is responsible for any work done by a worker within MPI
        """
        raise NotImplementedError("run_worker is not implemented by the {} backend".format(self.__class__.__name__))

    def _run_worker(self, packet):
        # the worker receives the bundle serialized, so we need to unpack it
        logger.debug("rank:{}/{} _run_worker".format(mpi.myrank, mpi.nprocs))
        packet['b'] = phoebe.frontend.bundle.Bundle(packet['b'])
        # do the computations requested for this worker
        rpacketlists = self.run_worker(**packet)
        # send the results back to the master (root=0)
        mpi.comm.gather(rpacketlists, root=0)

    def run(self, b, solver, compute, **kwargs):
        """
        if within mpirun, workers should call _run_worker instead of run
        """
        self.run_checks(b, solver,  compute, **kwargs)

        logger.debug("rank:{}/{} calling get_packet_and_feedback".format(mpi.myrank, mpi.nprocs))
        packet, feedback_ps = self.get_packet_and_feedback(b, solver, **kwargs)

        if mpi.enabled:
            # broadcast the packet to ALL workers
            logger.debug("rank:{}/{} broadcasting to all workers".format(mpi.myrank, mpi.nprocs))
            mpi.comm.bcast(packet, root=0)

            # now even the master can become a worker and take on a chunk
            packet['b'] = b
            rpacketlists = self.run_worker(**packet)

            # now receive all packetlists
            logger.debug("rank:{}/{} gathering packetlists from all workers".format(mpi.myrank, mpi.nprocs))
            rpacketlists_per_worker = mpi.comm.gather(rpacketlists, root=0)

        else:
            rpacketlists_per_worker = [self.run_worker(**packet)]

        logger.debug("rank:{}/{} calling _fill_feedback".format(mpi.myrank, mpi.nprocs))
        return self._fill_feedback(feedback_ps, rpacketlists_per_worker)




class EmceeBackend(BaseSolverBackend):
    """
    See <phoebe.parameters.solver.sampler.emcee>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_solver>
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
    def run_checks(self, b, solver, compute, **kwargs):
        # check whether emcee is installed

        if not _use_emcee:
            raise ImportError("could not import emcee, schwimbbad, and h5py")

        if LooseVersion(emcee.__version__) < LooseVersion("3.0.0"):
            raise ImportError("emcee backend requires emcee 3.0+, {} found".format(emcee.__version__))

        solver_ps = b.get_solver(solver)
        if not len(solver_ps.get_value(qualifier='init_from', init_from=kwargs.get('init_from', None))):
            raise ValueError("cannot run emcee without any distributions in init_from")

        filename = solver_ps.get_value(qualifier='filename', filename=kwargs.get('filename', None))
        continue_previous_run = solver_ps.get_value(qualifier='continue_previous_run', continue_previous_run=kwargs.get('continue_previous_run', None))
        if continue_previous_run and not os.path.exists(filename):
            raise ValueError("cannot file filename='{}', cannot use continue_previous_run=True".format(filename))


    def _get_packet_and_feedback(self, b, solver, **kwargs):
        # NOTE: b, solver, compute, backend will be added by get_packet_and_feedback

        feedback_params = []
        feedback_params += [_parameters.StringParameter(qualifier='filename', value=kwargs.get('filename', None), description='filename of emcee progress file (contents loaded on the fly, DO NOT DELETE FILE)')]
        feedback_params += [_parameters.ArrayParameter(qualifier='fitted_parameters', value=[], description='uniqueids of parameters fitted by the minimizer')]

        return kwargs, _parameters.ParameterSet(feedback_params)

    def _run_worker(self, packet):
        # here we'll override loading the bundle since it is not needed
        # in run_worker (for the workers.... note that the master
        # will enter run_worker through run, not here)
        return self.run_worker(**packet)

    def run_worker(self, b, solver, compute, **kwargs):
        # emcee handles workers itself.  So here we'll just take the workers
        # from our own waiting loop in phoebe's __init__.py and subscribe them
        # to emcee's pool.

        if mpi.within_mpirun:
            pool = schwimmbad.MPIPool()
            is_master = pool.is_master()
        else:
            pool = schwimmbad.MultiPool()
            is_master = True

        if is_master:
            niters = kwargs.get('niters')
            nwalkers = kwargs.get('nwalkers')
            init_from = kwargs.get('init_from')
            init_from_combine = kwargs.get('init_from_combine')
            priors = kwargs.get('priors')
            priors_combine = kwargs.get('priors_combine')

            filename = os.path.join(os.getcwd(), kwargs.get('filename'))
            continue_previous_run = kwargs.get('continue_previous_run')

            sample_dict = b.sample_distribution(distribution=init_from, N=nwalkers,
                                                combine=init_from_combine,
                                                include_constrained=False,
                                                keys='uniqueid', set_value=False)
            params_uniqueids, p0 = list(sample_dict.keys()), np.asarray(list(sample_dict.values()))

            # EnsembleSampler kwargs
            esargs = {}
            esargs['pool'] = pool
            esargs['nwalkers'] = nwalkers
            esargs['ndim'] = len(params_uniqueids)
            esargs['log_prob_fn'] = _lnlikelihood
            # esargs['a'] = kwargs.pop('a', None),
            # esargs['moves'] = kwargs.pop('moves', None)
            # esargs['args'] = None

            esargs['kwargs'] = {'bjson': _bjson(b, solver, compute, init_from+priors),
                                'params_uniqueids': params_uniqueids,
                                'compute': compute,
                                'priors': priors,
                                'priors_combine': priors_combine,
                                'feedback': kwargs.get('feedback', None),
                                'compute_kwargs': {k:v for k,v in kwargs.items() if k in b.get_compute(compute=compute, **_skip_filter_checks).qualifiers}}

            # esargs['live_dangerously'] = kwargs.pop('live_dangerously', None)
            # esargs['runtime_sortingfn'] = kwargs.pop('runtime_sortingfn', None)

            # TODO: consider supporting passing name=feedback... but that
            # seems to cause and hdf bug and also will need to be careful
            # to match feedback in order to use continue_previous_run
            logger.debug("using backend=HDFBackend('{}')".format(filename))
            backend = emcee.backends.HDFBackend(filename) #, name=kwargs.get('feedback', None))
            if not continue_previous_run:
                logger.debug("backend.reset({}, {})".format(nwalkers, len(params_uniqueids)))
                backend.reset(nwalkers, len(params_uniqueids))

            esargs['backend'] = backend

            logger.debug("EnsembleSampler({})".format(esargs))
            sampler = emcee.EnsembleSampler(**esargs)


            sargs = {}
            # sargs['log_prob0'] = kwargs.pop('log_prob0', None)
            # sargs['rstate0'] = kwargs.pop('rstate0', None)
            # sargs['blobs0'] = kwargs.pop('blobls0', None)
            sargs['iterations'] = niters
            # sargs['thin'] = kwargs.pop('thin', 1)  # TODO: make parameter - check if thin or thin_by
            # sargs['store'] = True
            sargs['progress'] = False  # TODO: make parameter? or set to True?  or check if necessary library is imported?
            sargs['skip_initial_state_check'] = True  # TODO: remove this?  Or can we reproduce the logic in a warning?

            logger.debug("sampler.sample(p0, {})".format(sargs))
            # TODO: parameters for checking convergence
            for sample in sampler.sample(p0.T, **sargs):
                # Only check convergence every 10 steps
                logger.info("emcee: iteration {} complete".format(sampler.iteration))


                # if sampler.iteration % 10:
                #     logger.debug("completed interation {}".format(sampler.iteration))
                #     continue

                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
                # logger.debug("checking for convergence on iteration {}".format(sampler.iteration))
                # tau = sampler.get_autocorr_time(tol=0)
                # autocorr[index] = np.mean(tau)
                # index += 1
                #
                # # Check convergence
                # converged = np.all(tau * 100 < sampler.iteration)
                # converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                # if converged:
                #     break
                # old_tau = tau

        else:
            # NOTE: because we overrode self._run_worker to skip loading the
            # bundle, b is just a json string here.  If we ever need the
            # bundle in here, just remove the override for self._run_worker.

            # temporarily disable MPI within run_compute to disabled parallelizing
            # per-time.
            within_mpirun = mpi.within_mpirun
            mpi_enabled = mpi.enabled
            mpi._within_mpirun = False
            mpi._enabled = False

            pool.wait()

            # restore previous MPI state
            mpi._within_mpirun = within_mpirun
            mpi._enabled = mpi_enabled

        if pool is not None:
            pool.close()

        if is_master:
            return [[{'qualifier': 'fitted_parameters', 'value': params_uniqueids}]]
        return {}


class Nelder_MeadBackend(BaseSolverBackend):
    """
    See <phoebe.parameters.solver.optimizer.nelder_mead>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_solver>
    * <phoebe.frontend.bundle.Bundle.run_solver>
    """
    def run_checks(self, b, solver, compute, **kwargs):
        solver_ps = b.get_solver(solver)
        if not len(solver_ps.get_value(qualifier='init_from', init_from=kwargs.get('init_from', None))):
            raise ValueError("cannot run scipy.optimize.minimize(method='nelder-mead') without any distributions in init_from")


    def _get_packet_and_feedback(self, b, solver, **kwargs):
        # NOTE: b, solver, compute, backend will be added by get_packet_and_feedback
        feedback_params = []

        feedback_params += [_parameters.StringParameter(qualifier='message', value='', description='message from the minimizer')]
        feedback_params += [_parameters.IntParameter(qualifier='nfev', value=0, limits=(0,None), description='number of completed function evaluations (forward models)')]
        feedback_params += [_parameters.IntParameter(qualifier='niter', value=0, limits=(0,None), description='number of completed iterations')]
        feedback_params += [_parameters.BoolParameter(qualifier='success', value=False, description='whether the minimizer returned a success message')]
        feedback_params += [_parameters.ArrayParameter(qualifier='fitted_parameters', value=[], description='uniqueids of parameters fitted by the minimizer')]
        # TODO: double check units here... is it current default units or those used by the backend?
        feedback_params += [_parameters.FloatArrayParameter(qualifier='fitted_values', value=[], description='final values returned by the minimizer (in current default units of each parameter)')]

        return kwargs, _parameters.ParameterSet(feedback_params)

    # def _run_worker(self, packet):
    #     # here we'll override loading the bundle since it is not needed
    #     # in run_worker (for the workers.... note that the master
    #     # will enter run_worker through run, not here)
    #     return self.run_worker(**packet)

    def run_worker(self, b, solver, compute, **kwargs):
        if mpi.within_mpirun:
            raise NotImplementedError("mpi support for scipy.optimize not yet implemented")
            # TODO: we need to tell the workers to join the pool for time-parallelization?

        init_from = kwargs.get('init_from')
        init_from_combine = kwargs.get('init_from_combine')
        priors = kwargs.get('priors')
        priors_combine = kwargs.get('priors_combine')

        # TODO: replace this with twig SelectParameter
        sample_dict = b.sample_distribution(distribution=init_from,
                                            N=None, keys='uniqueid',
                                            combine=init_from_combine,
                                            include_constrained=False,
                                            set_value=False)
        params_uniqueids, p0 = list(sample_dict.keys()), np.asarray(list(sample_dict.values()))

        compute_kwargs = {k:v for k,v in kwargs.items() if k in b.get_compute(compute=compute, **_skip_filter_checks).qualifiers}

        options = {k:v for k,v in kwargs.items() if k in ['maxiter', 'maxfex', 'xatol', 'fatol', 'adaptive']}

        logger.debug("calling scipy.optimize.minimize(_lnlikelihood_negative, p0, method='nelder-mead', args=(bjson, {}, {}, {}, {}, {}), options={})".format(params_uniqueids, compute, priors, kwargs.get('feedback', None), compute_kwargs, options))
        # TODO: would it be cheaper to pass the whole bundle (or just make one copy originally so we restore original values) than copying for each iteration?
        res = optimize.minimize(_lnlikelihood_negative, p0,
                                method='nelder-mead',
                                args=(_bjson(b, solver, compute, init_from+priors), params_uniqueids, compute, priors, priors_combine, kwargs.get('feedback', None), compute_kwargs),
                                options=options)


        return [[{'qualifier': 'message', 'value': res.message},
                {'qualifier': 'nfev', 'value': res.nfev},
                {'qualifier': 'niter', 'value': res.nit},
                {'qualifier': 'success', 'value': res.success},
                {'qualifier': 'fitted_parameters', 'value': params_uniqueids},
                {'qualifier': 'fitted_values', 'value': res.x}]]