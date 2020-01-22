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

import logging
logger = logging.getLogger("FITTING")
logger.addHandler(logging.NullHandler())

_skip_filter_checks = {'check_default': False, 'check_visible': False}

class BaseFittingBackend(object):
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

    def _get_packet_and_feedback(self, b, fitting, compute, **kwargs):
        """
        see get_packet_and_feedback.  _get_packet_and_feedback provides the custom parts
        of the packet that are Backend-dependent.

        This should return the packet to send to all workers and the new_syns to
        be sent to the master.

        return packet, feedback_ps
        """
        raise NotImplementedError("_get_packet_and_feedback is not implemented by the {} backend".format(self.__class__.__name__))

    def get_packet_and_feedback(self, b, fitting, compute, **kwargs):
        """
        get_packet_and_feedback is called by the master and must get all information necessary
        to send to all workers.  The returned packet will be passed on as
        _run_chunk(**packet) with the following exceptions:

        * b: the bundle will be included in the packet serialized
        * fitting: the label of the fitting options will be included in the packet
        * compute: the label of the compute options will be included in the packet
        * backend: the class name will be passed on in the packet so the worker can call the correct backend
        * all kwargs will be passed on verbatim
        """
        packet, feedback_ps = self._get_packet_and_feedback(b, fitting, compute, **kwargs)
        for k,v in kwargs.items():
            packet[k] = v

        # if kwargs.get('max_computations', None) is not None:
        #     if len(packet.get('infolists', packet.get('infolist', []))) > kwargs.get('max_computations'):
        #         raise ValueError("more than {} computations detected ({} estimated).".format(kwargs.get('max_computations'), len(packet['infolists'])))

        packet['b'] = b.to_json() if mpi.enabled else b
        packet['fitting'] = fitting
        packet['compute'] = compute
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

    def run(self, b, fitting, compute, **kwargs):
        """
        if within mpirun, workers should call _run_worker instead of run
        """
        self.run_checks(b, fitting,  compute, **kwargs)

        logger.debug("rank:{}/{} calling get_packet_and_feedback".format(mpi.myrank, mpi.nprocs))
        packet, feedback_ps = self.get_packet_and_feedback(b, fitting, compute, **kwargs)

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




class EmceeBackend(BaseFittingBackend):
    """
    See <phoebe.parameters.fitting.emcee>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_fitting>
    * <phoebe.frontend.bundle.Bundle.run_fitting>
    """
    def run_checks(self, b, fitting, compute, **kwargs):
        # check whether emcee is installed

        if not _use_emcee:
            raise ImportError("could not import emcee, schwimbbad, and h5py")

        if LooseVersion(emcee.__version__) < LooseVersion("3.0.0"):
            raise ImportError("emcee backend requires emcee 3.0+, {} found".format(emcee.__version__))

        fitting_ps = b.get_fitting(fitting)
        if not len(fitting_ps.get_value(qualifier='init_from', init_from=kwargs.get('init_from', None))):
            raise ValueError("cannot run emcee without any distributions in init_from")

        filename = fitting_ps.get_value(qualifier='filename', filename=kwargs.get('filename', None))
        continue_previous_run = fitting_ps.get_value(qualifier='continue_previous_run', continue_previous_run=kwargs.get('continue_previous_run', None))
        if continue_previous_run and not os.path.exists(filename):
            raise ValueError("cannot file filename='{}', cannot use continue_previous_run=True".format(filename))


    def _get_packet_and_feedback(self, b, fitting, compute, **kwargs):
        # NOTE: b, fitting, compute, backend will be added by get_packet_and_feedback

        fitting_ps = b.get_fitting(fitting=fitting)

        for param in fitting_ps.to_list():
            kwargs.setdefault(param.qualifier, param.get_value())

        feedback_params = []
        feedback_params += [_parameters.StringParameter(qualifier='filename', value=fitting_ps.get_value(qualifier='filename', filename=kwargs.get('filename', None)), description='filename of emcee progress file (contents loaded on the fly, DO NOT DELETE FILE)')]

        return kwargs, _parameters.ParameterSet(feedback_params)

    @staticmethod
    def _lnlikelihood(sampled_values, bjson, params_uniqueids, compute, priors, feedback, compute_kwargs={}):
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
                return -np.info

        # print("*** _lnlikelihood run_compute from rank: {}".format(mpi.myrank))
        try:
            b.run_compute(compute=compute, model=feedback, **compute_kwargs)
        except Exception as err:
            logger.warning("received error from run_compute: {}.  lnlikelihood=-inf".format(err))
            return -np.inf

        # print("*** _lnlikelihood returning from rank: {}".format(mpi.myrank))
        return b.calculate_lnp(distribution=priors) + b.calculate_lnlikelihood(model=feedback)


    def _run_worker(self, packet):
        # here we'll override loading the bundle since it is not needed
        # in run_worker (for the workers.... note that the master
        # will enter run_worker through run, not here)
        return self.run_worker(**packet)

    def run_worker(self, b, fitting, compute, **kwargs):
        # emcee handles workers itself.  So here we'll just take the workers
        # from our own waiting loop in phoebe's __init__.py and subscribe them
        # to emcee's pool.

        if mpi.within_mpirun:
            pool = schwimmbad.MPIPool()
            is_master = pool.is_master()
        else:
            pool = None
            is_master = True

        if is_master:
            fitting_ps = b.get_fitting(fitting=fitting)
            niters = fitting_ps.get_value(qualifier='niters', niters=kwargs.get('niters', None))
            nwalkers = fitting_ps.get_value(qualifier='nwalkers', nwalkers=kwargs.get('nwalkers', None))
            init_from = fitting_ps.get_value(qualifier='init_from', init_from=kwargs.get('init_from', None), expand=True)
            priors = fitting_ps.get_value(qualifier='priors', priors=kwargs.get('priors', None), expand=True)

            filename = fitting_ps.get_value(qualifier='filename', filename=kwargs.get('filename', None))
            continue_previous_run = fitting_ps.get_value(qualifier='continue_previous_run', continue_previous_run=kwargs.get('continue_previous_run', None))


            sample_dict = b.sample_distribution(distribution=init_from, N=nwalkers, keys='uniqueid', set_value=False)
            params_uniqueids, p0 = list(sample_dict.keys()), np.asarray(list(sample_dict.values()))

            # EnsembleSampler kwargs
            esargs = {}
            esargs['pool'] = pool
            esargs['nwalkers'] = nwalkers
            esargs['ndim'] = len(params_uniqueids)
            esargs['log_prob_fn'] = self._lnlikelihood
            # esargs['a'] = kwargs.pop('a', None),
            # esargs['moves'] = kwargs.pop('moves', None)
            # esargs['args'] = None

            # TODO: OPTIMIZE exclude disabled datasets?
            # TODO: re-enable removing unused compute options - currently causes some constraints to fail
            bjson = b.exclude(context=['model', 'feedback', 'figure'], **_skip_filter_checks).exclude(
                              fitting=[f for f in b.fittings if f!=fitting and fitting is not None], **_skip_filter_checks).exclude(
                              # compute=[c for c in b.computes if c!=compute and compute is not None], **_skip_filter_checks).exclude(
                              distribution=[d for d in b.distributions if d not in priors+init_from], **_skip_filter_checks).to_json(incl_uniqueid=True, exclude=['description', 'advanced', 'copy_for'])

            esargs['kwargs'] = {'bjson': bjson,
                                'params_uniqueids': params_uniqueids,
                                'compute': compute,
                                'priors': priors,
                                'feedback': kwargs.get('feedback', None),
                                'compute_kwargs': {k:v for k,v in kwargs.items() if k in b.get_compute(compute=compute).qualifiers}}

            # esargs['live_dangerously'] = kwargs.pop('live_dangerously', None)
            # esargs['runtime_sortingfn'] = kwargs.pop('runtime_sortingfn', None)

            filename = os.path.join(os.getcwd(), filename)
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

        return {}
