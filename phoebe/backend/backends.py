import os
import numpy as np

try:
  import commands
except:
  import subprocess as commands

import tempfile
from copy import deepcopy
import itertools

from phoebe.parameters import dataset as _dataset
from phoebe.parameters import StringParameter, DictParameter, ArrayParameter, ParameterSet
from phoebe.parameters.parameters import _extract_index_from_string
from phoebe import dynamics
from phoebe.backend import universe, etvs, horizon_analytic
from phoebe.atmospheres import passbands
from phoebe.distortions  import roche
from phoebe.frontend import io
import phoebe.frontend.bundle
from phoebe.dependencies.nparray.nparray import Array as _nparrayArray
from phoebe import u, c
from phoebe import conf, mpi
from phoebe import pool as _pool

#global variable to prevent phb1 configuration from happening
#multiple times
global _phb1_config
_phb1_config = False

try:
    import phoebe_legacy as phb1
except ImportError:
    try:
        import phoebeBackend as phb1
    except ImportError:
        _use_phb1 = False
    else:
        _use_phb1 = True
else:
    _use_phb1 = True



try:
    import ellc
except ImportError:
    _use_ellc = False
else:
    _use_ellc = True


from tqdm import tqdm as _tqdm

def _progressbar(args, total=None, show_progressbar=True):
    if show_progressbar:
        return _tqdm(args, total=total)
    else:
        return args

from scipy.stats import norm as _norm


import logging
logger = logging.getLogger("BACKENDS")
logger.addHandler(logging.NullHandler())

# the following list is for backends that use numerical meshes
_backends_that_require_meshing = ['phoebe', 'legacy']

_skip_filter_checks = {'check_default': False, 'check_visible': False}

def _simplify_error_message(msg):
    # simplify error messages so values, etc, don't create separate
    # entries in the returned dictionary.
    msg = str(msg) # in case an exception object
    if 'not within limits' in msg:
        msg = 'outside parameter limits'
    elif 'value further than' in msg:
        msg = 'outside angle wrapping limits'
    elif 'overflow' in msg:
        msg = 'roche overflow'
    elif 'lookup ld_coeffs' in msg:
        msg = 'ld_coeffs lookup out-of-bounds'
    elif 'compute_pblums failed' in msg:
        msg = 'atm out-of-bounds during compute_pblums'
    elif 'Atmosphere parameters out of bounds' in msg:
        msg = 'atm out-of-bounds'
    elif 'Could not compute ldint' in msg:
        msg = 'could not compute ldint with provided atm and ld_mode'
    elif 'not compatible for ld_func' in msg:
        msg = 'ld_coeffs and ld_func incompatible'
    return msg

def _needs_mesh(b, dataset, kind, component, compute):
    """
    """
    # print "*** _needs_mesh", kind
    compute_kind = b.get_compute(compute).kind
    if compute_kind not in _backends_that_require_meshing:
        # then we don't have meshes for this backend, so all should be False
        return False

    if kind not in ['mesh', 'lc', 'rv', 'lp']:
        return False

    # if kind == 'lc' and compute_kind=='phoebe' and b.get_value(qualifier='lc_method', compute=compute, dataset=dataset, context='compute')=='analytical':
    #     return False

    if kind == 'rv' and (compute_kind == 'legacy' or b.get_value(qualifier='rv_method', compute=compute, component=component, dataset=dataset, context='compute', **_skip_filter_checks)=='dynamical'):
        return False

    return True


def _timequalifier_by_kind(kind):
    if kind=='etv':
        return 'time_ephems'
    else:
        return 'times'

def _expand_mesh_times(b, dataset_ps, component):
    def get_times(b, include_times_entry):
        if include_times_entry in b.datasets:
            add_ps = b.filter(dataset=include_times_entry, context='dataset', **_skip_filter_checks)
            add_timequalifier = _timequalifier_by_kind(add_ps.kind)
            add_ps_compute_times_components = add_ps.filter(qualifier='compute_times', **_skip_filter_checks).components
            if len(add_ps.times):
                add_times = np.array([float(t) for t in add_ps.times])
            elif len(add_ps_compute_times_components):
                # then we need to concatenate over all components_
                # (times@rv@primary and times@rv@secondary are not necessarily
                # identical)
                add_times = np.unique(np.append(*[add_ps.get_value(qualifier='compute_times', component=c, **_skip_filter_checks) for c in add_ps_compute_times_components]))
            else:
                # then we're adding from some dataset at the system-level (like lcs)
                # that have component=None
                add_times = add_ps.get_value(qualifier='compute_times', component=None, unit=u.d, **_skip_filter_checks)

            if not len(add_times):
                add_ps_components = add_ps.filter(qualifier=add_timequalifier, **_skip_filter_checks).components
                if len(add_ps_components):
                    add_times = np.unique(np.append(*[add_ps.get_value(qualifier=add_timequalifier, component=c, **_skip_filter_checks) for c in add_ps_components]))
                else:
                    add_times = add_ps.get_value(qualifier=add_timequalifier, component=None, unit=u.d, **_skip_filter_checks)
        else:
            # then some sort of t0 from context='component' or 'system'
            add_times = [b.get_value(include_times_entry, context=['component', 'system'], **_skip_filter_checks)]

        return add_times

    # print "*** _expand_mesh_times", dataset_ps, dataset_ps.kind, component
    if dataset_ps.kind != 'mesh':
        raise TypeError("_expand_mesh_times only works for mesh datasets")

    # we're first going to access the compute_times@mesh... this should not have a component tag
    this_times = dataset_ps.get_value(qualifier='compute_times', component=None, unit=u.d)
    this_times = np.unique(np.append(this_times,
                                     [get_times(b, include_times_entry) for include_times_entry in dataset_ps.get_value(qualifier='include_times', expand=True, **_skip_filter_checks)]
                                     )
                           )

    return this_times

def _extract_from_bundle(b, compute, dataset=None, times=None,
                         by_time=True, include_mesh=True, **kwargs):
    """
    Extract a list of sorted times and the datasets that need to be
    computed at each of those times.  Any backend can then loop through
    these times and see what quantities are needed for that time step.

    Empty copies of synthetics for each applicable dataset are then
    created and returned so that they can be filled by the given backend.
    Setting of other meta-data should be handled by the bundle once
    the backend returns the filled synthetics.

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :return: times (list of floats or dictionary of lists of floats),
        infos (list of lists of dictionaries),
        new_syns (ParameterSet containing all new parameters)
    :raises NotImplementedError: if for some reason there is a problem getting
        a unique match to a dataset (shouldn't ever happen unless
        the user overrides a label)
    """
    provided_times = times
    times = []
    infolists = []
    needed_syns = []

    # The general format of the datastructures used within PHOEBE are as follows:
    # if by_time:
    # - times (list within _extract_from_bundle_by_time but then casted to np.array)
    # - infolists (list of <infolist>, same shape and order as times)
    # - infolist (list of <info>)
    # - info (dict containing information for a given dataset-component computation at the given time)
    #
    # else:
    # - infolist (same as by_time, one per-dataset, called infolists within this function only)
    # - info (same as by_time)
    #
    # The workers then return a similar format:
    # - packetlist (list of <packet>)
    # - packet (dict ready to be sent to new_syns.set_value(**packet))
    # where packet has a similar structure to info (but with value and possibly time keys)
    # but packetlist may be longer than infolist (since mesh passband-columns allow
    # now have their own entries.)

    if dataset is None:
        datasets = b.filter(qualifier='enabled', compute=compute, value=True, **_skip_filter_checks).datasets
    else:
        datasets = b.filter(dataset=dataset, context='dataset', **_skip_filter_checks).datasets

    for dataset in datasets:
        dataset_ps = b.filter(context='dataset', dataset=dataset, **_skip_filter_checks)
        dataset_compute_ps = b.filter(context='compute', dataset=dataset, compute=compute, **_skip_filter_checks)
        dataset_kind = dataset_ps.kind
        time_qualifier = _timequalifier_by_kind(dataset_kind)
        if dataset_kind in ['lc']:
            # then the Parameters in the model only exist at the system-level
            # and are not tagged by component
            dataset_components = [None]
        elif dataset_kind in ['lp']:
            # TODO: eventually spectra and RVs as well (maybe even LCs and ORBs)
            dataset_components = b.hierarchy.get_stars() + b.hierarchy.get_orbits()
        else:
            dataset_components = b.hierarchy.get_stars()

        for component in dataset_components:
            if isinstance(provided_times, dict) and dataset in provided_times.keys():
                this_times = provided_times.get(dataset)
            elif provided_times is not None and not isinstance(provided_times, dict):
                this_times = provided_times
            elif dataset_kind == 'mesh' and include_mesh:
                this_times = _expand_mesh_times(b, dataset_ps, component)
            elif dataset_kind in ['lp']:
                this_times = np.unique(dataset_ps.get_value(qualifier='compute_times', unit=u.d, **_skip_filter_checks))
                if not len(this_times):
                    # then we have Parameters tagged by times, this will probably
                    # also apply to spectra.
                    this_times = [float(t) for t in dataset_ps.times]
            else:
                timequalifier = _timequalifier_by_kind(dataset_kind)
                timecomponent = component if dataset_kind not in ['mesh', 'lc'] else None
                # print "*****", dataset_kind, dataset_ps.kinds, timequalifier, timecomponent
                # NOTE: compute_times is not component-dependent, but times can be (i.e. for RV datasets)
                this_times = dataset_ps.get_value(qualifier='compute_times', unit=u.d, **_skip_filter_checks)
                if not len(this_times):
                    this_times = dataset_ps.get_value(qualifier=timequalifier, component=timecomponent, unit=u.d, **_skip_filter_checks)

                # we may also need to compute at other times if requested by a
                # mesh with this dataset in datasets@mesh
                # for mesh_datasets_parameter in mesh_datasets_parameters:
                    # if dataset in mesh_datasets_parameter.get_value():
                        # mesh_obs_ps = b.filter(context='dataset', dataset=mesh_datasets_parameter.dataset, component=None)
                        # TODO: not sure about the component=None on the next line... what will this do for rvs with different times per-component?
                        # mesh_times = _expand_mesh_times(b, mesh_obs_ps, component=None)
                        # this_times = np.unique(np.append(this_times, mesh_times))

            if dataset_kind in ['lc'] and \
                    b.get_value(qualifier='exptime', dataset=dataset, **_skip_filter_checks) > 0 and \
                    dataset_compute_ps.get_value(qualifier='fti_method', fti_method=kwargs.get('fti_method', None), **_skip_filter_checks)=='oversample':

                # Then we need to override the times retrieved from the dataset
                # with the oversampled times.  Later we'll do an average over
                # the exposure.
                # NOTE: here we assume that the dataset times are at mid-exposure,
                # if we want to allow more flexibility, we'll need a parameter
                # that gives this option and different logic for each case.
                exptime = dataset_ps.get_value(qualifier='exptime', unit=u.d, **_skip_filter_checks)
                fti_oversample = dataset_compute_ps.get_value(qualifier='fti_oversample', check_visible=False, **kwargs)
                # NOTE: if changing this, also change in bundle.run_compute
                this_times = np.array([np.linspace(t-exptime/2., t+exptime/2., fti_oversample) for t in this_times]).flatten()

            if dataset_kind in ['lp']:
                # for line profiles and spectra, we only need to compute synthetic
                # model if there are defined wavelengths
                this_wavelengths = dataset_ps.get_value(qualifier='wavelengths', component=component, **_skip_filter_checks)
            else:
                this_wavelengths = None

            if len(this_times) and (this_wavelengths is None or len(this_wavelengths)):

                info = {'dataset': dataset,
                        'component': component,
                        'kind': dataset_kind,
                        'needs_mesh': _needs_mesh(b, dataset, dataset_kind, component, compute),
                        }

                if dataset_kind == 'mesh' and include_mesh:
                    # then we may be requesting passband-dependent columns be
                    # copied to the mesh from other datasets based on the values
                    # of columns@mesh.  Let's store the needed information here,
                    # where mesh_datasets and mesh_kinds correspond to each
                    # other (but mesh_columns does not).
                    info['mesh_coordinates'] = dataset_ps.get_value(qualifier='coordinates', expand=True, **_skip_filter_checks)
                    info['mesh_columns'] = dataset_ps.get_value(qualifier='columns', expand=True, **_skip_filter_checks)
                    info['mesh_datasets'] = list(set([c.split('@')[1] for c in info['mesh_columns'] if len(c.split('@'))>1]))
                    info['mesh_kinds'] = [b.filter(dataset=ds, context='dataset', **_skip_filter_checks).kind for ds in info['mesh_datasets']]

                if by_time:
                    for time_ in this_times:
                        # TODO: handle some deltatime allowance here?
                        if time_ in times:
                            ind = times.index(time_)
                            infolists[ind].append(info)
                        else:
                            times.append(time_)
                            infolists.append([info])
                else:
                    # TODO: this doesn't appear to be different than needed_syns,
                    # unless we change the structure to be per-dataset.
                    info['times'] = this_times
                    infolists.append(info)

                # we need the times for _create_syns but not within the infolists,
                # so we'll do this last and make a copy so times aren't passed
                # to everybody...
                needed_syn_info = info.copy()
                needed_syn_info['times'] = this_times
                needed_syns.append(needed_syn_info)

    if by_time and len(times):
        ti = list(zip(times, infolists))
        ti.sort()
        times, infolists = zip(*ti)

    if by_time:
        # print "*** _extract_from_bundle return(times, infolists, syns)", times, infolists, needed_syns
        return np.asarray(times), infolists, _create_syns(b, needed_syns)
    else:
        # print "*** _extract_from_bundle return(infolists, syns)", infolists, needed_syns
        return infolists, _create_syns(b, needed_syns)

def _create_syns(b, needed_syns):
    """
    Create empty synthetics

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter list needed_syns: list of dictionaries containing kwargs to access
        the dataset (dataset, component, kind)
    :return: :class:`phoebe.parameters.parameters.ParameterSet` of all new parameters
    """

    # needs_mesh = {info['dataset']: info['kind'] for info in needed_syns if info['needs_mesh']}

    params = []
    for needed_syn in needed_syns:
        # print "*** _create_syns needed_syn", needed_syn
        # used to be {}_syn
        syn_kind = '{}'.format(needed_syn['kind'])
        # if needed_syn['kind']=='mesh':
            # parameters.dataset.mesh will handle creating the necessary columns
            # needed_syn['dataset_fields'] = needs_mesh

            # needed_syn['columns'] = b.get_value(qualifier='columns', dataset=needed_syn['dataset'], context='dataset')
            # datasets = b.get_value(qualifier='datasets', dataset=needed_syn['dataset'], context='dataset')
            # needed_syn['datasets'] = {ds: b.filter(datset=ds, context='dataset').kind for ds in datasets}

        # phoebe will compute everything sorted - even if the input times array
        # is out of order, so let's make sure the exposed times array is in
        # the correct (sorted) order
        if 'times' in needed_syn.keys():
            needed_syn['times'].sort()

            needed_syn['empty_arrays_len'] = len(needed_syn['times'])

        these_params, these_constraints = getattr(_dataset, syn_kind.lower())(syn=True, **needed_syn)
        # TODO: do we need to handle constraints?
        these_params = these_params.to_list()
        for param in these_params:
            if param._dataset is None:
                # dataset may be set for mesh columns
                param._dataset = needed_syn['dataset']

            param._kind = syn_kind
            param._component = needed_syn['component']
            # reset copy_for... model Parameters should never copy
            param._copy_for = {}

            # context, model, etc will be handle by the bundle once these are returned

        params += these_params

    return ParameterSet(params)

def _make_packet(qualifier, value, time, info, **kwargs):
    """
    where kwargs overrides info
    """
    packet = {'dataset': kwargs.get('dataset', info['dataset']),
              'component': kwargs.get('component', info['component']),
              'kind': kwargs.get('kind', info['kind']),
              'qualifier': qualifier,
              'value': value,
              'time': time
              }

    return packet

class BaseBackend(object):
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

    def _get_packet_and_syns(self, b, compute, dataset=None, times=[], **kwargs):
        """
        see get_packet_and_syns.  _get_packet_and_syns provides the custom parts
        of the packet that are Backend-dependent.

        This should return the packet to send to all workers and the new_syns to
        be sent to the master.

        return packet, new_syns
        """
        raise NotImplementedError("_get_packet_and_syns is not implemented by the {} backend".format(self.__class__.__name__))

    def get_packet_and_syns(self, b, compute, dataset=None, times=[], **kwargs):
        """
        get_packet is called by the master and must get all information necessary
        to send to all workers.  The returned packet will be passed on as
        _run_chunk(**packet) with the following exceptions:

        * b: the bundle will be included in the packet serialized
        * compute: the label of the compute options will be included in the packet
        * backend: the class name will be passed on in the packet so the worker can call the correct backend
        * all kwargs will be passed on verbatim
        """
        packet, new_syns = self._get_packet_and_syns(b, compute, dataset, times, **kwargs)
        for k,v in kwargs.items():
            if k=='system':
                # force the workers to rebuild the system instead of attempting to
                # send via MPI
                continue
            packet[k] = v

        if kwargs.get('max_computations', None) is not None:
            if len(packet.get('infolists', packet.get('infolist', []))) > kwargs.get('max_computations'):
                raise ValueError("more than {} computations detected ({} estimated).".format(kwargs.get('max_computations'), len(packet['infolists'])))

        packet['b'] = b.to_json() if mpi.enabled else b
        packet['compute'] = compute
        packet['backend'] = self.__class__.__name__

        return packet, new_syns

    def _run_chunk(self):
        """
        _run_chunk will receive the packet (with the bundle deserialized if necessary)
        and is responsible for computing and returning results.
        """
        # np.array_split(any_input_array, nprocs)[myrank]
        raise NotImplementedError("_run_chunk is not implemented by the {} backend".format(self.__class__.__name__))

    def _fill_syns(self, new_syns, rpacketlists_per_worker):
        """
        rpacket_per_worker is a list of packetlists as returned by _run_chunk
        """
        # TODO: move to BaseBackendByDataset or BaseBackend?
        logger.debug("rank:{}/{} {}._fill_syns".format(mpi.myrank, mpi.nprocs, self.__class__.__name__))

        for packetlists in rpacketlists_per_worker:
            # single worker
            for packetlist in packetlists:
                # single time/dataset
                for packet in packetlist:
                    # single parameter
                    try:
                        new_syns.set_value(check_visible=False, check_default=False, ignore_readonly=True, **packet)
                    except Exception as err:
                        raise ValueError("failed to set value from packet: {}.  Original error: {}".format(packet, str(err)))

        return new_syns

    def _run_worker(self, packet):
        # the worker receives the bundle serialized, so we need to unpack it
        logger.debug("rank:{}/{} _run_worker".format(mpi.myrank, mpi.nprocs))
        packet['b'] = phoebe.frontend.bundle.Bundle(packet['b'])
        # do the computations requested for this worker
        rpacketlists = self._run_chunk(**packet)
        # send the results back to the master (root=0)
        mpi.comm.gather(rpacketlists, root=0)

    def run(self, b, compute, dataset=None, times=[], **kwargs):
        """
        if within mpirun, workers should call _run_worker instead of run
        """
        self.run_checks(b, compute, times, **kwargs)

        logger.debug("rank:{}/{} calling get_packet_and_syns".format(mpi.myrank, mpi.nprocs))
        packet, new_syns = self.get_packet_and_syns(b, compute, dataset, times, **kwargs)

        if mpi.enabled:
            # broadcast the packet to ALL workers
            logger.debug("rank:{}/{} broadcasting to all workers".format(mpi.myrank, mpi.nprocs))
            mpi.comm.bcast(packet, root=0)

            # now even the master can become a worker and take on a chunk
            packet['b'] = b
            rpacketlists = self._run_chunk(**packet)

            # now receive all packetlists
            logger.debug("rank:{}/{} gathering packetlists from all workers".format(mpi.myrank, mpi.nprocs))
            rpacketlists_per_worker = mpi.comm.gather(rpacketlists, root=0)

        else:
            rpacketlists_per_worker = [self._run_chunk(**packet)]

        logger.debug("rank:{}/{} calling _fill_syns".format(mpi.myrank, mpi.nprocs))
        return self._fill_syns(new_syns, rpacketlists_per_worker)


class BaseBackendByTime(BaseBackend):


    def _worker_setup(self, b, compute, times, infolists, **kwargs):
        """
        """
        raise NotImplementedError("worker_setup not implemented by the {} backend".format(self.__class__.__name__))
        return dict()

    def _run_single_time(self, b, i, time, infolist, **kwargs):
        """
        """
        raise NotImplementedError("_run_single_time not implemented by the {} backend".format(self.__class__.__name__))

    def _get_packet_and_syns(self, b, compute, dataset=None, times=[], **kwargs):
        # extract times/infolists/new_syns from the bundle
        # if the input for times is an empty list, we'll obey dataset times
        # otherwise all datasets will be overridden with the times provided
        # see documentation in _extract_from_bundle for details on the output variables.
        times, infolists, new_syns = _extract_from_bundle(b, compute=compute,
                                                          dataset=dataset,
                                                          times=times,
                                                          by_time=True,
                                                          **kwargs)

        # prepare the packet to be sent to the workers.
        # this packet will be sent to _run_worker as **packet
        packet = {'times': times,
                  'infolists': infolists}

        return packet, new_syns


    def _run_chunk(self, b, compute, times, infolists, **kwargs):
        logger.debug("rank:{}/{} _run_chunk".format(mpi.myrank, mpi.nprocs))

        worker_setup_kwargs = self._worker_setup(b, compute, times, infolists, **kwargs)

        inds = range(len(times))

        if mpi.enabled:
            # np.array_split(any_input_array, mpi.nprocs)[mpi.myrank]
            inds = np.array_split(inds, mpi.nprocs)[mpi.myrank]
            times = np.array_split(times, mpi.nprocs)[mpi.myrank]
            infolists = np.array_split(infolists, mpi.nprocs)[mpi.myrank]

        packetlists = [] # entry per-time
        for i, time, infolist in _progressbar(zip(inds, times, infolists), total=len(times), show_progressbar=not b._within_solver and kwargs.get('progressbar', False)):
            if kwargs.get('out_fname', False) and os.path.isfile(kwargs.get('out_fname')+'.kill'):
                logger.warning("received kill signal, exiting sampler loop")
                break

            packetlist = self._run_single_time(b, i, time, infolist, **worker_setup_kwargs)
            packetlists.append(packetlist)

        logger.debug("rank:{}/{} _run_chunk returning packetlist".format(mpi.myrank, mpi.nprocs))
        return packetlists


class BaseBackendByDataset(BaseBackend):
    def _worker_setup(self, b, compute, infolist, **kwargs):
        """
        """
        raise NotImplementedError("worker_setup not implemented by the {} backend".format(self.__class__.__name__))
        return dict()

    def _run_single_dataset(self, b, info, **kwargs):
        """
        """
        raise NotImplementedError("_run_single_dataset not implemented by the {} backend".format(self.__class__.__name__))

    def _get_packet_and_syns(self, b, compute, dataset=None, times=[], **kwargs):
        # self.run_checks(b, compute, times, **kwargs)

        # see documentation in _extract_from_bundle for details on the output variables.
        infolist, new_syns = _extract_from_bundle(b, compute=compute,
                                                  dataset=dataset,
                                                  times=times, by_time=False)

        packet = {'infolist': infolist}

        return packet, new_syns


    def _run_chunk(self, b, compute, infolist, **kwargs):
        worker_setup_kwargs = self._worker_setup(b, compute, infolist, **kwargs)

        if mpi.enabled:
            # np.array_split(any_input_array, mpi.nprocs)[mpi.myrank]
            infolist = np.array_split(infolist, mpi.nprocs)[mpi.myrank]

        packetlists = [] # entry per-dataset
        for info in _progressbar(infolist, total=len(infolist), show_progressbar=not b._within_solver and kwargs.get('progressbar', False)):
            if kwargs.get('out_fname', False) and os.path.isfile(kwargs.get('out_fname')+'.kill'):
                logger.warning("received kill signal, exiting sampler loop")
                break
            packetlist = self._run_single_dataset(b, info, **worker_setup_kwargs)
            packetlists.append(packetlist)

        return packetlists

def _call_run_single_model(args):
    # NOTE: b should be a deepcopy here to prevent conflicts
    b, samples, sample_kwargs, compute, dataset, times, compute_kwargs, expose_samples, expose_failed, i, allow_retries = args
    # override sample_from
    compute_kwargs['sample_from'] = []
    compute_kwargs['progressbar'] = False

    success_samples = []
    failed_samples = {}

    while True:
        # print("trying with samples={}".format(samples))
        for uniqueid, value in samples.items():
            uniqueid, index = _extract_index_from_string(uniqueid)

            # TODO: for some reason when redrawing we're getting arrays with length
            # one as if we had passed N=1 to sample_distribution_collection.  For now, we'll
            # just work around the issue.
            if isinstance(value, np.ndarray):
                value = value[0]
            # print("setting uniqueid={}, value={}".format(uniqueid, value))
            ref_param = b.get_parameter(uniqueid=uniqueid, **_skip_filter_checks)
            try:
                if index is None:
                    ref_param.set_value(value)
                else:
                    ref_param.set_index_value(index, value)
            except Exception as err:
                if expose_samples:
                    msg = _simplify_error_message(err)
                    failed_samples[msg] = failed_samples.get(msg, []) + [list(samples.values())]

                samples = b.sample_distribution_collection(N=None, keys='uniqueid', **sample_kwargs)
                break

        try:
            model_ps = b.run_compute(compute=compute, dataset=dataset, times=times, do_create_fig_params=False, model='sample_{}'.format(i), **compute_kwargs)
        except Exception as err:
            # new random draw for the next attempt
            logger.warning("model failed: drawing new sample")
            if expose_failed:
                msg = _simplify_error_message(err)
                # TODO: remove the list comprehension here once the bug is fixed in distributions that is sometimes returning an array with one entry
                failed_samples[msg] = failed_samples.get(msg, []) + [list([s[0] if isinstance(s, np.ndarray) else s for s in samples.values()])]
                # failed_samples[msg] = failed_samples.get(msg, []) + [list(samples.values())]

            if allow_retries:
                samples = b.sample_distribution_collection(N=None, keys='uniqueid', **sample_kwargs)
                # continue the next iteration in the while loop
                allow_retries -= 1
            else:
                raise ValueError("single model exceeded number of allowed retries.  Check sampling distribution to ensure physical models exist.  Latest model error: {}".format(err))
        else:
            if expose_samples:
                # TODO: remove the list comprehension here once the bug is fixed in distributions that is sometimes returning an array with one entry
                success_samples += list([s[0] if isinstance(s, np.ndarray) else s for s in samples.values()])
                # success_samples += list(samples.values())

            return model_ps.to_json(), success_samples, failed_samples

def _test_single_sample(args):
    b_copy, uniqueids, sample_per_param, dc, require_priors, require_compute, require_checks, allow_retries = args
    success = False

    while not success:
        for uniqueid, sample_value in zip(uniqueids, sample_per_param):

            uniqueid, index = _extract_index_from_string(uniqueid)
            ref_param = b_copy.get_parameter(uniqueid=uniqueid, **_skip_filter_checks)

            try:
                if index is None:
                    ref_param.set_value(sample_value)
                else:
                    ref_param.set_index_value(index, sample_value)


            except:
                success = False
            else:
                success = True

        compute_for_checks = None
        if require_compute not in [True, False]:
            compute_for_checks = require_compute
        elif require_checks not in [True, False]:
            compute_for_checks = require_checks

        if require_priors and success:
            logp = b_copy.calculate_lnp(require_priors, include_constrained=True)
            if not np.isfinite(logp):
                success = False

        if (require_checks or require_compute) and success:

            if not b_copy.run_checks_compute(compute=compute_for_checks).passed:
                success = False
            else:
                success = True

        if require_compute and success:
            try:
                b_copy.run_compute(compute=compute_for_checks, skip_checks=True, progressbar=False, model='test', overwrite=True)
            except Exception as e:
                success = False
            else:
                success = True

        if not success:
            if allow_retries:
                allow_retries -= 1
                replacement_values = dc.sample(size=1)
                sample_per_param = replacement_values[0]
            else:
                raise ValueError("single sample exceeded number of allowed retries.  Check sampling distribution to make sure enough valid entries exist.")

    return sample_per_param



class SampleOverModel(object):
    def __init__(self):
        return

    def run_worker(self, packet):
        """
        run_worker will receive the packet (with the bundle deserialized if necessary)
        and is responsible for any work done by a worker within MPI
        """
        logger.debug("rank:{}/{} run_worker".format(mpi.myrank, mpi.nprocs))
        # packet['b'] = phoebe.frontend.bundle.Bundle(packet['b'])
        # do the computations requested for this worker
        # rpacketlists = self.run_single_model(**packet)
        # send the results back to the master (root=0)
        # mpi.comm.gather(rpacketlists, root=0)
        return self.run(**packet)

    def run(self, b, compute, dataset=None, times=[], **kwargs):
        """
        if within mpirun, workers should call _run_worker instead of run
        """
        # TODO: can we run the checks of the requested backend?
        #self.run_checks(b, compute, times, **kwargs)

        if mpi.within_mpirun:
            logger.info("run_compute sample_from using MPI")
            pool = _pool.MPIPool()
            is_master = pool.is_master()
        elif conf.multiprocessing_nprocs==0 or b.get_value(qualifier='sample_num', compute=compute, sample_num=kwargs.get('sample_num', None), **_skip_filter_checks) == 1:
            logger.info("run_compute sample_from: serial mode")
            pool = _pool.SerialPool()
            is_master = True
        else:
            logger.info("run_compute sample_from using multiprocessing with {} procs".format(conf.multiprocessing_nprocs))
            pool = _pool.MultiPool(processes=conf._multiprocessing_nprocs)
            is_master = True

        # temporarily disable MPI within run_compute to disabled parallelizing
        # per-time.
        within_mpirun = mpi.within_mpirun
        mpi_enabled = mpi.enabled
        mpi._within_mpirun = False
        mpi._enabled = False

        if is_master:
            compute_ps = b.get_compute(compute=compute, **_skip_filter_checks)
            compute_kwargs = {k:v for k,v in kwargs.items() if k in compute_ps.qualifiers+['progressbar', 'skip_checks', 'times'] and 'sample' not in k}

            # sample_from = compute_ps.get_value(qualifier='sample_from', sample_from=kwargs.get('sample_from', None), expand=True, **_skip_filter_checks)
            # sample_from_combine = compute_ps.get_value(qualifier='sample_from_combine', sample_from_combine=kwargs.get('sample_from_combine', None), **_skip_filter_checks)
            sample_num = compute_ps.get_value(qualifier='sample_num', sample_num=kwargs.get('sample_num', None), **_skip_filter_checks)
            sample_mode = compute_ps.get_value(qualifier='sample_mode', sample_mode=kwargs.get('sample_mode', None), **_skip_filter_checks)
            expose_samples = compute_ps.get_value(qualifier='expose_samples', expose_samples=kwargs.get('expose_samples', None), **_skip_filter_checks)
            expose_failed = compute_ps.get_value(qualifier='expose_failed', expose_failed=kwargs.get('expose_failed', None), **_skip_filter_checks)

            # samples = range(sample_num)
            # note: sample_from can be any combination of solutions and distributions
            distribution_filters, combine, include_constrained, to_univariates, to_uniforms, require_limits, require_checks, require_compute, require_priors = b._distribution_collection_defaults(qualifier='sample_from', context='compute', compute=compute_ps.compute, **kwargs)
            sample_kwargs = {'distribution_filters': distribution_filters,
                             'combine': combine,
                             'include_constrained': include_constrained,
                             'to_univariates': to_univariates,
                             'to_uniforms': to_uniforms,
                             'require_limits': require_limits,
                             'require_checks': require_checks,
                             'require_compute': require_compute,
                             'require_priors': require_priors}

            sample_dict = b.sample_distribution_collection(N=sample_num,
                                                keys='uniqueid',
                                                **sample_kwargs)

            # determine if ALL samples are the same
            allow_retries = 10
            if sample_num == 1:
                allow_retries = 0
            elif np.all([len(np.unique(v))==1 for v in sample_dict.values()]):
                allow_retries = 0

            if len(list(sample_dict.values())[0]) == 1 and sample_mode != 'all':
                logger.warning("only one sample, falling back on sample_mode='all', sample_num=1 instead of sample_mode='{}', sample_num={}".format(sample_mode, sample_num))
                sample_num = 1
                sample_mode = 'all'

            global _active_pbar
            if kwargs.get('progressbar', True):
                _active_pbar = _tqdm(total=sample_num)
            else:
                _active_pbar = None

            def _sample_progress(*args):
                global _active_pbar
                if _active_pbar is not None:
                    _active_pbar.update(1)

            bexcl = b.copy()
            bexcl.remove_parameters_all(context=['model', 'solver', 'solutoin', 'figure'], **_skip_filter_checks)
            bexcl.remove_parameters_all(kind=['orb', 'mesh'], context='dataset', **_skip_filter_checks)
            args_per_sample = [(bexcl.copy(), {k:v[i] for k,v in sample_dict.items()}, sample_kwargs, compute, dataset, times, compute_kwargs, expose_samples, expose_failed, i, allow_retries) for i in range(sample_num)]
            models_success_failed = list(pool.map(_call_run_single_model, args_per_sample, callback=_sample_progress))
        else:
            pool.wait()

        if pool is not None:
            pool.close()

        if _active_pbar is not None:
            _active_pbar.close()

        # restore previous MPI state
        mpi._within_mpirun = within_mpirun
        mpi._enabled = mpi_enabled

        if is_master:
            # TODO: merge the models as requested by sample_mode

            # the first entry only has the figure parameter, so we'll make a copy of that to re-popluate with new values
            models = [msf[0] for msf in models_success_failed]
            ret_ps = ParameterSet(models[0])
            all_models_ps = ParameterSet(list(itertools.chain.from_iterable(models)))

            # merge failed dicts
            if expose_samples:
                success_samples = [msf[1] for msf in models_success_failed]

            if expose_failed:
                samples_dicts = [msf[2] for msf in models_success_failed]
                failed_samples = {}
                for samples_dict in samples_dicts:
                    for msg, samples in samples_dict.items():
                        failed_samples[msg] = failed_samples.get(msg, []) + samples

            for param in ret_ps.to_list():
                param._bundle = None
                if param.qualifier in ['fluxes', 'fluxes_nogps', 'gps', 'rvs']:
                    all_values = np.array([p.get_value() for p in all_models_ps.filter(qualifier=param.qualifier, dataset=param.dataset, component=param.component, **_skip_filter_checks).to_list()])
                    if sample_mode == 'all':
                        param.set_value(all_values, ignore_readonly=True)
                    elif sample_mode == 'median':
                        param.set_value(np.median(all_values, axis=0), ignore_readonly=True)
                    elif sample_mode in ['1-sigma', '3-sigma', '5-sigma']:
                        sigma = int(sample_mode[0])
                        bounds = np.percentile(all_values, 100 * _norm.cdf([-sigma, 0, sigma]), axis=0)
                        param.set_value(bounds, ignore_readonly=True)
                    else:
                        raise NotImplementedError("sample_mode='{}' not implemented".format(sample_mode))
                param._bundle = b
                param._model = kwargs.get('model')

            addl_params = []
            addl_params += [StringParameter(qualifier='sample_mode', value=sample_mode, readonly=True, description='mode used for sampling')]
            addl_params += [ArrayParameter(qualifier='sampled_uniqueids', value=list(sample_dict.keys()), advanced=True, readonly=True, description='uniqueids of sampled parameters')]
            addl_params += [ArrayParameter(qualifier='sampled_twigs', value=[b.get_parameter(uniqueid=uniqueid, **_skip_filter_checks).twig for uniqueid in sample_dict.keys()], readonly=True, description='twigs of sampled parameters')]
            if expose_samples:
                addl_params += [ArrayParameter(qualifier='samples', value=success_samples, readonly=True, description='samples that were drawn and successfully computed (in the units at the time run_compute was called).')]
            if expose_failed:
                addl_params += [DictParameter(qualifier='failed_samples', value=failed_samples, readonly=True, description='samples that were drawn but failed to compute (in the units at the time run_compute was called).  Dictionary keys are the messages with values being an array with shape (N, len(fitted_uniqueids))')]

            return ret_ps + ParameterSet(addl_params)
        return



class PhoebeBackend(BaseBackendByTime):
    """
    See <phoebe.parameters.compute.phoebe>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_compute>
    * <phoebe.frontend.bundle.Bundle.run_compute>
    """

    def run_checks(self, b, compute, times=[], **kwargs):
        logger.debug("rank:{}/{} run_checks".format(mpi.myrank, mpi.nprocs))

        computeparams = b.get_compute(compute, force_ps=True)
        hier = b.get_hierarchy()

        starrefs  = hier.get_stars()
        meshablerefs = hier.get_meshables()

        if len(starrefs)==1 and computeparams.get_value(qualifier='distortion_method', component=starrefs[0], **kwargs) in ['roche', 'none']:
            raise ValueError("distortion_method='{}' not valid for single star".format(computeparams.get_value(qualifier='distortion_method', component=starrefs[0], **kwargs)))

    def _compute_intrinsic_system_at_t0(self, b, compute,
                                          dynamics_method=None,
                                          hier=None,
                                          meshablerefs=None,
                                          datasets=None,
                                          reset=True,
                                          lc_only=True,
                                          **kwargs):

        logger.debug("rank:{}/{} PhoebeBackend._create_system_and_compute_pblums: calling universe.System.from_bundle".format(mpi.myrank, mpi.nprocs))
        system = universe.System.from_bundle(b, compute, datasets=b.datasets, **kwargs)

        if dynamics_method is None:
            if compute is None:
                dynamics_method = 'keplerian'
            else:
                computeparams = b.get_compute(compute, force_ps=True)
                dynamics_method = computeparams.get_value(qualifier='dynamics_method', dynamics_method=kwargs.get('dynamics_method', None), default='keplerian', **_skip_filter_checks)

        if hier is None:
            hier = b.get_hierarchy()

        if meshablerefs is None:
            starrefs  = hier.get_stars()
            meshablerefs = hier.get_meshables()

        t0 = b.get_value(qualifier='t0', context='system', unit=u.d, t0=kwargs.get('t0', None), **_skip_filter_checks)

        if len(meshablerefs) > 1 or hier.get_kind_of(meshablerefs[0])=='envelope':
            logger.debug("rank:{}/{} PhoebeBackend._create_system_and_compute_pblums: computing dynamics at t0".format(mpi.myrank, mpi.nprocs))
            if dynamics_method in ['nbody', 'rebound']:
                t0, xs0, ys0, zs0, vxs0, vys0, vzs0, inst_ds0, inst_Fs0, ethetas0, elongans0, eincls0 = dynamics.nbody.dynamics_from_bundle(b, [t0], compute, return_roche_euler=True, **kwargs)

            elif dynamics_method == 'bs':
                # TODO: pass stepsize
                # TODO: pass orbiterror
                # TODO: make sure that this takes systemic velocity and corrects positions and velocities (including ltte effects if enabled)
                t0, xs0, ys0, zs0, vxs0, vys0, vzs0, inst_ds0, inst_Fs0, ethetas0, elongans0, eincls0 = dynamics.nbody.dynamics_from_bundle_bs(b, [t0], compute, return_roche_euler=True, **kwargs)

            elif dynamics_method=='keplerian':
                # TODO: make sure that this takes systemic velocity and corrects positions and velocities (including ltte effects if enabled)
                t0, xs0, ys0, zs0, vxs0, vys0, vzs0, ethetas0, elongans0, eincls0 = dynamics.keplerian.dynamics_from_bundle(b, [t0], compute, return_euler=True, **kwargs)

            else:
                raise NotImplementedError

            x0, y0, z0, vx0, vy0, vz0, etheta0, elongan0, eincl0 = dynamics.dynamics_at_i(xs0, ys0, zs0, vxs0, vys0, vzs0, ethetas0, elongans0, eincls0, i=0)

        else:
            # singlestar case
            incl = b.get_value(qualifier='incl', component=meshablerefs[0], unit=u.rad)
            long_an = b.get_value(qualifier='long_an', component=meshablerefs[0], unit=u.rad)

            x0, y0, z0 = [0.], [0.], [0.]
            vx0, vy0, vz0 = [0.], [0.], [0.]
            etheta0, elongan0, eincl0 = [0.], [long_an], [incl]

        # Now we need to compute intensities at t0 in order to scale pblums for all future times
        # but only if any of the enabled datasets require intensities
        enabled_ps = b.filter(qualifier='enabled', compute=compute, value=True)
        if datasets is None:
            datasets = enabled_ps.datasets
        # kinds = [b.get_dataset(dataset=ds).kind for ds in datasets]

        system.update_positions(t0, x0, y0, z0, vx0, vy0, vz0, etheta0, elongan0, eincl0, ignore_effects=True)
        system.populate_observables(t0, ['lc' for dataset in datasets], datasets, ignore_effects=True)


        if reset:
            logger.debug("rank:{}/{} PhoebeBackend._create_system_and_compute_pblums: resetting system".format(mpi.myrank, mpi.nprocs))
            system.reset(force_recompute_instantaneous=True)

        return system


    def _worker_setup(self, b, compute, times, infolists, **kwargs):
        logger.debug("rank:{}/{} PhoebeBackend._worker_setup: extracting parameters".format(mpi.myrank, mpi.nprocs))
        computeparams = b.get_compute(compute, force_ps=True)
        hier = b.get_hierarchy()
        starrefs  = hier.get_stars()
        meshablerefs = hier.get_meshables()
        do_horizon = False #computeparams.get_value(qualifier='horizon', **kwargs)
        dynamics_method = computeparams.get_value(qualifier='dynamics_method', dynamics_method=kwargs.pop('dynamics_method', None), **_skip_filter_checks)
        ltte = computeparams.get_value(qualifier='ltte', ltte=kwargs.pop('ltte', None), **_skip_filter_checks)

        # b.compute_ld_coeffs(set_value=True) # TODO: only need if irradiation is enabled and only for bolometric

        system = kwargs.get('system', universe.System.from_bundle(b, compute, datasets=b.datasets, **kwargs))
        # pblums_scale computed within run_compute and then passed as kwarg to run (so should be in kwargs sent to each worker)
        pblums_scale = kwargs.get('pblums_scale')
        for dataset in list(pblums_scale.keys()):
            for comp, pblum_scale in pblums_scale[dataset].items():
                system.get_body(comp).set_pblum_scale(dataset, component=comp, pblum_scale=pblum_scale)

        if len(meshablerefs) > 1 or hier.get_kind_of(meshablerefs[0])=='envelope':
            logger.debug("rank:{}/{} PhoebeBackend._worker_setup: computing dynamics at all times".format(mpi.myrank, mpi.nprocs))
            if dynamics_method in ['nbody', 'rebound']:
                ts, xs, ys, zs, vxs, vys, vzs, inst_ds, inst_Fs, ethetas, elongans, eincls = dynamics.nbody.dynamics_from_bundle(b, times, compute, return_roche_euler=True, **kwargs)

            elif dynamics_method == 'bs':
                # if distortion_method == 'roche':
                    # raise ValueError("distortion_method '{}' not compatible with dynamics_method '{}'".format(distortion_method, dynamics_method))

                # TODO: pass stepsize
                # TODO: pass orbiterror
                # TODO: make sure that this takes systemic velocity and corrects positions and velocities (including ltte effects if enabled)
                ts, xs, ys, zs, vxs, vys, vzs, inst_ds, inst_Fs, ethetas, elongans, eincls = dynamics.nbody.dynamics_from_bundle_bs(b, times, compute, return_roche_euler=True, **kwargs)

            elif dynamics_method=='keplerian':
                # TODO: make sure that this takes systemic velocity and corrects positions and velocities (including ltte effects if enabled)
                ts, xs, ys, zs, vxs, vys, vzs, ethetas, elongans, eincls = dynamics.keplerian.dynamics_from_bundle(b, times, compute, return_euler=True, **kwargs)

            else:
                raise NotImplementedError

        else:
            # singlestar case
            incl = b.get_value(qualifier='incl', component=meshablerefs[0], unit=u.rad)
            long_an = b.get_value(qualifier='long_an', component=meshablerefs[0], unit=u.rad)
            vgamma = b.get_value(qualifier='vgamma', context='system', unit=u.solRad/u.d)
            t0 = b.get_value(qualifier='t0', context='system', unit=u.d)

            ts = [times]
            vxs, vys, vzs = [np.zeros(len(times))], [np.zeros(len(times))], [np.zeros(len(times))]
            xs, ys, zs = [np.zeros(len(times))], [np.zeros(len(times))], [np.full(len(times), vgamma)]
            ethetas, elongans, eincls = [np.zeros(len(times))], [np.full(len(times), long_an)], [np.full(len(times), incl)]

            for i,t in enumerate(times):
                zs[0][i] = vgamma*(t-t0)

        return dict(system=system,
                    hier=hier,
                    meshablerefs=meshablerefs,
                    starrefs=starrefs,
                    dynamics_method=dynamics_method,
                    ts=ts, xs=xs, ys=ys, zs=zs,
                    vxs=vxs, vys=vys, vzs=vzs,
                    ethetas=ethetas, elongans=elongans, eincls=eincls)

    def _run_single_time(self, b, i, time, infolist, **kwargs):
        logger.debug("rank:{}/{} PhoebeBackend._run_single_time(i={}, time={}, infolist={}, **kwargs.keys={})".format(mpi.myrank, mpi.nprocs, i, time, infolist, kwargs.keys()))

        # unpack all backend-dependent things from the received packet
        system = kwargs.get('system')
        hier = kwargs.get('hier')
        meshablerefs = kwargs.get('meshablerefs')
        starrefs = kwargs.get('starrefs')
        dynamics_method = kwargs.get('dynamics_method')
        xs = kwargs.get('xs')
        ys = kwargs.get('ys')
        zs = kwargs.get('zs')
        vxs = kwargs.get('vxs')
        vys = kwargs.get('vys')
        vzs = kwargs.get('vzs')
        ethetas = kwargs.get('ethetas')
        elongans = kwargs.get('elongans')
        eincls = kwargs.get('eincls')

        # Check to see what we might need to do that requires a mesh
        # TODO: make sure to use the requested distortion_method

        # we need to extract positions, velocities, and euler angles of ALL bodies at THIS TIME (i)
        logger.debug("rank:{}/{} PhoebeBackend._run_single_time: extracting dynamics at time={}".format(mpi.myrank, mpi.nprocs, time))
        xi, yi, zi, vxi, vyi, vzi, ethetai, elongani, eincli = dynamics.dynamics_at_i(xs, ys, zs, vxs, vys, vzs, ethetas, elongans, eincls, i=i)

        if True in [info['needs_mesh'] for info in infolist]:

            if dynamics_method in ['nbody', 'rebound']:
                di = dynamics.at_i(inst_ds, i)
                Fi = dynamics.at_i(inst_Fs, i)
                # by passing these along to update_positions, volume conservation will
                # handle remeshing the stars
            else:
                # then allow d to be determined from orbit and original sma
                # and F to remain fixed
                di = None
                Fi = None


            # TODO: eventually we can pass instantaneous masses and sma as kwargs if they're time dependent
            # masses = [b.get_value(qualifier='mass', component=star, context='component', time=time, unit=u.solMass) for star in starrefs]
            # sma = b.get_value(qualifier='sma', component=starrefs[body.ind_self], context='component', time=time, unit=u.solRad)

            logger.debug("rank:{}/{} PhoebeBackend._run_single_time: calling system.update_positions at time={}".format(mpi.myrank, mpi.nprocs, time))
            system.update_positions(time, xi, yi, zi, vxi, vyi, vzi, ethetai, elongani, eincli, ds=di, Fs=Fi)

            # Now we need to determine which triangles are visible and handle subdivision
            # NOTE: this should come after populate_observables so that each subdivided triangle
            # will have identical local quantities.  The only downside to this is that we can't
            # make a shortcut and only populate observables at known-visible triangles - but
            # frankly that wouldn't save much time anyways and would then be annoying when
            # inspecting or plotting the mesh
            # NOTE: this has been moved before populate observables now to make use
            # of per-vertex weights which are used to determine the physical quantities
            # (ie teff, logg) that should be used in computing observables (ie intensity)

            # expose_horizon = 'mesh' in [info['kind'] for info in infolist] and do_horizon
            expose_horizon = False
            logger.debug("rank:{}/{} PhoebeBackend._run_single_time: calling system.handle_eclipses at time={}".format(mpi.myrank, mpi.nprocs, time))
            horizons = system.handle_eclipses(expose_horizon=expose_horizon)

            # Now we can fill the observables per-triangle.  We'll wait to integrate
            # until we're ready to fill the synthetics
            # First we need to determine which datasets must be populated at this
            # time.  populate_kinds must be the same length (i.e. not necessarily
            # a unique set... there may be several lcs and/or several rvs) as
            # populate_datasets which should be a unique set
            populate_datasets = []
            populate_kinds = []
            for info in infolist:
                if info['dataset'] not in populate_datasets:
                    populate_datasets.append(info['dataset'])
                    populate_kinds.append(info['kind'])

                    if info['kind'] == 'mesh':
                        # then we also need to populate based on any requested
                        # passband-dependent columns
                        for mesh_kind, mesh_dataset in zip(info['mesh_kinds'], info['mesh_datasets']):
                            if mesh_dataset not in populate_datasets:
                                populate_datasets.append(mesh_dataset)
                                populate_kinds.append(mesh_kind)

            logger.debug("rank:{}/{} PhoebeBackend._run_single_time: calling system.populate_observables at time={}".format(mpi.myrank, mpi.nprocs, time))
            system.populate_observables(time, populate_kinds, populate_datasets)

        logger.debug("rank:{}/{} PhoebeBackend._run_single_time: filling packets at time={}".format(mpi.myrank, mpi.nprocs, time))
        # now let's loop through and prepare a packet which will fill the synthetics
        packetlist = []
        for k, info in enumerate(infolist):
            packet = dict()

            # i, time, info['kind'], info['component'], info['dataset']
            cind = starrefs.index(info['component']) if info['component'] in starrefs else None
            # ts[i], xs[cind][i], ys[cind][i], zs[cind][i], vxs[cind][i], vys[cind][i], vzs[cind][i]
            kind = info['kind']

            # now check the kind to see what we need to fill
            if kind=='lp':
                profile_func = b.get_value(qualifier='profile_func',
                                           dataset=info['dataset'],
                                           context='dataset')

                profile_rest = b.get_value(qualifier='profile_rest',
                                           dataset=info['dataset'],
                                           context='dataset')

                profile_sv = b.get_value(qualifier='profile_sv',
                                         dataset=info['dataset'],
                                         context='dataset')  # UNITS???

                wavelengths = b.get_value(qualifier='wavelengths',
                                          component=info['component'],
                                          dataset=info['dataset'],
                                          context='dataset',
                                          unit=u.nm)

                if info['component'] in b.hierarchy.get_stars():
                    lp_components = info['component']
                elif info['component'] in b.hierarchy.get_orbits():
                    lp_components = b.hierarchy.get_stars_of_children_of(info['component'])
                else:
                    raise NotImplementedError

                obs = system.observe(info['dataset'],
                                     kind=kind,
                                     components=lp_components,
                                     profile_func=profile_func,
                                     profile_rest=profile_rest,
                                     profile_sv=profile_sv,
                                     wavelengths=wavelengths)

                # TODO: copy the original for wavelengths just like we do with
                # times and don't use packets at all
                packetlist.append(_make_packet('wavelengths',
                                              wavelengths*u.nm,
                                              None, info))

                packetlist.append(_make_packet('flux_densities',
                                              obs['flux_densities']*u.W/(u.m**2*u.nm),
                                              time, info))
            elif kind=='rv':
                ### this_syn['times'].append(time) # time array was set when initializing the syns
                if info['needs_mesh']:
                    obs = system.observe(info['dataset'],
                                         kind=kind,
                                         components=info['component'])

                    rv = obs['rv'] + b.get_value(qualifier='rv_offset',
                                                 component=info['component'],
                                                 dataset=info['dataset'],
                                                 context='dataset',
                                                unit=u.solRad/u.d,
                                                 **_skip_filter_checks)
                else:
                    # then rv_method == 'dynamical'
                    rv = -1*vzi[cind]

                packetlist.append(_make_packet('rvs',
                                              rv*u.solRad/u.d,
                                              time, info))

            elif kind=='lc':
                obs = system.observe(info['dataset'],
                                     kind=kind,
                                     components=info['component'])

                packetlist.append(_make_packet('fluxes',
                                              obs['flux']*u.W/u.m**2,
                                              time, info))

            elif kind=='etv':

                # TODO: add support for other etv kinds (barycentric, robust, others?)
                time_ecl = etvs.crossing(b, info['component'], time, dynamics_method, ltte, tol=computeparams.get_value(qualifier='etv_tol', unit=u.d, dataset=info['dataset'], component=info['component']))

                this_obs = b.filter(dataset=info['dataset'], component=info['component'], context='dataset')

                # TODO: there must be a better/cleaner way to get to Ns
                packetlist.append(_make_packet('Ns',
                                              this_obs.get_parameter(qualifier='Ns').interp_value(time_ephems=time),
                                              time, info))

                # NOTE: no longer under constraint control
                packetlist.append(_make_packet('time_ephems',
                                              time,
                                              time, info))

                packetlist.append(_make_packet('time_ecls',
                                              time_ecl,
                                              time, info))

                # NOTE: no longer under constraint control
                packetlist.append(_make_packet('etvs',
                                              time_ecl-time,
                                              time, info))


            elif kind=='orb':
                # ts[i], xs[cind][i], ys[cind][i], zs[cind][i], vxs[cind][i], vys[cind][i], vzs[cind][i]

                # times array was set when creating the synthetic ParameterSet

                packetlist.append(_make_packet('us',
                                              xi[cind] * u.solRad,
                                              time, info))

                packetlist.append(_make_packet('vs',
                                              yi[cind] * u.solRad,
                                              time, info))

                packetlist.append(_make_packet('ws',
                                              zi[cind] * u.solRad,
                                              time, info))

                packetlist.append(_make_packet('vus',
                                              vxi[cind] * u.solRad/u.d,
                                              time, info))

                packetlist.append(_make_packet('vvs',
                                              vyi[cind] * u.solRad/u.d,
                                              time, info))

                packetlist.append(_make_packet('vws',
                                              vzi[cind] * u.solRad/u.d,
                                              time, info))

            elif kind=='mesh':
                body = system.get_body(info['component'])
                if body.mesh is None:
                    continue

                if 'uvw' in info['mesh_coordinates']:
                    packetlist.append(_make_packet('uvw_elements',
                                                  body.mesh.vertices_per_triangle,
                                                  time, info))
                    packetlist.append(_make_packet('uvw_normals',
                                                  body.mesh.tnormals,
                                                  time, info))

                if 'xyz' in info['mesh_coordinates']:
                    packetlist.append(_make_packet('xyz_elements',
                                                  body.mesh.roche_vertices_per_triangle,
                                                  time, info))
                    packetlist.append(_make_packet('xyz_normals',
                                                  body.mesh.roche_tnormals,
                                                  time, info))

                # if 'pot' in info['mesh_columns']:
                    # packetlist.append(_make_packet('pot',
                                                  # body._instantaneous_pot,
                                                  # time, info))
                # if 'rpole' in info['mesh_columns']:
                #     packetlist.append(_make_packet('rpole',
                #                                   roche.potential2rpole(body._instantaneous_pot, body.q, body.ecc, body.F, body._scale, component=body.comp_no),
                #                                   time, info))
                if 'volume' in info['mesh_columns']:
                    packetlist.append(_make_packet('volume',
                                                  body.mesh.volume,
                                                  time, info))

                if 'us' in info['mesh_columns']:
                    # UNIT: u.solRad
                    packetlist.append(_make_packet('us',
                                                  body.mesh.centers[:,0],
                                                  time, info))
                if 'vs' in info['mesh_columns']:
                    # UNIT: u.solRad
                    packetlist.append(_make_packet('vs',
                                                  body.mesh.centers[:,1],
                                                  time, info))
                if 'ws' in info['mesh_columns']:
                    # UNIT: u.solRad
                    packetlist.append(_make_packet('ws',
                                                  body.mesh.centers[:,2],
                                                  time, info))

                if 'vus' in info['mesh_columns']:
                    packetlist.append(_make_packet('vus',
                                                  body.mesh.velocities.centers[:,0] * u.solRad/u.d,
                                                  time, info))
                if 'vvs' in info['mesh_columns']:
                    packetlist.append(_make_packet('vvs',
                                                  body.mesh.velocities.centers[:,1] * u.solRad/u.d,
                                                  time, info))
                if 'vws' in info['mesh_columns']:
                    packetlist.append(_make_packet('vws',
                                                  body.mesh.velocities.centers[:,2] * u.solRad/u.d,
                                                  time, info))

                # if 'uvw_normals' in info['mesh_columns']:
                #     packetlist.append(_make_packet('uvw_normals',
                #                                   body.mesh.tnormals,
                #                                   time, info))

                if 'nus' in info['mesh_columns']:
                    packetlist.append(_make_packet('nus',
                                                  body.mesh.tnormals[:,0],
                                                  time, info))
                if 'nvs' in info['mesh_columns']:
                    packetlist.append(_make_packet('nvs',
                                                  body.mesh.tnormals[:,1],
                                                  time, info))
                if 'nws' in info['mesh_columns']:
                    packetlist.append(_make_packet('nws',
                                                  body.mesh.tnormals[:,2],
                                                  time, info))


                if 'xs' in info['mesh_columns']:
                    packetlist.append(_make_packet('xs',
                                                  body.mesh.roche_centers[:,0],
                                                  time, info))
                if 'ys' in info['mesh_columns']:
                    packetlist.append(_make_packet('ys',
                                                  body.mesh.roche_centers[:,1],
                                                  time, info))
                if 'zs' in info['mesh_columns']:
                    packetlist.append(_make_packet('zs',
                                                  body.mesh.roche_centers[:,2],
                                                  time, info))

                if 'vxs' in info['mesh_columns']:
                    packetlist.append(_make_packet('vxs',
                                                  body.mesh.roche_cvelocities[:,0] * u.solRad/u.d,
                                                  time, info))
                if 'vys' in info['mesh_columns']:
                    packetlist.append(_make_packet('vys',
                                                  body.mesh.roche_cvelocities[:,1] * u.solRad/u.d,
                                                  time, info))
                if 'vzs' in info['mesh_columns']:
                    packetlist.append(_make_packet('vzs',
                                                  body.mesh.roche_cvelocities[:,2] * u.solRad/u.d,
                                                  time, info))

                # if 'xyz_normals' in info['mesh_columns']:
                #     packetlist.append(_make_packet('xyz_normals',
                #                                   body.mesh.tnormals,
                #                                   time, info))

                if 'nxs' in info['mesh_columns']:
                    packetlist.append(_make_packet('nxs',
                                                  body.mesh.roche_tnormals[:,0],
                                                  time, info))
                if 'nys' in info['mesh_columns']:
                    packetlist.append(_make_packet('nys',
                                                  body.mesh.roche_tnormals[:,1],
                                                  time, info))
                if 'nzs' in info['mesh_columns']:
                    packetlist.append(_make_packet('nzs',
                                                  body.mesh.roche_tnormals[:,2],
                                                  time, info))


                if 'areas' in info['mesh_columns']:
                    # UNIT: u.solRad**2
                    packetlist.append(_make_packet('areas',
                                                  body.mesh.areas,
                                                  time, info))
                # if 'tareas' in info['mesh_columns']:
                    # packetlist.append(_make_packet('tareas',
                                                  # body.mesh.tareas,
                                                  # time, info))




                if 'rs' in info['mesh_columns']:
                    packetlist.append(_make_packet('rs',
                                                  body.mesh.rs.centers*body._scale,
                                                  time, info))
                # if 'cosbetas' in info['mesh_columns']:
                    # packetlist.append(_make_packet('cosbetas',
                                                  # body.mesh.cosbetas,
                                                  # time, info))


                if 'loggs' in info['mesh_columns']:
                    packetlist.append(_make_packet('loggs',
                                                  body.mesh.loggs.centers,
                                                  time, info))
                if 'teffs' in info['mesh_columns']:
                    packetlist.append(_make_packet('teffs',
                                                  body.mesh.teffs.centers,
                                                  time, info))


                if 'rprojs' in info['mesh_columns']:
                    packetlist.append(_make_packet('rprojs',
                                                  body.mesh.rprojs.centers*body._scale,
                                                  time, info))
                if 'mus' in info['mesh_columns']:
                    packetlist.append(_make_packet('mus',
                                                  body.mesh.mus,
                                                  time, info))
                if 'visibilities' in info['mesh_columns']:
                    packetlist.append(_make_packet('visibilities',
                                                  body.mesh.visibilities,
                                                  time, info))
                if 'visible_centroids' in info['mesh_columns']:
                    vcs = np.sum(body.mesh.vertices_per_triangle*body.mesh.weights[:,:,np.newaxis], axis=1)
                    for i,vc in enumerate(vcs):
                        if np.all(vc==np.array([0,0,0])):
                            vcs[i] = np.full(3, np.nan)

                    packetlist.append(_make_packet('visible_centroids',
                                                  vcs,
                                                  time, info))


                # Dataset-dependent quantities
                for mesh_dataset in info['mesh_datasets']:
                    if 'pblum_ext@{}'.format(mesh_dataset) in info['mesh_columns']:
                        packetlist.append(_make_packet('pblum_ext',
                                                      body.compute_luminosity(mesh_dataset),
                                                      time, info,
                                                      dataset=mesh_dataset,
                                                      component=info['component']))

                    if 'abs_pblum_ext@{}'.format(mesh_dataset) in info['mesh_columns']:
                        packetlist.append(_make_packet('abs_pblum_ext',
                                                      body.compute_luminosity(mesh_dataset, scaled=False),
                                                      time, info,
                                                      dataset=mesh_dataset,
                                                      component=info['component']))

                    if 'ptfarea@{}'.format(mesh_dataset) in info['mesh_columns']:
                        packetlist.append(_make_packet('ptfarea',
                                                      body.get_ptfarea(mesh_dataset),
                                                      time, info,
                                                      dataset=mesh_dataset,
                                                      component=info['component']))

                    for indep in ['rvs', 'dls',
                                  'intensities', 'normal_intensities',
                                  'abs_intensities', 'abs_normal_intensities',
                                  'boost_factors', 'ldint']:

                        if "{}@{}".format(indep, mesh_dataset) in info['mesh_columns']:
                            key = "{}:{}".format(indep, mesh_dataset)

                            value = body.mesh[key].centers

                            if indep in ['intensities', 'abs_intensities']:
                                # replace elements in the back with nan (these
                                # were computed internally with abs(mus) to
                                # prevent a crash)
                                mus = body.mesh.mus
                                value[mus<0] = np.nan

                            if indep=='rvs':
                                # rvs use solRad/d internally, but default to km/s in the dataset
                                value *= u.solRad/u.d

                            packetlist.append(_make_packet(indep,
                                                          value,
                                                          time, info,
                                                          dataset=mesh_dataset,
                                                          component=info['component']))

            else:
                raise NotImplementedError("kind {} not yet supported by this backend".format(kind))

        logger.debug("rank:{}/{} PhoebeBackend._run_single_time: returning packetlist at time={}".format(mpi.myrank, mpi.nprocs, time))

        return packetlist


class LegacyBackend(BaseBackendByDataset):
    """
    See <phoebe.parameters.compute.legacy>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_compute>
    * <phoebe.frontend.bundle.Bundle.run_compute>
    """

    def run_checks(self, b, compute, times=[], **kwargs):
        computeparams = b.get_compute(compute, force_ps=True)
        hier = b.get_hierarchy()

        starrefs  = hier.get_stars()
        meshablerefs = hier.get_meshables()

        # check whether phoebe legacy is installed
        if not _use_phb1:
            raise ImportError("phoebeBackend for phoebe legacy not found.  Install (see phoebe-project.org/1.0) and restart phoebe")

        if len(starrefs)!=2:
            raise ValueError("only binaries are supported by legacy backend")


    def _worker_setup(self, b, compute, infolist, **kwargs):
        """
        """
        logger.debug("rank:{}/{} LegacyBackend._worker_setup: creating temporary phoebe file".format(mpi.myrank, mpi.nprocs))
        global _phb1_config
        # make phoebe 1 file
        # tmp_filename = temp_name = next(tempfile._get_candidate_names())
        computeparams = b.get_compute(compute, force_ps=True)
 #       print('_phb1_config', _phb1_config)
        if _phb1_config == False:
            phb1.init()
            try:
                if hasattr(phb1, 'auto_configure'):
                    # then phb1 is phoebe_legacy
                    phb1.auto_configure()
                    _phb1_config = True
                else:
                    # then phb1 is phoebeBackend
                    phb1.configure()
                    _phb1_config = True
            except SystemError:
                raise SystemError("PHOEBE config failed: try creating PHOEBE config file through GUI")


        computeparams = b.get_compute(compute, force_ps=True)
        legacy_dict = io.pass_to_legacy(b, compute=compute, disable_l3=True, **kwargs)

        io.import_to_legacy(legacy_dict)

        # build lookup tables between the dataset labels and the indices needed
        # to pass to phoebe legacy
        lcinds = {}
        rvinds = {}
        for lcind in range(0, phb1.getpar('phoebe_lcno')):
            lcinds[phb1.getpar('phoebe_lc_id', lcind)] = lcind
        for rvind in range(0, phb1.getpar('phoebe_rvno')):
        #    print "rv curves", phb1.getpar('phoebe_rvno')
        #    print "creating dict", phb1.getpar('phoebe_rv_id', rvind), rvind
        #    rvinds[rvind] = phb1.getpar('phoebe_rv_id', rvind)
            rvid = phb1.getpar('phoebe_rv_id', rvind)
            comp = phb1.getpar('phoebe_rv_dep', rvind).split(' ')[0].lower()
            rvcurve = rvid+'-'+comp
            rvinds[rvcurve] = rvind

        return dict(lcinds=lcinds,
                    rvinds=rvinds,
                    computeparams=computeparams)

    def _run_single_dataset(self, b, info, **kwargs):
        """
        """
        logger.debug("rank:{}/{} LegacyBackend._run_single_dataset(info['dataset']={} info['component']={} info.keys={}, **kwargs.keys={})".format(mpi.myrank, mpi.nprocs, info['dataset'], info['component'], info.keys(), kwargs.keys()))

        def mqtf(value, direction=None):
            """
            mesh quadrant to full
            """
            # value is a 1-dimensional array, but probably a tuple if directly
            # from legacy
            value = np.asarray(value)

            # direction is either 'x', 'y', 'z' or None and gives the component
            # of the current array.
            # Plan for quadrants: no flip, flip y, flip z, flip y&z (must be same
            # order as in potentials.discretize_wd_style so we can compare meshes)
            if direction=='x':
                a = [+1,+1,+1,+1]
            elif direction=='y':
                a = [+1,-1,+1,-1]
            elif direction=='z':
                a = [+1,+1,-1,-1]
            else:
                # non-vector-component (like teffs/loggs)
                a = [1,1,1,1]

            return np.concatenate([a[0]*value, a[1]*value, a[2]*value, a[3]*value])


        lcinds = kwargs.get('lcinds')
        rvinds = kwargs.get('rvinds')
        computeparams = kwargs.get('computeparams')

        packetlist = []

        if info['kind'] == 'lc':
            # print(info['dataset'])
            # print('lcinds', lcinds)
            lcind = lcinds[info['dataset']]
            fluxes = np.array(phb1.lc(tuple(info['times'].tolist()), lcind))
            packetlist.append(_make_packet('fluxes',
                                           fluxes,
                                           None,
                                           info))

        elif info['kind'] == 'rv':
            comp = b.hierarchy.get_primary_or_secondary(info['component'])
            rvid =  info['dataset']
            rvcurve = rvid+'-'+comp
            rvind = rvinds[rvcurve]

            if comp == 'primary':
                proximity_par = 'phoebe_proximity_rv1_switch'
                rv_call = getattr(phb1, 'rv1')
            else:
                proximity_par = 'phoebe_proximity_rv2_switch'
                rv_call = getattr(phb1, 'rv2')

            # toggle the proximity parameter.  Legacy handles this
            # globally per-component, whereas phoebe2 has per-dataset
            # per-component switches.
            rv_method = computeparams.get_value(qualifier ='rv_method',
                                                component=info['component'],
                                                dataset=info['dataset'],
                                                **_skip_filter_checks)

            phb1.setpar(proximity_par, rv_method=='flux-weighted')

            rvs = np.array(rv_call(tuple(info['times'].tolist()), rvind))
            rvs += b.get_value(qualifier='rv_offset',
                               component=info['component'],
                               dataset=info['dataset'],
                               context='dataset',
                               unit=u.km/u.s,
                               **_skip_filter_checks)

            packetlist.append(_make_packet('rvs',
                                           rvs*u.km/u.s,
                                           None,
                                           info))


        elif info['kind'] == 'mesh':
            # NOTE: for each call we'll loop over times because the returned
            # mesh is concatenated over all times and we can't easily slice in
            # case the number of elements changes in time.

            # All columns in the legacy mesh are assigned to a component
            # by its index.  Within the infolists loop we're already assigned
            # a single component so let's determine which index we need to access
            # for all of the columns.  This will create some overhead, as legacy
            # will compute all of these meshes twice... but that's probably easier
            # than caching the results until we have the other component.
            # Note that this isn't a problem for RVs as phoebe2 allows different
            # times per-component and legacy has separate calls to rv1 and rv2
            if b.hierarchy.get_primary_or_secondary(info['component']) == 'primary':
                cind = 1
            else:
                cind = 2

            # first we'll get the geometric/bolometric data
            for time in info['times']:
                # TODO: what happens if there are no LCs attached?
                flux, mesh = phb1.lc((time,), 0, True)

                # legacy stores everything in Roche coordinates
                xs = mqtf(mesh['vcx{}'.format(cind)], 'x')
                ys = mqtf(mesh['vcy{}'.format(cind)], 'y')
                zs = mqtf(mesh['vcz{}'.format(cind)], 'z')
                xyz_elements = np.array([xs, ys, zs]).T   # Nx3
                nxs = mqtf(mesh['grx{}'.format(cind)], 'x')
                nys = mqtf(mesh['gry{}'.format(cind)], 'y')
                nzs = mqtf(mesh['grz{}'.format(cind)], 'z')
                # TODO: add velocities once supported by PHOEBE 1
                # vxs =
                # vys =
                # vzs =


                # TODO: convert to POS for this given time
                # us =
                # vs =
                # ws =
                # uvw_elements = np.array([us, vs, ws]).T   # Nx3
                # nus =
                # nvs =
                # nws =
                # vus =
                # vvs =
                # vws =

                packetlist.append(_make_packet('xyz_elements',
                                               xyz_elements,
                                               time,
                                               info))


                # this_syn.set_value(time=time, qualifier='uvw_elements', value=uvw_elements)

                # if 'us' in info['mesh_columns']:
                #     this_syn.set_value(time=time, qualifier='us', value=us)
                # if 'vs' in info['mesh_columns']:
                #     this_syn.set_value(time=time, qualifier='vs', value=vs)
                # if 'ws' in info['mesh_columns']:
                #     this_syn.set_value(time=time, qualifier='ws', value=ws)
                # if 'nus' in info['mesh_columns']:
                #     this_syn.set_value(time=time, qualifier='nus', value=nus)
                # if 'nvs' in info['mesh_columns']:
                #     this_syn.set_value(time=time, qualifier='nvs', value=nvs)
                # if 'nws' in info['mesh_columns']:
                #     this_syn.set_value(time=time, qualifier='nws', value=nws)
                # if 'vus' in info['mesh_columns']:
                #     this_syn.set_value(time=time, qualifier='vus', value=vus)
                # if 'vvs' in info['mesh_columns']:
                #     this_syn.set_value(time=time, qualifier='vvs', value=vvs)
                # if 'vws' in info['mesh_columns']:
                #     this_syn.set_value(time=time, qualifier='vws', value=vws)


                if 'xs' in info['mesh_columns']:
                    packetlist.append(_make_packet('xs',
                                                   xs,
                                                   time,
                                                   info))
                if 'ys' in info['mesh_columns']:
                    packetlist.append(_make_packet('ys',
                                                   ys,
                                                   time,
                                                   info))
                if 'zs' in info['mesh_columns']:
                    packetlist.append(_make_packet('zs',
                                                   zs,
                                                   time,
                                                   info))
                if 'nxs' in info['mesh_columns']:
                    packetlist.append(_make_packet('nxs',
                                                   nxs,
                                                   time,
                                                   info))
                if 'nys' in info['mesh_columns']:
                    packetlist.append(_make_packet('nys',
                                                   nys,
                                                   time,
                                                   info))
                if 'nzs' in info['mesh_columns']:
                    packetlist.append(_make_packet('nzs',
                                                   nzs,
                                                   time,
                                                   info))

                # if 'vxs' in info['mesh_columns']:
                    # this_syn.set_value(time=time, qualifier='vxs', value=vxs)
                # if 'vys' in info['mesh_columns']:
                    # this_syn.set_value(time=time, qualifier='vys', value=vys)
                # if 'vzs' in info['mesh_columns']:
                    # this_syn.set_value(time=time, qualifier='vzs', value=vzs)

                if 'rs' in info['mesh_columns']:
                    # NOTE: here we can assume there is only one sma@orbit since
                    # legacy only supports binary systems
                    packetlist.append(_make_packet('rs',
                                                   mqtf(mesh['rad{}'.format(cind)])*sma*u.solRad,
                                                   time,
                                                   info))

                # if 'cosbetas' in info['mesh_columns']:
                    # this_syn.set_value(time=time, qualifier='cosbetas', value=mesh['csbt{}'.format(cind)])

                # if 'areas' in info['mesh_columns']:
                    # TODO: compute and store areas from glump

                if 'loggs' in info['mesh_columns']:
                    packetlist.append(_make_packet('loggs',
                                                   mqtf(mesh['glog{}'.format(cind)]),
                                                   time,
                                                   info))


                if 'teffs' in info['mesh_columns']:
                    packetlist.append(_make_packet('teffs',
                                                   mqtf(mesh['tloc{}'.format(cind)])*u.K,
                                                   time,
                                                   info))

            # now we'll loop over the passband-datasets with requested columns
            for mesh_kind, mesh_dataset in zip(info['mesh_kinds'], info['mesh_datasets']):
                # print "*** legacy mesh pb", info['dataset'], mesh_dataset, mesh_kind
                if mesh_kind == 'lc':
                    lcind = lcinds[mesh_dataset]
                    for time in info['times']:
                        flux, mesh = phb1.lc((time,), lcind, True)

                        if 'abs_normal_intensities@{}'.format(mesh_dataset) in info['mesh_columns']:
                            packetlist.append(_make_packet('abs_normal_intensities',
                                                           mqtf(mesh['Inorm{}'.format(cind)])*u.erg*u.s**-1*u.cm**-3,
                                                           time,
                                                           info,
                                                           dataset=mesh_dataset))

                else:
                    # TODO: once phoebeBackend supports exporting meshes for RVs,
                    # add an elif (or possibly a separate if since the intensities
                    # will hopefully still be supported for the rv)
                    logger.warning("legacy cannot export mesh columns for dataset with kind='{}'".format(mesh_kind))

        else:
            raise NotImplementedError("dataset '{}' with kind '{}' not supported by legacy backend".format(info['dataset'], info['kind']))


        return packetlist


class PhotodynamBackend(BaseBackendByDataset):
    """
    See <phoebe.parameters.compute.photodynam>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_compute>
    * <phoebe.frontend.bundle.Bundle.run_compute>
    """
    def run_checks(self, b, compute, times=[], **kwargs):
        # check whether photodynam is installed
        out = commands.getoutput('photodynam')
        if 'not found' in out:
            raise ImportError('photodynam executable not found.  Install manually and try again.')


    def _worker_setup(self, b, compute, infolist, **kwargs):
        """
        """
        logger.debug("rank:{}/{} PhotodynamBackend._worker_setup".format(mpi.myrank, mpi.nprocs))

        computeparams = b.get_compute(compute, force_ps=True)

        hier = b.get_hierarchy()

        starrefs  = hier.get_stars()
        orbitrefs = hier.get_orbits()

        step_size = computeparams.get_value(qualifier='stepsize', **kwargs)
        orbit_error = computeparams.get_value(qualifier='orbiterror', **kwargs)
        time0 = b.get_value(qualifier='t0', context='system', unit=u.d, **kwargs)


        return dict(compute=compute,
                    starrefs=starrefs,
                    orbitrefs=orbitrefs,
                    step_size=step_size,
                    orbit_error=orbit_error,
                    time0=time0)

    def _run_single_dataset(self, b, info, **kwargs):
        """
        """
        logger.debug("rank:{}/{} PhotodynamBackend._run_single_dataset(info['dataset']={} info['component']={} info.keys={}, **kwargs.keys={})".format(mpi.myrank, mpi.nprocs, info['dataset'], info['component'], info.keys(), kwargs.keys()))


        compute = kwargs.get('compute')
        starrefs = kwargs.get('starrefs')
        orbitrefs = kwargs.get('orbitrefs')
        step_size = kwargs.get('step_size')
        orbit_error = kwargs.get('orbit_error')
        time0 = kwargs.get('time0')

        # write the input file
        # TODO: need to use TemporaryFiles to be MPI safe
        fi = open('_tmp_pd_inp', 'w')
        fi.write('{} {}\n'.format(len(starrefs), time0))
        fi.write('{} {}\n'.format(step_size, orbit_error))
        fi.write('\n')
        fi.write(' '.join([str(b.get_value(qualifier='mass', component=star,
                context='component', unit=u.solMass) * c.G.to('AU3 / (Msun d2)').value)
                for star in starrefs])+'\n') # GM

        fi.write(' '.join([str(b.get_value(qualifier='requiv', component=star,
                context='component', unit=u.AU))
                for star in starrefs])+'\n')

        if info['kind'] == 'lc':
            # TODO: this will make two meshing calls, let's create and extract from the dictionary instead, or use set_value=True
            pblums = [b.get_value(qualifier='pblum', dataset=info['dataset'], component=starref, unit=u.W, check_visible=False) for starref in starrefs]

            u1s, u2s = [], []
            for star in starrefs:
                if b.get_value(qualifier='ld_func', component=star, dataset=info['dataset'], context='dataset') == 'quadratic':
                    ld_coeffs = b.get_value(qualifier='ld_coeffs', component=star, dataset=info['dataset'], context='dataset', check_visible=False)
                else:
                    # TODO: can we still interpolate for quadratic manually using b.compute_ld_coeffs?
                    ld_coeffs = (0,0)
                    logger.warning("ld_func for {} {} must be 'quadratic' for the photodynam backend, but is not: defaulting to quadratic with coeffs of {}".format(star, info['dataset'], ld_coeffs))

                u1s.append(str(ld_coeffs[0]))
                u2s.append(str(ld_coeffs[1]))

        else:
            # we only care about the dynamics, so let's just pass dummy values
            pblums = [1 for star in starrefs]
            u1s = ['0' for star in starrefs]
            u2s = ['0' for star in starrefs]

        if -1 in pblums:
            raise ValueError('pblums must be set in order to run photodynam')

        fi.write(' '.join([str(pbl / (4*np.pi)) for pbl in pblums])+'\n')

        fi.write(' '.join(u1s)+'\n')
        fi.write(' '.join(u2s)+'\n')

        fi.write('\n')

        for orbitref in orbitrefs:
            a = b.get_value(qualifier='sma', component=orbitref,
                context='component', unit=u.AU)
            e = b.get_value(qualifier='ecc', component=orbitref,
                context='component')
            i = b.get_value(qualifier='incl', component=orbitref,
                context='component', unit=u.rad)
            o = b.get_value(qualifier='per0', component=orbitref,
                context='component', unit=u.rad)
            l = b.get_value(qualifier='long_an', component=orbitref,
                context='component', unit=u.rad)

            # t0 = b.get_value(qualifier='t0_perpass', component=orbitref,
                # context='component', unit=u.d)
            # period = b.get_value(qualifier='period', component=orbitref,
                # context='component', unit=u.d)

            # om = 2 * np.pi * (time0 - t0) / period
            om = b.get_value(qualifier='mean_anom', component=orbitref,
                             context='component', unit=u.rad)

            fi.write('{} {} {} {} {} {}\n'.format(a, e, i, o, l, om))
        fi.close()

        # write the report file
        fr = open('_tmp_pd_rep', 'w')
        # t times
        # F fluxes
        # x light-time corrected positions
        # v light-time corrected velocities
        fr.write('t F x v \n')   # TODO: don't always get all?

        ds = b.get_dataset(dataset=info['dataset'])
        times = ds.get_value(qualifier='compute_times', unit=u.d)
        if not len(times) and 'times' in ds.qualifiers:
            times = b.get_value(qualifier='times', component=info['component'], unit=u.d)

        for t in times:
            fr.write('{}\n'.format(t))
        fr.close()

        # run photodynam
        cmd = 'photodynam _tmp_pd_inp _tmp_pd_rep > _tmp_pd_out'
        logger.info("running photodynam backend: '{}'".format(cmd))
        out = commands.getoutput(cmd)
        stuff = np.loadtxt('_tmp_pd_out', unpack=True)

        # parse output to fill packets
        packetlist = []

        nbodies = len(starrefs)
        if info['kind']=='lc':
            packetlist.append(_make_packet('times',
                                           stuff[0]*u.d,
                                           None,
                                           info))

            packetlist.append(_make_packet('fluxes',
                                           stuff[1] +b.get_value(qualifier='pbflux', dataset=info['dataset'], unit=u.W/u.m**2, check_visible=False) - 1,
                                           None,
                                           info))

        elif info['kind']=='orb':
            cind = starrefs.index(info['component'])

            packetlist.append(_make_packet('times',
                                           stuff[0]*u.d,
                                           None,
                                           info))

            packetlist.append(_make_packet('us',
                                           -1*stuff[2+(cind*3)] * u.AU,
                                           None,
                                           info))

            packetlist.append(_make_packet('vs',
                                           -1*stuff[3+(cind*3)] * u.AU,
                                           None,
                                           info))

            packetlist.append(_make_packet('ws',
                                           stuff[4+(cind*3)] * u.AU,
                                           None,
                                           info))

            packetlist.append(_make_packet('vus',
                                           -1*stuff[3*nbodies+2+(cind*3)] * u.AU/u.d,
                                           None,
                                           info))

            packetlist.append(_make_packet('vvs',
                                           -1*stuff[3*nbodies+3+(cind*3)] * u.AU/u.d,
                                           None,
                                           info))

            packetlist.append(_make_packet('vws',
                                           stuff[3*nbodies+4+(cind*3)] * u.AU/u.d,
                                           None,
                                           info))


        elif info['kind']=='rv':
            cind = starrefs.index(info['component'])

            rvs = -stuff[3*nbodies+4+(cind*3)]

            packetlist.append(_make_packet('times',
                                           stuff[0]*u.d,
                                           None,
                                           info))

            packetlist.append(_make_packet('rvs',
                                           rvs * u.AU/u.d,
                                           None,
                                           info))

        else:
            raise NotImplementedError("kind {} not yet supported by this backend".format(info['kind']))

        return packetlist

# According to jktebop's readme.txt:
# The possible entries for the type of limb darkening law are 'lin' (for linear)
# 'log' (logarithmic), 'sqrt' (square-root), 'quad' (quadratic) or 'cub' (cubic)
_jktebop_ld_func = {'linear': 'lin',
                    'logarithmic': 'log',
                    'square_root': 'sqrt',
                    'quadratic': 'quad'}

class JktebopBackend(BaseBackendByDataset):
    """
    See <phoebe.parameters.compute.jktebop>.

    This run method in this class will almost always be called through the bundle, using
        * :meth:`phoebe.frontend.bundle.Bundle.add_compute`
        * :meth:`phoebe.frontend.bundle.Bundle.run_compute`
    """
    def run_checks(self, b, compute, times=[], **kwargs):
        # check whether jktebop is installed
        out = commands.getoutput('jktebop')
        if 'not found' in out:
            raise ImportError('jktebop executable not found.  Install manually and try again.')
        version = out.split('JKTEBOP  ')[1].split(' ')[0]
        try:
            version_int = int(float(version[1:]))
        except Exception as e:
            print(e)
            raise ImportError("could not parse jktebop version.  PHOEBE is tested for v40, but may work on newer versions.")
        else:
            min_version = 40
            if version_int < min_version:
                raise ImportError("PHOEBE requires jktebop v{}+, found v{}".format(min_version, version_int))

        hier = b.get_hierarchy()

        starrefs  = hier.get_stars()
        orbitrefs = hier.get_orbits()

        if len(starrefs) != 2 or len(orbitrefs) != 1:
            raise ValueError("jktebop backend only accepts binary systems")

        # handled in bundle checks
        # for dataset in b.filter(compute=compute, context='compute', qualifier='enabled', value=True).datasets:
        #     for comp in starrefs:
        #         if b.get_value(qualifier='ld_func', component=comp, dataset=datset, context='dataset') == 'interp':
        #             raise ValueError("jktebop backend does not accept ld_func='interp'")


    def _worker_setup(self, b, compute, infolist, **kwargs):
        """
        """
        logger.debug("rank:{}/{} JktebopBackend._worker_setup".format(mpi.myrank, mpi.nprocs))

        computeparams = b.get_compute(compute, force_ps=True)

        hier = b.get_hierarchy()

        starrefs  = hier.get_stars()
        orbitrefs = hier.get_orbits()

        orbitref = orbitrefs[0]

        ringsize = computeparams.get_value(qualifier='ringsize', unit=u.deg, ringsize=kwargs.get('ringsize', None), **_skip_filter_checks)
        distortion_method = computeparams.get_value(qualifier='distortion_method', distortion_method=kwargs.get('distortion_method', None), **_skip_filter_checks)
        irrad_method = computeparams.get_value(qualifier='irrad_method', irrad_method=kwargs.get('irrad_method', None), **_skip_filter_checks)

        rA = b.get_value(qualifier='requiv', component=starrefs[0], context='component', unit=u.solRad, **_skip_filter_checks)
        rB = b.get_value(qualifier='requiv', component=starrefs[1], context='component', unit=u.solRad, **_skip_filter_checks)
        sma = b.get_value(qualifier='sma', component=orbitref, context='component', unit=u.solRad, **_skip_filter_checks)
        sma_A = b.get_value(qualifier='sma', component=starrefs[0], context='component', unit=u.solRad, **_skip_filter_checks)
        sma_B = b.get_value(qualifier='sma', component=starrefs[1], context='component', unit=u.solRad, **_skip_filter_checks)
        incl = b.get_value(qualifier='incl', component=orbitref, context='component', unit=u.deg, **_skip_filter_checks)
        q = b.get_value(qualifier='q', component=orbitref, context='component', **_skip_filter_checks)
        ecc = b.get_value(qualifier='ecc', component=orbitref, context='component', **_skip_filter_checks)
        ecosw = b.get_value(qualifier='ecosw', component=orbitref, context='component', **_skip_filter_checks)
        esinw = b.get_value(qualifier='esinw', component=orbitref, context='component', **_skip_filter_checks)

        gravbA = b.get_value(qualifier='gravb_bol', component=starrefs[0], context='component', **_skip_filter_checks)
        gravbB = b.get_value(qualifier='gravb_bol', component=starrefs[1], context='component', **_skip_filter_checks)

        period = b.get_value(qualifier='period', component=orbitref, context='component', unit=u.d, **_skip_filter_checks)
        t0_supconj = b.get_value(qualifier='t0_supconj', component=orbitref, context='component', unit=u.d, **_skip_filter_checks)

        return dict(compute=compute,
                    starrefs=starrefs,
                    oritref=orbitref,
                    ringsize=ringsize,
                    distortion_method=distortion_method,
                    irrad_method=irrad_method,
                    rA=rA, rB=rB,
                    sma=sma, incl=incl, q=q,
                    ecosw=ecosw, esinw=esinw,
                    gravbA=gravbA, gravbB=gravbB,
                    period=period, t0_supconj=t0_supconj,
                    pblums=kwargs.get('pblums'),
                    sma_A=sma_A,
                    sma_B=sma_B,
                    ecc=ecc
                    )

    def _run_single_dataset(self, b, info, **kwargs):
        """
        """
        logger.debug("rank:{}/{} JktebopBackend._run_single_dataset(info['dataset']={} info['component']={} info.keys={}, **kwargs.keys={})".format(mpi.myrank, mpi.nprocs, info['dataset'], info['component'], info.keys(), kwargs.keys()))

        compute = kwargs.get('compute')
        starrefs = kwargs.get('starrefs')
        orbitref = kwargs.get('orbitref')
        ringsize = kwargs.get('ringsize')
        distortion_method = kwargs.get('distortion_method')
        irrad_method = kwargs.get('irrad_method')
        rA = kwargs.get('rA')
        rB = kwargs.get('rB')
        sma = kwargs.get('sma')
        sma_A = kwargs.get('sma_A')
        sma_B = kwargs.get('sma_B')
        incl = kwargs.get('incl')
        q = kwargs.get('q')
        if distortion_method == 'sphere':
            q *= -1
        ecosw = kwargs.get('ecosw')
        esinw = kwargs.get('esinw')
        ecc = kwargs.get('ecc')
        gravbA = kwargs.get('gravbA')
        gravbB = kwargs.get('gravbB')
        period = kwargs.get('period')
        t0_supconj = kwargs.get('t0_supconj')

        # get dataset-dependent things that we need
        ldfuncA = b.get_value(qualifier='ld_func', component=starrefs[0], dataset=info['dataset'], context='dataset', **_skip_filter_checks)
        ldfuncB = b.get_value(qualifier='ld_func', component=starrefs[1], dataset=info['dataset'], context='dataset', **_skip_filter_checks)

        # use check_visible=False to access the ld_coeffs from
        # compute_ld_coeffs(set_value=True) done in _worker_setup
        ldcoeffsA = b.get_value(qualifier='ld_coeffs', component=starrefs[0], dataset=info['dataset'], context='dataset', **_skip_filter_checks)
        ldcoeffsB = b.get_value(qualifier='ld_coeffs', component=starrefs[1], dataset=info['dataset'], context='dataset', **_skip_filter_checks)

        if irrad_method == "biaxial-spheroid":
            albA = b.get_value(qualifier='irrad_frac_refl_bol', component=starrefs[0], context='component', **_skip_filter_checks)
            albB = b.get_value(qualifier='irrad_frac_refl_bol', component=starrefs[1], context='component', **_skip_filter_checks)
        elif irrad_method == 'none':
            albA = 0.0
            albB = 0.0
        else:
            raise NotImplementedError("irrad_method '{}' not supported".format(irrad_method))

        pblums = kwargs.get('pblums').get(info['dataset'])
        sbratio = (pblums.get(starrefs[1])/b.get_value(qualifier='requiv', component=starrefs[1], context='component', unit=u.solRad)**2)/(pblums.get(starrefs[0])/b.get_value(qualifier='requiv', component=starrefs[0], context='component', unit=u.solRad)**2)

        # let's make sure we'll be able to make the translation later
        if ldfuncA not in _jktebop_ld_func.keys() or ldfuncB not in _jktebop_ld_func.keys():
            # NOTE: this is now handle in b.run_checks, so should never happen
            # TODO: provide a more useful error statement
            raise ValueError("jktebop only accepts the following options for ld_func: {}".format(ldfuncs.keys()))

        # create the input file for jktebop
        # uncomment this block, comment out the following block and the os.remove at the end
        # for testing
        # tmpfilenamein = 'jktebop.in'
        # tmpfilenamelcin = 'jktebop.lc.in'
        # tmpfilenamelcout = 'jktebop.lc.out'
        # tmpfilenamervin = 'jktebop.rv.in'
        # tmpfilenamervout = 'jktebop.rv.out'
        # tmpfilenameparamout = 'jktebop.param.out'
        # tmpfilenamemodelout = 'jktebop.model.out'
        # tmpfilenameout = 'jktebop.out'
        # try:
        #     os.remove(tmpfilenamein)
        # except: pass
        # try:
        #     os.remove(tmpfilenamelcin)
        # except: pass
        # try:
        #     os.remove(tmpfilenamelcout)
        # except: pass
        # try:
        #     os.remove(tmpfilenamervin)
        # except: pass
        # try:
        #     os.remove(tmpfilenamervout)
        # except: pass
        # try:
        #     os.remove(tmpfilenameparamout)
        # except: pass
        # try:
        #     os.remove(tmpfilenamemodelout)
        # except: pass
        # try:
        #     os.remove(tmpfilenameout)
        # except: pass

        tmpfilenamein = next(tempfile._get_candidate_names())
        tmpfilenamelcin = next(tempfile._get_candidate_names())
        tmpfilenamelcout = next(tempfile._get_candidate_names())
        if info['kind'] == 'rv':
            tmpfilenamervin = next(tempfile._get_candidate_names())
            tmpfilenamervout = next(tempfile._get_candidate_names())
        tmpfilenameparamout = next(tempfile._get_candidate_names())
        tmpfilenamemodelout = next(tempfile._get_candidate_names())
        tmpfilenameout = next(tempfile._get_candidate_names())


        fi = open(tmpfilenamein, 'w')

        # Task 3	This inputs a parameter file (containing estimated parameter
        # values) and an observed light curve. It fits the light curve using
        # Levenberg-Marquardt minimisation and produces an output parameter file,
        # a file of residuals of the observations, and file containing the best
        # fit to the light curve (as in Task 2). The parameter values have formal
        # errors (from the covariance matrix found by the minimisation algorithm)
        # but these are not overall uncertainties. You will need to run other
        # tasks to get reliable parameter uncertainties.
        fi.write('{:5} {:11} Task to do (from 1 to 9)   Integ. ring size (deg)\n'.format(3, ringsize))
        fi.write('{:5} {:11} Sum of the radii           Ratio of the radii\n'.format((rA+rB)/sma, rB/rA))
        fi.write('{:5} {:11} Orbital inclination (deg)  Mass ratio of the system\n'.format(incl, q))

        # we'll provide ecosw and esinw instead of ecc and long_an
        # jktebop's readme.txt states that so long as ecc is < 10,
        # it will be intrepreted as ecosw and esinw (otherwise would
        # need to be ecc+10 and long_an (deg)
        fi.write('{:5} {:11} Orbital eccentricity       Periastron longitude deg\n'.format(ecosw, esinw))


        fi.write('{:5} {:11} Gravity darkening (starA)  Grav darkening (starB)\n'.format(gravbA, gravbB))
        fi.write('{:5} {:11} Surface brightness ratio   Amount of third light\n'.format(sbratio, 0.0))


        fi.write('{:5} {:11} LD law type for star A     LD law type for star B\n'.format(_jktebop_ld_func[ldfuncA], _jktebop_ld_func[ldfuncB]))
        fi.write('{:5} {:11} LD star A (linear coeff)   LD star B (linear coeff)\n'.format(ldcoeffsA[0], ldcoeffsB[0]))
        fi.write('{:5} {:11} LD star A (nonlin coeff)   LD star B (nonlin coeff)\n'.format(ldcoeffsA[1] if len(ldcoeffsA)==2 else 0.0, ldcoeffsB[1] if len(ldcoeffsB)==2 else 0.0))

        fi.write('{:5} {:11} Reflection effect star A   Reflection effect star B\n'.format(albA, albB))
        fi.write('{:5} {:11} Phase of primary eclipse   Light scale factor (mag)\n'.format(0.0, 1.0))
        fi.write('{:13}      Orbital period of eclipsing binary system (days)\n'.format(period))
        fi.write('{:13}      Reference time of primary minimum (HJD)\n'.format(t0_supconj))

        # All fitting will be done with PHOEBE wrappers, so we need to set
        # all jktebop options for adjust to False (0)
        fi.write(' {:d}  {:d}             Adjust RADII SUM or RADII RATIO (0, 1, 2, 3)\n'.format(0, 0))
        fi.write(' {:d}  {:d}             Adjust INCLINATION or MASSRATIO (0, 1, 2, 3)\n'.format(0, 0))
        fi.write(' {:d}  {:d}             Adjust ECCENTRICITY or OMEGA (0, 1, 2, 3)\n'.format(0, 0))
        fi.write(' {:d}  {:d}             Adjust GRAVDARK1 or GRAVDARK2 (0, 1, 2, 3)\n'.format(0, 0))
        fi.write(' {:d}  {:d}             Adjust SURFACEBRIGH2 or THIRDLIGHT (0, 1, 2, 3)\n'.format(0, 0))
        fi.write(' {:d}  {:d}             Adjust LD-lin1 or LD-lin2 (0, 1, 2, 3)\n'.format(0, 0))
        fi.write(' {:d}  {:d}             Adjust LD-nonlin1 or LD-nonlin2 (0, 1, 2, 3)\n'.format(0, 0))
        fi.write(' {:d}  {:d}             Adjust REFLECTION COEFFS 1 and 2 (-1, 0, 1, 2, 3)\n'.format(0, 0))
        fi.write(' {:d}  {:d}             Adjust PHASESHIFT or SCALE FACTOR (0, 1, 2, 3)\n'.format(0, 0))
        fi.write(' {:d}  {:d}             Adjust PERIOD or TZERO (min light) (0, 1, 2, 3)\n'.format(0, 0))

        fi.write('{}  Name of file containing light curve\n'.format(tmpfilenamelcin))
        fi.write('{}  Name of output parameter file\n'.format(tmpfilenameparamout))
        fi.write('{}  Name of output light curve file\n'.format(tmpfilenamelcout))
        fi.write('{}  Name of output model light curve fit file\n'.format(tmpfilenamemodelout))

        # According to jktebop's readme.txt:
        # FITTING FOR RADIAL VELOCITIES:    the observed RVs should be in separate files
        # for the two stars and the data should be in the same format as the light curve
        # data. Then add a line below the main input parameters for each rv file:
        #   RV1  [infile]  [outfile]  [K]  [Vsys]  [vary(K)]  [vary(Vsys)]
        #   RV2  [infile]  [outfile]  [K]  [Vsys]  [vary(K)]  [vary(Vsys)]
        # where RV1 is for primary star velocities, RV2 is for secondary star velocities
        # [infile] is the input data file, [outfile] is the output data file, [K] is the
        # velocity amplitude of the star (km/s), [Vsys] is its systemic velocity (km/s),
        # and [vary(K)] and [vary(Vsys)] are 0 to fix and 1 to fit for these quantities.
        # The mass ratio parameter is not used for the RVs, only for the light curve.
        # If you want to fix the systemic velocity for star B to that for star A, simply
        # set vary(Vsys) for star B to be equal to -1
        #~ fi.write('rv1 llaqr-rv1.dat llaqr-rv1.out 55.0 -10.0 0 0\n')
        #~ fi.write('rv2 llaqr-rv2.dat llaqr-rv2.out 55.0 -10.0 0 0\n')
        if info['kind'] == 'rv':
            # NOTE: we disable systemic velocity as it will be added in bundle.run_compute
            sma_ = sma_A if info['component'] == starrefs[0] else sma_B
            K = np.pi * 2*(sma_*u.solRad).to(u.km).value * np.sin((incl*u.deg).to(u.rad).value) / ((period*u.d).to(u.s).value * np.sqrt(1-ecc**2))
            fi.write('{} {} {} {} {} 0 0\n'.format('rv1' if info['component'] == starrefs[0] else 'rv2', tmpfilenamervin, tmpfilenamervout, K, 0.0))


        # According to jktebop's readme.txt:
        # NUMERICAL INTEGRATION:  long exposure times can be split up into NUMINT points
        # occupying a total time interval of NINTERVAL (seconds) by including this line:
        #   NUMI  [numint]  [ninterval]

        # TODO: allow exposure times?


        fi.close()

        if info['kind'] == 'lc':
            np.savetxt(tmpfilenamelcin, np.asarray([info['times'], np.ones_like(info['times'])]).T, fmt='%f')
        elif info['kind'] == 'rv':
            # we don't technically need them, but otherwise jktebop complains about not enough data to "fit" for task 3, even though we're holding everything fixed
            np.savetxt(tmpfilenamelcin, np.asarray([info['times'], np.ones_like(info['times'])]).T, fmt='%f')
            np.savetxt(tmpfilenamervin, np.asarray([info['times'], np.ones_like(info['times'])]).T, fmt='%f')
        else:
            raise NotImplementedError()

        # run jktebop
        out = commands.getoutput("jktebop {} > {}".format(tmpfilenamein, tmpfilenameout))



        # fill packets
        packetlist = []

        if info['kind'] == 'lc':
            # parse output
            times, _, _, _, mags, _ = np.loadtxt(tmpfilenamelcout, unpack=True)

            logger.warning("converting from mags from jktebop to flux")
            fluxes = 10**((0.0-mags)/2.5)
            fluxes /= np.max(fluxes)

            packetlist.append(_make_packet('times',
                                           info['times']*u.d,
                                           None,
                                           info))

            packetlist.append(_make_packet('fluxes',
                                           fluxes,
                                           None,
                                           info))

        elif info['kind'] == 'rv':
            times, _, _, _, rvs, _ = np.loadtxt(tmpfilenamervout, unpack=True)

            packetlist.append(_make_packet('times',
                                           info['times']*u.d,
                                           None,
                                           info))

            packetlist.append(_make_packet('rvs',
                                           rvs*u.km/u.s,
                                           None,
                                           info))

        else:
            raise NotImplementedError()


        os.remove(tmpfilenamein)
        os.remove(tmpfilenamelcin)
        os.remove(tmpfilenamelcout)
        if info['kind'] == 'rv':
            os.remove(tmpfilenamervin)
            os.remove(tmpfilenamervout)
        os.remove(tmpfilenameparamout)
        os.remove(tmpfilenamemodelout)
        os.remove(tmpfilenameout)

        return packetlist


_ellc_ld_func = {'linear': 'lin', 'quadratic': 'quad', 'logarithmic': 'log', 'square_root': 'sqrt', 'power': 'power-2'}

 # The "power-2" limb darkening law is taken from  Morello et al.,
 # 2017AJ....154..111M:
 #  I_lambda(mu)/I_lambda(1) = 1 - ldc_1[0] * (1 - mu**ldc_1[1])
 # For the option "mugrid", ldc_1 must be an array of specific intensity
 # values as a function of mu=cos(theta), where theta is the angle between the
 # line of sight and the normal to a given point of the stellar surface. The
 # value of mu corresponding to element i of ldc_1 is i/(len(ldc_1)-1),  where
 # i=0,1, ... len(ldc_1)-1.
 # Default is None

class EllcBackend(BaseBackendByDataset):
    """
    See <phoebe.parameters.compute.ellc>.

    The run method in this class will almost always be called through the bundle, using
    * <phoebe.frontend.bundle.Bundle.add_compute>
    * <phoebe.frontend.bundle.Bundle.run_compute>
    """
    def run_checks(self, b, compute, times=[], **kwargs):
        # check whether ellc is installed
        if not _use_ellc:
            raise ImportError("could not import ellc.  Install (pip install ellc) and restart phoebe")

        if not (hasattr(ellc, 'lc') and hasattr(ellc, 'rv')):
            try:
                from ellc import lc as _ellc_lc
                from ellc import rv as _ellc_rv
            except Exception as e:
                raise ImportError("ellc failed to load required lc and/or rv functions with error: {}".format(e))

        hier = b.get_hierarchy()

        starrefs  = hier.get_stars()
        orbitrefs = hier.get_orbits()

        if len(starrefs) != 2 or len(orbitrefs) != 1:
            raise ValueError("ellc backend only accepts binary systems")

    def _worker_setup(self, b, compute, infolist, **kwargs):
        """
        """
        logger.debug("rank:{}/{} EllcBackend._worker_setup".format(mpi.myrank, mpi.nprocs))

        computeparams = b.get_compute(compute, force_ps=True, **_skip_filter_checks)
        t0_system = b.get_value(qualifier='t0', context='system', unit=u.d, **_skip_filter_checks)

        hier = b.get_hierarchy()

        starrefs  = hier.get_stars()
        orbitrefs = hier.get_orbits()

        orbitref = orbitrefs[0]

        shape_1 = computeparams.get_value(qualifier='distortion_method', component=starrefs[0], distortion_method=kwargs.get('distortion_method', None), **_skip_filter_checks)
        shape_2 = computeparams.get_value(qualifier='distortion_method', component=starrefs[1], distortion_method=kwargs.get('distortion_method', None), **_skip_filter_checks)

        hf_1 = computeparams.get_value(qualifier='hf', component=starrefs[0], hf=kwargs.get('hf', None), **_skip_filter_checks)
        hf_2 = computeparams.get_value(qualifier='hf', component=starrefs[1], hf=kwargs.get('hf', None), **_skip_filter_checks)

        grid_1 = computeparams.get_value(qualifier='grid', component=starrefs[0], grid=kwargs.get('grid', None), **_skip_filter_checks)
        grid_2 = computeparams.get_value(qualifier='grid', component=starrefs[1], grid=kwargs.get('grid', None), **_skip_filter_checks)

        exact_grav = computeparams.get_value(qualifier='exact_grav', exact_grav=kwargs.get('grav', None), **_skip_filter_checks)

        comp_ps = b.filter(context='component', **_skip_filter_checks)

        a = comp_ps.get_value(qualifier='sma', component=orbitref, unit=u.solRad, **_skip_filter_checks)
        radius_1 = comp_ps.get_value(qualifier='requiv', component=starrefs[0], unit=u.solRad, **_skip_filter_checks) / a
        radius_2 = comp_ps.get_value(qualifier='requiv', component=starrefs[1], unit=u.solRad, **_skip_filter_checks) / a

        period_anom = comp_ps.get_value(qualifier='period_anom', component=orbitref, unit=u.d, **_skip_filter_checks)
        q = comp_ps.get_value(qualifier='q', component=orbitref, **_skip_filter_checks)

        t_zero = comp_ps.get_value(qualifier='t0_supconj', component=orbitref, unit=u.d, **_skip_filter_checks)

        incl = comp_ps.get_value(qualifier='incl', component=orbitref, unit=u.deg, **_skip_filter_checks)
        didt = 0.0
        # didt = b.get_value(qualifier='dincldt', component=orbitref, context='component', unit=u.deg/u.d) * period
        # incl += didt * (t_zero - t0_system)

        ecc = comp_ps.get_value(qualifier='ecc', component=orbitref, **_skip_filter_checks)
        w = comp_ps.get_value(qualifier='per0', component=orbitref, unit=u.rad, **_skip_filter_checks)

        # need to correct w (per0) to be at t_zero (t0_supconj) instead of t0@system as defined in PHOEBE
        logger.debug("per0(t0@system): {}".format(w))
        domdt_rad = comp_ps.get_value(qualifier='dperdt', component=orbitref, unit=u.rad/u.d, **_skip_filter_checks)
        w += domdt_rad * (t_zero - t0_system)
        logger.debug("per0(t0_supconj): {}".format(w))

        # NOTE: domdt is listed in ellc as deg/anomalistic period, but as deg/sidereal period in the fortran source (which agrees with comparisons)
        # NOTE: this does NOT need to be iterative, because the original dperdt is in deg/d and independent of period
        logger.debug("dperdt (rad/d): {}".format(domdt_rad))
        period_sid = comp_ps.get_value(qualifier='period', component=orbitref, unit=u.d, **_skip_filter_checks)
        # NOTE: period_sidereal does not need to be corrected from t0@system -> t0_supconj because ellc does not support dpdt
        logger.debug("period_sidereal(t0@system,t0_ref,dpdt=0): {}".format(period_sid))
        domdt = comp_ps.get_value(qualifier='dperdt', component=orbitref, unit=u.deg/u.d, **_skip_filter_checks) * period_sid
        logger.debug("dperdt (deg/d * period_sidereal): {}".format(domdt))

        f_c = np.sqrt(ecc) * np.cos(w)
        f_s = np.sqrt(ecc) * np.sin(w)

        gdc_1 = comp_ps.get_value(qualifier='gravb_bol', component=starrefs[0], **_skip_filter_checks)
        gdc_2 = comp_ps.get_value(qualifier='gravb_bol', component=starrefs[1], **_skip_filter_checks)

        rotfac_1 = comp_ps.get_value(qualifier='syncpar', component=starrefs[0], **_skip_filter_checks)
        rotfac_2 = comp_ps.get_value(qualifier='syncpar', component=starrefs[1], **_skip_filter_checks)

        enabled_features = b.filter(qualifier='enabled', compute=compute, value=True, **_skip_filter_checks).features
        spots = b.filter(feature=enabled_features, kind='spot', **_skip_filter_checks).features
        if len(spots):
            # from ELLC docs:
            # spots_1 : (4, n_spots_1) array_like
            # Parameters of the spots on star 1. For each spot the parameters, in order,
            # are longitude, latitude, size and brightness factor. All three angles are
            # in degrees.

            # TODO: do we need to correct these from t0_system to t_zero (if rotfac!=1)?

            spots_1 = []
            spots_2 = []
            for spot in spots:
                spot_ps = b.get_feature(feature=spot, **_skip_filter_checks)
                spot_comp = spot_ps.component
                spot_args = [-1*spot_ps.get_value(qualifier='long', unit=u.deg, **_skip_filter_checks),
                             spot_ps.get_value(qualifier='colat', unit=u.deg, **_skip_filter_checks),
                             spot_ps.get_value(qualifier='radius', unit=u.deg, **_skip_filter_checks),
                             spot_ps.get_value(qualifier='relteff', **_skip_filter_checks)**4]
                if spot_comp == starrefs[0]:
                    spots_1.append(spot_args)
                elif spot_comp == starrefs[1]:
                    spots_2.append(spot_args)

            if not len(spots_1):
                spots_1 = None
            else:
                spots_1 = np.asarray(spots_1).T
            if not len(spots_2):
                spots_2 = None
            else:
                spots_2 = np.asarray(spots_2).T

        else:
            spots_1 = None
            spots_2 = None

        # The simplified reflection model is approximately equivalent to Lambert
        #     law scattering with the coefficients heat_1 and heat_2  being equal to
        #     A_g/2, where A_g is the geometric albedo.
        irrad_method = computeparams.get_value(qualifier='irrad_method', irrad_method=kwargs.get('irrad_method', None), **_skip_filter_checks)
        if irrad_method == 'lambert':
            heat_1 = b.get_value(qualifier='irrad_frac_refl_bol', component=starrefs[0], context='component', **_skip_filter_checks) / 2.
            heat_2 = b.get_value(qualifier='irrad_frac_refl_bol', component=starrefs[1], context='component', **_skip_filter_checks) / 2.
            # let's save ourselves, and also allow for flux-weighted RVs
            if heat_1 == 0 and heat_2 == 0:
                heat_1 = None
                heat_2 = None
        elif irrad_method == 'none':
            heat_1 = None
            heat_2 = None
        else:
            raise NotImplementedError("irrad_method='{}' not supported".format(irrad_method))

        # lambda_1/2 : {None, float},  optional
         # Sky-projected angle between orbital and rotation axes, star 1/2 [degrees]
         # N.B. lambda_1/2 is only used if shape_1='sphere'
        lambda_1 = b.get_value(qualifier='yaw', component=starrefs[0], context='component', unit=u.deg)
        lambda_2 = b.get_value(qualifier='yaw', component=starrefs[1], context='component', unit=u.deg)

        return dict(compute=compute,
                    starrefs=starrefs,
                    oritref=orbitref,
                    shape_1=shape_1, shape_2=shape_2,
                    grid_1=grid_1, grid_2=grid_2,
                    exact_grav=exact_grav,
                    radius_1=radius_1, radius_2=radius_2,
                    incl=incl,
                    t_zero=t_zero,
                    period_anom=period_anom,
                    q=q,
                    a=a,
                    f_c=f_c, f_s=f_s,
                    didt=didt, domdt=domdt,
                    gdc_1=gdc_1, gdc_2=gdc_2,
                    rotfac_1=rotfac_1, rotfac_2=rotfac_2,
                    heat_1=heat_1, heat_2=heat_2,
                    lambda_1=lambda_1, lambda_2=lambda_2,
                    spots_1=spots_1, spots_2=spots_2,
                    pblums=kwargs.get('pblums', {}))

    def export(self, b, filename, compute, pblums, dataset, times):
        infolist, new_syns = _extract_from_bundle(b, compute=compute,
                                                  dataset=dataset,
                                                  times=times, by_time=False)

        setup_kwargs = self._worker_setup(b, compute, infolist, pblums=pblums)

        if filename is not None:
            f = open(filename, 'w')
            f.write('import ellc\n\n')
            # f.write('from phoebe.dependencies import nparray\n\n')

        def _format_value(v):
            # if isinstance(v, _nparrayArray):
            #     return "nparray.from_dict({})".format(v.to_dict())
            if hasattr(v, 'tolist'):
                return v.tolist()
            if isinstance(v, str):
                return "\'{}\'".format(v)
            return v

        ret_dict = {}
        for info in infolist:

            ellc_kwargs = self._run_single_dataset(b, info, return_dict_only=True, **setup_kwargs)

            ret_dict[info['dataset'] if info['kind'] == 'lc' else "{}@{}".format(info['dataset'], info['component'])] = {'function': info['kind'], 'kwargs': ellc_kwargs}
            if filename is not None:
                f.write("# dataset=\'{}\'\n".format(info['dataset']))

                if info['kind'] == 'lc':
                    f.write("fluxes_{} = ellc.lc({})\n\n".format(info['dataset'],
                                                                 ", ".join(["{}={}".format(k,_format_value(v)) for k,v in ellc_kwargs.items()])))
                else:
                    f.write("rvs_{}_{} = ellc.rvs({})[{}]\n\n".format(info['dataset'], info['component'],
                                                                      ", ".join(["{}={}".format(k,_format_value(v)) for k,v in ellc_kwargs.items()]),
                                                                      b.hierarchy.get_primary_or_secondary(info['component'], return_ind=True)-1))


        if filename is not None:
            f.close()
        return ret_dict


    def _run_single_dataset(self, b, info, **kwargs):
        """
        """
        logger.debug("rank:{}/{} EllcBackend._run_single_dataset(info['dataset']={} info['component']={} info.keys={}, **kwargs.keys={})".format(mpi.myrank, mpi.nprocs, info['dataset'], info['component'], info.keys(), kwargs.keys()))

        compute = kwargs.get('compute')
        starrefs = kwargs.get('starrefs')
        orbitref = kwargs.get('orbitref')

        grid_1 = kwargs.get('grid_1')
        grid_2 = kwargs.get('grid_2')
        shape_1 = kwargs.get('shape_1')
        shape_2 = kwargs.get('shape_2')
        hf_1 = kwargs.get('hf_1')
        hf_2 = kwargs.get('hf_2')

        exact_grav = kwargs.get('exact_grav')

        radius_1 = kwargs.get('radius_1')
        radius_2 = kwargs.get('radius_2')

        incl = kwargs.get('incl')

        t_zero = kwargs.get('t_zero')
        period_anom = kwargs.get('period_anom')
        a = kwargs.get('a')
        q = kwargs.get('q')

        f_c = kwargs.get('f_c')
        f_s = kwargs.get('f_s')

        didt = kwargs.get('didt')
        domdt = kwargs.get('domdt')

        gdc_1 = kwargs.get('gdc_1')
        gdc_2 = kwargs.get('gdc_2')

        rotfac_1 = kwargs.get('rotfac_1')
        rotfac_2 = kwargs.get('rotfac_2')

        heat_1 = kwargs.get('heat_1')
        heat_2 = kwargs.get('heat_2')

        spots_1 = kwargs.get('spots_1')
        spots_2 = kwargs.get('spots_2')

        ds_ps = b.get_dataset(dataset=info['dataset'], **_skip_filter_checks)
        # get dataset-dependent things that we need
        ldfuncA = ds_ps.get_value(qualifier='ld_func', component=starrefs[0], **_skip_filter_checks)
        ldfuncB = ds_ps.get_value(qualifier='ld_func', component=starrefs[1], **_skip_filter_checks)

        # use check_visible=False to access the ld_coeffs from
        # compute_ld_coeffs(set_value=True) done in _worker_setup
        ldcoeffsA = ds_ps.get_value(qualifier='ld_coeffs', component=starrefs[0], **_skip_filter_checks)
        ldcoeffsB = ds_ps.get_value(qualifier='ld_coeffs', component=starrefs[1], **_skip_filter_checks)

        ld_1 = _ellc_ld_func.get(ds_ps.get_value(qualifier='ld_func', component=starrefs[0], **_skip_filter_checks))
        ldc_1 = ds_ps.get_value(qualifier='ld_coeffs', component=starrefs[0], **_skip_filter_checks)
        ld_2 = _ellc_ld_func.get(ds_ps.get_value(qualifier='ld_func', component=starrefs[1], **_skip_filter_checks))
        ldc_2 = ds_ps.get_value(qualifier='ld_coeffs', component=starrefs[1], **_skip_filter_checks)

        pblums = kwargs.get('pblums', {}).get(info['dataset'], {})
        sbratio = (pblums.get(starrefs[1])/b.get_value(qualifier='requiv', component=starrefs[1], context='component', unit=u.solRad, **_skip_filter_checks)**2)/(pblums.get(starrefs[0])/b.get_value(qualifier='requiv', component=starrefs[0], context='component', unit=u.solRad, **_skip_filter_checks)**2)

        if info['kind'] == 'lc':
            # third light handled by run_compute
            light_3 = 0.0

            t_exp = ds_ps.get_value(qualifier='exptime', **_skip_filter_checks)

            # move outside above 'lc' if-statement once exptime is supported for RVs in phoebe
            if b.get_value(qualifier='fti_method', compute=compute, dataset=info['dataset'], context='compute', **_skip_filter_checks) == 'ellc':
                n_int = b.get_value(qualifier='fti_oversample', compute=compute, dataset=info['dataset'], context='compute', **_skip_filter_checks)
            else:
                n_int = 1

            lc_kwargs = dict(t_obs=info['times'],
                             radius_1=radius_1, radius_2=radius_2,
                             sbratio=sbratio,
                             incl=incl,
                             light_3=light_3,
                             t_zero=t_zero,
                             period=period_anom,
                             a=a,
                             q=q,
                             f_c=f_c, f_s=f_s,
                             ld_1=ld_1, ld_2=ld_2,
                             ldc_1=ldc_1, ldc_2=ldc_2,
                             gdc_1=gdc_1, gdc_2=gdc_2,
                             didt=didt, domdt=domdt,
                             rotfac_1=rotfac_1, rotfac_2=rotfac_2,
                             hf_1=hf_1, hf_2=hf_2,
                             bfac_1=None, bfac_2=None,
                             heat_1=heat_1, heat_2=heat_2,
                             lambda_1=None, lambda_2=None,
                             vsini_1=None, vsini_2=None,
                             t_exp=t_exp, n_int=n_int,
                             grid_1=grid_1, grid_2=grid_2,
                             shape_1=shape_1, shape_2=shape_2,
                             spots_1=spots_1, spots_2=spots_2,
                             exact_grav=exact_grav,
                             verbose=1)

            if kwargs.get('return_dict_only', False):
                return lc_kwargs

            logger.info("calling ellc.lc for dataset='{}'".format(info['dataset']))
            logger.debug("ellc.lc(**{})".format(lc_kwargs))

            fluxes = ellc.lc(**lc_kwargs)

            # ellc returns "arbitrary" flux values... these will be rescaled
            # by run compute to pbflux

            # fill packets
            packetlist = []

            packetlist.append(_make_packet('times',
                                           info['times']*u.d,
                                           None,
                                           info))

            packetlist.append(_make_packet('fluxes',
                                           fluxes,
                                           None,
                                           info))

        elif info['kind'] == 'rv':
            rv_method = b.get_value(qualifier='rv_method', compute=compute, dataset=info['dataset'], component=info['component'], context='compute', **_skip_filter_checks)

            flux_weighted = rv_method == 'flux-weighted'
            # if flux_weighted:
            #     # TODO: may just be that we need to estimate and pass vsini
            #     raise NotImplementedError("flux-weighted does not seem to work in ellc")
            if flux_weighted and (heat_1 is not None or heat_2 is not None):
                raise NotImplementedError("ellc cannot compute flux-weighted RVs with irradiation")

            if flux_weighted and period_anom == 1.0: # add VersionCheck once bug fixed (https://github.com/pmaxted/ellc/issues/4)
                logger.warning("ellc does not allow period=1.0 with flux_weighted RVs (see  https://github.com/pmaxted/ellc/issues/4).  Overriding period to 1.0+1e-6 for {}@{}".format(info['component'], info['dataset']))
                period_anom += 1e-6
            # enable once exptime for RVs is supported in PHOEBE
            # t_exp = b.get_value(qualifier='exptime', dataset=info['dataset'], context='dataset')
            t_exp = 0
            n_int = 1

            lambda_1 = kwargs.get('lambda_1')
            lambda_2 = kwargs.get('lambda_2')

            vsini_1 = (2*np.pi*radius_1*(a*u.solRad)*np.sin((incl*u.deg).to(u.rad))/((period_anom*u.d)/rotfac_1)).to(u.km/u.s)
            vsini_2 = (2*np.pi*radius_2*(a*u.solRad)*np.sin((incl*u.deg).to(u.rad))/((period_anom*u.d)/rotfac_2)).to(u.km/u.s)

            rv_kwargs = dict(t_obs=info['times'],
                             radius_1=radius_1, radius_2=radius_2,
                             sbratio=sbratio,
                             incl=incl,
                             t_zero=t_zero,
                             period=period_anom,
                             a=a,
                             q=q,
                             f_c=f_c, f_s=f_s,
                             ld_1=ld_1, ld_2=ld_2,
                             ldc_1=ldc_1, ldc_2=ldc_2,
                             gdc_1=gdc_1, gdc_2=gdc_2,
                             didt=didt, domdt=domdt,
                             rotfac_1=rotfac_1, rotfac_2=rotfac_2,
                             hf_1=hf_1, hf_2=hf_2,
                             bfac_1=None, bfac_2=None,
                             heat_1=heat_1, heat_2=heat_2,
                             lambda_1=lambda_1, lambda_2=lambda_2,
                             vsini_1=vsini_1.value, vsini_2=vsini_2.value,
                             t_exp=t_exp, n_int=n_int,
                             grid_1=grid_1, grid_2=grid_2,
                             shape_1=shape_1, shape_2=shape_2,
                             spots_1=spots_1, spots_2=spots_2,
                             flux_weighted=flux_weighted,
                             verbose=1)

            if kwargs.get('return_dict_only', False):
                return rv_kwargs

            logger.info("calling ellc.rv for dataset='{}'".format(info['dataset']))
            logger.debug("ellc.rv(**{})".format(rv_kwargs))
            rvs1, rvs2 = ellc.rv(**rv_kwargs)


            # fill packets
            packetlist = []

            packetlist.append(_make_packet('times',
                                           info['times']*u.d,
                                           None,
                                           info))

            rvs = rvs1 if b.hierarchy.get_primary_or_secondary(info['component'])=='primary' else rvs2

            packetlist.append(_make_packet('rvs',
                                           rvs*u.km/u.s,
                                           None,
                                           info))
        else:
            raise TypeError("ellc only supports 'lc' and 'rv' datasets")

        return packetlist
