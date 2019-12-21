import os
import numpy as np

try:
  import commands
except:
  import subprocess as commands

import tempfile
from phoebe.parameters import dataset as _dataset
from phoebe.parameters import ParameterSet
from phoebe import dynamics
from phoebe.backend import universe, etvs, horizon_analytic
from phoebe.atmospheres import passbands
from phoebe.distortions  import roche
from phoebe.frontend import io
import phoebe.frontend.bundle
from phoebe import u, c
from phoebe import conf, mpi

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

import logging
logger = logging.getLogger("BACKENDS")
logger.addHandler(logging.NullHandler())

# the following list is for backends that use numerical meshes
_backends_that_require_meshing = ['phoebe', 'legacy']

_skip_filter_checks = {'check_default': False, 'check_visible': False}

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

    if kind == 'lc' and compute_kind=='phoebe' and b.get_value(qualifier='lc_method', compute=compute, dataset=dataset, context='compute')=='analytical':
        return False

    if kind == 'rv' and (compute_kind == 'legacy' or b.get_value(qualifier='rv_method', compute=compute, component=component, dataset=dataset, context='compute')=='dynamical'):
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
            add_ps = b.filter(dataset=include_times_entry, context='dataset')
            add_timequalifier = _timequalifier_by_kind(add_ps.kind)
            add_ps_components = add_ps.filter(qualifier=add_timequalifier).components
            # print "*** add_ps_components", add_dataset, add_ps_components
            if len(add_ps.times):
                add_times = np.array([float(t) for t in add_ps.times])
            elif len(add_ps_components):
                # then we need to concatenate over all components_
                # (times@rv@primary and times@rv@secondary are not necessarily
                # identical)
                add_times = np.unique(np.append(*[add_ps.get_value(qualifier='compute_times', component=c) for c in add_ps_components]))
                if not len(add_times):
                    add_times = np.unique(np.append(*[add_ps.get_value(qualifier=add_timequalifier, component=c) for c in add_ps_components]))
            else:
                # then we're adding from some dataset at the system-level (like lcs)
                # that have component=None
                add_times = add_ps.get_value(qualifier='compute_times', component=None, unit=u.d)
                if not len(add_times):
                    add_times = add_ps.get_value(qualifier=add_timequalifier, component=None, unit=u.d)
        else:
            # then some sort of t0 from context='component' or 'system'
            add_times = [b.get_value(include_times_entry, context=['component', 'system'])]

        return add_times

    # print "*** _expand_mesh_times", dataset_ps, dataset_ps.kind, component
    if dataset_ps.kind != 'mesh':
        raise TypeError("_expand_mesh_times only works for mesh datasets")

    # we're first going to access the compute_times@mesh... this should not have a component tag
    this_times = dataset_ps.get_value(qualifier='compute_times', component=None, unit=u.d)
    this_times = np.unique(np.append(this_times,
                                     [get_times(b, include_times_entry) for include_times_entry in dataset_ps.get_value(qualifier='include_times', expand=True)]
                                     )
                           )

    return this_times

def _extract_from_bundle(b, compute, times=None, allow_oversample=False,
                         by_time=True, **kwargs):
    """
    Extract a list of sorted times and the datasets that need to be
    computed at each of those times.  Any backend can then loop through
    these times and see what quantities are needed for that time step.

    Empty copies of synthetics for each applicable dataset are then
    created and returned so that they can be filled by the given backend.
    Setting of other meta-data should be handled by the bundle once
    the backend returns the filled synthetics.

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :return: times (list of floats), infos (list of lists of dictionaries),
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

    for dataset in b.filter(qualifier='enabled', compute=compute, value=True).datasets:
        dataset_ps = b.filter(context='dataset', dataset=dataset)
        dataset_compute_ps = b.filter(context='compute', dataset=dataset, compute=compute)
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
            if provided_times:
                this_times = provided_times
            elif dataset_kind == 'mesh':
                this_times = _expand_mesh_times(b, dataset_ps, component)
            elif dataset_kind in ['lp']:
                this_times = np.unique(dataset_ps.get_value(qualifier='compute_times', unit=u.d))
                if not len(this_times):
                    # then we have Parameters tagged by times, this will probably
                    # also apply to spectra.
                    this_times = [float(t) for t in dataset_ps.times]
            else:
                timequalifier = _timequalifier_by_kind(dataset_kind)
                timecomponent = component if dataset_kind not in ['mesh', 'lc'] else None
                # print "*****", dataset_kind, dataset_ps.kinds, timequalifier, timecomponent
                # NOTE: compute_times is not component-dependent, but times can be (i.e. for RV datasets)
                this_times = dataset_ps.get_value(qualifier='compute_times', unit=u.d)
                if not len(this_times):
                    this_times = dataset_ps.get_value(qualifier=timequalifier, component=timecomponent, unit=u.d)

                # we may also need to compute at other times if requested by a
                # mesh with this dataset in datasets@mesh
                # for mesh_datasets_parameter in mesh_datasets_parameters:
                    # if dataset in mesh_datasets_parameter.get_value():
                        # mesh_obs_ps = b.filter(context='dataset', dataset=mesh_datasets_parameter.dataset, component=None)
                        # TODO: not sure about the component=None on the next line... what will this do for rvs with different times per-component?
                        # mesh_times = _expand_mesh_times(b, mesh_obs_ps, component=None)
                        # this_times = np.unique(np.append(this_times, mesh_times))

            if allow_oversample and \
                    dataset_kind in ['lc'] and \
                    b.get_value(qualifier='exptime', dataset=dataset) > 0 and \
                    dataset_compute_ps.get_value(qualifier='fti_method', check_visible=False, **kwargs)=='oversample':

                # Then we need to override the times retrieved from the dataset
                # with the oversampled times.  Later we'll do an average over
                # the exposure.
                # NOTE: here we assume that the dataset times are at mid-exposure,
                # if we want to allow more flexibility, we'll need a parameter
                # that gives this option and different logic for each case.
                exptime = dataset_ps.get_value(qualifier='exptime', unit=u.d)
                fti_oversample = dataset_compute_ps.get_value(qualifier='fti_oversample', check_visible=False, **kwargs)
                # NOTE: if changing this, also change in bundle.run_compute
                this_times = np.array([np.linspace(t-exptime/2., t+exptime/2., fti_oversample) for t in this_times]).flatten()

            if dataset_kind in ['lp']:
                # for line profiles and spectra, we only need to compute synthetic
                # model if there are defined wavelengths
                this_wavelengths = dataset_ps.get_value(qualifier='wavelengths', component=component)
            else:
                this_wavelengths = None

            if len(this_times) and (this_wavelengths is None or len(this_wavelengths)):

                info = {'dataset': dataset,
                        'component': component,
                        'kind': dataset_kind,
                        'needs_mesh': _needs_mesh(b, dataset, dataset_kind, component, compute),
                        }

                if dataset_kind == 'mesh':
                    # then we may be requesting passband-dependent columns be
                    # copied to the mesh from other datasets based on the values
                    # of columns@mesh.  Let's store the needed information here,
                    # where mesh_datasets and mesh_kinds correspond to each
                    # other (but mesh_columns does not).
                    info['mesh_coordinates'] = dataset_ps.get_value(qualifier='coordinates', expand=True)
                    info['mesh_columns'] = dataset_ps.get_value(qualifier='columns', expand=True)
                    info['mesh_datasets'] = list(set([c.split('@')[1] for c in info['mesh_columns'] if len(c.split('@'))>1]))
                    info['mesh_kinds'] = [b.filter(dataset=ds, context='dataset').kind for ds in info['mesh_datasets']]

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

    def _get_packet_and_syns(self, b, compute, times=[], **kwargs):
        """
        see get_packet_and_syns.  _get_packet_and_syns provides the custom parts
        of the packet that are Backend-dependent.

        This should return the packet to send to all workers and the new_syns to
        be sent to the master.

        return packet, new_syns
        """
        raise NotImplementedError("_get_packet_and_syns is not implemented by the {} backend".format(self.__class__.__name__))

    def get_packet_and_syns(self, b, compute, times=[], **kwargs):
        """
        get_packet is called by the master and must get all information necessary
        to send to all workers.  The returned packet will be passed on as
        _run_chunk(**packet) with the following exceptions:

        * b: the bundle will be included in the packet serialized
        * compute: the label of the compute options will be included in the packet
        * backend: the class name will be passed on in the packet so the worker can call the correct backend
        * all kwargs will be passed on verbatim
        """
        packet, new_syns = self._get_packet_and_syns(b, compute, times, **kwargs)
        for k,v in kwargs.items():
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
                        new_syns.set_value(check_visible=False, check_default=False, **packet)
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

    def run(self, b, compute, times=[], **kwargs):
        """
        if within mpirun, workers should call _run_worker instead of run
        """
        self.run_checks(b, compute, times, **kwargs)

        logger.debug("rank:{}/{} calling get_packet_and_syns".format(mpi.myrank, mpi.nprocs))
        packet, new_syns = self.get_packet_and_syns(b, compute, times, **kwargs)

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

    def _get_packet_and_syns(self, b, compute, times=[], **kwargs):
        # extract times/infolists/new_syns from the bundle
        # if the input for times is an empty list, we'll obey dataset times
        # otherwise all datasets will be overridden with the times provided
        # see documentation in _extract_from_bundle for details on the output variables.
        times, infolists, new_syns = _extract_from_bundle(b, compute=compute,
                                                          times=times,
                                                          allow_oversample=True,
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
        for i, time, infolist in zip(inds, times, infolists):
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

    def _get_packet_and_syns(self, b, compute, times=[], **kwargs):
        # self.run_checks(b, compute, times, **kwargs)

        # see documentation in _extract_from_bundle for details on the output variables.
        infolist, new_syns = _extract_from_bundle(b, compute=compute,
                                               times=times, by_time=False)

        packet = {'infolist': infolist}

        return packet, new_syns


    def _run_chunk(self, b, compute, infolist, **kwargs):
        worker_setup_kwargs = self._worker_setup(b, compute, infolist, **kwargs)

        if mpi.enabled:
            # np.array_split(any_input_array, mpi.nprocs)[mpi.myrank]
            infolist = np.array_split(infolist, mpi.nprocs)[mpi.myrank]

        packetlists = [] # entry per-dataset
        for info in infolist:
            packetlist = self._run_single_dataset(b, info, **worker_setup_kwargs)
            packetlists.append(packetlist)

        return packetlists





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

    def _create_system_and_compute_pblums(self, b, compute,
                                          dynamics_method=None,
                                          hier=None,
                                          meshablerefs=None,
                                          datasets=None,
                                          compute_l3=True,
                                          compute_l3_frac=False,
                                          compute_extrinsic=False,
                                          reset=True,
                                          lc_only=True,
                                          **kwargs):

        logger.debug("rank:{}/{} PhoebeBackend._create_system_and_compute_pblums: calling universe.System.from_bundle".format(mpi.myrank, mpi.nprocs))
        system = universe.System.from_bundle(b, compute, datasets=b.datasets, **kwargs)

        if dynamics_method is None:
            computeparams = b.get_compute(compute, force_ps=True)
            dynamics_method = computeparams.get_value(qualifier='dynamics_method', **kwargs)

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

        logger.debug("rank:{}/{} PhoebeBackend._create_system_and_compute_pblums: handling pblum scaling".format(mpi.myrank, mpi.nprocs))
        # NOTE: system.compute_pblum_scalings populates at t0 with ignore_effect=True (so intrinsic pblum)
        system.compute_pblum_scalings(b, datasets, t0, x0, y0, z0, vx0, vy0, vz0, etheta0, elongan0, eincl0, reset=False, lc_only=lc_only)
        if compute_l3 or compute_extrinsic:
            if len(b.features):
                # then the features may affect intrinsic vs extrinsic pblums,
                # so we need to reset and force re-meshing
                system.reset(force_remesh=True)

        if compute_l3 and (compute_l3_frac or "frac" in [list(l3.keys())[0] for l3 in system.l3s.values()]):
            logger.debug("rank:{}/{} PhoebeBackend._create_system_and_compute_pblums: computing l3s".format(mpi.myrank, mpi.nprocs))
            system.compute_l3s(datasets, t0, x0, y0, z0, vx0, vy0, vz0, etheta0, elongan0, eincl0, compute_l3_frac=compute_l3_frac, reset=False)
        elif compute_extrinsic:
            logger.debug("rank:{}/{} PhoebeBackend._create_system_and_compute_pblums: recomputing with extrinsic effects enabled".format(mpi.myrank, mpi.nprocs))
            system.update_positions(t0, x0, y0, z0, vx0, vy0, vz0, etheta0, elongan0, eincl0, ignore_effects=True)

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

        # if ld_mode_bol is lookup, we need to pre-compute those and store
        # them in the (hidden) ld_coeffs_bol parameters
        # TODO [optimize]: skip this if irrad_method is 'none' or albedos are 0?
        b._compute_necessary_values(computeparams, **kwargs)

        do_horizon = False #computeparams.get_value(qualifier='horizon', **kwargs)
        dynamics_method = computeparams.get_value(qualifier='dynamics_method', dynamics_method=kwargs.pop('dynamics_method', None), **_skip_filter_checks)
        ltte = computeparams.get_value(qualifier='ltte', ltte=kwargs.pop('ltte', None), **_skip_filter_checks)
        distance = b.get_value(qualifier='distance', context='system', unit=u.m, distance=kwargs.pop('distance', None), **_skip_filter_checks)

        # TODO: skip initializing system if we NEVER need meshes
        system = self._create_system_and_compute_pblums(b, compute,
                                                        dynamics_method=dynamics_method,
                                                        hier=hier,
                                                        meshablerefs=meshablerefs,
                                                        compute_l3=True,
                                                        compute_extrinsic=False,
                                                        **kwargs)

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
                    distance=distance,
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
        distance = kwargs.get('distance')
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

                    rv = obs['rv']
                else:
                    # then rv_method == 'dynamical'
                    rv = -1*vzi[cind]

                packetlist.append(_make_packet('rvs',
                                              rv*u.solRad/u.d,
                                              time, info))

            elif kind=='lc':
                obs = system.observe(info['dataset'],
                                     kind=kind,
                                     components=info['component'],
                                     distance=distance)

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
            raise ImportError("phoebeBackend for phoebe legacy not found")

        if len(starrefs)!=2:
            raise ValueError("only binaries are supported by legacy backend")


    def _worker_setup(self, b, compute, infolist, **kwargs):
        """
        """
        logger.debug("rank:{}/{} LegacyBackend._worker_setup: creating temporary phoebe file".format(mpi.myrank, mpi.nprocs))

        # make phoebe 1 file
        tmp_filename = temp_name = next(tempfile._get_candidate_names())
        io.pass_to_legacy(b, filename=tmp_filename, compute=compute, **kwargs)
        phb1.init()
        try:
            if hasattr(phb1, 'auto_configure'):
                # then phb1 is phoebe_legacy
                phb1.auto_configure()
            else:
                # then phb1 is phoebeBackend
                phb1.configure()
        except SystemError:
            raise SystemError("PHOEBE config failed: try creating PHOEBE config file through GUI")

        phb1.open(tmp_filename)

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

        computeparams = b.get_compute(compute, force_ps=True)

        os.remove(tmp_filename)

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
                                                dataset=info['dataset'])

            phb1.setpar(proximity_par, rv_method=='flux-weighted')

            rvs = np.array(rv_call(tuple(info['times'].tolist()), rvind))

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
            raise ImportError('photodynam executable not found')


    def _worker_setup(self, b, compute, infolist, **kwargs):
        """
        """
        logger.debug("rank:{}/{} PhotodynamBackend._worker_setup".format(mpi.myrank, mpi.nprocs))

        computeparams = b.get_compute(compute, force_ps=True)

        b._compute_necessary_values(computeparams)

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

            packetlist.append(_make_packet('times',
                                           stuff[0]*u.d,
                                           None,
                                           info))

            packetlist.append(_make_packet('rvs',
                                           -stuff[3*nbodies+4+(cind*3)] * u.AU/u.d,
                                           None,
                                           info))

        else:
            raise NotImplementedError("kind {} not yet supported by this backend".format(info['kind']))

        return packetlist



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
            raise ImportError('jktebop executable not found.')

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

        logger.warning("jktebop backend is still in development/testing and is VERY experimental")


    def _worker_setup(self, b, compute, infolist, **kwargs):
        """
        """
        logger.debug("rank:{}/{} JktebopBackend._worker_setup".format(mpi.myrank, mpi.nprocs))

        computeparams = b.get_compute(compute, force_ps=True)

        b._compute_necessary_values(computeparams)

        hier = b.get_hierarchy()

        starrefs  = hier.get_stars()
        orbitrefs = hier.get_orbits()

        orbitref = orbitrefs[0]

        ringsize = computeparams.get_value(qualifier='ringsize', unit=u.deg, **kwargs)

        rA = b.get_value(qualifier='requiv', component=starrefs[0], context='component', unit=u.solRad)
        rB = b.get_value(qualifier='requiv', component=starrefs[1], context='component', unit=u.solRad)
        sma = b.get_value(qualifier='sma', component=orbitref, context='component', unit=u.solRad)
        incl = b.get_value(qualifier='incl', component=orbitref, context='component', unit=u.deg)
        q = b.get_value(qualifier='q', component=orbitref, context='component')
        ecosw = b.get_value(qualifier='ecosw', component=orbitref, context='component')
        esinw = b.get_value(qualifier='esinw', component=orbitref, context='component')

        gravbA = b.get_value(qualifier='gravb_bol', component=starrefs[0], context='component')
        gravbB = b.get_value(qualifier='gravb_bol', component=starrefs[1], context='component')


        period = b.get_value(qualifier='period', component=orbitref, context='component', unit=u.d)
        t0_supconj = b.get_value(qualifier='t0_supconj', component=orbitref, context='component', unit=u.d)


        return dict(compute=compute,
                    starrefs=starrefs,
                    oritref=orbitref,
                    ringsize=ringsize,
                    rA=rA, rB=rB,
                    sma=sma, incl=incl, q=q,
                    ecosw=ecosw, esinw=esinw,
                    gravbA=gravbA, gravbB=gravbB,
                    period=period, t0_supconj=t0_supconj)

    def _run_single_dataset(self, b, info, **kwargs):
        """
        """
        logger.debug("rank:{}/{} JktebopBackend._run_single_dataset(info['dataset']={} info['component']={} info.keys={}, **kwargs.keys={})".format(mpi.myrank, mpi.nprocs, info['dataset'], info['component'], info.keys(), kwargs.keys()))

        compute = kwargs.get('compute')
        starrefs = kwargs.get('starrefs')
        orbitref = kwargs.get('orbitref')
        ringsize = kwargs.get('ringsize')
        rA = kwargs.get('rA')
        rB = kwargs.get('rB')
        sma = kwargs.get('sma')
        incl = kwargs.get('incl')
        q = kwargs.get('q')
        ecosw = kwargs.get('ecosw')
        esinw = kwargs.get('esinw')
        gravbA = kwargs.get('gravbA')
        gravbB = kwargs.get('gravbB')
        period = kwargs.get('period')
        t0_supconj = kwargs.get('t0_supconj')

        # get dataset-dependent things that we need
        l3 = b.get_value(qualifier='l3', dataset=info['dataset'], context='dataset')

        ldfuncA = b.get_value(qualifier='ld_func', component=starrefs[0], dataset=info['dataset'], context='dataset')
        ldfuncB = b.get_value(qualifier='ld_func', component=starrefs[1], dataset=info['dataset'], context='dataset')

        # use check_visible=False to access the ld_coeffs from
        # compute_ld_coeffs(set_value=True) done in _worker_setup
        ldcoeffsA = b.get_value(qualifier='ld_coeffs', component=starrefs[0], dataset=info['dataset'], context='dataset', check_visible=False)
        ldcoeffsB = b.get_value(qualifier='ld_coeffs', component=starrefs[1], dataset=info['dataset'], context='dataset', check_visible=False)

        irrad_method = b.get_value(qualifier="irrad_method", compute=compute, context='compute')
        if irrad_method == "biaxial spheroid":
            albA = b.get_value(qualifier='irrad_frac_refl_bol', component=starrefs[0], context='component')
            albB = b.get_value(qualifier='irrad_frac_refl_bol', component=starrefs[1], context='component')
        elif irrad_method == 'none':
            albA = 0.0
            albB = 0.0
        else:
            raise NotImplementedError("irrad_method '{}' not supported".format(irrad_method))

        logger.debug("estimating surface brightness ratio from pblum and requiv")
        # note: these aren't true surface brightnesses, but the ratio should be fine
        sb_primary = b.get_value(qualifier='pblum', component=starrefs[0], dataset=info['dataset'], context='dataset', unit=u.W, check_visible=False) / b.get_value(qualifier='requiv', component=starrefs[0], context='component', unit=u.solRad)**2
        sb_secondary = b.get_value(qualifier='pblum', component=starrefs[1], dataset=info['dataset'], context='dataset', unit=u.W, check_visible=False) / b.get_value(qualifier='requiv', component=starrefs[1], context='component', unit=u.solRad)**2
        sb_ratio =  sb_secondary / sb_primary

        # provide translation from phoebe's 'ld_func' to jktebop's 'LD law type'
        ldfuncs = {'linear': 'lin',
                   'logarithmic': 'log',
                   'square_root': 'sqrt',
                   'quadratic': 'quad'}

        # let's make sure we'll be able to make the translation later
        if ldfuncA not in ldfuncs.keys() or ldfuncB not in ldfuncs.keys():
            # NOTE: this is now handle in b.run_checks, so should never happen
            # TODO: provide a more useful error statement
            raise ValueError("jktebop only accepts the following options for ld_func: {}".format(ldfuncs.keys()))

        # create the input file for jktebop
        fi = open('_tmp_jktebop_in', 'w')
        #~ fi.write("# JKTEBOP input file created by PHOEBE\n")

        # We always want task 2 - according to jktebop's website:
        # Task 2 	This inputs a parameter file and calculates a
        # synthetic light curve (10000 points between phases 0 and 1)
        # using the parameters you put in the file.
        fi.write('{:5} {:11} Task to do (from 1 to 9)   Integ. ring size (deg)\n'.format(2, ringsize))
        fi.write('{:5} {:11} Sum of the radii           Ratio of the radii\n'.format((rA+rB)/sma, rA/rB))
        fi.write('{:5} {:11} Orbital inclination (deg)  Mass ratio of the system\n'.format(incl, q))

        # we'll provide ecosw and esinw instead of ecc and long_an
        # jktebop's readme.txt states that so long as ecc is < 10,
        # it will be intrepreted as ecosw and esinw (otherwise would
        # need to be ecc+10 and long_an (deg)
        fi.write('{:5} {:11} Orbital eccentricity       Periastron longitude deg\n'.format(ecosw, esinw))


        fi.write('{:5} {:11} Gravity darkening (starA)  Grav darkening (starB)\n'.format(gravbA, gravbB))
        fi.write('{:5} {:11} Surface brightness ratio   Amount of third light\n'.format(sb_ratio, l3))


        # According to jktebop's readme.txt:
        # The possible entries for the type of limb darkening law are 'lin' (for linear)
        # 'log' (logarithmic), 'sqrt' (square-root), 'quad' (quadratic) or 'cub' (cubic)

        fi.write('{:5} {:11} LD law type for star A     LD law type for star B\n'.format(ldfuncs[ldfuncA], ldfuncs[ldfuncB]))
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

        #~ fi.write('{}  Name of file containing light curve\n'.format("_tmp_jktebop_lc_in"))
        #~ fi.write('{}  Name of output parameter file\n'.format("_tmp_jktebop_param"))
        #~ fi.write('{}  Name of output light curve file\n'.format("_tmp_jktebop_lc_out"))
        #~ fi.write('{}  Name of output model light curve fit file\n'.format("_tmp_jktebop_modelfit_out"))
        #~ fi.write('{}\n')

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
        #~ fi.write('rv1 llaqr-rv1.dat llaqr-rv1.out 55.0 -10.0 1 1\n')
        #~ fi.write('rv2 llaqr-rv2.dat llaqr-rv2.out 55.0 -10.0 1 1\n')


        # According to jktebop's readme.txt:
        # NUMERICAL INTEGRATION:  long exposure times can be split up into NUMINT points
        # occupying a total time interval of NINTERVAL (seconds) by including this line:
        #   NUMI  [numint]  [ninterval]

        # TODO: allow exposure times


        fi.close()

        # TODO: create_tmp_jktebop_lc_in - probably with times and dummy fluxes if none are in the obs
        #~ flc = open('_tmp_jktebop_lc_in', 'w')
        #~ times = b.get_value(qualifier='times', component=info['component'], dataset=info['dataset'], context='dataset', unit=u.d)
        #~ fluxes = b.get_value(qualifier='flux', component=info['component'], dataset=info['dataset'], context='dataset', unit=u.d)

        #~ if len(fluxes) < len(times):
            #~ # then just provide dummy fluxes - we're not using
            #~ # jktebop for fitting anyways, it really just needs the times
            #~ fluxes = [1.]*len(times)

        #~ for t,f in zip(times, fluxes):
            #~ flc.write('{}\t{}\n'.format(t,t))
        #~ flc.close()

        # run jktebop
        out = commands.getoutput("jktebop _tmp_jktebop_in > _tmp_jktebop_lc_out")

        # parse output
        phases_all, mags_all, l1, l2, l3 = np.loadtxt(str(period), unpack=True)
        #~ time, flux = np.loadtxt("_tmp_jktebop_lc_out", unpack=True)

        # fill packets
        packetlist = []

        # phases_all, mags_all are 10001 evenly-spaced phases, so we need to interpolate
        # to get at the desired times
        times_all = b.to_time(phases_all)  # in days
        mags_interp = np.interp(info['times'], times_all, mags_all)

        logger.warning("converting from mags from jktebop to flux")
        fluxes = 10**((0.0-mags_interp)/2.5) * b.get_value(qualifier='pbflux', dataset=info['dataset'], context='dataset', unit=u.W/u.m**2, check_visible=False)

        packetlist.append(_make_packet('times',
                                       info['times']*u.d,
                                       None,
                                       info))

        packetlist.append(_make_packet('fluxes',
                                       fluxes,
                                       None,
                                       info))

        return packetlist

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
            raise ImportError("could not import ellc")

        hier = b.get_hierarchy()

        starrefs  = hier.get_stars()
        orbitrefs = hier.get_orbits()

        if len(starrefs) != 2 or len(orbitrefs) != 1:
            raise ValueError("ellc backend only accepts binary systems")

        logger.warning("ellc backend is still in development/testing and is VERY experimental")


    def _worker_setup(self, b, compute, infolist, **kwargs):
        """
        """
        logger.debug("rank:{}/{} EllcBackend._worker_setup".format(mpi.myrank, mpi.nprocs))

        computeparams = b.get_compute(compute, force_ps=True, check_visible=False)

        b._compute_necessary_values(computeparams)

        hier = b.get_hierarchy()

        starrefs  = hier.get_stars()
        orbitrefs = hier.get_orbits()

        orbitref = orbitrefs[0]

        shape_1 = computeparams.get_value(qualifier='distortion_method', component=starrefs[0])
        shape_2 = computeparams.get_value(qualifier='distortion_method', component=starrefs[1])

        hf_1 = computeparams.get_value(qualifier='hf', component=starrefs[0], check_visible=False)
        hf_2 = computeparams.get_value(qualifier='hf', component=starrefs[1], check_visible=False)

        grid_1 = computeparams.get_value(qualifier='grid', component=starrefs[0])
        grid_2 = computeparams.get_value(qualifier='grid', component=starrefs[1])

        exact_grav = computeparams.get_value(qualifier='exact_grav')

        a = b.get_value(qualifier='sma', component=orbitref, context='component', unit=u.solRad)
        radius_1 = b.get_value(qualifier='requiv', component=starrefs[0], context='component', unit=u.solRad) / a
        radius_2 = b.get_value(qualifier='requiv', component=starrefs[1], context='component', unit=u.solRad) / a

        period = b.get_value(qualifier='period', component=orbitref, context='component', unit=u.d)
        q = b.get_value(qualifier='q', component=orbitref, context='component')

        # TODO: there seems to be a convention flip between primary and secondary star in ellc... maybe we can just address via t_zero?
        t_zero = b.get_value(qualifier='t0_supconj', component=orbitref, context='component', unit=u.d)

        incl = b.get_value(qualifier='incl', component=orbitref, context='component', unit=u.deg)
        didt = 0.0
        # didt = b.get_value(qualifier='dincldt', component=orbitref, context='component', unit=u.deg/u.d) * period

        ecc = b.get_value(qualifier='ecc', component=orbitref, context='component')
        w = b.get_value(qualifier='per0', component=orbitref, context='component', unit=u.rad)

        domdt = b.get_value(qualifier='dperdt', component=orbitref, context='component', unit=u.deg/u.d) * period

        gdc_1 = b.get_value(qualifier='gravb_bol', component=starrefs[0], context='component')
        gdc_2 = b.get_value(qualifier='gravb_bol', component=starrefs[1], context='component')

        rotfac_1 = b.get_value(qualifier='syncpar', component=starrefs[0], context='component')
        rotfac_2 = b.get_value(qualifier='syncpar', component=starrefs[1], context='component')

        f_c = np.sqrt(ecc) * np.cos(w)
        f_s = np.sqrt(ecc) * np.sin(w)


        return dict(compute=compute,
                    starrefs=starrefs,
                    oritref=orbitref,
                    shape_1=shape_1, shape_2=shape_2,
                    grid_1=grid_1, grid_2=grid_2,
                    exact_grav=exact_grav,
                    radius_1=radius_1, radius_2=radius_2,
                    incl=incl,
                    t_zero=t_zero,
                    period=period,
                    q=q,
                    a=a,
                    f_c=f_c, f_s=f_s,
                    didt=didt, domdt=domdt,
                    gdc_1=gdc_1, gdc_2=gdc_2,
                    rotfac_1=rotfac_1, rotfac_2=rotfac_2)

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

        radius_1 = kwargs.get('radius_2')
        radius_2 = kwargs.get('radius_1')

        incl = kwargs.get('incl')

        t_zero = kwargs.get('t_zero')
        period = kwargs.get('period')
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


        # get dataset-dependent things that we need
        ldfuncA = b.get_value(qualifier='ld_func', component=starrefs[0], dataset=info['dataset'], context='dataset')
        ldfuncB = b.get_value(qualifier='ld_func', component=starrefs[1], dataset=info['dataset'], context='dataset')

        # use check_visible=False to access the ld_coeffs from
        # compute_ld_coeffs(set_value=True) done in _worker_setup
        ldcoeffsA = b.get_value(qualifier='ld_coeffs', component=starrefs[0], dataset=info['dataset'], context='dataset', check_visible=False)
        ldcoeffsB = b.get_value(qualifier='ld_coeffs', component=starrefs[1], dataset=info['dataset'], context='dataset', check_visible=False)

        # albA = b.get_value(qualifier='irrad_frac_refl_bol', component=starrefs[0], context='component')
        # albB = b.get_value(qualifier='irrad_frac_refl_bol', component=starrefs[1], context='component')

        if info['kind'] == 'lc':
            light_3 = b.get_value(qualifier='l3_frac', dataset=info['dataset'], context='dataset', check_visible=False)

            # this is just a hack for now, we'll eventually want the true sb ratio
            logger.info("computing sb_ratio from pblums and requivs for dataset='{}'".format(info['dataset']))
            # note: these aren't true surface brightnesses, but the ratio should be fine
            sb_primary = b.get_value(qualifier='pblum', component=starrefs[0], dataset=info['dataset'], context='dataset', unit=u.W, check_visible=False) / b.get_value(qualifier='requiv', component=starrefs[0], context='component', unit=u.solRad)**2
            sb_secondary = b.get_value(qualifier='pblum', component=starrefs[1], dataset=info['dataset'], context='dataset', unit=u.W, check_visible=False) / b.get_value(qualifier='requiv', component=starrefs[1], context='component', unit=u.solRad)**2
            sb_ratio =  sb_secondary / sb_primary

            t_exp = b.get_value(qualifier='exptime', dataset=info['dataset'], context='dataset')

            # move outside above 'lc' if-statement once exptime is supported for RVs in phoebe
            if b.get_value(qualifier='fti_method', compute=compute, dataset=info['dataset'], context='compute') == 'oversample':
                n_int = b.get_value(qualifier='fti_oversample', compute=compute, dataset=info['dataset'], context='compute')
            else:
                n_int = 1

            logger.info("calling ellc.lc for dataset='{}'".format(info['dataset']))
            fluxes = ellc.lc(info['times'],
                             radius_1, radius_2,
                             sb_ratio,
                             incl,
                             light_3,
                             t_zero, period, a, q,
                             f_c, f_s,
                             ldc_1=None, ldc_2=None,
                             gdc_1=gdc_1, gdc_2=gdc_2,
                             didt=didt, domdt=domdt,
                             rotfac_1=rotfac_1, rotfac_2=rotfac_2,
                             hf_1=hf_1, hf_2=hf_2,
                             bfac_1=None, bfac_2=None,
                             heat_1=None, heat_2=None,
                             lambda_1=None, lambda_2=None,
                             vsini_1=None, vsini_2=None,
                             t_exp=t_exp, n_int=n_int,
                             grid_1=grid_1, grid_2=grid_2,
                             ld_1=None, ld_2=None,
                             shape_1=shape_1, shape_2=shape_2,
                             spots_1=None, spots_2=None,
                             exact_grav=exact_grav,
                             verbose=1)

            # ellc returns "arbitrary" flux values... let's try to rescale
            # to our flux units to be compatible with other backends
            fluxes *= b.get_value(qualifier='pbflux', dataset=info['dataset'], context='dataset', unit=u.W/u.m**2, check_visible=False)

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
            rv_method = b.get_value(qualifier='rv_method', compute=compute, dataset=info['dataset'], component=info['component'], context='compute')
            flux_weighted = rv_method == 'flux-weighted'
            if flux_weighted:
                raise NotImplementedError("flux-weighted does not seem to work in ellc")

            # surface-brightness ratio shouldn't matter for rvs...
            sb_ratio = 1.0

            # enable once exptime for RVs is supported in PHOEBE
            # t_exp = b.get_value(qualifier='exptime', dataset=info['dataset'], context='dataset')
            t_exp = 0
            n_int = 1

            logger.info("calling ellc.rv for dataset='{}'".format(info['dataset']))
            rvs1, rvs2 = ellc.rv(info['times'],
                                  radius_1, radius_2,
                                  sb_ratio,
                                  incl,
                                  t_zero, period, a, q,
                                  f_c, f_s,
                                  ldc_1=None, ldc_2=None,
                                  gdc_1=gdc_1, gdc_2=gdc_2,
                                  didt=didt, domdt=domdt,
                                  rotfac_1=rotfac_1, rotfac_2=rotfac_2,
                                  hf_1=hf_1, hf_2=hf_2,
                                  bfac_1=None, bfac_2=None,
                                  heat_1=None, heat_2=None,
                                  lambda_1=None, lambda_2=None,
                                  vsini_1=None, vsini_2=None,
                                  t_exp=t_exp, n_int=n_int,
                                  grid_1=grid_1, grid_2=grid_2,
                                  ld_1=None, ld_2=None,
                                  shape_1=shape_1, shape_2=shape_2,
                                  spots_1=None, spots_2=None,
                                  flux_weighted=flux_weighted,
                                  verbose=1)


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
