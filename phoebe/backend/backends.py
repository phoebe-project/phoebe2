import os
import numpy as np
import commands
import tempfile
from phoebe.parameters import dataset as _dataset
from phoebe.parameters import ParameterSet
from phoebe import dynamics
from phoebe.backend import universe, etvs, horizon_analytic
from phoebe.distortions  import roche
from phoebe.frontend import io
from phoebe import u, c
from phoebe import conf

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


# this is a bit of a hack and will only work with openmpi, but environment
# variables seem to be the only way to detect whether the script was run
# via mpirun or not
if 'OMPI_COMM_WORLD_SIZE' in os.environ.keys():
    from mpi4py import MPI
    _use_mpi = True

    comm   = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    nprocs = comm.Get_size()

    TAG_REQ  = 41
    TAG_DATA = 42

    if nprocs==1:
        raise ImportError("need more than 1 processor to run with mpi")

else:
    _use_mpi = False

import logging
logger = logging.getLogger("BACKENDS")
logger.addHandler(logging.NullHandler())

# the following list is for backends that use numerical meshes
_backends_that_require_meshing = ['phoebe', 'legacy']

def _needs_mesh(b, dataset, kind, component, compute):
    """
    """
    # print "*** _needs_mesh", kind
    compute_kind = b.get_compute(compute).kind
    if compute_kind not in _backends_that_require_meshing:
        # then we don't have meshes for this backend, so all should be False
        return False

    if kind not in ['mesh', 'lc', 'rv']:
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
            if len(add_ps_components):
                # then we need to concatenate over all components_
                # (times@rv@primary and times@rv@secondary are not necessarily
                # identical)
                add_times = np.unique(np.append(*[add_ps.get_value(qualifier=add_timequalifier, component=c) for c in add_ps_components]))
            else:
                # then we're adding from some dataset at the system-level (like lcs)
                # that have component=None
                add_times = add_ps.get_value(qualifier=add_timequalifier, component=None, unit=u.d)
        else:
            # then some sort of t0 from context='component' or 'system'
            add_times = [b.get_value(include_times_entry, context=['component', 'system'])]

        return add_times

    # print "*** _expand_mesh_times", dataset_ps, dataset_ps.kind, component
    if dataset_ps.kind != 'mesh':
        raise TypeError("_expand_mesh_times only works for mesh datasets")

    # we're first going to access the times@mesh... this should not have a component tag
    this_times = dataset_ps.get_value(qualifier='times', component=None, unit=u.d)
    this_times = np.unique(np.append(this_times,
                                     [get_times(b, include_times_entry) for include_times_entry in dataset_ps.get_value(qualifier='include_times', expand=True)]
                                     )
                           )

    return this_times

def _extract_from_bundle_by_time(b, compute, times=None, allow_oversample=False, **kwargs):
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
    # - times (list within _extract_from_bundle_by_time but then casted to np.array)
    # - infolists (list of <infolist>, same shape and order as times)
    # - infolist (list of <info>)
    # - info (dict containing information for a given dataset-component computation at the given time)
    #
    # The workers then return a similar format:
    # - packetlist (list of <packet>)
    # - packet (dict ready to be sent to new_syns.set_value(**packet))
    # where packet has a similar structure to info (but with value and possibly time keys)
    # but packetlist may be longer than infolist (since mesh passband-columns allow
    # now have their own entries.)

    for dataset in b.filter(qualifier='enabled', compute=compute, value=True).datasets:
        dataset_ps = b.filter(context='dataset', dataset=dataset).exclude(kind='*_dep')
        dataset_compute_ps = b.filter(context='compute', dataset=dataset, compute=compute, check_visible=False)
        dataset_kind = dataset_ps.kind
        time_qualifier = _timequalifier_by_kind(dataset_kind)
        if dataset_kind in ['lc']:
            dataset_components = [None]
        else:
            dataset_components = b.hierarchy.get_stars()

        for component in dataset_components:
            if provided_times:
                this_times = provided_times
            elif dataset_kind == 'mesh':
                this_times = _expand_mesh_times(b, dataset_ps, component)
            else:
                timequalifier = _timequalifier_by_kind(dataset_kind)
                timecomponent = component if dataset_kind not in ['mesh', 'lc'] else None
                # print "*****", dataset_kind, dataset_ps.kinds, timequalifier, timecomponent
                this_times = dataset_ps.get_value(qualifier=timequalifier, component=timecomponent, unit=u.d)

                # we may also need to compute at other times if requested by a
                # mesh with this dataset in datasets@mesh
                # for mesh_datasets_parameter in mesh_datasets_parameters:
                    # if dataset in mesh_datasets_parameter.get_value():
                        # mesh_obs_ps = b.filter(context='dataset', dataset=mesh_datasets_parameter.dataset, component=None).exclude(kind='*_dep')
                        # TODO: not sure about the component=None on the next line... what will this do for rvs with different times per-component?
                        # mesh_times = _expand_mesh_times(b, mesh_obs_ps, component=None)
                        # this_times = np.unique(np.append(this_times, mesh_times))

            # TODO: also copy this logic for _extract_from_bundle_by_dataset if
            # we decide to support oversamling with other backends
            if allow_oversample and \
                    dataset_kind in ['lc'] and \
                    b.get_value(qualifier='exptime', dataset=dataset, check_visible=False) > 0 and \
                    dataset_compute_ps.get_value(qualifier='fti_method', **kwargs)=='oversample':

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

            if len(this_times):

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
                    info['mesh_columns'] = dataset_ps.get_value('columns', expand=True)
                    info['mesh_datasets'] = list(set([c.split('@')[1] for c in info['mesh_columns'] if len(c.split('@'))>1]))
                    info['mesh_kinds'] = [b.filter(dataset=ds, context='dataset').exclude(kind='*_dep').kind for ds in info['mesh_datasets']]

                for time_ in this_times:
                    # TODO: handle some deltatime allowance here?
                    if time_ in times:
                        ind = times.index(time_)
                        infolists[ind].append(info)
                    else:
                        times.append(time_)
                        infolists.append([info])

                # we need the times for _create_syns but not within the infolists,
                # so we'll do this last and make a copy so times aren't passed
                # to everybody...
                needed_syn_info = info.copy()
                needed_syn_info['times'] = this_times
                needed_syns.append(needed_syn_info)

    if len(times):
        ti = zip(times, infolists)
        ti.sort()
        times, infolists = zip(*ti)

    # print "*** _extract_from_bundle_by_time return(times, infos, syns)", times, infolists, needed_syns
    return np.asarray(times), infolists, _create_syns(b, needed_syns)

def _extract_from_bundle_by_dataset(b, compute, times=[]):
    """
    Extract a list of enabled dataset from the bundle.

    Empty copies of synthetics for each applicable dataset are then
    created and returned so that they can be filled by the given backend.
    Setting of other meta-data should be handled by the bundle once
    the backend returns the filled synthetics.

    Unlike :func:`_extract_from_bundle_by_time`, this does not sort
    by times and combine datasets that need to be computed at the same
    timestamp.  In general, this function will be used by non-PHOEBE
    backends.

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

    for dataset in b.filter(qualifier='enabled', compute=compute, value=True).datasets:
        for component in b.hierarchy.get_stars()+[None]:
            obs_ps = b.filter(context='dataset', dataset=dataset, component=component).exclude(kind='*_dep')
            # only certain kinds accept a component of None
            if component is None and obs_ps.kind not in ['lc', 'mesh']:
                # TODO: should we change things like lightcurves to be tagged with the component label of the orbit instead of None?
                # anything that can accept observations on the "system" level should accept component is None
                continue

            timequalifier = _timequalifier_by_kind(obs_ps.kind)
            try:
                this_times = obs_ps.get_value(qualifier=timequalifier, component=component, unit=u.d)
            except ValueError: #TODO: custom exception for no parameter
                continue

            if obs_ps.kind == 'mesh':
                # then we also have the ability to append times from another dataset
                this_times = _expand_mesh_times(b, obs_ps, component)

            if len(this_times) and provided_times is not None:
                # then overrride the dataset times with the passed times
                #  (as kwarg to run_compute)
                this_times = provided_times

            if len(this_times):

                if component is None and obs_ps.kind in ['mesh']:
                    components = b.hierarchy.get_meshables()
                else:
                    components = [component]

                for component in components:

                    info = {'dataset': dataset,
                            'component': component,
                            'kind': obs_ps.kind,
                            'needs_mesh': _needs_mesh(b, dataset, obs_ps.kind, component, compute),
                            'times': this_times
                            }
                    needed_syns.append(info)

                    infolists.append([info])

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
            # needed_syn['datasets'] = {ds: b.filter(datset=ds, context='dataset').exclude(kind='*_dep').kind for ds in datasets}

        # phoebe will compute everything sorted - even if the input times array
        # is out of order, so let's make sure the exposed times array is in
        # the correct (sorted) order
        if 'times' in needed_syn.keys():
            needed_syn['times'].sort()

            needed_syn['empty_arrays_len'] = len(needed_syn['times'])

        these_params, these_constraints = getattr(_dataset, "{}_syn".format(syn_kind.lower()))(**needed_syn)
        # TODO: do we need to handle constraints?
        these_params = these_params.to_list()
        for param in these_params:
            param._component = needed_syn['component']
            if param._dataset is None:
                # dataset may be set for mesh columns
                param._dataset = needed_syn['dataset']
            param._kind = syn_kind
            # context, model, etc will be handle by the bundle once these are returned

        params += these_params

    return ParameterSet(params)

def phoebe(b, compute, times=[], as_generator=False, **kwargs):
    """
    Run the PHOEBE 2.0 backend.  This is the default built-in backend
    so no other pre-requisites are required.

    When using this backend, please cite
        * TODO: include list of citations

    When using dynamics_method=='nbody', please cite:
        * TODO: include list of citations for reboundx

    Parameters that are used by this backend:

    * Compute:
        * all parameters in :func:`phoebe.parameters.compute.phoebe`
    * Orbit:
        * TOOD: list these
    * Star:
        * TODO: list these
    * lc dataset:
        * TODO: list these

    Values that are filled by this backend:

    * lc:
        * times
        * fluxes
    * rv (dynamical only):
        * times
        * rvs

    This function will almost always be called through the bundle, using
        * :meth:`phoebe.frontend.bundle.Bundle.add_compute`
        * :meth:`phoebe.frontend.bundle.Bundle.run_compute`

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle` containing the system
        and datasets
    :parameter str compute: the label of the computeoptions to use (in the bundle).
        These computeoptions must have a kind of 'phoebe'.
    :parameter **kwargs: any temporary overrides to computeoptions
    :return: a list of new synthetic :class:`phoebe.parameters.parameters.ParameterSet`s
    """

    computeparams = b.get_compute(compute, force_ps=True, check_visible=False)
    hier = b.get_hierarchy()

    starrefs  = hier.get_stars()
    meshablerefs = hier.get_meshables()

    do_horizon = False #computeparams.get_value('horizon', **kwargs)

    times, infolists, new_syns = _extract_from_bundle_by_time(b, compute=compute,
                                                              times=times,
                                                              allow_oversample=True,
                                                              **kwargs)

    dynamics_method = computeparams.get_value('dynamics_method', **kwargs)
    ltte = computeparams.get_value('ltte', **kwargs)

    distance = b.get_value(qualifier='distance', context='system', unit=u.m, **kwargs)
    t0 = b.get_value(qualifier='t0', context='system', unit=u.d, **kwargs)

    if len(starrefs)==1 and computeparams.get_value('distortion_method', component=starrefs[0], **kwargs) in ['roche']:
        raise ValueError("distortion_method='{}' not valid for single star".format(computeparams.get_value('distortion_method', component=starrefs[0], **kwargs)))

    if len(meshablerefs) > 1 or hier.get_kind_of(meshablerefs[0])=='envelope':
        if dynamics_method in ['nbody', 'rebound']:
            t0, xs0, ys0, zs0, vxs0, vys0, vzs0, inst_ds0, inst_Fs0, ethetas0, elongans0, eincls0 = dynamics.nbody.dynamics_from_bundle(b, [t0], compute, return_roche_euler=True, **kwargs)
            ts, xs, ys, zs, vxs, vys, vzs, inst_ds, inst_Fs, ethetas, elongans, eincls = dynamics.nbody.dynamics_from_bundle(b, times, compute, return_roche_euler=True, **kwargs)

        elif dynamics_method == 'bs':
            # if distortion_method == 'roche':
                # raise ValueError("distortion_method '{}' not compatible with dynamics_method '{}'".format(distortion_method, dynamics_method))

            # TODO: pass stepsize
            # TODO: pass orbiterror
            # TODO: make sure that this takes systemic velocity and corrects positions and velocities (including ltte effects if enabled)
            t0, xs0, ys0, zs0, vxs0, vys0, vzs0, inst_ds0, inst_Fs0, ethetas0, elongans0, eincls0 = dynamics.nbody.dynamics_from_bundle_bs(b, [t0], compute, return_roche_euler=True, **kwargs)
            # ethetas0, elongans0, eincls0 = None, None, None
            ts, xs, ys, zs, vxs, vys, vzs, inst_ds, inst_Fs, ethetas, elongans, eincls = dynamics.nbody.dynamics_from_bundle_bs(b, times, compute, return_roche_euler=True, **kwargs)
            # ethetas, elongans, eincls = None, None, None


        elif dynamics_method=='keplerian':

            # TODO: make sure that this takes systemic velocity and corrects positions and velocities (including ltte effects if enabled)
            t0, xs0, ys0, zs0, vxs0, vys0, vzs0, ethetas0, elongans0, eincls0 = dynamics.keplerian.dynamics_from_bundle(b, [t0], compute, return_euler=True, **kwargs)
            ts, xs, ys, zs, vxs, vys, vzs, ethetas, elongans, eincls = dynamics.keplerian.dynamics_from_bundle(b, times, compute, return_euler=True, **kwargs)

        else:
            raise NotImplementedError

    # TODO: automatically guess body type for each case... based on things like whether the stars are aligned
    # TODO: handle different distortion_methods
    # TODO: skip initializing system if we NEVER need meshes
    system = universe.System.from_bundle(b, compute, datasets=b.datasets, **kwargs)


    # We need to create the mesh at periastron for any of the following reasons:
    # - protomesh
    # - volume-conservation for eccentric orbits
    # We'll assume that this is always done - so even for circular orbits, the initial mesh will just be a scaled version of this mesh
    system.initialize_meshes()

    # Now we need to compute intensities at t0 in order to scale pblums for all future times
    # TODO: only do this if we need the mesh for actual computations
    # TODO: move as much of this pblum logic into mesh.py as possible

    enabled_ps = b.filter(qualifier='enabled', compute=compute, value=True)
    kinds = enabled_ps.kinds
    datasets = enabled_ps.datasets

    if 'lc' in kinds or 'rv' in kinds:  # TODO this needs to be WAY more general

        if len(meshablerefs) > 1 or hier.get_kind_of(meshablerefs[0])=='envelope':
            x0, y0, z0, vx0, vy0, vz0, etheta0, elongan0, eincl0 = dynamics.dynamics_at_i(xs0, ys0, zs0, vxs0, vys0, vzs0, ethetas0, elongans0, eincls0, i=0)
        else:
            # singlestar case
            x0, y0, z0 = [0.], [0.], [0.]
            vx0, vy0, vz0 = [0.], [0.], [0.]
            # TODO: star needs long_an (yaw?)
            etheta0, elongan0, eincl0 = [0.], [0.], [b.get_value('incl', unit=u.rad)]

        system.update_positions(t0, x0, y0, z0, vx0, vy0, vz0, etheta0, elongan0, eincl0, ignore_effects=True)

        for dataset in datasets:
            ds = b.get_dataset(dataset=dataset).exclude(kind='*_dep')
            kind = ds.kind

            if kind not in ['lc']:
                # only LCs need pblum scaling
                continue

            # TODO: remove this for-loop... it really doesn't do anything
            # for component in ds.components:
                # if component=='_default':
                    # continue

            system.populate_observables(t0, [kind], [dataset],
                                        ignore_effects=True)

            # now for each component we need to store the scaling factor between
            # absolute and relative intensities
            pblum_copy = {}
            # for component in meshablerefs:
            for component in ds.filter(qualifier='pblum_ref').components:
                if component=='_default':
                    continue
                pblum_ref = ds.get_value(qualifier='pblum_ref', component=component)
                if pblum_ref=='self':
                    pblum = ds.get_value(qualifier='pblum', component=component)
                    ld_func = ds.get_value(qualifier='ld_func', component=component)
                    ld_coeffs = b.get_value(qualifier='ld_coeffs', component=component, dataset=dataset, context='dataset', check_visible=False)

                    # TODO: system.get_body(component) needs to be smart enough to handle primary/secondary within contact_envelope... and then smart enough to handle the pblum_scale
                    system.get_body(component).compute_pblum_scale(dataset, pblum, ld_func=ld_func, ld_coeffs=ld_coeffs, component=component)
                else:
                    # then this component wants to copy the scale from another component
                    # in the system.  We'll just store this now so that we make sure the
                    # component we're copying from has a chance to compute its scale
                    # first.
                    pblum_copy[component] = pblum_ref


            # now let's copy all the scales for those that are just referencing another component
            for comp, comp_copy in pblum_copy.items():
                pblum_scale = system.get_body(comp_copy).get_pblum_scale(dataset, component=comp_copy)
                system.get_body(comp).set_pblum_scale(dataset, component=comp, pblum_scale=pblum_scale)


#######################################################################################################################################################

    def master_populate_syns(new_syns, packetlist):
        for packet in packetlist:
            # print "*** master_populate_syns packet:", packet
            new_syns.set_value(**packet)

        return new_syns

    def worker(i, time, infolist):
        # print('work order %d received by processor %d' % (i, myrank))

        # Check to see what we might need to do that requires a mesh
        # TODO: make sure to use the requested distortion_method

        # we need to extract positions, velocities, and euler angles of ALL bodies at THIS TIME (i)
        if len(meshablerefs) > 1 or hier.get_kind_of(meshablerefs[0])=='envelope':
            xi, yi, zi, vxi, vyi, vzi, ethetai, elongani, eincli = dynamics.dynamics_at_i(xs, ys, zs, vxs, vys, vzs, ethetas, elongans, eincls, i=i)
        else:
            # then singlestar case
            xi, yi, zi = [0.], [0.], [0.]
            vxi, vyi, vzi = [0.], [0.], [0.]
            # TODO: star needs long_an (yaw?)

            ethetai, elongani, eincli = [0.], [0.], [b.get_value('incl', component=meshablerefs[0], unit=u.rad)]

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
            # masses = [b.get_value('mass', component=star, context='component', time=time, unit=u.solMass) for star in starrefs]
            # sma = b.get_value('sma', component=starrefs[body.ind_self], context='component', time=time, unit=u.solRad)

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

            expose_horizon = 'mesh' in [info['kind'] for info in infolist] and do_horizon
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

            system.populate_observables(time, populate_kinds, populate_datasets)


        # now let's loop through and prepare a packet which will fill the synthetics
        packetlist = []

        def make_packet(qualifier, value, time, info, **kwargs):
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

        for k, info in enumerate(infolist):
            packet = dict()

            # i, time, info['kind'], info['component'], info['dataset']
            cind = starrefs.index(info['component']) if info['component'] in starrefs else None
            # ts[i], xs[cind][i], ys[cind][i], zs[cind][i], vxs[cind][i], vys[cind][i], vzs[cind][i]
            kind = info['kind']

            # now check the kind to see what we need to fill
            if kind=='rv':
                ### this_syn['times'].append(time) # time array was set when initializing the syns
                if info['needs_mesh']:
                    # TODO: we have to call get here because twig access will trigger on kind=rv and qualifier=rv
                    # print "***", this_syn.filter(qualifier='rv').twigs, this_syn.filter(qualifier='rv').kinds, this_syn.filter(qualifier='rv').components
                    # if len(this_syn.filter(qualifier='rv').twigs)>1:
                        # print "***2", this_syn.filter(qualifier='rv')[1].kind, this_syn.filter(qualifier='rv')[1].component
                    rv = system.observe(info['dataset'], kind=kind, components=info['component'], distance=distance)['rv']
                else:
                    # then rv_method == 'dynamical'
                    rv = -1*vzi[cind]

                packetlist.append(make_packet('rvs',
                                              rv*u.solRad/u.d,
                                              time, info))

            elif kind=='lc':
                l3 = b.get_value(qualifier='l3', dataset=info['dataset'], context='dataset')

                packetlist.append(make_packet('fluxes',
                                              system.observe(info['dataset'], kind=kind, components=info['component'], distance=distance, l3=l3)['flux']*u.W/u.m**2,
                                              time, info))

            elif kind=='etv':

                # TODO: add support for other etv kinds (barycentric, robust, others?)
                time_ecl = etvs.crossing(b, info['component'], time, dynamics_method, ltte, tol=computeparams.get_value('etv_tol', u.d, dataset=info['dataset'], component=info['component']))

                this_obs = b.filter(dataset=info['dataset'], component=info['component'], context='dataset')

                # TODO: there must be a better/cleaner way to get to Ns
                packetlist.append(make_packet('Ns',
                                              this_obs.get_parameter(qualifier='Ns').interp_value(time_ephems=time),
                                              time, info))

                # NOTE: no longer under constraint control
                packetlist.append(make_packet('time_ephems',
                                              time,
                                              time, info))

                packetlist.append(make_packet('time_ecls',
                                              time_ecl,
                                              time, info))

                # NOTE: no longer under constraint control
                packetlist.append(make_packet('etvs',
                                              time_ecl-time,
                                              time, info))


            elif kind=='orb':
                # ts[i], xs[cind][i], ys[cind][i], zs[cind][i], vxs[cind][i], vys[cind][i], vzs[cind][i]

                # times array was set when creating the synthetic ParameterSet

                packetlist.append(make_packet('us',
                                              xi[cind],
                                              time, info))

                packetlist.append(make_packet('vs',
                                              yi[cind],
                                              time, info))

                packetlist.append(make_packet('ws',
                                              zi[cind],
                                              time, info))

                packetlist.append(make_packet('vus',
                                              vxi[cind],
                                              time, info))

                packetlist.append(make_packet('vvs',
                                              vyi[cind],
                                              time, info))

                packetlist.append(make_packet('vws',
                                              vzi[cind],
                                              time, info))

            elif kind=='mesh':
                # print "*** info['component']", info['component'], " info['dataset']", info['dataset']
                # print "*** this_syn.twigs", this_syn.twigs
                body = system.get_body(info['component'])

                packetlist.append(make_packet('uvw_elements',
                                              body.mesh.vertices_per_triangle,
                                              time, info))
                packetlist.append(make_packet('xyz_elements',
                                              body.mesh.roche_vertices_per_triangle,
                                              time, info))

                if 'pot' in info['mesh_columns']:
                    packetlist.append(make_packet('pot',
                                                  body._instantaneous_pot,
                                                  time, info))
                if 'rpole' in info['mesh_columns']:
                    packetlist.append(make_packet('rpole',
                                                  roche.potential2rpole(body._instantaneous_pot, body.q, body.ecc, body.F, body._scale, component=body.comp_no),
                                                  time, info))
                if 'volume' in info['mesh_columns']:
                    packetlist.append(make_packet('volume',
                                                  body.volume,
                                                  time, info))

                if 'us' in info['mesh_columns']:
                    # UNIT: u.solRad
                    packetlist.append(make_packet('us',
                                                  body.mesh.centers[:,0],
                                                  time, info))
                if 'vs' in info['mesh_columns']:
                    # UNIT: u.solRad
                    packetlist.append(make_packet('vs',
                                                  body.mesh.centers[:,1],
                                                  time, info))
                if 'ws' in info['mesh_columns']:
                    # UNIT: u.solRad
                    packetlist.append(make_packet('ws',
                                                  body.mesh.centers[:,2],
                                                  time, info))

                if 'vus' in info['mesh_columns']:
                    packetlist.append(make_packet('vus',
                                                  body.mesh.velocities.centers[:,0] * u.solRad/u.d,
                                                  time, info))
                if 'vvs' in info['mesh_columns']:
                    packetlist.append(make_packet('vvs',
                                                  body.mesh.velocities.centers[:,1] * u.solRad/u.d,
                                                  time, info))
                if 'vws' in info['mesh_columns']:
                    packetlist.append(make_packet('vws',
                                                  body.mesh.velocities.centers[:,2] * u.solRad/u.d,
                                                  time, info))

                # if 'uvw_normals' in info['mesh_columns']:
                #     packetlist.append(make_packet('uvw_normals',
                #                                   body.mesh.tnormals,
                #                                   time, info))

                if 'nus' in info['mesh_columns']:
                    packetlist.append(make_packet('nus',
                                                  body.mesh.tnormals[:,0],
                                                  time, info))
                if 'nvs' in info['mesh_columns']:
                    packetlist.append(make_packet('nvs',
                                                  body.mesh.tnormals[:,1],
                                                  time, info))
                if 'nws' in info['mesh_columns']:
                    packetlist.append(make_packet('nws',
                                                  body.mesh.tnormals[:,2],
                                                  time, info))


                if 'xs' in info['mesh_columns']:
                    packetlist.append(make_packet('xs',
                                                  body.mesh.roche_centers[:,0],
                                                  time, info))
                if 'ys' in info['mesh_columns']:
                    packetlist.append(make_packet('ys',
                                                  body.mesh.roche_centers[:,1],
                                                  time, info))
                if 'zs' in info['mesh_columns']:
                    packetlist.append(make_packet('zs',
                                                  body.mesh.roche_centers[:,2],
                                                  time, info))

                if 'vxs' in info['mesh_columns']:
                    packetlist.append(make_packet('vxs',
                                                  body.mesh.roche_cvelocities[:,0] * u.solRad/u.d,
                                                  time, info))
                if 'vys' in info['mesh_columns']:
                    packetlist.append(make_packet('vys',
                                                  body.mesh.roche_cvelocities[:,1] * u.solRad/u.d,
                                                  time, info))
                if 'vzs' in info['mesh_columns']:
                    packetlist.append(make_packet('vzs',
                                                  body.mesh.roche_cvelocities[:,2] * u.solRad/u.d,
                                                  time, info))

                # if 'xyz_normals' in info['mesh_columns']:
                #     packetlist.append(make_packet('xyz_normals',
                #                                   body.mesh.tnormals,
                #                                   time, info))

                if 'nxs' in info['mesh_columns']:
                    packetlist.append(make_packet('nxs',
                                                  body.mesh.roche_tnormals[:,0],
                                                  time, info))
                if 'nys' in info['mesh_columns']:
                    packetlist.append(make_packet('nys',
                                                  body.mesh.roche_tnormals[:,1],
                                                  time, info))
                if 'nzs' in info['mesh_columns']:
                    packetlist.append(make_packet('nzs',
                                                  body.mesh.roche_tnormals[:,2],
                                                  time, info))


                if 'areas' in info['mesh_columns']:
                    # UNIT: u.solRad**2
                    packetlist.append(make_packet('areas',
                                                  body.mesh.areas,
                                                  time, info))
                # if 'tareas' in info['mesh_columns']:
                    # packetlist.append(make_packet('tareas',
                                                  # body.mesh.tareas,
                                                  # time, info))




                if 'rs' in info['mesh_columns']:
                    packetlist.append(make_packet('rs',
                                                  body.mesh.rs.centers,
                                                  time, info))
                # if 'cosbetas' in info['mesh_columns']:
                    # packetlist.append(make_packet('cosbetas',
                                                  # body.mesh.cosbetas,
                                                  # time, info))


                if 'loggs' in info['mesh_columns']:
                    packetlist.append(make_packet('loggs',
                                                  body.mesh.loggs.centers,
                                                  time, info))
                if 'teffs' in info['mesh_columns']:
                    packetlist.append(make_packet('teffs',
                                                  body.mesh.teffs.centers,
                                                  time, info))


                if 'r_projs' in info['mesh_columns']:
                    packetlist.append(make_packet('r_projs',
                                                  body.mesh.rprojs.centers,
                                                  time, info))
                if 'mus' in info['mesh_columns']:
                    packetlist.append(make_packet('mus',
                                                  body.mesh.mus,
                                                  time, info))
                if 'visibilities' in info['mesh_columns']:
                    packetlist.append(make_packet('visibilities',
                                                  body.mesh.visibilities,
                                                  time, info))
                if 'visible_centroids' in info['mesh_columns']:
                    vcs = np.sum(body.mesh.vertices_per_triangle*body.mesh.weights[:,:,np.newaxis], axis=1)
                    for i,vc in enumerate(vcs):
                        if np.all(vc==np.array([0,0,0])):
                            vcs[i] = np.full(3, np.nan)

                    packetlist.append(make_packet('visible_centroids',
                                                  vcs,
                                                  time, info))


                # Eclipse horizon
                # if do_horizon and horizons is not None:
                    # packet[k]['horizon_xs'] = horizons[cind][:,0]
                    # packet[k]['horizon_ys'] = horizons[cind][:,1]
                    # packet[k]['horizon_zs'] = horizons[cind][:,2]

                # Analytic horizon
                # if do_horizon:
                #     if body.distortion_method == 'roche':
                #         if body.mesh_method == 'marching':
                #             q, F, d, Phi = body._mesh_args
                #             scale = body._scale
                #             euler = [ethetai[cind], elongani[cind], eincli[cind]]
                #             pos = [xi[cind], yi[cind], zi[cind]]
                #             ha = horizon_analytic.marching(q, F, d, Phi, scale, euler, pos)
                #         elif body.mesh_method == 'wd':
                #             scale = body._scale
                #             pos = [xi[cind], yi[cind], zi[cind]]
                #             ha = horizon_analytic.wd(b, time, scale, pos)
                #         else:
                #             raise NotImplementedError("analytic horizon not implemented for mesh_method='{}'".format(body.mesh_method))
                #
                #         packet[k]['horizon_analytic_xs'] = ha['xs']
                #         packet[k]['horizon_analytic_ys'] = ha['ys']
                #         packet[k]['horizon_analytic_zs'] = ha['zs']

                # Dataset-dependent quantities
                for mesh_dataset in info['mesh_datasets']:
                    if 'pblum@{}'.format(mesh_dataset) in info['mesh_columns']:
                        packetlist.append(make_packet('pblum',
                                                      body.compute_luminosity(mesh_dataset),
                                                      time, info,
                                                      dataset=mesh_dataset,
                                                      component=info['component']))

                    if 'ptfarea@{}'.format(mesh_dataset) in info['mesh_columns']:
                        packetlist.append(make_packet('ptfarea',
                                                      body.get_ptfarea(mesh_dataset),
                                                      time, info,
                                                      dataset=mesh_dataset,
                                                      component=info['component']))

                    for indep in ['rvs', 'intensities', 'normal_intensities',
                                  'abs_intensities', 'abs_normal_intensities',
                                  'boost_factors', 'ldint']:

                        if "{}@{}".format(indep, mesh_dataset) in info['mesh_columns']:
                            key = "{}:{}".format(indep, mesh_dataset)

                            value = body.mesh[key].centers

                            packetlist.append(make_packet(indep,
                                                          value,
                                                          time, info,
                                                          dataset=mesh_dataset,
                                                          component=info['component']))

            else:
                raise NotImplementedError("kind {} not yet supported by this backend".format(kind))

        return packetlist


    if _use_mpi and not conf.force_serial:
        if myrank == 0:
            # then this is the master process which is responsible for sending
            # jobs to the workers and processing the returned packets to fill
            # the synthetic parameters

            # receive the packet from each time sent by a worker
            req = [0]*len(times)
            for i in range(len(times)):
                req[i] = comm.irecv(source=MPI.ANY_SOURCE, tag=TAG_DATA)

            # send tasks to the workers
            # this is the main compute loop in MPI mode
            for i,time,infolist in zip(range(len(times)),times,infolists):
                node = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_REQ)
                packet = {'i': i, 'time': time, 'infolist': infolist}
                comm.send(packet, node, tag=TAG_DATA)

            # send kill command to all workers
            for i in range(1, nprocs):
                node = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_REQ)
                comm.send({'i': -1}, node, tag=TAG_DATA)

            # parse and process the received packets to fill the syns
            for i in range(len(req)):
                r = req[i].wait()

                i = r['i']
                packetlist = r['packet']

                new_syns = master_populate_syns(new_syns, packetlist)

                if as_generator:
                    # this is mainly used for live-streaming animation support
                    yield (new_syns, times[i])

            if not as_generator:
                yield new_syns

        else: # if myrank != 0:
            # then this is a worker processor, which must receive a job request
            # from the master and return the results
            while True:
                # tell the master that the worker is ready for another task
                comm.send(myrank, 0, tag=TAG_REQ)
                # receive the next job
                packet = comm.recv(tag=TAG_DATA)

                i = packet['i']
                if i == -1:
                    # then all jobs are complete, so kill the worker process
                    # by exiting the while loop
                    break

                # parse the packet and run computations for this single time
                time = packet['time']
                infolist = packet['infolist']

                packet = worker(i, time, infolist)

                # return the result packet to the master
                comm.send({'i': i, 'packet': packet}, 0, tag=TAG_DATA)

            yield ParameterSet([])
    else:
        # not _use_mpi
        # this is the main compute loop in serial mode
        req = [0]*len(times)
        for i,time,infolist in zip(range(len(times)),times,infolists):
            packetlist = worker(i, time, infolist)

            # In serial mode we will process the returned packet per-time
            # so that we can immediately yield the updated synthetics.
            # This may be a slight overhead, but allows us to stream the results
            # for live-updating plots (for example)
            new_syns = master_populate_syns(new_syns, packetlist)

            if as_generator:
                # this is mainly used for live-streaming animation support
                yield (new_syns, times[i])

        if not as_generator:
            yield new_syns




def legacy(b, compute, times=[], **kwargs): #, **kwargs):#(b, compute, **kwargs):

    """
    Use PHOEBE 1.0 (legacy) which is based on the Wilson-Devinney code
    to compute radial velocities and light curves for binary systems
    (>2 stars not supported).  The code is available here:

    http://phoebe-project.org/1.0

    PHOEBE 1.0 and the 'phoebeBackend' python interface must be installed
    and available on the system in order to use this plugin.

    When using this backend, please cite
        * Prsa & Zwitter (2005), ApJ, 628, 426

    Parameters that are used by this backend:

    * Compute:
        * all parameters in :func:`phoebe.parameters.compute.legacy`
    * Orbit:
        * TOOD: list these
    * Star:
        * TODO: list these
    * lc dataset:
        * TODO: list these

    Values that are filled by this backend:

    * lc:
        * times
        * fluxes
    * rv (dynamical only):
        * times
        * rvs

    This function will almost always be called through the bundle, using
        * :meth:`phoebe.frontend.bundle.Bundle.add_compute`
        * :meth:`phoebe.frontend.bundle.Bundle.run_compute`

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle` containing the system
        and datasets
    :parameter str compute: the label of the computeoptions to use (in the bundle).
        These computeoptions must have a kind of 'legacy'.
    :parameter **kwargs: any temporary overrides to computeoptions
    :return: a list of new synthetic :class:`phoebe.parameters.parameters.ParameterSet`s

    """


    """

    build up keys for each phoebe1 parameter, so that they correspond
    to the correct phoebe2 parameter. Since many values of x, y, and z
    it is unecessary to make specific dictionary items for them. This
    function does this and also makes sure the correct component is
    being called.

    Args:
        key: The root of the key word used in phoebe1: component: the
        name of the phoebe2 component associated with the given mesh.

    Returns:
        new key(s) which are component specific and vector direction
        specific (xs, ys, zs) for phoebe1 and phoebe2.

    Raises:
        ImportError if the python 'phoebeBackend' interface to PHOEBE
        legacy is not found/installed



    """
    p2to1 = {'tloc':'teffs', 'glog':'loggs', 'vcx':'xs', 'vcy':'ys', 'vcz':'zs', 'grx':'nxs', 'gry':'nys', 'grz':'nzs', 'csbt':'cosbetas', 'rad':'rs','Inorm':'abs_normal_intensities'}

    def ret_dict(key, stars):
        """
        Build up dictionary for each phoebe1 parameter, so that they
        correspond to the correct phoebe2 parameter.
        Args:
            key: The root of the key word used in phoebe1:
            component: the name of the phoebe2 component associated with
            the given mesh.

        Returns:
            dictionary of values which should be unique to a single
            parameter in phoebe 2.


        """
        d= {}
        comp = int(key[-1])
        key = key[:-1]
        #determine component
        if comp == 1:
            # TODO: is this hardcoding component names?  We should really access
            # from the hierarchy instead (we can safely assume a binary) by doing
            # b.hierarchy.get_stars() and b.hierarchy.get_primary_or_secondary()
            d['component'] = stars[0]
        elif comp== 2:
            d['component'] = stars[1]
        else:
            #This really shouldn't happen
            raise ValueError("All mesh keys should be component specific.")
        try:
            d['qualifier'] = p2to1[key]
        except:
            d['qualifier'] = key
        if key == 'Inorm':
             d['unit'] = u.erg*u.s**-1*u.cm**-3
        return d


    def fill_mesh(mesh, type, stars, time=None):
        """
        Fill phoebe2 mesh with values from phoebe1

        Args:
            key: Phoebe1 mesh for all time points
            type: mesh type "protomesh" or "pbmesh"
            time: array of times (only applicable for pbmesh)
        Returns:
            None

        Raises:
            ValueError if the anything other than pbmesh or protomesh is given for type.
        """
        keys = mesh.keys()

        if type == 'protomesh':
            grx1 = np.array(mesh['grx1'])
            gry1 = np.array(mesh['gry1'])
            grz1 = np.array(mesh['grz1'])
            # TODO: rewrite this to use np.linalg.norm
            grtot1 = grx1**2+gry1**2+grz1**2
            grx2 = np.array(mesh['grx2'])
            gry2 = np.array(mesh['gry2'])
            grz2 = np.array(mesh['grz2'])
            grtot2 = grx2**2+gry2**2+grz2**2
            grtot = [np.sqrt(grtot1),np.sqrt(grtot2)]

        for key in keys:
            d = ret_dict(key, stars)
     #       key_values =  np.array_split(mesh[key],n)
            if type == 'protomesh':
                # take care of the protomesh
                prot_val = np.array(mesh[key])#key_values[-1]

                d['dataset'] = 'protomesh'
                if ('vcx' in key) or ('grx' in key):
                    key_val = np.array(zip(prot_val, prot_val, prot_val, prot_val)).flatten()#, prot_val, prot_val, prot_val)).flatten()#, -prot_val, -prot_val, -prot_val, -prot_val)).flatten()

                elif ('vcy' in key) or ('gry' in key):
                    key_val = np.array(zip(prot_val, -1.0*prot_val,prot_val, -1.0*prot_val)).flatten()#, -prot_val, -prot_val, prot_val)).flatten()#, prot_val, -prot_val, -prot_val, prot_val)).flatten()

                elif ('vcz' in key) or ('grz' in key):
                    key_val = np.array(zip(prot_val, -1.0*prot_val, -1.0*prot_val, prot_val)).flatten()#, prot_val, -prot_val, -prot_val)).flatten()#, prot_val, -prot_val, -prot_val)).flatten()

                else:
                    key_val = np.array(zip(prot_val, prot_val, prot_val, prot_val)).flatten()
                #     grtotn = grtot[int(key[-1])-1]

                #     grtotn = np.array(zip(grtotn, grtotn, grtotn, grtotn, grtotn, grtotn, grtotn, grtotn)).flatten()

                # if 'vcx' or 'grx' in keyot_val, -prot_val, -prot_val, -prot_val)).flatten()
                # if 'vcy' or 'gry' in key:
                #     key_val = np.array(zip(prot_val, -prot_val, -prot_val, prot_val, prot_val, -prot_val, -prot_val, prot_val)).flatten()
                # if 'vcz' or 'grz' in key:
                #     key_val = np.array(zip(prot_val, prot_val, -prot_val, -prot_val, prot_val, prot_val, -prot_val, -prot_val)).flatten()
                # else:
                #     key_val = np.array(zip(prot_val, prot_val, prot_val, prot_val, prot_val, prot_val, prot_val, prot_val)).flatten()
                if key[:2] =='gr':
                    grtotn = grtot[int(key[-1])-1]

                    grtotn = np.array(zip(grtotn, grtotn, grtotn, grtotn)).flatten()#, grtotn, grtotn, grtotn)).flatten()#, grtotn, grtotn, grtotn, grtotn)).flatten()

                    # normals should be normalized
                    d['value'] = -key_val /grtotn
                else:
                    d['value'] = key_val
                         #TODO fill the normals column it is just (nx, ny, nz)

                try:
                    new_syns.set_value(**d)
                except:
                    logger.warning('{} has no corresponding value in phoebe 2 protomesh'.format(key))

            elif type == 'pbmesh':
                logger.warning('Only values which do not depend on the stars location are currently reported.')
                n = len(time)
                key_values =  np.array_split(mesh[key],n)
                #TODO change time inserted to time = time[:-1]
                for t in range(len(time)):
                # d = ret_dict(key)
                    d['dataset'] = 'pbmesh'
                    if key in ['Inorm1', 'Inorm2']:
                        d['dataset'] = dataset

                        d['time'] = time[t]
                        #prepare data
                        if key[:2] in ['vc', 'gr']:
                            # I need to change coordinates but not yet done
                            pass

                            #TODO Change these values so that they are placed in orbit

                        else:
                            key_val= np.array(key_values[t])
                            key_val = np.array(zip(key_val, key_val, key_val, key_val, key_val, key_val, key_val, key_val)).flatten()

                            param = new_syns.filter(**d)
                            if param:

                                d['value'] = key_val
                                new_syns.set_value(**d)
                            else:
                                logger.warning('{} has no corresponding value in phoebe 2 pbmesh'.format(key))
            else:
                raise ValueError("Only 'pbmesh' and 'protomesh' are acceptable mesh types.")


        return

    # check whether phoebe legacy is installed
    if not _use_phb1:
        raise ImportError("phoebeBackend for phoebe legacy not found")

    computeparams = b.get_compute(compute, force_ps=True)
#    computeparams = b.get_compute(compute, force_ps=True)
#    hier = b.get_hierarchy()

#    starrefs  = hier.get_stars()
#    orbitrefs = hier.get_orbits()

    stars = b.hierarchy.get_stars()
    primary, secondary = stars
    #need for protomesh
    perpass = b.get_value(qualifier='t0_perpass', kind='orbit', context='component')
    # print primary, secondary
    #make phoebe 1 file


    #create temporary file
    tmp_file = tempfile.NamedTemporaryFile()
#   testing
 #   filename = 'check.phoebe'
#   real
    io.pass_to_legacy(b, filename=tmp_file.name, compute=compute, **kwargs)
#   testing
#    filename = 'check.phoebe'
#    io.pass_to_legacy(b, filename=filename, compute=compute, **kwargs)
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
#   real
    phb1.open(tmp_file.name)
#   testing
#    phb1.open(filename)
#    phb1.updateLD()
    # TODO BERT: why are we saving here?
#   testing
 #   phb1.save('after.phoebe')
    lcnum = 0
    rvnum = 0
    rvid = None
    infos, new_syns = _extract_from_bundle_by_dataset(b, compute=compute, times=times)


#    print "INFOS", len(infos)
#    print "info 1",  infos[0]
#    print "info 2-1",  infos[0][1]
#    print "info 3-1",  infos[0][2]
#    quit()
    if protomesh:
        time = [perpass]
        # print 'TIME', time
#        rlcno = phb1.getpar('phoebe_lcno')
#        phb1.setpar('phoebe_lcno', 1)
        flux, mesh = phb1.lc(tuple(time), 0, lcnum+1)
        fill_mesh(mesh, 'protomesh', stars)
#        phb1.setpar('phoebe_lcno', rlcno)

    for info in infos:
        info = info[0]
        this_syn = new_syns.filter(component=info['component'], dataset=info['dataset'])
        time = info['times']
        dataset=info['dataset']

        if info['kind'] == 'lc':
            if not pbmesh:
                # print "lcnum", lcnum
            # print "*********************", this_syn.qualifiers
                flux= np.array(phb1.lc(tuple(time.tolist()), lcnum))
                lcnum = lcnum+1
                #get rid of the extra periastron passage
                this_syn['fluxes'] = flux

            else:
            #    time = np.append(time, perpass)
            #    print "TIME", time, perpass
                flux, mesh = phb1.lc(tuple(time.tolist()), 0, lcnum+1)
                flux = np.array(flux)
            # take care of the lc first
                this_syn['fluxes'] = flux

                fill_mesh(mesh, 'pbmesh', stars, time=time)
            # now deal with parameters
    #            keys = mesh.keys()
    #            n = len(time)


            # calculate the normal 'magnitude' for normalizing vectors
#                 grx1 = np.array_split(mesh['grx1'],n)[-1]
#                 gry1 = np.array_split(mesh['gry1'],n)[-1]
#                 grz1 = np.array_split(mesh['grz1'],n)[-1]
#                 # TODO: rewrite this to use np.linalg.norm
#                 grtot1 = grx1**2+gry1**2+grz1**2
#                 grx2 = np.array_split(mesh['grx1'],n)[-1]
#                 gry2 = np.array_split(mesh['gry1'],n)[-1]
#                 grz2 = np.array_split(mesh['grz1'],n)[-1]
#                 grtot2 = grx2**2+gry2**2+grz2**2
#                 grtot = [np.sqrt(grtot1),np.sqrt(grtot2)]
#                 for key in keys:
#                     key_values =  np.array_split(mesh[key],n)
#                     # take care of the protomesh
#                     prot_val = key_values[-1]
#                     d = ret_dict(key)
#                     d['dataset'] = 'protomesh'
#                     key_val = np.array(zip(prot_val, prot_val, prot_val, prot_val, prot_val, prot_val, prot_val, prot_val)).flatten()
#                     if key[:2] =='gr':
#                         grtotn = grtot[int(key[-1])-1]

#                         grtotn = np.array(zip(grtotn, grtotn, grtotn, grtotn, grtotn, grtotn, grtotn, grtotn)).flatten()

#                         # normals should be normalized
#                         d['value'] = -key_val /grtotn
#                     else:
#                         d['value'] = key_val
#                     #TODO fill the normals column it is just (nx, ny, nz)

#                     try:
#                         new_syns.set_value(**d)
#                     except:
#                         logger.warning('{} has no corresponding value in phoebe 2 protomesh'.format(key))

#                     #Normalize the normals that have been put in protomesh

#                     # now take care of pbmesh time point by time point
#                     for t in range(len(time[:-1])):
# #                        d = ret_dict(key)
#                         d['dataset'] = 'pbmesh'
#                         if key in ['Inorm1', 'Inorm2']:
#                             d['dataset'] = dataset

#                         d['times'] = time[t]
#                     #prepare data
#                         if key[:2] in ['vc', 'gr']:
#                             # I need to change coordinates but not yet done
#                             pass

#                             #TODO Change these values so that they are placed in orbit

#                         else:
#                             key_val= key_values[t]
#                             key_val = np.array(zip(key_val, key_val, key_val, key_val, key_val, key_val, key_val, key_val)).flatten()

#                             param = new_syns.filter(**d)
#                             if param:
#                                 d['value'] = key_val
#                                 new_syns.set_value(**d)
#                             else:
#                                 logger.warning('{} has no corresponding value in phoebe 2 pbmesh'.format(key))

#                 time = time[:-1]

        elif info['kind'] == 'rv':

            if rvid == None:
                dep =  phb1.getpar('phoebe_rv_dep', 0)
            else:
                dep =  phb1.getpar('phoebe_rv_dep', 1)
            dep = dep.split(' ')[0].lower()

            rvid = info['dataset']

            if dep == 'primary':
                comp = primary
            elif dep == 'secondary':
                comp = secondary

            proximity = computeparams.filter(qualifier ='rv_method', component=comp, dataset=rvid).get_value()
            if proximity == 'flux-weighted':
                rveffects = 1
            else:
                rveffects = 0

            if dep == 'primary':
                # print 'primary'
                phb1.setpar('phoebe_proximity_rv1_switch', rveffects)
                rv = np.array(phb1.rv1(tuple(time.tolist()), 0))
                rvnum = rvnum+1

            elif dep == 'secondary':
                # print 'secondary'
                phb1.setpar('phoebe_proximity_rv2_switch', rveffects)
                rv = np.array(phb1.rv2(tuple(time.tolist()), 0))
                rvnum = rvnum+1
            else:
                raise ValueError(str(info['component'])+' is not the primary or the secondary star')


                 #print "***", u.solRad.to(u.km)
            this_syn.set_value(qualifier='rvs', value=rv*u.km/u.s)
#########################################################################################################
#            if rvid == phb1.getpar('phoebe_rv_id', 0):

#                dep =  phb1.getpar('phoebe_rv_dep', 0)
#                dep = dep.split(' ')[0].lower()
           # must account for rv datasets with multiple components
#                if dep == 'primary':
#                    component = primary
#                elif dep == 'secondary':
#                    component = secondary


#            elif rvid == phb1.getpar('phoebe_rv_id', 1):
#                dep =  phb1.getpar('phoebe_rv_dep', 1)
#                dep = dep.split(' ')[0].lower()
           # must account for rv datasets with multiple components
#                if dep != info['component']:
#                    dep = info['component']

#            proximity = computeparams.filter(qualifier ='rv_method', component=component, dataset=rvid).get_value()

#            if proximity == 'flux-weighted':
#                rveffects = 1
#            else:
#                rveffects = 0
#            try:
#                dep2 =  phb1.getpar('phoebe_rv_dep', 1)
#                dep2 = dep2.split(' ')[0].lower()
#            except:
#                dep2 = None
#            print "dep", dep
#            print "dep2", dep2
#            print "COMPONENT", info['component']
#            if dep == 'primary':
#                phb1.setpar('phoebe_proximity_rv1_switch', rveffects)
#                rv = np.array(phb1.rv1(tuple(time.tolist()), 0))
#                rvnum = rvnum+1

#            elif dep == 'secondary':
#                phb1.setpar('phoebe_proximity_rv2_switch', rveffects)
#                rv = np.array(phb1.rv2(tuple(time.tolist()), 0))

#                rvnum = rvnum+1
#            else:
#                raise ValueError(str(info['component'])+' is not the primary or the secondary star')


            #print "***", u.solRad.to(u.km)
#            this_syn.set_value(qualifier='rvs', value=rv*u.km/u.s, component = component)
#            print "INFO", info
#            print "SYN", this_syn

            #print 'THIS SYN', this_syn

        elif info['kind']=='mesh':
            pass
#            print "I made it HERE"
#        if info['kind'] == 'mesh':
#            meshcol = {'tloc':'teff', 'glog':'logg','gr':'_o_normal_', 'vc':'_o_center' }

#            keys = ['tloc', 'glog']#, 'gr', 'vc']
#            n = len(time)
#            for i in keys:
#                p1keys, p2keys = par_build(i, info['component'])

#               for k in range(p1keys):
               # get parameter and copy because phoebe1 only does a quarter hemisphere.
#                    parn =  np.array_split(mesh[k],n)
#                    parn = np.array(zip(parn, parn, parn, parn, parn, parn, parn, parn)).flatten()

               # copy into correct location in phoebe2

                    #pary =  np.array_split(mesh[keyny],n)
                    #parz =  np.array_split(mesh[keynz],n)


#                for j in range(len(time)):

#                if i == 'gr' or i == 'vc':

#                    xd = i+'x'
#                    yd = i+'y'
#                    zd = i+'z'

#                par1x, par2x, parx = par_mesh(omesh, xd)
#                par1y, par2y, pary = par_mesh(omesh, yd)
#                par1z, par2z, parz = par_mesh(omesh, zd)

#                this_syn[meshcol[i]] = body.mesh[meshcol[i]]
#                this_syn['teff'] = body.mesh['teff']




    yield new_syns

def photodynam(b, compute, times=[], **kwargs):
    """
    Use Josh Carter's photodynamical code (photodynam) to compute
    velocities (dynamical only), orbital positions and velocities
    (center of mass only), and light curves (assumes spherical stars).
    The code is available here:

    https://github.com/dfm/photodynam

    photodynam must be installed and available on the system in order to
    use this plugin.

    Please cite both

    * Science 4 February 2011: Vol. 331 no. 6017 pp. 562-565 DOI:10.1126/science.1201274
    * MNRAS (2012) 420 (2): 1630-1635. doi: 10.1111/j.1365-2966.2011.20151.x

    when using this code.

    Parameters that are used by this backend:

    * Compute:
        - all parameters in :func:`phoebe.parameters.compute.photodynam`

    * Orbit:
        - sma
        - ecc
        - incl
        - per0
        - long_an
        - t0_perpass

    * Star:
        - mass
        - radius

    * lc dataset:
        - pblum
        - ld_coeffs (if ld_func=='linear')

    Values that are filled by this backend:

    * lc:
        - times
        - fluxes

    * rv (dynamical only):
        - times
        - rvs

    This function will almost always be called through the bundle, using
        * :meth:`phoebe.frontend.bundle.Bundle.add_compute`
        * :meth:`phoebe.frontend.bundle.Bundle.run_compute`

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle` containing the system
        and datasets
    :parameter str compute: the label of the computeoptions to use (in the bundle).
        These computeoptions must have a kind of 'photodynam'.
    :parameter **kwargs: any temporary overrides to computeoptions
    :return: a list of new synthetic :class:`phoebe.parameters.parameters.ParameterSet`s
    :raises ImportError: if the photodynam executable cannot be found or is not installed
    :raises ValueError: if pblums are invalid
    """
    # check whether photodynam is installed
    out = commands.getoutput('photodynam')
    if 'not found' in out:
        raise ImportError('photodynam executable not found')

    computeparams = b.get_compute(compute, force_ps=True)
    hier = b.get_hierarchy()

    starrefs  = hier.get_stars()
    orbitrefs = hier.get_orbits()

    infos, new_syns = _extract_from_bundle_by_dataset(b, compute=compute, times=times)

    step_size = computeparams.get_value('stepsize', **kwargs)
    orbit_error = computeparams.get_value('orbiterror', **kwargs)
    time0 = b.get_value(qualifier='t0', context='system', unit=u.d, **kwargs)

    for info in infos:
        info = info[0] # TODO: make sure this is an ok assumption

        # write the input file
        fi = open('_tmp_pd_inp', 'w')
        fi.write('{} {}\n'.format(len(starrefs), time0))
        fi.write('{} {}\n'.format(step_size, orbit_error))
        fi.write('\n')
        fi.write(' '.join([str(b.get_value('mass', component=star,
                context='component', unit=u.solMass) * c.G.to('AU3 / (Msun d2)').value)
                for star in starrefs])+'\n') # GM

        fi.write(' '.join([str(b.get_value('rpole', component=star,
                context='component', unit=u.AU))
                for star in starrefs])+'\n')

        if info['kind'] == 'lc':
            pblums = [b.get_value(qualifier='pblum', component=star,
                    context='dataset', dataset=info['dataset'])
                    for star in starrefs]  # TODO: units or unitless?
            u1s, u2s = [], []
            for star in starrefs:
                if b.get_value(qualifier='ld_func', component=star, dataset=info['dataset'], context='dataset') == 'quadratic':
                    ld_coeffs = b.get_value(qualifier='ld_coeffs', component=star, dataset=info['dataset'], context='dataset')
                else:
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
            a = b.get_value('sma', component=orbitref,
                context='component', unit=u.AU)
            e = b.get_value('ecc', component=orbitref,
                context='component')
            i = b.get_value('incl', component=orbitref,
                context='component', unit=u.rad)
            o = b.get_value('per0', component=orbitref,
                context='component', unit=u.rad)
            l = b.get_value('long_an', component=orbitref,
                context='component', unit=u.rad)

            # t0 = b.get_value('t0_perpass', component=orbitref,
                # context='component', unit=u.d)
            # period = b.get_value('period', component=orbitref,
                # context='component', unit=u.d)

            # om = 2 * np.pi * (time0 - t0) / period
            om = b.get_value('mean_anom', component=orbitref,
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

        for t in b.get_value('times', component=info['component'], dataset=info['dataset'], context='dataset', unit=u.d):
            fr.write('{}\n'.format(t))
        fr.close()

        # run photodynam
        cmd = 'photodynam _tmp_pd_inp _tmp_pd_rep > _tmp_pd_out'
        logger.info("running photodynam backend: '{}'".format(cmd))
        out = commands.getoutput(cmd)
        stuff = np.loadtxt('_tmp_pd_out', unpack=True)

        # parse output to fill syns
        this_syn = new_syns.filter(component=info['component'], dataset=info['dataset'])

        nbodies = len(starrefs)
        if info['kind']=='lc':
            this_syn['times'] = stuff[0] * u.d
            this_syn['fluxes'] = stuff[1] # + 1  # TODO: figure out why and actually fix the problem instead of fudging it!?!!?
        elif info['kind']=='orb':
            cind = starrefs.index(info['component'])
            this_syn['times'] = stuff[0] * u.d
            this_syn['xs'] = -1*stuff[2+(cind*3)] * u.AU
            this_syn['ys'] = -1*stuff[3+(cind*3)] * u.AU
            this_syn['zs'] = stuff[4+(cind*3)] * u.AU
            this_syn['vxs'] = -1*stuff[3*nbodies+2+(cind*3)] * u.AU/u.d
            this_syn['vys'] = -1*stuff[3*nbodies+3+(cind*3)] * u.AU/u.d
            this_syn['vzs'] = stuff[3*nbodies+4+(cind*3)] * u.AU/u.d
        elif info['kind']=='rv':
            cind = starrefs.index(info['component'])
            this_syn['times'] = stuff[0] * u.d
            this_syn['rvs'] = -stuff[3*nbodies+4+(cind*3)] * u.AU/u.d
        else:
            raise NotImplementedError("kind {} not yet supported by this backend".format(kind))

    yield new_syns

def jktebop(b, compute, times=[], **kwargs):
    """
    Use John Southworth's code (jktebop) to compute radial velocities
    and light curves.  The code is available here:

    http://www.astro.keele.ac.uk/jkt/codes/jktebop.html

    jktebop must be installed and available on the system in order to
    use this plugin.

    Please see the link above for a list of publications to cite when using this
    code.

    According to jktebop's website:

        JKTEBOP models the two components as biaxial spheroids for the
        calculation of the reflection and ellipsoidal effects,
        and as spheres for the eclipse shapes.

    Note that the wrapper around jktebop only uses its forward model.
    Jktebop also includes its own fitting kinds, including bootstrapping.
    Those capabilities cannot be accessed from PHOEBE.

    Parameters that are used by this backend:

    * Compute:
        -  all parameters in :func:`phoebe.parameters.compute.jktebop`

    * Orbit:
        - TODO: list these

    * Star:
        - TODO: list these

    * lc dataset:
        - TODO :list these

    Values that are filled by this backend:

    * lc:
        - times
        - fluxes

    * rv (dynamical only):
        - times
        - rvs

    This function will almost always be called through the bundle, using
        * :meth:`phoebe.frontend.bundle.Bundle.add_compute`
        * :meth:`phoebe.frontend.bundle.Bundle.run_compute`

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle` containing the system
        and datasets
    :parameter str compute: the label of the computeoptions to use (in the bundle).
        These computeoptions must have a kind of 'jktebop'.
    :parameter **kwargs: any temporary overrides to computeoptions
    :return: a list of new synthetic :class:`phoebe.parameters.parameters.ParameterSet`s
    :raise ImportError: if the jktebop executable cannot be found or is not installed
    :raises ValueError: if an ld_func is not valid for the jktebop backedn
    """

    # check whether jktebop is installed
    out = commands.getoutput('jktebop')
    if 'not found' in out:
        raise ImportError('jktebop executable not found')

    computeparams = b.get_compute(compute, force_ps=True)
    hier = b.get_hierarchy()

    starrefs  = hier.get_stars()
    orbitrefs = hier.get_orbits()

    if len(starrefs) != 2 or len(orbitrefs) != 1:
        raise ValueError("jktebop backend only accepts binary systems")

    logger.warning("JKTEBOP backend is still in development/testing and is VERY experimental")

    orbitref = orbitrefs[0]

    infos, new_syns = _extract_from_bundle_by_dataset(b, compute=compute, times=times)  # TODO: add rv support (see commented context below)

    ringsize = computeparams.get_value('ringsize', unit=u.deg, **kwargs)

    rA = b.get_value('rpole', component=starrefs[0], context='component', unit=u.solRad)
    rB = b.get_value('rpole', component=starrefs[1], context='component', unit=u.solRad)
    sma = b.get_value('sma', component=orbitref, context='component', unit=u.solRad)
    incl = b.get_value('incl', component=orbitref, context='component', unit=u.deg)
    q = b.get_value('q', component=orbitref, context='component')
    ecosw = b.get_value('ecosw', component=orbitref, context='component')
    esinw = b.get_value('esinw', component=orbitref, context='component')

    gravbA = b.get_value('gravb_bol', component=starrefs[0], context='component')
    gravbB = b.get_value('gravb_bol', component=starrefs[1], context='component')


    period = b.get_value('period', component=orbitref, context='component', unit=u.d)
    t0_supconj = b.get_value('t0_supconj', component=orbitref, context='component', unit=u.d)

    for info in infos:
        info = info[0]

        # get dataset-dependent things that we need
        l3 = b.get_value('l3', dataset=info['dataset'], context='dataset')
        # TODO: need to sum up pblums of each component - so need to write a function which will use the phoebe2 backend
        # to compute pblums that are coupled (or they need to be computed as constraints - I guess we'll see how fast that function runs)
        try:
            pblum = sum([b.get_value('pblum', dataset=info['dataset'], component=starref, context='dataset') for starref in starrefs])  # TODO: supposed to be in mags?
        except:
            raise ValueError("jktebop backend currently only supports decoupled pblums (b.set_value_all('pblum_ref', 'self'))")

        logger.warning("pblum in jktebop is sum of pblums (per-component): {}".format(pblum))
        pblum = -2.5 * np.log10(pblum) + 0.0

        ldfuncA = b.get_value('ld_func', component=starrefs[0], dataset=info['dataset'], context='dataset')
        ldfuncB = b.get_value('ld_func', component=starrefs[1], dataset=info['dataset'], context='dataset')

        ldcoeffsA = b.get_value('ld_coeffs', component=starrefs[0], dataset=info['dataset'], context='dataset')
        ldcoeffsB = b.get_value('ld_coeffs', component=starrefs[1], dataset=info['dataset'], context='dataset')

        if len(ldcoeffsA) != 2:
            logger.warning("ld_coeffs not compatible with jktebop - setting to (0.5,0.5)")
            ldcoeffsA = (0.5,0.5)
        if len(ldcoeffsB) != 2:
            logger.warning("ld_coeffs not compatible with jktebop - setting to (0.5,0.5)")
            ldcoeffsB = (0.5,0.5)

        albA = b.get_value('irrad_frac_refl_bol', component=starrefs[0], context='component')
        albB = b.get_value('irrad_frac_refl_bol', component=starrefs[1], context='component')

        tratio = b.get_value('teff', component=starrefs[0], context='component', unit=u.K) / b.get_value('teff', component=starrefs[1], context='component', unit=u.K)

        # provide translation from phoebe's 'ld_func' to jktebop's 'LD law type'
        ldfuncs = {'linear': 'lin',
                'logarithmic': 'log',
                'square_root': 'sqrt',
                'quadratic': 'quad'}

        # let's make sure we'll be able to make the translation later
        if ldfuncA not in ldfuncs.keys() or ldfuncB not in ldfuncs.keys():
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
        fi.write('{:5} {:11} Surface brightness ratio   Amount of third light\n'.format(tratio, l3))


        # According to jktebop's readme.txt:
        # The possible entries for the type of limb darkening law are 'lin' (for linear)
        # 'log' (logarithmic), 'sqrt' (square-root), 'quad' (quadratic) or 'cub' (cubic)

        fi.write('{:5} {:11} LD law type for star A     LD law type for star B\n'.format(ldfuncs[ldfuncA], ldfuncs[ldfuncB]))
        fi.write('{:5} {:11} LD star A (linear coeff)   LD star B (linear coeff)\n'.format(ldcoeffsA[0], ldcoeffsB[0]))
        fi.write('{:5} {:11} LD star A (nonlin coeff)   LD star B (nonlin coeff)\n'.format(ldcoeffsA[1], ldcoeffsB[1]))

        fi.write('{:5} {:11} Reflection effect star A   Reflection effect star B\n'.format(albA, albB))
        fi.write('{:5} {:11} Phase of primary eclipse   Light scale factor (mag)\n'.format(0.0, pblum))
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
        #~ times = b.get_value('times', component=info['component'], dataset=info['dataset'], context='dataset', unit=u.d)
        #~ fluxes = b.get_value('flux', component=info['component'], dataset=info['dataset'], context='dataset', unit=u.d)

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

        # fill syn
        this_syn = new_syns.filter(component=info['component'], dataset=info['dataset'])

        # phases_all, mags_all are 10001 evenly-spaced phases, so we need to interpolate
        # to get at the desired times
        times_all = b.to_time(phases_all)  # in days
        mags_interp = np.interp(info['times'], times_all, mags_all)

        this_syn['times'] = info['times'] * u.d # (period was requested in days)
        logger.warning("converting from mags from JKTEBOP to flux")
        ref_mag = 0  # TODO: what should we do with this?? - option in jktebop compute?
        this_syn['fluxes'] = 10**((mags_interp-ref_mag)/-2.5) * 2  # 2 seems to be necessary - probably from a difference in pblum conventions (or just normalization???)

    yield new_syns
