
import numpy as np
import commands

from phoebe.parameters import dataset as _dataset
from phoebe.parameters import ParameterSet
from phoebe import dynamics
from phoebe.backend import universe, etvs
from phoebe.distortions  import roche
from phoebe.frontend import io
from phoebe import u, c

try:
    import phoebeBackend as phb1
except ImportError:
    _use_phb1 = False
else:
    _use_phb1 = True

import logging
logger = logging.getLogger("BACKENDS")
logger.addHandler(logging.NullHandler())


# protomesh is the mesh at periastron in the reference frame of each individual star
_backends_that_support_protomesh = ['phoebe', 'legacy']
# automesh is meshes with filled observable columns (fluxes etc) at each point at which the mesh is used
_backends_that_support_automesh = ['phoebe', 'legacy']
# the following list is for backends that use numerical meshes
_backends_that_require_meshing = ['phoebe', 'legacy']

def _needs_mesh(b, dataset, method, component, compute):
    """
    """
    # print "*** _needs_mesh", method
    compute_method = b.get_compute(compute).method
    if compute_method not in _backends_that_require_meshing:
        # then we don't have meshes for this backend, so all should be False
        return False

    if method not in ['MESH', 'LC', 'RV']:
        return False

    if method == 'LC' and compute_method=='phoebe' and b.get_value(qualifier='lc_method', compute=compute, dataset=dataset, context='compute')=='analytical':
        return False

    if method == 'RV' and (compute_method == 'legacy' or b.get_value(qualifier='rv_method', compute=compute, component=component, dataset=dataset, context='compute')=='dynamical'):
        return False

    return True


def _timequalifier_by_method(method):
    if method=='ETV':
        return 'time_ephem'
    else:
        return 'time'

def _extract_from_bundle_by_time(b, compute, store_mesh=False, time=None):
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
    times = []
    infos = []
    needed_syns = []

    # for each dataset, component pair we need to check if enabled
    for dataset in b.datasets:

        if dataset=='_default':
            continue

        try:
            dataset_enabled = b.get_value(qualifier='enabled', dataset=dataset, compute=compute, context='compute')
        except ValueError: # TODO: custom exception for no parameter
            # then this backend doesn't support this method
            continue

        if not dataset_enabled:
            # then this dataset is disabled for these compute options
            continue

        for component in b.hierarchy.get_stars()+[None]:
            obs_ps = b.filter(context='dataset', dataset=dataset, component=component).exclude(method='*_dep')
            # only certain methods accept a component of None
            if component is None and obs_ps.method not in ['LC', 'MESH']:
                # TODO: should we change things like lightcurves to be tagged with the component label of the orbit instead of None?
                # anything that can accept observations on the "system" level should accept component is None
                continue

            timequalifier = _timequalifier_by_method(obs_ps.method)
            try:
                this_times = obs_ps.get_value(qualifier=timequalifier, component=component, unit=u.d)
            except ValueError: #TODO: custom exception for no parameter
                continue


            #if not len(this_times):
            #    # then override with passed times if available
            #    this_times = time
            if len(this_times) and time is not None:
                # then overrride the dataset times with the passed times
                this_times = time

            if len(this_times):
                if component is None and obs_ps.method in ['MESH']:
                    components = b.hierarchy.get_meshables()
                else:
                    components = [component]

                for component in components:

                    this_info = {'dataset': dataset,
                            'component': component,
                            'method': obs_ps.method,
                            'needs_mesh': _needs_mesh(b, dataset, obs_ps.method, component, compute),
                            'time': this_times}

                    needed_syns.append(this_info)

                    for time_ in this_times:
                        # print "***", time, this_info
                        # TODO: handle some deltatime allowance here
                        if time_ in times:
                            ind = times.index(time_)
                            infos[ind].append(this_info)
                        else:
                            times.append(time_)
                            infos.append([this_info])


    if store_mesh:
        needed_syns, infos = _handle_protomesh(b, compute, needed_syns, infos)

    if store_mesh:
        needed_syns, infos = _handle_automesh(b, compute, needed_syns, infos, times=times)

    if len(times):
        ti = zip(times, infos)
        ti.sort()
        times, infos = zip(*ti)

    return np.array(times), infos, _create_syns(b, needed_syns)

def _extract_from_bundle_by_dataset(b, compute, store_mesh=False, time=[]):
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
    times = []
    infos = []
    needed_syns = []
    # for each dataset, component pair we need to check if enabled
    for dataset in b.datasets:

        if dataset=='_default':
            continue

        try:
            dataset_enabled = b.get_value(qualifier='enabled', dataset=dataset, compute=compute, context='compute')
        except ValueError: # TODO: custom exception for no parameter
            # then this backend doesn't support this method
            continue

        if not dataset_enabled:
            # then this dataset is disabled for these compute options
            continue

        for component in b.hierarchy.get_stars()+[None]:
            obs_ps = b.filter(context='dataset', dataset=dataset, component=component).exclude(method='*_dep')
            # only certain methods accept a component of None
            if component is None and obs_ps.method not in ['LC', 'MESH']:
                # TODO: should we change things like lightcurves to be tagged with the component label of the orbit instead of None?
                # anything that can accept observations on the "system" level should accept component is None
                continue

            timequalifier = _timequalifier_by_method(obs_ps.method)
            try:
                this_times = obs_ps.get_value(qualifier=timequalifier, component=component, unit=u.d)
            except ValueError: #TODO: custom exception for no parameter
                continue

            if not len(this_times):
                # then override with passed times if available
                this_times = time

            if len(this_times):

                if component is None and obs_ps.method in ['MESH']:
                    components = b.hierarchy.get_meshables()
                else:
                    components = [component]

                for component in components:

                    this_info = {'dataset': dataset,
                            'component': component,
                            'method': obs_ps.method,
                            'needs_mesh': _needs_mesh(b, dataset, obs_ps.method, component, compute),
                            'time': this_times
                            }
                    needed_syns.append(this_info)

                    infos.append([this_info])

    if store_mesh:
        needed_syns, infos = _handle_protomesh(b, compute, needed_syns, infos)

    if store_mesh:
        needed_syns, infos = _handle_automesh(b, compute, needed_syns, infos, times=False)


#    print "NEEDED", needed_syns
    return infos, _create_syns(b, needed_syns)

def _handle_protomesh(b, compute, needed_syns, infos):
    """
    helper function for functionality needed in both _extract_from_bundle_by_dataset
    and _extract_from_bundle_by_times
    """
    # now add "datasets" for the "protomesh"
    if b.get_compute(compute).method in _backends_that_support_protomesh:
        for component in b.hierarchy.get_meshables():
            # then we need the prototype synthetics
            this_info = {'dataset': 'protomesh',
                    'component': component,
                    'method': 'MESH',
                    'needs_mesh': False,
                    'time': [None]}
            needed_syns.append(this_info)

    return needed_syns, infos

def _handle_automesh(b, compute, needed_syns, infos, times=None):
    """
    helper function for functionality needed in both _extract_from_bundle_by_dataset
    and _extract_from_bundle_by_times
    """
    # now add "datasets" for each timepoint at which needs_mesh is True, if store_mesh
    if b.get_compute(compute).method in _backends_that_support_automesh:

        # building synthetics for the automesh is a little different.  Here we
        # want to add a single "this_info" for each component, and fill that with
        # the total times from all valid datasets.

        # So first let's build a list of datasets we need to consider when finding
        # the times.  We'll do this by first building a dictionary, to avoid duplicates
        # that could occur for datasets with multiple components

        needs_mesh_infos = {info['dataset']: info for info in needed_syns if info['needs_mesh'] and info['method'] not in ['MESH']}.values()

        # now let's loop over ALL components first... even if a single component
        # is used in the dataset (ie RVs attached to a single component), we'll
        # still build a synthetic for all components
        # TODO: double check to make sure this is fine with the backend and that
        # the backend still fills these correctly (or are they just empty)?
        for component in b.hierarchy.get_meshables():

            # now let's find a list of all times used in all datasets that require
            # the use of a mesh, avoiding duplicates, and maintaining sort order
            this_times = np.array([])
            for info in needs_mesh_infos:
                this_times = np.append(this_times, info['time'])

            # TODO: there must be a better way to do this
            this_times = np.array(list(set(this_times.tolist())))
            this_times.sort()


            this_info = {'dataset': 'automesh',
                'component': component,
                'method': 'MESH',
                'needs_mesh': True,
                'time': this_times}

            if times:
                for time in this_times:
                    ind = times.index(time)
                    infos[ind].append(this_info)

            needed_syns.append(this_info)

    return needed_syns, infos


def _create_syns(b, needed_syns, store_mesh=False):
    """
    Create empty synthetics

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter list needed_syns: list of dictionaries containing kwargs to access
        the dataset (dataset, component, method)
    :return: :class:`phoebe.parameters.parameters.ParameterSet` of all new parameters
    """

    needs_mesh = {info['dataset']: info['method'] for info in needed_syns if info['needs_mesh']}

    params = []
    for needed_syn in needed_syns:
        # used to be {}_syn
        syn_method = '{}'.format(needed_syn['method'])
        if needed_syn['method']=='MESH':
            # parameters.dataset.mesh will handle creating the necessary columns
            needed_syn['dataset_fields'] = needs_mesh

        these_params, these_constraints = getattr(_dataset, "{}_syn".format(syn_method.lower()))(**needed_syn)
        # TODO: do we need to handle constraints?
        these_params = these_params.to_list()
        for param in these_params:
            param._component = needed_syn['component']
            if param._dataset is None:
                # dataset may be set for mesh columns
                param._dataset = needed_syn['dataset']
            param._method = syn_method
            # context, model, etc will be handle by the bundle once these are returned

        params += these_params

    return ParameterSet(params)

def phoebe(b, compute, time=[], as_generator=False, **kwargs):
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
        * time
        * flux
    * rv (dynamical only):
        * time
        * rv

    This function will almost always be called through the bundle, using
        * :meth:`phoebe.frontend.bundle.Bundle.add_compute`
        * :meth:`phoebe.frontend.bundle.Bundle.run_compute`

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle` containing the system
        and datasets
    :parameter str compute: the label of the computeoptions to use (in the bundle).
        These computeoptions must have a method of 'phoebe'.
    :parameter **kwargs: any temporary overrides to computeoptions
    :return: a list of new synthetic :class:`phoebe.parameters.parameters.ParameterSet`s
    """

    computeparams = b.get_compute(compute, force_ps=True, check_relevant=False)
    hier = b.get_hierarchy()

    starrefs  = hier.get_stars()
    meshablerefs = hier.get_meshables()
    # starorbitrefs = [hier.get_parent_of(star) for star in starrefs]
    # orbitrefs = hier.get_orbits()


    store_mesh = computeparams.get_value('store_mesh', **kwargs)

    times, infos, new_syns = _extract_from_bundle_by_time(b, compute=compute, time=time, store_mesh=store_mesh)

    dynamics_method = computeparams.get_value('dynamics_method', **kwargs)
    ltte = computeparams.get_value('ltte', **kwargs)

    distance = b.get_value(qualifier='distance', context='system', unit=u.m)
    t0 = b.get_value(qualifier='t0', context='system', unit=u.d)

    if len(starrefs)==1 and computeparams.get_value('distortion_method', component=starrefs[0], **kwargs) in ['roche']:
        raise ValueError("distortion_method='{}' not valid for single star".format(computeparams.get_value('distortion_method', component=starrefs[0], **kwargs)))

    if len(meshablerefs) > 1 or hier.get_kind_of(meshablerefs[0])=='envelope':
        if dynamics_method in ['nbody', 'rebound']:
            # if distortion_method == 'roche':
                # raise ValueError("distortion method '{}' not compatible with dynamics_method '{}'".format(distortion_method, dynamics_method))

            # gr = computeparams.get_value('gr', check_relevant=False, **kwargs)

            # TODO: pass stepsize
            t0, xs0, ys0, zs0, vxs0, vys0, vzs0 = dynamics.nbody.dynamics_from_bundle(b, [t0], compute, **kwargs)
            ethetas0, elongans0, eincls0 = None, None, None
            ts, xs, ys, zs, vxs, vys, vzs = dynamics.nbody.dynamics_from_bundle(b, times, compute, **kwargs)

        elif dynamics_method == 'bs':
            # if distortion_method == 'roche':
                # raise ValueError("distortion method '{}' not compatible with dynamics_method '{}'".format(distortion_method, dynamics_method))

            # TODO: pass stepsize
            # TODO: pass orbiterror
            # TODO: make sure that this takes systemic velocity and corrects positions and velocities (including ltte effects if enabled)
            t0, xs0, ys0, zs0, vxs0, vys0, vzs0 = dynamics.nbody.dynamics_from_bundle_bs(b, [t0], compute, **kwargs)
            ethetas0, elongans0, eincls0 = None, None, None
            ts, xs, ys, zs, vxs, vys, vzs = dynamics.nbody.dynamics_from_bundle_bs(b, times, compute, **kwargs)


        elif dynamics_method=='keplerian':

            # TODO: change syntax to be same format as rebound above (compute, **kwargs)
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

    # Now we should store the protomesh
    if store_mesh:
        for component in meshablerefs:
            body = system.get_body(component)

            protomesh = body.get_standard_mesh(scaled=False)  # TODO: provide theta=0.0 when supported
            body._compute_instantaneous_quantities([], [], [], d=1-body.ecc)
            body._fill_loggs(mesh=protomesh)
            body._fill_gravs(mesh=protomesh)
            body._fill_teffs(mesh=protomesh)

            this_syn = new_syns.filter(component=component, dataset='protomesh')

            # protomesh = body.get_standard_mesh(scaled=False)  # TODO: provide theta=0.0 when supported

            this_syn['x'] = protomesh.centers[:,0]# * u.solRad
            this_syn['y'] = protomesh.centers[:,1]# * u.solRad
            this_syn['z'] = protomesh.centers[:,2]# * u.solRad
            this_syn['vertices'] = protomesh.vertices_per_triangle
            this_syn['areas'] = protomesh.areas # * u.solRad**2
            this_syn['tareas'] = protomesh.tareas # * u.solRad**2
            this_syn['normals'] = protomesh.tnormals
            this_syn['nx'] = protomesh.tnormals[:,0]
            this_syn['ny'] = protomesh.tnormals[:,1]
            this_syn['nz'] = protomesh.tnormals[:,2]

            this_syn['logg'] = protomesh.loggs.centers
            this_syn['teff'] = protomesh.teffs.centers
            # this_syn['mu'] = protomesh.mus  # mus aren't filled until placed in orbit

            # NOTE: this is a computed column, meaning the 'r' is not the radius to centers, but rather the
            # radius at which computables have been determined.  This way r should not suffer from a course
            # grid (so much).  Same goes for cosbeta below.
            this_syn['r'] = protomesh.rs.centers
            # NOTE: no r_proj for protomeshs since we don't have LOS information

            # TODO: need to test the new (ComputedColumn) version of this
            this_syn['cosbeta'] = protomesh.cosbetas.centers
            # this_syn['cosbeta'] = [np.dot(c,n)/ (np.sqrt((c*c).sum())*np.sqrt((n*n).sum())) for c,n in zip(protomesh.centers, protomesh.tnormals)]


    # Now we need to compute intensities at t0 in order to scale pblums for all future times
    # TODO: only do this if we need the mesh for actual computations
    # TODO: move as much of this pblum logic into mesh.py as possible

    def dynamics_at_i(xs, ys, zs, vxs, vys, vzs, ethetas=None, elongans=None, eincls=None, i=0):
        xi, yi, zi = [x[i].to(u.solRad) for x in xs], [y[i].to(u.solRad) for y in ys], [z[i].to(u.solRad) for z in zs]
        vxi, vyi, vzi = [vx[i].to(u.solRad/u.d) for vx in vxs], [vy[i].to(u.solRad/u.d) for vy in vys], [vz[i].to(u.solRad/u.d) for vz in vzs]
        if ethetas is not None:
            ethetai, elongani, eincli = [etheta[i].to(u.rad) for etheta in ethetas], [elongan[i].to(u.rad) for elongan in elongans], [eincl[i].to(u.rad) for eincl in eincls]
        else:
            ethetai, elongani, eincli = None, None, None

        return xi, yi, zi, vxi, vyi, vzi, ethetai, elongani, eincli

    methods = b.get_dataset().methods
    if 'LC' in methods or 'RV' in methods:  # TODO this needs to be WAY more general
        # we only need to handle pblum_scale if we have a dataset method which requires
        # intensities
        if len(meshablerefs) > 1:
            x0, y0, z0, vx0, vy0, vz0, etheta0, elongan0, eincl0 = dynamics_at_i(xs0, ys0, zs0, vxs0, vys0, vzs0, ethetas0, elongans0, eincls0, i=0)
        else:
            x0, y0, z0 = [0.], [0.], [0.]
            vx0, vy0, vz0 = [0.], [0.], [0.]
            # TODO: star needs long_an (yaw?)
            etheta0, elongan0, eincl0 = [0.], [0.], [b.get_value('incl', unit=u.rad)]

        system.update_positions(t0, x0, y0, z0, vx0, vy0, vz0, etheta0, elongan0, eincl0)

        for dataset in b.datasets:
            ds = b.get_dataset(dataset=dataset, method='*dep')

            if ds.method is None:
                continue

            method = ds.method[:-4]

            #print "***", dataset, method
            if method not in ['LC', 'RV']:
                continue

            for component in ds.components:
                if component=='_default':
                    continue

                # let's populate the intensities for THIS dataset at t0
                #print "***", method, dataset, component, {p.qualifier: p.get_value() for p in b.get_dataset(dataset, component=component, method='*dep').to_list()+b.get_compute(compute, component=component).to_list()}
                kwargss = [{p.qualifier: p.get_value() for p in b.get_dataset(dataset, component=component, method='*dep').to_list()+b.get_compute(compute, component=component).to_list()+b.filter(qualifier='passband', dataset=dataset, method='*dep').to_list()}]

                system.populate_observables(t0,
                    [method], [dataset],
                    kwargss)

            # now for each component we need to store the scaling factor between
            # absolute and relative intensities
            pbscale_copy = {}
            for component in starrefs:
                if component=='_default':
                    continue
                #print "*** pbscale", component, dataset
                pbscale = b.get_value(qualifier='pbscale', component=component, dataset=dataset, context='dataset')
                if pbscale=='pblum':
                    pblum = b.get_value(qualifier='pblum', component=component, dataset=dataset, context='dataset')
                    ld_func = b.get_value(qualifier='ld_func', component=component, dataset=dataset, context='dataset')
                    ld_coeffs = b.get_value(qualifier='ld_coeffs', component=component, dataset=dataset, context='dataset')

                    system.get_body(component).compute_pblum_scale(dataset, pblum, ld_func=ld_func, ld_coeffs=ld_coeffs)
                else:
                    # then this component wants to copy the scale from another component
                    # in the system.  We'll just store this now so that we make sure the
                    # component we're copying from has a chance to compute its scale
                    # first.
                    pbscale_copy[component] = pbscale

            # now let's copy all the scales for those that are just referencing another component
            for comp, comp_copy in pbscale_copy.items():
                system.get_body(comp)._pblum_scale[dataset] = system.get_body(comp_copy).get_pblum_scale(dataset)


    # MAIN COMPUTE LOOP
    # the outermost loop will be over times.  infolist will be a list of dictionaries
    # with component, method, and dataset as keys applicable for that current time.
    for i,time,infolist in zip(range(len(times)),times,infos):
        # Check to see what we might need to do that requires a mesh
        # TOOD: make sure to use the requested distortion_method

        if True in [info['needs_mesh'] for info in infolist]:


            if dynamics_method == 'nbody':
                raise NotImplementedError('nbody not supported for meshes yet')

            # we need to extract positions, velocities, and euler angles of ALL bodies at THIS TIME (i)
            if len(meshablerefs) > 1:
                xi, yi, zi, vxi, vyi, vzi, ethetai, elongani, eincli = dynamics_at_i(xs, ys, zs, vxs, vys, vzs, ethetas, elongans, eincls, i=i)
            else:
                xi, yi, zi = [0.], [0.], [0.]
                vxi, vyi, vzi = [0.], [0.], [0.]
                # TODO: star needs long_an (yaw?)

                ethetai, elongani, eincli = [0.], [0.], [b.get_value('incl', component=hier.get_parent_of(meshablerefs[0]), unit=u.rad)]

            # TODO: eventually we can pass instantaneous masses and sma as kwargs if they're time dependent
            # masses = [b.get_value('mass', component=star, context='component', time=time, unit=u.solMass) for star in starrefs]
            # sma = b.get_value('sma', component=starrefs[body.ind_self], context='component', time=time, unit=u.solRad)

            system.update_positions(time, xi, yi, zi, vxi, vyi, vzi, ethetai, elongani, eincli)

            # Now we need to determine which triangles are visible and handle subdivision
            # NOTE: this should come after populate_observables so that each subdivided triangle
            # will have identical local quantities.  The only downside to this is that we can't
            # make a shortcut and only populate observables at known-visible triangles - but
            # frankly that wouldn't save much time anyways and would then be annoying when
            # inspecting or plotting the mesh
            # NOTE: this has been moved before populate observables now to make use
            # of per-vertex weights which are used to determine the physical quantities
            # (ie teff, logg) that should be used in computing observables (ie intensity)

            # TODO: for testing only
            # if computeparams.get_value('eclipse_alg') == 'wd_horizon':
            #     io.pass_to_legacy(b, filename='_tmp_legacy_inp')

            system.handle_eclipses()

            # Now we can fill the observables per-triangle.  We'll wait to integrate
            # until we're ready to fill the synthetics
            # print "*** system.populate_observables", [info['method'] for info in infolist if info['needs_mesh']], [info['dataset'] for info in infolist if info['needs_mesh']]
            kwargss = [{p.qualifier: p.get_value() for p in b.get_dataset(info['dataset'], component=info['component'], method='*dep').to_list()+b.get_compute(compute, component=info['component']).to_list()+b.filter(qualifier='passband', dataset=info['dataset'], method='*dep').to_list()} for info in infolist if info['needs_mesh']]

            system.populate_observables(time,
                    [info['method'] for info in infolist if info['needs_mesh']],
                    [info['dataset'] for info in infolist if info['needs_mesh']],
                    kwargss
                    )


        # now let's loop through and fill any synthetics at this time step
        # TODO: make this MPI ready by ditching appends and instead filling with all nans and then filling correct index
        for info in infolist:
            # i, time, info['method'], info['component'], info['dataset']
            cind = starrefs.index(info['component']) if info['component'] in starrefs else None
            # ts[i], xs[cind][i], ys[cind][i], zs[cind][i], vxs[cind][i], vys[cind][i], vzs[cind][i]
            method = info['method']


            if method in ['MESH', 'SP']:
                # print "*** new_syns", new_syns.twigs
                # print "*** filtering new_syns", info['component'], info['dataset'], method, time
                # print "*** this_syn.twigs", new_syns.filter(method=method, time=time).twigs
                this_syn = new_syns.filter(component=info['component'], dataset=info['dataset'], method=method, time=time)
            else:
                # print "*** new_syns", new_syns.twigs
                # print "*** filtering new_syns", info['component'], info['dataset'], method
                # print "*** this_syn.twigs", new_syns.filter(component=info['component'], dataset=info['dataset'], method=method).twigs
                this_syn = new_syns.filter(component=info['component'], dataset=info['dataset'], method=method)

            # now check the method to see what we need to fill
            if method=='RV':
                ### this_syn['time'].append(time) # time array was set when initializing the syns
                if info['needs_mesh']:
                    # TODO: we have to call get here because twig access will trigger on method=rv and qualifier=rv
                    # print "***", this_syn.filter(qualifier='rv').twigs, this_syn.filter(qualifier='rv').methods, this_syn.filter(qualifier='rv').components
                    # if len(this_syn.filter(qualifier='rv').twigs)>1:
                        # print "***2", this_syn.filter(qualifier='rv')[1].method, this_syn.filter(qualifier='rv')[1].component
                    rv = system.observe(info['dataset'], method=method, components=info['component'], distance=distance)['rv']
                    this_syn['rv'].append(rv*u.solRad/u.d)
                else:
                    # then rv_method == 'dynamical'
                    # TODO: send in with units
                    this_syn['rv'].append(-1*vzs[cind][i])

            elif method=='LC':

                # print "***", info['component']
                # print "***", system.observe(info['dataset'], method=method, components=info['component'])
                l3 = b.get_value(qualifier='l3', dataset=info['dataset'], context='dataset')
                this_syn['flux'].append(system.observe(info['dataset'], method=method, components=info['component'], distance=distance, l3=l3)['flux'])

            elif method=='ETV':

                # TODO: add support for other ETV methods (barycentric, robust, others?)
                time_ecl = etvs.crossing(b, info['component'], time, dynamics_method, ltte, tol=computeparams.get_value('etv_tol', u.d, dataset=info['dataset'], component=info['component']))

                this_obs = b.filter(dataset=info['dataset'], component=info['component'], context='dataset')
                this_syn['N'].append(this_obs.get_parameter(qualifier='N').interp_value(time_ephem=time))  # TODO: there must be a better/cleaner way to do this
                this_syn['time_ephem'].append(time)  # NOTE: no longer under constraint control
                this_syn['time_ecl'].append(time_ecl)
                this_syn['etv'].append(time_ecl-time)  # NOTE: no longer under constraint control

            elif method=='IFM':
                observables_ifm = system.observe(info['dataset'], method=method, components=info['component'], distance=distance)
                for key in observables_ifm.keys():
                    this_syn[key] = observables_ifm[key]

            elif method=='ORB':
                # ts[i], xs[cind][i], ys[cind][i], zs[cind][i], vxs[cind][i], vys[cind][i], vzs[cind][i]

                ### this_syn['time'].append(ts[i])  # time array was set when initializing the syns
                this_syn['x'].append(xs[cind][i])
                this_syn['y'].append(ys[cind][i])
                this_syn['z'].append(zs[cind][i])
                this_syn['vx'].append(vxs[cind][i])
                this_syn['vy'].append(vys[cind][i])
                this_syn['vz'].append(vzs[cind][i])

            elif method=='MESH':
                # print "*** info['component']", info['component'], " info['dataset']", info['dataset']
                # print "*** this_syn.twigs", this_syn.twigs
                body = system.get_body(info['component'])

                this_syn['pot'] = body._instantaneous_pot
                this_syn['rpole'] = roche.potential2rpole(body._instantaneous_pot, body.q, body.ecc, body.F, body._scale, component=body.comp_no)
                this_syn['volume'] = body.volume

                # TODO: should x, y, z be computed columns of the vertices???
                # could easily have a read-only property at the ProtoMesh level
                # that returns a ComputedColumn for xs, ys, zs (like rs)
                # (also do same for protomesh)
                this_syn['x'] = body.mesh.centers[:,0]# * u.solRad
                this_syn['y'] = body.mesh.centers[:,1]# * u.solRad
                this_syn['z'] = body.mesh.centers[:,2]# * u.solRad
                this_syn['vx'] = body.mesh.velocities.centers[:,0] * u.solRad/u.d # TODO: check units!!!
                this_syn['vy'] = body.mesh.velocities.centers[:,1] * u.solRad/u.d
                this_syn['vz'] = body.mesh.velocities.centers[:,2] * u.solRad/u.d
                this_syn['vertices'] = body.mesh.vertices_per_triangle
                this_syn['areas'] = body.mesh.areas # * u.solRad**2
                # TODO remove this 'normals' vector now that we have nx,ny,nz?
                this_syn['normals'] = body.mesh.tnormals
                this_syn['nx'] = body.mesh.tnormals[:,0]
                this_syn['ny'] = body.mesh.tnormals[:,1]
                this_syn['nz'] = body.mesh.tnormals[:,2]
                this_syn['mu'] = body.mesh.mus

                this_syn['logg'] = body.mesh.loggs.centers
                this_syn['teff'] = body.mesh.teffs.centers
                # TODO: include abun? (body.mesh.abuns.centers)

                # NOTE: these are computed columns, so are not based on the
                # "center" coordinates provided by x, y, z, etc, but rather are
                # the average value across each triangle.  For this reason,
                # they are less susceptible to a coarse grid.
                this_syn['r'] = body.mesh.rs.centers
                this_syn['r_proj'] = body.mesh.rprojs.centers

                this_syn['visibility'] = body.mesh.visibilities

                # vcs = np.sum(body.mesh.vertices_per_triangle*body.mesh.weights[:,:,np.newaxis], axis=1)
                # for i,vc in enumerate(vcs):
                #     if np.all(vc==np.array([0,0,0])):
                #         vcs[i] = np.full(3, np.nan)
                # this_syn['visible_centroids'] = vcs
                this_syn['visible_centroids'] = []

                indeps = {'RV': ['rv', 'intens_norm_abs', 'intens_norm_rel', 'intens_proj_abs', 'intens_proj_rel', 'ampl_boost', 'ld'], 'LC': ['intens_norm_abs', 'intens_norm_rel', 'intens_proj_abs', 'intens_proj_rel', 'ampl_boost', 'ld'], 'IFM': ['intens_norm_abs', 'intens_norm_rel', 'intens_proj_abs', 'intens_proj_rel']}
                for infomesh in infolist:
                    if infomesh['needs_mesh'] and infomesh['method'] != 'MESH':
                        new_syns.set_value(qualifier='pblum', time=time, dataset=infomesh['dataset'], component=info['component'], method='MESH', value=body.compute_luminosity(infomesh['dataset']))

                        for indep in indeps[infomesh['method']]:
                            key = "{}:{}".format(indep, infomesh['dataset'])
                            # print "***", key, indep, new_syns.qualifiers
                            # print "***", indep, time, infomesh['dataset'], info['component'], 'mesh', new_syns.filter(time=time, method='mesh').twigs
                            try:
                                new_syns.set_value(qualifier=indep, time=time, dataset=infomesh['dataset'], component=info['component'], method='MESH', value=body.mesh[key].centers)
                            except ValueError:
                                raise ValueError("more than 1 result found: {}".format(",".join(new_syns.filter(qualifier=indep, time=time, dataset=infomesh['dataset'], component=info['component'], method='MESH').twigs)))


            else:
                raise NotImplementedError("method {} not yet supported by this backend".format(method))

        if as_generator:
            # this is mainly used for live-streaming animation support
            yield (new_syns, time)

    if not as_generator:
        yield new_syns


def legacy(b, compute, time=[], **kwargs): #, **kwargs):#(b, compute, **kwargs):

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
        * time
        * flux
    * rv (dynamical only):
        * time
        * rv

    This function will almost always be called through the bundle, using
        * :meth:`phoebe.frontend.bundle.Bundle.add_compute`
        * :meth:`phoebe.frontend.bundle.Bundle.run_compute`

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle` containing the system
        and datasets
    :parameter str compute: the label of the computeoptions to use (in the bundle).
        These computeoptions must have a method of 'legacy'.
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
        specific (x, y, z) for phoebe1 and phoebe2.

    Raises:
        ImportError if the python 'phoebeBackend' interface to PHOEBE
        legacy is not found/installed



    """
    p2to1 = {'tloc':'teff', 'glog':'logg', 'vcx':'x', 'vcy':'y', 'vcz':'z', 'grx':'nx', 'gry':'ny', 'grz':'nz', 'csbt':'cosbeta', 'rad':'r','Inorm':'intens_norm_abs'}

    def ret_dict(key):
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
            d['component'] = 'primary'
        elif comp== 2:
            d['component'] = 'secondary'
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


    # check whether phoebe legacy is installed
    if not _use_phb1:
        raise ImportError("phoebeBackend for phoebe legacy not found")

    computeparams = b.get_compute(compute, force_ps=True)
    store_mesh = computeparams.get_value('store_mesh', **kwargs)
#    computeparams = b.get_compute(compute, force_ps=True)
#    hier = b.get_hierarchy()

#    starrefs  = hier.get_stars()
#    orbitrefs = hier.get_orbits()

    stars = b.hierarchy.get_stars()
    primary, secondary = stars
    #need for protomesh
    perpass = b.get_value(qualifier='t0_perpass', method='orbit', context='component')
    # print primary, secondary
    #make phoebe 1 file

    # TODO BERT: this really should be a random name (tmpfile) so two instances won't clash
    io.pass_to_legacy(b, filename='_tmp_legacy_inp')
    phb1.init()
    try:
        phb1.configure()
    except SystemError:
        raise SystemError("PHOEBE config failed: try creating PHOEBE config file through GUI")
    phb1.open('_tmp_legacy_inp')
    # TODO BERT: why are we saving here?
    phb1.save('after.phoebe')
    lcnum = 0
    rvnum = 0
    infos, new_syns = _extract_from_bundle_by_dataset(b, compute=compute, time=time, store_mesh=store_mesh)


#    print "INFOS", len(infos)
#    print "info 1",  infos[0]
#    print "info 2-1",  infos[0][1]
#    print "info 3-1",  infos[0][2]
#    quit()
    for info in infos:
        info = info[0]
        this_syn = new_syns.filter(component=info['component'], dataset=info['dataset'])
        time = info['time']
        dataset=info['dataset']
        if info['method'] == 'LC':
            if not store_mesh:
            # print "*********************", this_syn.qualifiers
                flux= np.array(phb1.lc(tuple(time.tolist()), lcnum))
                lcnum = lcnum+1
                #get rid of the extra periastron passage
                this_syn['flux'] = flux

            else:
                time = np.append(time, perpass)
                flux, mesh = phb1.lc(tuple(time.tolist()), 0, lcnum+1)
                flux = np.array(flux)
            # take care of the lc first
                this_syn['flux'] = flux[:-1]


            # now deal with parameters
                keys = mesh.keys()
                n = len(time)

            # calculate the normal 'magnitude' for normalizing vectors
                grx1 = np.array_split(mesh['grx1'],n)[-1]
                gry1 = np.array_split(mesh['gry1'],n)[-1]
                grz1 = np.array_split(mesh['grz1'],n)[-1]
                # TODO: rewrite this to use np.linalg.norm
                grtot1 = grx1**2+gry1**2+grz1**2
                grx2 = np.array_split(mesh['grx1'],n)[-1]
                gry2 = np.array_split(mesh['gry1'],n)[-1]
                grz2 = np.array_split(mesh['grz1'],n)[-1]
                grtot2 = grx2**2+gry2**2+grz2**2
                grtot = [np.sqrt(grtot1),np.sqrt(grtot2)]
                for key in keys:
                    key_values =  np.array_split(mesh[key],n)
                    # take care of the protomesh
                    prot_val = key_values[-1]
                    d = ret_dict(key)
                    d['dataset'] = 'protomesh'
                    key_val = np.array(zip(prot_val, prot_val, prot_val, prot_val, prot_val, prot_val, prot_val, prot_val)).flatten()
                    if key[:2] =='gr':
                        grtotn = grtot[int(key[-1])-1]

                        grtotn = np.array(zip(grtotn, grtotn, grtotn, grtotn, grtotn, grtotn, grtotn, grtotn)).flatten()

                        # normals should be normalized
                        d['value'] = -key_val /grtotn
                    else:
                        d['value'] = key_val
                    #TODO fill the normals column it is just (nx, ny, nz)

                    try:
                        new_syns.set_value(**d)
                    except:
                        logger.warning('{} has no corresponding value in phoebe 2 protomesh'.format(key))

                    #Normalize the normals that have been put in protomesh

                    # now take care of automesh time point by time point
                    for t in range(len(time[:-1])):
#                        d = ret_dict(key)
                        d['dataset'] = 'automesh'
                        if key in ['Inorm1', 'Inorm2']:
                            d['dataset'] = dataset

                        d['time'] = time[t]
                    #prepare data
                        if key[:2] in ['vc', 'gr']:
                            # I need to change coordinates but not yet done
                            pass

                            #TODO Change these values so that they are placed in orbit

                        else:
                            key_val= key_values[t]
                            key_val = np.array(zip(key_val, key_val, key_val, key_val, key_val, key_val, key_val, key_val)).flatten()

                            param = new_syns.filter(**d)
                            if param:
                                d['value'] = key_val
                                new_syns.set_value(**d)
                            else:
                                logger.warning('{} has no corresponding value in phoebe 2 automesh'.format(key))

                time = time[:-1]

        elif info['method'] == 'RV':
#            print "SYN", this_syn
            rvid = info['dataset']
            #print "rvid", info
#            quit()

            if rvid == phb1.getpar('phoebe_rv_id', 0):

                dep =  phb1.getpar('phoebe_rv_dep', 0)
                dep = dep.split(' ')[0].lower()
           # must account for rv datasets with multiple components
                if dep != info['component']:
                    dep = info['component']

            elif rvid == phb1.getpar('phoebe_rv_id', 1):
                dep =  phb1.getpar('phoebe_rv_dep', 1)
                dep = dep.split(' ')[0].lower()
           # must account for rv datasets with multiple components
                if dep != info['component']:
                    dep = info['component']

            proximity = computeparams.filter(qualifier ='rv_method', component='primary', dataset=rvid).get_value()

            if proximity == 'flux-weighted':
                rveffects = 1
            else:
                rveffects = 0
#            try:
#                dep2 =  phb1.getpar('phoebe_rv_dep', 1)
#                dep2 = dep2.split(' ')[0].lower()
#            except:
#                dep2 = None
#            print "dep", dep
#            print "dep2", dep2
#            print "COMPONENT", info['component']
            if dep == 'primary':
                phb1.setpar('phoebe_proximity_rv1_switch', rveffects)
                rv = np.array(phb1.rv1(tuple(time.tolist()), 0))
                rvnum = rvnum+1

            elif dep == 'secondary':
                phb1.setpar('phoebe_proximity_rv2_switch', rveffects)
                rv = np.array(phb1.rv2(tuple(time.tolist()), 0))
                rvnum = rvnum+1
            else:
                raise ValueError(str(info['component'])+' is not the primary or the secondary star')


            #print "***", u.solRad.to(u.km)
            this_syn.set_value(qualifier='rv', value=rv*u.km/u.s)
#            print "INFO", info
#            print "SYN", this_syn

            #print 'THIS SYN', this_syn

        elif info['method']=='MESH':
            pass
#            print "I made it HERE"
#        if info['method'] == 'MESH':
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

def photodynam(b, compute, time=[], **kwargs):
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
        - time
        - flux

    * rv (dynamical only):
        - time
        - rv

    This function will almost always be called through the bundle, using
        * :meth:`phoebe.frontend.bundle.Bundle.add_compute`
        * :meth:`phoebe.frontend.bundle.Bundle.run_compute`

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle` containing the system
        and datasets
    :parameter str compute: the label of the computeoptions to use (in the bundle).
        These computeoptions must have a method of 'photodynam'.
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

    infos, new_syns = _extract_from_bundle_by_dataset(b, compute=compute, time=time)

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

        if info['method'] == 'LC':
            pblums = [b.get_value(qualifier='pblum', component=star,
                    context='dataset', dataset=info['dataset'])
                    for star in starrefs]  # TODO: units or unitless?
            u1s, u2s = [], []
            for star in starrefs:
                if b.get_value(qualifier='ld_func', component=star, dataset=info['dataset'], context='dataset') == 'linear':
                    ld_coeffs = b.get_value(qualifier='ld_coeffs', component=star, dataset=info['dataset'], context='dataset')
                else:
                    ld_coeffs = (0,0)
                    logger.warning("ld_func for {} {} must be linear for the photodynam backend, but is not: defaulting to linear with coeffs of {}".format(star, info['dataset'], ld_coeffs))

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

            t0 = b.get_value('t0_perpass', component=orbitref,
                context='component', unit=u.d)
            period = b.get_value('period', component=orbitref,
                context='component', unit=u.d)

            om = 2 * np.pi * (time0 - t0) / period
            fi.write('{} {} {} {} {} {}\n'.format(a, e, i, o, l, om))
        fi.close()

        # write the report file
        fr = open('_tmp_pd_rep', 'w')
        # t times
        # F fluxes
        # x light-time corrected positions
        # v light-time corrected velocities
        fr.write('t F x v \n')   # TODO: don't always get all?

        for t in b.get_value('time', component=info['component'], dataset=info['dataset'], context='dataset', unit=u.d):
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
        if info['method']=='LC':
            this_syn['time'] = stuff[0] * u.d
            this_syn['flux'] = stuff[1] + 1  # TODO: figure out why and actually fix the problem instead of fudging it!?!!?
        elif info['method']=='ORB':
            cind = starrefs.index(info['component'])
            this_syn['time'] = stuff[0] * u.d
            this_syn['x'] = -1*stuff[2+(cind*3)] * u.AU
            this_syn['y'] = -1*stuff[3+(cind*3)] * u.AU
            this_syn['z'] = stuff[4+(cind*3)] * u.AU
            this_syn['vx'] = -1*stuff[3*nbodies+2+(cind*3)] * u.AU/u.d
            this_syn['vy'] = -1*stuff[3*nbodies+3+(cind*3)] * u.AU/u.d
            this_syn['vz'] = stuff[3*nbodies+4+(cind*3)] * u.AU/u.d
        elif info['method']=='RV':
            cind = starrefs.index(info['component'])
            this_syn['time'] = stuff[0] * u.d
            this_syn['rv'] = -stuff[3*nbodies+4+(cind*3)] * u.AU/u.d
        else:
            raise NotImplementedError("method {} not yet supported by this backend".format(method))

    yield new_syns

def jktebop(b, compute, time=[], **kwargs):
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
    Jktebop also includes its own fitting methods, including bootstrapping.
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
        - time
        - flux

    * rv (dynamical only):
        - time
        - rv

    This function will almost always be called through the bundle, using
        * :meth:`phoebe.frontend.bundle.Bundle.add_compute`
        * :meth:`phoebe.frontend.bundle.Bundle.run_compute`

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle` containing the system
        and datasets
    :parameter str compute: the label of the computeoptions to use (in the bundle).
        These computeoptions must have a method of 'jktebop'.
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

    infos, new_syns = _extract_from_bundle_by_dataset(b, compute=compute, time=time)  # TODO: add rv support (see commented context below)

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
            raise ValueError("jktebop backend currently only supports decoupled pblums (b.set_value_all('pbscale', 'pblum'))")

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

        albA = b.get_value('alb', component=starrefs[0], dataset=info['dataset'], context='dataset')
        albB = b.get_value('alb', component=starrefs[1], dataset=info['dataset'], context='dataset')

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
        # data. Then add a line below the main input parameters for each RV file:
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
        #~ times = b.get_value('time', component=info['component'], dataset=info['dataset'], context='dataset', unit=u.d)
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
        mags_interp = np.interp(info['time'], times_all, mags_all)

        this_syn['time'] = info['time'] * u.d # (period was requested in days)
        logger.warning("converting from mags from JKTEBOP to flux")
        ref_mag = 0  # TODO: what should we do with this?? - option in jktebop compute?
        this_syn['flux'] = 10**((mags_interp-ref_mag)/-2.5) * 2  # 2 seems to be necessary - probably from a difference in pblum conventions (or just normalization???)

    yield new_syns
