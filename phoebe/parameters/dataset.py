
from phoebe.parameters import *
from phoebe.parameters import constraint
from phoebe.atmospheres import passbands  # need to load pbtable (dictionary of available passbands)
from phoebe import u
from phoebe import conf

_ld_func_choices = ['interp', 'linear', 'logarithmic', 'quadratic', 'square_root', 'power']

global _pbdep_kinds
_pbdep_kinds = ['lc', 'rv']

def _empty_array(kwargs, qualifier):
    if qualifier in kwargs.keys():
        return kwargs.get(qualifier)
    elif 'empty_arrays_len' in kwargs.keys():
        return np.empty(kwargs.get('empty_arrays_len'))
    else:
        return []

def lc(**kwargs):
    """
    Create parameters for a new light curve dataset.

    Generally, this will be used as an input to the kind argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_dataset`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """

    obs_params = []

    syn_params, constraints = lc_syn(syn=False, **kwargs)
    obs_params += syn_params.to_list()
    #obs_params += lc_dep(**kwargs).to_list()

    #~ obs_params += [FloatArrayParameter(qualifier='flag', value=kwargs.get('flag', []), default_unit=None, description='Signal flag')]
    #~ obs_params += [FloatArrayParameter(qualifier='weight', value=kwargs.get('weight', []), default_unit=None, description='Signal weight')]

    #~ obs_params += [FloatParameter(qualifier='timeoffset', value=kwargs.get('timeoffset', 0.0), default_unit=u.d, description='Zeropoint date offset for observations')]
    #~ obs_params += [FloatParameter(qualifier='statweight', value=kwargs.get('statweight', 0.0), default_unit=None, description='Statistical weight in overall fitting')]

    return ParameterSet(obs_params), constraints

def lc_syn(syn=True, **kwargs):

    syn_params = []

    syn_params += [FloatArrayParameter(qualifier='times', value=kwargs.get('times', []), default_unit=u.d, description='Observed times')]
    syn_params += [FloatArrayParameter(qualifier='fluxes', value=_empty_array(kwargs, 'fluxes'), default_unit=u.W/u.m**2, description='Observed flux')]
    if not syn:
        syn_params += [FloatArrayParameter(qualifier='sigmas', value=_empty_array(kwargs, 'sigmas'), default_unit=u.W/u.m**2, description='Observed uncertainty on flux')]


    #~ syn_params += [FloatArrayParameter(qualifier='exptime', value=kwargs.get('exptime', []), default_unit=u.s, description='Signal exposure time')]

    constraints = []

    return ParameterSet(syn_params), constraints

def lc_dep(is_lc=True, **kwargs):
    dep_params = []

    # NOTE: these need to be added to the exception in bundle.add_dataset so that the kwargs get applied correctly
    dep_params += [ChoiceParameter(qualifier='ld_func', copy_for={'kind': ['star', 'envelope'], 'component': '*'}, component='_default', value=kwargs.get('ld_func', 'interp'), choices=_ld_func_choices, description='Limb darkening model')]
    dep_params += [FloatArrayParameter(qualifier='ld_coeffs', visible_if='ld_func:!interp', copy_for={'kind': ['star', 'envelope'], 'component': '*'}, component='_default', value=kwargs.get('ld_coeffs', [0.5, 0.5]), default_unit=u.dimensionless_unscaled, allow_none=True, description='Limb darkening coefficients')]
    passbands.init_passbands()  # NOTE: this only actually does something on the first call
    dep_params += [ChoiceParameter(qualifier='passband', value=kwargs.get('passband', 'Johnson:V'), choices=passbands.list_passbands(), description='Passband')]
    dep_params += [ChoiceParameter(qualifier='intens_weighting', value=kwargs.get('intens_weighting', 'energy'), choices=['energy', 'photon'], description='Whether passband intensities are weighted by energy of photons')]
    if is_lc:
        dep_params += [ChoiceParameter(qualifier='pblum_ref', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('pblum_ref', ''), choices=['self', '']+kwargs.get('starrefs', []), description='Whether to use this components pblum or to couple to that from another component in the system')]
        dep_params += [FloatParameter(qualifier='pblum', visible_if='pblum_ref:self', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('pblum', 4*np.pi), default_unit=u.W, description='Passband luminosity (defined at t0)')]
        dep_params += [FloatParameter(qualifier='l3', value=kwargs.get('l3', 0.), default_unit=u.W/u.m**3, description='Third light')]

    # dep_params += [FloatParameter(qualifier='alb', copy_for={'kind': ['star', 'envelope'], 'component': '*'}, component='_default', value=kwargs.get('alb', 0.), default_unit=u.dimensionless_unscaled, description='Passband Bond\'s albedo, alb=0 is no reflection')]

    if is_lc:
        dep_params += [FloatParameter(qualifier='exptime', value=kwargs.get('exptime', 0.0), default_unit=u.s, description='Exposure time (time is defined as mid-exposure)')]

    return ParameterSet(dep_params)

def rv(**kwargs):
    """
    Create parameters for a new radial velocity dataset.

    Generally, this will be used as an input to the kind argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_dataset`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """

    obs_params = []

    #obs_params += [FloatParameter(qualifier='statweight', value = kwargs.get('statweight', 1.0), default_unit=u.dimensionless_unscaled, description='Statistical weight in overall fitting')]
    syn_params, constraints = rv_syn(syn=False, **kwargs)
    obs_params += syn_params.to_list()
    #obs_params += rv_dep(**kwargs).to_list()

    return ParameterSet(obs_params), constraints

def rv_syn(syn=True, **kwargs):
    """
    """

    syn_params = []

    syn_params += [FloatArrayParameter(qualifier='times', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('times', []), default_unit=u.d, description='Observed times')]
    syn_params += [FloatArrayParameter(qualifier='rvs', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=_empty_array(kwargs, 'rvs'), default_unit=u.km/u.s, description='Observed radial velocity')]
    if not syn:
        syn_params += [FloatArrayParameter(qualifier='sigmas', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=_empty_array(kwargs, 'sigmas'), default_unit=u.km/u.s, description='Observed uncertainty on rv')]


    constraints = []

    return ParameterSet(syn_params), constraints

def rv_dep(**kwargs):
    """
    """

    dep_params = []
    # TODO: only relevent-if rv_method='flux-weighted'

    dep_params += lc_dep(is_lc=False, **kwargs).to_list()


    return ParameterSet(dep_params)


def etv(**kwargs):
    """
    Create parameters for a new eclipse timing variations dataset.

    Generally, this will be used as an input to the kind argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_dataset`

    :parameter **kwargs: default for the values of any of the ParameterSet
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """
    if not conf.devel:
        raise NotImplementedError("'etv' dataset not officially supported for this release.  Enable developer mode to test.")

    obs_params = []

    syn_params, constraints = etv_syn(syn=False, **kwargs)
    obs_params += syn_params.to_list()
    #obs_params += etv_dep(**kwargs).to_list()

    return ParameterSet(obs_params), constraints

def etv_syn(syn=True, **kwargs):
    """
    """

    syn_params = []

    #syn_params += [IntArrayParameter(qualifier='N', value=kwargs.get('N', []), description='Epoch since t0')]
    syn_params += [FloatArrayParameter(qualifier='Ns', copy_for={'kind': ['star', 'envelope'], 'component': '*'}, component='_default', value=_empty_array(kwargs, 'Ns'), default_unit=u.dimensionless_unscaled, description='Epoch since t0')]
    syn_params += [FloatArrayParameter(qualifier='time_ecls', copy_for={'kind': ['star', 'envelope'], 'component': '*'}, component='_default', value=_empty_array(kwargs, 'times_ecl'), default_unit=u.d, description='Time of eclipse')]
    syn_params += [FloatArrayParameter(qualifier='time_ephems', copy_for={'kind': ['star', 'envelope'], 'component': '*'}, component='_default', value=_empty_array(kwargs, 'times_ephem'), default_unit=u.d, description='Expected time of eclipse from the current ephemeris')]
    syn_params += [FloatArrayParameter(qualifier='etvs', copy_for={'kind': ['star', 'envelope'], 'component': '*'}, component='_default', value=_empty_array(kwargs, 'etvs'), default_unit=u.d, description='Eclipse timing variation (time_obs - time_ephem)')]
    if not syn:
        syn_params += [FloatArrayParameter(qualifier='sigmas', value=_empty_array(kwargs, 'sigmas'), default_unit=u.d, description='Observed uncertainty on time_obs')]

    constraints = []
    constraints += [(constraint.time_ephem, kwargs.get('component', '_default'), kwargs.get('dataset', None))]
    constraints += [(constraint.etv, kwargs.get('component', '_default'), kwargs.get('dataset', None))]

    return ParameterSet(syn_params), constraints

def etv_dep(**kwargs):
    """
    """

    dep_params = []

    # TODO: only relevent-if rv_method='flux-weighted'
    # TODO: add these back in if we implement an etv_method that actually needs fluxes
    #dep_params += [ChoiceParameter(qualifier='ld_func', value=kwargs.get('ld_func', 'logarithmic'), choices=_ld_func_choices, description='Limb darkening model')]
    #dep_params += [FloatArrayParameter(qualifier='ld_coeffs', value=kwargs.get('ld_coeffs', [0.5, 0.5]), default_unit=None, description='Limb darkening coefficients')]
    #passbands.init_passbands()  # TODO: possibly move to the import of the passbands module
    #dep_params += [ChoiceParameter(qualifier='passband', value=kwargs.get('passband', 'Johnson:V'), choices=passbands._pbtable.keys(), description='Passband')]


    return ParameterSet(dep_params)

def orb(**kwargs):
    """
    Create parameters for a new orbit dataset.

    Generally, this will be used as an input to the kind argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_dataset`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """

    obs_params = []

    #~ obs_params += [FloatArrayParameter(qualifier='exptime', value=kwargs.get('exptime', []), default_unit=u.s, description='Signal exposure time')]
    #~ obs_params += [FloatArrayParameter(qualifier='flag', value=kwargs.get('flag', []), default_unit=None, description='Signal flag')]
    #~ obs_params += [FloatArrayParameter(qualifier='weight', value=kwargs.get('weight', []), default_unit=None, description='Signal weight')]

    #~ obs_params += [FloatParameter(qualifier='timeoffset', value=kwargs.get('timeoffset', 0.0), default_unit=u.d, description='Zeropoint date offset for observations')]
    #~ obs_params += [FloatParameter(qualifier='statweight', value=kwargs.get('statweight', 0.0), default_unit=None, description='Statistical weight in overall fitting')]

    syn_params, constraints = orb_syn(syn=False, **kwargs)
    obs_params += syn_params.to_list()
    #obs_params += orb_dep(**kwargs).to_list()

    return ParameterSet(obs_params), []

def orb_syn(syn=True, **kwargs):

    syn_params = []

    syn_params += [FloatArrayParameter(qualifier='times', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('times', []), default_unit=u.d, description='{} times'.format('Synthetic' if syn else 'Observed'))]

    if syn:
        # syns ignore copy_for anyways
        syn_params += [FloatArrayParameter(qualifier='xs', value=_empty_array(kwargs, 'xs'), default_unit=u.solRad, description='X position')]
        syn_params += [FloatArrayParameter(qualifier='ys', value=_empty_array(kwargs, 'ys'), default_unit=u.solRad, description='Y position')]
        syn_params += [FloatArrayParameter(qualifier='zs', value=_empty_array(kwargs, 'zs'), default_unit=u.solRad, description='Z position')]
        syn_params += [FloatArrayParameter(qualifier='vxs', value=_empty_array(kwargs, 'vxs'), default_unit=u.solRad/u.d, description='X velocity')]
        syn_params += [FloatArrayParameter(qualifier='vys', value=_empty_array(kwargs, 'vys'), default_unit=u.solRad/u.d, description='Y velocity')]
        syn_params += [FloatArrayParameter(qualifier='vzs', value=_empty_array(kwargs, 'vzs'), default_unit=u.solRad/u.d, description='Z velocity')]

    constraints = []

    return ParameterSet(syn_params), constraints

def orb_dep(**kwargs):

    dep_params = []

    # ltte (per dep and per compute???)???

    return ParameterSet(dep_params)

def mesh(**kwargs):
    """
    Create parameters for a new mesh dataset.

    Generally, this will be used as an input to the kind argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_dataset`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """

    obs_params = []

    syn_params, constraints = mesh_syn(syn=False, **kwargs)
    obs_params += syn_params.to_list()

    obs_params += [SelectParameter(qualifier='include_times', value=kwargs.get('include_times', []), description='append to times from the following datasets/time standards', choices=[])]

    obs_params += [SelectParameter(qualifier='datasets', value=kwargs.get('datasets', []), description='datasets to expose as mesh columns', choices=[])]
    columns_choices = []

    columns_choices += ['pot', 'rpole', 'volume']

    columns_choices += ['xs', 'ys', 'zs']
    columns_choices += ['roche_xs', 'roche_ys', 'roche_zs']
    columns_choices += ['vxs', 'vys', 'vzs']
    # columns_choices += ['horizon_xs', 'horizon_ys', 'horizon_zs', 'horizon_analytic_xs', 'horizon_analytic_ys', 'horizon_analytic_zs']
    columns_choices += ['areas'] #, 'tareas']
    columns_choices += ['normals', 'nxs', 'nys', 'nzs']
    columns_choices += ['loggs', 'teffs']

    columns_choices += ['rprojs', 'mus', 'visibilities', 'visible_centroids']
    columns_choices += ['rs'] #, 'cosbetas']

    columns_choices += ['intensities', 'normal_intensities', 'abs_intensities', 'abs_normal_intensities']
    columns_choices += ['boost_factors', 'ldint']
    columns_choices += ['rvs']
    columns_choices += ['pblum', 'ptfarea']

    # TODO: split this into columns and dataset_columns???
    obs_params += [SelectParameter(qualifier='columns', value=kwargs.get('columns', ['teffs']), description='columns to expose within the mesh', choices=columns_choices)]
    #obs_params += mesh_dep(**kwargs).to_list()

    return ParameterSet(obs_params), constraints

def mesh_syn(syn=True, **kwargs):

    syn_params = []

    times = kwargs.get('times', [])

    syn_params += [FloatArrayParameter(qualifier='times', value=kwargs.get('times', []), default_unit=u.d, description='{} times'.format('Synthetic' if syn else 'Observed'))]

    # the following will all be arrays (value per triangle) per time
    if syn:
        columns = kwargs.get('columns', [])
        datasets = kwargs.get('datasets', [])

        for t in times:
            if not isinstance(t, float):
                raise ValueError("times must all be of type float")


            # always include basic geometric columns
            syn_params += [FloatArrayParameter(qualifier='vertices', time=t, value=kwargs.get('vertices', []), default_unit=u.solRad, description='Vertices of triangles in the plane-of-sky')]
            syn_params += [FloatArrayParameter(qualifier='roche_vertices', time=t, value=kwargs.get('roche_vertices', []), default_unit=u.dimensionless_unscaled, description='Vertices of triangles in Roche coordinates')]

            # NOTE: if changing the parameters which are optional, changes must
            # be made here, in the choices for the columns Parameter, and in
            # backends.py when the values are extracted and included in the
            # packet

            if 'pot' in columns:
                syn_params += [FloatParameter(qualifier='pot', time=t, value=kwargs.get('pot', 0.0), default_unit=u.dimensionless_unscaled, description='Equipotential of the stellar surface')]
            if 'rpole' in columns:
                syn_params += [FloatParameter(qualifier='rpole', time=t, value=kwargs.get('rpole', 0.0), default_unit=u.solRad, description='Polar radius of the stellar surface')]
            if 'volume' in columns:
                syn_params += [FloatParameter(qualifier='volume', time=t, value=kwargs.get('volume', 0.0), default_unit=u.solRad**3, description='Volume of the stellar surface')]


            if 'xs' in columns:
                syn_params += [FloatArrayParameter(qualifier='xs', time=t, value=kwargs.get('xs', []), default_unit=u.solRad, description='X coordinate of center of triangles in the plane-of-sky')]
            if 'ys' in columns:
                syn_params += [FloatArrayParameter(qualifier='ys', time=t, value=kwargs.get('ys', []), default_unit=u.solRad, description='Y coordinate of center of triangles in the plane-of-sky')]
            if 'zs' in columns:
                syn_params += [FloatArrayParameter(qualifier='zs', time=t, value=kwargs.get('zs', []), default_unit=u.solRad, description='Z coordinate of center of triangles in the plane-of-sky')]

            if 'roche_xs' in columns:
                syn_params += [FloatArrayParameter(qualifier='roche_xs', time=t, value=kwargs.get('roche_xs', []), default_unit=u.dimensionless_unscaled, description='X coordinate of center of triangles in Roche coordinates')]
            if 'roche_ys' in columns:
                syn_params += [FloatArrayParameter(qualifier='roche_ys', time=t, value=kwargs.get('roche_ys', []), default_unit=u.dimensionless_unscaled, description='Y coordinate of center of triangles in Roche coordinates')]
            if 'roche_zs' in columns:
                syn_params += [FloatArrayParameter(qualifier='roche_zs', time=t, value=kwargs.get('roche_zs', []), default_unit=u.dimensionless_unscaled, description='Z coordinate of center of triangles in Roche coordinates')]


            if 'vxs' in columns:
                syn_params += [FloatArrayParameter(qualifier='vxs', time=t, value=kwargs.get('vxs', []), default_unit=u.solRad/u.d, description='X velocity of center of triangles')]
            if 'vys' in columns:
                syn_params += [FloatArrayParameter(qualifier='vys', time=t, value=kwargs.get('vys', []), default_unit=u.solRad/u.d, description='Y velocity of center of triangles')]
            if 'vzs' in columns:
                syn_params += [FloatArrayParameter(qualifier='vzs', time=t, value=kwargs.get('vzs', []), default_unit=u.solRad/u.d, description='Z velocity of center of triangles')]

            if 'areas' in columns:
                syn_params += [FloatArrayParameter(qualifier='areas', time=t, value=kwargs.get('areas', []), default_unit=u.solRad**2, description='Area of triangles')]
            # if 'tareas' in columns:
                # syn_params += [FloatArrayParameter(qualifier='tareas', time=t, value=kwargs.get('areas', []), default_unit=u.solRad**2, description='Area of WD triangles')]

            if 'normals' in columns:
                syn_params += [FloatArrayParameter(qualifier='normals', time=t, value=kwargs.get('normals', []), default_unit=u.dimensionless_unscaled, description='Normals of triangles')]
            if 'nxs' in columns:
                syn_params += [FloatArrayParameter(qualifier='nxs', time=t, value=kwargs.get('nxs', []), default_unit=u.dimensionless_unscaled, description='X component of normals')]
            if 'nys' in columns:
                syn_params += [FloatArrayParameter(qualifier='nys', time=t, value=kwargs.get('nys', []), default_unit=u.dimensionless_unscaled, description='Y component of normals')]
            if 'nzs' in columns:
                syn_params += [FloatArrayParameter(qualifier='nzs', time=t, value=kwargs.get('nzs', []), default_unit=u.dimensionless_unscaled, description='Z component of normals')]

            if 'rs' in columns:
                syn_params += [FloatArrayParameter(qualifier='rs', time=t, value=kwargs.get('rs', []), default_unit=u.solRad if t is not None else u.dimensionless_unscaled, description='Distance of each triangle from center of mass')]
            # if 'cosbetas' in columns:
            #     syn_params += [FloatArrayParameter(qualifier='cosbetas', time=t, value=kwargs.get('cosbetas', []), default_unit=u.solRad if t is not None else u.dimensionless_unscaled, description='')]


            if 'loggs' in columns:
                syn_params += [FloatArrayParameter(qualifier='loggs', time=t, value=kwargs.get('loggs', []), default_unit=u.dimensionless_unscaled, description='Local surface gravity')]
            if 'teffs' in columns:
                syn_params += [FloatArrayParameter(qualifier='teffs', time=t, value=kwargs.get('teffs', []), default_unit=u.K, description='Local effective temperature')]

            if 'r_projs' in columns:
                syn_params += [FloatArrayParameter(qualifier='r_projs', time=t, value=kwargs.get('r_projs', []), default_unit=u.solRad, description='Projected distance (on plane of sky) of each triangle from center of mass')]
            if 'mus' in columns:
                syn_params += [FloatArrayParameter(qualifier='mus', time=t, value=kwargs.get('mus', []), default_unit=u.dimensionless_unscaled, description='Mu')]
            if 'visible_centroids' in columns:
                syn_params += [FloatArrayParameter(qualifier='visible_centroids', time=t, value=kwargs.get('visible_centroids', []), default_unit=u.solRad  if t is not None else u.dimensionless_unscaled, description='Center of the visible portion of each triangle')]
            if 'visibilities' in columns:
                syn_params += [FloatArrayParameter(qualifier='visibilities', time=t, value=kwargs.get('visibilities', []), default_unit=u.dimensionless_unscaled, description='Visiblity of triangles (1=visible, 0.5=partial, 0=hidden)')]

            # syn_params += [FloatArrayParameter(qualifier='horizon_xs', time=t, value=kwargs.get('horizon_xs', []), default_unit=u.solRad, description='Horizon of the mesh (x component)')]
            # syn_params += [FloatArrayParameter(qualifier='horizon_ys', time=t, value=kwargs.get('horizon_ys', []), default_unit=u.solRad, description='Horizon of the mesh (y component)')]
            # syn_params += [FloatArrayParameter(qualifier='horizon_zs', time=t, value=kwargs.get('horizon_zs', []), default_unit=u.solRad, description='Horizon of the mesh (z component)')]
            # syn_params += [FloatArrayParameter(qualifier='horizon_analytic_xs', time=t, value=kwargs.get('horizon_analytic_xs', []), default_unit=u.solRad, description='Analytic horizon (interpolated, x component)')]
            # syn_params += [FloatArrayParameter(qualifier='horizon_analytic_ys', time=t, value=kwargs.get('horizon_analytic_ys', []), default_unit=u.solRad, description='Analytic horizon (interpolated, y component)')]
            # syn_params += [FloatArrayParameter(qualifier='horizon_analytic_zs', time=t, value=kwargs.get('horizon_analytic_zs', []), default_unit=u.solRad, description='Analytic horizon (interpolated, z component)')]

            for dataset, kind in datasets.items():
                if 'rvs' in columns and kind=='rv':
                    syn_params += [FloatArrayParameter(qualifier='rvs', dataset=dataset, time=t, value=[], default_unit=u.solRad/u.d, description='Per-element value of rvs for {} dataset'.format(dataset))]
                if 'intensities' in columns:
                    syn_params += [FloatArrayParameter(qualifier='intensities', dataset=dataset, time=t, value=[], default_unit=u.W/u.m**3, description='Per-element value of intensities for {} dataset'.format(dataset))]
                if 'normal_intensities' in columns:
                    syn_params += [FloatArrayParameter(qualifier='normal_intensities', dataset=dataset, time=t, value=[], default_unit=u.W/u.m**3, description='Per-element value of normal_intensities for {} dataset'.format(dataset))]
                if 'abs_intensities' in columns:
                    syn_params += [FloatArrayParameter(qualifier='abs_intensities', dataset=dataset, time=t, value=[], default_unit=u.W/u.m**3, description='Per-element value of abs_intensities for {} dataset'.format(dataset))]
                if 'abs_normal_intensities' in columns:
                    syn_params += [FloatArrayParameter(qualifier='abs_normal_intensities', dataset=dataset, time=t, value=[], default_unit=u.W/u.m**3, description='Per-element value of abs_normal_intensities for {} dataset'.format(dataset))]
                if 'boost_factors' in columns:
                    syn_params += [FloatArrayParameter(qualifier='boost_factors', dataset=dataset, time=t, value=[], default_unit=u.dimensionless_unscaled, description='Per-element value of boost_factors for {} dataset'.format(dataset))]
                if 'ldint' in columns:
                    syn_params += [FloatArrayParameter(qualifier='ldint', dataset=dataset, time=t, value=kwargs.get('ldint', []), default_unit=u.dimensionless_unscaled, description='Integral of the limb-darkening function')]

                if 'ptfarea' in columns:
                    syn_params += [FloatParameter(qualifier='ptfarea', dataset=dataset, time=t, value=kwargs.get('ptfarea', 1.0), default_unit=u.m, description='Area of the passband transmission function')]
                if 'pblum' in columns:
                    syn_params += [FloatParameter(qualifier='pblum', dataset=dataset, time=t, value=kwargs.get('pblum', 0.0), default_unit=u.W, description='Passband Luminosity of entire star')]

    constraints = []

    return ParameterSet(syn_params), constraints

def mesh_dep(**kwargs):

    dep_params = []

    return ParameterSet(dep_params)
