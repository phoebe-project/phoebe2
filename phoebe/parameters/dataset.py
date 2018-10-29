
from phoebe.parameters import *
from phoebe.parameters import constraint
from phoebe.atmospheres import passbands  # need to load pbtable (dictionary of available passbands)
from phoebe import u
from phoebe import conf

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

_ld_func_choices = ['interp', 'linear', 'logarithmic', 'quadratic', 'square_root', 'power']

global _mesh_columns
global _pbdep_columns

_mesh_columns = []
# _mesh_columns += ['pot', 'rpole']
_mesh_columns += ['volume']

_mesh_columns += ['xs', 'ys', 'zs']
_mesh_columns += ['vxs', 'vys', 'vzs']
_mesh_columns += ['nxs', 'nys', 'nzs']

_mesh_columns += ['us', 'vs', 'ws']
_mesh_columns += ['vus', 'vvs', 'vws']
_mesh_columns += ['nus', 'nvs', 'nws']

# _mesh_columns += ['horizon_xs', 'horizon_ys', 'horizon_zs', 'horizon_analytic_xs', 'horizon_analytic_ys', 'horizon_analytic_zs']
_mesh_columns += ['areas'] #, 'tareas']
_mesh_columns += ['loggs', 'teffs']

_mesh_columns += ['rprojs', 'mus', 'visibilities', 'visible_centroids']
_mesh_columns += ['rs'] #, 'cosbetas']

lc_columns = []
lc_columns += ['intensities', 'normal_intensities', 'abs_intensities', 'abs_normal_intensities']
lc_columns += ['boost_factors', 'ldint']
lc_columns += ['pblum', 'abs_pblum', 'ptfarea']

rv_columns = lc_columns[:]
rv_columns += ['rvs']

lp_columns = rv_columns[:]
# lp_columns += ['dls']


_pbdep_columns = {'lc': lc_columns,
                  'rv': rv_columns,
                  'lp': lp_columns}

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
    dep_params += [ChoiceParameter(qualifier='ld_func', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('ld_func', 'interp'), choices=_ld_func_choices, description='Limb darkening model')]
    dep_params += [FloatArrayParameter(qualifier='ld_coeffs', visible_if='ld_func:!interp', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('ld_coeffs', [0.5, 0.5]), default_unit=u.dimensionless_unscaled, allow_none=True, description='Limb darkening coefficients')]
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
    syn_params += [FloatArrayParameter(qualifier='rvs', visible_if='times:<notempty>', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=_empty_array(kwargs, 'rvs'), default_unit=u.km/u.s, description='Observed radial velocity')]
    if not syn:
        syn_params += [FloatArrayParameter(qualifier='sigmas', visible_if='times:<notempty>', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=_empty_array(kwargs, 'sigmas'), default_unit=u.km/u.s, description='Observed uncertainty on rv')]


    constraints = []

    return ParameterSet(syn_params), constraints

def rv_dep(**kwargs):
    """
    """

    dep_params = []
    # TODO: only relevent-if rv_method='flux-weighted'

    dep_params += lc_dep(is_lc=False, **kwargs).to_list()


    return ParameterSet(dep_params)

def lp(**kwargs):
    """
    Create parameters for a new line profile dataset.

    Generally, this will be used as an input to the kind argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_dataset`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """

    obs_params = []

    #obs_params += [FloatParameter(qualifier='statweight', value = kwargs.get('statweight', 1.0), default_unit=u.dimensionless_unscaled, description='Statistical weight in overall fitting')]
    syn_params, constraints = lp_syn(syn=False, **kwargs)
    obs_params += syn_params.to_list()
    #obs_params += rv_dep(**kwargs).to_list()

    return ParameterSet(obs_params), constraints

def lp_syn(syn=True, **kwargs):
    """
    """

    times = kwargs.get('times', [])

    syn_params = []

    # if syn:
        # wavelengths array is copied per-time for the model
        # for time in times:
            # syn_params += [FloatArrayParameter(qualifier='wavelengths', copy_for={'kind': ['star', 'envelope', 'orbit'], 'component': '*'}, component='_default', time=time, value=_empty_array(kwargs, 'wavelengths'), default_unit=u.nm, description='Wavelengths of the observations')]

    # else:
    # wavelengths is time-independent
    syn_params += [FloatArrayParameter(qualifier='wavelengths', copy_for={'kind': ['star', 'orbit'], 'component': '*'}, component='_default', value=_empty_array(kwargs, 'wavelengths'), default_unit=u.nm, description='Wavelengths of the observations')]

    for time in times:
        # but do allow per-component flux_densities and sigmas
        syn_params += [FloatArrayParameter(qualifier='flux_densities', visible_if='[time]wavelengths:<notempty>', copy_for={'kind': ['star', 'orbit'], 'component': '*'}, component='_default', time=time, value=_empty_array(kwargs, 'flux_densities'), default_unit=u.W/(u.m**2*u.nm), description='Flux density per wavelength (must be same length as wavelengths or empty)')]
        if not syn:
            syn_params += [FloatArrayParameter(qualifier='sigmas', visible_if='[time]wavelengths:<notempty>', copy_for={'kind': ['star', 'orbit'], 'component': '*'}, component='_default', time=time, value=_empty_array(kwargs, 'sigmas'), default_unit=u.W/(u.m**2*u.nm), description='Observed uncertainty on flux_densities')]

    constraints = []

    return ParameterSet(syn_params), constraints

def lp_dep(**kwargs):
    """
    """

    dep_params = []

    dep_params += lc_dep(is_lc=False, **kwargs).to_list()

    dep_params += [ChoiceParameter(qualifier='profile_func', value=kwargs.get('profile_func', 'gaussian'), choices=['gaussian', 'lorentzian'], description='Function to use for the rest line profile')]
    dep_params += [FloatParameter(qualifier='profile_rest', value=kwargs.get('profile_rest', 550), default_unit=u.nm, limits=(0, None), description='Rest central wavelength of the profile')]
    dep_params += [FloatParameter(qualifier='profile_sv', value=kwargs.get('profile_sv', 1e-4), default_unit=u.dimensionless_unscaled, limits=(0, None), description='Subsidiary value of the profile')]

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
        syn_params += [FloatArrayParameter(qualifier='us', value=_empty_array(kwargs, 'us'), default_unit=u.solRad, description='U position')]
        syn_params += [FloatArrayParameter(qualifier='vs', value=_empty_array(kwargs, 'vs'), default_unit=u.solRad, description='V position')]
        syn_params += [FloatArrayParameter(qualifier='ws', value=_empty_array(kwargs, 'ws'), default_unit=u.solRad, description='W position')]
        syn_params += [FloatArrayParameter(qualifier='vus', value=_empty_array(kwargs, 'vus'), default_unit=u.km/u.s, description='U velocity')]
        syn_params += [FloatArrayParameter(qualifier='vvs', value=_empty_array(kwargs, 'vvs'), default_unit=u.km/u.s, description='V velocity')]
        syn_params += [FloatArrayParameter(qualifier='vws', value=_empty_array(kwargs, 'vws'), default_unit=u.km/u.s, description='W velocity')]

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

    obs_params += [SelectParameter(qualifier='include_times', value=kwargs.get('include_times', []), description='append to times from the following datasets/time standards', choices=['t0@system'])]

    obs_params += [SelectParameter(qualifier='columns', value=kwargs.get('columns', []), description='columns to expose within the mesh', choices=_mesh_columns)]
    #obs_params += mesh_dep(**kwargs).to_list()

    return ParameterSet(obs_params), constraints

def mesh_syn(syn=True, **kwargs):

    syn_params = []

    times = kwargs.get('times', [])

    syn_params += [FloatArrayParameter(qualifier='times', value=kwargs.get('times', []), default_unit=u.d, description='{} times'.format('Synthetic' if syn else 'Observed'))]

    # the following will all be arrays (value per triangle) per time
    if syn:
        columns = kwargs.get('mesh_columns', [])
        mesh_datasets = kwargs.get('mesh_datasets', [])

        for t in times:
            if not isinstance(t, float):
                raise ValueError("times must all be of type float")


            # always include basic geometric columns
            syn_params += [FloatArrayParameter(qualifier='uvw_elements', time=t, value=kwargs.get('uvw_elements', []), default_unit=u.solRad, description='Vertices of triangles in the plane-of-sky')]
            syn_params += [FloatArrayParameter(qualifier='xyz_elements', time=t, value=kwargs.get('xyz_elements ', []), default_unit=u.dimensionless_unscaled, description='Vertices of triangles in Roche coordinates')]

            # NOTE: if changing the parameters which are optional, changes must
            # be made here, in the choices for the columns Parameter, and in
            # backends.py when the values are extracted and included in the
            # packet

            # if 'pot' in columns:
                # syn_params += [FloatParameter(qualifier='pot', time=t, value=kwargs.get('pot', 0.0), default_unit=u.dimensionless_unscaled, description='Equipotential of the stellar surface')]
            # if 'rpole' in columns:
            #     syn_params += [FloatParameter(qualifier='rpole', time=t, value=kwargs.get('rpole', 0.0), default_unit=u.solRad, description='Polar radius of the stellar surface')]
            if 'volume' in columns:
                syn_params += [FloatParameter(qualifier='volume', time=t, value=kwargs.get('volume', 0.0), default_unit=u.solRad**3, description='Volume of the stellar surface')]


            if 'xs' in columns:
                syn_params += [FloatArrayParameter(qualifier='xs', time=t, value=kwargs.get('xs', []), default_unit=u.solRad, description='X coordinate of center of triangles in the plane-of-sky')]
            if 'ys' in columns:
                syn_params += [FloatArrayParameter(qualifier='ys', time=t, value=kwargs.get('ys', []), default_unit=u.solRad, description='Y coordinate of center of triangles in the plane-of-sky')]
            if 'zs' in columns:
                syn_params += [FloatArrayParameter(qualifier='zs', time=t, value=kwargs.get('zs', []), default_unit=u.solRad, description='Z coordinate of center of triangles in the plane-of-sky')]

            if 'vxs' in columns:
                syn_params += [FloatArrayParameter(qualifier='vxs', time=t, value=kwargs.get('vxs', []), default_unit=u.km/u.s, description='X velocity of center of triangles')]
            if 'vys' in columns:
                syn_params += [FloatArrayParameter(qualifier='vys', time=t, value=kwargs.get('vys', []), default_unit=u.km/u.s, description='Y velocity of center of triangles')]
            if 'vzs' in columns:
                syn_params += [FloatArrayParameter(qualifier='vzs', time=t, value=kwargs.get('vzs', []), default_unit=u.km/u.s, description='Z velocity of center of triangles')]

            if 'nxs' in columns:
                syn_params += [FloatArrayParameter(qualifier='nxs', time=t, value=kwargs.get('nxs', []), default_unit=u.dimensionless_unscaled, description='X component of normals')]
            if 'nys' in columns:
                syn_params += [FloatArrayParameter(qualifier='nys', time=t, value=kwargs.get('nys', []), default_unit=u.dimensionless_unscaled, description='Y component of normals')]
            if 'nzs' in columns:
                syn_params += [FloatArrayParameter(qualifier='nzs', time=t, value=kwargs.get('nzs', []), default_unit=u.dimensionless_unscaled, description='Z component of normals')]

            if 'us' in columns:
                syn_params += [FloatArrayParameter(qualifier='us', time=t, value=kwargs.get('us', []), default_unit=u.solRad, description='U coordinate of center of triangles in the plane-of-sky')]
            if 'vs' in columns:
                syn_params += [FloatArrayParameter(qualifier='vs', time=t, value=kwargs.get('vs', []), default_unit=u.solRad, description='V coordinate of center of triangles in the plane-of-sky')]
            if 'ws' in columns:
                syn_params += [FloatArrayParameter(qualifier='ws', time=t, value=kwargs.get('ws', []), default_unit=u.solRad, description='W coordinate of center of triangles in the plane-of-sky')]

            if 'vus' in columns:
                syn_params += [FloatArrayParameter(qualifier='vus', time=t, value=kwargs.get('vus', []), default_unit=u.km/u.s, description='U velocity of center of triangles')]
            if 'vvs' in columns:
                syn_params += [FloatArrayParameter(qualifier='vvs', time=t, value=kwargs.get('vvs', []), default_unit=u.km/u.s, description='V velocity of center of triangles')]
            if 'vws' in columns:
                syn_params += [FloatArrayParameter(qualifier='vws', time=t, value=kwargs.get('vws', []), default_unit=u.km/u.s, description='W velocity of center of triangles')]

            if 'nus' in columns:
                syn_params += [FloatArrayParameter(qualifier='nus', time=t, value=kwargs.get('nus', []), default_unit=u.dimensionless_unscaled, description='U component of normals')]
            if 'nvs' in columns:
                syn_params += [FloatArrayParameter(qualifier='nvs', time=t, value=kwargs.get('nvs', []), default_unit=u.dimensionless_unscaled, description='V component of normals')]
            if 'nws' in columns:
                syn_params += [FloatArrayParameter(qualifier='nws', time=t, value=kwargs.get('nws', []), default_unit=u.dimensionless_unscaled, description='W component of normals')]


            if 'areas' in columns:
                syn_params += [FloatArrayParameter(qualifier='areas', time=t, value=kwargs.get('areas', []), default_unit=u.solRad**2, description='Area of triangles')]
            # if 'tareas' in columns:
                # syn_params += [FloatArrayParameter(qualifier='tareas', time=t, value=kwargs.get('areas', []), default_unit=u.solRad**2, description='Area of WD triangles')]


            if 'rs' in columns:
                syn_params += [FloatArrayParameter(qualifier='rs', time=t, value=kwargs.get('rs', []), default_unit=u.solRad, description='Distance of each triangle from center of mass (of the half-envelope for contacts)')]
            # if 'cosbetas' in columns:
            #     syn_params += [FloatArrayParameter(qualifier='cosbetas', time=t, value=kwargs.get('cosbetas', []), default_unit=u.solRad, description='')]


            if 'loggs' in columns:
                syn_params += [FloatArrayParameter(qualifier='loggs', time=t, value=kwargs.get('loggs', []), default_unit=u.dimensionless_unscaled, description='Local surface gravity')]
            if 'teffs' in columns:
                syn_params += [FloatArrayParameter(qualifier='teffs', time=t, value=kwargs.get('teffs', []), default_unit=u.K, description='Local effective temperature')]

            if 'rprojs' in columns:
                syn_params += [FloatArrayParameter(qualifier='rprojs', time=t, value=kwargs.get('rprojs', []), default_unit=u.solRad, description='Projected distance (on plane of sky) of each triangle from center of mass (of the half-envelope for contacts)')]
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

            for dataset in mesh_datasets:
                # if 'dls@{}'.format(dataset) in columns:
                    # syn_params += [FloatArrayParameter(qualifier='dls', dataset=dataset, time=t, value=[], default_unit=u.nm, description='Per-element delta-lambda caused by doppler shift'.format(dataset))]
                if 'rvs@{}'.format(dataset) in columns:
                    syn_params += [FloatArrayParameter(qualifier='rvs', dataset=dataset, time=t, value=[], default_unit=u.km/u.s, description='Per-element value of rvs for {} dataset'.format(dataset))]
                if 'intensities@{}'.format(dataset) in columns:
                    syn_params += [FloatArrayParameter(qualifier='intensities', dataset=dataset, time=t, value=[], default_unit=u.W/u.m**3, description='Per-element value of intensities for {} dataset'.format(dataset))]
                if 'normal_intensities@{}'.format(dataset) in columns:
                    syn_params += [FloatArrayParameter(qualifier='normal_intensities', dataset=dataset, time=t, value=[], default_unit=u.W/u.m**3, description='Per-element value of normal_intensities for {} dataset'.format(dataset))]
                if 'abs_intensities@{}'.format(dataset) in columns:
                    syn_params += [FloatArrayParameter(qualifier='abs_intensities', dataset=dataset, time=t, value=[], default_unit=u.W/u.m**3, description='Per-element value of abs_intensities for {} dataset'.format(dataset))]
                if 'abs_normal_intensities@{}'.format(dataset) in columns:
                    syn_params += [FloatArrayParameter(qualifier='abs_normal_intensities', dataset=dataset, time=t, value=[], default_unit=u.W/u.m**3, description='Per-element value of abs_normal_intensities for {} dataset'.format(dataset))]
                if 'boost_factors@{}'.format(dataset) in columns:
                    syn_params += [FloatArrayParameter(qualifier='boost_factors', dataset=dataset, time=t, value=[], default_unit=u.dimensionless_unscaled, description='Per-element value of boost_factors for {} dataset'.format(dataset))]
                if 'ldint@{}'.format(dataset) in columns:
                    syn_params += [FloatArrayParameter(qualifier='ldint', dataset=dataset, time=t, value=kwargs.get('ldint', []), default_unit=u.dimensionless_unscaled, description='Integral of the limb-darkening function')]

                if 'ptfarea@{}'.format(dataset) in columns:
                    syn_params += [FloatParameter(qualifier='ptfarea', dataset=dataset, time=t, value=kwargs.get('ptfarea', 1.0), default_unit=u.m, description='Area of the passband transmission function')]
                if 'pblum@{}'.format(dataset) in columns:
                    syn_params += [FloatParameter(qualifier='pblum', dataset=dataset, time=t, value=kwargs.get('pblum', 0.0), default_unit=u.W, description='Passband Luminosity of entire star (after pblum scaling)')]
                if 'abs_pblum@{}'.format(dataset) in columns:
                    syn_params += [FloatParameter(qualifier='abs_pblum', dataset=dataset, time=t, value=kwargs.get('abs_pblum', 0.0), default_unit=u.W, description='Passband Luminosity of entire star (before pblum scaling)')]

    constraints = []

    return ParameterSet(syn_params), constraints

def mesh_dep(**kwargs):

    dep_params = []

    return ParameterSet(dep_params)
