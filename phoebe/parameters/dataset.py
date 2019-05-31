
from phoebe.parameters import *
from phoebe.parameters import constraint
from phoebe.atmospheres import passbands  # need to load pbtable (dictionary of available passbands)
from phoebe import u
from phoebe import conf

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

_ld_func_choices = ['interp', 'linear', 'logarithmic', 'quadratic', 'square_root', 'power']


passbands._init_passbands()  # TODO: move to module import
_ld_coeffs_source_choices = ['none', 'auto'] + list(set([atm for pb in passbands._pbtable.values() for atm in pb['atms_ld']]))

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
lc_columns += ['pblum_ext', 'abs_pblum_ext', 'ptfarea']

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

def lc(syn=False, as_ps=True, is_lc=True, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for a light curve dataset.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_dataset> as
    `b.add_dataset('lc')`.  In this case, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Arguments
    ----------
    * `syn` (bool, optional, default=False): whether to create the parameters
        for the synthetic (model) instead of the observational (dataset).
    * `as_ps` (bool, optional, default=True): whether to return the parameters
        as a <phoebe.parameters.ParameterSet> instead of a list of
        <phoebe.parameters.Parameter> objects.
    * `is_lc` (bool, optional, default=True): whether to build all lc parameters
        or just passband-dependent parameters.  Generally this is only used
        internally by other dataset types that need passband-dependent parameters.
    * `times` (array/quantity, optional): observed times.  Only applicable
        if `is_lc` is True.
    * `fluxes` (array/quantity, optional): observed flux.  Only applicable
        if `is_lc` is True.
    * `sigmas` (array/quantity, optional): errors on flux measurements.  Only
        applicable if `syn` is False and `is_lc` is True.
    * `compute_times` (array/quantity, optional): times at which to compute
        the model.  Only applicable if `syn` is False and `is_lc` is True.
    * `compute_phases` (array/quantity, optional): phases at which to compute
        the model.  Only applicable if `syn` is False and `is_lc` is True.
    * `ld_func` (string, optional): limb-darkening model.  Only applicable
        if `syn` is False.
    * `ld_coeffs_source` (string, optional, default='auto'): source for limb-darkening
        coefficients ('none' to provide manually, 'auto' to interpolate from
        the applicable table according to the 'atm' parameter, or the name of
        a specific atmosphere table).  Not applicable if `ld_func` is 'interp'.
        Only applicable if `syn` is False.
    * `ld_coeffs` (list, optional): limb-darkening coefficients.  Only applicable
       if `ld_coeffs_source` is 'none' (and therefore `ld_func` is not 'interp').
    * `passband` (string, optional): passband.  Only applicable if `syn` is False.
    * `intens_weighting` (string, optional): whether passband intensities are
        weighted by energy of photons.  Only applicable if `syn` is False.
    * `pblum_mode` (string, optional, default='provided'): mode for scaling
        passband luminosities.  Only applicable if `syn` is False and `is_lc`
        is True.
    * `pblum_ref` (string, optional): whether to use this components pblum or to
        couple to that from another component in the system.  Only applicable
        if `pblum_mode` is 'provided'.  Only applicable if `syn` is False and
        `is_lc` is True.
    * `pblum` (float/quantity, optional): passband luminosity (defined at t0).
        Only applicable if `syn` is False and `is_lc` is True.
    * `l3_mode` (string, optional, default='flux'): mode for providing third
        light (`l3`).  Only applicable if `syn` is False and `is_lc` is True.
    * `l3` (float/quantity, optional): third light in flux units (only applicable
        if `l3_mode` is 'flux'). Only applicable if `syn` is False and `is_lc`
        is True.
    * `l3_frac` (float/quantity, optional): third light in fraction of total light
        (only applicable if `l3_mode` is 'fraction of total light').
        Only applicable if `syn` is False and `is_lc` is True.
    * `exptime` (float/quantity, optional): exposure time of the observations
        (`times` is defined at the mid-exposure).
        Only applicable if `syn` is False and `is_lc` is True.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet> or list, list): ParameterSet (if `as_ps`)
        or list of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params, constraints = [], []


    if is_lc:
        params += [FloatArrayParameter(qualifier='times', value=kwargs.get('times', []), default_unit=u.d, description='Observed times')]
        params += [FloatArrayParameter(qualifier='fluxes', value=_empty_array(kwargs, 'fluxes'), default_unit=u.W/u.m**2, description='Observed flux')]

    if not syn:
        params += [ChoiceParameter(qualifier='ld_func', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('ld_func', 'interp'), choices=_ld_func_choices, description='Limb darkening model')]
        params += [ChoiceParameter(qualifier='ld_coeffs_source', visible_if='ld_func:!interp', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('ld_coeffs_source', 'auto'), choices=_ld_coeffs_source_choices, description='Source for limb darkening coefficients (\'none\' to provide manually, \'auto\' to interpolate from the applicable table according to the \'atm\' parameter, or the name of a specific atmosphere table)')]
        params += [FloatArrayParameter(qualifier='ld_coeffs', visible_if='ld_func:!interp,ld_coeffs_source:none', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('ld_coeffs', [0.5, 0.5]), default_unit=u.dimensionless_unscaled, description='Limb darkening coefficients')]
        passbands._init_passbands()  # NOTE: this only actually does something on the first call
        params += [ChoiceParameter(qualifier='passband', value=kwargs.get('passband', 'Johnson:V'), choices=passbands.list_passbands(), description='Passband')]
        params += [ChoiceParameter(qualifier='intens_weighting', value=kwargs.get('intens_weighting', 'energy'), choices=['energy', 'photon'], description='Whether passband intensities are weighted by energy of photons')]

    if is_lc and not syn:
        params += [FloatArrayParameter(qualifier='compute_times', value=kwargs.get('compute_times', []), default_unit=u.d, description='Times to use during run_compute.  If empty, will use times parameter')]
        params += [FloatArrayParameter(qualifier='compute_phases', component=kwargs.get('component_top', None), value=kwargs.get('compute_phases', []), default_unit=u.dimensionless_unscaled, description='Phases associated with compute_times.  Does not account for t0: for true phases, use b.to_phase or b.to_time')]

        constraints += [(constraint.compute_phases, kwargs.get('component_top', None), kwargs.get('dataset', None))]

        params += [FloatArrayParameter(qualifier='sigmas', value=_empty_array(kwargs, 'sigmas'), default_unit=u.W/u.m**2, description='Observed uncertainty on flux')]

        params += [ChoiceParameter(qualifier='pblum_mode', value=kwargs.get('pblum_mode', 'provided'),
                                   choices=['provided', 'color coupled', 'system flux', 'total flux', 'scale to data', 'absolute'] if conf.devel else ['provided', 'color coupled', 'total flux', 'scale to data', 'absolute'],
                                   description='Mode for scaling passband luminosities')]

        params += [ChoiceParameter(visible_if='[component]pblum_mode:provided', qualifier='pblum_ref', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('pblum_ref', ''), choices=['self', '']+kwargs.get('starrefs', []), description='Whether to use this components pblum or to couple to that from another component in the system')]
        params += [FloatParameter(qualifier='pblum', visible_if='[component]pblum_mode:provided,pblum_ref:self', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('pblum', 4*np.pi), default_unit=u.W, description='Passband luminosity (defined at t0)')]

        params += [ChoiceParameter(visible_if='pblum_mode:color coupled', qualifier='pblum_ref', value=kwargs.get('pblum_ref', ''), choices=['']+kwargs.get('lcrefs', []), description='Dataset with which to couple luminosities based on color')]

        params += [FloatParameter(visible_if='pblum_mode:system flux|total flux', qualifier='pbflux', value=kwargs.get('pbflux', 1.0), default_unit=u.W/u.m**2, description='Total inrinsic (excluding features and irradiation) passband flux (at t0, including l3 if pblum_mode=\'total flux\')')]

        params += [ChoiceParameter(qualifier='l3_mode', value=kwargs.get('l3_mode', 'flux'), choices=['flux', 'fraction of total light'], description='Whether third light is given in units of flux or as a fraction of total light')]
        params += [FloatParameter(visible_if='l3_mode:flux', qualifier='l3', value=kwargs.get('l3', 0.), limits=[0, None], default_unit=u.W/u.m**2, description='Third light in flux units')]
        params += [FloatParameter(visible_if='l3_mode:fraction of total light', qualifier='l3_frac', value=kwargs.get('l3_frac', 0.), limits=[0, 1], default_unit=u.dimensionless_unscaled, description='Third light as a fraction of total light')]

        params += [FloatParameter(qualifier='exptime', value=kwargs.get('exptime', 0.0), default_unit=u.s, description='Exposure time (time is defined as mid-exposure)')]


    return ParameterSet(params) if as_ps else params, constraints

def rv(syn=False, as_ps=True, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for a radial velocity dataset.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_dataset> as
    `b.add_dataset('rv')`.  In this case, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Arguments
    ----------
    * `syn` (bool, optional, default=False): whether to create the parameters
        for the synthetic (model) instead of the observational (dataset).
    * `as_ps` (bool, optional, default=True): whether to return the parameters
        as a <phoebe.parameters.ParameterSet> instead of a list of
        <phoebe.parameters.Parameter> objects.
    * `times` (array/quantity, optional): observed times.
    * `rvs` (array/quantity, optional): observed radial velocities.
    * `sigmas` (array/quantity, optional): errors on radial velocity measurements.
        Only applicable if `syn` is False.
    * `compute_times` (array/quantity, optional): times at which to compute
        the model.  Only applicable if `syn` is False.
    * `compute_phases` (array/quantity, optional): phases at which to compute
        the model.  Only applicable if `syn` is False.
    * `ld_func` (string, optional): limb-darkening model.  Only applicable if
        `syn` is False.
    * `ld_coeffs` (list, optional): limb-darkening coefficients.  Only applicable
        if `syn` is False.
    * `passband` (string, optional): passband.  Only applicable if `syn` is False.
    * `intens_weighting` (string, optional): whether passband intensities are
        weighted by energy of photons.  Only applicable if `syn` is False.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet> or list, list): ParameterSet (if `as_ps`)
        or list of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params, constraints = [], []

    params += [FloatArrayParameter(qualifier='times', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('times', []), default_unit=u.d, description='Observed times')]
    params += [FloatArrayParameter(qualifier='rvs', visible_if='times:<notempty>', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=_empty_array(kwargs, 'rvs'), default_unit=u.km/u.s, description='Observed radial velocity')]

    if not syn:
        params += [FloatArrayParameter(qualifier='sigmas', visible_if='times:<notempty>', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=_empty_array(kwargs, 'sigmas'), default_unit=u.km/u.s, description='Observed uncertainty on rv')]

        params += [FloatArrayParameter(qualifier='compute_times', value=kwargs.get('compute_times', []), default_unit=u.d, description='Times to use during run_compute.  If empty, will use times parameter')]
        params += [FloatArrayParameter(qualifier='compute_phases', component=kwargs.get('component_top', None), value=kwargs.get('compute_phases', []), default_unit=u.dimensionless_unscaled, description='Phases associated with compute_times.  Does not account for t0: for true phases, use b.to_phase or b.to_time')]

        constraints += [(constraint.compute_phases, kwargs.get('component_top', None), kwargs.get('dataset', None))]

    lc_params, lc_constraints = lc(syn=syn, as_ps=False, is_lc=False, **kwargs)
    params += lc_params
    constraints += lc_constraints

    return ParameterSet(params) if as_ps else params, constraints

def lp(syn=False, as_ps=True, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for a line profile dataset.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_dataset> as
    `b.add_dataset('lp', times=[...])`.  In this case, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Note that `times` **must** be passed during creation and cannot be changed
    after the fact as this function creates copies of the `flux_densities`
    and `sigmas` parameters per-time instead of creating a `times` parameter.

    Arguments
    ----------
    * `syn` (bool, optional, default=False): whether to create the parameters
        for the synthetic (model) instead of the observational (dataset).
    * `as_ps` (bool, optional, default=True): whether to return the parameters
        as a <phoebe.parameters.ParameterSet> instead of a list of
        <phoebe.parameters.Parameter> objects.
    * `times` (array/quantity): times at which the dataset should be defined.
        **IMPORTANT**: times is not a parameter and must be passed during creation,
        see note above.
    * `wavelengths` (array/quantity, optional): observed wavelengths.
    * `flux_densities` (array/quantity, optional): observed flux densities.
    * `sigmas` (array/quantity, optional): errors on flux densities measurements.
        Only applicable if `syn` is False.
    * `ld_func` (string, optional): limb-darkening model.  Only applicable if
    `syn` is False.
    * `ld_coeffs` (list, optional): limb-darkening coefficients.  Only
    applicable if `syn` is False.
    * `passband` (string, optional): passband.  Only applicable if `syn` is False.
    * `intens_weighting` (string, optional): whether passband intensities are
        weighted by energy of photons.  Only applicable if `syn` is False.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet> or list, list): ParameterSet (if `as_ps`)
        or list of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params, constraints = [], []

    times = kwargs.get('times', [])

    # wavelengths is time-independent
    params += [FloatArrayParameter(qualifier='wavelengths', copy_for={'kind': ['star', 'orbit'], 'component': '*'}, component='_default', value=_empty_array(kwargs, 'wavelengths'), default_unit=u.nm, description='Wavelengths of the observations')]

    for time in times:
        # but do allow per-component flux_densities and sigmas
        params += [FloatArrayParameter(qualifier='flux_densities', visible_if='[time]wavelengths:<notempty>', copy_for={'kind': ['star', 'orbit'], 'component': '*'},
                                       component='_default', time=time, value=_empty_array(kwargs, 'flux_densities'), default_unit=u.W/(u.m**2*u.nm),
                                       description='Flux density per wavelength (must be same length as wavelengths or empty)')]

        if not syn:
            params += [FloatArrayParameter(qualifier='sigmas', visible_if='[time]wavelengths:<notempty>', copy_for={'kind': ['star', 'orbit'], 'component': '*'}, component='_default', time=time, value=_empty_array(kwargs, 'sigmas'), default_unit=u.W/(u.m**2*u.nm), description='Observed uncertainty on flux_densities')]

    if not syn:
        params += [ChoiceParameter(qualifier='profile_func', value=kwargs.get('profile_func', 'gaussian'), choices=['gaussian', 'lorentzian'], description='Function to use for the rest line profile')]
        params += [FloatParameter(qualifier='profile_rest', value=kwargs.get('profile_rest', 550), default_unit=u.nm, limits=(0, None), description='Rest central wavelength of the profile')]
        params += [FloatParameter(qualifier='profile_sv', value=kwargs.get('profile_sv', 1e-4), default_unit=u.dimensionless_unscaled, limits=(0, None), description='Subsidiary value of the profile')]

    lc_params, lc_constraints = lc(syn=syn, as_ps=False, is_lc=False, **kwargs)
    params += lc_params
    constraints += lc_constraints

    return ParameterSet(params) if as_ps else params, constraints


def orb(syn=False, as_ps=True, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for an orbit dataset.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_dataset> as
    `b.add_dataset('orb')`.  In this case, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Arguments
    ----------
    * `syn` (bool, optional, default=False): whether to create the parameters
        for the synthetic (model) instead of the observational (dataset).
    * `as_ps` (bool, optional, default=True): whether to return the parameters
        as a <phoebe.parameters.ParameterSet> instead of a list of
        <phoebe.parameters.Parameter> objects.
    * `times` (array/quantity, optional): observed times.
    * `compute_times` (array/quantity, optional): times at which to compute
        the model.  Only applicable if `syn` is False.
    * `compute_phases` (array/quantity, optional): phases at which to compute
        the model.  Only applicable if `syn` is False.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet> or list, list): ParameterSet (if `as_ps`)
        or list of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params, constraints = [], []

    params += [FloatArrayParameter(qualifier='times', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('times', []), default_unit=u.d, description='{} times'.format('Synthetic' if syn else 'Observed'))]

    if syn:
        params += [FloatArrayParameter(qualifier='us', value=_empty_array(kwargs, 'us'), default_unit=u.solRad, description='U position')]
        params += [FloatArrayParameter(qualifier='vs', value=_empty_array(kwargs, 'vs'), default_unit=u.solRad, description='V position')]
        params += [FloatArrayParameter(qualifier='ws', value=_empty_array(kwargs, 'ws'), default_unit=u.solRad, description='W position')]
        params += [FloatArrayParameter(qualifier='vus', value=_empty_array(kwargs, 'vus'), default_unit=u.km/u.s, description='U velocity')]
        params += [FloatArrayParameter(qualifier='vvs', value=_empty_array(kwargs, 'vvs'), default_unit=u.km/u.s, description='V velocity')]
        params += [FloatArrayParameter(qualifier='vws', value=_empty_array(kwargs, 'vws'), default_unit=u.km/u.s, description='W velocity')]

    if not syn:
        params += [FloatArrayParameter(qualifier='compute_times', value=kwargs.get('compute_times', []), default_unit=u.d, description='Times to use during run_compute.  If empty, will use times parameter')]
        params += [FloatArrayParameter(qualifier='compute_phases', component=kwargs.get('component_top', None), value=kwargs.get('compute_phases', []), default_unit=u.dimensionless_unscaled, description='Phases associated with compute_times.  Does not account for t0: for true phases, use b.to_phase or b.to_time')]

        constraints += [(constraint.compute_phases, kwargs.get('component_top', None), kwargs.get('dataset', None))]

    return ParameterSet(params) if as_ps else params, constraints

def mesh(syn=False, as_ps=True, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for a mesh dataset.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_dataset> as
    `b.add_dataset('mesh')`.  In this case, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Arguments
    ----------
    * `syn` (bool, optional, default=False): whether to create the parameters
        for the synthetic (model) instead of the observational (dataset).
    * `as_ps` (bool, optional, default=True): whether to return the parameters
        as a <phoebe.parameters.ParameterSet> instead of a list of
        <phoebe.parameters.Parameter> objects.
    * `times` (array/quantity, optional): observed times.  Only applicable
        if `syn` is False.
    * `include_times` (string, optional): append to times from the following
        datasets/time standards.  Only applicable if `syn` is False.
    * `columns` (list, optional): columns to expose within the mesh.
        Only applicable if `syn` is False.
    * `**kwargs`: if `syn` is True, additional kwargs will be applied to the
        exposed columns according to the passed lists for `mesh_columns`
        and `mesh_datasets`.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet> or list, list): ParameterSet (if `as_ps`)
        or list of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params, constraints = [], []

    times = kwargs.get('times', [])

    params += [FloatArrayParameter(qualifier='times', value=kwargs.get('times', []), default_unit=u.d, description='{} times'.format('Synthetic' if syn else 'Observed'))]

    if not syn:
        params += [SelectParameter(qualifier='include_times', value=kwargs.get('include_times', []), description='append to times from the following datasets/time standards', choices=['t0@system'])]
        params += [SelectParameter(qualifier='columns', value=kwargs.get('columns', []), description='columns to expose within the mesh', choices=_mesh_columns)]

    # the following will all be arrays (value per triangle) per time
    if syn:
        columns = kwargs.get('mesh_columns', [])
        mesh_datasets = kwargs.get('mesh_datasets', [])

        for t in times:
            if not isinstance(t, float):
                raise ValueError("times must all be of type float")


            # always include basic geometric columns
            params += [FloatArrayParameter(qualifier='uvw_elements', time=t, value=kwargs.get('uvw_elements', []), default_unit=u.solRad, description='Vertices of triangles in the plane-of-sky')]
            params += [FloatArrayParameter(qualifier='xyz_elements', time=t, value=kwargs.get('xyz_elements ', []), default_unit=u.dimensionless_unscaled, description='Vertices of triangles in Roche coordinates')]

            # NOTE: if changing the parameters which are optional, changes must
            # be made here, in the choices for the columns Parameter, and in
            # backends.py when the values are extracted and included in the
            # packet

            # if 'pot' in columns:
                # params += [FloatParameter(qualifier='pot', time=t, value=kwargs.get('pot', 0.0), default_unit=u.dimensionless_unscaled, description='Equipotential of the stellar surface')]
            # if 'rpole' in columns:
            #     params += [FloatParameter(qualifier='rpole', time=t, value=kwargs.get('rpole', 0.0), default_unit=u.solRad, description='Polar radius of the stellar surface')]
            if 'volume' in columns:
                params += [FloatParameter(qualifier='volume', time=t, value=kwargs.get('volume', 0.0), default_unit=u.solRad**3, description='Volume of the stellar surface')]


            if 'xs' in columns:
                params += [FloatArrayParameter(qualifier='xs', time=t, value=kwargs.get('xs', []), default_unit=u.solRad, description='X coordinate of center of triangles in the plane-of-sky')]
            if 'ys' in columns:
                params += [FloatArrayParameter(qualifier='ys', time=t, value=kwargs.get('ys', []), default_unit=u.solRad, description='Y coordinate of center of triangles in the plane-of-sky')]
            if 'zs' in columns:
                params += [FloatArrayParameter(qualifier='zs', time=t, value=kwargs.get('zs', []), default_unit=u.solRad, description='Z coordinate of center of triangles in the plane-of-sky')]

            if 'vxs' in columns:
                params += [FloatArrayParameter(qualifier='vxs', time=t, value=kwargs.get('vxs', []), default_unit=u.km/u.s, description='X velocity of center of triangles')]
            if 'vys' in columns:
                params += [FloatArrayParameter(qualifier='vys', time=t, value=kwargs.get('vys', []), default_unit=u.km/u.s, description='Y velocity of center of triangles')]
            if 'vzs' in columns:
                params += [FloatArrayParameter(qualifier='vzs', time=t, value=kwargs.get('vzs', []), default_unit=u.km/u.s, description='Z velocity of center of triangles')]

            if 'nxs' in columns:
                params += [FloatArrayParameter(qualifier='nxs', time=t, value=kwargs.get('nxs', []), default_unit=u.dimensionless_unscaled, description='X component of normals')]
            if 'nys' in columns:
                params += [FloatArrayParameter(qualifier='nys', time=t, value=kwargs.get('nys', []), default_unit=u.dimensionless_unscaled, description='Y component of normals')]
            if 'nzs' in columns:
                params += [FloatArrayParameter(qualifier='nzs', time=t, value=kwargs.get('nzs', []), default_unit=u.dimensionless_unscaled, description='Z component of normals')]

            if 'us' in columns:
                params += [FloatArrayParameter(qualifier='us', time=t, value=kwargs.get('us', []), default_unit=u.solRad, description='U coordinate of center of triangles in the plane-of-sky')]
            if 'vs' in columns:
                params += [FloatArrayParameter(qualifier='vs', time=t, value=kwargs.get('vs', []), default_unit=u.solRad, description='V coordinate of center of triangles in the plane-of-sky')]
            if 'ws' in columns:
                params += [FloatArrayParameter(qualifier='ws', time=t, value=kwargs.get('ws', []), default_unit=u.solRad, description='W coordinate of center of triangles in the plane-of-sky')]

            if 'vus' in columns:
                params += [FloatArrayParameter(qualifier='vus', time=t, value=kwargs.get('vus', []), default_unit=u.km/u.s, description='U velocity of center of triangles')]
            if 'vvs' in columns:
                params += [FloatArrayParameter(qualifier='vvs', time=t, value=kwargs.get('vvs', []), default_unit=u.km/u.s, description='V velocity of center of triangles')]
            if 'vws' in columns:
                params += [FloatArrayParameter(qualifier='vws', time=t, value=kwargs.get('vws', []), default_unit=u.km/u.s, description='W velocity of center of triangles')]

            if 'nus' in columns:
                params += [FloatArrayParameter(qualifier='nus', time=t, value=kwargs.get('nus', []), default_unit=u.dimensionless_unscaled, description='U component of normals')]
            if 'nvs' in columns:
                params += [FloatArrayParameter(qualifier='nvs', time=t, value=kwargs.get('nvs', []), default_unit=u.dimensionless_unscaled, description='V component of normals')]
            if 'nws' in columns:
                params += [FloatArrayParameter(qualifier='nws', time=t, value=kwargs.get('nws', []), default_unit=u.dimensionless_unscaled, description='W component of normals')]


            if 'areas' in columns:
                params += [FloatArrayParameter(qualifier='areas', time=t, value=kwargs.get('areas', []), default_unit=u.solRad**2, description='Area of triangles')]
            # if 'tareas' in columns:
                # params += [FloatArrayParameter(qualifier='tareas', time=t, value=kwargs.get('areas', []), default_unit=u.solRad**2, description='Area of WD triangles')]


            if 'rs' in columns:
                params += [FloatArrayParameter(qualifier='rs', time=t, value=kwargs.get('rs', []), default_unit=u.solRad, description='Distance of each triangle from center of mass (of the half-envelope for contacts)')]
            # if 'cosbetas' in columns:
            #     params += [FloatArrayParameter(qualifier='cosbetas', time=t, value=kwargs.get('cosbetas', []), default_unit=u.solRad, description='')]


            if 'loggs' in columns:
                params += [FloatArrayParameter(qualifier='loggs', time=t, value=kwargs.get('loggs', []), default_unit=u.dimensionless_unscaled, description='Local surface gravity')]
            if 'teffs' in columns:
                params += [FloatArrayParameter(qualifier='teffs', time=t, value=kwargs.get('teffs', []), default_unit=u.K, description='Local effective temperature')]

            if 'rprojs' in columns:
                params += [FloatArrayParameter(qualifier='rprojs', time=t, value=kwargs.get('rprojs', []), default_unit=u.solRad, description='Projected distance (on plane of sky) of each triangle from center of mass (of the half-envelope for contacts)')]
            if 'mus' in columns:
                params += [FloatArrayParameter(qualifier='mus', time=t, value=kwargs.get('mus', []), default_unit=u.dimensionless_unscaled, description='Mu')]
            if 'visible_centroids' in columns:
                params += [FloatArrayParameter(qualifier='visible_centroids', time=t, value=kwargs.get('visible_centroids', []), default_unit=u.solRad  if t is not None else u.dimensionless_unscaled, description='Center of the visible portion of each triangle')]
            if 'visibilities' in columns:
                params += [FloatArrayParameter(qualifier='visibilities', time=t, value=kwargs.get('visibilities', []), default_unit=u.dimensionless_unscaled, description='Visiblity of triangles (1=visible, 0.5=partial, 0=hidden)')]

            # params += [FloatArrayParameter(qualifier='horizon_xs', time=t, value=kwargs.get('horizon_xs', []), default_unit=u.solRad, description='Horizon of the mesh (x component)')]
            # params += [FloatArrayParameter(qualifier='horizon_ys', time=t, value=kwargs.get('horizon_ys', []), default_unit=u.solRad, description='Horizon of the mesh (y component)')]
            # params += [FloatArrayParameter(qualifier='horizon_zs', time=t, value=kwargs.get('horizon_zs', []), default_unit=u.solRad, description='Horizon of the mesh (z component)')]
            # params += [FloatArrayParameter(qualifier='horizon_analytic_xs', time=t, value=kwargs.get('horizon_analytic_xs', []), default_unit=u.solRad, description='Analytic horizon (interpolated, x component)')]
            # params += [FloatArrayParameter(qualifier='horizon_analytic_ys', time=t, value=kwargs.get('horizon_analytic_ys', []), default_unit=u.solRad, description='Analytic horizon (interpolated, y component)')]
            # params += [FloatArrayParameter(qualifier='horizon_analytic_zs', time=t, value=kwargs.get('horizon_analytic_zs', []), default_unit=u.solRad, description='Analytic horizon (interpolated, z component)')]

            for dataset in mesh_datasets:
                # if 'dls@{}'.format(dataset) in columns:
                    # params += [FloatArrayParameter(qualifier='dls', dataset=dataset, time=t, value=[], default_unit=u.nm, description='Per-element delta-lambda caused by doppler shift'.format(dataset))]
                if 'rvs@{}'.format(dataset) in columns:
                    params += [FloatArrayParameter(qualifier='rvs', dataset=dataset, time=t, value=[], default_unit=u.km/u.s, description='Per-element value of rvs for {} dataset'.format(dataset))]
                if 'intensities@{}'.format(dataset) in columns:
                    params += [FloatArrayParameter(qualifier='intensities', dataset=dataset, time=t, value=[], default_unit=u.W/u.m**3, description='Per-element value of intensities for {} dataset'.format(dataset))]
                if 'normal_intensities@{}'.format(dataset) in columns:
                    params += [FloatArrayParameter(qualifier='normal_intensities', dataset=dataset, time=t, value=[], default_unit=u.W/u.m**3, description='Per-element value of normal_intensities for {} dataset'.format(dataset))]
                if 'abs_intensities@{}'.format(dataset) in columns:
                    params += [FloatArrayParameter(qualifier='abs_intensities', dataset=dataset, time=t, value=[], default_unit=u.W/u.m**3, description='Per-element value of abs_intensities for {} dataset'.format(dataset))]
                if 'abs_normal_intensities@{}'.format(dataset) in columns:
                    params += [FloatArrayParameter(qualifier='abs_normal_intensities', dataset=dataset, time=t, value=[], default_unit=u.W/u.m**3, description='Per-element value of abs_normal_intensities for {} dataset'.format(dataset))]
                if 'boost_factors@{}'.format(dataset) in columns:
                    params += [FloatArrayParameter(qualifier='boost_factors', dataset=dataset, time=t, value=[], default_unit=u.dimensionless_unscaled, description='Per-element value of boost_factors for {} dataset'.format(dataset))]
                if 'ldint@{}'.format(dataset) in columns:
                    params += [FloatArrayParameter(qualifier='ldint', dataset=dataset, time=t, value=kwargs.get('ldint', []), default_unit=u.dimensionless_unscaled, description='Integral of the limb-darkening function')]

                if 'ptfarea@{}'.format(dataset) in columns:
                    params += [FloatParameter(qualifier='ptfarea', dataset=dataset, time=t, value=kwargs.get('ptfarea', 1.0), default_unit=u.m, description='Area of the passband transmission function')]
                if 'pblum_ext@{}'.format(dataset) in columns:
                    params += [FloatParameter(qualifier='pblum_ext', dataset=dataset, time=t, value=kwargs.get('pblum_ext', 0.0), default_unit=u.W, description='Passband Luminosity of entire star (after pblum scaling)')]
                if 'abs_pblum_ext@{}'.format(dataset) in columns:
                    params += [FloatParameter(qualifier='abs_pblum_ext', dataset=dataset, time=t, value=kwargs.get('abs_pblum_ext', 0.0), default_unit=u.W, description='Passband Luminosity of entire star (before pblum scaling)')]


    return ParameterSet(params) if as_ps else params, constraints
