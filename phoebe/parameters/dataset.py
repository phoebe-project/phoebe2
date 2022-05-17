
from phoebe.parameters import *
from phoebe.parameters import constraint
from phoebe.atmospheres import passbands  # need to load pbtable (dictionary of available passbands)
from phoebe import u
from phoebe import conf

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

_ld_func_choices = ['linear', 'logarithmic', 'quadratic', 'square_root', 'power']


passbands._init_passbands()  # TODO: move to module import
_ld_coeffs_source_choices = ['auto'] + list(set([atm for pb in passbands._pbtable.values() for atm in pb['atms_ld']]))

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

# rv columns have all pb-dependent columns except those that require pblum scaling
rv_columns = [c for c in lc_columns[:] if c not in ['intensities', 'normal_intensities', 'pblum_ext']]
rv_columns += ['rvs']

lp_columns = rv_columns[:]
# lp_columns += ['dls']


_pbdep_columns = {'lc': lc_columns,
                  'rv': rv_columns,
                  'lp': lp_columns}

import logging
logger = logging.getLogger("DATASET")
logger.addHandler(logging.NullHandler())

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
    * `phases_period` (string, optional, default='period'): period to use
        when converting between `compute_phases` and `compute_times` as well as
        when applying `mask_phases`.  Only applicable if `syn` is False. and `is_lc` is True.
        Not applicable for single stars (in which case period is always used)
        or if `dperdt == 0`.
    * `phases_dpdt` (string, optional, default='dpdt'): dpdt to use when
        converting between compute_times and compute_phases as well as when
        applying mask_phases.  Not applicable for single stars or if `dpdt == 0`
    * `phases_t0` (string, optional, default='t0_supconj'): t0 to use
        when converting between `compute_phases` and `compute_times` as well as
        when applying `mask_phases`.  Only
        applicable if `syn` is False and `is_lc` is True.  Not applicable for
        single stars (in which case t0@system is always used).
    * `mask_enabled` (bool, optional, default=True): whether to apply the mask
        in mask_phases during plotting, calculate_residuals, calculate_chi2,
        calculate_lnlikelihood, and run_solver
    * `mask_phases` (list of tuples, optional, default=[]): List of phase-tuples.
        Any observations inside the range set by any of the tuples will be included.
    * `ld_mode` (string, optional, default='interp'): mode to use for handling
        limb-darkening.  Note that 'interp' is not available for all values
        of `atm` (availability can be checked by calling
        <phoebe.frontend.bundle.Bundle.run_checks> and will automatically be checked
        during <phoebe.frontend.bundle.Bundle.run_compute>).  Only applicable
        if `syn` is False.
    * `ld_func` (string, optional, default='logarithmic'): function/law to use for
        limb-darkening model. Not applicable if `ld_mode` is 'interp'.  Only
        applicable if `syn` is False.
    * `ld_coeffs_source` (string, optional, default='auto'): source for limb-darkening
        coefficients ('auto' to interpolate from the applicable table according
        to the 'atm' parameter, or the name of a specific atmosphere table).
        Only applicable if `ld_mode` is 'lookup'.  Only applicable if
        `syn` is False.
    * `ld_coeffs` (list, optional): limb-darkening coefficients.  Must be of
        the approriate length given the value of `ld_coeffs_source` which can
        be checked by calling <phoebe.frontend.bundle.Bundle.run_checks>
        and will automtically be checked during
        <phoebe.frontend.bundle.Bundle.run_compute>.  Only applicable
       if `ld_mode` is 'manual'.  Only applicable if `syn` is False.
    * `passband` (string, optional): passband.  Only applicable if `syn` is False.
    * `intens_weighting` (string, optional): whether passband intensities are
        weighted by energy or photons.  Only applicable if `syn` is False.
    * `pblum_mode` (string, optional, default='manual'): mode for scaling
        passband luminosities.  Only applicable if `syn` is False and `is_lc`
        is True.
    * `pblum_dataset` (string, optional):  Dataset to reference for coupling
        luminosities.  Only applicable if `pblum_mode` is 'dataset-coupled'.
    * `pblum_component` (string, optional): Component to provide `pblum`.
        Only applicable if `pblum_mode` is 'component-coupled'.
    * `pblum` (float/quantity or string, optional): passband luminosity (defined at t0).
        If `pblum_mode` is 'decoupled', then one entry per-star will be applicable.
        If `pblum_mode` is 'component-coupled', then only the entry for the primary
        component, according to <phoebe.parameters.HierarchyParameter> will be
        available.  To change the provided component, see `pblum_component`.
        Only applicable if `syn` is False and `is_lc` is True.
    * `l3_mode` (string, optional, default='flux'): mode for providing third
        light (`l3`).  Only applicable if `syn` is False and `is_lc` is True.
    * `l3` (float/quantity, optional): third light in flux units (only applicable
        if `l3_mode` is 'flux'). Only applicable if `syn` is False and `is_lc`
        is True.
    * `l3_frac` (float/quantity, optional): third light in fraction of total light.
        (only applicable if `l3_mode` is 'fraction').
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
        params += [FloatArrayParameter(qualifier='times', value=kwargs.get('times', []), required_shape=[None], readonly=syn, default_unit=u.d, description='Model (synthetic) times' if syn else 'Observed times')]
        params += [FloatArrayParameter(qualifier='fluxes', value=_empty_array(kwargs, 'fluxes'), required_shape=[None] if not syn else None, readonly=syn, default_unit=u.W/u.m**2, description='Model (synthetic) flux' if syn else 'Observed flux')]

    if not syn:
        # TODO: should we move all limb-darkening to compute options since
        # not all backends support interp (and lookup is atm-dependent)
        params += [ChoiceParameter(qualifier='ld_mode', copy_for={'kind': ['star'], 'component': '*'}, component='_default',
                                   value=kwargs.get('ld_mode', 'interp'), choices=['interp', 'lookup', 'manual'],
                                   description='Mode to use for limb-darkening')]
        params += [ChoiceParameter(visible_if='ld_mode:lookup|manual', qualifier='ld_func',
                                   copy_for={'kind': ['star'], 'component': '*'}, component='_default',
                                   value=kwargs.get('ld_func', 'logarithmic'), choices=_ld_func_choices,
                                   description='Limb darkening model')]
        params += [ChoiceParameter(visible_if='ld_mode:lookup', qualifier='ld_coeffs_source',
                                   copy_for={'kind': ['star'], 'component': '*'}, component='_default',
                                   value=kwargs.get('ld_coeffs_source', 'auto'), choices=_ld_coeffs_source_choices,
                                   advanced=True,
                                   description='Source for limb darkening coefficients (\'auto\' to interpolate from the applicable table according to the \'atm\' parameter, or the name of a specific atmosphere table)')]
        params += [FloatArrayParameter(visible_if='ld_mode:manual', qualifier='ld_coeffs',
                                       latexfmt=r'\mathrm{{ ldc }}_\mathrm{{ {dataset}, {component} }}',
                                       copy_for={'kind': ['star'], 'component': '*'}, component='_default',
                                       value=kwargs.get('ld_coeffs', [0.5, 0.5]), default_unit=u.dimensionless_unscaled,
                                       required_shape=[None],
                                       description='Limb darkening coefficients')]

        passbands._init_passbands()  # NOTE: this only actually does something on the first call
        params += [ChoiceParameter(qualifier='passband', value=kwargs.get('passband', 'Johnson:V'), choices=passbands.list_passbands(), description='Passband')]
        params += [ChoiceParameter(qualifier='intens_weighting', value=kwargs.get('intens_weighting', 'energy'), choices=['energy', 'photon'], advanced=True, description='Whether passband intensities are weighted by energy or photons')]



    if is_lc and not syn:
        params += [FloatArrayParameter(qualifier='compute_times', value=kwargs.get('compute_times', []), required_shape=[None], default_unit=u.d, description='Times to use during run_compute.  If empty, will use times parameter')]
        params += [FloatArrayParameter(qualifier='compute_phases', component=kwargs.get('component_top', None), required_shape=[None], value=kwargs.get('compute_phases', []), default_unit=u.dimensionless_unscaled, description='Phases associated with compute_times.')]
        params += [ChoiceParameter(qualifier='phases_period', visible_if='[dataset][context][kind]dperdt:!0.0,hierarchy.is_meshable:False', component=kwargs.get('component_top', None), value=kwargs.get('phases_period', 'period'), choices=['period', 'period_anom'], advanced=True, description='period to use when converting between compute_times and compute_phases as well as when applying mask_phases')]
        params += [ChoiceParameter(qualifier='phases_dpdt', visible_if='[dataset][context][kind]dpdt:!0.0,hierarchy.is_meshable:False', component=kwargs.get('component_top', None), value=kwargs.get('phases_dpdt', 'dpdt'), choices=['dpdt', 'none'], advanced=True, description='dpdt to use when converting between compute_times and compute_phases as well as when applying mask_phases')]
        params += [ChoiceParameter(qualifier='phases_t0', visible_if='hierarchy.is_meshable:False', component=kwargs.get('component_top', None), value=kwargs.get('phases_t0', 't0_supconj'), choices=['t0_supconj', 't0_perpass', 't0_ref'], advanced=True, description='t0 to use when converting between compute_times and compute_phases as well as when applying mask_phases')]
        constraints += [(constraint.compute_phases, kwargs.get('component_top', None), kwargs.get('dataset', None))]

        params += [BoolParameter(qualifier='mask_enabled', value=kwargs.get('mask_enabled', True), description='Whether to apply the mask in mask_phases during plotting, calculate_residuals, calculate_chi2, calculate_lnlikelihood, and run_solver')]
        params += [FloatArrayParameter(visible_if='[component]mask_enabled:True', qualifier='mask_phases', component=kwargs.get('component_top', None), value=kwargs.get('mask_phases', []), default_unit=u.dimensionless_unscaled, required_shape=[None, 2], description='List of phase-tuples.  Any observations inside the range set by any of the tuples will be included.')]

        params += [ChoiceParameter(qualifier='solver_times', value=kwargs.get('solver_times', 'auto'), choices=['auto', 'compute_times', 'times'], description='times to use within run_solver.  All options will properly account for masking from mask_times.  To see how this is parsed, see b.parse_solver_times.  auto: use compute_times if provided and shorter than times, otherwise use times.  compute_times: use compute_times if provided.  times: use times array.')]

        params += [FloatArrayParameter(qualifier='sigmas', value=_empty_array(kwargs, 'sigmas'), required_shape=[None], default_unit=u.W/u.m**2, description='Observed uncertainty on flux')]
        params += [FloatParameter(qualifier='sigmas_lnf', latexfmt=r'\sigma_\mathrm{{ lnf, {dataset} }}', visible_if='sigmas:<notempty>', value=kwargs.get('sigmas_lnf', -np.inf), default_unit=u.dimensionless_unscaled, limits=(None, None), description='Natural log of the fractional amount to sigmas are underestimate (when calculating chi2/lnlikelihood)')]

        params += [ChoiceParameter(qualifier='pblum_mode', value=kwargs.get('pblum_mode', 'component-coupled'),
                                   choices=['decoupled', 'component-coupled', 'dataset-coupled', 'dataset-scaled', 'absolute'],
                                   description='Mode for scaling passband luminosities')]

        # pblum_mode = 'component-coupled' or 'decoupled'
        params += [ChoiceParameter(visible_if='pblum_mode:component-coupled', qualifier='pblum_component', value=kwargs.get('pblum_component', ''), choices=kwargs.get('starrefs', ['']), advanced=True, description='Which component\'s pblum will be provided')]
        params += [FloatParameter(qualifier='pblum', latexfmt=r'L_\mathrm{{ pb, {dataset} }}', visible_if='[component]pblum_mode:decoupled||[component]pblum_mode:component-coupled,[component]pblum_component:<component>', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('pblum', 4*np.pi), default_unit=u.W, description='Passband luminosity (defined at t0)')]

        # pblum_mode = 'dataset-coupled'
        params += [ChoiceParameter(visible_if='pblum_mode:dataset-coupled', qualifier='pblum_dataset', value=kwargs.get('pblum_dataset', ''), choices=['']+kwargs.get('lcrefs', []), description='Dataset with which to couple luminosities based on color')]

        # pblum_mode = 'pbflux'
        params += [FloatParameter(visible_if='false', qualifier='pbflux', value=kwargs.get('pbflux', 1.0), default_unit=u.W/u.m**2, description='Total inrinsic (excluding features and irradiation) passband flux (at t0, including l3 if pblum_mode=\'flux\')')]

        params += [ChoiceParameter(qualifier='l3_mode', value=kwargs.get('l3_mode', 'flux'), choices=['flux', 'fraction'], description='Whether third light is given in units of flux or as a fraction of total light')]
        params += [FloatParameter(visible_if='l3_mode:flux', qualifier='l3', latexfmt=r'l_\mathrm{{ 3, {dataset} }}', value=kwargs.get('l3', 0.), limits=[0, None], default_unit=u.W/u.m**2, description='Third light in flux units')]
        params += [FloatParameter(visible_if='l3_mode:fraction', qualifier='l3_frac', latexfmt=r'l_\mathrm{{ 3, frac, {dataset} }}', value=kwargs.get('l3_frac', 0.), limits=[0, 1], default_unit=u.dimensionless_unscaled, description='Third light as a fraction of total flux (both system and third light)')]

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
    * `phases_period` (string, optional, default='period'): period to use
        when converting between `compute_phases` and `compute_times` as well as
        when applying `mask_phases`.  Only applicable if `syn` is False.  Not applicable for
        single stars (in which case period is always used) or if `dperdt == 0.0`.
    * `phases_dpdt` (string, optional, default='dpdt'): dpdt to use when
        converting between compute_times and compute_phases as well as when
        applying mask_phases.  Not applicable for single stars or if `dpdt == 0`
    * `phases_t0` (string, optional, default='t0_supconj'): t0 to use
        when converting between `compute_phases` and `compute_times` as well as
        when applying `mask_phases`.  Only applicable if `syn` is False.  Not applicable for
        single stars (in which case t0@system is always used).
    * `mask_enabled` (bool, optional, default=True): whether to apply the mask
        in mask_phases during plotting, calculate_residuals, calculate_chi2,
        calculate_lnlikelihood, and run_solver
    * `mask_phases` (list of tuples, optional, default=[]): List of phase-tuples.
        Any observations inside the range set by any of the tuples will be included.
    * `ld_mode` (string, optional, default='interp'): mode to use for handling
        limb-darkening.  Note that 'interp' is not available for all values
        of `atm` (availability can be checked by calling
        <phoebe.frontend.bundle.Bundle.run_checks> and will automatically be checked
        during <phoebe.frontend.bundle.Bundle.run_compute>).  Only applicable
        if `syn` is False.
    * `ld_func` (string, optional, default='linear'): function/law to use for
        limb-darkening model. Not applicable if `ld_mode` is 'interp'.  Only
        applicable if `syn` is False.
    * `ld_coeffs_source` (string, optional, default='auto'): source for limb-darkening
        coefficients ('auto' to interpolate from the applicable table according
        to the 'atm' parameter, or the name of a specific atmosphere table).
        Only applicable if `ld_mode` is 'lookup'.  Only applicable if
        `syn` is False.
    * `ld_coeffs` (list, optional): limb-darkening coefficients.  Must be of
        the approriate length given the value of `ld_coeffs_source` which can
        be checked by calling <phoebe.frontend.bundle.Bundle.run_checks>
        and will automtically be checked during
        <phoebe.frontend.bundle.Bundle.run_compute>.  Only applicable
       if `ld_mode` is 'manual'.  Only applicable if `syn` is False.
    * `passband` (string, optional): passband.  Only applicable if `syn` is False.
    * `intens_weighting` (string, optional): whether passband intensities are
        weighted by energy or photons.  Only applicable if `syn` is False.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet> or list, list): ParameterSet (if `as_ps`)
        or list of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params, constraints = [], []

    params += [FloatArrayParameter(qualifier='times', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('times', []), required_shape=[None], readonly=syn, default_unit=u.d, description='Model (synthetic) times' if syn else 'Observed times')]
    params += [FloatArrayParameter(qualifier='rvs', visible_if='times:<notempty>', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=_empty_array(kwargs, 'rvs'), required_shape=[None] if not syn else None, readonly=syn, default_unit=u.km/u.s, description='Model (synthetic) radial velocities' if syn else 'Observed radial velocity')]

    if not syn:
        params += [FloatArrayParameter(qualifier='sigmas', visible_if='times:<notempty>', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=_empty_array(kwargs, 'sigmas'), required_shape=None, default_unit=u.km/u.s, description='Observed uncertainty on rv')]
        params += [FloatParameter(qualifier='sigmas_lnf', latexfmt=r'\sigma_\mathrm{{ lnf, {dataset} }}', visible_if='sigmas:<notempty>', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('sigmas_lnf', -np.inf), default_unit=u.dimensionless_unscaled, limits=(None,None), description='Natural log of the fractional amount to sigmas are underestimate (when calculating chi2/lnlikelihood)')]

        params += [FloatParameter(qualifier='rv_offset', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('rv_offset', 0.0), default_unit=u.km/u.s, description='Per-component offset to add to synthetic RVs (i.e. for hot stars)')]

        params += [FloatArrayParameter(qualifier='compute_times', value=kwargs.get('compute_times', []), required_shape=[None], default_unit=u.d, description='Times to use during run_compute.  If empty, will use times parameter')]
        params += [FloatArrayParameter(qualifier='compute_phases', component=kwargs.get('component_top', None), value=kwargs.get('compute_phases', []), required_shape=[None], default_unit=u.dimensionless_unscaled, description='Phases associated with compute_times.')]
        params += [ChoiceParameter(qualifier='phases_period', visible_if='[dataset][context][kind]dperdt:!0.0,hierarchy.is_meshable:False', component=kwargs.get('component_top', None), value=kwargs.get('phases_period', 'period'), choices=['period', 'period_anom'], advanced=True, description='period to use when converting between compute_times and compute_phases as well as when applying mask_phases')]
        params += [ChoiceParameter(qualifier='phases_dpdt', visible_if='[dataset][context][kind]dpdt:!0.0,hierarchy.is_meshable:False', component=kwargs.get('component_top', None), value=kwargs.get('phases_dpdt', 'dpdt'), choices=['dpdt', 'none'], advanced=True, description='dpdt to use when converting between compute_times and compute_phases as well as when applying mask_phases')]
        params += [ChoiceParameter(qualifier='phases_t0', visible_if='hierarchy.is_meshable:False', component=kwargs.get('component_top', None), value=kwargs.get('phases_t0', 't0_supconj'), choices=['t0_supconj', 't0_perpass', 't0_ref'], advanced=True, description='t0 to use when converting between compute_times and compute_phases as well as when applying mask_phases')]
        constraints += [(constraint.compute_phases, kwargs.get('component_top', None), kwargs.get('dataset', None))]

        params += [BoolParameter(qualifier='mask_enabled', value=kwargs.get('mask_enabled', True), description='Whether to apply the mask in mask_phases during plotting, calculate_residuals, calculate_chi2, calculate_lnlikelihood, and run_solver')]
        params += [FloatArrayParameter(visible_if='[component]mask_enabled:True', qualifier='mask_phases', component=kwargs.get('component_top', None), value=kwargs.get('mask_phases', []), default_unit=u.dimensionless_unscaled, required_shape=[None, 2], description='List of phase-tuples.  Any observations inside the range set by any of the tuples will be included.')]

        params += [ChoiceParameter(qualifier='solver_times', value=kwargs.get('solver_times', 'auto'), choices=['auto', 'compute_times', 'times'], description='times to use within run_solver.  auto: use compute_times if provided and shorter than times, otherwise use times.  compute_times: use compute_times if provided.  times: use times array.')]


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
        see note above.  If `syn` is True, a `times` parameter will be created,
        but all other parameters will be tagged with individual times.
    * `wavelengths` (array/quantity, optional): observed wavelengths.
    * `flux_densities` (array/quantity, optional): observed flux densities.
        A copy of this parameter will exist per-time (as passed to the `times`
        argument at creation, see above) and will be tagged with that time.
    * `sigmas` (array/quantity, optional): errors on flux densities measurements.
        Only applicable if `syn` is False.  A copy of this parameter will exist
        per-time (as passed to the `times` argument at creation, see above) and
        will be tagged with that time.
    * `compute_times` (array/quantity, optional): times at which to compute
        the model.  If provided, this will override the tagged times as defined
        by `times` (note that interpolating between the model computed at
        `compute_times` and the dataset defined at `times` is not currently
        supported).  Only applicable if `syn` is False.
    * `compute_phases` (array/quantity, optional): phases at which to compute
        the model.  Only applicable if `syn` is False.
    * `phases_period` (string, optional, default='period'): period to use
        when converting between `compute_phases` and `compute_times` as well as
        when applying `mask_phases`.  Only applicable if `syn` is False.  Not applicable for
        single stars (in which case period is always used) or if `dperdt == 0`.
    * `phases_dpdt` (string, optional, default='dpdt'): dpdt to use when
        converting between compute_times and compute_phases as well as when
        applying mask_phases.  Not applicable for single stars or if `dpdt == 0`
    * `phases_t0` (string, optional, default='t0_supconj'): t0 to use
        when converting between `compute_phases` and `compute_times`.  Only
        applicable if `syn` is False.  Not applicable for
        single stars (in which case t0@system is always used).
    * `profile_func` (string, optional, default='gaussian'): function to use
        for the rest line profile.
    * `profile_rest` (float, optional, default=550): rest central wavelength
        for the line profile.
    * `profile_sv` (float, optional, default=1e-4): subsidiary value of the
        profile.
    * `ld_mode` (string, optional, default='interp'): mode to use for handling
        limb-darkening.  Note that 'interp' is not available for all values
        of `atm` (availability can be checked by calling
        <phoebe.frontend.bundle.Bundle.run_checks> and will automatically be checked
        during <phoebe.frontend.bundle.Bundle.run_compute>).  Only applicable
        if `syn` is False.
    * `ld_func` (string, optional, default='linear'): function/law to use for
        limb-darkening model. Not applicable if `ld_mode` is 'interp'.  Only
        applicable if `syn` is False.
    * `ld_coeffs_source` (string, optional, default='auto'): source for limb-darkening
        coefficients ('auto' to interpolate from the applicable table according
        to the 'atm' parameter, or the name of a specific atmosphere table).
        Only applicable if `ld_mode` is 'lookup'.  Only applicable if
        `syn` is False.
    * `ld_coeffs` (list, optional): limb-darkening coefficients.  Must be of
        the approriate length given the value of `ld_coeffs_source` which can
        be checked by calling <phoebe.frontend.bundle.Bundle.run_checks>
        and will automtically be checked during
        <phoebe.frontend.bundle.Bundle.run_compute>.  Only applicable
       if `ld_mode` is 'manual'.  Only applicable if `syn` is False.
    * `passband` (string, optional): passband.  Only applicable if `syn` is False.
    * `intens_weighting` (string, optional): whether passband intensities are
        weighted by energy or photons.  Only applicable if `syn` is False.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet> or list, list): ParameterSet (if `as_ps`)
        or list of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params, constraints = [], []

    times = kwargs.get('times', [])

    # if syn:
        # expose the computed times as we do for a mesh, even though the actual
        # parameters will be **tagged** with times
        # TODO: it would be nice if this wasn't copied per-component in the model... but it is also somewhat useful
        # NOTE: enabling this requires some changes to plotting logic.  We have it for meshes so you can plot pot vs time, for example, but lp doesn't have any syn FloatParameters
        # params += [FloatArrayParameter(qualifier='times', value=kwargs.get('times', []), default_unit=u.d, description='{} times'.format('Synthetic' if syn else 'Observed'))]


    # wavelengths is time-independent
    params += [FloatArrayParameter(qualifier='wavelengths', copy_for={'kind': ['star', 'orbit'], 'component': '*'}, component='_default', value=_empty_array(kwargs, 'wavelengths'), required_shape=[None], readonly=syn, default_unit=u.nm, description='Wavelengths of the model (synthetic)' if syn else 'Wavelengths of the observations')]

    for time in times:
        # but do allow per-component flux_densities and sigmas
        params += [FloatArrayParameter(qualifier='flux_densities', visible_if='[time]wavelengths:<notempty>', copy_for={'kind': ['star', 'orbit'], 'component': '*'},
                                       component='_default', time=time, value=_empty_array(kwargs, 'flux_densities'),
                                       required_shape=[None], readonly=syn, default_unit=u.W/(u.m**2*u.nm),
                                       description='Flux density per wavelength (must be same length as wavelengths or empty)')]

        if not syn:
            params += [FloatArrayParameter(qualifier='sigmas', visible_if='[time]wavelengths:<notempty>', copy_for={'kind': ['star', 'orbit'], 'component': '*'}, component='_default', time=time, value=_empty_array(kwargs, 'sigmas'), required_shape=[None], default_unit=u.W/(u.m**2*u.nm), description='Observed uncertainty on flux_densities')]
            params += [FloatParameter(qualifier='sigmas_lnf',  latexfmt=r'\sigma_\mathrm{{ lnf, {dataset} }}', visible_if='[time]sigmas:<notempty>', copy_for={'kind': ['star', 'orbit'], 'component': '*'}, component='_default', time=time, value=kwargs.get('sigmas_lnf', -np.inf), default_unit=u.dimensionless_unscaled, limits=(None, None), description='Natural log of the fractional amount to sigmas are underestimate (when calculating chi2/lnlikelihood)')]

    if not syn:
        params += [FloatArrayParameter(qualifier='compute_times', value=kwargs.get('compute_times', []), required_shape=[None], default_unit=u.d, description='Times to use during run_compute.  If empty, will use times of individual entries.  Note that interpolation is not currently supported for lp datasets.')]
        params += [FloatArrayParameter(qualifier='compute_phases', component=kwargs.get('component_top', None), value=kwargs.get('compute_phases', []), required_shape=[None], default_unit=u.dimensionless_unscaled, description='Phases associated with compute_times.')]
        params += [ChoiceParameter(qualifier='phases_period', visible_if='[dataset][context][kind]dperdt:!0.0,hierarchy.is_meshable:False', component=kwargs.get('component_top', None), value=kwargs.get('phases_period', 'period'), choices=['period', 'period_anom'], advanced=True, description='period to use when converting between compute_times and compute_phases as well as when applying mask_phases')]
        params += [ChoiceParameter(qualifier='phases_dpdt', visible_if='[dataset][context][kind]dpdt:!0.0,hierarchy.is_meshable:False', component=kwargs.get('component_top', None), value=kwargs.get('phases_dpdt', 'dpdt'), choices=['dpdt', 'none'], advanced=True, description='dpdt to use when converting between compute_times and compute_phases as well as when applying mask_phases')]
        params += [ChoiceParameter(qualifier='phases_t0', visible_if='hierarchy.is_meshable:False', component=kwargs.get('component_top', None), value=kwargs.get('phases_t0', 't0_supconj'), choices=['t0_supconj', 't0_perpass', 't0_ref'], advanced=True, description='t0 to use when converting between compute_times and compute_phases as well as when applying mask_phases')]
        constraints += [(constraint.compute_phases, kwargs.get('component_top', None), kwargs.get('dataset', None))]

        # params += [BoolParameter(qualifier='mask_enabled', value=kwargs.get('mask_enabled', True), description='Whether to apply the mask in mask_phases during plotting, calculate_residuals, calculate_chi2, calculate_lnlikelihood, and run_solver')]
        # params += [FloatArrayParameter(visible_if='[component]mask_enabled:True', qualifier='mask_phases', component=kwargs.get('component_top', None), value=kwargs.get('mask_phases', []), default_unit=u.dimensionless_unscaled, required_shape=[None, 2], description='List of phase-tuples.  Any observations inside the range set by any of the tuples will be included.')]

        params += [ChoiceParameter(qualifier='solver_times', value=kwargs.get('solver_times', 'auto'), choices=['auto', 'compute_times', 'times'], description='times to use within run_solver.  auto: use compute_times if provided and shorter than times, otherwise use times.  compute_times: use compute_times if provided.  times: use times array.')]

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
    * `phases_period` (string, optional, default='period'): period to use
        when converting between `compute_phases` and `compute_times` as well as
        when applying `mask_phases`.  Only applicable if `syn` is False.  Not applicable for
        single stars (in which case period is always used) or if `dperdt == 0.0`.
    * `phases_dpdt` (string, optional, default='dpdt'): dpdt to use when
        converting between compute_times and compute_phases as well as when
        applying mask_phases.  Not applicable for single stars or if `dpdt == 0`
    * `phases_t0` (string, optional, default='t0_supconj'): t0 to use
        when converting between `compute_phases` and `compute_times`.  Only
        applicable if `syn` is False.  Not applicable for
        single stars (in which case t0@system is always used).

    Returns
    --------
    * (<phoebe.parameters.ParameterSet> or list, list): ParameterSet (if `as_ps`)
        or list of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params, constraints = [], []

    if syn:
        params += [FloatArrayParameter(qualifier='times', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('times', []), readonly=syn, default_unit=u.d, description='Model (synthetic) times' if syn else 'Observed times')]

    if syn:
        params += [FloatArrayParameter(qualifier='us', value=_empty_array(kwargs, 'us'), readonly=syn, default_unit=u.solRad, description='U position')]
        params += [FloatArrayParameter(qualifier='vs', value=_empty_array(kwargs, 'vs'), readonly=syn, default_unit=u.solRad, description='V position')]
        params += [FloatArrayParameter(qualifier='ws', value=_empty_array(kwargs, 'ws'), readonly=syn, default_unit=u.solRad, description='W position')]
        params += [FloatArrayParameter(qualifier='vus', value=_empty_array(kwargs, 'vus'), readonly=syn, default_unit=u.km/u.s, description='U velocity')]
        params += [FloatArrayParameter(qualifier='vvs', value=_empty_array(kwargs, 'vvs'), readonly=syn, default_unit=u.km/u.s, description='V velocity')]
        params += [FloatArrayParameter(qualifier='vws', value=_empty_array(kwargs, 'vws'), readonly=syn, default_unit=u.km/u.s, description='W velocity')]

    if not syn:
        params += [FloatArrayParameter(qualifier='compute_times', value=kwargs.get('compute_times', []), required_shape=[None], default_unit=u.d, description='Times to use during run_compute.  If empty, will use times parameter')]
        params += [FloatArrayParameter(qualifier='compute_phases', component=kwargs.get('component_top', None), value=kwargs.get('compute_phases', []), required_shape=[None], default_unit=u.dimensionless_unscaled, description='Phases associated with compute_times.')]
        params += [ChoiceParameter(qualifier='phases_period', visible_if='[dataset][context][kind]dperdt:!0.0,hierarchy.is_meshable:False', component=kwargs.get('component_top', None), value=kwargs.get('phases_period', 'period'), choices=['period', 'period_anom'], advanced=True, description='period to use when converting between compute_times and compute_phases as well as when applying mask_phases')]
        params += [ChoiceParameter(qualifier='phases_dpdt', visible_if='[dataset][context][kind]dpdt:!0.0,hierarchy.is_meshable:False', component=kwargs.get('component_top', None), value=kwargs.get('phases_dpdt', 'dpdt'), choices=['dpdt', 'none'], advanced=True, description='dpdt to use when converting between compute_times and compute_phases as well as when applying mask_phases')]
        params += [ChoiceParameter(qualifier='phases_t0', visible_if='hierarchy.is_meshable:False', component=kwargs.get('component_top', None), value=kwargs.get('phases_t0', 't0_supconj'), choices=['t0_supconj', 't0_perpass', 't0_ref'], advanced=True, description='t0 to use when converting between compute_times and compute_phases as well as when applying mask_phases')]
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
        if `syn` is True.  When `syn` is False: if provided, but `compute_times`
        is not provided, this will write to `compute_times` with a warning
        in the logger.
    * `compute_times` (array/quantity, optional): times at which to compute
        the model.  Only applicable if `syn` is False.
    * `compute_phases` (array/quantity, optional): phases at which to compute
        the model.  Only applicable if `syn` is False.
    * `phases_period` (string, optional, default='period'): period to use
        when converting between `compute_phases` and `compute_times` as well as
        when applying `mask_phases`.  Only applicable if `syn` is False.  Not applicable for
        single stars (in which case period is always used) or if `dperdt == 0.0`.
    * `phases_dpdt` (string, optional, default='dpdt'): dpdt to use when
        converting between compute_times and compute_phases as well as when
        applying mask_phases.  Not applicable for single stars or if `dpdt == 0`
    * `phases_t0` (string, optional, default='t0_supconj'): t0 to use
        when converting between `compute_phases` and `compute_times`.  Only
        applicable if `syn` is False.  Not applicable for
        single stars (in which case t0@system is always used).
    * `include_times` (string, optional): append to `compute_times` from the
        following datasets/time standards.  If referring to other datasets,
        this will copy the computed times of that dataset (whether that be
        from the `times` or `compute_times` of the respective dataset).
        Only applicable if `syn` is False.
    * `coordinates` (list, optional, default=['xyz', 'uvw']): coordinates to
        expose the mesh.  uvw (plane of sky) and/or xyz (roche).
    * `columns` (list, optional, default=[]): columns to expose within the mesh.
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

    if syn:
        # TODO: it would be nice if this wasn't copied per-component in the model... but it is also somewhat useful
        params += [FloatArrayParameter(qualifier='times', value=kwargs.get('times', []), readonly=syn, default_unit=u.d, description='Model (synthetic) times' if syn else 'Observed times')]

    if not syn:
        if 'times' in kwargs.keys():
            if 'compute_times' in kwargs.keys():
                raise KeyError("mesh dataset cannot accept both 'times' and 'compute_times'.")
            else:
                logger.warning("mesh datasets do not have a 'times' parameter.  Applying value sent to 'times' to 'compute_times'")
                compute_times = kwargs.get('times', [])
        else:
            compute_times = kwargs.get('compute_times', [])


        params += [FloatArrayParameter(qualifier='compute_times', value=compute_times, required_shape=[None], default_unit=u.d, description='Times to use during run_compute.')]
        params += [FloatArrayParameter(qualifier='compute_phases', component=kwargs.get('component_top', None), value=kwargs.get('compute_phases', []), required_shape=[None], default_unit=u.dimensionless_unscaled, description='Phases associated with compute_times.')]
        params += [ChoiceParameter(qualifier='phases_period', visible_if='[dataset][context][kind]dperdt:!0.0,hierarchy.is_meshable:False', component=kwargs.get('component_top', None), value=kwargs.get('phases_period', 'period'), choices=['period', 'period_anom'], advanced=True, description='period to use when converting between compute_times and compute_phases as well as when applying mask_phases')]
        params += [ChoiceParameter(qualifier='phases_dpdt', visible_if='[dataset][context][kind]dpdt:!0.0,hierarchy.is_meshable:False', component=kwargs.get('component_top', None), value=kwargs.get('phases_dpdt', 'dpdt'), choices=['dpdt', 'none'], advanced=True, description='dpdt to use when converting between compute_times and compute_phases as well as when applying mask_phases')]
        params += [ChoiceParameter(qualifier='phases_t0', visible_if='hierarchy.is_meshable:False', component=kwargs.get('component_top', None), value=kwargs.get('phases_t0', 't0_supconj'), choices=['t0_supconj', 't0_perpass', 't0_ref'], advanced=True, description='t0 to use when converting between compute_times and compute_phases as well as when applying mask_phases')]
        constraints += [(constraint.compute_phases, kwargs.get('component_top', None), kwargs.get('dataset', None))]

        params += [SelectParameter(qualifier='include_times', value=kwargs.get('include_times', []), required_shape=[None], advanced=False, description='append to compute_times from the following datasets/time standards', choices=['t0@system'])]
        params += [SelectParameter(qualifier='coordinates', value=kwargs.get('coordinates', ['xyz', 'uvw']), choices=['xyz', 'uvw'], advanced=True, description='coordinates to expose the mesh.  uvw (plane of sky) and/or xyz (roche)')]
        params += [SelectParameter(qualifier='columns', value=kwargs.get('columns', []), description='columns to expose within the mesh', choices=_mesh_columns)]

    # the following will all be arrays (value per triangle) per time
    if syn:
        columns = kwargs.get('mesh_columns', [])
        coordinates = kwargs.get('mesh_coordinates', ['xyz', 'uvw'])
        mesh_datasets = kwargs.get('mesh_datasets', [])

        for t in times:
            if not isinstance(t, float):
                raise ValueError("times must all be of type float")


            # basic geometric columns
            if 'uvw' in coordinates:
                params += [FloatArrayParameter(qualifier='uvw_elements', time=t, value=kwargs.get('uvw_elements', []), readonly=True, default_unit=u.solRad, advanced=True, description='Vertices of triangles in the plane-of-sky')]
                params += [FloatArrayParameter(qualifier='uvw_normals', time=t, value=kwargs.get('uvw_normals', []), readonly=True, default_unit=u.solRad, advanced=True, description='Normals of triangles in the plane-of-sky')]

            if 'xyz' in coordinates:
                params += [FloatArrayParameter(qualifier='xyz_elements', time=t, value=kwargs.get('xyz_elements ', []), readonly=True, default_unit=u.dimensionless_unscaled, advanced=True, description='Vertices of triangles in Roche coordinates')]
                params += [FloatArrayParameter(qualifier='xyz_normals', time=t, value=kwargs.get('xyz_normals ', []), readonly=True, default_unit=u.dimensionless_unscaled, advanced=True, description='Normals of triangles in Roche coordinates')]

            # NOTE: if changing the parameters which are optional, changes must
            # be made here, in the choices for the columns Parameter, and in
            # backends.py when the values are extracted and included in the
            # packet

            # if 'pot' in columns:
                # params += [FloatParameter(qualifier='pot', time=t, value=kwargs.get('pot', 0.0), default_unit=u.dimensionless_unscaled, description='Equipotential of the stellar surface')]
            # if 'rpole' in columns:
            #     params += [FloatParameter(qualifier='rpole', time=t, value=kwargs.get('rpole', 0.0), default_unit=u.solRad, description='Polar radius of the stellar surface')]
            if 'volume' in columns:
                params += [FloatParameter(qualifier='volume', time=t, value=kwargs.get('volume', 0.0), readonly=True, default_unit=u.solRad**3, description='Volume of the stellar surface')]


            if 'xs' in columns:
                params += [FloatArrayParameter(qualifier='xs', time=t, value=kwargs.get('xs', []), readonly=True, default_unit=u.solRad, description='X coordinate of center of triangles in the plane-of-sky')]
            if 'ys' in columns:
                params += [FloatArrayParameter(qualifier='ys', time=t, value=kwargs.get('ys', []), readonly=True, default_unit=u.solRad, description='Y coordinate of center of triangles in the plane-of-sky')]
            if 'zs' in columns:
                params += [FloatArrayParameter(qualifier='zs', time=t, value=kwargs.get('zs', []), readonly=True, default_unit=u.solRad, description='Z coordinate of center of triangles in the plane-of-sky')]

            if 'vxs' in columns:
                params += [FloatArrayParameter(qualifier='vxs', time=t, value=kwargs.get('vxs', []), readonly=True, default_unit=u.km/u.s, description='X velocity of center of triangles')]
            if 'vys' in columns:
                params += [FloatArrayParameter(qualifier='vys', time=t, value=kwargs.get('vys', []), readonly=True, default_unit=u.km/u.s, description='Y velocity of center of triangles')]
            if 'vzs' in columns:
                params += [FloatArrayParameter(qualifier='vzs', time=t, value=kwargs.get('vzs', []), readonly=True, default_unit=u.km/u.s, description='Z velocity of center of triangles')]

            if 'nxs' in columns:
                params += [FloatArrayParameter(qualifier='nxs', time=t, value=kwargs.get('nxs', []), readonly=True, default_unit=u.dimensionless_unscaled, description='X component of normals')]
            if 'nys' in columns:
                params += [FloatArrayParameter(qualifier='nys', time=t, value=kwargs.get('nys', []), readonly=True, default_unit=u.dimensionless_unscaled, description='Y component of normals')]
            if 'nzs' in columns:
                params += [FloatArrayParameter(qualifier='nzs', time=t, value=kwargs.get('nzs', []), readonly=True, default_unit=u.dimensionless_unscaled, description='Z component of normals')]

            if 'us' in columns:
                params += [FloatArrayParameter(qualifier='us', time=t, value=kwargs.get('us', []), readonly=True, default_unit=u.solRad, description='U coordinate of center of triangles in the plane-of-sky')]
            if 'vs' in columns:
                params += [FloatArrayParameter(qualifier='vs', time=t, value=kwargs.get('vs', []), readonly=True, default_unit=u.solRad, description='V coordinate of center of triangles in the plane-of-sky')]
            if 'ws' in columns:
                params += [FloatArrayParameter(qualifier='ws', time=t, value=kwargs.get('ws', []), readonly=True, default_unit=u.solRad, description='W coordinate of center of triangles in the plane-of-sky')]

            if 'vus' in columns:
                params += [FloatArrayParameter(qualifier='vus', time=t, value=kwargs.get('vus', []), readonly=True, default_unit=u.km/u.s, description='U velocity of center of triangles')]
            if 'vvs' in columns:
                params += [FloatArrayParameter(qualifier='vvs', time=t, value=kwargs.get('vvs', []), readonly=True, default_unit=u.km/u.s, description='V velocity of center of triangles')]
            if 'vws' in columns:
                params += [FloatArrayParameter(qualifier='vws', time=t, value=kwargs.get('vws', []), readonly=True, default_unit=u.km/u.s, description='W velocity of center of triangles')]

            if 'nus' in columns:
                params += [FloatArrayParameter(qualifier='nus', time=t, value=kwargs.get('nus', []), readonly=True, default_unit=u.dimensionless_unscaled, description='U component of normals')]
            if 'nvs' in columns:
                params += [FloatArrayParameter(qualifier='nvs', time=t, value=kwargs.get('nvs', []), readonly=True, default_unit=u.dimensionless_unscaled, description='V component of normals')]
            if 'nws' in columns:
                params += [FloatArrayParameter(qualifier='nws', time=t, value=kwargs.get('nws', []), readonly=True, default_unit=u.dimensionless_unscaled, description='W component of normals')]


            if 'areas' in columns:
                params += [FloatArrayParameter(qualifier='areas', time=t, value=kwargs.get('areas', []), readonly=True, default_unit=u.solRad**2, description='Area of triangles')]
            # if 'tareas' in columns:
                # params += [FloatArrayParameter(qualifier='tareas', time=t, value=kwargs.get('areas', []), readonly=True, default_unit=u.solRad**2, description='Area of WD triangles')]


            if 'rs' in columns:
                params += [FloatArrayParameter(qualifier='rs', time=t, value=kwargs.get('rs', []), readonly=True, default_unit=u.solRad, description='Distance of each triangle from center of mass (of the half-envelope for contacts)')]
            # if 'cosbetas' in columns:
            #     params += [FloatArrayParameter(qualifier='cosbetas', time=t, value=kwargs.get('cosbetas', []), readonly=True, default_unit=u.solRad, description='')]


            if 'loggs' in columns:
                params += [FloatArrayParameter(qualifier='loggs', time=t, value=kwargs.get('loggs', []), readonly=True, default_unit=u.dimensionless_unscaled, description='Local surface gravity')]
            if 'teffs' in columns:
                params += [FloatArrayParameter(qualifier='teffs', time=t, value=kwargs.get('teffs', []), readonly=True, default_unit=u.K, description='Local effective temperature')]

            if 'rprojs' in columns:
                params += [FloatArrayParameter(qualifier='rprojs', time=t, value=kwargs.get('rprojs', []), readonly=True, default_unit=u.solRad, description='Projected distance (on plane of sky) of each triangle from center of mass (of the half-envelope for contacts)')]
            if 'mus' in columns:
                params += [FloatArrayParameter(qualifier='mus', time=t, value=kwargs.get('mus', []), readonly=True, default_unit=u.dimensionless_unscaled, description='Mu')]
            if 'visible_centroids' in columns:
                params += [FloatArrayParameter(qualifier='visible_centroids', time=t, value=kwargs.get('visible_centroids', []), readonly=True, default_unit=u.solRad  if t is not None else u.dimensionless_unscaled, description='Center of the visible portion of each triangle')]
            if 'visibilities' in columns:
                params += [FloatArrayParameter(qualifier='visibilities', time=t, value=kwargs.get('visibilities', []), readonly=True, default_unit=u.dimensionless_unscaled, description='Visiblity of triangles (1=visible, 0.5=partial, 0=hidden)')]

            # params += [FloatArrayParameter(qualifier='horizon_xs', time=t, value=kwargs.get('horizon_xs', []), readonly=True, default_unit=u.solRad, description='Horizon of the mesh (x component)')]
            # params += [FloatArrayParameter(qualifier='horizon_ys', time=t, value=kwargs.get('horizon_ys', []), readonly=True, default_unit=u.solRad, description='Horizon of the mesh (y component)')]
            # params += [FloatArrayParameter(qualifier='horizon_zs', time=t, value=kwargs.get('horizon_zs', []), readonly=True, default_unit=u.solRad, description='Horizon of the mesh (z component)')]
            # params += [FloatArrayParameter(qualifier='horizon_analytic_xs', time=t, value=kwargs.get('horizon_analytic_xs', []), readonly=True, default_unit=u.solRad, description='Analytic horizon (interpolated, x component)')]
            # params += [FloatArrayParameter(qualifier='horizon_analytic_ys', time=t, value=kwargs.get('horizon_analytic_ys', []), readonly=True, default_unit=u.solRad, description='Analytic horizon (interpolated, y component)')]
            # params += [FloatArrayParameter(qualifier='horizon_analytic_zs', time=t, value=kwargs.get('horizon_analytic_zs', []), readonly=True, default_unit=u.solRad, description='Analytic horizon (interpolated, z component)')]

            for dataset in mesh_datasets:
                # if 'dls@{}'.format(dataset) in columns:
                    # params += [FloatArrayParameter(qualifier='dls', dataset=dataset, time=t, value=[], default_unit=u.nm, description='Per-element delta-lambda caused by doppler shift'.format(dataset))]
                if 'rvs@{}'.format(dataset) in columns:
                    params += [FloatArrayParameter(qualifier='rvs', dataset=dataset, time=t, value=[], readonly=True, default_unit=u.km/u.s, description='Per-element value of rvs for {} dataset'.format(dataset))]
                if 'intensities@{}'.format(dataset) in columns:
                    params += [FloatArrayParameter(qualifier='intensities', dataset=dataset, time=t, value=[], readonly=True, default_unit=u.W/u.m**3, description='Per-element value of intensities for {} dataset'.format(dataset))]
                if 'normal_intensities@{}'.format(dataset) in columns:
                    params += [FloatArrayParameter(qualifier='normal_intensities', dataset=dataset, time=t, value=[], readonly=True, default_unit=u.W/u.m**3, description='Per-element value of normal_intensities for {} dataset'.format(dataset))]
                if 'abs_intensities@{}'.format(dataset) in columns:
                    params += [FloatArrayParameter(qualifier='abs_intensities', dataset=dataset, time=t, value=[], readonly=True, default_unit=u.W/u.m**3, description='Per-element value of abs_intensities for {} dataset'.format(dataset))]
                if 'abs_normal_intensities@{}'.format(dataset) in columns:
                    params += [FloatArrayParameter(qualifier='abs_normal_intensities', dataset=dataset, time=t, value=[], readonly=True, default_unit=u.W/u.m**3, description='Per-element value of abs_normal_intensities for {} dataset'.format(dataset))]
                if 'boost_factors@{}'.format(dataset) in columns:
                    params += [FloatArrayParameter(qualifier='boost_factors', dataset=dataset, time=t, value=[], readonly=True, default_unit=u.dimensionless_unscaled, description='Per-element value of boost_factors for {} dataset'.format(dataset))]
                if 'ldint@{}'.format(dataset) in columns:
                    params += [FloatArrayParameter(qualifier='ldint', dataset=dataset, time=t, value=kwargs.get('ldint', []), readonly=True, default_unit=u.dimensionless_unscaled, description='Integral of the limb-darkening function')]

                if 'ptfarea@{}'.format(dataset) in columns:
                    params += [FloatParameter(qualifier='ptfarea', dataset=dataset, time=t, value=kwargs.get('ptfarea', 1.0), readonly=True, default_unit=u.m, description='Area of the passband transmission function')]
                if 'pblum_ext@{}'.format(dataset) in columns:
                    params += [FloatParameter(qualifier='pblum_ext', dataset=dataset, time=t, value=kwargs.get('pblum_ext', 0.0), readonly=True, default_unit=u.W, description='Passband Luminosity of entire star (after pblum scaling)')]
                if 'abs_pblum_ext@{}'.format(dataset) in columns:
                    params += [FloatParameter(qualifier='abs_pblum_ext', dataset=dataset, time=t, value=kwargs.get('abs_pblum_ext', 0.0), readonly=True, default_unit=u.W, description='Passband Luminosity of entire star (before pblum scaling)')]


    return ParameterSet(params) if as_ps else params, constraints


# del _empty_array
# del deepcopy
# del download_passband, list_installed_passbands, list_online_passbands, list_passbands, parameter_from_json, parse_json, send_if_client, update_if_client
# del fnmatch
