
import numpy as np

from phoebe.parameters import *
from phoebe.parameters import dataset as _dataset
import phoebe.dynamics as dynamics
from phoebe.atmospheres import passbands # needed to get choices for 'atm' parameter
from phoebe import u
from phoebe import conf

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

passbands._init_passbands()  # TODO: move to module import
_atm_choices = list(set([atm for pb in passbands._pbtable.values() for atm in pb['atms'] if atm in passbands._supported_atms]))

def _sampling_params(**kwargs):
    """
    """
    params = []

    params += [SelectParameter(qualifier='sample_from', value=kwargs.get('sample_from', []), choices=[], description='distributions or solutions to use for sampling.  If pointing to a solution, adopt_solution(as_distributions=True, **kwargs) will be called to create a temporary distribution which will be removed automatically.  If all distributions are delta functions (face-values), sample_mode and sample_num will be ignored with a warning.')]
    params += [ChoiceParameter(visible_if='sample_from:<notempty>', qualifier='sample_from_combine', value=kwargs.get('sample_from_combine', 'first'), choices=['first'], description='Method to use to combine multiple distributions from sample_from for the same parameter.  first: ignore duplicate entries and take the first in the sample_from parameter')]
    params += [IntParameter(visible_if='sample_from:<notempty>', qualifier='sample_num', value=kwargs.get('sample_num', 10), limits=(8, 1e6), description='Number of forward models to run sampling from the distributions defined in sample_from and sample_from_combine.')]
    params += [ChoiceParameter(visible_if='sample_from:<notempty>', qualifier='sample_mode', value=kwargs.get('sample_mode', '1-sigma'), choices=['all', 'median', '1-sigma', '3-sigma', '5-sigma'], description='Mode to use when exposing model after sampling.  all: expose all sampled forward-models.  median: only return the median of all sampled models.  1/3/5-sigma: expose the synthetic variable at the median and +/- n-sigma.')]
    params += [BoolParameter(visible_if='sample_from:<notempty>', qualifier='expose_samples', value=kwargs.get('expose_samples', True), description='Whether to expose failed samples along with the simplified error messages.')]
    params += [BoolParameter(visible_if='sample_from:<notempty>', qualifier='expose_failed', value=kwargs.get('expose_failed', True), description='Whether to expose failed samples along with the simplified error messages.')]

    return params

def _comments_params(**kwargs):
    """
    """
    params = []

    params += [StringParameter(qualifier='comments', value=kwargs.get('comments', ''), description='User-provided comments for these compute-options.  Feel free to place any notes here - if not overridden, they will be copied to any resulting models.')]
    return params

def phoebe(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for compute options for the
    PHOEBE 2 backend.  This is the default built-in backend so no other
    pre-requisites are required.

    When using this backend, please see the
    http://phoebe-project.org/publications and cite
    the appropriate references.

    See also:
    * <phoebe.frontend.bundle.Bundle.references>

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_compute>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_compute>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_compute('phoebe')
    b.run_compute(kind='phoebe')
    ```

    Note that default bundles (<phoebe.frontend.bundle.Bundle.default_binary>, for example)
    include a set of compute options for the phoebe backend.

    Arguments
    ----------
    * `enabled` (bool, optional, default=True): whether to create synthetics in
        compute/solver runs.
    * `dynamics_method` (string, optional, default='keplerian'): which method to
        use to determine the dynamics of components.
    * `ltte` (bool, optional, default=False): whether to correct for light
        travel time effects.
    * `atm` (string, optional, default='ck2004'): atmosphere tables.
    * `irrad_method` (string, optional, default='horvat'): which method to use
        to handle irradiation.
    * `boosting_method` (string, optional, default='none'): type of boosting method.
    * `mesh_method` (string, optional, default='marching'): which method to use
        for discretizing the surface.
    * `ntriangles` (int, optional, default=1500): target number of triangles
        (only applicable if `mesh_method` is 'marching').
    * `distortion_method` (string, optional, default='roche'): what type of
        distortion to use when meshing the surface (only applicable
        if `mesh_method` is 'marching').
    * `eclipse_method` (string, optional, default='native'): which method to use
        for determinging eclipses.
    * `lc_method` (string, optional, default='numerical'): which method to use
        for computing light curves.
    * `fti_method` (string, optional, default='oversample'): method to use for
        handling finite-time of integration (exptime).
    * `fti_oversample` (int, optional, default=5): number of times to sample
        per-datapoint for finite-time integration (only applicable if
        `fti_method` is 'oversample').
    * `rv_method` (string, optional, default='flux-weighted'): which method to
        use for computing radial velocities.  If 'dynamical', Rossiter-McLaughlin
        effects will not be computed.
    * `rv_grav` (bool, optional, default=False): whether gravitational redshift
        effects are enabled for RVs (only applicable if `rv_method` is
        'flux-weighted')

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _sampling_params(**kwargs)
    params += _comments_params(**kwargs)

    params += [BoolParameter(qualifier='enabled', copy_for={'context': 'dataset', 'dataset': '*'}, dataset='_default', value=kwargs.get('enabled', True), description='Whether to create synthetics in compute/solver run')]
    params += [BoolParameter(qualifier='enabled', copy_for={'context': 'feature', 'feature': '*'}, feature='_default', value=kwargs.get('enabled', True), description='Whether to enable the feature in compute/solver run')]

    # DYNAMICS
    params += [ChoiceParameter(qualifier='dynamics_method', value=kwargs.get('dynamics_method', 'keplerian'), choices=['keplerian'], description='Which method to use to determine the dynamics of components')]
    params += [BoolParameter(qualifier='ltte', value=kwargs.get('ltte', False), description='Correct for light travel time effects')]

    if conf.devel:
        # note: even though bs isn't an option, its manually added as an option in test_dynamics and test_dynamics_grid
        params += [BoolParameter(visible_if='dynamics_method:bs', qualifier='gr', value=kwargs.get('gr', False), description='Whether to account for general relativity effects')]
        params += [FloatParameter(visible_if='dynamics_method:bs', qualifier='stepsize', value=kwargs.get('stepsize', 0.01), default_unit=None, description='stepsize for the N-body integrator')]         # TODO: improve description (and units??)
        params += [ChoiceParameter(visible_if='dynamics_method:bs', qualifier='integrator', value=kwargs.get('integrator', 'ias15'), choices=['ias15', 'whfast', 'sei', 'leapfrog', 'hermes'], description='Which integrator to use within rebound')]


    # PHYSICS
    # TODO: should either of these be per-dataset... if so: copy_for={'kind': ['rv_dep', 'lc_dep'], 'dataset': '*'}, dataset='_default' and then edit universe.py to pull for the correct dataset (will need to become dataset-dependent dictionary a la ld_func)
    params += [ChoiceParameter(qualifier='irrad_method', value=kwargs.get('irrad_method', 'horvat'), choices=['none', 'wilson', 'horvat'], description='Which method to use to handle all irradiation effects (reflection, redistribution)')]
    params += [ChoiceParameter(qualifier='boosting_method', value=kwargs.get('boosting_method', 'none'), choices=['none'], advanced=True, description='Type of boosting method')]

    # MESH
    # -- these parameters all need to exist per-component --
    # copy_for = {'kind': ['star', 'disk', 'custombody'], 'component': '*'}
    # means that this should exist for each component (since that has a wildcard) which
    # has a kind in [star, disk, custombody]
    # params += [BoolParameter(qualifier='horizon', value=kwargs.get('horizon', False), description='Store horizon for all meshes (except protomeshes)')]
    params += [ChoiceParameter(visible_if='hierarchy.is_meshable:true', copy_for={'kind': ['star', 'envelope'], 'component': '*'},
                               component='_default', qualifier='mesh_method',
                               value=kwargs.get('mesh_method', 'marching'), choices=['marching', 'wd'] if conf.devel else ['marching'],
                               description='Which method to use for discretizing the surface')]

    # NOTE: although the default here is 1500 for ntriangles, add_compute will
    # override this for envelopes already existing in the hierarchy (although
    # any new envelopes in which copy_for triggers a new ntriangles parameter
    # will still get 1500 as a default)
    params += [IntParameter(visible_if='mesh_method:marching,hierarchy.is_meshable:true', copy_for={'kind': ['star', 'envelope'], 'component': '*'}, component='_default', qualifier='ntriangles', value=kwargs.get('ntriangles', 1500), limits=(100,None), default_unit=u.dimensionless_unscaled, description='Requested number of triangles (won\'t be exact).')]
    params += [ChoiceParameter(visible_if='mesh_method:marching,hierarchy.is_meshable:true', copy_for={'kind': ['star'], 'component': '*'}, component='_default', qualifier='distortion_method', value=kwargs.get('distortion_method', 'roche'), choices=['roche', 'rotstar', 'sphere', 'none'], description='Method to use for distorting stars')]

    if conf.devel:
        # TODO: can we have this computed from ntriangles? - and then do the same for the legacy compute options?
        # NOTE: if removing from developer mode - also need to remove if conf.devel in io.py line ~800
        params += [IntParameter(visible_if='mesh_method:wd', copy_for={'kind': ['star', 'envelope'], 'component': '*'}, component='_default', qualifier='gridsize', value=kwargs.get('gridsize', 60), limits=(10,None), default_unit=u.dimensionless_unscaled, description='Number of meshpoints for WD method')]

    if conf.devel:
        params += [BoolParameter(qualifier='mesh_offset', value=kwargs.get('mesh_offset', True), advanced=True, description='Whether to adjust the mesh to have the correct surface area')]
        params += [FloatParameter(visible_if='mesh_method:marching', copy_for={'kind': ['star', 'envelope'], 'component': '*'}, component='_default', qualifier='mesh_init_phi', value=kwargs.get('mesh_init_phi', 0.0), default_unit=u.rad, limits=(0,2*np.pi), advanced=True, description='Initial rotation offset for mesh')]

    # ECLIPSE DETECTION
    params += [ChoiceParameter(qualifier='eclipse_method', value=kwargs.get('eclipse_method', 'native'), choices=['only_horizon', 'graham', 'none', 'visible_partial', 'native', 'wd_horizon'] if conf.devel else ['native', 'only_horizon'], advanced=True, description='Type of eclipse algorithm')]
    params += [ChoiceParameter(visible_if='eclipse_method:native', qualifier='horizon_method', value=kwargs.get('horizon_method', 'boolean'), choices=['boolean', 'linear'] if conf.devel else ['boolean'], advanced=True, description='Type of horizon method')]


    # PER-COMPONENT
    params += [ChoiceParameter(copy_for = {'kind': ['star'], 'component': '*'}, component='_default', qualifier='atm', value=kwargs.get('atm', 'ck2004'), choices=_atm_choices, description='Atmosphere table')]

    # PER-DATASET

    # -- these parameters all need to exist per-rvobs or lcobs --
    # copy_for = {'kind': ['rv_dep'], 'component': '*', 'dataset': '*'}
    # means that this should exist for each component/dataset pair with the
    # rv_dep kind
    # params += [ChoiceParameter(qualifier='lc_method', copy_for = {'kind': ['lc'], 'dataset': '*'}, dataset='_default', value=kwargs.get('lc_method', 'numerical'), choices=['numerical', 'analytical'] if conf.devel else ['numerical'], advanced=True, description='Method to use for computing LC fluxes')]
    params += [ChoiceParameter(qualifier='fti_method', copy_for = {'kind': ['lc'], 'dataset': '*'}, dataset='_default', value=kwargs.get('fti_method', 'none'), choices=['none', 'oversample'], description='How to handle finite-time integration (when non-zero exptime)')]
    params += [IntParameter(visible_if='fti_method:oversample', qualifier='fti_oversample', copy_for={'kind': ['lc'], 'dataset': '*'}, dataset='_default', value=kwargs.get('fti_oversample', 5), limits=(1,None), default_unit=u.dimensionless_unscaled, description='Number of times to sample per-datapoint for finite-time integration')]

    params += [ChoiceParameter(qualifier='rv_method', copy_for={'component': {'kind': 'star'}, 'dataset': {'kind': 'rv'}}, component='_default', dataset='_default', value=kwargs.get('rv_method', 'flux-weighted'), choices=['flux-weighted', 'dynamical'], description='Method to use for computing RVs (must be flux-weighted for Rossiter-McLaughlin effects)')]
    params += [BoolParameter(visible_if='rv_method:flux-weighted', qualifier='rv_grav', copy_for={'component': {'kind': 'star'}, 'dataset': {'kind': 'rv'}}, component='_default', dataset='_default', value=kwargs.get('rv_grav', False), description='Whether gravitational redshift effects are enabled for RVs')]

    if conf.devel:
        params += [ChoiceParameter(qualifier='etv_method', copy_for = {'kind': ['etv'], 'component': '*', 'dataset': '*'}, component='_default', dataset='_default', value=kwargs.get('etv_method', 'crossing'), choices=['crossing'], description='Method to use for computing ETVs')]
        params += [FloatParameter(visible_if='etv_method:crossing', qualifier='etv_tol', copy_for = {'kind': ['etv'], 'component': '*', 'dataset': '*'}, component='_default', dataset='_default', value=kwargs.get('etv_tol', 1e-4), default_unit=u.d, description='Precision with which to determine eclipse timings')]


    return ParameterSet(params)


def legacy(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for compute options for the
    [PHOEBE 1.0 legacy](http://phoebe-project.org/1.0) backend.

    See also:
    * <phoebe.frontend.bundle.Bundle.export_legacy>
    * <phoebe.frontend.bundle.Bundle.from_legacy>

    Use PHOEBE 1.0 (legacy) which is based on the Wilson-Devinney code
    to compute radial velocities and light curves for binary systems
    (>2 stars not supported).  The code is available here:

    http://phoebe-project.org/1.0

    PHOEBE 1.0 and the 'phoebeBackend' python interface must be installed
    and available on the system in order to use this plugin.

    When using this backend, please cite
    * Prsa & Zwitter (2005), ApJ, 628, 426

    See also:
    * <phoebe.frontend.bundle.Bundle.references>

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_compute>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_compute>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_compute('legacy')
    b.run_compute(kind='legacy')
    ```

    Arguments
    ----------
    * `enabled` (bool, optional, default=True): whether to create synthetics in
        compute/solver run.
    * `atm` (string, optional, default='extern_atmx'): atmosphere tables.
    * `pblum_method` (string, optional, default='phoebe'): Method to estimate
        passband luminosities and handle scaling of returned fluxes from legacy.
        stefan-boltzmann: approximate the star as a uniform sphere and estimate
        the luminosities from teff, requiv, logg, and abun from the internal
        passband and atmosphere tables.
        phoebe: build the mesh using roche distortion at time t0 and compute
        luminosities use the internal atmosphere tables (considerable overhead,
        but more accurate for distorted stars).
    * `gridsize` (int, optional, default=60): number of meshpoints for WD.
    * `distortion_method` (string, optional, default='roche'): method to use
        for distorting stars (legacy only supports roche).
    * `irrad_method` (string, optional, default='wilson'): which method to use
        to handle irradiation.
    * `refl_num` (int, optional, default=1): number of reflections (only applicable
        if `irrad_method` is 'wilson').
    * `ie` (bool, optional, default=False): whether data should be de-reddened.
    * `rv_method` (string, optional, default='flux-weighted'): which method to
        use for computing radial velocities.
    * `fti_method` (string, optional, default='none'): How to handle finite-time
        integration (when non-zero exptime)
    * `fit_oversample` (int, optiona, default=5): Number of times to sample
        per-datapoint for finite-time integration (only applicable when `fit_method`
        is 'oversample')

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _sampling_params(**kwargs)
    params += _comments_params(**kwargs)

    params += [BoolParameter(qualifier='enabled', copy_for={'context': 'dataset', 'kind': ['lc', 'rv', 'mesh'], 'dataset': '*'}, dataset='_default', value=kwargs.get('enabled', True), description='Whether to create synthetics in compute/solver run')]
    params += [BoolParameter(qualifier='enabled', copy_for={'context': 'feature', 'kind': ['spot', 'gaussian_process'], 'feature': '*'}, feature='_default', value=kwargs.get('enabled', True), description='Whether to enable the feature in compute/solver run')]

    params += [ChoiceParameter(copy_for = {'kind': ['star'], 'component': '*'}, component='_default', qualifier='atm', value=kwargs.get('atm', 'extern_atmx'), choices=['extern_atmx', 'extern_planckint'], description='Atmosphere table to use within legacy.  For estimating passband luminosities and flux scaling (see pblum_method), extern_atmx will use ck2004 and extern_planckint will use blackbody.')]
    params += [ChoiceParameter(qualifier='pblum_method', value=kwargs.get('pblum_method', 'phoebe'), choices=['stefan-boltzmann', 'phoebe'], description='Method to estimate passband luminosities and handle scaling of returned fluxes from legacy.  stefan-boltzmann: approximate the star as a uniform sphere and estimate the luminosities from teff, requiv, logg, and abun from the internal passband and atmosphere tables.  phoebe: build the mesh using roche distortion at time t0 and compute luminosities use the internal atmosphere tables (considerable overhead, but more accurate for distorted stars).')]

    params += [IntParameter(copy_for={'kind': ['star'], 'component': '*'}, component='_default', qualifier='gridsize', value=kwargs.get('gridsize', 60), limits=(10, 200), description='Number of meshpoints for WD')]

    params += [ChoiceParameter(copy_for={'kind': ['star'], 'component': '*'}, component='_default', qualifier='distortion_method', value=kwargs.get('distortion_method', 'roche'), choices=["roche"], description='Method to use for distorting stars (legacy only supports roche distortion)')]

    params += [ChoiceParameter(qualifier='irrad_method', value=kwargs.get('irrad_method', 'wilson'), choices=['none', 'wilson'], description='Which method to use to handle irradiation/reflection effects')]
    params += [IntParameter(visible_if='irrad_method:wilson', qualifier='refl_num', value=kwargs.get('refl_num', 1), limits=(0,None), description='Number of reflections')]


    # TODO: can we come up with a better qualifier for reddening (and be consistent when we enable in phoebe2)
    params += [BoolParameter(qualifier='ie', value=kwargs.get('ie', False), description='Should data be de-reddened')]

    params += [ChoiceParameter(qualifier='rv_method', copy_for={'component': {'kind': 'star'}, 'dataset': {'kind': 'rv'}}, component='_default', dataset='_default',
                               value=kwargs.get('rv_method', 'flux-weighted'), choices=['flux-weighted', 'dynamical'], description='Method to use for computing RVs (must be flux-weighted for Rossiter-McLaughlin)')]

    params += [ChoiceParameter(qualifier='fti_method', copy_for = {'kind': ['lc'], 'dataset': '*'}, dataset='_default', value=kwargs.get('fti_method', 'none'), choices=['none', 'oversample'], description='How to handle finite-time integration (when non-zero exptime)')]
    params += [IntParameter(visible_if='fti_method:oversample', qualifier='fti_oversample', copy_for={'kind': ['lc'], 'dataset': '*'}, dataset='_default', value=kwargs.get('fti_oversample', 5), limits=(1,None), default_unit=u.dimensionless_unscaled, description='Number of times to sample per-datapoint for finite-time integration')]


    return ParameterSet(params)

def photodynam(**kwargs):
    """
    **This backend is EXPERIMENTAL and requires developer mode to be enabled**

    **DO NOT USE FOR SCIENCE**

    Create a <phoebe.parameters.ParameterSet> for compute options for Josh
    Carter's [photodynam](http://github.com/phoebe-project/photodynam) code.

    Use photodynam to compute radial velocities and light curves.
    photodynam must be installed and available on the system in order to use
    this plugin.  The code is available here:

    http://github.com/phoebe-project/photodynam

    When using this backend, please cite
    * Science 4 February 2011: Vol. 331 no. 6017 pp. 562-565 DOI:10.1126/science.1201274
    * MNRAS (2012) 420 (2): 1630-1635. doi: 10.1111/j.1365-2966.2011.20151.x

    See also:
    * <phoebe.frontend.bundle.Bundle.references>

    The following parameters are "exported/translated" when using the photodynam
    backend:

    System:
    * t0

    Star:
    * mass
    * requiv

    Orbit:
    * sma
    * ecc
    * incl
    * per0
    * long_an
    * mean_anom

    Dataset:
    * ld_func (only supports quadratic)
    * ld_coeffs (will use <phoebe.frontend.bundle.Bundle.compute_ld_coeffs> if necessary)
    * pblum (will use <phoebe.frontend.bundle.Bundle.compute_pblums> if necessary)


    The following parameters are populated in the resulting model when using the
    photodynam backend:

    LCs:
    * times
    * fluxes

    RVs (dynamical only):
    * times
    * rvs

    ORBs:
    * times
    * us
    * vs
    * ws
    * vus
    * vvs
    * vws

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_compute>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_compute>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_compute('photodynam')
    b.run_compute(kind='photodynam')
    ```

    Arguments
    ----------
    * `enabled` (bool, optional, default=True): whether to create synthetics in
        compute/solver runs.
    * `stepsize` (float, optional, default=0.01): stepsize to use for dynamics
        integration.
    * `orbiterror` (float, optional, default=1e-20): error to use for dynamics
        integration.
    * `distortion_method` (string, optional, default='sphere'): method to use
        for distorting stars (photodynam only supports spherical stars).
    * `irrad_method` (string, optional, default='none'): method to use for
        irradiation (photodynam does not support irradiation).

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    if not conf.devel:
        raise NotImplementedError("'photodynam' backend not officially supported for this release.  Enable developer mode to test.")

    params = _sampling_params(**kwargs)
    params += _comments_params(**kwargs)

    params += [BoolParameter(qualifier='enabled', copy_for={'context': 'dataset', 'kind': ['lc', 'rv', 'orb'], 'dataset': '*'}, dataset='_default', value=kwargs.get('enabled', True), description='Whether to create synthetics in compute/solver run')]
    params += [BoolParameter(qualifier='enabled', copy_for={'context': 'feature', 'kind': ['gaussian_process'], 'feature': '*'}, feature='_default', value=kwargs.get('enabled', True), description='Whether to enable the feature in compute/solver run')]

    params += [ChoiceParameter(copy_for = {'kind': ['star'], 'component': '*'}, component='_default', qualifier='atm', value=kwargs.get('atm', 'ck2004'), advanced=True, choices=_atm_choices, description='Atmosphere table to use when estimating passband luminosities and flux scaling (see pblum_method).  Note photodynam itself does not support atmospheres.')]
    params += [ChoiceParameter(qualifier='pblum_method', value=kwargs.get('pblum_method', 'stefan-boltzmann'), choices=['stefan-boltzmann', 'phoebe'], description='Method to estimate passband luminosities and handle scaling of returned fluxes from photodynam.  stefan-boltzmann: approximate the star as a uniform sphere and estimate the luminosities from teff, requiv, logg, and abun from the internal passband and atmosphere tables.  phoebe: build the mesh using roche distortion at time t0 and compute luminosities use the internal atmosphere tables (considerable overhead, but more accurate for distorted stars).')]

    params += [FloatParameter(qualifier='stepsize', value=kwargs.get('stepsize', 0.01), default_unit=None, description='Stepsize to use for dynamics integration')]
    params += [FloatParameter(qualifier='orbiterror', value=kwargs.get('orbiterror', 1e-20), default_unit=None, description='Error to use for dynamics integraton')]

    params += [ChoiceParameter(copy_for={'kind': ['star'], 'component': '*'}, component='_default', qualifier='distortion_method', value=kwargs.get('distortion_method', 'sphere'), choices=["sphere"], description='Method to use for distorting stars (photodynam only supports spherical stars)')]

    params += [ChoiceParameter(qualifier='irrad_method', value=kwargs.get('irrad_method', 'none'), choices=['none'], description='Which method to use to handle all irradiation effects (ellc does not support irradiation)')]

    params += [ChoiceParameter(qualifier='fti_method', copy_for = {'kind': ['lc'], 'dataset': '*'}, dataset='_default', value=kwargs.get('fti_method', 'none'), choices=['none', 'oversample'], description='How to handle finite-time integration (when non-zero exptime)')]
    params += [IntParameter(visible_if='fti_method:oversample', qualifier='fti_oversample', copy_for={'kind': ['lc'], 'dataset': '*'}, dataset='_default', value=kwargs.get('fti_oversample', 5), limits=(1,None), default_unit=u.dimensionless_unscaled, description='Number of times to sample per-datapoint for finite-time integration')]


    return ParameterSet(params)

def jktebop(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for compute options for John
    Southworth's [jktebop](https://www.astro.keele.ac.uk/jkt/codes/jktebop.html) code.

    Use jktebop to compute radial velocities and light curves for binary systems.
    jktebop must be installed and available on the system in order to use
    this plugin.  The code is available here (currently tested with v40, requires
    v40+):

    http://www.astro.keele.ac.uk/jkt/codes/jktebop.html

    Please see the link above for a list of publications to cite when using this
    code.

    See also:
    * <phoebe.frontend.bundle.Bundle.references>

    Note on `distortion_method`: according to jktebop's website, "jktebop models
    the two components as biaxial spheroids for the calculation of the reflection
    and ellipsoidal effects, and as spheres for the eclipse shapes."

    Note that the wrapper around jktebop only uses its forward model.
    jktebop also includes its own solver methods, including bootstrapping.
    Those capabilities cannot be accessed from PHOEBE.

    The following parameters are "exported/translated" when using the jktebop
    backend:

    Star:
    * requiv
    * gravb_bol
    * irrad_frac_refl_bol
    * teff (surface-brightness approximated as pblum ratio according to pblum_method divided by square of respective requivs)

    Orbit:
    * sma
    * incl
    * q
    * ecosw
    * esinw
    * period
    * t0_supconj

    Dataset:
    * l3_frac (will be estimated if l3_mode=='flux', but will cost time)
    * ld_mode (cannot be 'interp'.  If 'lookup', coefficients are queried from PHOEBE tables and passed as ld_coeffs)
    * ld_func (supports linear, logarithmic, square_root, quadratic)
    * ld_coeffs (will call <phoebe.frontend.bundle.Bundle.compute_ld_coeffs> if necessary)
    * pblum (will use <phoebe.frontend.bundle.Bundle.compute_pblums> if necessary)

    Note that jktebop works in magnitudes (not fluxes) and is normalized at
    quadrature.  Once converted to fluxes, these are then re-scaled according
    to `pblum_method`.  This renormalization
    is crude and should not be trusted to give absolute fluxes, but should behave
    reasonably with plbum_mode='dataset-scaled'.

    The following parameters are populated in the resulting model when using the
    jktebop backend:

    LCs:
    * times
    * fluxes

    RVs:
    * times
    * rvs

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_compute>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_compute>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_compute('jktebop')
    b.run_compute(kind='jktebop')
    ```

    Arguments
    ----------
    * `enabled` (bool, optional, default=True): whether to create synthetics in
        compute/solver runs.
    * `requiv_max_limit` (float, optional, default=0.5): Maximum allowed fraction
        of requiv_max (as jktebop does not handle highly distorted systems)
        before raising an error in <phoebe.frontend.bundle.Bundle.run_checks_compute>.
    * `atm` (string, optional, default='ck2003'): Atmosphere table to use when
        estimating passband luminosities and flux scaling (see pblum_method).
        Note jktebop itself does not support atmospheres.
    * `pblum_method` (string, optional, default='stefan-boltzmann'): Method to
        estimate passband luminosities and handle scaling of returned fluxes from
        jktebop.  stefan-boltzmann: approximate the star as a uniform sphere and
        estimate the luminosities from teff, requiv, logg, and abun from the
        internal passband and atmosphere tables.  phoebe: build the mesh using
        roche distortion at time t0 and compute luminosities use the internal
         atmosphere tables (considerable overhead, but more accurate for
         distorted stars).
    * `ringsize` (float, optional, default=5): integration ring size.
    * `rv_method` (string, optional, default='dynamical'): Method to use for
        computing RVs.  jktebop only supports dynamical (Keplerian) RVs.
    * `distortion_method` (string, optional, default='sphere/biaxial spheroid'):
        Method to use for distorting stars (applies to all components).
        sphere/biaxial-spheroid: spheres for eclipse shapes and biaxial spheroid
        for calculation of ellipsoidal effects and reflection,
        sphere: sphere for eclipse shapes and no ellipsoidal or reflection effects
    * `irrad_method` (string, optional, default='biaxial spheroid'): method
        to use for computing irradiation.  See note above regarding jktebop's
        treatment of `distortion_method`.
    * `irrad_method` (string, optional, default='none'): method to use for
        irradiation (ellc does not support irradiation).

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _sampling_params(**kwargs)
    params += _comments_params(**kwargs)

    params += [BoolParameter(qualifier='enabled', copy_for={'context': 'dataset', 'kind': ['lc', 'rv'], 'dataset': '*'}, dataset='_default', value=kwargs.get('enabled', True), description='Whether to create synthetics in compute/solver run')]
    params += [BoolParameter(qualifier='enabled', copy_for={'context': 'feature', 'kind': ['gaussian_process'], 'feature': '*'}, feature='_default', value=kwargs.get('enabled', True), description='Whether to enable the feature in compute/solver run')]

    params += [FloatParameter(qualifier='requiv_max_limit', value=kwargs.get('requiv_max_limit', 0.5), limits=(0.01,1), default_unit=u.dimensionless_unscaled, advanced=True, description='Maximum allowed fraction of requiv_max (as jktebop does not handle highly distorted systems) before raising an error in run_checks_compute.')]

    params += [ChoiceParameter(copy_for = {'kind': ['star'], 'component': '*'}, component='_default', qualifier='atm', value=kwargs.get('atm', 'ck2004'), advanced=True, choices=_atm_choices, description='Atmosphere table to use when estimating passband luminosities and flux scaling (see pblum_method).  Note jktebop itself does not support atmospheres.')]
    params += [ChoiceParameter(qualifier='pblum_method', value=kwargs.get('pblum_method', 'stefan-boltzmann'), choices=['stefan-boltzmann', 'phoebe'], description='Method to estimate passband luminosities and handle scaling of returned fluxes from jktebop.  stefan-boltzmann: approximate the star as a uniform sphere and estimate the luminosities from teff, requiv, logg, and abun from the internal passband and atmosphere tables.  phoebe: build the mesh using roche distortion at time t0 and compute luminosities use the internal atmosphere tables (considerable overhead, but more accurate for distorted stars).')]

    params += [FloatParameter(qualifier='ringsize', value=kwargs.get('ringsize', 5), default_unit=u.deg, description='Integration ring size')]

    params += [ChoiceParameter(qualifier='rv_method', copy_for = {'component': {'kind': 'star'}, 'dataset': {'kind': 'rv'}}, component='_default', dataset='_default',
                               value=kwargs.get('rv_method', 'dynamical'), choices=['dynamical'], description='Method to use for computing RVs.  jktebop only supports dynamical (Keplerian) RVs.')]


    params += [ChoiceParameter(qualifier='distortion_method', value=kwargs.get('distortion_method', 'sphere/biaxial spheroid'), choices=["sphere/biaxial spheroid", "sphere"], description='Method to use for distorting stars (applies to all components). sphere/biaxial-spheroid: spheres for eclipse shapes and biaxial spheroid for calculation of ellipsoidal effects and reflection, sphere: sphere for eclipse shapes and no ellipsoidal or reflection effects')]
    params += [ChoiceParameter(qualifier='irrad_method', value=kwargs.get('irrad_method', 'biaxial-spheroid'), choices=['none', 'biaxial-spheroid'], description='Which method to use to handle all irradiation effects')]

    params += [ChoiceParameter(qualifier='fti_method', copy_for = {'kind': ['lc'], 'dataset': '*'}, dataset='_default', value=kwargs.get('fti_method', 'none'), choices=['none', 'oversample'], description='How to handle finite-time integration (when non-zero exptime)')]
    params += [IntParameter(visible_if='fti_method:oversample', qualifier='fti_oversample', copy_for={'kind': ['lc'], 'dataset': '*'}, dataset='_default', value=kwargs.get('fti_oversample', 5), limits=(1,None), default_unit=u.dimensionless_unscaled, description='Number of times to sample per-datapoint for finite-time integration')]

    return ParameterSet(params)

def ellc(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for compute options for Pierre
    Maxted's [ellc](https://github.com/pmaxted/ellc) code.

    Use ellc to compute radial velocities and light curves for binary systems.
    ellc must be installed and available on the system in order to use
    this plugin (tested with v 1.8.1).  The code is available here:

    https://github.com/pmaxted/ellc

    and can be installed via pip:

    ```py
    pip install ellc
    ```

    Please cite the following when using this backend:

    https://ui.adsabs.harvard.edu/abs/2016A%26A...591A.111M/abstract

    See also:
    * <phoebe.frontend.bundle.Bundle.references>

    Note that the wrapper around ellc only uses its forward model.
    ellc also includes its own solver methods, including emcee.
    Those capabilities cannot be accessed from PHOEBE.

    The following parameters are "exported/translated" when using the ellc
    backend:

    Star:
    * requiv (passed as relative radii by dividing by sma)
    * syncpar
    * gravb_bol
    * teff (surface-brightness approximated as pblum ratio according to pblum_method divided by square of respective requivs)
    * irrad_frac_refl_bol
    * yaw (misalignment only supported with `distortion_method='sphere'` and only included for Rossiter-McLaughlin contribution to RVs)

    Orbit:
    * sma
    * period_anom
    * q
    * incl
    * ecc (passed as `sqrt(ecc)*cos(per0)` and `sqrt(ecc)*sin(per0)`)
    * per0 (passed as `sqrt(ecc)*cos(per0)` and `sqrt(ecc)*sin(per0)`, translated from t0@system to t0_supconj)
    * dperdt (passed as dperdt/period where period is the sidereal period)
    * t0_supconj

    System:
    * vgamma

    Feature (spots only):
    * colat (passed as latitude=-colat)
    * long
    * radius
    * relteff (passed as brightness_factor = relteff^4)

    Dataset (LC/RV only):
    * l3_frac (will be estimated if l3_mode=='flux', but will cost time)
    * ld_mode (cannot be 'interp'.  If 'lookup', coefficients are queried from PHOEBE tables and passed as ld_coeffs)
    * ld_func (supports linear, quadratic, logarithmic, square_root, power)
    * ld_coeffs (will call <phoebe.frontend.bundle.Bundle.compute_ld_coeffs> if necessary)
    * pblum (will use <phoebe.frontend.bundle.Bundle.compute_pblums> if necessary)

    Note: ellc returns fluxes that are normalized based on the sum of the irradiated
    faces of each of the components.  These are then rescaled according to
    `pblum_method`.  Note that this re-normalization is not exact, but should behave
    reasonably with plbum_mode='dataset-scaled'.

    The following parameters are populated in the resulting model when using the
    ellc backend:

    LCs:
    * times
    * fluxes

    RVs:
    * times
    * rvs

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_compute>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_compute>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_compute('ellc')
    b.run_compute(kind='ellc')
    ```

    Arguments
    ----------
    * `enabled` (bool, optional, default=True): whether to create synthetics in
        compute/solver runs.
    * `atm` (string, optional, default='ck2003'): Atmosphere table to use when
        estimating passband luminosities and flux scaling (see pblum_method).
        Note jktebop itself does not support atmospheres.
    * `pblum_method` (string, optional, default='stefan-boltzmann'): Method to
        estimate passband luminosities and handle scaling of returned fluxes from
        jktebop.  stefan-boltzmann: approximate the star as a uniform sphere and
        estimate the luminosities from teff, requiv, logg, and abun from the
        internal passband and atmosphere tables.  phoebe: build the mesh using
        roche distortion at time t0 and compute luminosities use the internal
         atmosphere tables (considerable overhead, but more accurate for
         distorted stars).
    * `distortion_method` (string, optional, default='roche'): method to use
        for distorting stars.
    * `hf` (float, optional, default=1.5): fluid second love number (only applicable
        if/when `distortion_method`='love')
    * `grid` (string, optional, default='default'): grid size used to calculate the flux.
    * `exact_grav` (bool, optional, default=False): whether to use point-by-point
        calculation of local surface gravity for calculation of gravity darkening
        or a (much faster) approximation based on functional form fit to local
        gravity at 4 points on the star.
    * `rv_method` (string, optional, default='flux-weighted'): which method to
        use for computing radial velocities.
    * `irrad_method` (string, optional, default='none'): method to use for
        irradiation (ellc does not support irradiation).
    * `fti_method` (string, optional, default='none'): method to use when accounting
        for finite exposure times.
    * `fti_oversample` (int, optional, default=1): number of integration points
        used to account for finite exposure time.  Only used if `fti_method`='oversample'.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _sampling_params(**kwargs)
    params += _comments_params(**kwargs)

    params += [BoolParameter(qualifier='enabled', copy_for={'context': 'dataset', 'kind': ['lc', 'rv'], 'dataset': '*'}, dataset='_default', value=kwargs.get('enabled', True), description='Whether to create synthetics in compute/solver run')]
    params += [BoolParameter(qualifier='enabled', copy_for={'context': 'feature', 'kind': ['spot', 'gaussian_process'], 'feature': '*'}, feature='_default', value=kwargs.get('enabled', True), description='Whether to enable the feature in compute/solver run')]

    params += [ChoiceParameter(copy_for = {'kind': ['star'], 'component': '*'}, component='_default', qualifier='atm', value=kwargs.get('atm', 'ck2004'), advanced=True, choices=_atm_choices, description='Atmosphere table to use when estimating passband luminosities and flux scaling (see pblum_method).  Note ellc itself does not support atmospheres.')]
    params += [ChoiceParameter(qualifier='pblum_method', value=kwargs.get('pblum_method', 'stefan-boltzmann'), choices=['stefan-boltzmann', 'phoebe'], description='Method to estimate passband luminosities and handle scaling of returned fluxes from ellc.  stefan-boltzmann: approximate the star as a uniform sphere and estimate the luminosities from teff, requiv, logg, and abun from the internal passband and atmosphere tables.  phoebe: build the mesh using roche distortion at time t0 and compute luminosities use the internal atmosphere tables (considerable overhead, but more accurate for distorted stars).')]

    params += [ChoiceParameter(copy_for={'kind': ['star'], 'component': '*'}, component='_default', qualifier='distortion_method', value=kwargs.get('distortion_method', 'roche'), choices=["roche", "roche_v", "sphere", "poly1p5", "poly3p0", "love"], description='Method to use for distorting stars')]
    params += [FloatParameter(visible_if='distortion_method:love', copy_for={'kind': ['star'], 'component': '*'}, component='_default', qualifier='hf', value=kwargs.get('hf', 1.5), limits=(0,None), default_unit=u.dimensionless_unscaled, description='fluid second love number for radial displacement')]


    params += [ChoiceParameter(copy_for={'kind': ['star'], 'component': '*'}, component='_default', qualifier='grid', value=kwargs.get('grid', 'default'), choices=['very_sparse', 'sparse', 'default', 'fine', 'very_fine'], description='Grid size used to calculate the flux.')]

    params += [BoolParameter(qualifier='exact_grav', value=kwargs.get('exact_grav', False), description='Whether to use point-by-point calculation of local surface gravity for calculation of gravity darkening or a (much faster) approximation based on functional form fit to local gravity at 4 points on the star.')]

    params += [ChoiceParameter(qualifier='rv_method', copy_for = {'component': {'kind': 'star'}, 'dataset': {'kind': 'rv'}}, component='_default', dataset='_default',
                               value=kwargs.get('rv_method', 'dynamical'), choices=['flux-weighted', 'dynamical'], description='Method to use for computing RVs (must be flux-weighted for Rossiter-McLaughlin).  Note that \'flux-weighted\' is not allowed and will raise an error if irradiation is enabled (see irrad_method).')]


    # copy for RV datasets once exptime support for RVs in phoebe
    params += [ChoiceParameter(qualifier='fti_method', copy_for = {'kind': ['lc'], 'dataset': '*'}, dataset='_default', value=kwargs.get('fti_method', 'none'), choices=['none', 'ellc', 'oversample'], description='How to handle finite-time integration (when non-zero exptime).  ellc: use ellcs native oversampling. oversample: use phoebe\'s oversampling')]
    params += [IntParameter(visible_if='fti_method:ellc|oversample', qualifier='fti_oversample', copy_for={'kind': ['lc'], 'dataset': '*'}, dataset='_default', value=kwargs.get('fti_oversample', 5), limits=(1, None), default_unit=u.dimensionless_unscaled, description='number of integration points used to account for finite exposure time.')]

    params += [ChoiceParameter(qualifier='irrad_method', value=kwargs.get('irrad_method', 'lambert'), choices=['lambert', 'none'], description='Which method to use to handle all irradiation effects.  Note that irradiation and rv_method=\'flux-weighted\' cannot be used together.')]

    return ParameterSet(params)

# del deepcopy
# del download_passband, list_installed_passbands, list_online_passbands, list_passbands, parameter_from_json, parse_json, send_if_client, update_if_client
# del fnmatch
