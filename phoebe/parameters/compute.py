
import numpy as np

from phoebe.parameters import *
from phoebe.parameters import dataset as _dataset
import phoebe.dynamics as dynamics
from phoebe.atmospheres import passbands # needed to get choices for 'atm' parameter
from phoebe import u
from phoebe import conf

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

passbands._init_passbands()  # TODO: move to module import
_atm_choices = list(set([atm for pb in passbands._pbtable.values() for atm in pb['atms']]))

def phoebe(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for compute options for the
    PHOEBE 2 backend.  This is the default built-in backend so no other
    pre-requisites are required.

    When using this backend, please see the
    [list of publications](https://phoebe-project/org/publications) and cite
    the appropriate references.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_compute>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_compute>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Arguments
    ----------
    * `enabled` (bool, optional): whether to create synthetics in compute/fitting
        run.
    * `dynamics_method` (string, optional): which method to use to determine the
        dynamics of components.
    * `ltte` (bool, optional): whether to correct for light travel time effects.
    * `atm` (string, optional): atmosphere tables
    * `irrad_method` (string, optional): which method to use to handle irradiation.
    * `boosting_method` (string, optional): type of boosting method.
    * `mesh_method` (string, optional): which method to use for discretizing
        the surface.
    * `eclipse_method` (string, optional): which method to use for determinging
        eclipses.
    * `rv_method` (string, optional): which method to use for compute radial
        velocities.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = []

    params += [BoolParameter(qualifier='enabled', copy_for={'context': 'dataset', 'dataset': '*'}, dataset='_default', value=kwargs.get('enabled', True), description='Whether to create synthetics in compute/fitting run')]

    # DYNAMICS
    params += [ChoiceParameter(qualifier='dynamics_method', value=kwargs.get('dynamics_method', 'keplerian'), choices=['keplerian', 'nbody', 'rebound', 'bs'] if conf.devel else ['keplerian'], description='Which method to use to determine the dynamics of components')]
    params += [BoolParameter(qualifier='ltte', value=kwargs.get('ltte', False), description='Correct for light travel time effects')]

    if conf.devel:
        params += [BoolParameter(visible_if='dynamics_method:nbody', qualifier='gr', value=kwargs.get('gr', False), description='Whether to account for general relativity effects')]
        params += [FloatParameter(visible_if='dynamics_method:nbody', qualifier='stepsize', value=kwargs.get('stepsize', 0.01), default_unit=None, description='stepsize for the N-body integrator')]         # TODO: improve description (and units??)
        params += [ChoiceParameter(visible_if='dynamics_method:nbody', qualifier='integrator', value=kwargs.get('integrator', 'ias15'), choices=['ias15', 'whfast', 'sei', 'leapfrog', 'hermes'], description='Which integrator to use within rebound')]

    # params += [FloatParameter(visible_if='dynamics_method:bs', qualifier='stepsize', value=kwargs.get('stepsize', 0.01), default_unit=None, description='stepsize for the N-body integrator')]         # TODO: improve description (and units??)
    # params += [FloatParameter(visible_if='dynamics_method:bs', qualifier='orbiterror', value=kwargs.get('orbiterror', 1e-20), default_unit=None, description='orbiterror for the N-body integrator')]  # TODO: improve description (and units??)


    # PHYSICS
    # TODO: should either of these be per-dataset... if so: copy_for={'kind': ['rv_dep', 'lc_dep'], 'dataset': '*'}, dataset='_default' and then edit universe.py to pull for the correct dataset (will need to become dataset-dependent dictionary a la ld_func)
    params += [ChoiceParameter(qualifier='irrad_method', value=kwargs.get('irrad_method', 'wilson'), choices=['none', 'wilson', 'horvat'], description='Which method to use to handle all irradiation effects (reflection, redistribution)')]
    params += [ChoiceParameter(qualifier='boosting_method', value=kwargs.get('boosting_method', 'none'), choices=['none', 'linear'], description='Type of boosting method')]

    # TODO: include scattering here? (used to be in lcdep)

    #params += [ChoiceParameter(qualifier='irradiation_alg', value=kwargs.get('irradiation_alg', 'point_source'), choices=['full', 'point_source'], description='Type of irradiation algorithm')]

    # MESH
    # -- these parameters all need to exist per-component --
    # copy_for = {'kind': ['star', 'disk', 'custombody'], 'component': '*'}
    # means that this should exist for each component (since that has a wildcard) which
    # has a kind in [star, disk, custombody]
    # params += [BoolParameter(qualifier='horizon', value=kwargs.get('horizon', False), description='Store horizon for all meshes (except protomeshes)')]
    params += [ChoiceParameter(copy_for={'kind': ['star', 'envelope'], 'component': '*'}, component='_default', qualifier='mesh_method', value=kwargs.get('mesh_method', 'marching'), choices=['marching', 'wd'] if conf.devel else ['marching'], description='Which method to use for discretizing the surface')]
    params += [IntParameter(visible_if='mesh_method:marching', copy_for={'kind': ['star', 'envelope'], 'component': '*'}, component='_default', qualifier='ntriangles', value=kwargs.get('ntriangles', 1500), limits=(100,None), default_unit=u.dimensionless_unscaled, description='Requested number of triangles (won\'t be exact).')]
    params += [ChoiceParameter(visible_if='mesh_method:marching', copy_for={'kind': ['star'], 'component': '*'}, component='_default', qualifier='distortion_method', value=kwargs.get('distortion_method', 'roche'), choices=['roche', 'rotstar', 'sphere'], description='Method to use for distorting stars')]


    if conf.devel:
        # TODO: can we have this computed from ntriangles? - and then do the same for the legacy compute options?
        # NOTE: if removing from developer mode - also need to remove if conf.devel in io.py line ~800
        params += [IntParameter(visible_if='mesh_method:wd', copy_for={'kind': ['star', 'envelope'], 'component': '*'}, component='_default', qualifier='gridsize', value=kwargs.get('gridsize', 60), limits=(10,None), default_unit=u.dimensionless_unscaled, description='Number of meshpoints for WD method')]
    # ------------------------------------------------------

    #params += [ChoiceParameter(qualifier='subdiv_alg', value=kwargs.get('subdiv_alg', 'edge'), choices=['edge'], description='Subdivision algorithm')]
    # params += [IntParameter(qualifier='subdiv_num', value=kwargs.get('subdiv_num', 3), limits=(0,None), description='Number of subdivisions')]



    if conf.devel:
        params += [BoolParameter(qualifier='mesh_offset', value=kwargs.get('mesh_offset', True), description='Whether to adjust the mesh to have the correct surface area (TESTING)')]
        params += [FloatParameter(visible_if='mesh_method:marching', qualifier='mesh_init_phi', value=kwargs.get('mesh_init_phi', 0.0), default_unit=u.rad, limits=(0,2*np.pi), description='Initial rotation offset for mesh (TESTING)')]

    # DISTORTION


    # ECLIPSE DETECTION
    params += [ChoiceParameter(qualifier='eclipse_method', value=kwargs.get('eclipse_method', 'native'), choices=['only_horizon', 'graham', 'none', 'visible_partial', 'native', 'wd_horizon'] if conf.devel else ['native', 'only_horizon'], description='Type of eclipse algorithm')]
    params += [ChoiceParameter(visible_if='eclipse_method:native', qualifier='horizon_method', value=kwargs.get('horizon_method', 'boolean'), choices=['boolean', 'linear'] if conf.devel else ['boolean'], description='Type of horizon method')]



    # PER-COMPONENT
    params += [ChoiceParameter(copy_for = {'kind': ['star'], 'component': '*'}, component='_default', qualifier='atm', value=kwargs.get('atm', 'ck2004'), choices=_atm_choices, description='Atmosphere table')]

    # PER-DATASET

    # -- these parameters all need to exist per-rvobs or lcobs --
    # copy_for = {'kind': ['rv_dep'], 'component': '*', 'dataset': '*'}
    # means that this should exist for each component/dataset pair with the
    # rv_dep kind
    params += [ChoiceParameter(qualifier='lc_method', copy_for = {'kind': ['lc'], 'dataset': '*'}, dataset='_default', value=kwargs.get('lc_method', 'numerical'), choices=['numerical', 'analytical'] if conf.devel else ['numerical'], description='Method to use for computing LC fluxes')]
    params += [ChoiceParameter(qualifier='fti_method', copy_for = {'kind': ['lc'], 'dataset': '*'}, dataset='_default', value=kwargs.get('fti_method', 'none'), choices=['none', 'oversample'], description='How to handle finite-time integration (when non-zero exptime)')]
    params += [IntParameter(visible_if='fti_method:oversample', qualifier='fti_oversample', copy_for={'kind': ['lc'], 'dataset': '*'}, dataset='_default', value=kwargs.get('fti_oversample', 5), default_unit=u.dimensionless_unscaled, description='Number of times to sample per-datapoint for finite-time integration')]
    params += [ChoiceParameter(qualifier='rv_method', copy_for = {'kind': ['rv'], 'component': '*', 'dataset': '*'}, component='_default', dataset='_default', value=kwargs.get('rv_method', 'flux-weighted'), choices=['flux-weighted', 'dynamical'], description='Method to use for computing RVs (must be flux-weighted for Rossiter-McLaughlin effects)')]
    params += [BoolParameter(visible_if='rv_method:flux-weighted', qualifier='rv_grav', copy_for = {'kind': ['rv'], 'component': '*', 'dataset': '*'}, component='_default', dataset='_default', value=kwargs.get('rv_grav', False), description='Whether gravitational redshift effects are enabled for RVs')]

    if conf.devel:
        params += [ChoiceParameter(qualifier='etv_method', copy_for = {'kind': ['etv'], 'component': '*', 'dataset': '*'}, component='_default', dataset='_default', value=kwargs.get('etv_method', 'crossing'), choices=['crossing'], description='Method to use for computing ETVs')]
        params += [FloatParameter(visible_if='etv_method:crossing', qualifier='etv_tol', copy_for = {'kind': ['etv'], 'component': '*', 'dataset': '*'}, component='_default', dataset='_default', value=kwargs.get('etv_tol', 1e-4), default_unit=u.d, description='Precision with which to determine eclipse timings')]
    # -----------------------------------------------------------



    return ParameterSet(params)


def legacy(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for compute options for the
    PHOEBE Legacy backend.

    Use PHOEBE 1.0 (legacy) which is based on the Wilson-Devinney code
    to compute radial velocities and light curves for binary systems
    (>2 stars not supported).  The code is available here:

    [http://phoebe-project.org/1.0](http://phoebe-project.org/1.0)

    PHOEBE 1.0 and the 'phoebeBackend' python interface must be installed
    and available on the system in order to use this plugin.

    When using this backend, please cite
    * Prsa & Zwitter (2005), ApJ, 628, 426

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_compute>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_compute>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Arguments
    ----------
    * `enabled` (bool, optional): whether to create synthetics in compute/fitting
        run.
    * `atm` (string, optional): atmosphere tables.
    * `gridsize` (float, optional): number of meshpoints for WD.
    * `irrad_method` (string, optional): which method to use to handle irradiation.
    * `ie` (bool, optional): whether data should be de-reddened.
    * `rv_method` (string, optional): which method to use for compute radial
        velocities.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = []

    params += [BoolParameter(qualifier='enabled', copy_for={'context': 'dataset', 'kind': ['lc', 'rv', 'mesh'], 'dataset': '*'}, dataset='_default', value=kwargs.get('enabled', True), description='Whether to create synthetics in compute/fitting run')]

    # TODO: the kwargs need to match the qualifier names!
    # TODO: include MORE meshing options
    params += [ChoiceParameter(copy_for = {'kind': ['star'], 'component': '*'}, component='_default', qualifier='atm', value=kwargs.get('atm', 'extern_atmx'), choices=['extern_atmx', 'extern_planckint'], description='Atmosphere table')]
#    params += [ChoiceParameter(copy_for = {'kind': ['star'], 'component': '*'}, component='_default', qualifier='atm', value=kwargs.get('atm', 'kurucz'), choices=['kurucz', 'blackbody'], description='Atmosphere table')]
#    params += [ChoiceParameter(qualifier='morphology', value=kwargs.get('morphology','Detached binary'), choices=['Unconstrained binary system', 'Detached binary', 'Overcontact binary of the W UMa type', 'Overcontact binary not in thermal contact'], description='System type constraint')]
#    params += [BoolParameter(qualifier='cindex', value=kwargs.get('cindex', False), description='Color index constraint')]
#    params += [IntParameter(visible_if='cindex_switch:True', qualifier='cindex', value=kwargs.get('cindex', np.array([1.0])), description='Number of reflections')]
#    params += [BoolParameter(qualifier='heating', value=kwargs.get('heating', True), description='Allow irradiators to heat other components')]
    params += [IntParameter(copy_for={'kind': ['star'], 'component': '*'}, component='_default', qualifier='gridsize', value=kwargs.get('gridsize', 60), limits=(10,None), description='Number of meshpoints for WD')]

    params += [ChoiceParameter(qualifier='irrad_method', value=kwargs.get('irrad_method', 'wilson'), choices=['none', 'wilson'], description='Which method to use to handle irradiation/reflection effects')]
    params += [IntParameter(visible_if='irrad_method:wilson', qualifier='refl_num', value=kwargs.get('refl_num', 1), limits=(0,None), description='Number of reflections')]

#    params += [BoolParameter(qualifier='msc1', value=kwargs.get('msc1', False), description='Mainsequence Constraint for star 1')]
#    params += [BoolParameter(qualifier='msc2', value=kwargs.get('msc2', False), description='Mainsequence Constraint for star 2')]


    # TODO: can we come up with a better qualifier for reddening (and be consistent when we enable in phoebe2)
    params += [BoolParameter(qualifier='ie', value=kwargs.get('ie', False), description='Should data be de-reddened')]

    # TODO: can we change this to rv_method = ['flux_weighted', 'dynamical'] to be consistent with phoebe2?
    # TODO: can proximity_rv (rv_method) be copied for each dataset (see how this is done for phoebe2)?  This would probably mean that the wrapper would need to loop and make separate calls since PHOEBE1 can't handle different settings per-RV dataset
    params += [ChoiceParameter(qualifier='rv_method', copy_for = {'kind': ['rv'], 'component': '*', 'dataset': '*'}, component='_default', dataset='_default',
                               value=kwargs.get('rv_method', 'flux-weighted'), choices=['flux-weighted', 'dynamical'], description='Method to use for computing RVs (must be flux-weighted for Rossiter-McLaughlin)')]

    return ParameterSet(params)

def photodynam(**kwargs):
    """
    Compute options for using Josh Carter's 'photodynam' code as a
    backend (must be installed).

    Generally, this will be used as an input to the kind argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_compute`

    Please see :func:`phoebe.backend.backends.photodynam` for a list of sources to
    cite when using this backend.

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """
    if not conf.devel:
        raise NotImplementedError("'photodynam' backend not officially supported for this release.  Enable developer mode to test.")

    params = []

    params += [BoolParameter(qualifier='enabled', copy_for={'context': 'dataset', 'kind': ['lc', 'rv', 'orb'], 'dataset': '*'}, dataset='_default', value=kwargs.get('enabled', True), description='Whether to create synthetics in compute/fitting run')]

    params += [FloatParameter(qualifier='stepsize', value=kwargs.get('stepsize', 0.01), default_unit=None, description='blah')]
    params += [FloatParameter(qualifier='orbiterror', value=kwargs.get('orbiterror', 1e-20), default_unit=None, description='blah')]

    # TODO: remove this option and instead use time0@system
    #params += [FloatParameter(qualifier='time0', value=kwargs.get('time0', 0.0), default_unit=u.d, description='Time to start the integration')]

    return ParameterSet(params)

def jktebop(**kwargs):
    """
    Compute options for using John Southworth's 'jktebop' code as a
    backend (must be installed).

    Generally, this will be used as an input to the kind argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_compute`

    Please see :func:`phoebe.backend.backends.jktebop` for a list of sources to
    cite when using this backend.

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """
    if not conf.devel:
        raise NotImplementedError("'jktebop' backend not officially supported for this release.  Enable developer mode to test.")

    params = []

    params += [BoolParameter(qualifier='enabled', copy_for={'context': 'dataset', 'kind': ['lc'], 'dataset': '*'}, dataset='_default', value=kwargs.get('enabled', True), description='Whether to create synthetics in compute/fitting run')]

    params += [FloatParameter(qualifier='ringsize', value=kwargs.get('ringsize', 5), default_unit=u.deg, description='Integ Ring Size')]

    return ParameterSet(params)
