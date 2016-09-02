
import numpy as np

from phoebe.parameters import *
from phoebe.parameters import dataset as _dataset
import phoebe.dynamics as dynamics
from phoebe.atmospheres import passbands # needed to get choices for 'atm' parameter
from phoebe import u

passbands.init_passbands()  # TODO: move to module import
_atm_choices = list(set([atm for pb in passbands._pbtable.values() for atm in pb['atms']]))

def phoebe(**kwargs):
    """
    Compute options for using the PHOEBE 2.0 backend.

    Generally, this will be used as an input to the method argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_compute`

    Please see :func:`phoebe.backend.backends.phoebe` for a list of sources to
    cite when using this backend.

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """
    params = []

    params += [BoolParameter(qualifier='enabled', copy_for={'dataset': '*'}, relevant_if='False', dataset='_default', value=kwargs.get('enabled', True), description='Whether to create synthetics in compute/fitting run')]

    # DYNAMICS
    params += [ChoiceParameter(qualifier='dynamics_method', value=kwargs.get('dynamics_method', 'keplerian'), choices=['keplerian', 'nbody', 'rebound', 'bs'], description='Which method to use to determine the dynamics of components')]
    params += [BoolParameter(qualifier='ltte', value=kwargs.get('ltte', False), description='Correct for light travel time effects')]

    params += [BoolParameter(relevant_if='dynamics_method:nbody', qualifier='gr', value=kwargs.get('gr', False), description='Whether to account for general relativity effects')]
    params += [FloatParameter(relevant_if='dynamics_method:nbody', qualifier='stepsize', value=kwargs.get('stepsize', 0.01), default_unit=None, description='stepsize for the N-body integrator')]         # TODO: improve description (and units??)
    params += [ChoiceParameter(relevant_if='dynamics_method:nbody', qualifier='integrator', value=kwargs.get('integrator', 'ias15'), choices=['ias15', 'whfast', 'sei', 'leapfrog', 'hermes'], description='Which integrator to use within rebound')]

    # params += [FloatParameter(relevant_if='dynamics_method:bs', qualifier='stepsize', value=kwargs.get('stepsize', 0.01), default_unit=None, description='stepsize for the N-body integrator')]         # TODO: improve description (and units??)
    # params += [FloatParameter(relevant_if='dynamics_method:bs', qualifier='orbiterror', value=kwargs.get('orbiterror', 1e-20), default_unit=None, description='orbiterror for the N-body integrator')]  # TODO: improve description (and units??)


    # PHYSICS
    #params += [BoolParameter(qualifier='heating', value=kwargs.get('heating', True), description='Allow irradiators to heat other components')]
    params += [BoolParameter(qualifier='refl', value=kwargs.get('refl', False), description='Enable reflection/heating/scattering')]
    #params += [IntParameter(relevant_if='refl:True', qualifier='refl_num', value=kwargs.get('refl_num', 1), limits=(0,None), description='Number of reflections')]

    # TODO: boosting alg should be per-lcdep (note: not lcobs) - maybe per rvdep as well since those require intensities?
    # would that be copy_for = {'method': ['RV_dep', 'LC_dep'], 'component': '*', 'dataset': '*'} ???
    #params += [ChoiceParameter(qualifier='boosting_alg', value=kwargs.get('boosting_alg', 'None'), choices=['None', 'simple', 'local', 'full'], description='Type of boosting algorithm')]

    # TODO: include scattering here? (used to be in lcdep)

    #params += [ChoiceParameter(qualifier='irradiation_alg', value=kwargs.get('irradiation_alg', 'point_source'), choices=['full', 'point_source'], description='Type of irradiation algorithm')]

    # MESH
    # -- these parameters all need to exist per-component --
    # copy_for = {'method': ['star', 'disk', 'custombody'], 'component': '*'}
    # means that this should exist for each component (since that has a wildcard) which
    # has a method in [star, disk, custombody]
    params += [BoolParameter(qualifier='store_mesh', value=kwargs.get('store_mesh', False), description='Store a protomesh (reference frame of stars) and filled meshes at each timepoint and which a mesh is computed')]
    params += [ChoiceParameter(copy_for={'method': ['star', 'envelope'], 'component': '*'}, component='_default', qualifier='mesh_method', value=kwargs.get('mesh_method', 'marching'), choices=['marching', 'wd'], descriptio='Which method to use for discretizing the surface')]
    params += [FloatParameter(relevant_if='mesh_method:marching', copy_for={'method': ['star', 'envelope'], 'component': '*'}, component='_default', qualifier='delta', value=kwargs.get('delta', 0.1), limits=(1e-9,None), default_unit=u.dimensionless_unscaled, description='Stepsize for mesh generation via marching method')]
    params += [IntParameter(relevant_if='mesh_method:marching', copy_for={'method': ['star', 'envelope'], 'component': '*'}, component='_default', qualifier='maxpoints', value=kwargs.get('maxpoints', 100000), limits=(10,None), default_unit=u.dimensionless_unscaled, description='Maximum number of triangles for marching method')]
    params += [ChoiceParameter(relevant_if='mesh_method:marching', copy_for={'method': ['star', 'envelope'], 'component': '*'}, component='_default', qualifier='distortion_method', value=kwargs.get('distortion_method', 'roche'), choices=['roche', 'rotstar', 'nbody', 'sphere'], description='Method to use for distorting stars')]
    params += [IntParameter(relevant_if='mesh_method:wd', copy_for={'method': ['star', 'envelope'], 'component': '*'}, component='_default', qualifier='gridsize', value=kwargs.get('gridsize', 40), limits=(10,None), default_unit=u.dimensionless_unscaled, description='Number of meshpoints for WD method')]
    # ------------------------------------------------------

    #params += [ChoiceParameter(qualifier='subdiv_alg', value=kwargs.get('subdiv_alg', 'edge'), choices=['edge'], description='Subdivision algorithm')]
    # params += [IntParameter(qualifier='subdiv_num', value=kwargs.get('subdiv_num', 3), limits=(0,None), description='Number of subdivisions')]


    params += [BoolParameter(qualifier='mesh_offset', value=kwargs.get('mesh_offset', True), description='Whether to adjust the mesh to have the correct surface area (TESTING)')]

    # DISTORTION


    # ECLIPSE DETECTION
    params += [ChoiceParameter(qualifier='eclipse_alg', value=kwargs.get('eclipse_alg', 'visible_ratio'), choices=['only_horizon', 'graham', 'none', 'visible_partial', 'visible_ratio', 'wd_horizon'], description='Type of eclipse algorithm')]



    # PER-COMPONENT
    params += [ChoiceParameter(copy_for = {'method': ['star','envelope'], 'component': '*'}, component='_default', qualifier='atm', value=kwargs.get('atm', 'ck2004'), choices=_atm_choices, description='Atmosphere table')]

    # PER-DATASET

    # -- these parameters all need to exist per-rvobs or lcobs --
    # copy_for = {'method': ['rv_dep'], 'component': '*', 'dataset': '*'}
    # means that this should exist for each component/dataset pair with the
    # rv_dep method
    params += [ChoiceParameter(qualifier='lc_method', copy_for = {'method': ['LC'], 'dataset': '*'}, component='_default', dataset='_default', value=kwargs.get('lc_method', 'numerical'), choices=['numerical', 'analytical'], description='Method to use for computing LC fluxes')]
    params += [ChoiceParameter(qualifier='rv_method', copy_for = {'method': ['RV'], 'component': '*', 'dataset': '*'}, component='_default', dataset='_default', value=kwargs.get('rv_method', 'flux-weighted'), choices=['flux-weighted', 'dynamical'], description='Method to use for computing RVs (must be flux-weighted for Rossiter-McLaughlin)')]
    params += [BoolParameter(relevant_if='rv_method:flux-weighted', qualifier='rv_grav', copy_for = {'method': ['RV'], 'component': '*', 'dataset': '*'}, component='_default', dataset='_default', value=kwargs.get('rv_grav', False), description='Whether gravitational redshift effects are enabled for RVs')]
    params += [ChoiceParameter(qualifier='etv_method', copy_for = {'method': ['ETV'], 'component': '*', 'dataset': '*'}, component='_default', dataset='_default', value=kwargs.get('etv_method', 'crossing'), choices=['crossing'], description='Method to use for computing ETVs')]
    params += [FloatParameter(relevant_if='etv_method:crossing', qualifier='etv_tol', copy_for = {'method': ['ETV'], 'component': '*', 'dataset': '*'}, component='_default', dataset='_default', value=kwargs.get('etv_tol', 1e-4), default_unit=u.d, description='Precision with which to determine eclipse timings')]
    # -----------------------------------------------------------



    return ParameterSet(params)


def legacy(**kwargs):
    """
    Compute options for using the PHOEBE 1.0 legacy backend (must be
    installed).

    Generally, this will be used as an input to the method argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_compute`

    Please see :func:`phoebe.backend.backends.legacy` for a list of sources to
    cite when using this backend.

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """
    params = []

    params += [BoolParameter(qualifier='enabled', copy_for={'method': ['LC', 'RV'], 'dataset': '*'}, relevant_if='False', dataset='_default', value=kwargs.get('enabled', True), description='Whether to create synthetics in compute/fitting run')]

    # TODO: the kwargs need to match the qualifier names!
    # TODO: include MORE meshing options
    params += [BoolParameter(qualifier='store_mesh', value=kwargs.get('store_mesh', False), description='Store meshes computed from phoebe1')]
    params += [ChoiceParameter(copy_for = {'method': ['star'], 'component': '*'}, component='_default', qualifier='atm', value=kwargs.get('atm', 'kurucz'), choices=['kurucz', 'blackbody'], description='Atmosphere table')]
    params += [ChoiceParameter(qualifier='morphology', value=kwargs.get('morphology','Detached binary'), choices=['Unconstrained binary system', 'Detached binary'], description='System type constraint')]
    params += [BoolParameter(qualifier='cindex', value=kwargs.get('cindex', False), description='Color index constraint')]
#    params += [IntParameter(relevant_if='cindex_switch:True', qualifier='cindex', value=kwargs.get('cindex', np.array([1.0])), description='Number of reflections')]
#    params += [BoolParameter(qualifier='heating', value=kwargs.get('heating', True), description='Allow irradiators to heat other components')]
    params += [IntParameter(copy_for={'method': ['star'], 'component': '*'}, component='_default', qualifier='gridsize', value=kwargs.get('gridsize', 40), limits=(10,None), description='Number of meshpoints for WD')]

#    params += [BoolParameter(qualifier='mult_refl', value=kwargs.get('mult_refl', False), description='Allow irradiated bodies to reflect light (for heating only) multiple times')]
    params += [IntParameter(qualifier='refl_num', value=kwargs.get('refl_num', 1), limits=(0,None), description='Number of reflections')]

#    params += [BoolParameter(qualifier='msc1', value=kwargs.get('msc1', False), description='Mainsequence Constraint for star 1')]
#    params += [BoolParameter(qualifier='msc2', value=kwargs.get('msc2', False), description='Mainsequence Constraint for star 2')]


    # TODO: can we come up with a better qualifier for reddening (and be consistent when we enable in phoebe2)
    params += [BoolParameter(qualifier='ie', value=kwargs.get('ie', False), description='Should data be de-reddened')]

    # TODO: can we change this to rv_method = ['flux_weighted', 'dynamical'] to be consistent with phoebe2?
    # TODO: can proximity_rv (rv_method) be copied for each dataset (see how this is done for phoebe2)?  This would probably mean that the wrapper would need to loop and make separate calls since PHOEBE1 can't handle different settings per-RV dataset
    params += [ChoiceParameter(qualifier='rv_method', copy_for = {'method': ['RV'], 'component': '*', 'dataset': '*'}, component='_default', dataset='_default', value=kwargs.get('rv_method', 'flux-weighted'), choices=['flux-weighted', 'dynamical'], description='Method to use for computing RVs (must be flux-weighted for Rossiter-McLaughlin)')]
#    params += [BoolParameter(copy_for={'method': ['star'], 'component': '*'}, qualifier='proximity_rv', component='_default', value=kwargs.get('proximity_rv', True), description='Rossiter effect')]


    return ParameterSet(params)

def photodynam(**kwargs):
    """
    Compute options for using Josh Carter's 'photodynam' code as a
    backend (must be installed).

    Generally, this will be used as an input to the method argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_compute`

    Please see :func:`phoebe.backend.backends.photodynam` for a list of sources to
    cite when using this backend.

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """
    params = []

    params += [BoolParameter(qualifier='enabled', copy_for={'method': ['LC', 'RV', 'ORB'], 'dataset': '*'}, relevant_if='False', dataset='_default', value=kwargs.get('enabled', True), description='Whether to create synthetics in compute/fitting run')]

    params += [FloatParameter(qualifier='stepsize', value=kwargs.get('stepsize', 0.01), default_unit=None, description='blah')]
    params += [FloatParameter(qualifier='orbiterror', value=kwargs.get('orbiterror', 1e-20), default_unit=None, description='blah')]

    # TODO: remove this option and instead use time0@system
    #params += [FloatParameter(qualifier='time0', value=kwargs.get('time0', 0.0), default_unit=u.d, description='Time to start the integration')]

    return ParameterSet(params)

def jktebop(**kwargs):
    """
    Compute options for using John Southworth's 'jktebop' code as a
    backend (must be installed).

    Generally, this will be used as an input to the method argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_compute`

    Please see :func:`phoebe.backend.backends.jktebop` for a list of sources to
    cite when using this backend.

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """

    params = []

    params += [BoolParameter(qualifier='enabled', copy_for={'method': ['LC'], 'dataset': '*'}, relevant_if='False', dataset='_default', value=kwargs.get('enabled', True), description='Whether to create synthetics in compute/fitting run')]

    params += [FloatParameter(qualifier='ringsize', value=kwargs.get('ringsize', 5), default_unit=u.deg, description='Integ Ring Size')]

    return ParameterSet(params)
