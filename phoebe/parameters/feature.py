

from phoebe.parameters import *
from phoebe.parameters import constraint
from phoebe import u
from phoebe import conf

import logging
logger = logging.getLogger("FEATURE")
logger.addHandler(logging.NullHandler())

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

def _component_allowed_for_feature(feature_kind, component_kind):
    _allowed = {}
    _allowed['spot'] = ['star', 'envelope']
    _allowed['pulsation'] = ['star', 'envelope']
    _allowed['gp_sklearn'] = [None]
    _allowed['gp_celerite2'] = [None]
    _allowed['gaussian_process'] = [None]  # deprecated: remove in 2.5

    return component_kind in _allowed.get(feature_kind, [None])

def _dataset_allowed_for_feature(feature_kind, dataset_kind):
    _allowed = {}
    _allowed['spot'] = [None]
    _allowed['pulsation'] = [None]
    _allowed['gp_sklearn'] = ['lc', 'rv', 'lp']
    _allowed['gp_celerite2'] = ['lc', 'rv', 'lp']
    _allowed['gaussian_process'] = ['lc', 'rv', 'lp']  # deprecated: remove in 2.5

    return dataset_kind in _allowed.get(feature_kind, [None])

def _solver_allowed_for_feature(feature_kind, solver_kind):
    _allowed = {}
    _allowed['emcee_move'] = ['emcee']

    return solver_kind in _allowed.get(feature_kind, [None])


def spot(feature, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for a spot feature.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_feature>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_feature>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Allowed to attach to:
    * components with kind: star
    * datasets: not allowed
    * solver: not allowed

    Arguments
    ----------
    * `colat` (float/quantity, optional): colatitude of the center of the spot
        wrt spin axis.
    * `long` (float/quantity, optional): longitude of the center of the spot wrt
        spin axis.
    * `radius` (float/quantity, optional): angular radius of the spot.
    * `relteff` (float/quantity, optional): temperature of the spot relative
        to the intrinsic temperature.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>, list): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params = []

    params += [FloatParameter(qualifier="colat", value=kwargs.get('colat', 0.0), default_unit=u.deg, description='Colatitude of the center of the spot wrt spin axis')]
    params += [FloatParameter(qualifier="long", value=kwargs.get('long', 0.0), default_unit=u.deg, description='Longitude of the center of the spot wrt spin axis')]
    params += [FloatParameter(qualifier='radius', value=kwargs.get('radius', 1.0), default_unit=u.deg, description='Angular radius of the spot')]
    # params += [FloatParameter(qualifier='area', value=kwargs.get('area', 1.0), default_unit=u.solRad, description='Surface area of the spot')]

    params += [FloatParameter(qualifier='relteff', value=kwargs.get('relteff', 1.0), limits=(0.,None), default_unit=u.dimensionless_unscaled, description='Temperature of the spot relative to the intrinsic temperature')]
    # params += [FloatParameter(qualifier='teff', value=kwargs.get('teff', 10000), default_unit=u.K, description='Temperature of the spot')]

    constraints = []

    return ParameterSet(params), constraints

def gp_sklearn(feature, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for a gp_sklearn feature.

    Requires scikit-learn to be installed.  See https://scikit-learn.org/stable/modules/gaussian_process.html/.
    If using gp_sklearn, consider citing:
    * https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html

    See also:
    * <phoebe.frontend.bundle.Bundle.references>

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_feature>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_feature>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Allowed to attach to:
    * components: not allowed
    * datasets with kind: lc

    If `compute_times` or `compute_phases` is used: the underlying model without
    gaussian_processes will be computed at the given times/phases but will then
    be interpolated into the times of the underlying dataset to include the
    contribution of gaussian processes and will be exposed at the dataset
    times (with a warning in the logger and in
    <phoebe.frontend.bundle.Bundle.run_checks_compute>).  If the system is
    time-dependent without GPs
    (see <phoebe.parameters.HierarchyParameter.is_time_dependent>), then
    the underlying model will need to cover the entire dataset or an error
    will be raised by <phoebe.frontend.bundle.Bundle.run_checks_compute>.


    Arguments
    ----------
    * `kernel` (string, optional, default='white'): Kernel for the gaussian
        process (see https://scikit-learn.org/stable/modules/gaussian_process.html#kernels-for-gaussian-processes)
    * `constant_value` (float, optional, default=1.0): only applicable if `kernel` is
        'constant'.
    * `noise_level` (float, optional, default=1.0): only applicable if `kernel` is 'white'.
    * `length_scale` (float, optional, default=1.0): only applicable if `kernel` is 'rbf', 'rational_quadratic',
        'exp_sine_squared' or 'matern'.
    * `nu` (float, optional, default=1.5): only applicable if `kernel` is 'matern'.
    * `alpha` (float, optional, default=1.0): only applicable if `kernel` is 'rational_quadratic'.
    * `periodicity` (float, optional, default=1.0): only applicable if `kernel` is 'exp_sine_sqaured'.
    * `sigma_0` (float, optional, default=1.0): only applicable if `kernel` is 'sigma_0'.
    * `alg_operation` (string, default='sum'): algebraic operation for the kernel with previously added ones.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>, list): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """
    params = []

    params += [ChoiceParameter(qualifier='kernel', value=kwargs.get('kernel', 'white'), choices=['constant', 'white', 'rbf', 'matern', 'rational_quadratic', 'exp_sine_squared', 'dot_product'], description='Kernel for the gaussian process (see https://scikit-learn.org/stable/modules/gaussian_process.html)')]

    # sklearn kernel parameters
    params += [FloatParameter(visible_if='kernel:constant', qualifier='constant_value', value=kwargs.get('constant_value', 1.0), default_unit=u.dimensionless_unscaled, description='Value of the constant kernel')]
    params += [FloatParameter(visible_if='kernel:white', qualifier='noise_level', value=kwargs.get('noise_level', 1.0), default_unit=u.dimensionless_unscaled, description='Noise level of the white kernel')]
    params += [FloatParameter(visible_if='kernel:rbf|rational_quadratic|exp_sine_squared|matern', qualifier='length_scale', value=kwargs.get('length_scale', 1.0), default_unit=u.dimensionless_unscaled, description='Length scale of the kernel')]
    params += [FloatParameter(visible_if='kernel:matern', qualifier='nu', value=kwargs.get('nu', 1.5), default_unit=u.dimensionless_unscaled, description='Smoothness factor of the Matern kernel')]
    params += [FloatParameter(visible_if='kernel:rational_quadratic', qualifier='alpha', value=kwargs.get('alpha', 1.0), default_unit=u.dimensionless_unscaled, description='Scale mixture parameter of the RationalQuadratic kernel')]
    params += [FloatParameter(visible_if='kernel:exp_sine_squared', qualifier='periodicity', value=kwargs.get('periodicity', 1.0), default_unit=u.dimensionless_unscaled, description='Periodicity parameter of the ExpSineSquared kernel')]
    params += [FloatParameter(visible_if='kernel:dot_product', qualifier='sigma_0', value=kwargs.get('sigma_0', 1.0), default_unit=u.dimensionless_unscaled, description='Constant factor of the DotProduct kernel')]

    params += [StringParameter(visible_if='kernel:constant', qualifier='constant_value_bounds', value='fixed', default_unit=u.dimensionless_unscaled, description='Value bounds of the constant kernel')]
    params += [StringParameter(visible_if='kernel:white', qualifier='noise_level_bounds', value='fixed', default_unit=u.dimensionless_unscaled, description='Noise level bounds of the white kernel')]
    params += [StringParameter(visible_if='kernel:rbf|rational_quadratic|exp_sine_squared|matern', qualifier='length_scale_bounds', value='fixed', default_unit=u.dimensionless_unscaled, description='Length scale bounds of the kernel')]
    params += [StringParameter(visible_if='kernel:matern', qualifier='nu_bounds', value='fixed', default_unit=u.dimensionless_unscaled, description='Smoothness factor bounds of the Matern kernel')]
    params += [StringParameter(visible_if='kernel:rational_quadratic', qualifier='alpha_bounds', value='fixed', default_unit=u.dimensionless_unscaled, description='Scale mixture parameter bounds of the RationalQuadratic kernel')]
    params += [StringParameter(visible_if='kernel:exp_sine_squared', qualifier='periodicity_bounds', value='fixed', default_unit=u.dimensionless_unscaled, description='Periodicity parameter bounds of the ExpSineSquared kernel')]
    params += [StringParameter(visible_if='kernel:dot_product', qualifier='sigma_0_bounds', value='fixed', default_unit=u.dimensionless_unscaled, description='Constant factor bounds of the DotProduct kernel')]

    # additional parameters for GPs
    params += [ChoiceParameter(qualifier='alg_operation', value='sum', choices=['sum', 'product'], default_unit=u.dimensionless_unscaled, description='Algebraic operation of this kernel with previous ones. Can be one of [sum, product]')]
    # params += [FloatArrayParameter(qualifier='exclude_phase_ranges', value=kwargs.get('exclude_phase_ranges', []), required_shape=[None, 2], default_unit=u.dimensionless_unscaled, description='Phase ranges to exclude from fitting the GP model (typically correspond to ingress and egress of eclipses).')]
    constraints = []

    return ParameterSet(params), constraints

def gp_celerite2(feature, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for a gp_celerite2 feature.

    Requires celerite2 to be installed.  See https://celerite2.readthedocs.io/en/stable/.
    If using gaussian processes, consider citing:
    * https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F

    See also:
    * <phoebe.frontend.bundle.Bundle.references>

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_feature>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_feature>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Allowed to attach to:
    * components: not allowed
    * datasets with kind: lc
    * solvers: not allowed

    If `compute_times` or `compute_phases` is used: the underlying model without
    gaussian_processes will be computed at the given times/phases but will then
    be interpolated into the times of the underlying dataset to include the
    contribution of gaussian processes and will be exposed at the dataset
    times (with a warning in the logger and in
    <phoebe.frontend.bundle.Bundle.run_checks_compute>).  If the system is
    time-dependent without GPs
    (see <phoebe.parameters.HierarchyParameter.is_time_dependent>), then
    the underlying model will need to cover the entire dataset or an error
    will be raised by <phoebe.frontend.bundle.Bundle.run_checks_compute>.


    Arguments
    ----------
    * `kernel` (string, optional, default='sho'): Kernel for the gaussian
        process (see https://celerite2.readthedocs.io/en/stable/api/python/#celerite2.terms)
    * `rho` (float, optional, default=1.0): only applicable if `kernel` is
        'sho' or 'matern32'.
    * `tau` (float, optional, default=1.0): only applicable if `kernel` is
        'sho'.
    * `sigma` (float, optional, default=1.0)
    * `period` (float, optional, default=1.0): only applicable if `kernel` is
        'rotation'.
    * `Q0` (float, optional, default=1.0): only applicable if `kernel` is
        'rotation'.
    * `dQ` (float, optional, default=1.0): only applicable if `kernel` is
        'rotation'.
    * `f` (float, optional, default=1.0): only applicable if `kernel` is
        'rotation'.
    * `eps` (float, optional, default=1e-5): only applicable if `kernel` is
        'sho' or 'matern32'.
    * `alg_operation` (string, default='sum'): algebraic operation for the kernel with previously added ones.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>, list): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params = []

    params += [ChoiceParameter(qualifier='kernel', value=kwargs.get('kernel', 'sho'), choices=['sho', 'rotation', 'matern32'], description='Kernel for the gaussian process')]


    # celerite2 kernel parameters
    params += [FloatParameter(visible_if='kernel:sho|matern32', qualifier='rho', value=kwargs.get('rho', 1.0), default_unit = u.dimensionless_unscaled, description='Periodicity of the SHO kernel.')]
    params += [FloatParameter(visible_if='kernel:sho', qualifier='tau', value=kwargs.get('tau', 1.0), default_unit = u.dimensionless_unscaled, description='Damping timescale of the SHO kernel.')]
    params += [FloatParameter(visible_if='kernel:sho|rotation|matern32', qualifier='sigma', value=kwargs.get('sigma', 1.0), default_unit = u.dimensionless_unscaled, description='Standard deviation of the process.')]
    params += [FloatParameter(visible_if='kernel:rotation', qualifier='period', value=kwargs.get('period', 1.0), default_unit = u.dimensionless_unscaled, description='The primary period of variability of the rotation kernel.')]
    params += [FloatParameter(visible_if='kernel:rotation', qualifier='Q0', value=kwargs.get('Q0', 1.0), default_unit = u.dimensionless_unscaled, description='The quality factor for the secondary oscillation.')]
    params += [FloatParameter(visible_if='kernel:rotation', qualifier='dQ', value=kwargs.get('dQ', 1.0), default_unit = u.dimensionless_unscaled, description='The difference between the quality factors of the first and the second modes.')]
    params += [FloatParameter(visible_if='kernel:rotation', qualifier='f', value=kwargs.get('f', 1.0), default_unit = u.dimensionless_unscaled, description='The fractional amplitude of the secondary mode compared to the primary.')]
    params += [FloatParameter(visible_if='kernel:sho|matern32', qualifier='eps', value=kwargs.get('eps', 1e-5), default_unit = u.dimensionless_unscaled, description='A regularization parameter used for numerical stability.')]

    # additional parameters for GPs
    params += [ChoiceParameter(qualifier='alg_operation', value='sum', choices=['sum', 'product'], default_unit=u.dimensionless_unscaled, description='Algebraic operation of this kernel with previous ones. Can be one of [sum, product]')]
    # params += [FloatArrayParameter(qualifier='exclude_phase_ranges', value=kwargs.get('exclude_phase_ranges', []), required_shape=[None, 2], default_unit=u.dimensionless_unscaled, description='Phase ranges to exclude from fitting the GP model (typically correspond to ingress and egress of eclipses).')]
    constraints = []

    return ParameterSet(params), constraints

def emcee_move(feature, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for an emcee_move feature to attach
    to an <phoebe.parameters.solver.sampler.emcee> solver.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_feature>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_feature>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Allowed to attach to:
    * components: not allowed
    * datasets: not allowed
    * solvers with kind: emcee


    Arguments
    ----------
    * `move` (string, optional, default='Stretch'): Type of move
        (see https://emcee.readthedocs.io/en/stable/user/moves/)
    * `weight` (float, optional, default=1.0): Weighted probability to apply to
        move.  Weights across all enabled emcee_move features will be renormalized
        to sum to 1 before passing to emcee.
    * `nsplits` (int, optional, default=2):
    * `randomize_split` (bool, optional, default=True):
    * `a` (float, optional, default=2.0):
    * `smode` (string, optional, default='auto'):
    * `s` (int, optional, default=16):
    * `bw_method` (string, optional, default='scott'):
    * `bw_constant` (float, optional, default=1.0):
    * `sigma` (float, optional, default=1e-5):
    * `gamma0_mode` (string, optional, default='auto'):
    * `gamma0` (float, optional, default=0.5):
    * `gammas` (float, optional, default=1.7):


    Returns
    --------
    * (<phoebe.parameters.ParameterSet>, list): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params = []

    params += [ChoiceParameter(qualifier='move', value=kwargs.get('move', 'Stretch'), choices=['Stretch', 'Walk', 'KDE', 'DE', 'DESnooker'], description='Type of move (see https://emcee.readthedocs.io/en/stable/user/moves/)')]
    params += [FloatParameter(qualifier='weight', value=kwargs.get('weight', 1.0), limits=(0,None), default_unit=u.dimensionless_unscaled, description='Weighted probability to apply to move.  Weights across all enabled emcee_move features will be renormalized to sum to 1 before passing to emcee.')]

    # NOTE: RedBlue requires subclassing
    # params += [IntParameter(visible_if='move:RedBlue', qualifier='nsplits', value=kwargs.get('nsplits', 2), limits=(1,100), description='Passed directly to emcee. The number of sub-ensembles to use. Each sub-ensemble is updated in parallel using the other sets as the complementary ensemble. The default value is 2 and you probably won’t need to change that.')]
    # params += [BoolParameter(visible_if='move:RedBlue', qualifier='randomize_split', value=kwargs.get('randomize_split', True), description='Passed directly to emcee. Randomly shuffle walkers between sub-ensembles. The same number of walkers will be assigned to each sub-ensemble on each iteration.')]

    params += [FloatParameter(visible_if='move:Stretch', qualifier='a', value=kwargs.get('a', 2.0), limits=(None, None), default_units=u.dimensionless_unscaled, description='Passed directly to emcee.  The stretch scale parameter.')]

    params += [ChoiceParameter(visible_if='move:Walk', qualifier='smode', value=kwargs.get('smode', 'auto'), choices=['auto', 'manual'], description='Whether to manually provide the s parameter (number of helper walkers) or use all walkers in the complement by passing None to emcee.')]
    params += [IntParameter(visible_if='move:Walk,smode:manual', qualifier='s', value=kwargs.get('s', 16), limits=(1,None), description='Passed directly to emcee.  The number of helper walkers to use.')]

    params += [ChoiceParameter(visible_if='move:KDE', qualifier='bw_method', value=kwargs.get('bw_method', 'scott'), choices=['scott', 'silverman', 'constant'], description='Passed directly to emcee. The bandwidth estimation method.  See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html')]
    params += [FloatParameter(visible_if='move:KDE,bw_method:constant', qualifier='bw_constant', value=kwargs.get('bw_constant', 1.0), limits=(None, None), default_unit=u.dimensionless_unscaled, description='Bandwidth estimation kde factor.  See https://docs.scipy.org/docs/scipy/reference/generated/scipy.stats.gaussian_kde.html')]

    params += [FloatParameter(visible_if='move:DE', qualifier='sigma', value=kwargs.get('sigma', 1e-5), limits=(0,None), default_unit=u.dimensionless_unscaled, description='Passed directly to emcee. The standard deviation of the Gaussian used to stretch the proposal vector.')]
    params += [ChoiceParameter(visible_if='move:DE', qualifier='gamma0_mode', value=kwargs.get('gamma0_mode', 'auto'), choices=['auto', 'manual'], description='Whether to manually provide gamma0 or default to 2.38/sqrt(2 * ndim)')]
    params += [FloatParameter(visible_if='move:DE,gamma0_mode:manual', qualifier='gamma0', value=kwargs.get('gamma0', 0.5), limits=(0,None), default_unit=u.dimensionless_unscaled, description='Passed directly to emcee.  The mean stretch factor for the proposal vector.')]

    params += [FloatParameter(visible_if='move:DESnooker', qualifier='gammas', value=kwargs.get('gammas', 1.7), limits=(0,None), default_unit=u.dimensionless_unscaled, description='Passed directly to emcee.  The mean stretch factor of the proposal vector.')]

    # NOTE: MH not implemented as it requires a callable
    # NOTE: Gaussian not implemented as it requires a covariance (as scalar, vector, or matrix)

    constraints = []

    return ParameterSet(params), constraints



# del deepcopy
# del _component_allowed_for_feature
# del download_passband, list_installed_passbands, list_online_passbands, list_passbands, parameter_from_json, parse_json, send_if_client, update_if_client
# del fnmatch
