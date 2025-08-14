import numpy as np
import astropy.units as u

import logging
logger = logging.getLogger("DATASET_FEATURES")
logger.addHandler(logging.NullHandler())

from phoebe.parameters import FloatParameter, ChoiceParameter, StringParameter, ParameterSet
from phoebe.features.common import BaseFeature

__all__ = ['DatasetFeature']

_skip_filter_checks = {'check_default': False, 'check_visible': False}


class DatasetFeature(BaseFeature):
    """
    DatasetFeatures modify the model dataset after it is returned from the backend.
    
    """
    allowed_component_kinds = [None]
    allowed_dataset_kinds = ['lc', 'rv', 'lp']

    def __repr__(self):
        return f"<DatasetFeature: {self.__class__.__name__}>"

    @classmethod
    def get_parameters(self, **kwargs):
        raise NotImplementedError("get_parameters must be implemented in the feature subclass")

    def modify_data_for_estimators(self, b, data_ps, **data_arrays):
        """
        Modify the data parameters for the estimators.
        This is called before the data is passed to the estimators.
        """
        return {}

    def modify_model(self, b, model_ps):
        raise NotImplementedError("modify_model must be implemented in the feature subclass")

class GPSKLearn(DatasetFeature):
    allowed_dataset_kinds = ['lc']

    @classmethod
    def create_feature_parameters(self, feature, **kwargs):
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

        return ParameterSet(params), []

    def modify_model(self, b, model_ps):
        # GPS are handled separately and all simultaneously
        return


class GPCelerite2(DatasetFeature):
    @classmethod
    def create_feature_parameters(self, feature, **kwargs):
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

        return ParameterSet(params), []

    def modify_model(self, b, model_ps):
        # GPS are handled separately and all simultaneously
        return


class RVOffset(DatasetFeature):
    allowed_component_kinds = ['star']
    allowed_dataset_kinds = ['rv']

    @classmethod
    def create_feature_parameters(self, feature, **kwargs):
        """
        Create a <phoebe.parameters.ParameterSet> for an rvoffset feature.

        Generally, this will be used as an input to the kind argument in
        <phoebe.frontend.bundle.Bundle.add_feature>.  If attaching through
        <phoebe.frontend.bundle.Bundle.add_feature>, all `**kwargs` will be
        passed on to set the values as described in the arguments below.  Alternatively,
        see <phoebe.parameters.ParameterSet.set_value> to set/change the values
        after creating the Parameters.

        Allowed to attach to:
        * datasets: rv
        """
        params = []
        params += [FloatParameter(qualifier='rv_offset', copy_for={'kind': ['star'], 'component': '*'}, component='_default', value=kwargs.get('rv_offset', 0.0), default_unit=u.km/u.s, description='Per-component offset to add to synthetic RVs (i.e. for hot stars)')]

        return ParameterSet(params), []

    @classmethod
    def parse_bundle(cls, b, feature_ps):
        """
        Initialize an RVOffset feature from the bundle
        """
        rv_offsets = feature_ps.filter(qualifier='rv_offset', **_skip_filter_checks)
        return {param.component: param.get_quantity(**_skip_filter_checks) for param in rv_offsets.to_list()}

    def modify_model(self, b, model_ps):
        for rv_param in model_ps.filter(qualifier='rvs', kind=['rv', 'mesh'], **_skip_filter_checks).to_list():
            rv_param.set_value(rv_param.get_value() + self.kwargs.get(rv_param.component).to_value(rv_param.default_unit), ignore_readonly=True, **_skip_filter_checks)
