import numpy as np
import astropy.units as u

import logging
logger = logging.getLogger("DATASET_FEATURES")
logger.addHandler(logging.NullHandler())

from phoebe.parameters import FloatParameter, ParameterSet
from phoebe.parameters import constraint

__all__ = ['DatasetFeature', 'SinusoidalThirdLight']

_skip_filter_checks = {'check_default': False, 'check_visible': False}

class DatasetFeature(object):
    """
    
    """
    allowed_component_kinds = [None]
    allowed_dataset_kinds = ['lc', 'rv', 'lp']
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_bundle(cls, b, feature_ps):
        return cls()

    @classmethod
    def add_feature(self, feature, **kwargs):
        raise NotImplementedError("add_feature must be implemented in the feature subclass")

    def modify_model(self, b, feature_ps, model_ps):
        raise NotImplementedError("modify_model must be implemented in the feature subclass")



class SinusoidalThirdLight(DatasetFeature):
    allowed_dataset_kinds = ['lc']

    @classmethod
    def add_feature(self, feature, **kwargs):
        params = []
        params += [FloatParameter(qualifier='amplitude', latexfmt=r'A_\mathrm{{ {feature} }}', value=kwargs.get('amplitude', 1.0), default_unit=u.W/u.m**2, description='Amplitude of the third light sinusoidal contribution')]
        params += [FloatParameter(qualifier='period', latexfmt=r'P_\mathrm{{ {feature} }}', value=kwargs.get('period', 1.0), default_unit=u.d, description='Period of the third light sinusoidal contribution')]
        params += [FloatParameter(qualifier='freq', latexfmt=r'f_\mathrm{{ {feature} }}', value=kwargs.get('freq', 2*np.pi/3.0), default_unit=u.rad/u.d, advanced=True, description='Orbital frequency (sidereal)')]

        constraints = [(constraint.freq, feature, 'feature')]

        return ParameterSet(params), constraints

    def modify_model(self, b, feature_ps, model_ps):
        period = feature_ps.get_value(qualifier='period', context='system', unit=u.d, **_skip_filter_checks)
        t0 = b.get_value(qualifier='t0', context='system', unit=u.d, **_skip_filter_checks)
        for flux_param in model_ps.filter(qualifier='fluxes', **_skip_filter_checks).tolist():
            ampl = feature_ps.get_value(qualifier='amplitude', unit=flux_param.unit, **_skip_filter_checks)
            times = model_ps.get_value(qualifier='times', dataset=flux_param.dataset, unit=u.d, **_skip_filter_checks)
            flux_param.set_value(flux_param.get_value() + ampl * np.sin(2 * np.pi * (times - t0)) / period, ignore_readonly=True, **_skip_filter_checks)

