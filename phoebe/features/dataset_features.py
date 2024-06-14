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
    _phoebe_custom_feature = 'dataset'
    allowed_component_kinds = [None]
    allowed_dataset_kinds = ['lc', 'rv', 'lp']

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def from_bundle(cls, b, feature_ps):
        return cls()

    @classmethod
    def get_parameters(self, feature, **kwargs):
        raise NotImplementedError("get_parameters must be implemented in the feature subclass")

    def modify_model(self, b, feature_ps, model_ps):
        raise NotImplementedError("modify_model must be implemented in the feature subclass")