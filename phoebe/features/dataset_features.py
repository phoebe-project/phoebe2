import numpy as np
import astropy.units as u

import logging
logger = logging.getLogger("DATASET_FEATURES")
logger.addHandler(logging.NullHandler())

from phoebe.parameters import FloatParameter, ParameterSet
from phoebe.parameters import constraint
import phoebe.parameters.feature as _parameters_feature
from phoebe.features.common import BaseFeature

__all__ = ['DatasetFeature']

_skip_filter_checks = {'check_default': False, 'check_visible': False}

def _register_feature(feature_cls, kind=None):
    if kind is None:
        kind = feature_cls.__name__.lower()

    _parameters_feature._register(feature_cls, kind)
    globals()[kind.title()] = feature_cls
    __all__.append(kind.title())

class DatasetFeature(BaseFeature):
    """
    
    """
    _phoebe_custom_feature = 'dataset'
    allowed_component_kinds = [None]
    allowed_dataset_kinds = ['lc', 'rv', 'lp']

    def __repr__(self):
        return f"<DatasetFeature: {self.__class__.__name__}>"

    @classmethod
    def get_parameters(self, feature, **kwargs):
        raise NotImplementedError("get_parameters must be implemented in the feature subclass")

    def modify_model(self, b, feature_ps, model_ps):
        raise NotImplementedError("modify_model must be implemented in the feature subclass")