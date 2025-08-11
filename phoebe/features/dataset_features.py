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


class DatasetFeature(BaseFeature):
    """
    DatasetFeatures modify the model dataset after it is returned from the backend.
    
    """
    _phoebe_custom_feature = 'dataset'
    allowed_component_kinds = [None]
    allowed_dataset_kinds = ['lc', 'rv', 'lp']

    def __repr__(self):
        return f"<DatasetFeature: {self.__class__.__name__}>"

    @classmethod
    def get_parameters(self, **kwargs):
        raise NotImplementedError("get_parameters must be implemented in the feature subclass")

    def modify_data_for_estimators(self, b, feature_ps, data_ps, **data_arrays):
        """
        Modify the data parameters for the estimators.
        This is called before the data is passed to the estimators.
        """
        return {}

    def modify_model(self, b, feature_ps, model_ps):
        raise NotImplementedError("modify_model must be implemented in the feature subclass")


class Rv_Offset(DatasetFeature):
    allowed_component_kinds = ['star']
    allowed_dataset_kinds = ['rv']

    @classmethod
    def parse_bundle(cls, b, feature_ps):
        """
        Initialize an RVOffset feature from the bundle
        """
        rv_offsets = feature_ps.filter(qualifier='rv_offset', **_skip_filter_checks)
        return {param.component: param.get_quantity(**_skip_filter_checks) for param in rv_offsets.to_list()}

    def modify_model(self, b, feature_ps, model_ps):
        for rv_param in model_ps.filter(qualifier='rvs', kind=['rv', 'mesh'], **_skip_filter_checks).to_list():
            rv_param.set_value(rv_param.get_value() + self.kwargs.get(rv_param.component).to_value(rv_param.default_unit), ignore_readonly=True, **_skip_filter_checks)

