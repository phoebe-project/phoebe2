

from phoebe.parameters import *
from phoebe.parameters import constraint
from phoebe.features.component_features import Spot
from phoebe.features.dataset_features import GPSKLearn, GPCelerite2, RVOffset
from phoebe import u
from phoebe import conf

import logging
logger = logging.getLogger("FEATURE")
logger.addHandler(logging.NullHandler())

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

_allowed_components = {'pulsation': ['star', 'envelope'],
                       'gp_sklearn': [None],
                       'gp_celerite2': [None]}

_allowed_datasets = {'pulsation': [None],
                     'gp_sklearn': ['lc', 'rv', 'lp'],
                     'gp_celerite2': ['lc', 'rv', 'lp']} 

_feature_classes = {}


def _component_allowed_for_feature(feature_kind, component_kind):
    return component_kind in getattr(feature_kind, 'allowed_component_kinds', _allowed_components.get(feature_kind, []))


def _dataset_allowed_for_feature(feature_kind, dataset_kind):
    return dataset_kind in getattr(feature_kind, 'allowed_dataset_kinds', _allowed_datasets.get(feature_kind, []))


def _register_feature(feature_cls, name):
    globals()[name] = feature_cls.create_feature_parameters
    _allowed_components[name] = getattr(feature_cls, 'allowed_component_kinds', [])
    _allowed_datasets[name] = getattr(feature_cls, 'allowed_dataset_kinds', [])
    _feature_classes[name] = feature_cls


_register_feature(Spot, 'spot')
_register_feature(RVOffset, 'rv_offset')
_register_feature(GPSKLearn, 'gp_sklearn')
_register_feature(GPCelerite2, 'gp_celerite2')