

from phoebe.parameters import *
from phoebe.features.component_features import Spot
from phoebe.features.dataset_features import GPSKLearn, GPCelerite2, RVOffset

import logging
logger = logging.getLogger("FEATURE")
logger.addHandler(logging.NullHandler())

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

_feature_classes = {}


def _resolve_feature_class(feature_kind):
    # Built-in kinds are usually passed as strings (for example "rv_offset").
    if isinstance(feature_kind, str):
        return _feature_classes.get(feature_kind)

    # Custom features may be passed directly as classes.
    if hasattr(feature_kind, 'allowed_component_kinds'):
        return feature_kind

    # Backward-compatible support for callables such as feature.spot.
    for name, feature_cls in _feature_classes.items():
        if globals().get(name) is feature_kind:
            return feature_cls

    return None


def _component_allowed_for_feature(feature_kind, component_kind):
    feature_cls = _resolve_feature_class(feature_kind)
    return component_kind in getattr(feature_cls, 'allowed_component_kinds', [])


def _dataset_allowed_for_feature(feature_kind, dataset_kind):
    feature_cls = _resolve_feature_class(feature_kind)
    return dataset_kind in getattr(feature_cls, 'allowed_dataset_kinds', [])


def _register_feature(feature_cls, name):
    globals()[name] = feature_cls.create_feature_parameters
    _feature_classes[name] = feature_cls


_register_feature(Spot, 'spot')
_register_feature(RVOffset, 'rv_offset')
_register_feature(GPSKLearn, 'gp_sklearn')
_register_feature(GPCelerite2, 'gp_celerite2')