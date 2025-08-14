import logging
logger = logging.getLogger("FEATURES")
logger.addHandler(logging.NullHandler())


__all__ = ['BaseFeature']


class BaseFeature:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.logger = logger

    @classmethod
    def create_feature_parameters(self, feature_ps, **kwargs):
        raise NotImplementedError("create_feature_parameters must be implemented in the feature subclass")

    @classmethod
    def parse_from_feature_ps(cls, b, feature_ps, param_list):
        _skip_filter_checks = {'check_default': False,
                               'check_visible': False,
                               'check_advanced': False}

        def item_to_kwargs(item):
            if isinstance(item, str):
                item = {'qualifier': item}
            if not isinstance(item, dict):
                raise TypeError("items in param_list must be qualifiers or dictionaries")
            return item

        kws = [item_to_kwargs(item) for item in param_list]
        return {kw['qualifier']: feature_ps.get_value(**kw, **_skip_filter_checks) for kw in kws}

    @classmethod
    def parse_bundle(cls, b, feature_ps):
        return {}

    @classmethod
    def _from_bundle(cls, b, feature_ps):
        return cls(**cls.parse_bundle(b, feature_ps))

    @classmethod
    def run_checks_compute(cls, b, feature_ps, compute_ps):
        return [{}]
