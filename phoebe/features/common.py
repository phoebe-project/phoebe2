__all__ = ['BaseFeature']


class BaseFeature:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def from_bundle(cls, b, feature):
        return cls(**cls.parse_bundle(b, feature))

    @classmethod
    def parse_bundle(cls, b, feature):
        return {}

    @classmethod
    def get_parameters(self, feature, **kwargs):
        raise NotImplementedError("get_parameters must be implemented in the feature subclass")

