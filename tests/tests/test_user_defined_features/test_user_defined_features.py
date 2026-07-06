import phoebe
from phoebe import u
from phoebe.features import ComponentFeature
from phoebe.parameters import FloatParameter, ParameterSet


class CustomComponentFeature(ComponentFeature):
    allowed_component_kinds = ['star', 'envelope', 'orbit']

    @classmethod
    def create_feature_parameters(cls, feature, **kwargs):
        params = [
            FloatParameter(
                qualifier='test_param',
                latexfmt=r'T_\\mathrm{{ {feature} }}',
                value=kwargs.get('test_param', 1),
                default_unit=u.dimensionless_unscaled,
                description='Just a test parameter!',
            )
        ]
        return ParameterSet(params), []

    @classmethod
    def parse_bundle(cls, b, feature_ps):
        return {}

    @classmethod
    def run_checks_compute(cls, b, feature_ps, compute_ps):
        return [{}]


def test_user_defined_component_feature_save_load(tmp_path):
    b = phoebe.default_binary()
    b.add_feature(
        CustomComponentFeature,
        component='primary',
        test_param=2,
        feature='my_custom_component_feature',
    )

    feature = 'my_custom_component_feature'
    custom_code_before = b.get_parameter(qualifier='custom_code', feature=feature).get_source_code()

    assert 'def create_feature_parameters' in custom_code_before
    assert "qualifier='test_param'" in custom_code_before
    assert b.get_value(qualifier='test_param', feature=feature) == 2

    bundle_path = tmp_path / 'custom_feature.bundle'
    b.save(bundle_path)

    b2 = phoebe.load(str(bundle_path))
    custom_code_after = b2.get_parameter(qualifier='custom_code', feature=feature).get_source_code()

    assert custom_code_after == custom_code_before
    assert b2.get_value(qualifier='test_param', feature=feature) == 2

    cls = b2.get_feature_code(feature=feature, instantiate=False)
    assert hasattr(cls, 'create_feature_parameters')
