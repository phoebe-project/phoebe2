import phoebe
import pytest
from numpy.testing import assert_allclose


def test_integrate_flux_from_mesh():
    b = phoebe.default_binary()
    b.add_dataset('lc', compute_phases=phoebe.linspace(0,1,4), dataset='lc01')
    b.add_dataset('mesh', include_times='lc01', dataset='mesh01')
    b.run_compute(model='latest')

    with pytest.raises(ValueError):
        phoebe.helpers.fluxes_from_mesh_model(b, 'latest', 'mesh01', 'lc01')

    b.set_value('columns', value=['visibilities', 'intensities@lc01', 'areas', 'ptfarea@lc01', 'mus'])
    b.run_compute(model='latest')

    times, fluxes = phoebe.helpers.fluxes_from_mesh_model(b, 'latest', 'mesh01', 'lc01')
    assert len(times) == 4
    assert len(fluxes) == 4

    fluxes_ml = b.get_value(qualifier='fluxes', dataset='lc01', context='model')
    assert_allclose(fluxes, fluxes_ml, atol=5e-5)

def test_integrate_rv_from_mesh():
    b = phoebe.default_binary()
    b.set_value(qualifier='q', value=0.95)
    b.add_dataset('rv', times=phoebe.linspace(0,1,4), dataset='rv01')
    b.add_dataset('mesh', include_times='rv01', dataset='mesh01')
    b.run_compute(model='latest')

    with pytest.raises(ValueError):
        phoebe.helpers.rvs_from_mesh_model(b, 'latest', 'mesh01', 'rv01', 'primary')

    b.set_value(qualifier='columns', value=['vws', 'visibilities', 'abs_intensities@rv01', 'areas', 'mus'])
    b.run_compute(model='latest')

    for comp in ('primary', 'secondary'):
        times, rvs = phoebe.helpers.rvs_from_mesh_model(b, 'latest', 'mesh01', 'rv01', comp)
        assert len(times) == 4
        assert len(rvs) == 4

        rvs_ml = b.get_value(qualifier='rvs', dataset='rv01', component=comp, context='model')
        assert_allclose(rvs, rvs_ml, atol=5e-5)

if __name__ == '__main__':
    test_integrate_flux_from_mesh()
    test_integrate_rv_from_mesh()
