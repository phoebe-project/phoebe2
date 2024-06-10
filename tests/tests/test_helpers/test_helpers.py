import phoebe
import pytest
from numpy.testing import assert_allclose


def test_integrate_flux_from_mesh():
    b = phoebe.default_binary()
    b.add_dataset('lc', compute_phases=phoebe.linspace(0,1,4), dataset='lc01')
    b.add_dataset('mesh', include_times='lc01', dataset='mesh01')
    b.run_compute(model='latest')

    with pytest.raises(ValueError):
        phoebe.helpers.integrate_flux_from_mesh(b, 'latest', 'mesh01', 'lc01')

    b.set_value('columns', value=['visibilities', 'intensities@lc01', 'areas', 'ptfarea@lc01', 'mus'])
    b.run_compute(model='latest')

    times, fluxes = phoebe.helpers.integrate_flux_from_mesh(b, 'latest', 'mesh01', 'lc01')
    assert len(times) == 4
    assert len(fluxes) == 4

    fluxes_lc = b.get_value(qualifier='fluxes', dataset='lc01', context='model')
    assert_allclose(fluxes, fluxes_lc, atol=5e-5)

if __name__ == '__main__':
    test_integrate_flux_from_mesh()
