import numpy as np
import pytest

import phoebe
from phoebe.features.gaussian_processes import _use_celerite2, _use_sklearn


def _build_minimal_bundle_with_observations():
    """Build a small LC bundle with synthetic observations and a smooth residual trend."""
    b = phoebe.default_binary()
    b.add_dataset('lc', compute_times=phoebe.linspace(0, 1, 41), passband='Johnson:V')

    b.add_compute(compute='fast_compute')
    b.set_value_all('ld_mode', value='manual')
    b.set_value('irrad_method', compute='fast_compute', value='none')
    b.set_value_all('distortion_method', compute='fast_compute', value='sphere')
    b.set_value_all('atm', value='ck2004')

    b.run_compute(model='baseline', compute='fast_compute')

    times = b.get_value(qualifier='times', context='model', model='baseline', dataset='lc01')
    fluxes = b.get_value(qualifier='fluxes', context='model', model='baseline', dataset='lc01')

    # Add a deterministic correlated trend so GP contribution is non-trivial.
    observed_fluxes = fluxes + 0.02 * np.sin(6.0 * np.pi * times)
    sigmas = np.full_like(observed_fluxes, 0.01)

    b.add_dataset('lc', dataset='lc01', times=times, fluxes=observed_fluxes,
                  sigmas=sigmas, passband='Johnson:V', overwrite=True)

    b.run_compute(model='without_gps', compute='fast_compute')
    return b


@pytest.mark.parametrize(
    "gp_kind,kernel,kernel_kwargs,is_available",
    [
        ('sklearn', 'rbf', {'length_scale': 0.1}, _use_sklearn),
        ('celerite2', 'sho', {'rho': 1.0, 'tau': 1.0, 'sigma': 0.2}, _use_celerite2),
    ],
)
def test_minimal_gp_forward_model_decomposition(gp_kind, kernel, kernel_kwargs, is_available):
    """
    Lightweight coverage adapted from docs/development/examples/minimal_GPs.py.

    Ensures the GP post-processing path runs and that exported model arrays satisfy
    fluxes = fluxes_nogps + gps for each supported GP backend.
    """
    if not is_available:
        pytest.skip(f"{gp_kind} backend not available in this environment")

    b = _build_minimal_bundle_with_observations()

    b.add_gaussian_process(gp_kind, dataset='lc01', kernel=kernel, **kernel_kwargs)
    b.flip_constraint('compute_phases', solve_for='compute_times')
    b.set_value('compute_phases', phoebe.linspace(0, 1, 61))
    b.run_compute(model='with_gps', compute='fast_compute')

    fluxes = b.get_value(qualifier='fluxes', context='model', model='with_gps', dataset='lc01')
    fluxes_nogps = b.get_value(qualifier='fluxes_nogps', context='model', model='with_gps', dataset='lc01')
    gps = b.get_value(qualifier='gps', context='model', model='with_gps', dataset='lc01')

    assert len(fluxes) == len(fluxes_nogps) == len(gps)
    assert np.all(np.isfinite(gps))
    np.testing.assert_allclose(fluxes, fluxes_nogps + gps)
    assert np.max(np.abs(gps)) > 0.0
