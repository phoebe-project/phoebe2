"""
Test suite for compute_l3s warnings and errors.

This test suite covers:
- Error when pblum_mode='dataset-scaled' and model is not provided
- Warning when both model and use_pbfluxes are provided
- Correct behavior when model is provided with dataset-scaled mode
- Correct behavior for non-dataset-scaled modes without model
"""

import phoebe
import numpy as np
import pytest
import logging


# Helper function to create mock light curve data
def create_mock_lc_data(n_points=21, noise_level=0.01):
    """Create mock light curve data for testing."""
    np.random.seed(42)
    times = np.linspace(0, 1, n_points)
    fluxes = 1.0 + 0.1 * np.sin(2 * np.pi * times)
    fluxes += np.random.normal(0, noise_level, n_points)
    sigmas = np.ones_like(fluxes) * noise_level
    return times, fluxes, sigmas


def test_compute_l3s_dataset_scaled_no_model_raises():
    """
    Test that compute_l3s raises ValueError when pblum_mode='dataset-scaled'
    and model is not provided.
    """
    b = phoebe.Bundle.default_binary()

    times, fluxes, sigmas = create_mock_lc_data()

    # Add dataset with dataset-scaled mode and l3_mode='fraction'
    b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas,
                  dataset='lc01', passband='Johnson:V',
                  pblum_mode='dataset-scaled', l3_mode='fraction')

    b.set_value('l3_frac', 0.1, dataset='lc01')

    # Should raise ValueError when calling compute_l3s without model
    with pytest.raises(ValueError) as excinfo:
        b.compute_l3s()

    assert "pblum_mode='dataset-scaled'" in str(excinfo.value)
    assert "model" in str(excinfo.value)


def test_compute_l3s_dataset_scaled_with_model_succeeds():
    """
    Test that compute_l3s works correctly when pblum_mode='dataset-scaled'
    and model is provided.
    """
    b = phoebe.Bundle.default_binary()

    times, fluxes, sigmas = create_mock_lc_data()

    # Add dataset with dataset-scaled mode and l3_mode='fraction'
    b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas,
                  dataset='lc01', passband='Johnson:V',
                  pblum_mode='dataset-scaled', l3_mode='fraction')

    b.set_value(qualifier='l3_frac', value=0.1, dataset='lc01')

    # Run compute first to generate the model with flux_scale
    b.run_compute(irrad_method='none', model='mymodel')

    # Now compute_l3s should work with model provided
    l3s = b.compute_l3s(model='mymodel')

    # Verify we got a result
    assert 'l3@lc01' in l3s
    # The l3 value should be a reasonable flux (not astronomically large)
    # compute_l3s returns Quantity objects for l3
    l3_val = l3s['l3@lc01'].value if hasattr(l3s['l3@lc01'], 'value') else l3s['l3@lc01']
    assert l3_val < 1e10  # sanity check


def test_compute_l3s_non_dataset_scaled_no_model_succeeds():
    """
    Test that compute_l3s works without model for non-dataset-scaled modes.
    """
    b = phoebe.Bundle.default_binary()

    times, fluxes, sigmas = create_mock_lc_data()

    # Add dataset with component-coupled mode (not dataset-scaled)
    b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas,
                  dataset='lc01', passband='Johnson:V',
                  pblum_mode='component-coupled', l3_mode='fraction')

    b.set_value(qualifier='l3_frac', value=0.1, dataset='lc01')

    # Should work without model since not dataset-scaled
    l3s = b.compute_l3s()

    assert 'l3@lc01' in l3s


def test_compute_l3s_model_and_use_pbfluxes_warning(caplog):
    """
    Test that compute_l3s emits a warning when both model and use_pbfluxes
    are provided for the same dataset.
    """
    b = phoebe.Bundle.default_binary()

    times, fluxes, sigmas = create_mock_lc_data()

    # Add dataset with component-coupled mode
    b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas,
                  dataset='lc01', passband='Johnson:V',
                  pblum_mode='component-coupled', l3_mode='fraction')

    b.set_value('l3_frac', 0.1, dataset='lc01')

    # Run compute to generate a model
    b.run_compute(irrad_method='none', model='mymodel')

    # Call compute_l3s with both model and use_pbfluxes
    with caplog.at_level(logging.WARNING):
        _ = b.compute_l3s(model='mymodel', use_pbfluxes={'lc01': 0.1})

    # Check that warning was emitted
    assert any('both model and use_pbfluxes' in record.message for record in caplog.records)


def test_compute_l3s_l3_frac_to_l3_conversion():
    """
    Test that the l3_frac to l3 conversion is mathematically correct.

    The formula should satisfy: l3_frac = l3 / (l3 + pbflux)
    Rearranging: l3 = l3_frac / (1 - l3_frac) * pbflux
    """
    b = phoebe.Bundle.default_binary()

    times, fluxes, sigmas = create_mock_lc_data()

    # Use component-coupled mode for simpler testing (no model scaling needed)
    # Use unique dataset name to avoid any test interference
    b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas,
                  dataset='lc_frac_to_l3', passband='Johnson:V',
                  pblum_mode='component-coupled', l3_mode='fraction')

    l3_frac_input = 0.15
    b.set_value('l3_frac', l3_frac_input, dataset='lc_frac_to_l3')

    # Get l3 from compute_l3s
    l3s = b.compute_l3s()
    l3_val = l3s['l3@lc_frac_to_l3']
    l3_computed = l3_val.value if hasattr(l3_val, 'value') else l3_val

    # Get pbflux from compute_pblums
    pblums = b.compute_pblums(pbflux=True)
    pbflux = pblums['pbflux@lc_frac_to_l3'].value

    # Verify the formula: l3 = l3_frac / (1 - l3_frac) * pbflux
    l3_expected = l3_frac_input / (1 - l3_frac_input) * pbflux

    assert np.isclose(l3_computed, l3_expected, rtol=1e-6)


def test_compute_l3s_l3_to_l3_frac_conversion():
    """
    Test that the l3 to l3_frac conversion is mathematically correct.

    The formula should satisfy: l3_frac = l3 / (l3 + pbflux)
    """
    b = phoebe.Bundle.default_binary()

    times, fluxes, sigmas = create_mock_lc_data()

    # Use component-coupled mode so pbflux doesn't depend on model scaling
    # Use unique dataset name to avoid any test interference
    b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas,
                  dataset='lc_l3_to_frac', passband='Johnson:V',
                  pblum_mode='component-coupled', l3_mode='flux')

    # Get pbflux first
    pblums = b.compute_pblums(pbflux=True)
    pbflux = pblums['pbflux@lc_l3_to_frac'].value

    # Set l3 to a specific fraction (10%) of pbflux
    desired_frac = 0.10
    # l3_frac = l3 / (l3 + pbflux), so l3 = l3_frac / (1 - l3_frac) * pbflux
    l3_input = desired_frac / (1 - desired_frac) * pbflux
    b.set_value('l3', l3_input, dataset='lc_l3_to_frac')

    # Get l3_frac from compute_l3s
    l3s = b.compute_l3s()
    l3_frac_computed = l3s['l3_frac@lc_l3_to_frac']

    # Should match our desired fraction
    assert np.isclose(l3_frac_computed, desired_frac, rtol=1e-6)


def test_compute_l3s_multiple_datasets_mixed_modes():
    """
    Test compute_l3s with multiple datasets where some are dataset-scaled
    and others are not.
    """
    b = phoebe.Bundle.default_binary()

    times, fluxes, sigmas = create_mock_lc_data()

    # Add first dataset with dataset-scaled mode
    b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas,
                  dataset='lc_scaled', passband='Johnson:V',
                  pblum_mode='dataset-scaled', l3_mode='fraction')

    # Add second dataset with component-coupled mode
    b.add_dataset('lc', times=times, fluxes=fluxes*1.1, sigmas=sigmas,
                  dataset='lc_coupled', passband='Johnson:B',
                  pblum_mode='component-coupled', l3_mode='fraction')

    b.set_value('l3_frac', 0.1, dataset='lc_scaled')
    b.set_value('l3_frac', 0.05, dataset='lc_coupled')

    # Without model, should raise error because of the dataset-scaled dataset
    with pytest.raises(ValueError) as excinfo:
        b.compute_l3s()

    assert 'lc_scaled' in str(excinfo.value)

    # Run compute
    b.run_compute(irrad_method='none', model='mymodel')

    # With model, should work for all datasets
    l3s = b.compute_l3s(model='mymodel')

    assert 'l3@lc_scaled' in l3s
    assert 'l3@lc_coupled' in l3s


def test_compute_l3s_specific_dataset_only():
    """
    Test compute_l3s with dataset parameter to compute only specific datasets.
    """
    b = phoebe.Bundle.default_binary()

    times, fluxes, sigmas = create_mock_lc_data()

    # Add dataset-scaled dataset
    b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas,
                  dataset='lc_scaled', passband='Johnson:V',
                  pblum_mode='dataset-scaled', l3_mode='fraction')

    # Add component-coupled dataset
    b.add_dataset('lc', times=times, fluxes=fluxes*1.1, sigmas=sigmas,
                  dataset='lc_coupled', passband='Johnson:B',
                  pblum_mode='component-coupled', l3_mode='fraction')

    b.set_value('l3_frac', 0.1, dataset='lc_scaled')
    b.set_value('l3_frac', 0.05, dataset='lc_coupled')

    # Request only the component-coupled dataset - should work without model
    l3s = b.compute_l3s(dataset='lc_coupled')

    assert 'l3@lc_coupled' in l3s
    assert 'l3@lc_scaled' not in l3s


if __name__ == '__main__':
    # Run tests with verbose output
    logger = phoebe.logger(clevel='WARNING')

    print("Testing compute_l3s dataset-scaled no model raises...")
    test_compute_l3s_dataset_scaled_no_model_raises()
    print("PASSED")

    print("Testing compute_l3s dataset-scaled with model succeeds...")
    test_compute_l3s_dataset_scaled_with_model_succeeds()
    print("PASSED")

    print("Testing compute_l3s non-dataset-scaled no model succeeds...")
    test_compute_l3s_non_dataset_scaled_no_model_succeeds()
    print("PASSED")

    print("Testing compute_l3s l3_frac to l3 conversion...")
    test_compute_l3s_l3_frac_to_l3_conversion()
    print("PASSED")

    print("Testing compute_l3s l3 to l3_frac conversion...")
    test_compute_l3s_l3_to_l3_frac_conversion()
    print("PASSED")

    print("Testing compute_l3s multiple datasets mixed modes...")
    test_compute_l3s_multiple_datasets_mixed_modes()
    print("PASSED")

    print("Testing compute_l3s specific dataset only...")
    test_compute_l3s_specific_dataset_only()
    print("PASSED")

    print("\nAll tests passed!")
