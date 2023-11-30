"""
Tests line profile implementation.
"""

import phoebe
import numpy as np


def test_interp(plot=False):
    """
    Tests whether internal wavelength interpolation works.

    Parameters
    ----------
    plot : bool, optional
        should line profiles be plotted, by default False

    Returns
    -------
    <phoebe.Bundle>
        Returns a properly initialized bundle.
    """

    b = phoebe.Bundle.default_binary()

    wavelengths = phoebe.linspace(549.0, 551.0, 101)
    interp_wavelengths = phoebe.linspace(549.0, 551.0, 100)

    b.add_dataset('lp', times=[0.25,], wavelengths=wavelengths, dataset='no_interp')
    b.add_dataset('lp', times=[0.25,], wavelengths=interp_wavelengths, dataset='interp')
    b.run_compute(irrad_method='none')

    fluxes = b['value@flux_densities@no_interp@model']
    interp_fluxes = b['value@flux_densities@interp@model']

    diffs = np.abs(interp_fluxes-np.interp(interp_wavelengths, wavelengths, fluxes))
    assert (np.all(diffs < 0.004))

    if plot:
        b.plot(show=True)

    return b


def test_multi_lp():
    """
    Checks whether line profiles are computed consistently for three successive timestamps at the same phase.

    Parameters
    ----------
    plot : bool, optional
        should line profiles be plotted, by default False

    Returns
    -------
    <phoebe.Bundle>
        Returns a properly initialized bundle.
    """

    b = phoebe.Bundle.default_binary()

    b.add_dataset('lp', times=phoebe.linspace(0, 1, 2), wavelengths=phoebe.linspace(549.0, 551.0, 101))
    b.run_compute(irrad_method='none')

    assert (np.allclose(b['values@00.000000@flux_densities@model'], b['values@01.000000@flux_densities@model'], rtol=1e-6, atol=1e-6))
    assert (np.allclose(b['values@01.000000@flux_densities@model'], b['values@02.000000@flux_densities@model'], rtol=1e-6, atol=1e-6))
    return b


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')

    b = test_interp()
    b = test_multi_lp()
