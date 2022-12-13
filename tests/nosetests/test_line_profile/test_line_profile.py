"""
This unit test checks whether line profiles are computed consistently for three successive timestamps at the same phase.
"""

import phoebe
import numpy as np


def test_binary():
    b = phoebe.Bundle.default_binary()

    b.add_dataset('lp', times=phoebe.linspace(0, 1, 2), wavelengths=phoebe.linspace(0, 1, 101))
    b.run_compute(irrad_method='none')

    assert(np.allclose(b['values@00.000000@flux_densities@model'], b['values@01.000000@flux_densities@model'], rtol=1e-6, atol=1e-6))
    assert(np.allclose(b['values@01.000000@flux_densities@model'], b['values@02.000000@flux_densities@model'], rtol=1e-6, atol=1e-6))
    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')

    b = test_binary()
