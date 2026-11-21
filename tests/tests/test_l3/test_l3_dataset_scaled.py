"""
Third light given as a fraction (l3_mode='fraction') must be the same with
pblum_mode='dataset-scaled' as with the default pblum_mode='component-coupled' (#1187).
"""

import numpy as np
import phoebe
from phoebe import u


def test_l3_frac_dataset_scaled(verbose=False):
    for contact_binary in [False, True]:
        b = phoebe.default_binary(contact_binary=contact_binary)
        b.add_dataset('lc', times=phoebe.linspace(0, 1, 21), l3_mode='fraction')
        b.set_value('l3_frac', 0.2)

        # a distance other than the default 1 m, so the distance factors are tested too
        b.set_value('distance', 10*u.pc)

        # synthetic observations, computed with the default pblum_mode='component-coupled'
        b.run_compute(irrad_method='none', model='coupled')
        fluxes = b.get_value('fluxes', model='coupled')
        b.set_value('fluxes', context='dataset', value=fluxes)

        # dataset-scaling only rescales the model, so it must reproduce these observations exactly
        b.set_value('pblum_mode', 'dataset-scaled')
        b.run_compute(irrad_method='none', model='scaled')
        scaled_fluxes = b.get_value('fluxes', model='scaled')

        if verbose:
            print(f"contact_binary={contact_binary}: max relative difference {abs(scaled_fluxes/fluxes - 1).max()}")

        assert np.allclose(scaled_fluxes, fluxes, rtol=1e-6, atol=0.), f"contact_binary={contact_binary}"


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    test_l3_frac_dataset_scaled(verbose=True)
