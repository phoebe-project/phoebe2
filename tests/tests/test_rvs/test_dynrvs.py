import phoebe
import numpy as np


def test_dynrvs(verbose=False):
    """
    This test checks whether PHOEBE computes dynamical RVs without building
    a mesh (as otherwise run_compute would fail with atm out-of-bounds errors), 
    and whether the computed RVs are close to the flux-weighted RVs.
    """
    b = phoebe.default_binary()
    b.add_dataset('rv', compute_times=phoebe.linspace(0, 1, 21), dataset='dynrv')
    b.set_value_all('rv_method', 'dynamical')
    b.set_value_all('teff', 1000000)
    b.run_compute()

    b.set_value_all('teff', 5772)
    b.add_dataset('rv', compute_times=phoebe.linspace(0, 1, 21), dataset='fluxrv')
    b.set_value('rv_method', component='primary', dataset='fluxrv', value='flux-weighted')
    b.set_value('rv_method', component='secondary', dataset='fluxrv', value='flux-weighted')
    b.run_compute(eclipse_method='only_horizon')

    if verbose:
        print('maximum primary RV offset:', np.abs(b['value@rvs@primary@dynrv']-b['value@rvs@primary@fluxrv']).max())
        print('maximum secondary RV offset:', np.abs(b['value@rvs@secondary@dynrv']-b['value@rvs@secondary@fluxrv']).max())

    assert np.allclose(b['value@rvs@primary@dynrv'], b['value@rvs@primary@fluxrv'], atol=0.3)
    assert np.allclose(b['value@rvs@secondary@dynrv'], b['value@rvs@secondary@fluxrv'], atol=0.3)


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    test_dynrvs(verbose=True)
