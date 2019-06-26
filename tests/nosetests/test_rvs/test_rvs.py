"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt


def _beta_vs_legacy(b, plot=False):

    period = b.get_value('period@orbit')
    times = np.linspace(-0.2,1.2*period,51)

    b.set_value('vgamma', 50)

    #turn off albedos (legacy requirement)
    b.set_value_all('irrad_frac_refl_bol',  0.0)

    b.add_dataset('rv', times=times, dataset='rv01', ld_mode='manual', ld_func='logarithmic', ld_coeffs=[0.5,0.5])

    b.add_compute('phoebe', compute='phnum', ltte=False, atm='extern_planckint', rv_method='flux-weighted', irrad_method='none')
    b.add_compute('legacy', compute='legnum', ltte=False, atm='extern_planckint', rv_method='flux-weighted', refl_num=0)


    b.run_compute('phnum', model='phnumresults')
    b.run_compute('legnum', model='legnumresults')

    if plot:
        b.plot(show=True)
        print("sma: {}, period: {}, q: {}".format(b.get_value('sma@binary'), b.get_value('period@binary'), b.get_value('q')))

    phoebe2_val = b.get_value('rvs@primary@phnumresults@phnum')
    phoebe1_val = b.get_value('rvs@primary@legnumresults@legnum')
    if plot:
        print("rv@primary max rel diff: {}".format(max(np.abs((phoebe1_val-phoebe2_val)/phoebe2_val))))
    assert(np.allclose(phoebe2_val, phoebe1_val, rtol=1e-1, atol=0.))

    phoebe2_val = b.get_value('rvs@secondary@phnumresults@phnum')
    phoebe1_val = b.get_value('rvs@secondary@legnumresults@legnum')
    if plot:
        print("rv@secondary max rel diff: {}".format(max(np.abs((phoebe1_val-phoebe2_val)/phoebe2_val))))
    assert(np.allclose(phoebe2_val, phoebe1_val, rtol=1e-1, atol=0.))


def test_binary(plot=False):

    ## system = [sma (solRad), period (d)]
    system1 = [11, 2.575]
    system2 = [215., 257.5]
    system3 = [8600., 65000.]

    for q in [0.5,1.]:
        for system in [system1, system2, system3]:

            b = phoebe.Bundle.default_binary()

            b.set_value('sma@binary',system[0])
            b.set_value('period@binary', system[1])
            b.set_value('q', q)

            _beta_vs_legacy(b, plot=plot)


if __name__ == '__main__':
    logger = phoebe.logger('debug')
    test_binary(plot=True)
