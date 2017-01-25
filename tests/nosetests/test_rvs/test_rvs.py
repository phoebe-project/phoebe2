"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt
import phoebeBackend as phb

phoebe.devel_on()
# phoebe.interactive_on()

def _beta_vs_legacy(b):

    period = b.get_value('period@orbit')
    times = np.linspace(-0.2,1.2*period,51)

    b.add_dataset('lc', times=times, dataset='lc01', ld_func='logarithmic', ld_coeffs = [0.5,0.5])
    b.add_dataset('rv', times=times, dataset='rv01', ld_func='logarithmic', ld_coeffs = [0.5,0.5])

    b.add_compute('phoebe', compute='phnum', ltte=False, atm='extern_planckint', rv_method='flux-weighted', irrad_method='none')
    b.add_compute('legacy', compute='legnum', ltte=False, atm='extern_planckint', rv_method='flux-weighted', refl_num=0)


    b.run_compute('phnum', model='phnumresults')
    b.run_compute('legnum', model='legnumresults')

    print "sma: {}, period: {}, q: {}".format(b.get_value('sma@binary'), b.get_value('period@binary'), b.get_value('q'))

    phoebe2_val = b.get_value('rvs@primary@phnumresults@phnum')
    phoebe1_val = b.get_value('rvs@primary@legnumresults@legnum')
    print "rv@primary max rel diff: {}".format(max(np.abs((phoebe1_val-phoebe2_val)/phoebe2_val)))
    assert(np.allclose(phoebe2_val, phoebe1_val, rtol=1e-1, atol=0.))

    phoebe2_val = b.get_value('rvs@secondary@phnumresults@phnum')
    phoebe1_val = b.get_value('rvs@secondary@legnumresults@legnum')
    print "rv@secondary max rel diff: {}".format(max(np.abs((phoebe1_val-phoebe2_val)/phoebe2_val)))
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

            _beta_vs_legacy(b)


if __name__ == '__main__':
    logger = phoebe.logger()
    test_binary()

