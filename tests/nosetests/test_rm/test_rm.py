"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt
import phoebeBackend as phb

phoebe.devel_on()
# phoebe.interactive_on()

def _beta_vs_legacy(b, plot=False):



    b.run_compute('phnum', model='phnumresults')
    b.run_compute('legnum', model='legnumresults')




    phoebe2_val = b.get_value('rvs@primary@phnumresults@phnum')
    phoebe1_val = b.get_value('rvs@primary@legnumresults@legnum')
    print "rv@primary max abs diff: {}".format(max(np.abs(phoebe1_val-phoebe2_val)))
    assert(np.allclose(phoebe2_val, phoebe1_val, rtol=0., atol=2.0))

    phoebe2_val = b.get_value('rvs@secondary@phnumresults@phnum')
    phoebe1_val = b.get_value('rvs@secondary@legnumresults@legnum')
    print "rv@secondary max abs diff: {}".format(max(np.abs(phoebe1_val-phoebe2_val)))
    assert(np.allclose(phoebe2_val, phoebe1_val, rtol=0., atol=2.0))

    if plot:
        plt.cla()
        b.plot()
        plt.show()


def test_binary(plot=False):

    b = phoebe.default_binary()

    period = b.get_value('period@orbit')
    times = np.linspace(-0.2,1.2*period,51)

    b.add_dataset('rv', times=times, dataset='rv01', ld_func='logarithmic', ld_coeffs = [0.5,0.5])

    b.add_compute('phoebe', compute='phnum', ltte=False, atm='extern_planckint', rv_method='flux-weighted', irrad_method='none')
    b.add_compute('legacy', compute='legnum', ltte=False, atm='blackbody', rv_method='flux-weighted', refl_num=0)

    for syncpar in [1./4, 4]:
        print "setting syncpar@primary to", syncpar
        b.set_value('syncpar@primary', syncpar)
        _beta_vs_legacy(b, plot=plot)

    return b


if __name__ == '__main__':
    logger = phoebe.logger()
    b = test_binary(plot=True)

