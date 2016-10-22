import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt


def test_binary(plot=False):

    oc = phoebe.Bundle.default_binary(overcontact=True)
    q = 1.
    oc['pot@common_envelope'] = 3.5
    oc['q'] = q
    oc['teff@primary'] = 5000.
    oc['teff@secondary'] = 5000.
    oc['incl'] = 90


    oc.add_dataset('lc', times=np.linspace(0,3,50), dataset='lc01')
    oc.add_dataset('rv', time=np.linspace(0,3,50), dataset='rv01')

    oc.add_compute('phoebe', compute='phoebe2', mesh_method='marching')
    oc.add_compute('legacy', compute='phoebe1', morphology = 'Overcontact binary not in thermal contact')

    oc.set_value_all('atm@phoebe2', 'extern_planckint')
    oc.set_value_all('atm@phoebe1', 'blackbody')

    # turn off limb-darkening:
    oc.set_value_all('ld_func_bol', 'logarithmic')
    oc.set_value_all('ld_coeffs_bol', [0.0, 0.0])

    oc.set_value_all('ld_func', 'logarithmic')
    oc.set_value_all('ld_coeffs', [0.0, 0.0])

    oc.set_value_all('refl_num',0)
    oc.set_value_all('rv_grav', False)
    oc.set_value_all('ltte', False)

    print "running phoebe2 model..."
    oc.run_compute(compute='phoebe2', reflection_method='none', model='phoebe2model')
    print "running phoebe1 model..."
    oc.run_compute(compute='phoebe1', refl_num=0, model='phoebe1model')

    phoebe2_val_lc = oc.get_value('fluxes@phoebe2model')
    phoebe1_val_lc = oc.get_value('fluxes@phoebe1model')
    phoebe2_val_rv1 = oc.get_value('rvs@primary@phoebe2model')[1:49]
    phoebe1_val_rv1 = oc.get_value('rvs@primary@phoebe1model')[1:49]
    phoebe2_val_rv2 = oc.get_value('rvs@secondary@phoebe2model')
    phoebe1_val_rv2 = oc.get_value('rvs@secondary@phoebe1model')

    if plot:
        oc.plot(dataset='lc01')
        plt.legend()
        plt.show()

        oc.plot(dataset='rv01')
        plt.legend()
        plt.show()

    assert(np.allclose(phoebe2_val_lc, phoebe1_val_lc, rtol=3e-3, atol=0.))
    assert(np.allclose(phoebe2_val_rv1, phoebe1_val_rv1, rtol=0., atol=0.2))
    assert(np.allclose(phoebe2_val_rv2, phoebe1_val_rv2, rtol=0., atol=0.2))
    return oc

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')

    oc = test_binary(plot=False)
