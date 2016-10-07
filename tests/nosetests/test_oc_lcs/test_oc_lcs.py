import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt


def test_binary(plot=False):

    logger = phoebe.logger()

    oc = phoebe.Bundle.default_binary(overcontact=True)
    q = 1.
    oc['pot@common_envelope'] = 3.5
    oc['q'] = q
    oc['teff@primary'] = 7000.
    oc['teff@secondary'] = 5000.
    oc['incl'] = 90


    oc.add_dataset('lc', time=np.linspace(0,3,100), dataset='lc01')
    oc.set_value_all('ld_func','logarithmic')

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

    phoebe2_val = oc.get_value('fluxes@phoebe2model')
    phoebe1_val = oc.get_value('fluxes@phoebe1model')

    if plot:
        oc.plot(dataset='lc01')
        plt.legend()
        plt.show()

    assert(np.allclose(phoebe2_val, phoebe1_val, rtol=5e-3, atol=0.))

    return oc

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    oc = test_binary(plot=False)
