"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt

phoebe.devel_on()

def test_binary(plot=False):
    b = phoebe.Bundle.default_binary()

    b.set_value('rpole', component='primary', value=1.2)
    b.set_value('rpole', component='secondary', value=0.8)

    for q in [0.1, 1.0, 0.95]:
        for syncpar in [0.0, 1.0, 2.0]:
            for e in [0.0, 0.3]:
                for sma in [8, 20]:
                    b.set_value('sma', component='binary', value=sma)
                    b.set_value('ecc', e)
                    b.set_value_all('syncpar', syncpar)
                    b.set_value('q', q)
                    b.run_delayed_constraints()

                    for component, comp_no in {'primary': 1, 'secondary': 2}.items():

                        rpole = b.get_value('rpole', component=component, context='component')
                        sma = b.get_value('sma', component='binary', context='component')
                        rpole = rpole/sma
                        q = b.get_value('q', component='binary', context='component')
                        e = b.get_value('ecc', component='binary', context='component')
                        F = b.get_value('syncpar', component=component, context='component')
                        Omega = b.get_value('pot', component=component, context='component')

                        pot = phoebe.distortions.roche.rpole2potential(rpole, q, e, F, component=comp_no)
                        rp = phoebe.distortions.roche.potential2rpole(pot, q, e, F, component=comp_no)

                        if plot:
                            print "pot", Omega, pot
                            print "rpole", rpole, rp

                        assert(abs(Omega-pot) < 1e-6)
                        assert(abs(rpole-rp)< 1e-6)

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    b = test_binary(plot=True)