"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt

def _get_ld_coeffs(ld_coeff, ld_func):
    # length of ld_coeffs depends on ld_func
    if ld_coeff is None:
        ld_coeffs = None
    elif ld_func in ['linear']:
        ld_coeffs = [ld_coeff]
    elif ld_func in ['logarithmic', 'square_root', 'quadratic']:
        ld_coeffs = [ld_coeff, ld_coeff]
    elif ld_func in ['power']:
        ld_coeffs = [ld_coeff, ld_coeff, ld_coeff, ld_coeff]
    elif ld_func in ['interp']:
        ld_coeffs = None
    else:
        raise NotImplementedError

    return ld_coeffs

def test_binary(plot=False):
    b = phoebe.Bundle.default_binary()


    b.add_dataset('lc', times=np.linspace(0,3,101))
    b.add_compute('phoebe', reflection_method='none', compute='phoebe2')
    b.add_compute('legacy', refl_num=0, compute='phoebe1')



    # set matching limb-darkening for bolometric
    b.set_value_all('ld_func_bol', 'logarithmic')
    b.set_value_all('ld_coeffs_bol', [0.0, 0.0])

    b.set_value_all('ld_func', 'logarithmic')
    b.set_value_all('ld_coeffs', [0.0, 0.0])

    for ld_func in []: #b.get('ld_func', component='primary').choices:
        # let's test all of these against legacy.  For some we don't have
        # exact comparisons, so we'll get close and leave a really lose
        # tolerance.

        ld_coeff_loop = [None] if ld_func=='interp' else [0.5]

        for ld_coeff in ld_coeff_loop:

            ld_coeffs = _get_ld_coeffs(ld_coeff, ld_func)


            if ld_func=='interp':
                atm = 'ck2004'
                atm_ph1 = 'kurucz'
                exact_comparison = False

            else:
                atm = 'extern_atmx'
                atm_ph1 = 'kurucz'
                exact_comparison = True


            # some ld_funcs aren't supported by legacy.  So let's fall back
            # on logarthmic at least to make sure there isn't a large offset
            if ld_func in ['logarithmic']:
                # TODO: add linear and square_root once bugs 111 and 112 are fixed
                ld_func_ph1 = ld_func
                ld_coeffs_ph1 = ld_coeffs
                exact_comparison = exact_comparison
            elif ld_func in ['linear']:
                # then we can fudge this by just sending the first coefficient
                # to a different ld_func
                ld_func_ph1 = 'logarithmic'
                if ld_coeffs is None:
                    ld_coeffs_ph1 = [0.0, 0.0]
                else:
                    ld_coeffs_ph1 = [ld_coeff, 0.0]
                exact_comparison = exact_comparison
            else:
                ld_func_ph1 = 'logarithmic'
                if ld_coeffs is None:
                    ld_coeffs_ph1 = [0.0, 0.0]
                else:
                    ld_coeffs_ph1 = [ld_coeff, ld_coeff]
                exact_comparison = False


            print "running phoebe2 model atm={}, ld_func={}, ld_coeffs={}...".format(atm, ld_func, ld_coeffs)


            b.set_value_all('atm@phoebe2', atm)
            b.set_value_all('ld_func', ld_func)
            if ld_coeffs is not None:
                b.set_value_all('ld_coeffs', ld_coeffs)


            b.run_compute(compute='phoebe2', model='phoebe2model')


            print "running phoebe1 model atm={}, ld_func={}, ld_coeffs={}...".format(atm_ph1, ld_func_ph1, ld_coeffs_ph1)

            b.set_value_all('atm@phoebe1', atm_ph1)
            b.set_value_all('ld_func', ld_func_ph1)
            b.set_value_all('ld_coeffs', ld_coeffs_ph1)

            b.run_compute(compute='phoebe1', model='phoebe1model')

            phoebe2_val = b.get_value('fluxes@phoebe2model')
            phoebe1_val = b.get_value('fluxes@phoebe1model')

            if plot:
                print "exact_comparison: {}, max (rel): {}".format(exact_comparison, abs((phoebe2_val-phoebe1_val)/phoebe1_val).max())

                b.plot(dataset='lc01')
                plt.legend()
                plt.show()

            assert(np.allclose(phoebe2_val, phoebe1_val, rtol=2e-3 if exact_comparison else 0.3, atol=0.))


    for ld_func in ['quadratic', 'power', 'square_root']:
        # TODO: remove square_root once supported fully above

        # these we just want to make sure the median value doesn't change
        # as a function of ld_coeffs - to make sure the pblum scaling is
        # accounting for limb-darkening correctly.

        b.set_value_all('atm@phoebe2', 'ck2004')
        b.set_value_all('ld_func', ld_func)

        med_fluxes = []
        for ld_coeff in [0.0, 0.25, 0.5, 0.75]:
            ld_coeffs = _get_ld_coeffs(ld_coeff, ld_func)

            b.set_value_all('ld_coeffs', ld_coeffs)

            b.run_compute(compute='phoebe2', model='phoebe2model')

            med_fluxes.append(np.median(b.get_value('fluxes@phoebe2model')))


        std = np.stats.std(np.array(med_fluxes))
        print "std(med_fluxes): {}".format(std)
        assert(std < 0.05)



    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    b = test_binary(plot=True)