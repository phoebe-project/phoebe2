"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt

phoebe.devel_on()

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


    period = b.get_value('period@binary')
    b.add_dataset('lc', times=np.linspace(0,period,21))
    b.add_compute('phoebe', irrad_method='none', compute='phoebe2')
    b.add_compute('legacy', refl_num=0, compute='phoebe1')



    # set matching limb-darkening for bolometric
    b.set_value_all('ld_func_bol', 'logarithmic')
    b.set_value_all('ld_coeffs_bol', [0.0, 0.0])

    b.set_value_all('ld_func', 'logarithmic')
    b.set_value_all('ld_coeffs', [0.0, 0.0])

    for ld_func in b.get('ld_func', component='primary').choices:
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
            if ld_func in ['logarithmic', 'linear', 'square_root']:
                # TODO: add linear and square_root once bugs 111 and 112 are fixed
                ld_func_ph1 = ld_func
                ld_coeffs_ph1 = ld_coeffs
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

            print "exact_comparison: {}, max (rel): {}".format(exact_comparison, abs((phoebe2_val-phoebe1_val)/phoebe1_val).max())

            if plot:
                b.plot(dataset='lc01')
                plt.legend()
                plt.show()

            assert(np.allclose(phoebe2_val, phoebe1_val, rtol=3e-3 if exact_comparison else 0.3, atol=0.))


    for atm in ['ck2004', 'blackbody']:
        # don't need much time-resolution at all since we're just using median
        # to compare
        b.set_value('times@dataset', np.linspace(0,3,11))
        b.set_value_all('atm@phoebe2', atm)

        for ld_func in b.get('ld_func', component='primary').choices:
            # now we just want to make sure the median value doesn't change
            # as a function of ld_coeffs - to make sure the pblum scaling is
            # accounting for limb-darkening correctly.
            # This is especially important for those we couldn't check above
            # vs legacy (quadratic, power), but also important to run all
            # with blackbody.
            if ld_func=='interp':
                # there are no ld_coeffs here to vary
                continue

            b.set_value_all('ld_func', ld_func)

            med_fluxes = []
            if ld_func == 'power':
                ld_coeff_loop = [0.0, 0.2]
            elif ld_func == 'logarithmic':
                ld_coeff_loop = [0.0, 0.4]
            else:
                ld_coeff_loop = [0.0, 0.8]

            for ld_coeff in ld_coeff_loop:
                ld_coeffs = _get_ld_coeffs(ld_coeff, ld_func)

                b.set_value_all('ld_coeffs', ld_coeffs)

                b.run_compute(compute='phoebe2', model='phoebe2model')

                med_fluxes.append(np.median(b.get_value('fluxes@phoebe2model')))

                if plot:
                    b.plot(model='phoebe2model')

            med_fluxes = np.array(med_fluxes)
            diff_med_fluxes = med_fluxes.max() - med_fluxes.min()
            print "atm={} ld_func={} range(med_fluxes): {}".format(atm, ld_func, diff_med_fluxes)

            if plot:
                b.show()

            assert(diff_med_fluxes < 0.025)



    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    b = test_binary(plot=False)