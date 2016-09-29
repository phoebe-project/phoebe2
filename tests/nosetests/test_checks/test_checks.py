"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt

phoebe.devel_on()


def test_checks():
    b = phoebe.Bundle.default_binary()


    b.add_dataset('lc')
    b.add_compute()

    # test overflow
    passed, msg = b.run_checks()
    if not passed:
        raise AssertionError(msg)

    b.set_value('rpole', component='primary', value=8)
    passed, msg = b.run_checks()
    if passed:
        raise AssertionError

    b.set_value('rpole', component='primary', value=1.0)

    # TODO: test overlap scenario


    # test ld_func vs ld_coeffs
    passed, msg = b.run_checks()
    if not passed:
        raise AssertionError(msg)

    b.set_value('ld_coeffs_bol', component='primary', value=[0.])
    passed, msg = b.run_checks()
    if passed:
        raise AssertionError
    b.set_value('ld_coeffs_bol', component='primary', value=[0., 0.])

    b.set_value('ld_func', component='primary', value='logarithmic')
    b.set_value('ld_coeffs', component='primary', value=[0.])
    passed, msg = b.run_checks()
    if passed:
        raise AssertionError
    b.set_value('ld_coeffs', component='primary', value=[0., 0.])
    b.set_value('ld_func', component='primary', value='interp')

    # test ld_func vs atm
    passed, msg = b.run_checks()
    if not passed:
        raise AssertionError(msg)

    b.set_value('atm', component='primary', value='blackbody')
    passed, msg = b.run_checks()
    if passed:
        raise AssertionError
    b.set_value('atm', component='primary', value='ck2004')


    # test gravb vs teff warning
    b.set_value('teff', component='primary', value=6000)
    b.set_value('gravb_bol', component='primary', value=1.0)
    passed, msg = b.run_checks()
    if passed is not None:
        raise AssertionError

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    b = test_checks()