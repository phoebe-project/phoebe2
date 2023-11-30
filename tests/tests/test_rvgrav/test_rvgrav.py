# Test gravitational redshift for the Sun. The theoretically
# predicted value from GTR is ~633 m/s.
# 
# The Sun-Jupiter system
# expected RV semi-amplitude of the Sun is 12.5 m/s.

# import numpy as np

import phoebe
from phoebe import u, c
# import libphoebe


def initiate_sun_jupiter_system():
    b = phoebe.default_binary()

    b['teff@primary'] = 1.*u.solTeff
    b['requiv@primary'] = 1.*u.solRad
    b['syncpar@primary'] = 14.61

    b['period@orbit'] = 11.852*u.yr
    b['q@orbit'] = 1.26687e17/1.3271244e20   # (GM)_J / (GM)_Sun
    b['sma@orbit'] = 5.2*u.au

    b['teff@secondary'] = (500, 'K')
    b['requiv@secondary'] = 1.*c.R_jup
    b['atm@secondary'] = 'blackbody'
    
    b['ld_mode_bol@secondary'] = 'manual'
    b['ld_func_bol@secondary'] = 'linear'
    b['ld_coeffs_bol@secondary'] = [0.5]

    b.add_dataset('rv', times=[0,], dataset='grv', passband='Johnson:V')

    b['rv_grav@secondary@grv'] = True
    b['ld_mode@secondary@grv'] = 'manual'
    b['ld_func@secondary@grv'] = 'linear'
    b['ld_coeffs@secondary@grv'] = [0.5]

    return b


def test_rvgrav():
    b = initiate_sun_jupiter_system()

    # first run without rv_grav:
    b.run_compute(rv_grav=False, model='without_rvg')

    # second run with rv_grav:
    b.run_compute(rv_grav=True, model='with_rvg')

    rv_diff = b['value@rvs@primary@with_rvg'][0]-b['value@rvs@primary@without_rvg'][0]

    assert abs(rv_diff-0.6363) < 1e-3


if __name__ == '__main__':
    # logger = phoebe.logger(clevel='INFO')

    test_rvgrav()
