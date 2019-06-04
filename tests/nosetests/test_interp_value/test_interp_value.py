import phoebe
from phoebe import u
# import numpy as np
# import matplotlib.pyplot as plt


def test_interp_value():
    b = phoebe.default_binary()
    b.add_dataset('lc', times=[0, 1], fluxes=[0.8, 0.7])

    assert(abs(b.get_parameter('fluxes').interp_value(times=0.5)-0.75) < 1e-12)

    times_in_s = (0.5*u.d).to(u.s)
    assert(abs(b.get_parameter('fluxes').interp_value(times=times_in_s)-0.75) < 1e-12)

    assert(abs(b.get_parameter('fluxes').interp_quantity(times=times_in_s).value-0.75) < 1e-12)

if __name__ == '__main__':
    logger = phoebe.logger(clevel='debug')

    test_interp_value()
