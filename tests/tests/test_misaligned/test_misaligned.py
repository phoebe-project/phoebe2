"""
"""
import phoebe
import numpy as np


def test_binary(plot=False):
    b = phoebe.Bundle.default_binary()

    b['pitch@primary'] = 10
    b['yaw@primary'] = 10

    b['pitch@secondary'] = 20
    b['yaw@secondary'] = 30

    b['distortion_method@secondary'] = 'rotstar'

    b.add_dataset('lc', times=np.linspace(0, 1, 21))
    b.run_compute(irrad_method='none')

    # TODO: add comparisons????
    if plot:
        b.plot(show=True)


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    test_binary(plot=True)
