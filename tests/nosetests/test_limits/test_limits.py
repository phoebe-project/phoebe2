"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt

from nose.tools import assert_raises

phoebe.logger('DEBUG')

def test_limits():
    b = phoebe.Bundle.default_binary()


    assert_raises(ValueError, b.set_value, 'teff@primary', -10)
    assert_raises(ValueError, b.set_value, 'teff@primary', -10*u.Celsius)

    assert_raises(ValueError, b.set_value, 'ecc@binary', 1.0)

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    b = test_limits()
