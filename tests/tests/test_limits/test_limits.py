"""
"""

import phoebe
from phoebe import u
import pytest

phoebe.logger('DEBUG')


def test_limits():
    b = phoebe.Bundle.default_binary()

    with pytest.raises(ValueError):
        b.set_value('teff@primary', -10)

    with pytest.raises(ValueError):
        b.set_value('requiv@primary', -10*u.km)

    with pytest.raises(ValueError):
        b.set_value('ecc@binary', 1.0)


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    test_limits()
