"""
"""

import phoebe


def test_binary():
    b = phoebe.Bundle.default_binary()

    b.set_hierarchy()
    b.set_hierarchy('orbit:binary(star:secondary, star:primary)')


if __name__ == '__main__':
    logger = phoebe.logger(clevel='DEBUG')
    test_binary()
