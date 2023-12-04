"""
"""

import phoebe

def test_binary():
    b = phoebe.Bundle.default_binary()

    b.set_hierarchy()

    b.set_hierarchy('orbit:binary(star:secondary, star:primary)')

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='DEBUG')

    b = test_binary()
