"""
"""

import phoebe

def test_binary():
    b = phoebe.Bundle.default_binary()

    b.set_hierarchy()
    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='DEBUG')

    b = test_binary()
