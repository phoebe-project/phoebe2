"""
"""

import phoebe

def test_all():
    print("phoebe.Bundle() ...")
    b = phoebe.Bundle()
    print("phoebe.default_star()) ...")
    b = phoebe.default_star()
    print("phoebe.default_binary() ...")
    b = phoebe.default_binary()
    print("phoebe.default_binary(contact_binary=True) ...")
    b = phoebe.default_binary(contact_binary=True)
    # b = phoebe.default_triple(inner_as_primary=True)
    # b = phoebe.default_triple(inner_as_primary=False)

if __name__ == '__main__':
    logger = phoebe.logger(clevel='WARNING')

    test_all()
