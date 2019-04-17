import phoebe
# from phoebe import u
# import numpy as np
# import matplotlib.pyplot as plt


def test_star():
    b1 = phoebe.default_star(force_build=True)
    b2 = phoebe.default_star()

    # TODO: add comparison

def test_binary():
    b1 = phoebe.default_binary(force_build=True)
    b2 = phoebe.default_binary()

    # TODO: add comparison

def test_contact_binary():
    b1 = phoebe.default_binary(contact_binary=True, force_build=True)
    b2 = phoebe.default_binary(contact_binary=True)

    # TODO: add comparison

if __name__ == '__main__':
    logger = phoebe.logger(clevel='debug')

    test_star()
    test_binary()
    test_contact_binary()
