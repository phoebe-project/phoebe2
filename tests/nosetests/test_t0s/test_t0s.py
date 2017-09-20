"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt

from nose.tools import assert_raises

phoebe.interactive_on()

def test_binary(verbose=False):
    def assert_t0s(p1_t0_ref, p1_t0_supconj, p1_t0_perpass, tol=1e-4):
        # b.run_delayed_constraints()

        p2_t0_ref = b.get_value('t0_ref@component')
        p2_t0_supconj = b.get_value('t0_supconj@component')
        p2_t0_perpass = b.get_value('t0_perpass@component')

        if verbose:
            print("{}=={}, {}=={}, {}=={}".format(p1_t0_ref, p2_t0_ref, p1_t0_supconj, p2_t0_supconj, p1_t0_perpass, p2_t0_perpass))

        assert(abs(p2_t0_ref-p1_t0_ref) < tol)
        assert(abs(p2_t0_supconj-p1_t0_supconj) < tol)
        assert(abs(p2_t0_perpass-p1_t0_perpass) < tol)

    b = phoebe.Bundle.default_binary()

    b = phoebe.default_binary()
    # ecc = 0, per0 = 0
    # compare against values in PHOEBE 1 by manually setting the HJD0 in PHOEBE 1
    # so that supconj matches the prescribed 0.0
    p1_ref = 0.0 # SET
    p1_supconj = 0.0
    p1_perpass = -0.25
    assert_t0s(p1_ref, p1_supconj, p1_perpass)

    b['ecc'] = 0.2
    p1_ref = 0.063235 # SET
    p1_supconj = 0.0
    p1_perpass = -0.186765
    assert_t0s(p1_ref, p1_supconj, p1_perpass)

    b['per0'] = 90
    p1_ref = 0.0 # SET
    p1_supconj = 0.0
    p1_perpass = 0.0
    assert_t0s(p1_ref, p1_supconj, p1_perpass)

    b.flip_constraint('t0_ref', solve_for='t0_supconj')
    # shouldn't change values
    assert_t0s(p1_ref, p1_supconj, p1_perpass)

    b['t0_ref@component'] = 0.0
    b['per0'] = 180
    p1_ref = 0.0 # SET
    p1_supconj = 0.063235
    p1_perpass = 0.25
    assert_t0s(p1_ref, p1_supconj, p1_perpass)

    b['t0_ref@component'] = 0.3
    b['per0'] = 45
    p1_ref = 0.3 # SET
    p1_supconj = 0.259489
    p1_perpass = 0.17500
    assert_t0s(p1_ref, p1_supconj, p1_perpass)

    # cannot flip both constraints to solve for t0_supconj
    assert_raises(ValueError, b.flip_constraint, 't0_perpass', solve_for='t0_supconj')

    b.flip_constraint('t0_supconj', solve_for='t0_ref')
    b.flip_constraint('t0_perpass', solve_for='t0_supconj')
    # shouldn't change values
    assert_t0s(p1_ref, p1_supconj, p1_perpass)

    b['t0_perpass'] = 0.0
    b['period@orbit'] = 3
    b['ecc'] = 0.3
    b['per0'] = 0.123, 'rad'
    # set p1 HJD0 = +0.691272
    p1_ref = 0.691272 # SET
    p1_supconj = 0.418709
    p1_perpass = 0.0
    assert_t0s(p1_ref, p1_supconj, p1_perpass)

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')

    b = test_binary(verbose=True)
