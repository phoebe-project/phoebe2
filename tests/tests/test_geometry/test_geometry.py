#!/usr/bin/env python3

import numpy as np

from phoebe.dynamics import geometry
from phoebe import u
from phoebe import c

TINY = 4.0e-16

def test_geometry(verbose=False):

    day = u.d.to('s')
    au = u.au.to('m')
    M_S = u.solMass.to('kg')
    G = c.G.to('kg^-1 m^3 s^-2').value
    gms = G*M_S / (au**3 * day**-2)

    m = gms*np.array([1.0, 0.0])
    elmts = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    rb, vb = geometry.geometry_hierarchical(m, elmts)

    if verbose:
        print("m = ", m)
        print("rb[0] = ", rb[0])
        print("rb[1] = ", rb[1])
        print("vb[0] = ", vb[0])
        print("vb[1] = ", vb[1])

    assert(abs(rb[1,0] - 1.0) < TINY)
    assert(abs(rb[1,1] - 0.0) < TINY)
    assert(abs(rb[1,2] - 0.0) < TINY)
    assert(abs(vb[1,0] - 0.0) < TINY)
    assert(abs(vb[1,1] - 1.7202098947281922e-02) < TINY)
    assert(abs(vb[1,2] - 0.0) < TINY)

    elmts, euler, roche = geometry.invgeometry_hierarchical(m, rb, vb)

    if verbose:
        print("")
        print("m = ", m)
        print("elmts[0] = ", elmts[0])
        print("euler[0] = ", euler[0])
        print("euler[1] = ", euler[1])
        print("roche[0] = ", roche[0])
        print("roche[1] = ", roche[1])

    assert(abs(elmts[0,0] - 1.0) < TINY)
    assert(abs(elmts[0,1] - 0.0) < TINY)
    assert(abs(elmts[0,2] - 0.0) < TINY)
    assert(abs(elmts[0,3] - 0.0) < TINY)
    assert(abs(elmts[0,4] - 0.0) < TINY)
    assert(abs(elmts[0,5] - 0.0) < TINY)

    m = gms*np.array([1.0, 1.0, 0.5, 0.5])
    elmts = []
    elmts.append([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    elmts.append([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    elmts.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    rb, vb = geometry.geometry_twopairs(m, elmts)

    if verbose:
        print("")
        print("m = ", m)
        print("rb[0] = ", rb[0])
        print("rb[1] = ", rb[1])
        print("rb[2] = ", rb[2])
        print("rb[3] = ", rb[3])
        print("vb[0] = ", vb[0])
        print("vb[1] = ", vb[1])
        print("vb[2] = ", vb[2])
        print("vb[3] = ", vb[3])

    elmts, euler, roche = geometry.invgeometry_twopairs(m, rb, vb)

    if verbose:
        print("")
        print("m = ", m)
        print("elmts[0] = ", elmts[0])
        print("elmts[1] = ", elmts[1])
        print("elmts[2] = ", elmts[2])
        print("euler[0] = ", euler[0])
        print("euler[1] = ", euler[1])
        print("euler[2] = ", euler[2])
        print("euler[3] = ", euler[3])
        print("roche[0] = ", roche[0])
        print("roche[1] = ", roche[1])
        print("roche[2] = ", roche[2])
        print("roche[3] = ", roche[3])

    assert(abs(elmts[0,0] - 0.1) < TINY)
    assert(abs(elmts[1,0] - 0.2) < TINY)
    assert(abs(elmts[2,0] - 1.0) < TINY)

if __name__ == "__main__":

    test_geometry(verbose=True)


