#!/usr/bin/env python3

import numpy as np

def coord_h2j(mass, rh, vh):
    """
    Converts from heliocentric to Jacobi coordinates.

    Rewritten from swift/coord/coord_h2j.f.

    Reference: Levison, H.F., Duncan, M.J., The long-term dynamical behavior of short-period comets. Icarus 108, 18-36, 1994.

    """

    nbod = len(mass)
    eta = np.array(nbod*[0.0])
    rj = np.array((nbod*[[0.0, 0.0, 0.0]]))
    vj = np.array((nbod*[[0.0, 0.0, 0.0]]))

    eta[0] = mass[0]
    for i in range(1,nbod):
       eta[i] = eta[i-1] + mass[i]

    rj[1] = rh[1]
    vj[1] = vh[1]

    sumr = mass[1]*rh[1]
    sumv = mass[1]*vh[1]
    capr = sumr/eta[1]
    capv = sumv/eta[1]

    for i in range(2,nbod):
        rj[i] = rh[i] - capr
        vj[i] = vh[i] - capv

        if i < nbod-1:
             sumr += mass[i]*rh[i]
             sumv += mass[i]*vh[i]
             capr = sumr/eta[i]
             capv = sumv/eta[i]

    return rj, vj

if __name__ == "__main__":

    mass = np.array([1.0, 1.0, 0.5])
    rh = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    vh = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    print("mass = ", mass)
    print("rh[0] = ", rh[0])
    print("rh[1] = ", rh[1])
    print("rh[2] = ", rh[2])

    rj, vj = coord_h2j(mass, rh, vh)

    print("rj[0] = ", rj[0])
    print("rj[1] = ", rj[1])
    print("rj[2] = ", rj[2])


