#!/usr/bin/env python3

import numpy as np

def coord_b2j(mass, rb, vb):
    """
    Converts from barycentric to Jacobi coordinates.

    Rewritten from swift/coord/coord_b2j.f.

    Reference: Levison, H.F., Duncan, M.J., The long-term dynamical behavior of short-period comets. Icarus 108, 18-36, 1994.

    """

    nbod = len(mass)
    eta = np.zeros((nbod))
    rj = np.zeros((nbod, 3))
    vj = np.zeros((nbod, 3))

    # First compute auxiliary variables, then the Jacobi positions
    eta[0] = mass[0]
    for i in range(1,nbod):
       eta[i] = eta[i-1] + mass[i]

    sumr = mass[0]*rb[0]
    sumv = mass[0]*vb[0]
    capr = rb[0]
    capv = vb[0]

    for i in range(1, nbod-1):
        rj[i] = rb[i] - capr
        vj[i] = vb[i] - capv

        sumr += mass[i]*rb[i]
        sumv += mass[i]*vb[i]
        capr = sumr/eta[i]
        capv = sumv/eta[i]

    rj[nbod-1] = rb[nbod-1] - capr
    vj[nbod-1] = vb[nbod-1] - capv
 
    return rj, vj

if __name__ == "__main__":

    mass = np.array([1.0, 1.0, 0.5])
    rb = np.array([[-0.5, -0.2, 0.0], [0.5, -0.2, 0.0], [0.0, 0.8, 0.0]])
    vb = np.array([[-0.5, -0.2, 0.0], [0.5, -0.2, 0.0], [0.0, 0.8, 0.0]])

    print("mass = ", mass)
    print("rb[0] = ", rb[0])
    print("rb[1] = ", rb[1])
    print("rb[2] = ", rb[2])

    rj, vj = coord_b2j(mass, rb, vb)

    print("rj[0] = ", rj[0])
    print("rj[1] = ", rj[1])
    print("rj[2] = ", rj[2])

