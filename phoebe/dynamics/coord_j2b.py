#!/usr/bin/env python3

import numpy as np

def coord_j2b(mass, rj, vj):
    """
    Converts from Jacobi to barycentric coordinates.

    Rewritten from swift/coord/coord_j2b.f.

    Reference: Levison, H.F., Duncan, M.J., The long-term dynamical behavior of short-period comets. Icarus 108, 18-36, 1994.

    """

    nbod = len(mass)
    eta = np.zeros((nbod))
    rb = np.zeros((nbod, 3))
    vb = np.zeros((nbod, 3))

    # First compute auxiliary variables, then the barycentric positions
    eta[0] = mass[0]
    for i in range(1,nbod):
       eta[i] = eta[i-1] + mass[i]

    i = nbod-1
    mtot = eta[i]
    rb[i] = eta[i-1]*rj[i]/mtot
    vb[i] = eta[i-1]*vj[i]/mtot

    capr = mass[i]*rj[i]/mtot
    capv = mass[i]*vj[i]/mtot

    for i in range(nbod-2,0,-1):
        rat = eta[i-1]/eta[i]
        rb[i] = rat*rj[i] - capr
        vb[i] = rat*vj[i] - capv
       
        rat2 = mass[i]/eta[i]
        capr += rat2*rj[i]
        capv += rat2*vj[i]
	
    # Now compute the Sun's barycentric position
    rtmp = np.array([0.0, 0.0, 0.0])
    vtmp = np.array([0.0, 0.0, 0.0])

    for i in range(1,nbod):
        rtmp += mass[i]*rb[i]
        vtmp += mass[i]*vb[i]

    rb[0] = -rtmp/mass[0]
    vb[0] = -vtmp/mass[0]

    return rb, vb

if __name__ == "__main__":

    mass = np.array([1.0, 1.0, 0.5])
    rj = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    vj = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    print("mass = ", mass)
    print("rj[0] = ", rj[0])
    print("rj[1] = ", rj[1])
    print("rj[2] = ", rj[2])

    rb, vb = coord_j2b(mass, rj, vj)

    print("rb[0] = ", rb[0])
    print("rb[1] = ", rb[1])
    print("rb[2] = ", rb[2])

    com = np.array([0.0, 0.0, 0.0])
    for i in range(3):
        com[i] = np.dot(mass, rb[:,i])/np.sum(mass)

    print("com = ", com)

