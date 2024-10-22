
from numpy import zeros

def coord_j2b(mass, rj, vj):
    """
    Converts from Jacobi to barycentric coordinates.

    Rewritten from swift/coord/coord_j2b.f.

    Reference: Levison, H.F., Duncan, M.J., The long-term dynamical behavior of short-period comets. Icarus 108, 18-36, 1994.

    """

    nbod = len(mass)
    eta = zeros((nbod))
    rb = zeros((nbod, 3))
    vb = zeros((nbod, 3))

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
    rtmp = zeros((3))
    vtmp = zeros((3))

    for i in range(1,nbod):
        rtmp += mass[i]*rb[i]
        vtmp += mass[i]*vb[i]

    rb[0] = -rtmp/mass[0]
    vb[0] = -vtmp/mass[0]

    return rb, vb


def coord_b2j(mass, rb, vb):
    """
    Converts from barycentric to Jacobi coordinates.

    Rewritten from swift/coord/coord_b2j.f.

    Reference: Levison, H.F., Duncan, M.J., The long-term dynamical behavior of short-period comets. Icarus 108, 18-36, 1994.

    """

    nbod = len(mass)
    eta = zeros((nbod))
    rj = zeros((nbod, 3))
    vj = zeros((nbod, 3))

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


def coord_h2j(mass, rh, vh):
    """
    Converts from heliocentric to Jacobi coordinates.

    Rewritten from swift/coord/coord_h2j.f.

    Reference: Levison, H.F., Duncan, M.J., The long-term dynamical behavior of short-period comets. Icarus 108, 18-36, 1994.

    """

    nbod = len(mass)
    eta = zeros((nbod))
    rj = zeros((nbod, 3))
    vj = zeros((nbod, 3))

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


