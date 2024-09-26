#!/usr/bin/env python3

from numpy import sin, cos, sqrt, array

from phoebe.dynamics import orbel_ehie

TINY = 4.0e-15

def orbel_el2xv(gm, ialpha, elmts):
    """
    Computes cartesian positions and velocities given orbital elements.

    Rewritten from swift/coord/coord_j2b.f.

    Reference: Levison, H.F., Duncan, M.J., The long-term dynamical behavior of short-period comets. Icarus 108, 18-36, 1994.

    """

    a, e, inc, capom, omega, capm = elmts

    if e < 0.0:
       print('WARNING in orbel_el2xv: e<0, setting e=0!')
       e = 0.0

    # check for inconsistencies between ialpha and e
    em1 = e - 1.0
    if (ialpha == 0 and abs(em1) > TINY) or \
        (ialpha < 0 and e > 1.0) or \
        (ialpha > 0 and e < 1.0):

        print('ERROR in orbel_el2xv: ialpha and e inconsistent')
        print('ialpha = ', ialpha)
        print('e = ', e)
        raise ValueError

    # Generate rotation matrices (on p. 42 of Fitzpatrick)
    sp = sin(omega); cp = cos(omega)
    so = sin(capom); co = cos(capom)
    si = sin(inc); ci = cos(inc)
    d1 = array([ cp*co - sp*so*ci,  cp*so + sp*co*ci, sp*si])
    d2 = array([-sp*co - cp*so*ci, -sp*so + cp*co*ci, cp*si])

    # Get the other quantities depending on orbit type (i.e., ialpha)
    if ialpha == -1:

        cape = orbel_ehie.orbel_ehie(e, capm)
        scap = sin(cape); ccap = cos(cape)
        sqe = sqrt(1.0 - e*e)
        sqgma = sqrt(gm*a)
        xfac1 = a*(ccap - e)
        xfac2 = a*sqe*scap
        ri = 1.0/(a*(1.0 - e*ccap))
        vfac1 = -ri * sqgma * scap
        vfac2 = ri * sqgma * sqe * ccap

    else:
        raise NotImplementedError

    r = d1*xfac1 + d2*xfac2
    v = d1*vfac1 + d2*vfac2

    return r, v

if __name__ == "__main__":

    day = 86400.
    au = 1.496e11
    M_S = 2.e30
    G = 6.67e-11
    gm = G*M_S / (au**3 * day**-2)

    elmts = [1.0, 0.1, 0.0, 0.0, 0.0, 0.0]

    print("gm = ", gm)
    print("elmts = ", elmts)

    r, v = orbel_el2xv(gm, -1, elmts)

    print("r = ", r)
    print("v = ", v)


