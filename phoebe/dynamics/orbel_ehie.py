#!/usr/bin/env python3

from numpy import sin, cos, pi

NMAX = 3
TWOPI = 2.0*pi

def orbel_ehie(e, m):
    """
    Solves Kepler's equation.

    Rewritten from swift/orbel/orbel_ehie.f.

    Reference: Levison, H.F., Duncan, M.J., The long-term dynamical behavior of short-period comets. Icarus 108, 18-36, 1994.

    """

    # In this section, bring M into the range (0,TWOPI) and if
    # the result is greater than PI, solve for (TWOPI - M).
    iflag = 0
    nper = int(m/TWOPI)
    m = m - nper*TWOPI
    if m < 0.0:
         m = m + TWOPI

    if m > pi:
        m = TWOPI - m
        iflag = 1

    # Make a first guess that works well for e near 1.
    x = (6.0*m)**(1.0/3.0) - m

    # Iteration loop
    for niter in range(NMAX):
        sa = sin(x + m); ca = cos(x + m)
        esa = e*sa
        eca = e*ca
        f = x - esa
        fp = 1.0 - eca
        dx = -f/fp
        dx = -f/(fp + 0.5*dx*esa)
        dx = -f/(fp + 0.5*dx*(esa+0.3333333333333333*eca*dx))
        x = x + dx

    cape = m + x

    if iflag == 1:
        cape = TWOPI - cape
        m = TWOPI - m

    return cape

if __name__ == "__main__":

    m = 0.25
    e = 0.6

    print("m = ", m)
    print("e = ", e)

    cape = orbel_ehie(e, m)

    print("cape = ", cape)

    print("")
    print("M = E - e*sin(E)")
    print(m, " = ", cape - e*sin(cape))

