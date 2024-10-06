#!/usr/bin/env python3

from numpy import pi, sin, cos, tan, arccos, arctan, arctan2, sqrt, dot, cross, array

TINY = 4.0e-15

cape = None
absr = None

def orbel_xv2el(gm, r, v):
    """
    Computes orbial elements given cartesian positions and velocities.

    Rewritten from swift/orbel/orbel_xv2el.f.

    Reference: Levison, H.F., Duncan, M.J., The long-term dynamical behavior of short-period comets. Icarus 108, 18-36, 1994.

    """
    global cape
    global absr

    # Compute the angular momentum, and thereby the inclination.
    h = cross(r, v)
    h2 = dot(h, h)
    absh = sqrt(h2)
    inc = arccos(h[2]/absh)

    fac = sqrt(h[0]*h[0] + h[1]*h[1])/absh

    if fac < TINY:
        capom = 0.0
        u = arctan2(r[1], r[0])
        if abs(inc - pi) < 10.0*TINY:
           u = -u
    else:
        capom = arctan2(h[0], -h[1])  
        u = arctan2(r[2]/sin(inc), r[0]*cos(capom) + r[1]*sin(capom))

    if capom < 0.0:
        capom += 2.0*pi

    if u < 0.0:
        u += 2.0*pi

    # Compute the radius R and velocity squared V2, and the dot product RDOTV, the energy per unit mass ENERGY .

    absr = sqrt(dot(r, r))
    v2 = dot(v, v)
    absv = sqrt(v2)
    vdotr = dot(r, v)
    energy = 0.5*v2 - gm/absr

    # Determine type of conic section and label it via IALPHA
    if abs(energy*absr/gm) < sqrt(TINY):
        ialpha = 0
    else:
        if energy < 0.0:
            ialpha = -1
        else:
            ialpha = +1

    # Depending on the conic type, determine the remaining elements
    if ialpha == -1:
        a = -0.5*gm/energy  
        fac = 1.0 - h2/(gm*a)

        if fac > TINY:
            e = sqrt(fac)
            face = (a-absr)/(a*e)

            if face > 1.0:
               cape = 0.0
            else:
                if face > -1.0:
                   cape = arccos(face)
                else:
                   cape = pi

            if vdotr < 0.0:
                cape = 2.0*pi - cape

            cw = (cos(cape) - e)/(1.0 - e*cos(cape))
            sw = sqrt(1.0 - e*e)*sin(cape)/(1.0 - e*cos(cape))
            w = arctan2(sw, cw)

            if w < 0.0:
                w += 2.0*pi

        else:
            e = 0.0
            w = u
            cape = u

        capm = cape - e*sin(cape)
        omega = u - w
        if omega < 0.0:
            omega += 2.0*pi
        omega = omega - int(omega/(2.0*pi))*2.0*pi

    else:
        raise NotImplementedError

    elmts = array([a, e, inc, capom, omega, capm])

    return elmts

def get_euler(gm, elmts):
    """
    Get corresponding Euler angles.

    Note: Module variable (cape) is used from previous computation in orbel_el2xv().

    """
    global cape

    a, e, inc, capom, omega, capm = elmts

    theta = 2.0*arctan(sqrt((1.0+e)/(1.0-e))*tan(cape/2.0))
    
    euler = array([0.0, 0.0, 0.0])
    euler[0] = theta + omega
    euler[1] = capom
    euler[2] = inc

    cape = None

    return euler

def get_roche(gm, elmts):
    """
    Get corresponding Roche parameters.

    Note: Module variable (absr) is used from previous computation in orbel_el2xv().

    Note: The division by rotperiod is done elsewhere...

    """
    global absr

    a, e, inc, capom, omega, capm = elmts

    n = sqrt(gm/a**3)
    P = 2.*pi/n

    roche = array([0.0, 0.0])
    roche[0] = absr/a
    roche[1] = P

    absr = None

    return roche

if __name__ == "__main__":

    day = 86400.
    au = 1.496e11
    M_S = 2.e30
    G = 6.67e-11
    gms = G*M_S / (au**3 * day**-2)

    r = array([0.9, 0.0, 0.0])
    v = array([0.0, 1.9066428749416830523700e-02, 0.0])

    print("gms = ", gms)
    print("r = ", r)
    print("v = ", v)

    elmts = orbel_xv2el(gms, r, v)
    
    print("elmts = ", elmts)


