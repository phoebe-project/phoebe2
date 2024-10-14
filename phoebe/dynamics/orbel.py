
from numpy import pi, sin, cos, tan, arccos, arctan, arctan2, sqrt, dot, cross, array

TINY = 4.0e-15
TWOPI = 2.0*pi

cape = None
absr = None

def orbel_el2xv(gm, ialpha, elmts):
    """
    Computes cartesian positions and velocities given orbital elements.

    Rewritten from swift/orbel/orbel_el2xv.f.

    Reference: Levison, H.F., Duncan, M.J., The long-term dynamical behavior of short-period comets. Icarus 108, 18-36, 1994.

    """
    global cape

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

        cape = orbel_ehie(e, capm)
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

def get_euler(elmts):
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
    NMAX = 3
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


