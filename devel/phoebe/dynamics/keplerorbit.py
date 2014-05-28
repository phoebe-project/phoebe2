# -*- coding: utf-8 -*-
"""
Properties of Keplerian orbits.

Section 1 Introduction
======================

This module computes Keplerian orbits, and provides convenience functions to
work with them or derive additional parameters.

Part of the convenience functions are meant to be able to deal with
hierarchical multiple systems.

Be B{careful} with the inclination angle: if you only want an image projected
onto the sky, it is not case-sensitive. Only for the radial velocities this
becomes important (the primary and secondary can be mixed).

The geometry of an orbit is generally constrained by three angles:

    - B{theta}: angle measured from the angle of periastron C{per0}, gives the position along the orbit in the orbital plane
    - B{incl}: inclination angle (90 degrees means edge on, 0 degrees means the orbital plane is coplanar with the plane of the sky
    - B{Omega}: the longitude of periastron C{long_an} measures the angle on the sky
  

B{Absolute times} are acquired via the specification of the B{zeropoint} C{t0},
which is the time of periastron passage. The coordinate system is defined such
that at the time of periastron passage, the primary component is eclipsed if
the argument of periastron is 90 degrees. For circular orbits, the argument
of periastron should always be 90 degrees, such that for circular orbits, the
time of periastron passage coincides with the primary star being eclipsed.
Note that we do not strictly enforce this latter convention for the sake of
generality in this module, it is the user's responsability (or higher level
wrappers) to make sure that the argument of periastron is 90 degrees for a
circular orbit.



Section 2 Binary systems
========================

Section 2.1 Time conventions
----------------------------

As stated above, C{t0} is the time of periastron passage. To illustrate these,
let's define some random orbital parameters:

>>> period = 10.
>>> ecc = 0.8
>>> sma1 = 1.
>>> sma2 = 2.
>>> t0 = 1.2345
>>> per0 = 1.2*np.pi
>>> long_an = 0.
>>> incl = 70/180.*np.pi

And compute the whole orbit, for the primary and secondary:

>>> times = np.linspace(t0,t0+period,1000)
>>> pos1,velo1,euler1 = get_orbit(times,period,ecc,sma1,t0,per0=per0,
...                       long_an=long_an,incl=incl,component='primary')
>>> pos2,velo2,euler2 = get_orbit(times,period,ecc,sma2,t0,per0=per0,
...                       long_an=long_an,incl=incl,component='secondary')

Critical times (converted from phases) can be computed as well:

>>> crit_times = calculate_critical_phases(per0,ecc)*period + t0
>>> cpos1,velo1,euler1 = get_orbit(crit_times,period,ecc,sma1,t0,per0=per0,
...                       long_an=long_an,incl=incl,component='primary')
>>> cpos2,velo2,euler2 = get_orbit(crit_times,period,ecc,sma2,t0,per0=per0,
...                       long_an=long_an,incl=incl,component='secondary')

Now put all these things together in a plot:

>>> p = pl.figure()
>>> p = pl.subplot(121,aspect='equal')
>>> p = pl.title("Line-of-sight")
>>> p = pl.plot(pos1[0],pos1[1],'k-',lw=2,label='Primary')
>>> p = pl.plot(pos2[0],pos2[1],'r-',lw=2,label='Secondary')
>>> p = pl.plot([cpos1[0][0],cpos2[0][0]],[cpos1[1][0],cpos2[1][0]],'bo-',ms=8,label='Periastron passage')
>>> p = pl.plot([cpos1[0][1],cpos2[0][1]],[cpos1[1][1],cpos2[1][1]],'go-',ms=8,label='Superior conjunction')
>>> p = pl.plot([cpos1[0][2],cpos2[0][2]],[cpos1[1][2],cpos2[1][2]],'co-',ms=8,label='Inferior conjunction')
>>> p = pl.plot([cpos1[0][3],cpos2[0][3]],[cpos1[1][3],cpos2[1][3]],'mo-',ms=8,label='Ascending phase')
>>> p = pl.plot([cpos1[0][4],cpos2[0][4]],[cpos1[1][4],cpos2[1][4]],'yo-',ms=8,label='Descending phase')
>>> p = pl.grid()
>>> ax = pl.subplot(122,aspect='equal')
>>> p = pl.title('Top view')
>>> p = pl.plot(pos1[0],pos1[2],'k-',lw=2,label='Primary (front)')
>>> p = pl.plot(pos1[0],pos1[2],'k-',label='Primary (back)')
>>> p = pl.plot(pos2[0],pos2[2],'r-',lw=2,label='Secondary')
>>> p = pl.plot(pos2[0],pos2[2],'r-',lw=2)
>>> p = pl.plot([cpos1[0][0],cpos2[0][0]],[cpos1[2][0],cpos2[2][0]],'bo-',ms=8,label='Periastron passage')
>>> p = pl.plot([cpos1[0][1],cpos2[0][1]],[cpos1[2][1],cpos2[2][1]],'go-',ms=8,label='Superior conjunction')
>>> p = pl.plot([cpos1[0][2],cpos2[0][2]],[cpos1[2][2],cpos2[2][2]],'co-',ms=8,label='Inferior conjunction')
>>> p = pl.plot([cpos1[0][3],cpos2[0][3]],[cpos1[2][3],cpos2[2][3]],'mo-',ms=8,label='Ascending phase')
>>> p = pl.plot([cpos1[0][4],cpos2[0][4]],[cpos1[2][4],cpos2[2][4]],'yo-',ms=8,label='Descending phase')
>>> p = ax.annotate('To the observer',xy=(1,1),xytext=(1,2.1),arrowprops=dict(facecolor='k',arrowstyle='<-'),ha='center')
>>> p = pl.ylim(pl.ylim()[::-1])
>>> p = pl.legend(loc='best').get_frame().set_alpha(0.5)
>>> p = pl.grid()

]include figure]]images/orbit_time.png]

Section 2.1 Evolution of orbital parameters
-------------------------------------------

What follows is an example of how to allow for an evolution the argument of
periastron. First, some parameters need to be defined.

>>> period = 5.741046 # period in days
>>> ecc = 0.7 # eccentricity
>>> per0,long_an,incl = 1.20,238., 42.75 # degrees
>>> t0 = 51194.6239 # year in days
>>> plx = 7.19 # mas
>>> M1,M2 = 1.58,0.236 # Msun
>>> dpdt = 0#3.64e-12
>>> dperdt = 1. # deg/d
>>> times = np.linspace(t0,t0+10*period,10000)

Use Kepler's third law to derive the semi-major axis of the absolute orbits
(the semi-major axis of the relative (or system) orbit is a1+a2).

>>> sma = third_law(totalmass=(M1+M2), period=period)*constants.au
>>> sma2 = sma / (1.0+M2/M1)
>>> sma1 = sma-sma2

Convert the angles to radians and times to seconds, to pass them to L{get_orbit}:

>>> per0,long_an,i = per0/180*pi,long_an/180*pi,incl/180*pi
>>> dperdt = conversions.convert('deg/d','rad/s',dperdt)
>>> times *= 24*3600
>>> period *= 24*3600
>>> t0 *= 24*3600

Finally compute the orbits of the primary and secondary in sky coordinates.

>>> pos1,velo1,euler1 = get_orbit(times,period,ecc,sma1,t0,per0=per0,
...                       long_an=long_an,incl=incl,dperdt=dperdt,dpdt=dpdt,
...                       component='primary')
>>> pos2,velo2,euler2 = get_orbit(times,period,ecc,sma2,t0,per0=per0,
...                       long_an=long_an,incl=incl,dperdt=dperdt,dpdt=dpdt,
...                       component='secondary')


It is also allowed to change the value of the period linearly in time. There
is some ambiguity here, however, because if the orbital period changes, either
the semi-major axis or the mass must change too, to satisfy Kepler's third
law. Therefore, there is an extra keyword argument called C{mass_conservation},
which is either C{True} (semi-major axis will change) or C{False} (semi-major
axis will stay the same). In this example, we immediately also show how to
use predefined Phoebe parameterSets to compute things. We can first define
some kind of ParameterSet:

>>> from pyphoebe.parameters import parameters
>>> ps = parameters.ParameterSet(context='orbit',period=100,sma=100,incl=0.)

After defining a time array, we can then set the period change, and see what
the influence of mass conservation is:

>>> times = np.linspace(ps['t0'],ps['t0']+5*ps['period'],10000)*24*3600
>>> loc1,velo1,euler1 = parse_ps(get_orbit,ps,times,dpdt=-1e-1)
>>> loc2,velo2,euler2 = parse_ps(get_orbit,ps,times,dpdt=-1e-1,mass_conservation=False)

And make a plot:

>>> p = pl.figure(figsize=(12,6))
>>> p = pl.subplot(121,aspect='equal')
>>> p = pl.title("Evolution of Argument of Periastron")
>>> p = pl.plot(pos1[0]/constants.Rsol,pos1[1]/constants.Rsol,'k-')
>>> p = pl.plot(pos2[0]/constants.Rsol,pos2[1]/constants.Rsol,'r-')
>>> p = pl.grid()
>>> p = pl.xlabel('X [Rsol]')
>>> p = pl.ylabel('Y [Rsol]')
>>> p = pl.subplot(122,aspect='equal')
>>> p = pl.title("Evolution of Period")
>>> p = pl.plot(loc1[0]/constants.Rsol,loc1[1]/constants.Rsol,'k-',label='Mass conservation')
>>> p = pl.plot(loc2[0]/constants.Rsol,loc2[1]/constants.Rsol,'r-',label='No mass conservation')
>>> p = pl.legend(loc='best').get_frame().set_alpha(0.5)
>>> p = pl.grid()
>>> p = pl.xlabel('X [Rsol]')
>>> p = pl.ylabel('Y [Rsol]')

]include figure]]images/orbit_evolution.png]


Section 2.2 Light time travel effects
-------------------------------------


Section 2.3 Real world examples
-------------------------------

Example 1: mu Cassiopeia
~~~~~~~~~~~~~~~~~~~~~~~~

We compute the absolute orbits of the two components of mu Cassiopeia, and
compute the relative orbit of the second component.

See ps file on http://www.chara.gsu.edu/~gudehus/binary.html and Drummond et al.,
1995 (the latter's orbital parameters are used).

Define the orbital elements, Euler angles and masses of the system

>>> period = 21.753*365 # period in days
>>> ecc = 0.561 # eccentricity
>>> per0,long_an,incl = 332.7, 47.3, 106.8 # degrees
>>> t0 = 1975.738*365 # year in days
>>> plx = 133.2 # parallax in mas
>>> M1,M2 = 0.719,0.168 # mass in Msol
>>> times = np.linspace(t0,t0+0.95*period,100) # 95% of the orbit

Use Kepler's third law to derive the semi-major axis of the absolute orbits
(the semi-major axis of the relative (or system) orbit is a1+a2).

>>> sma = third_law(totalmass=(M1+M2), period=period)*constants.au
>>> sma2 = sma / (1.0+M2/M1)
>>> sma1 = sma-sma2

Convert the angles to radians to pass them to L{get_orbit}:

>>> per0,long_an,incl = per0/180.*pi,long_an/180.*pi,incl/180.*pi

Finally compute the orbits of the primary and secondary in sky coordinates.

>>> pos1,velo1,euler1 = get_orbit(times,period,ecc,sma1,t0,per0=per0,\
                        long_an=long_an,incl=incl,component='primary')
>>> pos2,velo2,euler2 = get_orbit(times,period,ecc,sma2,t0,per0=per0,\
                        long_an=long_an,incl=incl,component='secondary')

Convert these coordinates (returned in SI units) to solar radii:

>>> distance = (1.0/plx,'kpc')
>>> pos1 = np.array(pos1)/constants.Rsol
>>> pos2 = np.array(pos2)/constants.Rsol
>>> velo1 = np.array(velo1)/constants.Rsol
>>> velo2 = np.array(velo2)/constants.Rsol

And those on its turn to angular coordinates on the sky via
:py:func:`truecoords_to_spherical`, since the coordinates were returned in the
Cartesian framework.

>>> ra1,dec1 = truecoords_to_spherical(pos1.T,distance=distance)
>>> ra2,dec2 = truecoords_to_spherical(pos2.T,distance=distance)

Convert radians to arcseconds:

>>> ra1,dec1 = conversions.convert('rad','as',ra1),conversions.convert('rad','as',dec1)
>>> ra2,dec2 = conversions.convert('rad','as',ra2),conversions.convert('rad','as',dec2)

Plot the two true orbits and their relative orbit:

>>> x1,y1,z1 = pos1
>>> x2,y2,z2 = pos2
>>> vx1,vy1,vz1 = velo1
>>> vx2,vy2,vz2 = velo2

>>> import pylab as pl
>>> colors = times/365.
>>> p = pl.figure()
>>> p = pl.title('mu Cassiopeia')
>>> p = pl.subplot(111,aspect='equal')
>>> p = pl.scatter(ra1,dec1,c=colors,edgecolors='none',cmap=pl.cm.spectral)
>>> p = pl.scatter(ra2,dec2,c=colors,edgecolors='none',cmap=pl.cm.spectral)
>>> p = pl.scatter(ra2-ra1,dec2-dec1,c=colors,edgecolors='none',cmap=pl.cm.spectral)
>>> p = pl.plot(ra2-ra1,dec2-dec1,'r-',label='Secondary relative')
>>> p = pl.colorbar()
>>> p = pl.legend(loc='lower right')
>>> p = pl.grid()
>>> p = pl.xlabel(r'Angular size (arcsec) --> N')
>>> p = pl.ylabel(r'Angular size (arcsec) --> E')
>>> p = pl.xlim(-0.5,1.5)
>>> p = pl.ylim(-0.5,1.5)

]include figure]]images/muCas_comparison.png]


Example 2: Capella
~~~~~~~~~~~~~~~~~~

Compute the absolute orbits of the two components of
Capella, and compute the relative orbit of the primary component.

See Hummel et al. 1994.

Neccesary imports:

>>> from phoebe.units.constants import *

Orbital elements, Euler angles and masses of the system

>>> period = 104.0233 # period in days
>>> ecc = 0.000 # eccentricity
>>> incl = 137.18 # degree
>>> t0 = 2447528.45 # year
>>> per0 = 0. # argument of periastron in degrees
>>> long_an = 40.8 # longitude of ascending node in degrees
>>> plx = 76.20 # parallax in mas
>>> M1 = 2.69 # mass in Msol
>>> M2 = 2.56 # mass in Msol
>>> RV0 = 0.
>>> times = np.linspace(t0,t0+0.95*period,100)

Use Kepler's third law to derive the semi-major axis of the absolute orbits
(the semi-major axis of the relative (or system) orbit is a1+a2).

>>> sma = third_law(totalmass=(M1+M2), period=period)*constants.au
>>> sma2 = sma / (1.0+M2/M1)
>>> sma1 = sma-sma2

Convert the angles to radians

>>> per0,long_an,incl = per0/180.*pi,long_an/180.*pi,incl/180.*pi

Finally compute the orbits of the primary and secondary in sky coordinates.

>>> pos1,velo1,euler1 = get_orbit(times,period,ecc,sma1,t0,per0=per0,\
                        long_an=long_an,incl=incl,component='primary')
>>> pos2,velo2,euler2 = get_orbit(times,period,ecc,sma2,t0,per0=per0,\
                        long_an=long_an,incl=incl,component='secondary')

Convert these coordinates (returned in SI units) to solar radii:

>>> distance = (1.0/plx,'kpc')
>>> pos1 = np.array(pos1)/constants.Rsol
>>> pos2 = np.array(pos2)/constants.Rsol
>>> velo1 = np.array(velo1)/constants.Rsol
>>> velo2 = np.array(velo2)/constants.Rsol

And those on its turn to angular coordinates on the sky, since the are returned
as cartesian coordinates.

>>> ra1,dec1 = truecoords_to_spherical(pos1.T,distance=distance)
>>> ra2,dec2 = truecoords_to_spherical(pos2.T,distance=distance)

Convert radians to milliarcseconds:

>>> ra1,dec1 = conversions.convert('rad','mas',ra1),conversions.convert('rad','mas',dec1)
>>> ra2,dec2 = conversions.convert('rad','mas',ra2),conversions.convert('rad','mas',dec2)

Plot the two true orbits and their relative orbit:

>>> x1,y1,z1 = pos1
>>> x2,y2,z2 = pos2
>>> vx1,vy1,vz1 = velo1
>>> vx2,vy2,vz2 = velo2

>>> import pylab as pl
>>> colors = times/365.
>>> p = pl.figure()
>>> p = pl.title('Capella')
>>> p = pl.subplot(111,aspect='equal')
>>> p = pl.scatter(ra1,dec1,c=colors,edgecolors='none',cmap=pl.cm.spectral)
>>> p = pl.scatter(ra2,dec2,c=colors,edgecolors='none',cmap=pl.cm.spectral)
>>> p = pl.scatter(ra1-ra2,dec1-dec2,c=colors,edgecolors='none',cmap=pl.cm.spectral)
>>> p = pl.plot(ra1-ra2,dec1-dec2,'r-',label='Secondary relative')
>>> p = pl.colorbar()
>>> p = pl.legend(loc='lower right')
>>> p = pl.grid()
>>> p = pl.xlabel(r'Angular size (arcsec) --> N')
>>> p = pl.ylabel(r'Angular size (arcsec) --> E')
>>> p = pl.xlim(60,-60)
>>> p = pl.ylim(-60,60)

]include figure]]images/capella_comparison.png]


Section 3 Multiple systems
==========================

Define three orbits: two narrow ones, and one wide one.

>>> from pyphoebe.parameters import parameters
>>> x1 = parameters.ParameterSet(context='orbit',label='wide',period=100,sma=100,incl=45.,t0=20)
>>> x2 = parameters.ParameterSet(context='orbit',label='xnarrow',period=5,sma=5,incl=45.,t0=0)
>>> x3 = parameters.ParameterSet(context='orbit',label='narrow',period=30,sma=30,incl=45.,t0=10)

Place the two narrow ones in the wider one.

>>> sysA = {x2:[True,True]}
>>> sysB = {x3:[True,True]}
>>> system = {x1:[sysA,sysB]}

Compute the orbits at specific times:

>>> times = np.linspace(0,100*24*3600,1000)
>>> output = get_hierarchical_orbits(times,system)

Plot the orbits and the radial velocities:

>>> p = pl.figure(figsize=(12,6))
>>> p = pl.subplot(121,aspect='equal')
>>> for body in output: p = pl.plot(body[0][0]/constants.Rsol,body[0][1]/constants.Rsol,'-',lw=2)
>>> p = pl.grid()
>>> p = pl.xlabel('X [Rsol]')
>>> p = pl.ylabel('Y [Rsol]')
>>> p = pl.subplot(122)
>>> for body in output: p = pl.plot(times/(24*3600),body[1][2]/1000.,'-',lw=2)
>>> p = pl.grid()
>>> p = pl.xlabel('Time [d]')
>>> p = pl.ylabel('Radial velocity [km/s]')

]include figure]]images/hierarchical_orbit_quadruple.png]


"""
import itertools
import functools
import inspect
import logging
import numpy as np
from numpy import pi,sqrt,cos,sin,tan,arctan
from scipy.optimize import newton,bisect, fmin, brentq
from phoebe.units import constants
from phoebe.units import conversions
from phoebe.utils import cgeometry
from phoebe.utils import fgeometry
from phoebe.dynamics import ftrans
from phoebe.dynamics import ctrans

logger = logging.getLogger('BINARY.ORBIT')

default_polar_dir = np.array([0,0,-1.0])
default_los_dir = np.array([0,0,+1.0])
default_zeros = np.array([0,0,0.0])

#{ General orbits    

def get_orbit(times, period, ecc, sma, t0, per0=0., long_an=0., incl=0.,
              dpdt=0., deccdt=0., dperdt=0., mass_conservation=True,
              component='primary', t0type='periastron passage'):
    r"""
    Construct an orbit in observer coordinates.
        
    Argument ``Parameters`` contains:
        1. period (s)
        2. eccentricity
        3. component semi-major axis (not system semi-major axis! see below
           for conversion formula from system to component semi-major axis) (m)
        4. time of periastron passage :math:`t_0` (not :math:`x_0`!) (s)
        
    Optional parameters for orientation in 3D:
        5. argument of periastron (radians) (:math:`\omega`)
        6. longitude of ascending node (radius) (:math:`\Omega`)
        7. inclination (radians) (:math:`i`)
    
    Optional parameters for time evolution of period and argument of
    periastron:
    
        8. dpdt: allow for a change of period (s/s). At :math:`t_0`, the period (P) is
           C{period}. When C{mass_conservation=True}, mass will be conserved
           which means that also the semi-major axis (a) will change according
           to:
            
           .. math::
            
              a(t) = \frac{a_0}{P_0^2} P^2(t)
        
        9. dperdt: allow for a change of argument of periastron (rad/s)
        10. deccdt: allow for a change of eccentricity (:math:`\mathrm{s}^{-1}`)
        
    By a freak-accident of nature, this function is purely geometrical and
    does not use any physical constants. This means that you are in principle
    allowed to B{pass any units} to this function, as long as you're
    consistent. If you want to give C{times} in days, go ahead, but also set
    the C{period} and C{t0} in days. The velocities should then be in m/days.
    Use at your own risk!
    
    See Hilditch p41 for a sketch of the coordinates. The difference with this
    is that all angles are inverted. In this approach, North is in the
    positive X direction, East is in the negative Y direction (?).
    
    The system semi-major axis and component semi-major axes are connected to 
    the mass ratio :math:`q=M_2/M_1` via:
    
    .. math::
    
        a_1 = \frac{a}{1 + \frac{1}{q}} \\
        a_2 = \frac{a}{1 + q}
    
    See also: :py:func:`get_binary_orbit`
    
    @param times: times of observations (s)
    @type times: array
    @param period: orbital period (s)
    @type period: float
    @param ecc: eccentricity
    @type ecc: float
    @param sma: component semi-major axis (m)
    @type sma: float
    @param t0: time of periastron passage (HJD)
    @type t0: float
    @param per0: argument of periastron (radian)
    @type per0: float
    @param long_an: longitude of ascending node (radian)
    @type long_an: float
    @param incl: inclination angle (radian)
    @type incl: float
    @param dpdt: linear period shift (s/s)
    @type dpdt: float
    @param deccdt: linear eccentricity shift
    @type deccdt: ecc/s
    @param dperdt: linear periastron shift (rad/s)
    @type dperdt: float
    @param mass_conservation: if C{True}, the semi-major axis will change if
                              the period changes
    @type mass_conservation: bool
    @param component: component to calculate the orbit of. If it's the secondary,
                      the angles will be shifted by 180 degrees
    @type component: string, one of 'primary' or 'secondary'
    @return: position vector, velocity vector, Euler angles
    @rtype: 3-tuple, 3-tuple, 3-tuple
    """
    # If t0 is not the time of periastron passage, convert it:
    if t0type == 'superior conjunction':
        phshift = 0.
        t0 = from_supconj_to_perpass(t0, period, per0, phshift=phshift)
    elif not t0type == 'periastron passage':
        raise ValueError('t0type needs to be one of "superior conjunction" or "periastron passage"')
    
    #-- if dpdt is non-zero, the period is actually an array, and the semi-
    #   major axis changes to match Kepler's third law (unless
    #   `mass_conservation` is set to False)
    if dpdt!=0:
        period_ = period
        period = dpdt*(times-t0) + period_
        if mass_conservation and not np.isscalar(period):
             sma = sma/period[0]**2*period**2
        elif mass_conservation:
             sma = sma/period_**2*period**2
    #-- if dperdt is non-zero, the argument of periastron is actually an
    #   array
    if dperdt!=0.:
        per0 = dperdt*(times-t0) + per0
    #-- if deccdt is non-zero, the eccentricity is actually an array
    if deccdt!=0:
        ecc = deccdt*(times-t0) + ecc
    #-- compute orbit
    n = 2*pi/period
    ma = n*(times-t0)
    E,theta = true_anomaly(ma,ecc)
    r = sma*(1-ecc*cos(E))
    PR = r*sin(theta)
    #-- compute rdot and thetadot
    l = r*(1+ecc*cos(theta))#-omega))
    L = 2*pi*sma**2/period*sqrt(1-ecc**2)
    rdot = L/l*ecc*sin(theta)#-omega)
    thetadot = L/r**2
    #-- the secondary is half an orbit further than the primary
    if 'sec' in component.lower():
        theta += pi
    #-- take care of Euler angles
    theta_ = theta+per0
    #-- convert to the right coordinate frame
    #-- create some shortcuts
    sin_theta_ = sin(theta_)
    cos_theta_ = cos(theta_)
    sin_longan = sin(long_an)
    cos_longan = cos(long_an)
    #-- spherical coordinates to cartesian coordinates. Note that we actually
    #   set incl=-incl (but of course it doesn't show up in the cosine). We
    #   do this to match the convention that superior conjunction (primary
    #   eclipsed) happens at periastron passage when per0=90 deg.
    x = r*(cos_longan*cos_theta_ - sin_longan*sin_theta_*cos(incl))
    y = r*(sin_longan*cos_theta_ + cos_longan*sin_theta_*cos(incl))
    z = r*(sin_theta_*sin(-incl))
    #-- spherical vectors to cartesian vectors, and then rotated for
    #   the Euler angles Omega and i.
    vx_ = cos_theta_*rdot - sin_theta_*r*thetadot
    vy_ = sin_theta_*rdot + cos_theta_*r*thetadot
    vx = cos_longan*vx_ - sin_longan*vy_*cos(incl) 
    vy = sin_longan*vx_ + cos_longan*vy_*cos(incl)
    vz = sin(-incl)*vy_
    #-- that's it!
    return (x,y,z),(vx,vy,vz),(theta_,long_an,incl)
    

def get_barycentric_orbit(bary_times, *args, **kwargs):
    """
    Construct on orbit in observer coordinates corrected for light LTT effects.
    
    @param bary_times: barycentric time coordinates
    @type bary_times: float or array
    @return: position vector, velocity vector, Euler angles, proper times
    @rtype: 3-tuple, 3-tuple, 3-tuple, array/float
    """
    #-- We need to set the location of the object such that it light arrives
    #   simultaneously with the barycentric time. Only the direction in the
    #   line-of-sight is important (z-coordinate). The time it takes for light
    #   to travel to us, relatively, to the barycentre, is z/cc [sec]. The
    #   time which we thus see the light from that object is t+z/cc, but we
    #   need it to be t_bary. Thus, we need to optimize it's location (i.e.
    #   proper time) until the observed time is the barycentric time.
    def propertime_barytime_residual(t):
        pos, velo, euler = get_orbit(t, *args, **kwargs)
        z = pos[2]
        #return t + z/constants.cc/(24*3600) - t_bary
        return t - z/constants.cc - t_bary
    #-- Finding that right time is easy with a Newton optimizer:
    propertimes = [newton(propertime_barytime_residual, t_bary) for t_bary in bary_times]
    propertimes = np.array(propertimes)
    #-- then make an orbit with these times!
    this_orbit = get_orbit(propertimes, *args, **kwargs)
    return list(this_orbit) + [propertimes]


def correct_barycentric_orbit(obs_times, *args, **kwargs):
    """
    Correct observed timings to be in the frame of reference of a component.
    
    This can be useful to e.g. analyse pulsations in the proper frame of
    reference
    
    In a way the reverse of :py:func:`get_barycentric_orbit`.
    
    @param obs_times: observed time coordinates
    @type obs_times: float or array
    @return: position vector, velocity vector, Euler angles, proper times
    @rtype: 3-tuple, 3-tuple, 3-tuple, array/float
    """
    def propertime_barytime_residual(t_bary):
        pos, velo, euler = get_orbit(t_bary, *args, **kwargs)
        z = pos[2]
        #return t + z/constants.cc/(24*3600) - t_bary
        return t_bary + z/constants.cc - t
    #-- Finding that right time is easy with a Newton optimizer:
    propertimes = [newton(propertime_barytime_residual, t) for t in obs_times]
    propertimes = np.array(propertimes)
    #-- then make an orbit with these times!
    this_orbit = get_orbit(propertimes, *args, **kwargs)
    return list(this_orbit) + [propertimes]


def get_hierarchical_orbit(times,orbits,comps):
    """
    Retrieve the orbit of one components in a hierarchical system.
    
    Velocity is wrong!
    
    @param times: time array
    @type times: array
    @param orbits: list of ParameterSets in the C{phoebe} frame and C{orbit}
    context
    @type orbits: list of ParameterSets of length N
    @param comps: list of integers denoting which component (0 for primary,
    1 for secondary) the object is for each orbit
    @type comps: list of integers
    @return: position vectors, velocity vectors
    @rtype: 3-tuple, 3-tuple
    """
    if np.isscalar(times):
        obj = np.zeros((1,3))
        vel = np.zeros((1,3))
    else:
        obj = np.zeros((len(times),3))
        vel = np.zeros((len(times),3))
    for i,(orbit,comp) in enumerate(list(zip(orbits,comps))):
        #-- retrieve the coordinates and euler angles for this orbit
        comp = ['primary','secondary'][comp]
        loc,velo,euler = parse_ps(get_orbit,orbit,times,component=comp)        
        #-- and incrementally add it to the present object
        obj[:,0] += loc[0].ravel()
        obj[:,1] += loc[1].ravel()
        obj[:,2] += loc[2].ravel()
        vel[:,0] += velo[0].ravel()
        vel[:,1] += velo[1].ravel()
        vel[:,2] += velo[2].ravel()
    return obj.T,vel.T
    

def get_barycentric_hierarchical_orbit(bary_times, orbits, components, barycentric=True):
    r"""
    Construct a hiearchical orbit corrected for light LTT effects.
    
    Light time travel effects are included by finding the system's proper time
    :math:`t` such that the (oberved) barycentric time :math:`t_\mathrm{bary}`
    is equal to:
    
    .. math::
    
        t_\mathrm{bary} = t + \frac{z(t)}{c},
        
    with :math:`c` the velocity of light. and :math:`z(t)` the location of the
    Body in the :math:`z` direction (radial direction) at time :math:`t`. We
    can't determine :math:`z` from this function algebraically because :math:`z`
    is a function of :math:`t`. The solution is to do a minimization of the function
    
    .. math::
    
        f(t) = t + \frac{z(t)}{c} - t_\mathrm{bary}
    
    via a simple Newton-Raphson procedure.
    
    Note that in Phoebe, :math:`-z` is in the direction of the observer, thus
    in Phoebe internal coordinates, you need to replace :math:`z` in the above
    equations with :math:`-z`.
    
    Get orbits and components via :py:func:`Body.get_orbits <phoebe.backend.universe.Body.get_orbits>`
    from the :py:mod:`phoebe.backend.universe` module.
    
    @param bary_times: barycentric time coordinates
    @type bary_times: float or array
    @param orbits: list of ParameterSets in the C{phoebe} frame and C{orbit} context
    @type orbits: list of ParameterSets of length N
    @param components: list of integers denoting which component (0 for primary, 1 for secondary) the object is for each orbit
    @type components: list of integers
    @return: proper times, position vector, velocity vector, Euler angles
    @rtype: 3-tuple, 3-tuple, array/float
    """
    bary_times = np.asarray(bary_times)
    
    # We need to set the location of the object such that it light arrives
    # simultaneously with the barycentric time. Only the direction in the
    # line-of-sight is important (z-coordinate). The time it takes for light to
    # travel to us, relatively, to the barycentre, is z/cc [sec]. The time which
    # we thus see the light from that object is t+z/cc, but we need it to be
    # t_bary. Thus, we need to optimize it's location (i.e. proper time) until
    # the observed time is the barycentric time.
    # Because -z is in the line of sight (instead of +z), we set -z below.
    scale_factor = 1.0/constants.cc * constants.Rsol/(24*3600.)
    def propertime_barytime_residual(t):
        obj,vel = get_hierarchical_orbit_phoebe(t, orbits, components)
        z = obj[2,0]
        return t - z*scale_factor - t_bary
    
    # Finding that right time is easy with a Newton optimizer:
    propertimes = [newton(propertime_barytime_residual, t_bary) for \
                       t_bary in bary_times] if barycentric else bary_times
    propertimes = np.array(propertimes).ravel()
    
    # then make an orbit with these times!
    this_orbit = get_hierarchical_orbit_phoebe(bary_times, orbits, components)
    return list(this_orbit) + [propertimes]
    

def get_hierarchical_orbits(times,system,barycentric=False):
    """
    Retrieve the orbits of all components in a hierarchical system.
    
    Passing parameters of a hierarchical system can be quite tricky. To
    simplify matters, we choose to use ParameterSets in dictionaries. Each
    orbit is one dictionary with one key. The value corresponding to that
    key is a list of two objects, where each object is a component. That
    component can itself again be an orbit. The key is of the dictionary is
    not a string, but a ParameterSet itself (see L{walk_hierarchical}).
    
    @return: list of position and velocity vectors for each object in the
    system. If B{barycentric=True}, also the proper times are returned
    @rtype: list of tuples
    """    
    output = []
    for i,(this,orbits,comps) in enumerate(walk_hierarchical(system)):
        if this is None: continue
        myorbits = orbits[::-1]
        mycompon = (comps + [this])[::-1]
        if barycentric:
            this_out = get_barycentric_hierarchical_orbit(times,myorbits,mycompon)
        else:
            this_out = get_hierarchical_orbit(times,myorbits,mycompon)
        output.append(this_out)
    return output



def rotate_into_orbit(obj, euler, loc=(0,0,0)):
    r"""
    Rotate something to a position in an orbit.
    
    The rotation goes like:
    
    .. math::
    
        R_Z(\Omega) \cdot R_X(-i) \cdot R_Z(\theta+\pi)
    
    where :math:`R_X` is a rotation around the x-axis, :math:`i` is the inclination,
    :math:`\Omega` the longitude of the ascending node and :math:`\theta` the
    position in the orbit (:math:`\theta` equals :math:`\omega` at periastron
    where :math:`\omega` is argument of periastron).
    
    To rotate a set of coordinates, set C{loc} to match the Cartesian
    location of the point in the orbit.
    
    To rotate a vector, set C{loc} to the null vector.
    
    @param obj: coordinates or vectors to position in the orbit
    @type obj: 3-tuple (each entry can be an array)
    @param euler: Euler angles matching the position in the orbit
    @type euler: 3-tuple
    @param loc: translation vector
    @type loc: 3-tuple (each entry can be an array)
    @return: rotated and translated coordinates
    @rtype: 3xarray
    """
    #x, y, z = obj
    theta,longan,incl = euler

    #-- C version
    X_Y_Z = cgeometry.rotate_into_orbit(obj.copy(),euler,loc).reshape((3,-1))
    return X_Y_Z



def apparent_coordinates(distance, ra, dec, pmra, pmdec,
                         observer_position, target_position=None,
                         epoch='J2000', t0=None):
    r"""
    Compute apparent coordinates on the sky wrt to an observer.
    
    Observer position needs to be in ecliptic coordinates with respect to the
    Sun and need to span the total time span of the required times.
    (time, x, y, z).
    
    Method description:
    
        1. Compute ecliptic coordinates of the system at distance :math:`d_\odot`,
           centered on the Sun. This gives coordinates
           
           .. math::
              :label: ecl_coords
           
                (d_\odot, \lambda_\odot, \beta_\odot).
           
           It is possible that these coordinates are time dependent, e.g. in a
           binary system. If so, the argument ``target_position`` should contain
           the Cartesian coordinates of the object in its own barycentric
           coordinates, in units of solar radii (i.e. the natural units of Phoebe).
           Then these coordinates will be converted to spherical coordinates via
           :py:func:`truecoords_to_spherical`, and used as the RA and DEC for
           the object different times :math:`t_i`,
           
           .. math::
           
                (d_\odot(t_i), \lambda_\odot(t_i), \beta_\odot(t_i)).
           
           If only one coordinate tuple is given,
           ``target_position`` is effectively a constant offset from the system's
           RA and DEC.
           
           In Cartesian coordinates, coordinates :eq:`ecl_coords` become:
           
           .. math::
           
                x_\odot & = d_\odot \sin\left(\frac{\pi}{2}-\beta_\odot\right) \cos(\lambda_\odot)\\
                y_\odot & = d_\odot \sin\left(\frac{\pi}{2}-\beta_\odot\right) \sin(\lambda_\odot)\\
                z_\odot & = d_\odot \cos\left(\frac{\pi}{2}-\beta_\odot\right)
          
        2. Next, compute the ecliptic coordinates centered of the observer
           (denoted with symbol :math:`\oplus`). First,
           compute the positions of the observer :math:`(X,Y,Z)` relative to the
           Sun at each time point :math:`t_i` (e.g. via the
           :ref:`JPL Horizons interface <label-jpl>`). Then convert the heliocentric ecliptic
           coordinates to observer-centric ones:
           
           .. math::
                
                x_\oplus(t_i) & = x_\odot - X(t_i)\\
                y_\oplus(t_i) & = y_\odot - Y(t_i)\\
                z_\oplus(t_i) & = z_\odot - Z(t_i).
                
           Note that also the :math:`\odot` coordinates might be time dependent,
           in which case:
           
           .. math::
                
                x_\oplus(t_i) & = x_\odot(t_i) - X(t_i)\\
                y_\oplus(t_i) & = y_\odot(t_i) - Y(t_i)\\
                z_\oplus(t_i) & = z_\odot(t_i) - Z(t_i).
          
           In observer-centric spherical coordinates this becomes:
           
           .. math::
           
                d_\oplus(t_i)        & = \sqrt{x_\oplus^2 + y_\oplus^2 + z_\oplus^2}\\
                \lambda_\oplus(t_i)  & = \frac{\pi}{2} - \arccos\left(\frac{z_\oplus}{d_\oplus}\right)\\
                \beta_\oplus(t_i)    & = \arctan(y,x)\\
                
        Now we have the apparent ecliptic coordinates as observed by the observer.
        This is the parallactic motion in ecliptic coordinates. It will be flat
        near the equator and circular around the pole if the observer is located
        in the ecliptic plane (like the earth is). To make it angular, you have
        to correct for the declination:
        
            .. math::
            
                \lambda_{\oplus,P}(t_i) & = \lambda_\oplus(t_i)\cos(\beta_\oplus(t_i)) \\
                \beta_{\oplus,P}(t_i) & = \beta_\oplus(t_i)
    
    To test, try Vega (large parallax), Eta Dra (near ecliptic pole) and
    lambda Aqr (near ecliptic plane).
    
    .. _label-jpl:
    
    .. note:: JPL Horizons interface
    
        The `JPL Horizons interface <http://ssd.jpl.nasa.gov/horizons.cgi>`_ can
        give you the location of the observer,
        being a spacecraft or solar system body, in time. For example for an
        Earth orbiting observer (e.g. the Hipparcos satellite), the following
        settings might be of some use::
        
            Ephemeris Type [change] :    VECTORS
            Target Body [change] :       Earth [Geocenter] [399]
            Coordinate Origin [change] : Sun (body center) [500@10]
            Time Span [change] :         Start=1990-08-22, Stop=2013-09-21, Step=1 d
            Table Settings [change] :    quantities code=1; CSV format=YES
            Display/Output [change] :    plain text
        
        You can read in the output text file, assumed it is saved to a file
        called ``observer.dat``, with something like::
        
            >>> times, X, Y, Z = np.genfromtxt('observer.dat',
                     delimiter=',', skip_header=53, skip_footer=32,
                     usecols=(0,2,3,4), unpack=True)
    
    
        
    @param distance: distance to the object (parsec)
    @type distance: float
    @param ra: object's right ascension (degrees)
    @type ra: float
    @param dec: object's declination (degrees)
    @type dec: float
    @param pmra: right ascension proper motion (mas/yr)
    @type pmra: float
    @param pmdec: declination proper motion (mas/yr)
    @type pmdec: float
    @param observer_position: observer times (JD), ecliptic X,Y,Z coordinates (au)
    @type observer_position: 4-tuple arrays of length N
    @param target_position: target position, (x, y, z (Rsol))
    @type target_position: 3-tuple arrays of length N
    @param epoch: epoch for ra and pmra (e.g. J2000, B1950, J1991.25)
    @type epoch: str
    """
    # What's our reference time?
    if t0 is None:
        t0 = conversions.convert('epoch','JD',epoch)
    
    # Get the apparent position of the star wrt the earth.
    # Earth's position in ecliptic coordinates wrt Sun
    times, x_earth, y_earth, z_earth = observer_position
    times = times - t0
    
    # What's our target's position?
    if target_position is None:
        target_position = np.zeros((3,1))
    target_position = np.asarray(target_position).T
    ra_target, dec_target, d_target = truecoords_to_spherical(target_position,
                      distance=(distance,'pc'), origin=(ra/180.*pi,dec/180.*pi))
    
    # Retrieve info on proper motions
    pmra = conversions.convert('mas/yr','deg/d', pmra)
    pmdec = conversions.convert('mas/yr','deg/d', pmdec)
    lamref, betref = conversions.convert('equatorial','ecliptic',(ra/180.*pi,
                                          dec/180.*pi), epoch=epoch[1:])
    
    
    # Get the coordinates of the system in ecliptic coordinates
    lam, bet = np.array([list(conversions.convert('equatorial','ecliptic',\
                         (ira,idec), epoch=epoch[1:])) \
                         for ira,idec in zip(ra_target,dec_target)]).T
    
    #   Spherical coordinate position of star wrt the Sun
    d = d_target*constants.Rsol/constants.au*np.ones_like(times)
    
    #   Cartesian position of star wrt the Sun
    x = d * sin(pi/2-bet) * cos(lam)
    y = d * sin(pi/2-bet) * sin(lam)
    z = d * cos(pi/2-bet)
    
    #   Cartesian position of star wrt the Earth
    x_ = x - x_earth
    y_ = y - y_earth
    z_ = z - z_earth

    #   Spherical coordinate position of star wrt the Earth (apparent coordinates)
    r_ = np.sqrt(x_**2 + y_**2 + z_**2)
    app_beta = pi/2 - np.arccos(z_/r_)
    app_lambda = np.arctan2(y_, x_)
        
    ras, decs = np.zeros_like(app_lambda),np.zeros_like(app_lambda)
    for i,(ilam, ibet) in enumerate(zip(app_lambda, app_beta)):
        ras[i], decs[i] = conversions.convert('ecliptic','equatorial',(ilam, ibet),
                                              epoch=epoch[1:])

    # Account for proper motions
    decs = decs/pi*180 + pmdec*times 
    ras = ras/pi*180 + pmra*times/cos(decs/180.*pi) 
    
    # Construct parallax circle:
    lam_ = lam if lam<np.pi else lam-2*np.pi
    par_circ_lam = (app_lambda)/pi*180 * cos(app_beta) - lam_/pi*180*cos(app_beta)
    par_circ_bet = (app_beta)/pi*180 - bet/pi*180
    
    #parallax = np.abs((par_circ_lam).min()-(par_circ_lam).max())/2.0*3600*1000
    #print("Parallax = {:e}".format(parallax))    
    
    # all output is in degrees    
    delta_ra = ras - ra
    delta_ra = delta_ra*cos(decs/180.*pi) 
    output = dict(ra=ra, dec=dec, delta_ra=delta_ra, delta_dec=decs-dec,
                  plx_lambda=par_circ_lam, plx_beta=par_circ_bet)
    return output

#}

#{ Time and phases

def true_anomaly(M,ecc,itermax=8):
    r"""
    Calculation of true and eccentric anomaly in Kepler orbits.
    
    ``M`` is the phase of the star, ``ecc`` is the eccentricity
    
    See p.39 of Hilditch, 'An Introduction To Close Binary Stars':
    
    Kepler's equation:
    
    .. math::
    
        E - e\sin E = \frac{2\pi}{P}(t-T)
        
    with :math:`E` the eccentric anomaly. The right hand size denotes the
    observed phase :math:`M`. This function returns the true anomaly, which is
    the position angle of the star in the orbit (:math:`\theta` in Hilditch'
    book). The relationship between the eccentric and true anomaly is as
    follows:
    
    .. math::
    
        \tan(\theta/2) = \sqrt{\frac{1+e}{1-e}} \tan(E/2)
    
    @parameter M: phase
    @type M: float
    @parameter ecc: eccentricity
    @type ecc: float
    @keyword itermax: maximum number of iterations
    @type itermax: integer
    @return: eccentric anomaly (E), true anomaly (theta)
    @rtype: float,float
    """
    # Initial value
    Fn = M + ecc*sin(M) + ecc**2/2.*sin(2*M)
    
    # Iterative solving of the transcendent Kepler's equation
    for i in range(itermax):
        F = Fn
        Mn = F-ecc*sin(F)
        Fn = F+(M-Mn)/(1.-ecc*cos(F))
        keep = F!=0 # take care of zerodivision
        if hasattr(F,'__iter__'):
            if np.all(abs((Fn-F)[keep]/F[keep])<0.00001):
                break
        elif (abs((Fn-F)/F)<0.00001):
            break
            
    # relationship between true anomaly (theta) and eccentric anomaly (Fn)
    true_an = 2.*arctan(sqrt((1.+ecc)/(1.-ecc))*tan(Fn/2.))
    
    return Fn,true_an

def mean_anomaly_to_time(M,total_mass,sma,T0=0):
    """
    Convert mean anomaly to time
    
    @param M: mean anomaly in radians
    @param total_mass: total mass of the system in kg
    @param sma: semi-major axis in m
    @return: time
    @rtype: float
    """
    return M/sqrt(constants.GG*total_mass/sma**3) + T0
    
def calculate_phase(T,ecc,per0,phshift=0):
    r"""
    Compute orbital phase from true anomaly T.
    
    The phase :math:`\Phi` is related to the true anomaly :math:`T`, eccentricity
    :math:`e`, argument of periastron :math:`\omega` and phase shift :math:`\Delta\phi`
    as:
    
    .. math::
    
        E = 2 \arctan\left[ \sqrt{\frac{1-e}{1+e}} \tan\left(\frac{T}{2}\right)\right] \\
        
        M = E - e\sin(E) \\
        
        \Phi = \frac{M+\omega}{2\pi} - \frac{1}{4} + \Delta\phi
    
    @parameter T: true anomaly
    @type T: float
    @parameter per0: argument of periastron (radians)
    @type per0: float
    @parameter ecc: eccentricity
    @type ecc: float
    @parameter phshift: phase shift
    @type phshift: float
    @return: phase of superior conjunction, phase of periastron passage
    @rtype: float,float
    """
    E = 2.0*arctan(sqrt((1-ecc)/(1+ecc)) * tan(T/2.0))
    M = E - ecc*sin(E)
    return (M+per0)/(2.0*pi) - 0.25 + phshift

    
def calculate_critical_phases(per0, ecc, phshift=0):
    r"""
    Computes critical phases in the orbit: periastron passage, superior conjunction,
    inferior conjunction, ascending node and descending node.
    
    The phase of periastron passage :math:`\Phi_\mathrm{per}` is related to the
    argument of periastron :math:`\omega` and the phase shift `\Delta\phi` as
    follows (Phoebe scientific reference Eq. ?):
    
    .. math::
    
        \Phi_\mathrm{per} = \frac{1}{2\pi}\left(\omega - \frac{\pi}{2}\right) + \Delta\phi
    
    Example usage:
    
    >>> per0 = pi/4.0
    >>> ecc = 0.3
    >>> print(calculate_critical_phases(per0,ecc))
    [ 0.          0.06735539 -0.29554513 -0.06735539  0.29554513]
    
    With superior conjunction as t0:
    (-0.125, -0.057644612788576133, -0.42054512757020118, -0.19235538721142384, 0.17054512757020118)
    
    @parameter per0: argument of periastron (radians)
    @type per0: float
    @parameter ecc: eccentricity
    @type ecc: float
    @parameter phshift: phase shift
    @type phshift: float
    @return: phase of periastron passage, superior conjunction, inferior conjunction, ascending, descending
    @rtype: 5xfloat
    """
    #-- Phase of periastron passage
    Phi_per0 = (per0 - pi/2.0)/(2.0*pi) + phshift
    Phi_per0 = calculate_phase(0.0, ecc, per0, phshift)
    #-- Phase of inferior/superior conjunction
    Phi_conj = calculate_phase(pi/2.0-per0, ecc, per0, phshift)
    Phi_inf  = calculate_phase(3.0*pi/2.0-per0, ecc, per0, phshift)
    Phi_asc  = calculate_phase(-per0, ecc, per0, phshift)
    Phi_desc = calculate_phase(pi-per0, ecc, per0, phshift)
    #-- convert to our stuff:
    phases = np.array([Phi_per0, Phi_conj, Phi_inf, Phi_asc, Phi_desc])
    return from_perpass_to_supconj(phases, 1.0, per0, phshift=phshift)

def calculate_critical_times(per0, ecc, period, t0, phshift=0.,
                             t0type='periastron passage'):
    """
    Computes critical times in the orbit.
    """
    
    if t0type == 'superior conjunction':
        t0 = from_supconj_to_perpass(t0, period, per0, phshift=phshift)
    elif not t0type == 'periastron passage':
        raise ValueError("Do not recognize t0type='{}'".format(t0type))
    
    crit_phases = calculate_critical_phases(per0, ecc, phshift=phshift)
    return crit_phases * period + t0


def from_supconj_to_perpass(t0, period, per0, phshift=0.):
    """
    Convert the time convention where t0 is superior conjunction to periastron passage.
    
    Typically, parameterSets coming from Wilson-Devinney or Phoebe Legacy
    have :math:`t_0` as superior conjunction.
    
    Inverse function is L{from_perpass_to_supconj}.
    
    See Phoebe Scientific reference Eqs. (3.30) and (3.35).
    
    Time zeropoint and period should be in the sam units
    
    @param t0: time of superior conjunction
    @type t0: float
    @param period: orbital period
    @type period: float
    @param per0: argument of periastron (rad)
    @type per0: float
    @param phshift: phase shift (cycle)
    @type phshift: float
    @return: new time zero point
    @rtype: float
    """
    t0 = t0 + (phshift - 0.25 + per0/(2*np.pi)) * period
    return t0

def from_perpass_to_supconj(t0,period,per0,phshift=0.):
    """
    Convert the time convention where :math:`t_0` is periastron passage to superior conjunction.
    
    Typically, parameterSets coming from Wilson-Devinney or Phoebe Legacy
    have T0 as superior conjunction.
    
    Inverse function is L{from_supconj_to_perpass}.
    
    See Phoebe Scientific reference Eqs. (3.30) and (3.35).
    
    @param t0: time of periastron passage
    @type t0: float
    @param period: orbital period
    @type period: float
    @param per0: argument of periastron (rad)
    @type per0: float
    @param phshift: phase shift (cycle)
    @type phshift: float
    @return: new time zero point
    @rtype: float
    """
    t0 = t0 - (phshift - 0.25 + per0/(2*np.pi)) * period
    return t0
    

def eclipse_separation(ecc,per0):
    r"""
    Calculate the eclipse separation between primary and secondary in a light curve.
    
    Minimum separation at ``per0`` = :math:`\pi`
    
    Maximum spearation at ``per0`` = 0
    
    .. math:: \Delta = \pi + 2\arctan\left(\frac{e\cos\omega}{\sqrt{1-e^2}}\right) + 2\frac{e\cos\omega\sqrt{1-e^2}}{1-e^2\sin^2\omega}
    
    @parameter ecc: eccentricity
    @type ecc: float
    @parameter per0: argument of periastron (radians)
    @type per0: float
    @return: separation in phase units (0.5 is half)
    @rtype: float
    """
    radians = pi+2*arctan(ecc*cos(per0)/sqrt(1-ecc**2)) + 2*ecc*cos(per0)*sqrt(1-ecc**2)/(1-ecc**2*sin(per0)**2)
    return radians/(2*pi)


def per0_from_eclipse_separation(separation,ecc):
    r"""
    Caculate the argument of periastron from the eclipse separation and eccentricity.
    
    separation in phase units.
    
    Separation :math:`\Delta` is defined as:
    
    .. math::
    
        \Delta = \frac{T_\mathrm{prim} - T_\mathrm{sec}}{P},\qquad T_\mathrm{prim} < T_\mathrm{sec}
        
        \Delta = 1.0 - \frac{T_\mathrm{prim} - T_\mathrm{sec}}{P},\qquad T_\mathrm{prim} > T_\mathrm{sec}
    
    Though for WD it seems to be the reverse
    
    @parameter separation: separation in phase units (0.5 is half)
    @type separation: float
    @parameter ecc: eccentricity
    @type ecc: float
    @return: per0 (argument of periastron) in radians
    @rtype: float
    """
    minsep_per0 = pi
    maxsep_per0 = 0.
    minsep = eclipse_separation(ecc,minsep_per0)
    maxsep = eclipse_separation(ecc,maxsep_per0)
    if separation<minsep or maxsep<separation:
        logger.warning('Phase separation must be between %.3g and %.3g when ecc=%.3g'%(minsep,maxsep,ecc))
        return np.nan
    else:
        per0 = bisect(lambda x:separation-eclipse_separation(ecc,x),maxsep_per0,minsep_per0)
        return per0
    
#}

#{ Kepler's laws and helper functions

def third_law(totalmass=None,sma=None,period=None):
    """
    Kepler's third law.
    
    Give two quantities, derived the third:
        - C{totalmass} = total mass system (solar units)
        - C{sma} = semi-major axis (au)
        - C{period} = period (d)
    
    The ``totalmass`` :math:`M` is computed as:
    
    .. math::
        
        M = 4\pi^2 \mathrm{sma}^3/\mathrm{period}^2/G\quad\mathrm{kg}
    
    The system semi-major axis ``sma`` as
    
    .. math::
    
        \mathrm{sma} = (G M\mathrm{period}^2/(4\pi^2))^{1/3}\quad\mathrm{m}

    And the ``period`` as
    
    .. math::
    
        \mathrm{period} = \sqrt{4\pi^2\mathrm{sma}^3/(G*M)} \quad\mathrm{s}
    
    
    Example of Jupiter (Jupiter mass negligeable):
    
    >>> print third_law(totalmass=1.,sma=5.204)
    4336.15053042
    >>> print third_law(sma=5.204,period=4332.59)
    1.00164427903
    >>> print third_law(totalmass=1.,period=4332.59)
    5.20115084661
    
    @parameter totalmass: total mass of the system (solar units)
    @type totalmass: float
    @parameter sma: system's semi-major axis (au)
    @type sma: float
    @parameter period: period of the system (d)
    @type period: float
    @return: the quantity that is not given
    @rtype: float
    """
    if sma is not None:
        sma *= constants.au
    if period is not None:
        period *= (24*3600.)
    if totalmass is not None:
        totalmass *= constants.Msol
    
    if totalmass is None:
        return 4*pi**2*sma**3/period**2/constants.GG/constants.Msol
    if sma is None:
        return (constants.GG*totalmass*period**2/(4*pi**2))**(1./3.)/constants.au
    if period is None:
        return sqrt(4*pi**2*sma**3/(constants.GG*totalmass))/(24*3600.)    

        
def calculate_asini(period, ecc, K1=0, K2=0):
    r"""
    Calculate projected semi-major axis of a component or the system (asini).
    
    Period, eccentricity and semi-amplitude are usually obtained from
    radial velocity measurements.
    
    .. math::
    
        a_1\sin i = K_1 \frac{P}{2\pi} \sqrt{ 1-e^2} & 
        
        a_2\sin i = K_2 \frac{P}{2\pi} \sqrt{ 1-e^2} & 
    
    and
    
    .. math::
        
        a\sin i = a_1 \sin i + a_2 \sin i
    
    For a doubled line binary, the system asini is obtained via::
    
        asini = calculate_asini(P,e,K1)+calculate_asini(P, e, K2)
    
    or in one go via::
    
        asini = calculate_asini(P, e, K1, K2)
    
    @param period: period (s)
    @type period: float
    @param ecc: eccentricity
    @type ecc: float
    @param K1: semi-amplitude of primary (m/s)
    @type K1: float
    @param K2: semi-amplitude of primary (m/s)
    @type K2: float
    @return: semi-major axis (m)
    @rtype: float
    """
    factor = period*np.sqrt((1.0-ecc**2))/2.0/pi
    asini = np.abs(K1)*factor
    asini+= np.abs(K2)*factor
    return asini
    
def calculate_mass_function(period, ecc, K):
    r"""
    Calculate mass function f(m).
    
    .. math::
        
        f(M) = (M_2\sin \mathrm{incl})^3/(M_1+M_2)^2
    
        f(M) = 1.0361\times 10^7 (1-\mathrm{ecc}^2)^{1.5} K^3 \mathrm{period}
    
    Fixing the primary and secondary mass, you can use this relation to compute
    the inclination angle.
    
    @param period: period (s)
    @type period: float
    @param ecc: eccentricity
    @type ecc: float
    @param K: semi-amplitude (m/s)
    @type K: float
    @return: mass function (kg)
    @rtype: float
    """
    return (1-ecc**2)**1.5 * K**3*period/(2*pi*constants.GG)

def get_incl_from_mass_function(mass_function,mass1,mass2):
    """
    Derive the inclination angle from the mass function
    
    .. math::
    
        \sin\mathrm{incl} = (f(M) M_\mathrm{tot}^2/ M_2^3)^{1/3}
    
    @param mass_function: mass function (kg)
    @type mass_function: float
    @param mass1: primary mass (kg)
    @type mass1: float
    @param mass2: secondary mass (kg)
    @type mass2: float
    @return: inclination angle (rad)
    @rtype: float
    """
    totalmass = mass1+mass2
    sin_incl = (mass_function*(mass1+mass2)**2/mass2**3)**(1./3.)
    return np.arcsin(sin_incl)
    
def get_incl_from_RV(a1sini,sma,q):
    r"""
    Compute inclination angle from RV information and individual masses.
    
    You can compute C{sma} from L{third_law} and assuming a total mass and
    the binary period. From the individual mass (i.e. the mass ratio), and an
    estimate of the projected semi-major axis of the primary, you can compute
    the inclination angle:
    
    .. math::
    
        a\sin i &= (a_1+a_2)\sin i
        
                &= \left(a_1+\frac{a_1}{q}\right)\sin i
        
        \sin i & = \left(1+\frac{1}{q}\right)\frac{a_1}{a}\sin i
    
    @param a1sini: projected semi-major axis of primary (m)
    @type a1sini: float
    @param sma: system semi-major axis (m)
    @type sma: float
    @param q: mass ratio
    @type q: float
    @return: inclination angle (rad)
    @rtype: float
    """
    return np.arcsin(a1sini/sma*(1+1./q))

def calculate_mass(component,sma,period,q):
    r"""
    Calculate the mass of a component given the orbital paramaters.
    
    Everything should be in SI units.
    
    .. math::
    
        M = 4\pi^2 \frac{a^3}{G (1+q) P^2}
    
    @rtype: float
    """
    if component==0:
        _q_ = q
    elif component==1:
        _q_ = 1./q
    else:
        raise ValueError("Component {} not understood".format(component))
    mass = 4*pi**2*sma**3/period**2/constants.GG/(1.0+_q_)
    return mass

def calculate_total_radius_from_eclipse_duration(eclipse_duration,incl,sma=1):
    r"""
    Calculate the total radius from the eclipse duration, i and sma.
    
    .. math::
    
        \Delta t = \frac{P}{\pi} \arcsin\left(\sqrt{ \frac{R_\mathrm{tot}^2}{a^2} - \cos^2i}\right)
        
    
    You can also use this function to compute the radius difference, and
    from both the individual radii: calling it with the same signature but
    replacing the C{eclipse_duration} with the duration of the total eclipse,
    you get the radius difference instead of radius sum::
    
        a = calculate_total_radius_from_eclipse_duration(eclipse_duration,sma,incl)
        b = calculate_total_radius_from_eclipse_duration(totally_eclipsed_duration,sma,incl)
        R1 = (a+b)/2
        R2 = (a-b)/2
    
    If you don't give the semi-major axis, the total radius will be in units
    of sma.
    
    Only for circular orbits!!
    
    @param eclipse_duration: fractional duration of the eclipse
    @type eclipse_duration: float
    @param sma: semi-major axis (m)
    @type sma: float
    @param incl: inclination angle (rad)
    @type incl: float
    @return: total radius (R1+R2) (m)
    @rtype: float
    """
    return sma*sqrt( sin(pi*eclipse_duration)**2 + cos(incl)**2)
    
    
def truecoords_to_spherical(coords, distance=0., origin=(0.,0.), units=None):
    """
    Convert 3D coords to spherical on-sky coordinates.
    
    You can use this to position an object absolutely on the sky, if you set
    C{origin} to some value.
    
    If you set C{distance=None}, a standard value of 10 pc will be used.
    
    @param coords: array of 3D coordinates in Rsol. X,Y must be sky-coordinates,
                   z must be the radial direction
    @type coords: (Nx3) array
    @param distance: distance to the source in Rsol or given as a tuple with unit
    @type distance: float
    @param origin: origin of spherical coordinate system, in radians
    @type origin: tuple of floats
    @return: spherical coordinates in radians
    @rtype: (float,float)
    """
    if distance is None: # then take 10 pc
        #distance = 443658100.4823813 # 10 pc in Rsol
        distance = 10*phoebe.constants.pc/phoebe.constants.Rsol
    elif isinstance(distance,tuple):
        distance = conversions.convert(distance[1],'Rsol',distance[0])
    
    # correct coords for distance
    z = coords[:,2]+distance
    
    # compute spherical angles
    alpha = 2*arctan(coords[:,0]/(2*z)) 
    beta = 2*arctan(coords[:,1]/(2*z))
    
    # correct spherical angles for offset
    if units is not None:
        ra = conversions.convert('rad',units,origin[0] + alpha)
        dec = conversions.convert('rad',units,origin[1] + beta)
        return ra, dec, z
    else:
        return origin[0]+alpha,origin[1]+beta, z

#}


#{ Time scales

def circularization(M,R,q,a,envelope='radiative'):
    """
    Circularization time scale.
    
    """
    #-- Tidal torque constants (Zahn 1975, 1977)
    E2_mass = [1.6,2.,3,5,7.,10,15.]
    E2_vals = [2.41e-9,1.45e-8,4.72e-8,1.53e-7,3.8e-7,1.02e-6,3.49e-6]
    E2 = np.interp(M,E2_mass,E2_vals)
    tcirc = 2./21.*sqrt(R**3/constants.GG*M) * 1./(q*(1+q)**(11./6.))*1./E2 * (a/R)**(21./2.)
    raise NotImplementedError
    return tcirc
    
#}

#{ Multibody problem

def Nbody(times,x_init,v_init,masses):
    """
    Integrate an Nbody problem over time.
    
    Initial conditions are x_init and v_init. They should be arrays
    of shape (Nx3), with N the number of bodies.
    
    Everything should be in SI units.
    
    If times is a float, it must represent dt and the system
    is advanced during one timestep. The return values then have
    the same shape as x_init and v_init, but represent the new
    timestep
    """
    #-- how many bodies do we have?
    Nbody = len(x_init)
    
    #-- prepare output arrays for orbit and velocities
    if not hasattr(times,'__iter__'):
        Ntime = 2
    else:
        Ntime = len(times)
    orbit = np.zeros((Ntime,3*Nbody))
    velos = np.zeros((Ntime,3*Nbody))
    orbit[0] = np.ravel(x_init)
    velos[0] = np.ravel(v_init)
    
    #-- now integrate over time array (if times is an array)
    if not hasattr(times,'__iter__'):
        delta_times = [times]
    else:
        delta_times = np.diff(times)
    for i,dt in enumerate(delta_times):
        #-- and calculate the forces from all objects on each object
        F = np.zeros((Nbody,3))
        for j in range(len(Ms)):
            xj = orbit[i,3*j:(3*j+3)]
            vj = velos[i,3*j:(3*j+3)]
            for k in range(len(Ms)):
                #-- a body excerts no force on itself
                if j==k: continue
                xk = orbit[i,3*k:(3*k+3)]
                r = xk-xj
                #-- compute magnitude and direction of the force
                F_mag = constants.GG*Ms[k]*Ms[j]/norm(r)**2
                F_dir = r/norm(r)
                F[j] += F_mag*F_dir
            #-- new x-location and velocity
            orbit[i+1,3*j:(3*j+3)] =         vj*dt + xj
            velos[i+1,3*j:(3*j+3)] = F[j]/Ms[j]*dt + vj
    if not hasattr(times,'__iter__'):
        orbit = orbit[-1].reshape(x_init.shape)
        velos = velos[-1].reshape(v_init.shape)
    #-- that's it
    return orbit,velos

#}
#{ Phoebe interface    

def parse_ps(func,ps,*args,**kwargs):
    """
    Parse default values from a ParameterSet to a function.
    
    In short, this function looks for a match between the names of the
    arguments in C{func}, and overrides their defaults (if named arguments)
    or takes their values from it (if positional arguments).
    
    In long: mandatory arguments to C{func} can be passed explicitly. If any
    are not given, the values are taken from the parameterSet. If those do not
    exist, a TypeError is raised.
    
    Named arguments can be passed explicitly. If any are not given, the
    values are taken from the parameterSet. If those do not exist, the default
    value from C{func} is used.
    
    Extra keyword arguments (**kwargs in C{func}) can be passed explicitly.
    Their values are never overriden since their names cannot easily be
    derived.
    
    B{warning:} Values are B{always} passed in B{SI units}, if a ParameterSet
    has units. It is the user's responsibility to know if C{func} needs SI
    units.
    
    @param func: function to be called
    @type func: callable
    @param ps: parameterSet to use for default values
    @type ps: parameterSet
    """
    argspec = inspect.getargspec(func)
    argnames = argspec.args
    defaults = argspec.defaults
    #-- mandatory arguments: check if any are given, if not, set their values
    #   from the parameterSet. If that doesn't work out, an Error will be
    #   raised because you must have done something wrong...
    argvals = []
    if len(args):
        argvals += list(args)
    argnr_start = len(argvals)
    if (argnr_start+len(defaults))<len(argnames):
        for argnr,argname in enumerate(argnames[argnr_start:-len(defaults)]):
            if argname in ps:
                # Take the component sma instead of system sma
                if argname == 'sma' and 'component' in kwargs:
                    argname = 'sma{:d}'.format([None,'primary','secondary'].index(kwargs['component']))
                    argvals.append(ps.request_value(argname))
                #-- if the paramer has units, make sure to take the SI value
                #   else, just take the value
                elif ps.has_unit(argname):
                    argvals.append(ps.request_value(argname,'SI'))
                else:
                    argvals.append(ps.request_value(argname))
            else:
                raise TypeError("{} takes at least {} arguments, {} given".format(func.__name__,len(argnames)-len(defaults),len(args)))
        argnr_start += argnr+1
    #-- keyword arguments: override their defaults with the ones from the
    #   parameterSet, unless they are explicitly set in the extra **kwargs
    for argnr,argname in enumerate(argnames[argnr_start:]):
        #-- fill with default value
        argvals.append(defaults[argnr])
        #-- override with parameterSet value
        if argname in ps:
            if ps.has_unit(argname):
                argvals[-1] = ps.request_value(argname,'SI')
            else:
                argvals[-1] = ps.request_value(argname)
        #-- override with keyword argument value
        if argname in kwargs:
            argvals[-1] = kwargs.pop(argname)
    #-- pass the unsed extra **kwargs
    return func(*argvals,**kwargs)
    

def walk_hierarchical(o,orbits=None,components=None):
    """
    Walk through a hierarchical orbit dictionary.
    
    A hierarchical definition should something like:
    
    Let's define five random orbits:
    
    >>> from pyphoebe.parameters import parameters
    >>> x1 = parameters.ParameterSet(context='orbit',label='system')
    >>> x2 = parameters.ParameterSet(context='orbit',label='sysD')
    >>> x3 = parameters.ParameterSet(context='orbit',label='sysC')
    >>> x4 = parameters.ParameterSet(context='orbit',label='sysB')
    >>> x5 = parameters.ParameterSet(context='orbit',label='sysA')
    get_orbit(time,P,e,a_comp,T0,per0=argper,long_an=long_an,
                               incl=in
    And nest them in some way, from deepest nesting to the upper level:
    
    >>> sysA = {x5:[True,True]}
    >>> sysC = {x3:[True,True]}
    >>> sysB = {x4:[True,sysA]}
    >>> sysD = {x2:[sysC,sysB]}
    >>> system = {x1:[True,sysD]}
    
    An attempt to do an ASCII representation::
    
        system ---- primary
               |
               +--- sysD    ---- sysC ---- primary
                            |         |
                            |         +--- secondary
                            |
                            +--- sysB ---- primary
                                      |
                                      +--- sysA       ---- primary
                                                      |
                                                      +--- secondary
    
    Then you can walk over the system:
    
    >>> for this_comp,orbits,comps in walk_hierarchical(system):
    ...     print(this_comp,[o['label'] for o in orbits],comps)
    (0, ['system'], [])
    (0, ['system', 'sysD', 'sysC'], [1, 0])
    (1, ['system', 'sysD', 'sysC'], [1, 0])
    (0, ['system', 'sysD', 'sysB'], [1, 1])
    (0, ['system', 'sysD', 'sysB', 'sysA'], [1, 1, 1])
    (1, ['system', 'sysD', 'sysB', 'sysA'], [1, 1, 1])
    
    In the above, the first number tells you if the object is a a primary (0)
    or secondary (1) in its own orbit. The following list of labels tells you
    what systems the object belongs to (hierarchically, starting with the top
    most or biggest system). The final list tells you if each subsystem is the
    primary (0) or secondary (1). The length of that list is 1 - system_list
    because the final component is given first.
    
    Setting one of the components to C{None} will make the algorithm skip
    that component.
    
    @param o: system to walk
    @type o: dictionary
    @param orbits: history of the orbits walked over
    @type orbits: list
    @param components: history of the components
    @type components: list
    @return: (current_component, orbits, components)
    @rtype: 3-tuple
    """
    #-- if there is no history, create a history
    if orbits is None: orbits = []
    if components is None: components = []
    #-- if we're iterating over a dictionary, remember it's key (since it is
    #   the parameterset containing the orbital parameters), and iterative
    #   over it's member. It's member should be a list containing two items
    #   which can either be "True" (then it is a component) or a parameterSet
    #   itself (in which case it is a nested orbit).
    if isinstance(o,dict):
        orbitps = list(o.keys())[0]
        new_orbits = orbits + [orbitps]
        for comp in walk_hierarchical(o[orbitps],orbits=new_orbits,components=components):
            yield comp
    #-- if we're iterating over a list, check if we're dealing with a
    #   component or again a nested orbit.
    elif isinstance(o,list):
        for ncomp,comp in enumerate(o):
            if isinstance(comp,dict):
                orbitps = list(comp.keys())[0]
                new_orbits = orbits + [orbitps]
                new_components = components + [ncomp]
                for mcomp,comp in enumerate(walk_hierarchical(comp[orbitps],orbits=new_orbits,components=new_components)):
                    yield comp
            else:
                if comp is True:
                    yield ncomp,orbits,components
                else:
                    yield None,orbits,components


def get_binary_orbit(time, orbit, component, barycentric=False, return_time=False):
    """
    Get the binary orbit from a Phoebe parameterSet.
    
    The output is in Phoebe units, i.e.:
    
        - distance is in Rsol
        - time is in days
        - velocity is in Rsol/day (conversion factor to km/s is 8.0498611111111)
        
    The output is in Phoebe coordinates, i.e.:
    
        - x, y are the coordinates in the plane of the sky
        - z is in the direction towards from the observer. Thus, negative z means
          the object is going away from the observer
    
    If :envvar::`barycentric` is ``True``, then, unsurprisingly, the barycentric
    orbit will be given (:py:func:`get_barycentric_orbit`). If you're interested
    in the adjust times, :envvar:`set return_time=True`.
    
    @param time: time array or float to compute the orbit on
    @type time: numpy array or float
    @param orbit: orbit ParameterSet
    @type orbit: ParameterSet
    @param component: 'primary' or 'secondary'
    @type component: str
    @param barycentric: correct for light time travel effects
    @type barycentric: bool
    """
    #-- get some information
    P = orbit.get_value('period','d')
    e = orbit.get_value('ecc')
    a = orbit.get_value('sma','Rsol')
    a1 = orbit.get_constraint('sma1','Rsol')
    a2 = orbit.get_constraint('sma2','Rsol')
    inclin = orbit.get_value('incl','rad')
    argper = orbit.get_value('per0','rad')
    long_an = orbit.get_value('long_an','rad')
    T0 = orbit.get_value('t0')
    t0type = orbit.get('t0type', 'periastron passage')
    dpdt = orbit.get_value('dpdt')
    dperdt = orbit.get_value('dperdt')/180.*np.pi / 365.25 # deg/yr to rad/d
    deccdt = 0.0
    a_comp = [a1,a2][['primary','secondary'].index(component)]
    
    if t0type == 'superior conjunction':
        time = time - orbit['phshift'] * P
    
    # Where in the orbit are we? We need everything in cartesian Rsol units
    if not barycentric:
        func = get_orbit
    else:
        func = get_barycentric_orbit
    
    # in the case of barycentric times, also the "proper times" are returned
    output = func(time, P, e, a_comp, T0, per0=argper,
                               long_an=long_an, incl=inclin, dpdt=dpdt,
                               deccdt=deccdt, dperdt=dperdt,
                               mass_conservation=True,
                               component=component, t0type=t0type)
    loc, velo, euler = output[:3]
    
    if not return_time:
        return loc,velo,euler
    elif return_time and barycentric:
        return loc,velo,euler,output[3]
    else:
        return loc,velo,euler,time


def place_in_binary_orbit(self,time):
    """
    Place a body in a binary orbit at a specific time.
    """
    #-- get some information
    P = self.params['orbit']['period']
    e = self.params['orbit']['ecc']
    a = self.params['orbit']['sma']
    q = self.params['orbit']['q']
    a1 = a / (1+1.0/q)
    a2 = a-a1
    deg2rad=pi/180.
    inclin = self.params['orbit']['incl'] *deg2rad
    argper = self.params['orbit']['per0'] *deg2rad
    long_an = self.params['orbit']['long_an']*deg2rad
    T0 = self.params['orbit'].get_value('t0')
    n_comp = self.get_component()
    component = ('primary', 'secondary')[n_comp]
    t0type = self.params['orbit'].get('t0type','periastron passage')

    if t0type == 'superior conjunction':
        time = time - self.params['orbit']['phshift'] * P

    a_comp = [a1, a2][n_comp]
    
    #-- where in the orbit are we? We need everything in cartesian Rsol units
    loc, velo, euler = get_orbit(time, P, e, a_comp, T0, per0=argper, 
                                 long_an=long_an, incl=inclin,
                                 component=component, t0type=t0type)
    
    #-- we need a new copy of the mesh
    mesh = self.mesh#.copy()
    
    #-- modify velocity vectors due to binarity and rotation within the orbit
    #   rotational velocity
    #   There's two posibilities: either we have a synchronicity parameter
    #   or we don't. If we don't, we assume synchronous rotation
    #-- rotational velocity
    if 'component' in self.params and 'syncpar' in self.params['component']:
        F = self.params['component']['syncpar']
        logmsg = 'using synchronicity parameter ({:.3g})'.format(F)
    else:
        F = 1.
        logmsg = 'could not find "syncpar"; assuming synchronisation'    
    omega_rot = F * 2*pi/P # rad/d

    #-- if we can't get the polar direction, assume it's in the negative Z-direction
    try:
        polar_dir = -self.get_polar_direction(norm=True)
    except:
        polar_dir = default_polar_dir
    
    fields = mesh.dtype.names
    
    ctrans.place_in_binary_orbit(mesh['mu'], mesh['center'],
                             mesh['triangle'], mesh['normal_'],
                             mesh['velo___bol_'], polar_dir*omega_rot,
                             mesh['_o_center'], euler, loc, velo)
    
    if 'B_' in fields:
        mesh['B_'] = ftrans.trans(mesh['B_'],euler,default_zeros,len(mesh['B_']))
    
    self.mesh = mesh



def place_in_binary_orbit_old(self,time):
    """
    Place a body in a binary orbit at a specific time.
    """
    #-- get some information
    P = self.params['orbit']['period']
    e = self.params['orbit']['ecc']
    a = self.params['orbit']['sma']
    q = self.params['orbit']['q']
    a1 = a / (1+1.0/q)
    a2 = a-a1
    deg2rad=pi/180.
    inclin = self.params['orbit']['incl'] *deg2rad
    argper = self.params['orbit']['per0'] *deg2rad
    long_an = self.params['orbit']['long_an']*deg2rad
    T0 = self.params['orbit'].get_value('t0')
    n_comp = self.get_component()
    component = ('primary', 'secondary')[n_comp]
    t0type = self.params['orbit'].get('t0type','periastron passage')

    if t0type == 'superior conjunction':
        time = time - self.params['orbit']['phshift'] * P

    a_comp = [a1, a2][n_comp]
    
    #-- where in the orbit are we? We need everything in cartesian Rsol units
    loc, velo, euler = get_orbit(time, P, e, a_comp, T0, per0=argper, 
                                 long_an=long_an, incl=inclin,
                                 component=component, t0type=t0type)
    
    #-- we need a new copy of the mesh
    mesh = self.mesh#.copy()
    
    #-- modify velocity vectors due to binarity and rotation within the orbit
    #   rotational velocity
    #   There's two posibilities: either we have a synchronicity parameter
    #   or we don't. If we don't, we assume synchronous rotation
    #-- rotational velocity
    if 'component' in self.params and 'syncpar' in self.params['component']:
        F = self.params['component']['syncpar']
        logmsg = 'using synchronicity parameter ({:.3g})'.format(F)
    else:
        F = 1.
        logmsg = 'could not find "syncpar"; assuming synchronisation'    
    omega_rot = F * 2*pi/P # rad/d

    #-- if we can't get the polar direction, assume it's in the negative Z-direction
    try:
        polar_dir = -self.get_polar_direction(norm=True)
    except:
        polar_dir = default_polar_dir
    
    fields = mesh.dtype.names
    
    velo_rot = fgeometry.cross_nx3_3(mesh['_o_center'], polar_dir*omega_rot)
    velo_rot = ftrans.trans(velo_rot,euler,default_zeros,len(velo_rot))
    mesh['center'] = ftrans.trans(mesh['center'],euler,loc,len(velo_rot))
    mesh['triangle'][:,0:3] = ftrans.trans(mesh['triangle'][:,0:3],euler,loc,len(mesh['triangle'][:,0:3]))
    mesh['triangle'][:,3:6] = ftrans.trans(mesh['triangle'][:,3:6],euler,loc,len(mesh['triangle'][:,3:6]))
    mesh['triangle'][:,6:9] = ftrans.trans(mesh['triangle'][:,6:9],euler,loc,len(mesh['triangle'][:,6:9]))
    mesh['normal_'] = ftrans.trans(mesh['normal_'],euler,default_zeros,len(mesh['normal_']))
    mesh['velo___bol_'] += velo_rot+velo
    mesh['mu'] = cgeometry.cos_theta(mesh['normal_'].ravel(order='F').reshape((-1,3)),default_los_dir)
    # Add systemic velocity:
    #globals = self.get_globals()
    #if globals is not None:
        ##vgamma = globals.request_value('vgamma', 'Rsol/d')
        #vgamma = globals['vgamma'] * 1000. / constants.Rsol * 24 * 3600
        #mesh['velo___bol_'][:,2] -= vgamma
    
    if 'B_' in fields:
        mesh['B_'] = ftrans.trans(mesh['B_'],euler,default_zeros,len(mesh['B_']))
    
    self.mesh = mesh

    #logger.info('Placed into orbit')

        
def get_hierarchical_orbit_phoebe(times, orbits, comps):
    """
    Retrieve the orbit of one components in a hierarchical system.
    
    Is velocity wrong?
    
    @param times: time array
    @type times: array
    @param orbits: list of ParameterSets in the C{phoebe} frame and C{orbit}
    context
    @type orbits: list of ParameterSets of length N
    @param comps: list of integers denoting which component (0 for primary,
    1 for secondary) the object is for each orbit
    @type comps: list of integers
    @return: position vectors, velocity vectors
    @rtype: 3-tuple, 3-tuple
    """
    if np.isscalar(times):
        obj = np.zeros((1,3))
        vel = np.zeros((1,3))
    else:
        obj = np.zeros((len(times),3))
        vel = np.zeros((len(times),3))
    for i,(orbit,comp) in enumerate(list(zip(orbits,comps))):
        #-- retrieve the coordinates and euler angles for this orbit
        #-- get some information
        P = orbit['period']#get_value('period', 'd')
        e = orbit['ecc']#.get_value('ecc')
        a = orbit['sma']#.get_value('sma', 'Rsol')
        dpdt = orbit['dpdt']/(365.25*24*3600)
        q = orbit['q']
        a1 = a / (1+1.0/q)
        a2 = a-a1
        inclin = orbit['incl'] / 180.*np.pi
        argper = orbit['per0'] / 180.*np.pi
        long_an = orbit['long_an'] / 180.*np.pi
        T0 = orbit['t0']
        component = ('primary', 'secondary')[comp]
        t0type = orbit['t0type']
        if t0type == 'superior conjunction':
            times = times - orbit['phshift'] * P
        a_comp = [a1, a2][comp]
        loc, velo, euler = get_orbit(times, P, e, a_comp, T0, per0=argper, 
                                 long_an=long_an, incl=inclin, dpdt=dpdt,
                                 component=component, t0type=t0type)
    
        #-- and incrementally add it to the present object
        obj[:,0] += loc[0].ravel()
        obj[:,1] += loc[1].ravel()
        obj[:,2] += loc[2].ravel()
        vel[:,0] += velo[0].ravel()
        vel[:,1] += velo[1].ravel()
        vel[:,2] += velo[2].ravel()
    return obj.T, vel.T    

    
#}

if __name__=="__main__":
    import doctest
    import pylab as pl
    doctest.testmod()
    pl.show()
    
    #-- things like below should go in a test suite
    #from pyphoebe.parameters import parameters
    #from pyphoebe.binary import keplerorbit
    

    
