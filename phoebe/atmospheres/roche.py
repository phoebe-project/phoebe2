"""
Derive parameters in the Roche approximation.

Most of these parameters are local atmospheric parameters.
"""
import logging
import numpy as np
from numpy import cos,sin,sqrt,pi,log,tan
from scipy.optimize import newton
from phoebe.algorithms import marching
from phoebe.units import constants
from phoebe.utils import coordinates

logger = logging.getLogger('ATM.ROCHE')


#{ General

def temperature_zeipel(system):
    """
    Calculate local temperature.
    
    beta is gravity darkening parameter.
    
    Definition:: 
    
        T = Tpole * (g/gpole)**(beta/4) [Eq. 4.23 in Phoebe Scientific Reference]
    
    beta/gravb = 1.00 for radiative atmospheres (Teff>8000K)
    beta/gravb = 0.32 for convective atmpsheres (Teff<5000K, but we're a
    little bit less strict and set 6600)
    
    Body C{system} needs to have the following parameters defined:
        
        - teff or teffpolar
        - gravb
        - g_pole
    
    @param system: object to compute temperature of
    @type system: Body
    """
    #-- necessary global information: this should be the first parameterSet
    #   in a Ordered dictionary
    body = system.params.values()[0]
    if body.has_qualifier('teffpolar'):
        Teff = body.request_value('teffpolar','K')
        type = 'polar'
    else:
        Teff = body.request_value('teff','K')
        type = 'mean'
    beta = body.request_value('gravb')
    gp = body.request_value('g_pole')
    #-- consistency check for gravity brightening
    if Teff>=8000. and beta!=1.00:
        logger.warning('Object probably has a radiative atm (Teff={:.0f}K>8000K), for which gravb=1.00 is a better approx than gravb={:.2f}'.format(Teff,beta))
    elif Teff<=6600. and beta!=0.32:
        logger.warning('Object probably has a convective atm (Teff={:.0f}K<6600K), for which gravb=0.32 is a better approx than gravb={:.2f}'.format(Teff,beta))
    elif beta<0.32 or beta>1.00:
        logger.warning('Object has intermittent temperature, gravb should be between 0.32-1.00')
    #-- compute G and Tpole
    Grav = abs(10**(system.mesh['logg']-2)/gp)**beta
    if type=='mean':
        Tpole = Teff*(np.sum(system.mesh['size']) / np.sum(Grav*system.mesh['size']))**(0.25)
    elif type=='polar':
        Tpole = Teff
    else:
        raise ValueError("Cannot interpret temperature type '{}' (needs to be one of ['mean','polar'])".format(type))
    #-- now we can compute the local temperatures. We add a constraint to
    #   the body so that we can access the polar temperature if needed
    system.mesh['teff'] = Grav**0.25 * Tpole
    body.add_constraint('{{t_pole}} = {0:.16g}'.format(Tpole))
    logger.info("derived effective temperature (Zeipel) (%.3f <= teff <= %.3f, Tp=%.3f)"%(system.mesh['teff'].min(),system.mesh['teff'].max(),Tpole))
    

def temperature_espinosa(system):
    """
    Calculate local temperature according to Espinosa Lara & Rieutord (2011).
    
    This model doesn't need a gravity darkening parameter.
    
    @param system: object to compute temperature of
    @type system: Body
    """
    M = system.params['star'].request_value('mass','kg')
    r_pole = system.params['star'].request_value('radius','m')
    teffpolar = system.params['star'].request_value('teff')
    #-- rotation frequency in units of critical rotation frequency
    omega_rot = 2*np.pi/system.params['star'].request_value('rotperiod','s')
    Omega_crit = np.sqrt( 8*constants.GG*M / (27.*r_pole**3))
    omega = omega_rot/Omega_crit
    #-- equatorial radius
    Re = fast_rotation_radius(np.pi/2,r_pole,omega)
    #-- surface coordinates
    index = np.array([1,0,2])
    r,phi,theta = coordinates.cart2spher_coord(*system.mesh['_o_center'].T[index])
    cr = r*constants.Rsol/Re
    cr_pole = r_pole/Re
    #-- find var_theta for all points on the surface:
    def funczero(var_theta,cr,theta):
        sol = cos(var_theta)+log(tan(0.5*var_theta))-1./3.*omega**2*cr**3*cos(theta)**3 - cos(theta)-log(tan(0.5*theta))
        return sol
    var_theta = np.array([newton(funczero,itheta,args=(icr,itheta)) for icr,itheta in zip(cr,theta)])
    #-- we scale everything to the polar effective temperature
    factor_pole = np.exp(0.5/3.*omega**2*cr_pole**3)
    #-- funczero is not defined for the equator and pole
    factor_eq = (1-omega**2)**(-0.5/3.)
    scale = teffpolar*np.sqrt(cr_pole)/factor_pole
    #-- now compute the effective temperature
    factor = sqrt(tan(var_theta)/tan(theta))
    factor[(factor>factor_eq) | (factor<factor_pole)] = factor_eq
    teff = scale * (cr**-4 + omega**4*cr**2*sin(theta)**2-2*omega**2*sin(theta)**2/cr)**0.125*factor
    system.mesh['teff'] = teff
    logger.info("derived effective temperature (Espinosa) (%.3f <= teff <= %.3f)"%(system.mesh['teff'].min(),system.mesh['teff'].max()))


    
def approximate_lagrangian_points(q,d=1.,sma=1.):
    """
    Approximate Langrangian points L1, L2 and L3.
    
    Give C{d} as a fraction of semi-major axis. The units of the lagrangian
    points that you get out are C{sma}. So if you give the C{sma} in km, then
    you get the lagrangian points in km (see examples below).
    
    >>> approximate_lagrangian_points(1.)
    (0.5034722608505247, 1.6893712419226086, -0.6995580853748713)
    >>> approximate_lagrangian_points(constants.Mearth/constants.Msol)
    (0.990028699155567, 1.010038030137821, -0.9999982474945123)
    
    The Sun/Earth L1 is located about 1.5 million km from Earth:
    >>> L1,L2,L3 = approximate_lagrangian_points(constants.Mearth/constants.Msol,sma=constants.au)
    >>> (constants.au-L1)/1000.
    1491685.3744362793
    
    The Earth/Moon L1 is located about 60.000 km from the Moon:
    >>> dlun = 384499. # km
    >>> L1,L2,L3 = approximate_lagrangian_points(constants.Mlun/constants.Mearth,sma=dlun)
    >>> (dlun-L1)
    58032.18404510553
    
    @parameter q: mass ratio M1/M2
    @type q: float
    @parameter d: separation between the two objects (in units of sma)
    @type d: float
    @parameter sma: system semi-major axis
    @type sma: float
    @return: L1, L2, L3 (zero is center of primary, d is center of secondary)
    @rtype: float,float,float
    """
    mu = q/(1.+q)
    z = (mu/3.)**(1./3.)
    x_L1 = z - 1./3.*z**2 - 1./9.*z**3 + 58./81.*z**4
    x_L2 = z + 1./3.*z**2 - 1./9.*z**3 + 50./81.*z**4
    x_L3 = 1. - 7./12.*mu - 1127./20736.*mu**3 - 7889./248832*mu**4
    return (d-x_L1*d)*sma,(d+x_L2*d)*sma,(-x_L3*d)*sma

def calculate_critical_potentials(q,F=1.,d=1.,component=1):
    """
    Calculate the (approximate) critical potentials in L1, L2 and L3.
    
    Only approximate for synchroneous case.
    
    You need to give orbital separation C{d} as a fraction of the semi-major
    axies. E.g. for circular orbits, C{d=1}.
    
    @parameter q: mass ratio M2/M1
    @parameter d: separation between the two objects
    @type d: float
    @param F: synchronicity parameter
    @type F: float
    @return: critical potentials at L1,L2 and L3
    @rtype: float,float,float
    """
    L1,L2,L3 = approximate_lagrangian_points(q,d=d,sma=1.)
    theta,phi = np.pi/2,0
    Phi_L1 = -binary_potential(L1,theta,phi,0.0,q,d,F,component=component)
    Phi_L2 = -binary_potential(L2,theta,phi,0.0,q,d,F,component=component)
    Phi_L3 = -binary_potential(L3,theta,phi,0.0,q,d,F,component=component)
    return Phi_L1,Phi_L2,Phi_L3    

def calculate_critical_radius(q,F=1.,d=1.,sma=1.,component=1,loc='pole'):
    """
    Calculate the (approximate) critical radius.
    
    When C{loc='pole'}, the critical radius is that polar radius for which the
    equatorial radius reaches L1. When C{loc='eq'}, the returned radius is the
    critical radius at the equator.
    
    Example: suppose you want to compute the critical radius in Rsol:
    
    >>> Req = calculate_critical_radius(0.45,component=1,loc='eq',sma=32.)
    >>> Rpl = calculate_critical_radius(0.45,component=1,loc='pole',sma=32.)
    >>> print Req,Rpl
    18.58379386 13.5364047855
    >>> print Req/Rpl
    1.37287515811
    
    And for the other component:
    >>> Req = calculate_critical_radius(0.45,component=2,loc='eq',sma=32.)
    >>> Rpl = calculate_critical_radius(0.45,component=2,loc='pole',sma=32.)
    >>> print Req,Rpl
    13.381671155 9.32997069024
    >>> print Req/Rpl
    1.43426722326
    
    @parameter q: mass ratio M2/M1
    @parameter d: separation between the two objects as a fraction of sma.
    @type d: float
    @parameter sma: system semi-major axis
    @type sma: float
    @param F: synchronicity parameter
    @type F: float
    @param component: component in the system (1 or 2)
    @type component: integer
    @return: critical radius at L1 (in units of sma)
    @rtype: float
    """
    pL1,pL2,pL3 = calculate_critical_potentials(q,F=F,d=d)
    rL1 = potential2radius(pL1,q,d=d,F=F,sma=sma,loc=loc,component=component)
    return rL1
    

def potential2radius(pot,q,d=1,F=1.,component=1,sma=1.,loc='pole',tol=1e-10,maxiter=50):
    """
    Convert a potential value to a radius.
    
    Returns radius in units of system semi-major axis (C{sma}), unless it is
    given. In that case, the radius is in the same units as the semi-major axis.
    
    To compute the polar radius, set C{loc='pole'}. For equatorial radius, set
    C{loc='eq'}.
    
    >>> M1,M2 = 1.5,0.75
    >>> q = M2/M1
    >>> L1,L2,L3 = calculate_critical_potentials(q)
    >>> print L1,L2,L3
    2.87584628665 2.57731605358 -0.082438791412
    
    Far away from the critical potentials, the polar and equatorial values are
    similar:
    
    >>> Rpole = potential2radius(2*L1,q,component=1,loc='pole')
    >>> Req = potential2radius(2*L1,q,component=1,loc='eq')
    >>> print Rpole,Req
    0.190096394806 0.192267598622
    
    At the critical potential, this is no longer the case.
    
    >>> Rpole = potential2radius(L1,q,component=1,loc='pole')
    >>> Req = potential2radius(L1,q,component=1,loc='eq')
    >>> print Rpole,Req
    0.414264812638 0.57038700404
    
    Example for primary and secondary component, in a system with C{q=0.45} and
    C{sma=32} Rsol:
    
    >>> Req1 = potential2radius(14.75,0.45,loc='eq',sma=32,component=1)
    >>> Req2 = potential2radius( 6.00,0.45,loc='eq',sma=32,component=2)
    >>> print Req1,Req2
    2.23868846442 3.0584436335
    
    @param pot: Roche potential value (unitless)
    @type pot: float
    @param q: mass ratio
    @type q: float
    @param d: separation (in units of semi-major axis)
    @type d: float
    @param F: synchronicity parameter
    @type F: float
    @param component: component in the system (1 or 2)
    @type component: integer
    @param sma: semi-major axis
    @type sma: value of the semi-major axis.
    @param loc: location of the radius ('pole' or 'eq')
    @type loc: str
    @return: potential value for a certain radius
    @rtype: float
    """
    if 'pol' in loc.lower():
        theta = 0
    elif 'eq' in loc.lower():
        theta = np.pi/2.
    else:
        ValueError,"don't understand loc=%s"%(loc)
    try:
        r_pole = newton(binary_potential,1e-5,args=(theta,0,pot,q,d,F,component),tol=tol,maxiter=maxiter)
    except RuntimeError:
        raise ValueError("Failed to converge, potential {} is probably too low".format(pot))
    return r_pole*sma

def radius2potential(radius,q,d=1.,F=1.,component=1,sma=1.,loc='pole',tol=1e-10,maxiter=50):
    """
    Convert a radius to a potential value.
    
    Radius should be in units of C{sma}: if you want to give radius in Rsol,
    make sure C{sma} is also given in Rsol. If you leave C{sma=1}, you need to
    give C{radius} as a fraction of C{sma}.
    
    For eccentric orbits, the radius/potential are dependent on the orbital
    phase. You should explicitly set C{d} to the instantaneous seperation in
    units of semi-major axis. Setting C{d=1} gives the potential at periastron.
    
    >>> M1,M2 = 1.5,0.75
    >>> q = M2/M1
    >>> L1,L2,L3 = calculate_critical_potentials(q)
    >>> print L1,L2,L3
    2.87584628665 2.57731605358 -0.082438791412
    
    In this case, a polar radius of 0.5 sma units has a lower potential than
    the critical potential, so the system should be in Roche lobe overflow.
    
    >>> pot1 = radius2potential(0.5,q,component=1,loc='pole')
    >>> pot2 = radius2potential(0.5,q,component=1,loc='eq')
    >>> print pot1,pot2
    2.4472135955 2.9375
    
    An inverse example of L{potential2radius}:
    
    >>> Req1 = potential2radius(14.75,0.45,loc='eq',sma=32.,component=1)
    >>> Req2 = potential2radius( 6.00,0.45,loc='eq',sma=32.,component=2)
    
    >>> pot1 = radius2potential(Req1,0.45,loc='eq',sma=32.,component=1)
    >>> pot2 = radius2potential(Req2,0.45,loc='eq',sma=32.,component=2)
    
    >>> print Req1,Req2
    2.23868846442 3.0584436335
    >>> print pot1,pot2
    14.75 6.0
        
    @param radius: radius (in same units as C{sma})
    @type radius: float
    @param q: mass ratio
    @type q: float
    @param d: separation (in units of semi-major axis)
    @type d: float
    @param F: synchronicity parameter
    @type F: float
    @param component: component in the system (1 or 2)
    @type component: integer
    @param sma: semi-major axis
    @type sma: value of the semi-major axis.
    @param loc: location of the radius ('pole' or 'eq')
    @type loc: str
    @return: radius corresponding to the potential
    @rtype: float
    """
    if 'pol' in loc.lower():
        theta = 0
    elif 'eq' in loc.lower():
        theta = np.pi/2.
    else:
        ValueError,"don't understand loc=%s"%(loc)
    def _binary_potential(Phi,r,theta,phi,q,d,F,component=1):
        return binary_potential(r,theta,phi,Phi,q,d,F,component=component)
    pot = newton(_binary_potential,1000.,args=(radius/sma,theta,0.,q,d,F,component),tol=tol,maxiter=maxiter)
    return pot

#}

#{ Binaries

def change_component(q,Phi):
    """
    Change mass ratio and value of the potential into the frame of the other
    component.
    
    >>> q = 0.736
    >>> Phi = 7.345
    >>> q,Phi = change_component(q,Phi)
    >>> print q,Phi
    1.35869565217 9.80027173913
    >>> q,Phi = change_component(q,Phi)
    >>> print q,Phi
    0.736 7.345
    
    @param q: mass ratio
    @type q: float
    @param Phi: Roche potential value (unitless)
    @type Phi: float
    """
    Phi = Phi/q + 0.5*(q-1.0)/q
    q = 1.0/q
    return q,Phi

def binary_surface_gravity(x,y,z,d,omega,mass1,mass2,normalize=False):
    """
    Calculate surface gravity in an eccentric asynchronous binary roche potential.
    
    Give everything in SI units please...
    
    @param x: cartesian x coordinate
    @type x: float
    @param y: cartesian y coordinate
    @type y: float
    @param z: cartesian z coordinate
    @type z: float
    @param d: separation between the two components
    @type d: float
    @param omega: rotation vector
    @type omega: float
    @param mass1: primary mass
    @type mass1: float
    @param mass2: secondary mass
    @type mass2: float
    """
    q = mass2/mass1
    x_com = q*d/(1+q)
    
    r = np.array([x,y,z])
    d_cf = np.array([d-x_com,0,0])
    d = np.array([d,0,0])
    h = d - r

    term1 = - constants.GG*mass1/np.linalg.norm(r)**3*r
    term2 = - constants.GG*mass2/np.linalg.norm(h)**3*h
    term3 = - omega**2 * d_cf
    
    g_pole = term1 + term2 + term3
    if normalize:
        return np.linalg.norm(g_pole)
    else:
        return g_pole

def binary_potential(r,theta,phi,Phi,q,d,F,component=1):
    """
    Unitless eccentric asynchronous Roche potential in spherical coordinates.
    
    See Wilson, 1979.
    
    The  synchronicity parameter F is 1 for synchronised circular orbits. For
    pseudo-synchronous eccentrical orbits, it is equal to (Hut, 1981)
    
    F = sqrt( (1+e)/ (1-e)^3)
    
    Periastron is reached when d = 1-e.
        
    @param r: radius of Roche volume at potential Phi (in units of semi-major axis)
    @type r: float
    @param theta: colatitude (0 at the pole, pi/2 at the equator)
    @type theta: float
    @param phi: longitude (0 in direction of COM)
    @type phi: float
    @param Phi: Roche potential value (unitless)
    @type Phi: float
    @param q: mass ratio
    @type q: float
    @param d: separation (in units of semi-major axis)
    @type d: float
    @param F: synchronicity parameter
    @type F: float
    @param component: component in the system (1 or 2)
    @type component: integer
    @return: residu between Phi and roche potential
    @rtype: float
    """
    #-- transform into system of component (defaults to primary)
    if component==2:
        q,Phi = change_component(q,Phi)
    lam,nu = cos(phi)*sin(theta),cos(theta)
    term1 = 1. / r
    term2 = q * ( 1./sqrt(d**2 - 2*lam*d*r + r**2) - lam*r/d**2)
    term3 = 0.5 * F**2 * (q+1) * r**2 * (1-nu**2)
    return (Phi - (term1 + term2 + term3))

def binary_potential_gradient(x,y,z,q,d,F,component=1,normalize=False):
    """
    Gradient of eccenctric asynchronous Roche potential in cartesian coordinates.
    
    See Phoebe scientific reference, http://phoebe.fiz.uni-lj.si/docs/phoebe_science.ps.gz
    
    x,y,z,d in real units! (otherwise you have to scale it yourself)
    
    @param x: x-axis
    @type x: float'
    @param y: y-axis
    @type y: float'
    @param z: z-axis
    @type z: float'
    @param q: mass ratio
    @type q: float
    @param d: separation (in units of semi-major axis)
    @type d: float
    @param F: synchronicity parameter
    @type F: float
    @param component: component in the system (1 or 2)
    @type component: integer
    @param normalize: flag to return magnitude (True) or vector form (False)
    @type normalize: boolean
    @return: Roche potential gradient
    @rtype: ndarray or float
    """
    #-- transform into system of component (defaults to primary)
    if component==2:
        q = 1.0/q
    r = np.sqrt(x**2 + y**2 + z**2)
    r_= np.sqrt((d-x)**2 + y**2 + z**2)
    dOmega_dx = - x / r**3 + q * (d-x) / r_**3 + F**2 * (1+q)*x - q/d**2
    dOmega_dy = - y / r**3 - q * y     / r_**3 + F**2 * (1+q)*y
    dOmega_dz = - z / r**3 - q * z     / r_**3
    
    dOmega = np.array([dOmega_dx,dOmega_dy,dOmega_dz])
    
    if normalize:
        return np.linalg.norm(dOmega)
    else:
        return dOmega

def improve_potential(V,V0,pot0,rpole0,d,q,F,sma=1.):
    """
    Return an approximate value for the potential in eccentric orbit.
    
    Given the volume V0 at a reference time t0, this function returns
    the approximate value for a new potential at another time at t1, to
    conserve the volume. At t1, the volume at the potential V(pot=pot(t0))
    equals V.
    
    The value is calculated as follows::
    
        Pot(t) = DeltaPot + Pot(t0)
        Pot(t) = DeltaPot/DeltaR * DeltaR + Pot(t0)
    
    We know that::
    
        DeltaV/DeltaR = 4*pi*R**2
        DeltaR = DeltaV/4*pi*R**2
        
    We compute DeltaPot/DeltaR (the gradient of the potential) at the
    pole with L{marching.dBinaryRochedz}. Thus::
    
        Pot(t) = DeltaPot/DeltaR * DeltaR + Pot(t0)
    
    If you give all values (V,V0,rpole0 and d) in Rsol, you need to give
    sma in Rsol also.
    
    @param V: reference volume, in units of sma (or give sma).
    @param V0: reference volume, in units of sma  (or give sma).
    @param rpole0: polar radius, in units of sma  (or give sma).
    @param d: separation, in units of sma  (or give sma).
    @param q: mass ratio M2/M1
    @param F: synchronicity parameter.
    """
    #-- put everything in the right units
    V0 = V0/sma**3
    V  = V /sma**3
    R = rpole0/sma
    d = d/sma
    
    #-- compute change in potential at the pole
    grad = -marching.dBinaryRochedz([0,0,R],d,q,F)
    DeltaR = (V-V0)/(4*np.pi*R**2)
    DeltaPot = DeltaR*grad
    
    newPot = DeltaPot+pot0
    return newPot



#}

#{ Rotating stars

def critical_angular_frequency(M,R_pole):
    """
    Compute the critical angular frequency (rad/s).
    
    Definition taken from Cranmer and Owocki, 1995 and equal to
    
    Omega_crit = sqrt( 8GM / 27Rp**3 )
    
    Example usage (includes conversion to period in days):
    
    >>> Omega = critical_angular_frequency(1.,1.)
    >>> P = 2*pi/Omega
    >>> print 'Critical rotation period of the Sun: %.3f days'%(P/(24*3600))
    Critical rotation period of the Sun: 0.213 days
        
    @param M: mass (solar masses)
    @type M: float
    @param R_pole: polar radius (solar radii)
    @type R_pole: float
    @return: critical angular velocity in rad/s
    @rtype: float
    """
    M = M*constants.Msol
    R = R_pole*constants.Rsol
    omega_crit = np.sqrt( 8*constants.GG*M / (27.*R**3))
    return omega_crit

def critical_velocity(M,R_pole):
    """
    Compute the critical velocity (km/s)
    
    Definition 1 from Cranmer and Owocki, 1995:
    
    v_c = 2 pi R_eq(omega_c) * omega_c
    
    Definition 2 from Townsend 2004:
    
    v_c = sqrt ( 2GM/3Rp )
    
    which both amount to the same value.
    
    >>> critical_velocity(1.,1.)
    356663.88096223801
    
    @param M: mass (solar masses)
    @type M: float
    @param R_pole: polar radius (solar radii)
    @type R_pole: float
    @return: critical velocity in m/s
    @rtype: float
    """
    veq = np.sqrt( 2*constants.GG * M*constants.Msol / (3*R_pole*constants.Rsol))
    return veq

def fast_rotation_radius(colat,r_pole,omega):
    Rstar = 3*r_pole/(omega*sin(colat)) * cos((np.pi + np.arccos(omega*sin(colat)))/3.)
    #-- solve singularities
    if np.isinf(Rstar) or sin(colat)<1e-10:
        Rstar = r_pole
    return Rstar

def diffrotlaw_to_internal(omega_pole,omega_eq):
    """
    Omega's in units of critical angular frequency.
    
    Assuming a differential rotation law of the form
    
    oemga = b1 + b2*s**2
    
    with s the distance to the rotational axis, compute the value of b2 given
    the polar and equatorial rotation period in units of critical angular
    velocity (in the case of solid body rotation).
    """
    #-- in dimensional units, the critical angular velocity is sqrt(8/27).
    omega_pole = omega_pole*0.54433105395181736
    omega_eq = omega_eq*0.54433105395181736
    b1 = omega_pole
    b2 = -0.1 if omega_eq<omega_pole else +0.1
    b3 = 0.
    def funczero(b2,omega_eq,b1):
        r0 = -marching.projectOntoPotential(np.array((-0.02, 0.0, 0.0)), 'DiffRotateRoche', b1,b2,b3,1.0).r[0]
        if np.isnan(b2):
            raise ValueError("Impossible differential rotation")
        return (omega_eq-b1)/r0**2-b2
    return newton(funczero,b2,args=(omega_eq,b1))


#}