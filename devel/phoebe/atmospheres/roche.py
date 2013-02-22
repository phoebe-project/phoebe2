"""
Derive parameters in the Roche approximation.

Most of these parameters are local atmospheric parameters.
"""
import logging
import numpy as np
from scipy.optimize import newton
from phoebe.algorithms import marching
from phoebe.units import constants

logger = logging.getLogger('ATM.ROCHE')


#{ General

def temperature(system):
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
    logger.info("derived effective temperature (%.3f <= teff <= %.3f, Tp=%.3f)"%(system.mesh['teff'].min(),system.mesh['teff'].max(),Tpole))
    
    
    
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

#{ Rotation

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