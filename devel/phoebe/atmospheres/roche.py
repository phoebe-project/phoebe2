r"""
Derive parameters in the Roche approximation.

Most of these parameters are local atmospheric parameters.

**Binaries**

.. autosummary::

    approximate_lagrangian_points
    calculate_critical_potentials
    calculate_critical_radius
    potential2radius
    radius2potential
    change_component
    binary_surface_gravity
    binary_potential
    binary_potential_gradient
    improve_potential

**Single stars**

.. autosummary::

    critical_angular_frequency
    critical_velocity
    fast_rotation_radius
    diffrotlaw_to_internal
    

**Phoebe specific:**

.. autosummary::

    temperature_zeipel
    temperature_espinosa
    zeipel_gravb_binary
    claret_gravb


**Gravity darkening**

And now for some unstructured gibberish that only I understand. Mental notes
they call it.

Our definition of the gravity darkening parameter is a bolometric definition,
and parameterized via :math:`\beta` (Eq. 4.23 in the Phoebe Scientific reference):

.. math::
    
        T_\mathrm{eff}^4 = T_\mathrm{pole}^4  \left(\frac{g}{g_\mathrm{pole}}\right)^\beta
        
with

    - :math:`\beta = 1.00` for radiative atmospheres (:math:`T_\mathrm{eff}> 8000\,\mathrm{K}`)
    - :math:`\beta = 0.32` for convective atmospheres (:math:`T_\mathrm{eff}< 5000\,\mathrm{K}`, but we're a little bit less strict and set 6600K)
        
More generally, gravity darkening relates effective temperature and surface gravity.
[Espinosa2011]_ derived values bla bla. The values from [Espinosa2012]_ are
0.25*our value.

Claret defines passband surface gravities bla bla. If interpreted bolometrically
(which is wrong), they are equal to our values (no times 4 or anything).
        


   

"""
import logging
import os
import numpy as np
from numpy import cos,sin,sqrt,pi,log,tan
from scipy.optimize import newton, brentq
from scipy.interpolate import RectBivariateSpline
from phoebe.algorithms import marching
from phoebe.algorithms import interp_nDgrid
from phoebe.units import constants
from phoebe.utils import coordinates
from phoebe.utils import decorators
from phoebe.atmospheres import froche
import time

logger = logging.getLogger('ATM.ROCHE')
basedir = os.path.dirname(os.path.abspath(__file__))

#{ General

def temperature_zeipel(system):
    r"""
    Calculate local temperature according to von Zeipel's law.
    
    See e.g. [VonZeipel1924]_ and [Lucy1967]_.
    
    This law assumes the star is spherically symmetric.
    
    :math:`\beta` is called the gravity darkening (or brightening) parameter.
    
    Definition [Eq. 4.23 in Phoebe Scientific Reference]:
    
    .. math::
    
        T_\mathrm{eff}^4 = T_\mathrm{pole}^4  \left(\frac{g}{g_\mathrm{pole}}\right)^{\beta}
        
    with
    
        - :math:`\beta/\mathrm{gravb} = 1.00` for radiative atmospheres (:math:`T_\mathrm{eff}\geq 8000\,\mathrm{K}`)
        - :math:`\beta/\mathrm{gravb} = 0.32` for convective atmpsheres (:math:`T_\mathrm{eff}\leq 5000\,\mathrm{K}`, but we're a little bit less strict and set 6600K)
    
    Body C{system} needs to have the following parameters defined:
        
        - ``teff`` or ``teffpolar``
        - ``gravb``
        - ``g_pole``
    
    @param system: object to compute temperature of
    @type system: Body
    """
    # Necessary global information: this should be the first parameterSet in an
    # Ordered dictionary
    body = list(system.params.values())[0]
    if body.has_qualifier('teffpolar'):
        Teff = body['teffpolar']
        typ = 'polar'
    else:
        Teff = body['teff']
        typ = 'mean'
    
    # We need the gravity brightening parameter and the polar surface gravity
    # as a reference point
    beta = body.request_value('gravb')
    gp = body.request_value('g_pole')
    
    # Consistency check for gravity brightening
    if Teff >= 8000. and beta < 0.9:
        logger.info('Object probably has a radiative atm (Teff={:.0f}K>8000K), for which gravb=1.00 might be a better approx than gravb={:.2f}'.format(Teff,beta))
    elif Teff <= 6600. and beta >= 0.9:
        logger.info('Object probably has a convective atm (Teff={:.0f}K<6600K), for which gravb=0.32 might be a better approx than gravb={:.2f}'.format(Teff,beta))
    elif beta < 0.32 or beta > 1.00:
        logger.info('Object has intermittent temperature, gravb should be between 0.32-1.00')
        
    # Compute G and Tpole
    Grav = abs(10**(system.mesh['logg']-2)/gp)**beta
    
    if typ == 'mean':
        Tpole = Teff*(np.sum(system.mesh['size']) / np.sum(Grav*system.mesh['size']))**(0.25)
    elif typ == 'polar':
        Tpole = Teff
    else:
        raise ValueError("Cannot interpret temperature type '{}' (needs to be one of ['mean','polar'])".format(type))
        
    # Now we can compute the local temperatures. We add a constraint to the body
    # so that we can access the polar temperature if needed
    #system.mesh['teff'] = (Grav**0.25 * Tpole) -- sometimes takes very long for some reason
    system.mesh['teff'] = (Grav * Tpole**4)**0.25
    
    #=== FORTRAN ATTEMPT DOESN'T GO ANY FASTER =======
    #Tpole,system.mesh['teff'] = froche.temperature_zeipel(system.mesh['logg'],
    #                   system.mesh['size'], Teff, ['mean','polar'].index(type),
    #                   beta, gp)
    body.add_constraint('{{t_pole}} = {0:.16g}'.format(Tpole))
    logger.info("derived effective temperature (Zeipel) (%.3f <= teff <= %.3f, Tp=%.3f)"%(system.mesh['teff'].min(),system.mesh['teff'].max(),Tpole))


def temperature_espinosa(system):
    r"""
    Calculate local temperature according to [Espinosa2011]_.
    
    This model doesn't need a gravity darkening parameter, and assumes a single,
    rotating star. The effecive temperature is computed via
    
    .. math::
    
        T_\mathrm{eff}^4 =  \left(\frac{L}{4\pi\sigma R_e^2}\right) \sqrt{\tilde{r}^{-4} + \omega^4\tilde{r}^2\sin^2\theta - \frac{2\omega^2\sin^2\theta}{\tilde{r}}}\quad  \frac{\tan^2\hat{\theta}}{\tan^2\theta}
        
        \cos\hat{\theta} + \log\tan\frac{\hat{\theta}}{2} - \frac{\omega^2\tilde{r}^3\cos^3\theta}{3} - \cos\theta - \log\tan\frac{\theta}{2} = 0
        
    with :math:`\tilde{r} = r/R_e` the radius coordinate relative to the equatorial radius, and :math:`\theta` the colatitude.
    
    Let's slightly rephrase some of the key sentences from [Espinosa2011]_:
    
    They assume that the flux :math:`\mathbf{F}` in the envelope of a rotating
    star is very close to
    
    .. math::
    
        \mathbf{F} = - f(r, \theta) \mathbf{g}_\mathrm{eff}
    
    Thus the energy flux vector is almost antiparallel to the effective gravity.
    The authors claim this to be justified in a convective atmosphere, where
    heat transport is mainly in the vertical direction, because convective
    bubbles rise against gravity. In radiative regions, they say the flux is
    antiparallel to the temperature gradient, which is slightly off the
    potential gradient. Still, they say that their models have shown that this
    deviation is very small, even for very fast rotation.
    
    @param system: object to compute temperature of
    @type system: Body
    """
    # Get some global and local parameters
    M = system.params['star'].request_value('mass', 'kg')
    r_pole = system.params['star'].request_value('radius', 'm')
    teffpolar = system.params['star'].request_value('teff')
    
    # Rotation frequency in units of critical rotation frequency
    omega_rot = 2*np.pi / system.params['star'].request_value('rotperiod', 's')
    Omega_crit = np.sqrt( 8*constants.GG*M / (27.*r_pole**3))
    omega = omega_rot / Omega_crit
    
    # Compute equatorial radius
    Re = fast_rotation_radius(np.pi/2.0, r_pole, omega)
    
    # But define in terms of Keplerian instead of Roche (the former is used in
    # the paper
    omega = omega_rot * np.sqrt(Re**3 / (constants.GG*M))
    
    # Surface coordinates
    index = np.array([1, 0, 2])
    r, phi, theta = coordinates.cart2spher_coord(*system.mesh['_o_center'].T[index])
    cr = r * constants.Rsol/Re
    cr_pole = r_pole / Re
    
    # Python version: find var_theta for all points on the surface:
    #def funczero(var_theta,cr,theta):
    #    sol = cos(var_theta)+log(tan(0.5*var_theta))-1./3.*omega**2*cr**3*cos(theta)**3 - cos(theta)-log(tan(0.5*theta))
    #    return sol
    #var_theta = np.array([newton(funczero,itheta,args=(icr,itheta)) for icr,itheta in zip(cr,theta)])
    
    # Find var_theta for all points on the surface (Fortran version):
    var_theta = froche.espinosa(cr, theta, omega)
    
    # We scale everything to the polar effective temperature
    factor_pole = np.exp(0.5/3. * omega**2*cr_pole**3)
    
    # Funczero is not defined for the equator and pole
    factor_eq = (1 - omega**2)**(-0.5/3.)
    scale = teffpolar * np.sqrt(cr_pole)/factor_pole
    
    # Now compute the effective temperature, but we include an empirical
    # correction for when tan(x)/tan(y) becomes very large (or tan(y) becomes
    # zero)
    factor = np.where(np.abs(theta-np.pi/2)<0.002, factor_eq, sqrt(tan(var_theta) / tan(theta)))
    teff = scale * (cr**-4 + omega**4*cr**2*sin(theta)**2-2*omega**2*sin(theta)**2/cr)**0.125*factor
    
    # And put it in the mesh
    system.mesh['teff'] = teff
    
    logger.info("derived effective temperature (Espinosa) (%.3f <= teff <= %.3f)"%(system.mesh['teff'].min(),system.mesh['teff'].max()))



@decorators.memoized
def zeipel_gravb_binary():
    """
    Return an interpolator to derive the approximate Zeipel gravity brightening.
    
    See [Espinosa2012]_.
    
    """
    data = np.loadtxt(os.path.join(basedir, 'tables', 'gravb', 'espinosa.dat'))
    x, y = np.log10(data[1:,0]), data[0,1:]
    z = 4.0*data[1:,1:]
    return RectBivariateSpline(x, y, z)


@decorators.memoized
def claret_gravb():
    """
    Return a grid to derive the Zeipel gravity brightening.
    
    See [Claret2012]_.
    
    You can interpolate the gravity brightening coefficients in the following way:
    
    >>> axis_values, pixgrid = claret_gravb()
    >>> gravb = interp_nDgrid.interpolate([teff, logg, abun], axis_values, pixgrid)
    >>> print(gravb)
    
    The variables ``teff``, ``logg`` and ``abun`` must be lists or arrays, even
    if you only have one variable.
    """
    data = np.loadtxt(os.path.join(basedir, 'tables', 'gravb', 'claret.dat')).T
    # rescale metallicity to solar value and take logarithm
    data[2] = np.log10(data[2]/constants.z_solar)
    axv, pix = interp_nDgrid.create_pixeltypegrid(data[:3], data[3:])
    return axv, pix

    
def approximate_lagrangian_points(q, sma=1.):
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
    >>> print(L1/1000.)
    1491685.37444
    
    The Earth/Moon L1 is located about 60.000 km from the Moon:
    
    >>> dlun = 384499. # km
    >>> L1,L2,L3 = approximate_lagrangian_points(constants.Mlun/constants.Mearth,sma=dlun)
    >>> print(L1)
    58032.1840451
    
    @parameter q: mass ratio M2/M1
    @type q: float
    @parameter sma: system semi-major axis
    @type sma: float
    @return: L1, L2, L3 (zero is center of primary, d is center of secondary)
    @rtype: float,float,float
    """
    raise NotImplementedError
    if q <= 1:
        mu = q / (1.0 + q)
    else:
        mu = 1.0 / (1.0 + q)
    
    z = (mu/3.)**(1./3.)
    x_L1 = z - 1./3.*z**2 - 1./9.*z**3 + 58./81.*z**4
    x_L2 = z + 1./3.*z**2 - 1./9.*z**3 + 50./81.*z**4
    x_L3 = 1. - 7./12.*mu - 1127./20736.*mu**3 - 7889./248832*mu**4
    
    if q <= 1:
        return (x_L1)*sma, (x_L2)*sma, (x_L3)*sma
    else:
        return (1.0-x_L1)*sma, (1-x_L2)*sma, (1-x_L3)*sma


def exact_lagrangian_points(q, F=1.0, d=1.0, sma=1.):
    """
    Exact Langrangian points L1, L2 and L3.
    
    Give C{d} as a fraction of semi-major axis. The units of the lagrangian
    points that you get out are C{sma}. So if you give the C{sma} in km, then
    you get the lagrangian points in km (see examples below).
    
    @parameter q: mass ratio M2/M1
    @type q: float
    @parameter sma: system semi-major axis
    @type sma: float
    @return: L1, L2, L3 (zero is center of primary, d is center of secondary)
    @rtype: float,float,float
    """
    
    dxL = 1.0
    L1 = 1e-3
    while abs(dxL) > 1e-6:
        dxL = - marching.dBinaryRochedx([L1, 0.0, 0.0], d, q, F) / marching.d2BinaryRochedx2([L1, 0.0, 0.0], d, q, F)
        L1 = L1 + dxL
    return L1*sma, 0.0, 0.0



def calculate_critical_potentials(q,F=1.,d=1.,component=1):
    """
    Calculate the (approximate) critical potentials in L1, L2 and L3.
    
    Only approximate for synchroneous case.
    
    You need to give orbital separation C{d} as a fraction of the semi-major
    axies. E.g. for circular orbits, C{d=1}.
    
    @parameter q: mass ratio M2/M1
    @type q: float
    @param F: synchronicity parameter
    @type F: float
    @parameter d: separation between the two objects
    @type d: float
    @param component: component in the system (1 or 2)
    @type component: integer
    @return: critical potentials at L1,L2 and L3
    @rtype: float,float,float
    """
    
    
    theta, phi = np.pi/2, 0.0
    if component == 1:
        L1, L2, L3 = exact_lagrangian_points(q, F, d)
        Phi_L1 = -binary_potential(L1, np.pi/2.0, 0.0, 0.0, q, d, F)
    elif component == 2:
        L1, L2, L3 = exact_lagrangian_points(1./q, F, d)
        Phi_L1 = -binary_potential(L1, np.pi/2.0, 0.0, 0.0, 1./q, d, F)
        Phi_L1 = change_component(1./q, Phi_L1)[1]
    Phi_L2 = 0.0
    Phi_L3 = 0.0
    return Phi_L1, Phi_L2, Phi_L3    


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
    if loc=='eq':
        rL1 = exact_lagrangian_points(q, F=F, d=d, sma=sma)[0]
    else:
        pL1, pL2, pL3 = calculate_critical_potentials(q ,F=F, d=d)
        rL1 = potential2radius(pL1, q, d=d,F=F, sma=sma, loc=loc, component=component)
    return rL1
    

def potential2radius(pot, q, d=1, F=1.0, component=1, sma=1.0, loc='pole',
                     tol=1e-10, maxiter=50):
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

def radius2potential(radius, q, d=1., F=1., component=1, sma=1., loc='pole',
                     tol=1e-10, maxiter=50):
    r"""
    Convert a radius to a potential value.
    
    Radius should be in units of C{sma}: if you want to give radius in Rsol,
    make sure C{sma} is also given in Rsol. If you leave C{sma=1}, you need to
    give C{radius} as a fraction of C{sma}.
    
    For eccentric orbits, the radius/potential are dependent on the orbital
    phase. You should explicitly set C{d} to the instantaneous seperation in
    units of semi-major axis. Setting C{d=1-ecc} gives the potential at
    periastron.
    
    The location in the star to convert the potential to radius is given by
    envvar:`loc`. If ``eq`` or ``pole``, then those radii will be returned. For
    arbitrary angles, :envvar:`loc` can also be given as a tuple of direction
    cosines;
    
    .. math::
    
        \lambda = \sin\theta\cos\phi
        \nu = \cos\theta
    
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
    @param loc: location of the radius ('pole' or 'eq') or (lambda,nu) coordinates (directional cosines)
    @type loc: str or tuple
    @return: radius corresponding to the potential
    @rtype: float
    """
    try:
        # Magic strings
        loc = loc.lower()
        if 'pol' in loc:
            theta = 0
        elif 'eq' in loc:
            theta = np.pi/2.
        else:
            ValueError,"don't understand loc=%s"%(loc)
    except AttributeError:
        # Direction cosines
        lam, nu = loc
        theta = np.arccos(nu)
        phi = np.arccos(lam / np.sqrt(1-nu**2))
        
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
    r"""
    Calculate surface gravity in an eccentric asynchronous binary roche potential.
    
    If ``x=0``, ``y=0`` and ``z=Rpole``, this compute the polar surface gravity:
    
    .. math::
    
        \mathbf{g}_p = -\frac{GM_1}{r^2_p} \frac{\mathbf{r_p}}{r_p} - \frac{GM_2}{h^2}\frac{\mathbf{h}}{h} - \omega^2(t) d_\mathrm{cf}\frac{\mathbf{d_\mathrm{cf}}}{d_\mathrm{cf}}
    
    with,
    
    .. math::
        
        h = \sqrt{r_p^2 + d^2}
    
    and :math:`d_\mathrm{cf}` is the perpendicular distance from the star's pole
    to the axis of rotation of the binary. For misaligned binaries, we neglect
    the influence of the precession on the surface gravity for now. This should
    be very small anyway.
        
    
    Reference: Phoebe Scientific reference.
    
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
    #q = mass1/mass2
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

def misaligned_binary_surface_gravity(x,y,z,d,omega,mass1,mass2,normalize=False,F=1.,Rpole=None):
    r"""
    Calculate surface gravity in a misaligned binary roche potential.
    
    If ``x=0``, ``y=0`` and ``z=Rpole``, this compute the polar surface gravity:
    
    .. math::
    
        \mathbf{g}_p = -\frac{GM_1}{r^2_p} \frac{\mathbf{r_p}}{r_p} - \frac{GM_2}{h^2}\frac{\mathbf{h}}{h} - \omega^2(t) d_\mathrm{cf}\frac{\mathbf{d_\mathrm{cf}}}{d_\mathrm{cf}}
    
    with,
    
    .. math::
        
        h = \sqrt{r_p^2 + d^2}
    
    and :math:`d_\mathrm{cf}` is the perpendicular distance from the star's pole
    to the axis of rotation of the binary. For misaligned binaries, we neglect
    the influence of the precession on the surface gravity for now. This should
    be very small anyway. But we also neglect any orbitally induces term of
    the surface gravity, which is probably not a good idea.
        
    
    Reference: Phoebe Scientific reference.
    
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
    shapearr1 = np.ones_like(x)
    shapearr2 = np.zeros_like(x)
    x_com = np.array([q*d/(1+q)*shapearr1,shapearr2,shapearr2])
    
    if Rpole is None:
        Rpole = np.array([shapearr2,shapearr2,shapearr1])
    
    r = np.array([x,y,z])
    d = np.array([d*shapearr1,shapearr2,shapearr2])
    d_cf = r-x_com
    d_cf[2] = 0
    h = d - r
    
    r_cf = np.cross(r.T,Rpole.T).T
    
    #import matplotlib.pyplot as plt
    #plt.plot(x/constants.Rsol,y/constants.Rsol,'ro')
    #from mayavi import mlab
    #mlab.figure(1)
    #mlab.points3d(*np.array([x,y,z]).reshape((3,-1))/constants.Rsol,color=(0,1,0))
    #mlab.points3d(*d_cf.reshape((3,-1))/constants.Rsol,color=(0,1,1))
    #mlab.points3d(*d.reshape((3,-1))/constants.Rsol,color=(0,0,1))
    #mlab.plot3d(np.array([0,d[0]])/constants.Rsol,np.array([0,d[1]]),np.array([0,d[2]]))
    #mlab.plot3d(2*np.array([0,x])/constants.Rsol,2*np.array([0,y])/constants.Rsol,2*np.array([0,z])/constants.Rsol)
    #mlab.plot3d(np.array([x,d[0]])/constants.Rsol,
                #np.array([y,d[1]])/constants.Rsol,
                #np.array([z,d[2]])/constants.Rsol)
    term1 = - constants.GG*mass1/coordinates.norm(r)**3*r
    term2 = - constants.GG*mass2/coordinates.norm(h)**3*h
    term3 = - (omega)**2 * d_cf
    term4 = - (omega*F)**2 * r_cf
    
    #print (np.log10(coordinates.norm(term1+term2+term3+term4))+2).min()
    #print (np.log10(coordinates.norm(term1+term2-term3+term4))+2).min()
    #print (np.log10(coordinates.norm(term1+term2-term3-term4))+2).min()
    #print (np.log10(coordinates.norm(term1+term2+term3-term4))+2).min()
    ##print '=========='
    #print term1.min(),term1.max()
    #print term2.min(),term2.max()
    #print term3.min(),term3.max()
    #print '=========='
    
    #import matplotlib.pyplot as plt
    #try:
        #if len(r_cf[0])>1:
            #g = np.log10(coordinates.norm(term1+term2+term3+term4))+2
            #print g.min(),g.max()
            #plt.figure()
            #plt.plot(r[2],g,'ko-')
            
            #plt.figure()
            #plt.subplot(111,aspect='equal')
            #n = coordinates.norm(r).max()
            #plt.scatter(x/n,y/n,c=coordinates.norm(r)/n,edgecolors='none')
            #plt.colorbar()
            
            
            #plt.figure()
            #plt.subplot(131,aspect='equal')
            #plt.plot(x,y,'ko')
            #plt.xlabel('x');plt.ylabel('y')
            #plt.subplot(132,aspect='equal')
            #plt.plot(x,z,'ko')
            #plt.xlabel('x');plt.ylabel('z')
            #plt.subplot(133,aspect='equal')
            #plt.plot(y,z,'ko')
            #plt.xlabel('y');plt.ylabel('z')
            
            #plt.figure()
            #plt.subplot(221,aspect='equal')
            #plt.scatter(x,y,c=np.log10(coordinates.norm(term1))+2,edgecolors='none')
            #plt.colorbar()
            #plt.subplot(222,aspect='equal')
            #r_cfn = coordinates.norm(r_cf)
            #plt.scatter(x,y,c=r_cfn/r_cfn.max(),edgecolors='none')
            #plt.colorbar()
            #plt.subplot(223,aspect='equal')
            #plt.scatter(x,y,c=np.log10(coordinates.norm(term2))+2,edgecolors='none')
            #plt.colorbar()
            #plt.subplot(224,aspect='equal')
            #plt.scatter(x,y,c=np.log10(coordinates.norm(term3))+2,edgecolors='none')
            #plt.colorbar()
            #plt.show()
            
    #except Exception, msg:
        #print msg
        #pass
    
    g_pole = term1 + term2 + term3 + term4
    if normalize:
        return coordinates.norm(g_pole)
    elif normalize is None:
        return coordinates.norm(term1),coordinates.norm(term2),coordinates.norm(term3),coordinates.norm(g_pole)
    else:
        return g_pole

def binary_potential(r,theta,phi,Phi,q,d,F,component=1):
    r"""
    Unitless eccentric asynchronous Roche potential in spherical coordinates.
    
    See [Wilson1979]_.
    
    The  synchronicity parameter F is 1 for synchronised circular orbits. For
    pseudo-synchronous eccentrical orbits, it is equal to [Hut1981]_
    
    .. math::
    
        F = \sqrt{ \frac{(1+e)}{(1-e)^3}}
    
    Periastron is reached when :math:`d = 1-e`.
        
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

def binary_potential_gradient(x, y, z, q, d, F, component=1, normalize=False,
                              output_type='array'):
    """
    Gradient of eccenctric asynchronous Roche potential in cartesian coordinates.
    
    See Phoebe scientific reference, http://phoebe.fiz.uni-lj.si/docs/phoebe_science.ps.gz
    
    x,y,z,d in real units! (otherwise you have to scale it yourself)
    
    by specifying output_type as list, you can avoid  unnecessary array conversions.
    
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
    # Transform into system of component (defaults to primary)
    if component == 2:
        q = 1.0/q
    y2 = y**2
    z2 = z**2
    y2pz2 = y2 + z2
    r3 = np.sqrt(x**2 + y2pz2)**3
    r3_= np.sqrt((d-x)**2 + y2pz2)**3
    dOmega_dx = - x / r3 + q * (d-x) / r3_ + F**2 * (1+q)*x - q/d**2
    dOmega_dy = - y / r3 - q * y     / r3_ + F**2 * (1+q)*y
    dOmega_dz = - z / r3 - q * z     / r3_
    
    if normalize:
        return np.sqrt(dOmega_dx**2 + dOmega_dy**2 + dOmega_dz**2)
    elif output_type=='array':
        return np.array([dOmega_dx, dOmega_dy, dOmega_dz])
    else:
        return dOmega_dx, dOmega_dy, dOmega_dz

def misaligned_binary_potential_gradient(x,y,z,q,d,F,theta,phi,component=1,normalize=False):
    """
    Gradient of misaligned cicular Roche potential in cartesian coordinates.
    
    x,y,z,d in real units! (otherwise you have to scale it yourself)
    
    I'm including d here, but you better set d==1! Perhaps we can change add
    eccentric misaligned orbits later.
    
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
    @param theta: inclination misalignment parameter
    @type theta: float
    @param phi: phase misalignment parameter
    @type phi: float
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
    x = x/d
    y = y/d
    z = z/d
    d = 1.0
    r = np.sqrt(x**2 + y**2 + z**2)
    r_= np.sqrt((d-x)**2 + y**2 + z**2)
    deltax = 2*(1-np.cos(phi)**2*np.sin(theta)**2)*x -\
                np.sin(theta)**2*np.sin(2*phi)*y -\
                np.sin(2*theta)*np.cos(phi)*z
    deltay = 2*(1-np.sin(phi)**2*np.sin(theta)**2)*y -\
                np.sin(theta)**2*np.sin(2*phi)*x -\
                np.sin(2*theta)*np.sin(phi)*z
    deltaz = 2*np.sin(theta)**2*z -\
            np.sin(2*theta)*np.cos(phi)*x -\
            np.sin(2*theta)*np.sin(phi)*y
    dOmega_dx = - x / r**3 + q * (d-x) / r_**3 + 0.5*F**2 * (1+q)*deltax - q/d**2
    dOmega_dy = - y / r**3 - q *   y   / r_**3 + 0.5*F**2 * (1+q)*deltay
    dOmega_dz = - z / r**3 - q *   z   / r_**3 + 0.5*F**2 * (1+q)*deltaz
    
    dOmega = np.array([dOmega_dx,dOmega_dy,dOmega_dz])
    
    if normalize:
        return np.linalg.norm(dOmega)
    else:
        return dOmega



def improve_potential(V,V0,pot0,rpole0,d,q,F,sma=1.):
    r"""
    Return an approximate value for the potential in eccentric orbit.
    
    Given the volume :math:`V_0` at a reference time :math:`t_0`, this
    function returns the approximate value for a new potential :math:`P(t_1)`
    at another time at :math:`t_1`, to conserve the volume. At :math:`t_1`,
    the volume at the potential :math:`V(P=P(t_0))` equals :math:`V_0`.
    
    The value is calculated as follows:
    
    .. math::
    
        P(t) & = \Delta P + P(t_0)
        
        P(t) & = \left(\frac{\Delta P}{\Delta R}\right) \Delta R + P(t_0)
    
    We know that:
    
    .. math::
    
        \frac{\Delta V}{\Delta R} & = 4\pi R^2
        
        \Delta R & = \frac{\Delta V}{4\pi R^2}
        
    We compute :math:`\Delta P / \Delta R` (the gradient of the potential) at the
    pole with L{marching.dBinaryRochedz}. Thus:
    
    .. math::
    
        P(t) = \left(\frac{\Delta P}{\Delta R}\right) \Delta R + P(t_0)
    
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
    r"""
    Compute the critical angular frequency (rad/s).
    
    Definition taken from [Cranmer1995]_ and equal to
    
    .. math::
    
        \Omega_\mathrm{crit} = \sqrt{\frac{8GM}{27R_p^3}}
    
    with :math:`R_p` the polar radius of the star.
    
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
    r"""
    Compute the critical velocity (km/s)
    
    Definition 1 from [Cranmer1995]_:
    
    .. math::
    
        v_c = 2 \pi R_\mathrm{eq}(\omega_c) \omega_c
    
    Definition 2 from [Townsend2004]_:
    
    .. math::
    
        v_c = \sqrt{\frac{2GM}{3R_p}}
    
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
    r"""
    Compute the radius of a fast rotating star.
    
    E.g. [Cranmer1995]_:
    
    .. math::
    
        R_*(\omega,\theta) = \frac{3R_p}{\omega\sin\theta}\cos\left[\frac{\pi+\arccos(\omega\sin\theta)}{3}\right]
    
    @param colat: colatitude (radians)
    @type colat: float
    @param r_pole: polar radius (whatever units)
    @type r_pole: float
    @param omega: angular rotation frequency as a fraction of the critical rotation frequency
    @type omega: float
    """
    Rstar = 3*r_pole/(omega*sin(colat)) * cos((np.pi + np.arccos(omega*sin(colat)))/3.)
    #-- solve singularities
    if np.isinf(Rstar) or sin(colat)<1e-10:
        Rstar = r_pole
    return Rstar

def diffrotlaw_to_internal(omega_pole,omega_eq):
    """
    Omega's in units of critical angular frequency.
    
    Assuming a differential rotation law of the form
    
    .. math::
    
        \omega = b_1 + b_2 s^2
    
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

if __name__=="__main__":
    import doctest
    doctest.testmod()