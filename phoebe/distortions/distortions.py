
from scipy.optimize import newton

def potential2radius(pot_func, pot, q, d=1, F=1.0, component=1, sma=1.0, loc='pole',
                     tol=1e-10, maxiter=50):
    """
    @param pot_func: the potential function to use
    @type pot_func: func
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

    potential = lambda r, theta, phi, q, d, F, c: binary_potential(r, theta, phi, q, d, F, c)-pot

    try:
        r_pole = newton(potential, x0=1./pot, args=(theta, 0, q, d, F, component), tol=tol, maxiter=maxiter)
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
    
