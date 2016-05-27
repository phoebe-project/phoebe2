
import numpy as np
from numpy import cos, sin, tan, pi, sqrt

from phoebe import c


# make criticial potential (c-wrapper) available from here
from phoebe_roche import critical_potential


from scipy.optimize import newton

def binary_potential(r, theta, phi, q, d, F, component=1):
    r"""
    Unitless eccentric asynchronous surface potential.
    
    See [Wilson1979]_.
    
    The  synchronicity parameter F is 1 for synchronized circular orbits. For
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
    @param q: mass ratio
    @type q: float
    @param d: separation (in units of semi-major axis)
    @type d: float
    @param F: synchronicity parameter
    @type F: float
    @param component: component in the system (1 or 2)
    @type component: integer
    @return: surface potential
    @rtype: float
    """

    lam, nu = cos(phi)*sin(theta), cos(theta)

    if component == 2:
        q = 1./q
    
    pot = 1./r + q*(1./sqrt(d**2-2*lam*d*r+r**2) - lam*r/d**2) + 0.5*F**2*(q+1)*r**2*(1-nu**2)
    if component==2:
        pot = pot/q + 0.5*(q-1)/q
    
    return pot

def binary_potential_derivative(r, theta, phi, q, d, F, component=1):
    """
    Computes a derivative of the potential with respect to r.
    
    @param r:      radius
    @param theta:  colatitude
    @param phi:    longitude
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    
    x, y, z = r*cos(phi)*sin(theta), r*sin(phi)*sin(theta), r*cos(theta)
    r2 = x*x+y*y+z*z
    r1 = np.sqrt(r2)
    
    return -1./r2 - q*(r1-r[0]/r1*D)/((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**1.5 - q*r[0]/r1/D/D + F*F*(1+q)*(1-r[2]*r[2]/r2)*r1

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
    Gradient of misaligned circular Roche potential in cartesian coordinates.
    
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

def binary_surface_gravity(x,y,z,d,omega,mass1,mass2,normalize=False):
    r"""
    Calculate surface gravity in an eccentric asynchronous binary roche potential.
    
    If ``x=0``, ``y=0`` and ``z=Rpole``, this computes the polar surface gravity:
    
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

    term1 = - c.G.value*mass1/np.linalg.norm(r)**3*r
    term2 = - c.G.value*mass2/np.linalg.norm(h)**3*h
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
    #plt.plot(x/c.Rsol,y/c.Rsol,'ro')
    #from mayavi import mlab
    #mlab.figure(1)
    #mlab.points3d(*np.array([x,y,z]).reshape((3,-1))/c.Rsol,color=(0,1,0))
    #mlab.points3d(*d_cf.reshape((3,-1))/c.Rsol,color=(0,1,1))
    #mlab.points3d(*d.reshape((3,-1))/c.Rsol,color=(0,0,1))
    #mlab.plot3d(np.array([0,d[0]])/c.Rsol,np.array([0,d[1]]),np.array([0,d[2]]))
    #mlab.plot3d(2*np.array([0,x])/c.Rsol,2*np.array([0,y])/c.Rsol,2*np.array([0,z])/c.Rsol)
    #mlab.plot3d(np.array([x,d[0]])/c.Rsol,
                #np.array([y,d[1]])/c.Rsol,
                #np.array([z,d[2]])/c.Rsol)
    term1 = - c.G.value*mass1/coordinates.norm(r)**3*r
    term2 = - c.G.value*mass2/coordinates.norm(h)**3*h
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



# @decorators.memoized
def zeipel_gravb_binary():
    """
    Return an interpolator to derive the approximate Zeipel gravity brightening.
    
    See [Espinosa2012]_.
    
    """
    data = np.loadtxt(os.path.join(basedir, 'tables', 'gravb', 'espinosa.dat'))
    x, y = np.log10(data[1:,0]), data[0,1:]
    z = 4.0*data[1:,1:]
    return RectBivariateSpline(x, y, z)


# @decorators.memoized
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
    data[2] = np.log10(data[2]/c.z_solar)
    axv, pix = interp_nDgrid.create_pixeltypegrid(data[:3], data[3:])
    return axv, pix



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



def rpole2potential(r_pole, q, e, F, sma=1.0, component=1, tol=1e-10, maxiter=50):
    """
    Tranfsorms polar radius to surface potential at periastron.
    """

    return binary_potential(r_pole/sma, 0, 0, q, 1-e, F, component)
    
def potential2rpole(pot, q, e, F, sma=1.0, component=1, tol=1e-10, maxiter=50):
    """
    Transforms surface potential to polar radius at periastron.
    """

    potential = lambda r, q, e, F, c: binary_potential(r, 0, 0, q, 1-e, F, c)-pot

    try:
        r_pole = newton(potential, x0=1./pot, args=(q, e, F, component), tol=tol, maxiter=maxiter)
    except RuntimeError:
        raise ValueError("Failed to converge, potential {} is probably too low".format(pot))
    # print "*** potential2rpole", pot, sma, r_pole, r_pole*sma
    return r_pole*sma
