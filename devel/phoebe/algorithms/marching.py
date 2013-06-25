"""
Marching method for mesh generation

**Some convenience functions**

.. autosummary::

   precision_to_delta
   delta_to_precision
   delta_to_nelements
   delta_to_gridsize
   nelements_to_precision
   nelements_to_delta
   nelements_to_gridsize
   gridsize_to_delta

"""
from math import sqrt, sin, cos, acos, atan2, trunc, pi
import numpy as np
enable_mayavi = False
if enable_mayavi:
    try:
        from enthought.mayavi import mlab
    except ImportError:
        try:
            from mayavi import mlab
        except:
            #print("Soft warning: Mayavi could not be found on your system, 3D plotting is disabled, as well as some debugging features")
            enable_mayavi = False
import cmarching
try:
    import marching2FLib        
except ImportError:
    pass
    #print("Cannot import C methods for grid computations")

#{ Sphere



def Sphere(r, R):
    """
    Implicit surface of a sphere.
    
    @param r: relative radius vector (3 components)
    @type r: 3-tuple/list/array
    @param R: absolute radius
    @type R: float
    @return: potential value
    @rtype: float
    """
    return r[0]*r[0]+r[1]*r[1]+r[2]*r[2]-R*R

def dSpheredx(r):
    """
    Derivative of the spherical equation in the x-direction.
    
    @param r: relative radius vector (3 components)
    @type r: 3-tuple/list/array
    @return: derivative
    @rtype: float
    """
    return 2*r[0]

def dSpheredy(r):
    """
    Derivative of the spherical equation in the y-direction.
    
    @param r: relative radius vector (3 components)
    @type r: 3-tuple/list/array
    @return: derivative
    @rtype: float
    """
    return 2*r[1]

def dSpheredz(r):
    """
    Derivative of the spherical equation in the z-direction.
    
    @param r: relative radius vector (3 components)
    @type r: 3-tuple/list/array
    @return: derivative
    @rtype: float
    """
    return 2*r[2]

#}
#{ Plain binary

def BinaryRoche (r, D, q, F, Omega=0.0):
    """
    Computes a value of the potential. If @Omega is passed, it computes
    the difference.
    
    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    @param Omega:  value of the potential
    """
    return 1.0/sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]) + q*(1.0/sqrt((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])-r[0]/D/D) + 0.5*F*F*(1+q)*(r[0]*r[0]+r[1]*r[1]) - Omega

def dBinaryRochedx (r, D, q, F):
    """
    Computes a derivative of the potential with respect to x.
    
    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    return -r[0]*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**-1.5 -q*(r[0]-D)*((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**-1.5 -q/D/D + F*F*(1+q)*r[0]

def d2BinaryRochedx2(r, D, q, F):
    """
    Computes second derivative of the potential with respect to x.
    
    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    return (2*r[0]*r[0]-r[1]*r[1]-r[2]*r[2])/(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**2.5 +\
          q*(2*(r[0]-D)*(r[0]-D)-r[1]*r[1]-r[2]*r[2])/((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**2.5 +\
          F*F*(1+q)


def dBinaryRochedy (r, D, q, F):
    """
    Computes a derivative of the potential with respect to y.
    
    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    return -r[1]*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**-1.5 -q*r[1]*((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**-1.5 + F*F*(1+q)*r[1]

def dBinaryRochedz (r, D, q, F):
    """
    Computes a derivative of the potential with respect to z.
    
    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    return -r[2]*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**-1.5 -q*r[2]*((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**-1.5

#}
#{ Misaligned binary

def MisalignedBinaryRoche (r, D, q, F, theta, phi, Omega=0.0):
    r"""
    Computes a value of the potential. If @Omega is passed, it computes
    the difference.
    
    See [Avni1982]_.
    
    The parameter :math:`\theta` is similar to the inclination, and
    is an invariant of the system. The parameter :math:`\phi`, however,
    is phase dependent, and given by:
    
    .. math::
    
        \phi = \phi_0 - 2\pi \frac{t}{T}
    
    The potential is given by:
    
    .. math::
        
        \delta = &  (1-\cos^2\phi\sin^2\theta) x^2 +(1-\sin^2\phi\sin^2\theta) y^2 + \sin^2\theta z^2\\
               &  - \sin^2\theta\sin 2\phi xy - \sin 2\theta \cos\phi xz - \sin 2\theta \sin\phi yz \\
        \Omega  = & \frac{1}{r} + \frac{q}{\sqrt{(x-d)^2 + y^2 + z^2}} - \frac{qx}{d^2} + \frac{F^2}{2}(1+q)\delta
    
    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    @param theta:  misalignment coordinate (kind of like inclination)
    @param phi:    misalignment coordinate (phase dependent)
    @param Omega:  value of the potential
    """
    delta = (1-np.cos(phi)**2*np.sin(theta)**2)*r[0]*r[0] +\
            (1-np.sin(phi)**2*np.sin(theta)**2)*r[1]*r[1] +\
            np.sin(theta)**2*r[2]*r[2] -\
            np.sin(theta)**2*np.sin(2*phi)*r[0]*r[1] -\
            np.sin(2*theta)*np.cos(phi)*r[0]*r[2] -\
            np.sin(2*theta)*np.sin(phi)*r[1]*r[2]
    return 1.0/sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]) + q*(1.0/sqrt((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])-r[0]/D/D) + 0.5*F*F*(1+q)*delta - Omega

def dMisalignedBinaryRochedx (r, D, q, F, theta, phi):
    r"""
    Computes a derivative of the potential with respect to x.
    
    .. math::
    
        \delta = & 2 (1-\cos^2\phi\sin^2\theta)x - \sin^2\theta\sin 2\phi y - \sin 2\theta\cos\phi z\\
        
        \frac{d\Omega}{dx}  = &  -\frac{x}{r^{3/2}} - \frac{q (x-d)}{ ((x-d)^2 + y^2 + z^2)^{3/2}} - \frac{q}{d^2} + \frac{F^2}{2}(1+q)\delta
    
    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    @param theta:  misalignment coordinate
    @param phi:    misalignment coordinate
    """
    delta = 2*(1-np.cos(phi)**2*np.sin(theta)**2)*r[0] -\
            np.sin(theta)**2*np.sin(2*phi)*r[1] -\
            np.sin(2*theta)*np.cos(phi)*r[2]
    return -r[0]*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**-1.5 -q*(r[0]-D)*((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**-1.5 -q/D/D + 0.5*F*F*(1+q)*delta

def dMisalignedBinaryRochedy (r, D, q, F, theta, phi):
    """
    Computes a derivative of the potential with respect to y.
    
    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    @param theta:  misalignment coordinate
    @param phi:    misalignment coordinate
    """
    delta = 2*(1-np.sin(phi)**2*np.sin(theta)**2)*r[1] -\
            np.sin(theta)**2*np.sin(2*phi)*r[0] -\
            np.sin(2*theta)*np.sin(phi)*r[2]
    return -r[1]*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**-1.5 -q*r[1]*((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**-1.5 + 0.5*F*F*(1+q)*delta

def dMisalignedBinaryRochedz (r, D, q, F, theta, phi):
    """
    Computes a derivative of the potential with respect to z.
    
    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    @param theta:  misalignment coordinate
    @param phi:    misalignment coordinate
    """
    delta = 2*np.sin(theta)**2*r[2] -\
            np.sin(2*theta)*np.cos(phi)*r[0] -\
            np.sin(2*theta)*np.sin(phi)*r[1]
    return -r[2]*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**-1.5 -q*r[2]*((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**-1.5 + 0.5*F*F*(1+q)*delta

#}
#{ Distorted binary

def DistortedBinaryRoche (r, D, q, F, theta, phi, Omega=0.0):
    """
    Computes a value of the potential. If @Omega is passed, it computes
    the difference.
    
    For differentially rotating or tidally distorted binaries.
    
    See Kumar, Kumar Lal, Singh & Saini, 2011, World Journal of modelling and
    simulation.
    
    Incomplete!
    
    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    @param theta:  misalignment coordinate
    @param phi:    misalignment coordinate
    @param Omega:  value of the potential
    """
    j = np.arange(2,10) # should be up to infinity
    factor = a[0]+a[1]+a[2]
    factor-= 0.5*(1-v**2)**2*r**4 * (a[0]*b[0]**2 + a[1]*b[1]**2 + a[2]*b[2]**2)
    return 1./r + q + q*np.sum(r**j*p) + 0.5*r**2*(1-v**2)*factor - Omega

#}
#{ Rotating single star
    
def RotateRoche(r, Omega, Rpole):
    """
    Roche shape of rotating star.
    
    In our units, 0.544... is the critical angular velocity if Rpole==1.
    Actually this is generally true, since Omega is the dimensionless
    angular rotation frequency, i.e. in units of sqrt(GM/R**3). The critical
    angular velocity is sqrt(8*G*M/(27*R**3)) or::
        
        Omega_crit = sqrt(8/27)
        Omega_crit = 0.54433105395181736
        
    You'd better use Rpole=1 and rescale afterwards.
    """
    Omega = Omega*0.54433105395181736
    r_ = (r[0]**2+r[1]**2+r[2]**2)**0.5
    return 1./Rpole - 1/r_ -0.5*Omega**2*(r[0]**2+r[1]**2)
    

def dRotateRochedx(r, Omega):
    Omega = Omega*0.54433105395181736
    return r[0]*(r[0]**2+r[1]**2+r[2]**2)**-1.5 - Omega**2*r[0]

def dRotateRochedy(r, Omega):
    Omega = Omega*0.54433105395181736
    return r[1]*(r[0]**2+r[1]**2+r[2]**2)**-1.5 - Omega**2*r[1]

def dRotateRochedz(r, Omega):
    Omega = Omega*0.54433105395181736
    return r[2]*(r[0]**2+r[1]**2+r[2]**2)**-1.5

#}
#{ Differentially rotating single star

def DiffRotateRoche(r, b1, b2, b3, Rpole):
    """
    Roche shape of rotating star.
    
    omega = b1 + b2*s**2 + b3*s**4 with s the distance from the rotation axis.
    
    To get the same result as with RotateRoche, set b2==b3==0 and b1 = Omega*0.54433105395181736
    
    See Kumar, Kumar Lal, Singh & Saini, 2011, World Journal of modelling and
    simulation.
    
    You'd better use Rpole=1. Then, if b2=0 and b3=0, this funciton is equivalent
    to L{RotateRoche}.
    """
    r_ = (r[0]**2+r[1]**2+r[2]**2)**0.5
    x2y2 = r[0]**2 + r[1]**2
    
    return 1./Rpole - 1/r_ -0.5* (b1**2                 * x2y2    + \
                                  b1*b2                 * x2y2**2 + \
                                  1./3.*(2*b1*b2+b2**2) * x2y2**3 + \
                                  0.5*b2*b3             * x2y2**4 + \
                                  1./5. * b3**2         * x2y2**5)
    

def dDiffRotateRochedx(r, b1, b2, b3,):
    x2y2 = r[0]**2 + r[1]**2
    fac1 = b1**2
    fac2 = b1*b2
    fac3 = 1./3.*(2*b1*b3 + b2**2)
    fac4 = 0.5*b2*b3
    fac5 = 1./5.*b3**2
    return r[0]*(r[0]**2+r[1]**2+r[2]**2)**-1.5 - 0.5* (fac1*2*r[0] + \
                                                        fac2*2*x2y2*2*r[0] + \
                                                        fac3*3*x2y2**2*2*r[0] + \
                                                        fac4*4*x2y2**3*2*r[0] +\
                                                        fac5*5*2*r[0])

def dDiffRotateRochedy(r, b1, b2, b3,):
    x2y2 = r[0]**2 + r[1]**2
    fac1 = b1**2
    fac2 = b1*b2
    fac3 = 1./3.*(2*b1*b3 + b2**2)
    fac4 = 0.5*b2*b3
    fac5 = 1./5.*b3**2
    return r[1]*(r[0]**2+r[1]**2+r[2]**2)**-1.5 - 0.5* (fac1*2*r[1] + \
                                                        fac2*2*x2y2*2*r[1] + \
                                                        fac3*3*x2y2**2*2*r[1] + \
                                                        fac4*4*x2y2**3*2*r[1] +\
                                                        fac5*5*2*r[1])


def dDiffRotateRochedz(r, b1, b2, b3,):
    return r[2]*(r[0]**2+r[1]**2+r[2]**2)**-1.5

#}
#{ Torus

def Torus(r,R,r_):
    return r_**2-R**2+2*R*(r[0]**2+r[1]**2)**0.5-r[0]**2-r[1]**2-r[2]**2

def dTorusdx(r,R):
    return 2*R*r[0]*(r[0]**2+r[1]**2)**-0.5-2*r[0]

def dTorusdy(r,R):
    return 2*R*r[1]*(r[0]**2+r[1]**2)**-0.5-2*r[1]

def dTorusdz(r,R):
    return -2*r[2]
    
    
#}

#{ Heart

def Heart(r,R):
    return     ((r[0]**2 + 9./4.*r[1]**2 + r[2]**2 - 1)**3 - r[0]**2*r[2]**3 - 9./80*r[1]**2*r[2]**3)

def dHeartdx(r):
    return (3 * (r[0]**2 + 9./4.*r[1]**2 + r[2]**2 - 1)**2*2*r[0] - 2*r[0]*r[2]**3)

def dHeartdy(r):
    return (3*(r[0]**2 + 9./4.*r[1]**2 + r[2]**2 - 1)**2*9./2.*r[1]                 -  9./40.*r[1]*r[2]**3)

def dHeartdz(r):
    return (3*(r[0]**2 + 9./4.*r[1]**2 + r[2]**2 - 1)**2*2*r[2] - 3*r[0]**2*r[2]**2 - 27./80.*r[1]**2*r[2]**2)

#}


class MeshVertex:
    def __init__(self, r, dpdx, dpdy, dpdz, *args):
        nx = dpdx(r, *args)
        ny = dpdy(r, *args)
        nz = dpdz(r, *args)
        nn = sqrt(nx*nx+ny*ny+nz*nz)
        nx /= nn
        ny /= nn
        nz /= nn

        if nx > 0.5 or ny > 0.5:
            nn = sqrt(ny*ny+nx*nx)
            t1x = ny/nn
            t1y = -nx/nn
            t1z = 0.0
        else:
            nn = sqrt(nx*nx+nz*nz)
            t1x = -nz/nn
            t1y = 0.0
            t1z = nx/nn

        t2x = ny*t1z - nz*t1y
        t2y = nz*t1x - nx*t1z
        t2z = nx*t1y - ny*t1x

        self.r = r
        self.n = np.array((nx, ny, nz))
        self.t1 = np.array((t1x, t1y, t1z))
        self.t2 = np.array((t2x, t2y, t2z))
        
    def __repr__(self):
        repstr  = " r = (% 3.3f, % 3.3f, % 3.3f)\t" % (self.r[0],  self.r[1], self.r[2])
        repstr += " n = (% 3.3f, % 3.3f, % 3.3f)\t" % (self.n[0],  self.n[1], self.n[2])
        repstr += "t1 = (% 3.3f, % 3.3f, % 3.3f)\t" % (self.t1[0], self.t1[1], self.t1[2])
        repstr += "t2 = (% 3.3f, % 3.3f, % 3.3f)"   % (self.t2[0], self.t2[1], self.t2[2])
        return repstr

#{ Marching method        
        
def projectOntoPotential(r, pot_name, *args):
    """
    Last term must be constant factor (i.e. the value of the potential)
    """
    ri = np.array((0,0,0))
    
    pot = globals()[pot_name]
    dpdx = globals()['d%sdx'%(pot_name)]
    dpdy = globals()['d%sdy'%(pot_name)]
    dpdz = globals()['d%sdz'%(pot_name)]
        
    n_iter = 0
    while ((r[0]-ri[0])*(r[0]-ri[0]) + (r[1]-ri[1])*(r[1]-ri[1]) + (r[2]-ri[2])*(r[2]-ri[2]) > 1e-12) and n_iter<100:
        ri = r
        g = np.array((dpdx(ri, *args[:-1]), dpdy(ri, *args[:-1]), dpdz(ri, *args[:-1])))
        grsq = g[0]*g[0]+g[1]*g[1]+g[2]*g[2]
        r = ri - pot(ri, *args)*g/grsq
        n_iter+=1
    if n_iter>=90:
        print('warning: projection did not converge')
    return MeshVertex(r, dpdx, dpdy, dpdz, *args[:-1])

    
def inverse(a11, a12, a13, a21, a22, a23, a31, a32, a33):
    detA = a11*a22*a33-a13*a22*a31+a12*a23*a31-a11*a23*a32+a13*a21*a32-a12*a21*a33

    b11 = a22*a33-a23*a32
    b12 = a13*a32-a12*a33
    b13 = a12*a23-a13*a22
    b21 = a23*a31-a21*a33
    b22 = a11*a33-a13*a31
    b23 = a13*a21-a11*a23
    b31 = a21*a32-a31*a22
    b32 = a12*a31-a11*a32
    b33 = a11*a22-a12*a21

    return (b11/detA, b12/detA, b13/detA, b21/detA, b22/detA, b23/detA, b31/detA, b32/detA, b33/detA)

def cart2local (v, r):
    """
    This function converts vector @r from the cartesian coordinate system
    (defined by i, j, k) to the local coordinate system (defined
    by n, t1, t2).
    
    @param v: MeshVertex that defines the local coordinate system
    @param r: Vector in the cartesian coordinate system
    """

    invM = inverse(v.n[0], v.t1[0], v.t2[0], v.n[1], v.t1[1], v.t2[1], v.n[2], v.t1[2], v.t2[2])
    xi, eta, zeta = invM[0]*r[0]+invM[1]*r[1]+invM[2]*r[2], invM[3]*r[0]+invM[4]*r[1]+invM[5]*r[2], invM[6]*r[0]+invM[7]*r[1]+invM[8]*r[2]
    return np.array((xi, eta, zeta))

def local2cart (v, r):
    """
    This function converts vector @r from the local coordinate system
    (defined by n, t1, t2) to the cartesian coordinate system (defined
    by i, j, k).
    
    @param v: MeshVertex that defines the local coordinate system
    @param r: Vector in the local coordinate system
    """
    
    x, y, z = v.n[0]*r[0]+v.t1[0]*r[1]+v.t2[0]*r[2], v.n[1]*r[0]+v.t1[1]*r[1]+v.t2[1]*r[2], v.n[2]*r[0]+v.t1[2]*r[1]+v.t2[2]*r[2]
    return np.array((x, y, z))

    
def gridsize_to_delta(gridsize):
    """
    Estimate the marching stepsize delta parameter from the WD gridsize parameter.
    
    This is an empirical relation, which will deviate more for less spherical
    surfaces.
    
    You will get about twice as much triangles with WD gridding, but WD actually
    uses sqaures, so you need to divide that number by two::
    
        delta = 0.037 # 23884
        gridsize = 60 # 24088
        delta = 0.045 # 16582
        gridsize = 50 # 16783
        delta = 0.0555 # 10720
        gridsize = 40  # 10760
        delta = 0.073 # 6092
        gridsize = 30 # 6096
        delta = 0.095654  # 3628
        gridsize = 23     # 3616
        delta = 0.145 # 1570
        gridsize = 15 # 1560
        delta = 0.275 # 470
        gridsize = 8 # 464
    
    @param gridsize: WD gridsize parameter
    @type gridsize: float
    """
    #return 10**(-0.9925473*np.log10(gridsize)+0.48526143)
    return 10**(-0.98359345*np.log10(gridsize)+0.4713824)

def delta_to_gridsize(delta):
    """
    Estimate the WD gridsize parameter from the marching delta parameter.
    
    @param delta: marching delta parameter
    @type delta: float
    """
    return gridsize_to_delta(delta)

def delta_to_nelements(delta):
    """
    Estimate the number of surface elements from the marching delta parameter.
    
    @param delta: marching delta parameter
    @type delta: float
    @return n: number of surface elements
    @rtype n: int
    """
    a = 0.50688733
    b = 0.78603153
    return 10**((b-np.log10(delta))/a)

def delta_to_precision(delta):
    """
    Estimate the precision of computed fluxes from the marching delta.
    
    This delivers an estimate of the precision between analytically computed
    and numerically computed fluxes, and is only strictly valid for spherical
    surfaces. For deformed shapes, this is only a rough estimate.
        
    @param delta: marching delta
    @type delta: int
    @return: estimate of relative precision of computed fluxes
    @rtype: float
    """
    n = delta_to_nelements(delta)
    return 6.23/n

def precision_to_delta(eps):
    """
    Estimate the delta from the required precision.
    
    Well applicable to spherical surfaces. For deformed shapes, this is only a rough estimate.
    
    @param eps: required precision
    @type eps: float
    @return: marching parameter delta
    @rtype: float
    """
    n = 6.23/eps
    return nelements_to_delta(n)
    

def nelements_to_delta(n):
    """
    Estimate the marching delta parameter from the number of surface elements.
    
    @param n: number of surface elements
    @type n: int
    """
    return 10**(-0.50688733*np.log10(n)+0.78603153)

def nelements_to_gridsize(n):
    """
    Estimate the WD gridsize parameter from the number of surface elements.
    
    If you want 3000 squares, give n=3000.
    
    @param n: number of surface elements
    @type n: int
    """
    return 10**(+0.51070499*np.log10(n)-0.30303537)

def nelements_to_precision(n,alg='marching'):
    """
    Estimate the precision of computed fluxes from the number of surface elements.
    
    This delivers an estimate of the precision between analytically computed
    and numerically computed fluxes, and is only strictly valid for spherical
    surfaces. For deformed shapes, this is only a rough estimate.
    
    This is different for WD and the marching method.
    
    @param n: number of surface elements
    @type n: int
    @param alg: type of algorithm
    @type alg: str, one of 'marching','wd'
    @return: estimate of relative precision of computed fluxes
    @rtype: float
    """
    if alg.lower()=='marching':
        return 6.23/n
    elif alg.lower()=='wd':
        return 3.12*n**(-0.85)
    
#}
#{ Main interface
    
def discretize_wd_style(N=30, potential='BinaryRoche', *args):
    # WD computes the center-point of the rectangle and projects it onto
    # the potential. It then computes the four vertices but does nothing
    # with them, i.e. they do *not* lie on the equipotential.
    #
    # Since we cannot use rectangles and need to use triangles, we must
    # modify the original approach so that the vertices do lie on the
    # equipotential.

    Ts = []
    r0 = -projectOntoPotential(np.array((-0.02, 0.0, 0.0)), potential, *args).r[0]

    #theta = [pi/2*(k-0.5)/N for k in range(1,N+1)]
    theta = [pi/2*(k-0.5)/N for k in range(1,2*N+1)]
    for th in theta:
        Mk = 1+int(1.3*N*sin(th))
        #phi = [pi*(l-0.5)/Mk for l in range(1,Mk+1)]
        phi = [pi*(l-0.5)/Mk for l in range(1,2*Mk+1)]
        
        for ph in phi:
            r1 = (r0*sin(th-pi/4/N)*cos(ph-pi/2/Mk), r0*sin(th-pi/4/N)*sin(ph-pi/2/Mk), r0*cos(th-pi/4/N))
            r2 = (r0*sin(th-pi/4/N)*cos(ph+pi/2/Mk), r0*sin(th-pi/4/N)*sin(ph+pi/2/Mk), r0*cos(th-pi/4/N))
            r3 = (r0*sin(th+pi/4/N)*cos(ph+pi/2/Mk), r0*sin(th+pi/4/N)*sin(ph+pi/2/Mk), r0*cos(th+pi/4/N))
            r4 = (r0*sin(th+pi/4/N)*cos(ph-pi/2/Mk), r0*sin(th+pi/4/N)*sin(ph-pi/2/Mk), r0*cos(th+pi/4/N))
            v1 = projectOntoPotential(r1, potential, *args)
            v2 = projectOntoPotential(r2, potential, *args)
            v3 = projectOntoPotential(r3, potential, *args)
            v4 = projectOntoPotential(r4, potential, *args)
            Ts += [(v1, v2, v4), (v2, v3, v4)]
    
    table = np.zeros((len(Ts), 16))
    for i in range(0,len(Ts)):
        cx = (Ts[i][0].r[0]+Ts[i][1].r[0]+Ts[i][2].r[0])/3
        cy = (Ts[i][0].r[1]+Ts[i][1].r[1]+Ts[i][2].r[1])/3
        cz = (Ts[i][0].r[2]+Ts[i][1].r[2]+Ts[i][2].r[2])/3
        c = projectOntoPotential((cx,cy,cz), potential, *args)
        side1 = sqrt((Ts[i][0].r[0]-Ts[i][1].r[0])**2+(Ts[i][0].r[1]-Ts[i][1].r[1])**2+(Ts[i][0].r[2]-Ts[i][1].r[2])**2)
        side2 = sqrt((Ts[i][0].r[0]-Ts[i][2].r[0])**2+(Ts[i][0].r[1]-Ts[i][2].r[1])**2+(Ts[i][0].r[2]-Ts[i][2].r[2])**2)
        side3 = sqrt((Ts[i][2].r[0]-Ts[i][1].r[0])**2+(Ts[i][2].r[1]-Ts[i][1].r[1])**2+(Ts[i][2].r[2]-Ts[i][1].r[2])**2)
        s = 0.5*(side1 + side2 + side3)

        table[i][ 0] = c.r[0]
        table[i][ 1] = c.r[1]
        table[i][ 2] = c.r[2]
        table[i][ 3] = sqrt(s*(s-side1)*(s-side2)*(s-side3))
        table[i][ 4] = Ts[i][0].r[0]
        table[i][ 5] = Ts[i][0].r[1]
        table[i][ 6] = Ts[i][0].r[2]
        table[i][ 7] = Ts[i][1].r[0]
        table[i][ 8] = Ts[i][1].r[1]
        table[i][ 9] = Ts[i][1].r[2]
        table[i][10] = Ts[i][2].r[0]
        table[i][11] = Ts[i][2].r[1]
        table[i][12] = Ts[i][2].r[2]
        table[i][13] = c.n[0]
        table[i][14] = c.n[1]
        table[i][15] = c.n[2]

    return table


def cdiscretize2(delta=0.1,  max_triangles=10000, potential='BinaryRoche', *args):
    """
    Discretize using c module written by Joe Giammarco.
    """
    pot = ['Sphere','BinaryRoche','MisalignedBinaryRoche','RotateRoche'].index(potential)
    args = list(args)+[delta]
    numparams = len(args)+1
    table = marching2FLib.getMesh(pot,max_triangles,numparams,*args)
    return table[0]

def cdiscretize(delta=0.1,  max_triangles=10000, potential='BinaryRoche', *args):
    """
    Discretize using c module written by Gal Matijevic.
    """
    if max_triangles is None:
        max_triangles = 0
    #print delta,max_triangles,potential,args
    table = cmarching.discretize(delta,max_triangles,potential,*args)
    return table


def discretize(delta=0.1,  max_triangles=None, potential='BinaryRoche', *args):
    """
    Computes and returns a table of triangulated surface elements.
    Table columns are:
    
    center-x, center-y, center-z, area, v1-x, v1-y, v1-z, v2-x, v2-y,
    v2-z, v3-x, v3-y, v3-z, normal-x, normal-y, normal-z
    
    Arguments:
        - delta:  triangle side in units of SMA
        - D:      instantaneous separation in units of SMA
        - q:      mass ratio
        - F:      synchronicity parameter
        - Omega:  value of the surface potential
    """
    # q=0.76631, Omega=6.1518, F=5 is failing.
    init = (-0.00002,0,0)
    #init = (50.,50,0.1)
    p0 = projectOntoPotential(np.array(init), potential, *args)
    V = [p0] # A list of all vertices
    P = []   # Actual front polygon
    Ts = []  # Triangles
    #print 'p0',p0
    # Create the first hexagon:
    for i in range(0,6):
        qk = np.array((p0.r[0]+delta*cos(i*pi/3)*p0.t1[0]+delta*sin(i*pi/3)*p0.t2[0], p0.r[1]+delta*cos(i*pi/3)*p0.t1[1]+delta*sin(i*pi/3)*p0.t2[1], p0.r[2]+delta*cos(i*pi/3)*p0.t1[2]+delta*sin(i*pi/3)*p0.t2[2]))
        #print i,'qk',qk
        #print qk
        pk = projectOntoPotential(qk, potential, *args)
        #print i,'pk',pk
        P.append (pk)
        V.append (pk)
    #       print "P[%d]: " % (i), P[i]
    Ts += [(V[0], V[1], V[2]), (V[0], V[2], V[3]), (V[0], V[3], V[4]), (V[0], V[4], V[5]), (V[0], V[5], V[6]), (V[0], V[6], V[1])]

    # Start triangulation:
    step = -1
    while len(P) > 0:
        step += 1
        #print step,max_triangles
        if max_triangles != None and step > max_triangles:
            break
        omega = np.zeros(len(P))
        for i in range(0,len(P)):
            xi1, eta1, zeta1 = cart2local(P[i], P[i-1].r-P[i].r)
            xi2, eta2, zeta2 = cart2local(P[i], P[i+1 if i < len(P)-1 else 0].r-P[i].r)
            omega[i] = (atan2(zeta2, eta2)-atan2(zeta1, eta1)) % (2*pi)
    #           print "%d: r[%d]=(% 3.3f, % 3.3f, % 3.3f), r[%d]=(% 3.3f, % 3.3f, % 3.3f), front_angle=% 3.3f" % (i, i-1 if i > 0 else len(P)-1, xi1, eta1, zeta1, i+1 if i < len(P)-1 else 0, xi2, eta2, zeta2, angle[i]/pi*180)

        minidx = omega.argmin()
        minangle = omega[minidx]
    #       print "Minimum angle: % 3.3f at index %d" % (omega[minidx]*180/pi, minidx)

        # The number of triangles to be generated:
        nt = trunc(minangle*3/pi)+1
        domega = minangle/nt
        if domega < 0.8 and nt > 1:
            nt -= 1
            domega = minangle/nt
        ### INSERT THE REMAINING HOOKS HERE!

    #       print "Number of triangles to be generated: %d; domega = % 3.3f" % (nt, domega)

        # Generate the triangles:

        p0m, v1, v2 = P[minidx], P[minidx-1], P[minidx+1 if minidx < len(P)-1 else 0]

        for i in range(1, nt):
            xi1, eta1, zeta1 = cart2local(p0m, v1.r-p0m.r)
            xi3, eta3, zeta3 = 0.0, eta1*cos(i*domega)-zeta1*sin(i*domega), eta1*sin(i*domega)+zeta1*cos(i*domega)
            norm3 = sqrt(eta3*eta3+zeta3*zeta3)
            eta3 /= norm3/delta
            zeta3 /= norm3/delta
            qk = p0m.r+local2cart (p0m, np.array((xi3, eta3, zeta3)))
            pk = projectOntoPotential(qk, potential, *args)
            V.append(pk)
            Ts += [(v1 if i == 1 else V[len(V)-2], pk, p0m)]

        if nt == 1:
            Ts += [(v1, v2, p0m)]
        else:
            Ts += [(V[len(V)-1], v2, p0m)]

        idx = P.index(p0m)
        P.remove(p0m)
        for i in range(1,nt):
            P.insert(idx+i-1, V[len(V)-nt+i])
    #print "Number of triangles:", len(Ts)

    table = np.zeros((len(Ts), 16))
    for i in range(0,len(Ts)):
        cx = (Ts[i][0].r[0]+Ts[i][1].r[0]+Ts[i][2].r[0])/3
        cy = (Ts[i][0].r[1]+Ts[i][1].r[1]+Ts[i][2].r[1])/3
        cz = (Ts[i][0].r[2]+Ts[i][1].r[2]+Ts[i][2].r[2])/3
        c = projectOntoPotential((cx,cy,cz), potential, *args)
        side1 = sqrt((Ts[i][0].r[0]-Ts[i][1].r[0])**2+(Ts[i][0].r[1]-Ts[i][1].r[1])**2+(Ts[i][0].r[2]-Ts[i][1].r[2])**2)
        side2 = sqrt((Ts[i][0].r[0]-Ts[i][2].r[0])**2+(Ts[i][0].r[1]-Ts[i][2].r[1])**2+(Ts[i][0].r[2]-Ts[i][2].r[2])**2)
        side3 = sqrt((Ts[i][2].r[0]-Ts[i][1].r[0])**2+(Ts[i][2].r[1]-Ts[i][1].r[1])**2+(Ts[i][2].r[2]-Ts[i][1].r[2])**2)
        s = 0.5*(side1 + side2 + side3)

        table[i][ 0] = c.r[0]
        table[i][ 1] = c.r[1]
        table[i][ 2] = c.r[2]
        table[i][ 3] = sqrt(s*(s-side1)*(s-side2)*(s-side3))
        table[i][ 4] = Ts[i][0].r[0]
        table[i][ 5] = Ts[i][0].r[1]
        table[i][ 6] = Ts[i][0].r[2]
        table[i][ 7] = Ts[i][1].r[0]
        table[i][ 8] = Ts[i][1].r[1]
        table[i][ 9] = Ts[i][1].r[2]
        table[i][10] = Ts[i][2].r[0]
        table[i][11] = Ts[i][2].r[1]
        table[i][12] = Ts[i][2].r[2]
        table[i][13] = c.n[0]
        table[i][14] = c.n[1]
        table[i][15] = c.n[2]
    
    return table
  
def reproject(table,*new_mesh_args):
    """
    Reproject the old coordinates. We assume they are fairly
    close to the real values.
    """
    new_table = np.zeros_like(table)
    
    for tri in range(len(table)):
    
        #-- reproject the triangle vertices
        
        new_table[tri, 4: 7] = projectOntoPotential(table[tri, 4: 7],*new_mesh_args).r
        new_table[tri, 7:10] = projectOntoPotential(table[tri, 7:10],*new_mesh_args).r
        new_table[tri,10:13] = projectOntoPotential(table[tri,10:13],*new_mesh_args).r
    
        #-- reproject also the center coordinates, together with their normals
    
        p0 = projectOntoPotential(table[tri, 0: 3],*new_mesh_args)
        new_table[tri,0:3] = p0.r     
        new_table[tri,13:16] = p0.n
   
    return new_table  
  
  
def creproject(table,*new_mesh_args):
    new_table = cmarching.reproject(table,*new_mesh_args)
    return new_table
#}

#     center-x, center-y, center-z, area, v1-x, v1-y, v1-z, v2-x, v2-y,
#     v2-z, v3-x, v3-y, v3-z, normal-x, normal-y, normal-z



if __name__=="__main__":
    from mayavi import mlab
    # heart shape: (2x**2+y**2+z**2-1)**3 - (0.1*x**2+y**2)*z**3 = 0

    D,q,F,Omega = 1.,1.,1.,3.8
    project,args = 'BinaryRoche',(D,q,F,Omega)
    #project,args = 'Sphere',(1.0,)
    #project,args = 'RotateRoche',(0.99,1.0) #delta=0.075
    #project,args = 'Torus',(1.0,0.1) # delta=0.1, max 2500 triangles
    project,args = 'Heart',(1.0,)
    
    table = discretize(0.1,2600,project,*args)
    #table = discretize_wd_style(20, project, *args)
    N = len(table)
    x = np.hstack([table[:,i+4] for i in range(0,9,3)])
    y = np.hstack([table[:,i+4] for i in range(1,9,3)])
    z = np.hstack([table[:,i+4] for i in range(2,9,3)])
    triangles = [(i,N+i,2*N+i) for i in range(N)]
    mlab.triangular_mesh(x,y,z,triangles)
    mlab.show()
