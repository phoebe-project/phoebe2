"""
Coordinate transformations.

In general, we define :math:`\phi` as the longitude between :math:`-\pi` and
:math:`+\pi`. The colatitude :math:`\theta` lies between :math:`0` and :math:`\pi`.
"""
import logging
import numpy as np
from numpy import sin,cos,sqrt,pi,arctan2
from phoebe.utils import cgeometry
#from pyphoebe import fgeometry

logger = logging.getLogger('UTILS.COORDS')

def rotate_in_place(obj,euler,pivot=(0,0,0),loc=(0,0,0)):
    """
    Rotate/translate an object in place.
    
    The rotation goes like:
    
    ROTZ(Omega) * ROTX(-i) * ROTZ(theta)
    
    where ROTX is a rotation around the x-axis, i is the inclination, Omega
    the angle on the sky
    
    To rotate a set of coordinates, set C{loc} to match the Cartesian
    location of the point in the orbit.
    
    To rotate a vector, set C{loc} to the null vector.
    
    @param obj: coordinates or vectors to the position.
    @type obj: 3-tuple (each entry can be an array)
    @param euler: Euler angles
    @type euler: 3-tuple
    @param loc: translation vector
    @type loc: 3-tuple (each entry can be an array)
    @return: rotated and translated coordinates
    @rtype: 3xarray
    """
    #-- Optimized Python version
    x,y,z = obj.T
    out = np.zeros_like(obj)
    x = x-pivot[0]
    y = y-pivot[1]
    z = z-pivot[2]
    theta,longan,incl = euler
    s1,c1 = np.sin(longan),np.cos(longan)
    s2,c2 = np.sin(-incl)  ,np.cos(-incl)
    s3,c3 = np.sin(theta) ,np.cos(theta)
    out[:,0] = ( c1*c3-c2*s1*s3)*x - (c1*s3+c2*c3*s1)*y + s1*s2*z + loc[0]
    out[:,1] = ( s1*c3+c2*c1*s3)*x + (c1*c2*c3-s1*s3)*y - c1*s2*z + loc[1]
    out[:,2] =           (s2*s3)*x +          (s2*c3)*y +    c2*z + loc[2]       
    return out


def rotate_and_translate(mesh,theta=0,incl=0,Omega=0,
             pivot=(0,0,0),loc=(0,0,0),
             los=(0,0,+1),incremental=False):
    """
    Rotate a body in 3D and compute angles between normals and LOS.
    
    Suppose we have a mesh consisting of a small star and a giant star, and
    we see the system edge on ((x,y) is the plane of the sky).
    
    Omega rotates around z-axis
    incl rotates around y-axis
    
    The changes in rotation are not cumulative, i.e. if you do twice a
    rotation with 45. degrees, you end up with a rotation 45. degrees. 
    
    Conv=YXZ should do the trick for binaries when looking at zminus
    """
    #-- only update orbital parameters when they are given
    mesh_ = mesh.copy()
    fields = mesh_.dtype.names
    if incremental:
        prefix = ''
    else:
        prefix = '_o_'
    logger.info("rotating body (%s) with theta=%.3g, incl=%.3g, Omega=%.3g"%(prefix,theta,incl,Omega))
    mesh_['center'] = rotate_in_place(mesh[prefix+'center'],(theta,Omega,incl),pivot,loc)
    mesh_['triangle'][:,0:3] = rotate_in_place(mesh[prefix+'triangle'][:,0:3],(theta,Omega,incl),pivot,loc)
    mesh_['triangle'][:,3:6] = rotate_in_place(mesh[prefix+'triangle'][:,3:6],(theta,Omega,incl),pivot,loc)
    mesh_['triangle'][:,6:9] = rotate_in_place(mesh[prefix+'triangle'][:,6:9],(theta,Omega,incl),pivot,loc)
    mesh_['normal_'] = rotate_in_place(mesh_[prefix+'normal_'],(theta,Omega,incl),(0,0,0))
    #-- rotate magnetic field if present
    if 'B_' in fields:
        mesh_['B_'] = rotate_in_place(mesh_[prefix+'B_'],(theta,Omega,incl),(0,0,0))
    for lbl in mesh_.dtype.names:
        if lbl[:4]=='velo':
            mesh_[lbl] = rotate_in_place(mesh_[prefix+lbl],(theta,Omega,incl),(0,0,0))
    
    mesh_['mu'] = cgeometry.cos_theta(mesh_['normal_'].ravel(order='F').reshape((-1,3)),np.array(los,float))
    #mesh_['mu'] = fgeometry.cos_theta(mesh_['normal_'],los)
    return mesh_


def norm(vec,axis=0):
    """
    Euclidian norm of a vector (or of a grid of vectors)
    
    Input vectors should be numpy arrays of size (3xN) when axis=0
    or of size (Nx3) when axis=1
    
    Single vector:
    
    >>> vec = np.array([4.,5.,6.])
    >>> print norm(vec)
    8.77496438739
    
    Array of vectors: if N vectors and arrays is Nx3, then make sure
    C{axis=1} or C{axis=-1}. When array is 3XN set C{axis=0}. In each case,
    an array of length N is returned
    
    >>> vecs = np.array([vec,2*vec])
    >>> print norm(vecs,axis=-1)
    [  8.77496439  17.54992877]
    
    @return: norm of input vector(s)
    @rtype: array/float
    """
    return sqrt((vec**2).sum(axis=axis))


def cart2spher_coord(x,y,z):
    """
    Cartesian to spherical coordinate transformation.
    
    @return: radius, phi (longitude), theta (colatitude)
    @rtype: 3-tuple
    """
    rho = sqrt(x**2+y**2+z**2)
    phi = arctan2(y,x)
    theta = arctan2(np.sqrt(x**2+y**2),z)
    return rho,phi,theta


def spher2cart_coord(r,phi,theta):
    """
    Spherical to Cartesian coordinate transformation.
    
    @return: x, y, z
    @rtype: 3-tuple
    """
    x = r*cos(phi)*sin(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(theta)
    return x,y,z

def spher2cart(position, direction):
    """
    Spherical to cartesian vector transformation.
    
    theta is angle from z-axis (colatitude)
    phi is longitude
    
    E.g. http://www.engin.brown.edu/courses/en3/Notes/Vector_Web2/Vectors6a/Vectors6a.htm
    
    >>> np.random.seed(111)
    >>> r,phi,theta = np.random.uniform(low=-1,high=1,size=(3,2))
    >>> a_r,a_phi,a_theta = np.random.uniform(low=-1,high=1,size=(3,2))
    >>> a_x,a_y,a_z = spher2cart((r,phi,theta),(a_r,a_phi,a_theta))
    
    @return: a_x, a_y, a_z
    @rtype: 3-tuple
    """
    (r,phi,theta), (a_r,a_phi,a_theta) = position, direction
    ax = sin(theta)*cos(phi)*a_r + cos(theta)*cos(phi)*a_theta - sin(phi)*a_phi
    ay = sin(theta)*sin(phi)*a_r + cos(theta)*sin(phi)*a_theta + cos(phi)*a_phi
    az = cos(theta)         *a_r - sin(theta)         *a_theta
    return ax,ay,az


def cos_angle(vec1,vec2,axis=0):
    """
    Compute cosine of angle between two vectors (or between two grids of vectors).
    
    Input vectors should be numpy arrays of size (3xN) when axis=0
    or of size (Nx3) when axis=1
    
    @return: cosine of angle between input vectors
    @rtype: array/float
    
    """
    return (vec1*vec2).sum(axis=axis) / (norm(vec1,axis=axis)*norm(vec2,axis=axis))



def normal_from_points(*args):
    """
    Construct a normal from three points.
    """
    if len(args)==3:
        A,B,C = args
        direction = np.cross(B-A,C-A)[:3]
    else:
        ax3 = np.ones(len(args))
        points = np.vstack([np.array(args).T,ax3]).T
        u,s,v = np.linalg.svd(points,0)
        direction = v.T[:3,3]
    return direction/np.sqrt((direction**2).sum(axis=0))    

def reflection_matrix(point, normal):
    """
    Return matrix to mirror at plane defined by point and normal vector.
    
    Inspired by Christoph Gohlke's transformations.py

    >>> v0 = np.random.random(4) - 0.5
    >>> v0[3] = 1.
    >>> v1 = np.random.random(3) - 0.5
    >>> R = reflection_matrix(v0, v1)
    >>> np.allclose(2, numpy.trace(R))
    True
    >>> np.allclose(v0, np.dot(R, v0))
    True
    >>> v2 = v0.copy()
    >>> v2[:3] += v1
    >>> v3 = v0.copy()
    >>> v2[:3] -= v1
    >>> np.allclose(v2, np.dot(R, v3))
    True

    """
    M = np.identity(4)
    M[:3, :3] -= 2.0 * np.outer(normal, normal)
    M[:3, 3] = (2.0 * np.dot(point[:3], normal)) * normal
    return M

def reflect(points_to_reflect,points_in_plane):
    """
    Reflect points around a plane.
    
    @param points_to_reflect = N x 3 points1...pointN
    @param points_in_plane: 3 X 3 (point1, point2, point3)
    @return: N x 3 reflected points
    """
    ax3 = np.ones(len(points_to_reflect))
    points = np.vstack([points_to_reflect.T,ax3])
    normal = normal_from_points(*points_in_plane)
    M = reflection_matrix(points_in_plane[0],normal)
    reflected_points = np.dot(M,points)
    return reflected_points[:3].T
