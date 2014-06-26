"""
Spot drawing functions.
"""
import logging
import pylab as pl
import numpy as np
from numpy import cos,arccos,pi,arcsin,sin,sqrt
from phoebe.utils import coordinates
from phoebe.atmospheres import cspots

logger = logging.getLogger('ATM.SPOTS')

def distance_on_sphere(loc1,loc2,R):
   """
   Compute the distance between two points on a sphere.
   
   Phi is longitude
   th is colatitude
   
   One of the two arguments may be vectors (R can then also be a vector).
   
   @parameter phi1: longitude of point 1
   @parameter th1: colatitude of point 1
   @parameter phi2: longitude of point 2
   @parameter th2: colatitude of point 2
   @parameter R: radius of the sphere.
   """
   (phi1,th1),(phi2,th2) = loc1, loc2
   #-- old way
   #z = arccos(sin(th1)*sin(th2)*cos(phi1-phi2) + cos(th1)*cos(th2))
   #-- new way
   z = 2*arcsin(sqrt(sin(0.5*(th1-th2))**2 +\
            sin(th1)*sin(th2)*sin(0.5*(phi2-phi1))**2))
   #return R*z
   return z


def add_circular_spot(the_system,time,spot_pars,update_temperature=True):
    """
    Add a circular spot to a body.
    """
    inside,boundary = detect_circular_spot(the_system,time,spot_pars)
    if update_temperature:
        teffratio = spot_pars.request_value('teffratio')#0.5
        abunratio = spot_pars.request_value('abunratio')#0.5
        mesh = the_system.mesh.copy()
        mesh['teff'][inside] = mesh['teff'][inside] * teffratio
        mesh['abun'][inside] = mesh['abun'][inside] + abunratio # it's in log!
        #mesh['teff'][boundary] = (1.-spot_deltatemp)*mesh['teff'][boundary]
        #mesh['teff'][-inside&-boundary] = 5777.
        mesh['partial'][inside] = False
        the_system.mesh = mesh
        #logger.debug("added circular spot, covering ({}+{}) triangles".format(inside.sum(),boundary.sum()))

def detect_circular_spot(the_system,time,spot_pars):
    """
    Mark triangles to be on the the spot radius, for future subdivision.
    
    This function returns spots inside and on the spotradius, so they can be
    easily subdivided and local parameter-adjusted.
    """
    spot_t0 = spot_pars.request_value('t0')
    spot_long = spot_pars.request_value('long','rad')#pi/4.
    spot_colat = spot_pars.request_value('colat','rad')#pi/2. # on the equator
    spot_radius = spot_pars.request_value('angrad','rad') #0.3
    #-- by default, the spot rotates on the surface of the star. If we want
    #   differential rotation, we need to do that wrt to the rotation
    if 'star' in the_system.params:
        rotperiod = the_system.params['star'].request_value('rotperiod','d')
        diffrot = the_system.params['star'].request_value('diffrot','d')
        if diffrot:
            latperiod = rotperiod + diffrot*np.sin(spot_colat)**2
            spot_long = spot_long - 2*np.pi/rotperiod*(time-spot_t0)
            spot_long = spot_long + 2*np.pi/latperiod*(time-spot_t0)
            
        
    
    mesh = the_system.mesh.copy()
    
    if False:
        x = cspots.detect_circular_spot(np.array(mesh['_o_triangle'].T,float),(spot_long,spot_colat,spot_radius))
        triangles_on_spotradius,triangles_in_spotradius = np.array(x.reshape((2,-1)),bool)
        mesh['partial'] = mesh['partial'] | triangles_on_spotradius
        the_system.mesh = mesh
        logger.info('marked circular spot (on:{0:d}, in:{1:d})'.format(sum(triangles_on_spotradius),sum(triangles_in_spotradius)))
        logger.info('marked circular spot')
        return triangles_in_spotradius,triangles_on_spotradius
    else:
        #-- and mark triangles on the edge of the spot for subdivision. Triangles
        #   on the edge or defined as those with at least on vertex inside and one
        #   vertex outside the spot. They will be marked to be partially visible,
        #   but their state of 'visible' or 'hidden' remains the same.
        r1,phi1,theta1 = coordinates.cart2spher_coord(*mesh['_o_triangle'][:,0:3].T)
        r2,phi2,theta2 = coordinates.cart2spher_coord(*mesh['_o_triangle'][:,3:6].T)
        r3,phi3,theta3 = coordinates.cart2spher_coord(*mesh['_o_triangle'][:,6:9].T)
        distance1 = distance_on_sphere((spot_long,spot_colat),(phi1,theta1),r1)
        distance2 = distance_on_sphere((spot_long,spot_colat),(phi2,theta2),r2)
        distance3 = distance_on_sphere((spot_long,spot_colat),(phi3,theta3),r3)
        
        allinside  = (distance1<=spot_radius)&(distance2<=spot_radius)&(distance3<=spot_radius) 
        alloutside = (distance1>spot_radius)&(distance2>spot_radius)&(distance3>spot_radius)
        triangles_on_spotradius = -allinside & -alloutside
        triangles_in_spotradius = allinside
        mesh['partial'] = mesh['partial'] | triangles_on_spotradius
        the_system.mesh = mesh
        logger.info('marked circular spot (on:{0:d}, in:{1:d})'.format(sum(triangles_on_spotradius),sum(triangles_in_spotradius)))
        return triangles_in_spotradius,triangles_on_spotradius
    
    
def worley_noise(system, time=0, feature_points=100, metric='f1', seed=None,
                 max_angle=-1):
    """
    Create a granulation type pattern of a quantity.
    
    This function returns values between 0 and 1, which can be used to scale
    certain quantities.
    """
    # Get the surface points in spherical coordinates
    r_, phi_, theta_ = system.get_coords()
    R = r_.max()
    
    # define the metric
    if metric == 'f1':
        metric = lambda f: f[0]
    elif metric == 'f2':
        metric = lambda f: f[1]
    elif metric == 'f2-f1':
        metric = lambda f: f[1]-f[0]
    elif metric == 'f1+f2':
        metric = lambda f: f[0]+f[1] 
    
    # Initiate the sender and generate random points
    if seed is not None:
        np.random.seed(seed)
    z = np.random.uniform(low=-R, high=R, size=feature_points)
    seeds_phi = np.random.uniform(low=-np.pi, high=np.pi, size=feature_points)
    seeds_theta = np.arcsin(z/R) + np.pi/2. # between 0 and pi
    
    # We should know what the previous state is (we can access the previous time
    # via self.time), and compute incrementatlly up to the current point
    # if time < self.time: raise ValueError
    # delta_time = time - self.time
    # steps = delta_time / elementary_delta_time
    # -- FOR EACH POINT
    #    -- FOR EACH ELEMENTARY DT
    #        if dissappear: generate new one and follow that one
    #
    #        distances = stats.distr.exponential.draw(size=steps)
    #        delta_thetas = stats.distr.uniform(0, distance, size=steps)
    #        delta_phis = compute_from_distances_and_theta
    # net_delta_theta = np.sum(delta_thetas)
    # net_delta_phi = np.sum(delta_phis)
    # -- now we know where the go to for each point
    
    # Set the seeder to be this time point for any other computation
    #deltat = 0.001
    
    # somewhere between this time step
    #np.random.seed(int(time/deltat))
    #distance1 = 1.0 # draw random distance
    #delta_phi1 = 1.0 # draw random value from between 0 and distance from uniform distribution
    #delta_theta1 = 1.0 # compute delta_theta such that the distance is recovered
    
    # and this time step
    #np.random.seed(int(time/deltat)+1)
    
    # 1. Define the delta_phi and deltha_theta to walk into:
    #      - draw random distance
    #      - define arbitrary deltha_phi
    #      - compute deltha_theta
    
    
    
    # Prepare arrays for output storage
    directions = np.zeros((2, len(phi_), 3))
    value = np.zeros_like(phi_)
    
    # Find max distance to two closest points
    maxDist1 = 0.0
    maxDist2 = 0.0
    
    for i, (kphi, kth) in enumerate(zip(phi_, theta_)):
        # create a sorted list of distances to all seed points
        dists = distance_on_sphere((seeds_phi, seeds_theta), (kphi, kth), 1.0)
        
        # Update the maximum distances
        dists.sort()
        if dists[0] > maxDist1:
            maxDist1 = dists[0]
        if dists[1] > maxDist2:
            maxDist2 = dists[1]
    
    # Set the values
    maxDist2 = maxDist2/maxDist1*2./np.sqrt(feature_points)
    maxDist1 = 2./np.sqrt(feature_points)
    
    for i, (kphi, kth) in enumerate(zip(phi_, theta_)):
        
        # create a sorted list of distances to all seed points
        dists = distance_on_sphere((seeds_phi, seeds_theta), (kphi, kth), 1.0)
        sa = np.argsort(dists)
        
        f1 = dists[sa[0]] / maxDist1
        f2 = dists[sa[1]] / maxDist2
        value[i] = metric((f1,f2))
        
        # Store coordinates of two points to create directional vectors
        directions[0,i] = (1.0, kphi, kth)
        directions[1,i] = (1.0, seeds_phi[sa[0]], seeds_theta[sa[0]])
    
    # Scale the values between 0 and 1
    #value = value - value.min()
    #value /= value.max()
    
    directions[0] = np.array(coordinates.spher2cart_coord(*directions[0].T)).T
    directions[1] = np.array(coordinates.spher2cart_coord(*directions[1].T)).T
    
    # Generate vectors:
    # Find plane through three points and center of sphere: The equation is
    # then plane[0]*x + plane[1]*y + plane[2]*z = 0
    index = np.array([1,0,2])
    plane = np.cross(directions[0], directions[1])
    
    # Find tangent vector at mesh point
    tangent = np.cross(directions[0], plane)#[:,index] # only include index if we don't want to rotate
    
    # No rotation
    if max_angle == 0:
        return tangent[:,index]
    
    # Normalise directional vector
    norm = np.sqrt(np.sum(tangent**2, axis=1))[:, None]
    tangent = tangent / norm
        
    # Auto-rotation
    if max_angle==-1:
        tangent2 = (directions[0] + tangent) * (1+value[:,None])
    # Invalid rotation
    elif max_angle<0 or max_angle>-90:
        raise ValueError("Invalid angle for granulation vector field (strictly 0<angle<90)")
    # Manual rotation
    else:
        angles = np.arctan(value)
        angles = angles / np.abs(angles).max() * max_angle / 180.*np.pi
        value = np.tan(angles)
        tangent2 = (directions[0] + tangent) * (1+value[:,None])
    
    tangent = tangent2 - directions[0]
    tangent = tangent[:, index]
    
    norm = np.sqrt(np.sum(tangent**2, axis=1))[:, None]
    tangent = tangent / norm
    
    # That's it!
    return value, tangent #/ np.sqrt(np.sum(directions**2, axis=1))[:,None]