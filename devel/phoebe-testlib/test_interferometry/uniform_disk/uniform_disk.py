import phoebe
import numpy as np
import astropy.units as units
import matplotlib.pyplot as plt
from scipy.special import j1

KEPLERMAX=2000

def apparentPosition(self, t, pars):
    """
    Computes position of a binary omn the celestial sphere.
    INPUT:
      t			= time array (JD)
      pars		= orbital parameters 
      (a,e,i,o,O,T0,P, do)
    """
  
    a = pars[0];
    e = pars[1];
    i = pars[2];
    o = pars[3];
    O = pars[4];
    T0 = pars[5];
    P = pars[6];
    # advance of the periapsis is not entirely
    # necessary, so it is not mandatory to pass
    # it
    try:
      do = pars[7]
    except:
      do = 0.0
   
    # the do is inputted in rad/yr, so we need 
    # convert it
    do = do/365.25
   
    # now compute the value of the periastron
    # length
    o = o + (t-T0)*do
      
    # true anomaly
    M = 2*np.pi*(t-T0)/P;
    
    #Eccentric anomaly	- solution of 
    # the Kepler equation

    # Do not solve the eq. at very low 
    # eccentricities
    if e < 1e-6:
      V	= M;
    else:
      E0 = M.copy();
      E1 = M+e*np.sin(E0);
      
      while (np.absolute(E1-E0) > KeplerEqToler).any() | (kepcount < KEPLERMAX):
	  E0 = E1
	  E1 = M + e*np.sin(E0)

      # True anomaly
      V				= 2*np.arctan2(np.sqrt((1+e)/(1-e))*np.tan(E1/2),1);
      
      # transform angles -pi, pi -> 0, 2pi
      ind 			= np.where(V < 0.0)
      V[ind]			= 2*np.pi-np.absolute(V[ind])
    
    # mutual distance
    R				= a*(1-e*e)/(1 + e*np.cos(V));
    
    # position angle
    pa				= (np.arctan2(np.sin(V + o)*np.cos(i), np.cos(V + o)) + O)%(2*np.pi);
    
    # separation - not singular
    si				= (np.sin(i))**2;
    co				= (np.sin(V+o))**2;
    rho 			= R*(1-si*co)**0.5;

    # returns the position in Cartesian coordinates 
    # x = ordinate, y = abscissa
    return rho*np.sin(pa), rho*np.cos(pa);  
  
def uniformDisk(u,v,pars):
  """
  Visibility variations of a uniform model in Polar coordinates.
  Call: V = pointSource(u,v,pars);
  INPUT:
    u			= x component of the spatial frequency
    v			= y component of the spatial frequency
    pars		= parameters defing the model pars=(x-cartesian coordinate,
			  y-cartesian coordinate, theta-diameter, L-luminosity fraction.)
  OUTPUT:
    V			= complex visibility
  """
  x			= (pars[0]*units.mas).to('rad').value;
  y			= (pars[1]*units.mas).to('rad').value;
  theta			= (pars[2]*units.mas).to('rad').value;
  
  L			= pars[3];
  argument		= np.pi*theta*((u**2 + v**2)**0.5);
  if pars[2] > 1e-6:
      V			= (2*L*j1(argument)/argument)*np.exp(-2j*np.pi*(u*x+v*y));
  else:
      V			= L*np.exp(-2j*np.pi*(u*x+v*y));
  
  return V;

def setup_star(bundle, address, properties):
    """
    Setups the star.
    """
    for prop in properties.keys():
	full_adress = '@'.join([address, prop])
	#print full_adress
	bundle.set_value(full_adress, properties[prop])
	
	
    return bundle

def save_observables_p2(bundle):
    """
    """
    
    for dataref in bundle.get_datarefs():
	bundle.write_syn(dataref, dataref+'.p2.syn')
	
def save_observables_arrs(u, v, ew, t, vis2=None, phase=None, u2=None, v2=None, t3a=None, t3p=None, dataref='', fmt='%15.10e'):
    """
    """
    ew = ew/1e-10
    names = ['ucoord', 'vcoord', 'eff_wave', 'time', 'vis2', 'vphase', 'ucoord2', 'vcoord2', 't3ampl', 't3phase']
    save_cols = []
    header = []
    for i,x in enumerate([u, v, ew, t, vis2, phase, u2, v2, t3a, t3p]):
	if x is not None:
	    save_cols.append(x)
	    header.append(names[i])
    header = "".join(["%15s " % (x) for x in header])	    
    np.savetxt(dataref+'.analytic.syn', np.column_stack(save_cols), fmt=fmt, header=header)
      

def sim_baselines(nbase, angs=[0.0, 90.], eff_wave=6.56e-7):
    """
    Generates spatial frequencies for kind of
    CHARA baselines.
    """
    bmin = 0.0
    bmax = 300.0
    
    # generates baselines
    b = bmin + (bmax-bmin)*np.random.random(nbase)
    
    # generates position angles or the baselines
    nang = len(angs)
    sublen = len(b)/nang
    u = np.zeros(len(b))
    v = np.zeros(len(b))
    for i,ang in enumerate(angs):
	imin = i*sublen
	imax = (i+1)*sublen
	ang = np.radians(ang)
	if imax < len(b):
	    u[imin:imax] = b[imin:imax]*np.sin(ang)
	    v[imin:imax] = b[imin:imax]*np.cos(ang)
	else:
	    u[imin:] = b[imin:]*np.sin(ang)
	    v[imin:] = b[imin:]*np.cos(ang)
	    
    return u,v,np.ones(nbase)*eff_wave
	       
def sort_elements(els):
    """
    Convert dictionary to a list for apparentPosition.
    """
    key_order = ['sma','ecc','incl','per0','long_an','hjd0','period', 'dperdt']
    
    pars = []
    for key in key_order:
	pars.append(els[key])
	
    return pars

def RSolarToMas(R, d):
    """
    """
    # converts radius in solar diameters to radius in mas
    R	 = R*units.Rsun;
    d	 = d*units.parsec;
    theta = R.to('km')/d.to('km');
    theta = theta.value*180*60*60*1000/np.pi;
    return theta;
	
def get_uniform_star(radius, sma, q):
    """
    Creates a bundle with one RocheStar..
    """
    
    #create non-rotating star
    pot = phoebe.atmospheres.roche.radius2potential(radius, q, sma=sma, F=0.0)
    #print pot
    
    # define the star - within an orbit 
    star = phoebe.ParameterSet(context='component', pot=pot, label='star', gravb=0.0)
    orbit = phoebe.ParameterSet(context='orbit', q=q, sma=(sma,'Rsol'), c1label='star', c2label='anything')
    
    
    # position of the star
    position = phoebe.ParameterSet(context='position')
    
    # mesh
    mesh = phoebe.ParameterSet(context='mesh:marching')
    
    # create the star
    bag = phoebe.BinaryRocheStar(star, orbit=orbit, mesh=mesh, position=position)
    
    bundle = phoebe.Bundle(bag)
    
    return bundle
  
def get_uniform_star_simple(distance=10.0, radius=1.0):
    """
    """
    # use predefined bundle 
    star = phoebe.Bundle('vega')
    #print star
    
    # setup some properties
    props_star = dict(ld_func='uniform',
		      ld_coeffs=[1.0],
		      rotperiod=(10000.0, 'd'),
		      radius=radius,
		      gravb=0.0,
		      atm='blackbody',
		      incl=(90.0, 'deg'), 
		      long=(0.0, 'deg'))
    props_pos =  dict(distance=distance)
    setup_star(star, 'position', props_pos)
    setup_star(star, 'star', props_star)
    
    return star

def main():
    """
    """
    
    radius = 1.0
    distance = 10.0
    ndata = 100
    
    # logger
    logger = phoebe.get_basic_logger(clevel='debug')
    
    # get the budle
    b = get_uniform_star_simple(radius=radius, distance=distance)
    
    # get one set of baselines, effective wavelengths and temperatures
    u,v,ew = sim_baselines(ndata)
    t = np.zeros(len(u))
    
    # attach data to bundle
    b.if_fromarrays(time=t, ucoord=u, vcoord=v, eff_wave=(ew, 'm'), dataref='if01')
    
    # compute the synthetic visibility P2
    b.run_compute()
    save_observables_p2(b)
    
    # angular size of the star
    theta = 2*RSolarToMas(radius, distance)

    # compute the synthetic visibility analytic
    vis = uniformDisk(u/ew,v/ew, [0.0,0.0,theta,1.0])
    #print vis
    vis2 = vis.real**2
    visphase = np.angle(vis)
    
    save_observables_arrs(u,v,ew,t,vis2=vis2, phase=visphase, dataref='if01')
    
if __name__== '__main__':
    main()