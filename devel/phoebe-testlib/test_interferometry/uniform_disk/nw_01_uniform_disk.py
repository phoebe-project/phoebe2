import phoebe
import astropy.units as u
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
  x			= pars[0]*const.MasToRad;
  y			= pars[1]*const.MasToRad;
  theta			= pars[2]*const.MasToRad;
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
	bundle.set_value(full_adress, properties[prop])
	
    return bundle
    
def sort_elements(els):
    """
    Convert dictionary to a list for apparentPosition.
    """
    key_order = ['sma','ecc','incl','per0','long_an','hjd0','period', 'dperdt']
    
    pars = []
    for key in key_order:
	pars.append(els[key])
	
    return pars
	
def main():
    # create a uniform disk within P2
    # star = uniform disk
    star = phoebe.ParameterSet(context='star', 
			       radius=1.0, mass=1.0, rotperiod=0.0, 
			       irradiator=False, gravb=0.0)
    # position of the star
    position = phoebe.ParameterSet(context='position')
    
    # mesh
    mesh = phoebe.ParameterSet(context='mesh:marching')
    
    # create the star
    bag = phoebe.Star(star, mesh=mesh)
    
    
if __name__== '__main__':
    main()