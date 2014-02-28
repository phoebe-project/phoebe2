import phoebe
import numpy as np
import os

basedir = os.path.dirname(os.path.abspath(__file__))

def setup_star(pmra, pmdec, ra, dec, distance):
    # Create the star
    star = phoebe.create.from_library('vega', create_body=True)
    star.params['globals']['epoch'] = 'J1991.25'
    star.params['globals']['pmra'] = pmra
    star.params['globals']['pmdec'] = pmdec
    star.params['globals']['ra'] = ra
    star.params['globals']['dec'] = dec
    star.params['globals']['distance'] = distance
    
    # Read in the Hipparcos positions
    times, x_earth, y_earth, z_earth = np.genfromtxt(os.path.join(basedir,'hipparcos_position.dat'),
                     delimiter=',', skip_header=53, skip_footer=32,
                     usecols=(0,2,3,4), unpack=True)
    
    # Create the dep and obs
    amdep = phoebe.PS(context='amdep', ref='myam')
    amobs = phoebe.DataSet('amobs', ref='myam', time=times, eclx=x_earth,
                       ecly=y_earth, eclz=z_earth)[::100]

    # Add the dep and obs, and compute
    star.add_pbdeps([amdep])
    star.add_obs([amobs])
    star.init_mesh()
    star.compute()

    # Get the synthetic stuff
    output = star.get_synthetic(category='am',ref='myam').asarray()
    return output['delta_ra']*3600*1000, output['delta_dec']*3600*1000


def test_eta_dra():
    """
    Astrometry: Eta Dra (near ecliptic pole)
    
    HIP80331
    """
    x, y = setup_star(-16.98,56.68, 245.99794523,61.51407536, 1000./37.18)
    x_,y_ = np.loadtxt(os.path.join(basedir,'eta_dra.pm'), unpack=True)
    assert(np.allclose(x,x_))
    assert(np.allclose(y,y_))
    #np.savetxt('eta_dra.pm', np.column_stack([x,y]))
    #return x,y
     
def test_polaris():
    """
    Astrometry: Polaris (near the equatorial pole)
    
    HIP11767
    """
    x, y = setup_star(44.22,-11.74,37.94614689,89.26413805, 1000./7.56)
    x_,y_ = np.loadtxt(os.path.join(basedir,'polaris.pm'), unpack=True)
    assert(np.allclose(x,x_))
    assert(np.allclose(y,y_))
    #np.savetxt('polaris.pm', np.column_stack([x,y]))
    #return x,y

def test_lam_aqr():
    """
    Astrometry: Lambda Aquarius (ecliptic plane)
    
    HIP112961
    """
    x, y = setup_star( 19.51,32.71,343.15360192,-7.5796787, 1000./8.33)
    x_,y_ = np.loadtxt(os.path.join(basedir,'lam_aqr.pm'), unpack=True)
    assert(np.allclose(x,x_))
    assert(np.allclose(y,y_))
    #np.savetxt('lam_aqr.pm', np.column_stack([x,y]))
    #return x,y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    logger = phoebe.get_basic_logger()
    
    x, y = test_eta_dra()
    plt.subplot(111,aspect='equal')
    plt.plot(x,y,'k-')
    
    x, y = test_polaris()
    plt.figure()
    plt.subplot(111,aspect='equal')
    plt.plot(x,y,'k-')
    
    x, y = test_lam_aqr()
    plt.figure()
    plt.subplot(111,aspect='equal')
    plt.plot(x,y,'k-')
    
    plt.show()
