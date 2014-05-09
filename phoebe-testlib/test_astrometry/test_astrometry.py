import phoebe
import numpy as np
import os

basedir = os.path.dirname(os.path.abspath(__file__))

def setup_star(pmra, pmdec, ra, dec, distance):
    # Create the star
    star = phoebe.create.from_library('vega', create_body=True)
    star.params['position']['epoch'] = 'J1991.25'
    star.params['position']['pmra'] = pmra
    star.params['position']['pmdec'] = pmdec
    star.params['position']['ra'] = ra
    star.params['position']['dec'] = dec
    star.params['position']['distance'] = distance
    
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
    assert(np.allclose(x,x_,atol=1.e-5))
    assert(np.allclose(y,y_,atol=1.e-5))
    #np.savetxt('eta_dra.pm', np.column_stack([x,y]))
    #return x,y
     
def test_polaris(return_output=False):
    """
    Astrometry: Polaris (near the equatorial pole)
    
    HIP11767
    """
    x, y = setup_star(44.22,-11.74,37.94614689,89.26413805, 1000./7.56)
    x_,y_ = np.loadtxt(os.path.join(basedir,'polaris.pm'), unpack=True)

    try:
        assert(np.allclose(x,x_,atol=1.e-5))
        assert(np.allclose(y,y_,atol=1.e-5))
    except AssertionError:
        if return_output:
            return (x,y), (x_,y_)
        else:
            raise
    #np.savetxt('polaris.pm', np.column_stack([x,y]))
    #return x,y

def test_lam_aqr():
    """
    Astrometry: Lambda Aquarius (ecliptic plane)
    
    HIP112961
    """
    x, y = setup_star( 19.51,32.71,343.15360192,-7.5796787, 1000./8.33)
    x_,y_ = np.loadtxt(os.path.join(basedir,'lam_aqr.pm'), unpack=True)
    assert(np.allclose(x,x_,atol=1.e-5))
    assert(np.allclose(y,y_,atol=1.e-5))
    #np.savetxt('lam_aqr.pm', np.column_stack([x,y]))
    #return x,y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    logger = phoebe.get_basic_logger()
    
    test_eta_dra()
    #plt.subplot(111,aspect='equal')
    #plt.plot(x,y,'k-')
    
    output =test_polaris(return_output=True)
    if output is not None:    
        plt.figure()
        plt.subplot(111,aspect='equal')
        plt.plot(output[0][0]-output[1][0],output[0][1]-output[1][1])
        plt.title('error in polaris')
        print(output[0])
        print(output[1])
        plt.show()
    
    test_lam_aqr()
    #plt.figure()
    #plt.subplot(111,aspect='equal')
    #plt.plot(x,y,'k-')
    
    #plt.show()
