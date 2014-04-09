import itertools
import numpy as np
import matplotlib.pyplot as plt
import phoebe
import os

np.random.seed(1983)



def test_binary():
    """
    Eclipse detection: various binaries
    """
    radii = [0.01,0.1,0.5,0.99999,1.0,1.0001,2.]
    if not os.path.isfile('test_graham_binary.phoebe'):
        mesh = phoebe.PS('mesh:marching',delta=0.05)
        star = phoebe.create.star_from_spectral_type('G2V', radius=1.00)
        star1 = phoebe.Star(star, mesh=mesh)
        
        for radius in radii:
            tennisball = phoebe.create.star_from_spectral_type('G2V',radius=radius, atm='blackbody',ld_func='uniform',ld_coeffs=[1.0])    
            star2 = phoebe.Star(tennisball, mesh=mesh)
            system = phoebe.BodyBag([star1,star2])
            system.set_time(0.)
            system.save('test_graham_binary_{:.6e}.phoebe'.format(radius))
    
    
    for radius in radii:    

        for nr,(x,y) in enumerate(np.random.uniform(size=(5,2))):
            if nr==0:
                x, y = 0, 0
            location = (x,y,-5)
            system = phoebe.load_body('test_graham_binary_{:.6e}.phoebe'.format(radius))
            star1, star2 = system.bodies
            
            star1.translate(loc=(x,y,-5))

            
            has_eclipse = phoebe.algorithms.eclipse.convex_graham(system.get_bodies(), first_iteration=True)

            for j in range(5):
                system.subdivide(threshold=0)
                has_eclipse = phoebe.algorithms.eclipse.convex_graham(system.get_bodies(), first_iteration=False)


            # Analytical:
            p = star2.params['star']['radius']/star1.params['star']['radius']
            z = np.sqrt((x/star1.params['star']['radius'])**2 + \
                        (y/star1.params['star']['radius'])**2)
            k1 = np.arccos( (1-p**2+z**2)/(2*z))
            k0 = np.arccos( (p**2+z**2-1)/(2*p*z))

            lam = 0
            if 1+p<z:
                lam = 0
            elif np.abs(1-p)<z<=(1+p):
                lam = 1.0/np.pi * (p**2*k0 + k1 - np.sqrt( (4*z**2- (1+z**2-p**2)**2)/4.))
            elif z<=(1-p):
                lam = p**2
            else:
                lam = 1

            analytical = 1-lam
            vis = star1.mesh['visible']
            par = star1.mesh['partial']
            vis_surface = (star1.mesh['size'][vis]*star1.mesh['mu'][vis]).sum()
            par_surface = (star1.mesh['size'][par]*star1.mesh['mu'][par]).sum()/2.0
            surface = vis_surface + par_surface
            numerical = surface/np.pi
            
            if analytical==0:
                continue
            
            assert(np.abs((analytical-numerical))<=5e-4)

if __name__=="__main__":
    phoebe.get_basic_logger()
    test_binary()