"""
Anisotropic scattering
========================

We compute three light curves of a binary system consisting of a hot and a cool
companion. We assume that the surface of the cool companion is completely
reflective (albedo=1), but that the direction of the scattering can be different.
The first case has strong backward scattering, the second case is isotropic, and
the third case has strong forward scattering.

Initialisation
--------------

"""
# First, import necessary modules.
import numpy as np
import matplotlib.pyplot as plt
import phoebe
from phoebe.parameters import tools

# Take care of output to the screen
logger = phoebe.get_basic_logger(clevel='INFO')

# Create the two stars and put them in a binary orbit
star1 = phoebe.PS('star', teff=30000, ld_func='linear', ld_coeffs=[0.5], radius=(1., 'Rsol'))
star2 = phoebe.PS('star', teff=1000, ld_func='linear', ld_coeffs=[0.5], radius=(2.,'Rsol'), alb=1)

comp1, comp2, orbit = phoebe.create.binary_from_stars(star1, star2, period=(1,'d'))
orbit['incl'] = 85.0

# Create the mesh and the light curves
mesh1 = phoebe.ParameterSet(context='mesh:marching', delta=0.075)
mesh2 = phoebe.ParameterSet(context='mesh:marching', delta=0.05)

lcdep1 = phoebe.PS('lcdep', ld_func='linear', ld_coeffs=[0.5], alb=0, ref='mylc')
lcdep2 = phoebe.PS('lcdep', ld_func='linear', ld_coeffs=[0.5], alb=1, ref='mylc')

# Add scattering properties to the light curve. We'll use the Henyey-Greenstein
# scattering phase function which only has one parameter (the asymmetry parameter)
tools.add_scattering(lcdep2, 'henyey')

# Let's compute the light curve in 200 points
time = np.linspace(0.05, 0.95*orbit['period'], 200)
lcobs = phoebe.LCDataSet(time=time, ref='mylc')

# Create the system consisting of two BinaryRocheStars
star1 = phoebe.BinaryRocheStar(comp1, orbit, mesh1, pbdep=[lcdep1])
star2 = phoebe.BinaryRocheStar(comp2, orbit, mesh2, pbdep=[lcdep2])
system = phoebe.BodyBag([star1, star2], obs=[lcobs])

# Cycle over the different asymmetry values and compute the light curve

for asymmetry in [-0.8,0.0,0.8]:
    
    # Set asymmetry factor
    lcdep2['asymmetry'] = asymmetry

    # Clear previous results and compute the LC
    system.reset_and_clear()
    system.compute()

    # Plot the LC
    phoebe.plotting.plot_lcsyn(system, 'o-', label='Asymmyetry = {}'.format(asymmetry))
    
    
plt.xlabel('Time')
plt.ylabel('Flux')
plt.legend(loc='best').get_frame().set_alpha(0.5)
plt.savefig('anisotropic_scattering.png')

"""

Note that in the video below, the images are artificially brightened with 
a factor of 4 to make the effect better visible.

+----------------------------------------------+
| .. image:: images_tut/anisotropy_movie.gif   |
|    :scale: 100 %                             |
+----------------------------------------------+

"""
