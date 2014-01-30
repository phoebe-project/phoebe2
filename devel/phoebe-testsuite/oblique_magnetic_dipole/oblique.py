"""
Beta Cephei (oblique magnetic dipole in a rotating pulsating star)
===================================================================

This tutorial builds a rotating, pulsating star with an oblique magnetic field,
much like Beta Cephei.

The point is to make a visualisation of the star, together with the evolution
and variability of the Stokes I and V profiles.

"""

# Import necessary modules and set up a logger
import matplotlib.pyplot as plt
import numpy as np
import phoebe
from phoebe.parameters import tools
from phoebe.parameters import datasets
from phoebe.utils import plotlib

logger = phoebe.get_basic_logger()

# Define the parameters of the star. We're inspired by [Donati1997]_,
# [Morel2006]_ and [Nieva2013]_.

star = phoebe.ParameterSet('star',label='beta Cephei')
star['atm'] = 'kurucz'
star['ld_func'] = 'claret'
star['ld_coeffs'] = 'kurucz'
star['abun'] = -0.2
star['teff'] = 27000.,'K'
star['rotperiod'] =  12.001663246,'d'
star['incl'] = 60.,'deg'
star['long'] = 20, 'deg'
star['mass'] = 14.,'Msol'
star['radius'] = 6.4,'Rsol'
star['vgamma'] = 7.632287,'km/s'

# Just for fun, also add the parallax, the surface gravity and vsini (and define
# a mesh).
tools.add_parallax(star,parallax=4.76,unit='mas')
tools.add_surfgrav(star,3.70,derive='radius')
tools.add_vsini(star,32.359421,unit='km/s',derive='incl')

mesh = phoebe.PS('mesh:marching',delta=0.05, alg='c')


# For the parameters of the pulsation mode, we're inspired by [Telting1997]_:
freq1 = phoebe.ParameterSet('puls',freq=  5.24965427519,phase=0.122545,ampl=0.10525/50.,l=3,m=2,amplteff=0.01)

# For the parameters of the magnetic dipole, we take guesses from our own
# research:
mag_field = phoebe.PS('magnetic_field')
mag_field['Bpolar'] = 276.01,'G'
mag_field['beta'] = 61.29,'deg'
mag_field['phi0'] = 80.522,'deg'


# Fake some observations, and specify the passband dependent parameters.
plobs = datasets.PLDataSet(time=np.linspace(0, 0.5*star['rotperiod'], 500),
                           wavelength=np.linspace(399.92,400.08,500), ref='mylsd',
                           columns=['time','wavelength','flux','V','continuum'])

pldep = phoebe.ParameterSet('pldep',ref='mylsd', weak_field=False,
                            atm='kurucz',ld_func='claret',ld_coeffs='kurucz')

# Now we can build the star itself.
betacep = phoebe.Star(star, mesh, puls=[freq1], magnetic_field=mag_field,
                      pbdep=pldep, obs=plobs)

# Before computing everything, we define a function that we'll call after every
# time point to make a nice plot
def make_figure(betacep, time, i, name='oblique_dipole', **kwargs):
    # Start the figure
    
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes([0.05,0.1,0.65,0.85],axisbg='0.7',aspect='equal')
    fig.set_facecolor('0.7')
    fig.set_edgecolor('0.7')

    # Plot the effective temperature map
    xlim, ylim, patchT = betacep.plot2D(ref='mylsd',select='teff', ax=ax)
    cbar = plt.colorbar(patchT)
    cbar.set_label("Effective temperature [K]")

    # Plot contours of constant mangetic field
    phoebe.observatory.contour(betacep, select='B', linewidths=2,
                            cmap=plt.cm.spectral_r, levels=np.arange(140,260,20),
                            )
                            #prop=dict( inline=1, fontsize=14, fmt='%.0f G'))

    # Plot longitudinal and latitudinal lines
    phoebe.observatory.contour(betacep, select='longitude', linewidths=1,
                            linestyles='solid', colors='k')
    phoebe.observatory.contour(betacep, select='latitude', linewidths=1,
                            linestyles='solid', colors='k')

    plt.xlabel("Plane of sky coordinate [$R_\odot$]")
    plt.ylabel("Plane of sky coordinate [$R_\odot$]")
    
    ax = plt.axes([0.77,0.08,0.22,0.41])
    syn = betacep.get_synthetic(category='pl', ref='mylsd').asarray()
    wave = syn['wavelength'][-1]
    velo = phoebe.convert('nm','km/s',wave,wave=(400., 'nm'))
    plt.plot(velo,syn['flux'][-1]/syn['continuum'][-1], 'k-', lw=2)
    plt.ylim(0.86,1.01)
    plt.xlim(velo[0],velo[-1])
    plt.xlabel(r"Velocity [km s$^{-1}$]")
    plt.ylabel("Normalised flux")
    plt.grid()
    
    ax = plt.axes([0.77,0.55,0.22,0.41])
    plt.plot(velo,syn['V'][-1]/syn['flux'][-1]*1e4, 'r-', lw=2)
    plt.ylim(-5,5)
    plt.xlim(velo[0],velo[-1])
    plt.ylabel("Stokes V / Stokes I")
    plt.grid()
    
    plt.savefig(('{}_{:.3f}'.format(name,time)).replace('.','_'))
    
    plt.close()


# All is left is to do the computations and make a movie!
params = phoebe.ParameterSet('compute',subdiv_num=0,eclipse_alg='only_horizon')
phoebe.compute(betacep,extra_func=[make_figure],params=params)
#mpi = phoebe.PS('mpi',np=4)
#phoebe.compute(betacep,params=params,mpi=mpi)

#plotlib.make_movie('oblique_dipole_*.png',output='oblique_dipole.gif')

"""

+---------------------------------------------------+----------------------------------------------------+
| .. image:: images_tut/oblique_dipole_0_000.png    | .. image:: images_tut/oblique_dipole_1_503.png     |
|    :scale: 50%                                    |    :scale: 50%                                     |
|    :width: 266px                                  |    :width: 266px                                   |
|    :height: 266px                                 |    :height: 266px                                  |
+---------------------------------------------------+----------------------------------------------------+
| .. image:: images_tut/oblique_dipole_3_006.png    | .. image:: images_tut/oblique_dipole_6_001.png     |
|    :scale: 50%                                    |    :scale: 50%                                     |
|    :width: 266px                                  |    :width: 266px                                   |
|    :height: 266px                                 |    :height: 266px                                  |
+---------------------------------------------------+----------------------------------------------------+

"""
