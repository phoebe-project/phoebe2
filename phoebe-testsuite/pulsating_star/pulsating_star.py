"""
Pulsating star (mode identification)
======================================

Last updated: ***time***

Simulation of a nonrotating star pulsating with one mode. The parameters of the
star are chosen to be similar to 12 Lacertae, but the rotational and pulsational
behaviour is chosen such that it is similar to the simulations done in Aerts et
al. 1993, so that comparison between the results presented there and those shown
here is facilitated. We generate the light curve, some effective temperature,
radial velocity and intensity maps, and a spectroscopic timeseries.

Inspired [Aerts1993]_

Initialisation
--------------

"""
# First, import necessary modules
import sys
import numpy as np
import matplotlib.pyplot as plt
import phoebe
from phoebe.utils import plotlib

logger = phoebe.get_basic_logger(clevel='INFO')

# Parameter preparation
# ---------------------
# Create a ParameterSet with parameters matching 12 Lac.
lac = phoebe.ParameterSet(context='star',add_constraints=True,label='12 Lac')
lac['teff'] = 25600.
lac['incl'] = 80.,'deg'
lac['radius'] = 8.8,'Rsol'
lac['mass'] = 14.4,'Msol'
lac['shape'] = 'sphere'
lac['atm'] = 'kurucz'
lac['ld_coeffs'] = 'kurucz'
lac['ld_func'] = 'claret'
mesh = phoebe.ParameterSet(context='mesh:marching',alg='c', maxpoints=100000)
mesh['delta'] = 0.08
lac.add(phoebe.Parameter(qualifier='veq',unit='km/s',value=200.))
lac.add_constraint('{rotperiod} = 2*np.pi*{radius}/{veq}')
print lac

# Create a ParameterSet with Parameters for the pulsation mode
freq_pars1 = phoebe.ParameterSet(context='puls',add_constraints=True)
freq_pars1['freq'] = 3.97,'cy/d'
freq_pars1['ampl'] = 0.01
freq_pars1['k'] = 0.0
freq_pars1['l'] = sys.argv[1] if len(sys.argv)>1 else 4
freq_pars1['m'] = sys.argv[2] if len(sys.argv)>2 else 4
freq_pars1['ledoux_coeff'] = 0.1585
freq_pars1['amplteff'] = 0.01
freq_pars1['scheme'] = 'nonrotating'

# Create a ParameterSet with parameters for the light curve
lcdep1 = phoebe.ParameterSet(context='lcdep',ref='light curve')
lcdep1['ld_func'] = 'claret'
lcdep1['ld_coeffs'] = 'kurucz'
lcdep1['atm'] = 'kurucz'

# Create a parameterSet with parameters for the spectra
spdep1 = phoebe.ParameterSet(context='spdep',ref='line profile')
spdep1['ld_func'] = 'claret'
spdep1['ld_coeffs'] = 'kurucz'
spdep1['atm'] = 'kurucz'

name = ""


# Computation of observables
# --------------------------

# Compute the light curve and spectra

name = 'pulsating_star_{0}rot_l{1:d}m{2:d}_{3}'.format((np.isinf(lac['rotperiod']) and 'no' or 'w'),freq_pars1['l'],freq_pars1['m'],name)

def extra_func(system,time,i_time):
    
    xlim,ylim,p = phoebe.image(system,select='rv',vmin=-200/8.,vmax=200/8.)
    plt.xlim(-9,9);plt.ylim(-9,9)
    plt.savefig('{}_rv_{:05d}.png'.format(name,i_time));plt.close()
    xlim,ylim,p = phoebe.image(system,select='teff',vmin=25500,vmax=25700)
    plt.xlim(-9,9);plt.ylim(-9,9)
    plt.savefig('{}_teff_{:05d}.png'.format(name,i_time));plt.close()
    xlim,ylim,p = phoebe.image(system,ref='light curve',select='proj2',vmax=1.2e5)
    plt.xlim(-9,9);plt.ylim(-9,9)
    plt.savefig('{}_proj2_{:05d}.png'.format(name,i_time));plt.close()


# Make sure to always have the same time period, so that we can see the frequency
# shifts
times = np.linspace(0,1./(freq_pars1['freq']-4./lac.get_value('rotperiod','d')),50)

# Create a DataSet with the template spectra and light curve
spobs1 = phoebe.SPDataSet(wavelength=np.linspace(4544, 4556, 200),
                          time=times, ref='line profile')
lcobs1 = phoebe.LCDataSet(time=times, ref='light curve')

star = phoebe.Star(lac,mesh,puls=[freq_pars1],
              pbdep=[lcdep1,spdep1], obs=[lcobs1, spobs1])

phoebe.compute(star, subdiv_num=0, extra_func=[extra_func])

"""

For synthetic Gaussian line profile templates:

+--------------------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------+--------------------------------------------------------------+
| l=4, m=4: line profile variations                      | l=4, m=4 effective temperature                               | l=4, m=4: radial velocity                              | l=4, m=4 intensity                                           |
+--------------------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------+--------------------------------------------------------------+ 
| .. image:: images_tut/pulsating_star_wrot_l4m4_.gif    | .. image:: images_tut/pulsating_star_wrot_l4m4__teff.gif     | .. image:: images_tut/pulsating_star_wrot_l4m4__rv.gif | .. image:: images_tut/pulsating_star_wrot_l4m4__proj2.gif    |
|    :scale: 100 %                                       |    :scale: 100 %                                             |    :scale: 100 %                                       |    :scale: 100 %                                             |
|    :width: 233px                                       |    :width: 233px                                             |    :width: 233px                                       |    :width: 233px                                             |
|    :height: 233px                                      |    :height: 233px                                            |    :height: 233px                                      |    :height: 233px                                            |
+--------------------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------+--------------------------------------------------------------+

+--------------------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------+--------------------------------------------------------------+
| l=2, m=2: line profile variations                      | l=2, m=2 effective temperature                               | l=2, m=2: radial velocity                              | l=2, m=2 intensity                                           |
+--------------------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------+--------------------------------------------------------------+ 
| .. image:: images_tut/pulsating_star_wrot_l2m2_.gif    | .. image:: images_tut/pulsating_star_wrot_l2m2__teff.gif     | .. image:: images_tut/pulsating_star_wrot_l2m2__rv.gif | .. image:: images_tut/pulsating_star_wrot_l2m2__proj2.gif    |
|    :scale: 100 %                                       |    :scale: 100 %                                             |    :scale: 100 %                                       |    :scale: 100 %                                             |
|    :width: 233px                                       |    :width: 233px                                             |    :width: 233px                                       |    :width: 233px                                             |
|    :height: 233px                                      |    :height: 233px                                            |    :height: 233px                                      |    :height: 233px                                            |
+--------------------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------+--------------------------------------------------------------+

+--------------------------------------------------------+----------------------------------------------------------------+--------------------------------------------------------+----------------------------------------------------------------+
| l=2, m=-2: line profile variations                     | l=2, m=-2 effective temperature                                | l=2, m=-2: radial velocity                             | l=2, m=-2 intensity                                            |
+--------------------------------------------------------+----------------------------------------------------------------+--------------------------------------------------------+----------------------------------------------------------------+ 
| .. image:: images_tut/pulsating_star_wrot_l2m-2_.gif   | .. image:: images_tut/pulsating_star_wrot_l2m-2__teff.gif      | .. image:: images_tut/pulsating_star_wrot_l2m-2__rv.gif| .. image:: images_tut/pulsating_star_wrot_l2m-2__proj2.gif     |
|    :scale: 100 %                                       |    :scale: 100 %                                               |    :scale: 100 %                                       |    :scale: 100 %                                               |
|    :width: 233px                                       |    :width: 233px                                               |    :width: 233px                                       |    :width: 233px                                               |
|    :height: 233px                                      |    :height: 233px                                              |    :height: 233px                                      |    :height: 233px                                              |
+--------------------------------------------------------+----------------------------------------------------------------+--------------------------------------------------------+----------------------------------------------------------------+

+--------------------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------+--------------------------------------------------------------+
| l=0, m=0: line profile variations                      | l=0, m=0 effective temperature                               | l=0, m=0: radial velocity                              | l=0, m=0 intensity                                           |
+--------------------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------+--------------------------------------------------------------+  
| .. image:: images_tut/pulsating_star_wrot_l0m0_.gif    | .. image:: images_tut/pulsating_star_wrot_l0m0__teff.gif     | .. image:: images_tut/pulsating_star_wrot_l0m0__rv.gif | .. image:: images_tut/pulsating_star_wrot_l0m0__proj2.gif    |
|    :scale: 100 %                                       |    :scale: 100 %                                             |    :scale: 100 %                                       |    :scale: 100 %                                             |
|    :width: 233px                                       |    :width: 233px                                             |    :width: 233px                                       |    :width: 233px                                             |
|    :height: 233px                                      |    :height: 233px                                            |    :height: 233px                                      |    :height: 233px                                            |
+--------------------------------------------------------+--------------------------------------------------------------+--------------------------------------------------------+--------------------------------------------------------------+


Comparison between Gaussian and *real* templates:

+-----------------------------------------------------+-------------------------------------------------------------+---------------------------------------------------------------+
| l=4, m=4: Gaussian template                         | l=4, m=4: Silicon line                                      | l=4, m=4 Helium line                                          |
+-----------------------------------------------------+-------------------------------------------------------------+---------------------------------------------------------------+ 
| .. image:: images_tut/pulsating_star_wrot_l4m4_.gif | .. image:: images_tut/pulsating_star_wrot_l4m4_silicon.gif  | .. image:: images_tut/pulsating_star_wrot_l4m4_helium.gif     |
|    :scale: 100 %                                    |    :scale: 100 %                                            |    :scale: 100 %                                              |
|    :width: 233px                                    |    :width: 233px                                            |    :width: 233px                                              |
|    :height: 233px                                   |    :height: 233px                                           |    :height: 233px                                             |
+-----------------------------------------------------+-------------------------------------------------------------+---------------------------------------------------------------+


"""         
         
         
         
# Analysis of results
# -------------------
# Collect the times of observations and calculated flux.
times = star.get_synthetic(category='lc',ref=0)['time']
flux = np.array(star.get_synthetic(category='lc',ref=0)['flux'])

# Plot light curve
plt.figure()
plt.plot(times,flux/flux.mean(),'ko-')
plt.xlim(times[0],times[-1])
#if 'helium' in name: plt.ylim(0.8,1.004)
#else:
plt.ylim(0.996,1.004)
plt.grid()
plt.xlabel('Time [d]')
plt.ylabel('Normalised flux')
plt.savefig('{0}_lc'.format(name))
plt.close()

"""
+--------------------------------------------------------+--------------------------------------------------------------+
| l=4, m=4: Light curve                                  | l=0, m=0 Light curve                                         |
+--------------------------------------------------------+--------------------------------------------------------------+ 
| .. image:: images_tut/pulsating_star_wrot_l4m4__lc.png | .. image:: images_tut/pulsating_star_wrot_l0m0__lc.png       |
|    :scale: 100 %                                       |    :scale: 100 %                                             |
|    :width: 233px                                       |    :width: 233px                                             |             
|    :height: 233px                                      |    :height: 233px                                            |             
+--------------------------------------------------------+--------------------------------------------------------------+
| l=2, m=2: Light curve                                  | l=2, m=-2 Light curve                                        |
+--------------------------------------------------------+--------------------------------------------------------------+ 
| .. image:: images_tut/pulsating_star_wrot_l2m2__lc.png | .. image:: images_tut/pulsating_star_wrot_l2m-2__lc.png      |
|    :scale: 100 %                                       |    :scale: 100 %                                             |
|    :width: 233px                                       |    :width: 233px                                             |             
|    :height: 233px                                      |    :height: 233px                                            |             
+--------------------------------------------------------+--------------------------------------------------------------+


"""

spectra = star.get_synthetic(category='sp',ref=0)

# Plot spectra of stars separately.
for j in range(len(spectra['wavelength'])):
    wave = spectra['wavelength'][j]
    spec = spectra['flux'][j]
    cont = spectra['continuum'][j]
    plt.figure()
    plt.plot(wave,spec/cont,'k-')
    plt.grid()
    plt.xlim(wave[0], wave[-1])
    if 'helium' in name: plt.ylim(0.8,1.004)
    else:                plt.ylim(0.93,1.005)
    plt.xlabel('Wavelength [Angstrom]')
    plt.ylabel('Normalized flux')
    plt.savefig('{0}_spec_{1:03d}'.format(name,j))
    plt.close()
    plt.figure(100)
    if j==0:
        ptp = (spec/cont).ptp()
    plt.plot(wave,spec/cont+j*ptp,'k-')
plt.figure(100)
plt.xlim(wave[0], wave[-1])
plt.ylim(1-1.2*ptp,1.+j*ptp*1.05)
plt.xlabel('Wavelength [Angstrom]')
plt.ylabel('Normalized flux')
plt.savefig('{0}spec'.format(name))
plt.close()    

for ext in ['.gif','.avi']:
    for root in ['spec','teff','rv','proj2']:
        plotlib.make_movie('{0}_{1}*.png'.format(name,root),output='{0}_{1}{2}'.format(name,root,ext))
    
    
