"""
Sirius
======

last updated: ***time***

Compute Sirius's interferometric visibilities (see [Kervella2003]_).

Input parameters:

* Kurucz atmospheres
* Claret's limbdarkening law
* Roche surface.

We compare the values of the total projected intensity (calculated
numerically and the observed values). We make an image and a plot of the radial
velocities of all surface elements.

Initialisation
--------------

"""
# First, import necessary modules.
import numpy as np
import pylab as pl
import phoebe

# Parameter preparation
# ---------------------

# Create a ParameterSet with parameters closely matching Sirius.
sirius = phoebe.ParameterSet(context='star',add_constraints=True)
sirius['teff'] = 9900,'K'
sirius['mass'] = 2.3,'Msol'
sirius['radius'] = 1.711,'Rsol'
sirius['rotperiod'] = np.inf
sirius['ld_func'] = 'claret'
sirius['atm'] = 'kurucz'
sirius['ld_coeffs'] = 'kurucz'

# Position parameters
pos = phoebe.ParameterSet(context='position')
pos['distance'] = 2.637,'pc'

# To compute interferometric visibilities:
ifdep1 = phoebe.ParameterSet(context='ifdep')
ifdep1['ld_func'] = 'claret'
ifdep1['ld_coeffs'] = 'kurucz'
ifdep1['passband'] = '2MASS.KS'

mesh1 = phoebe.ParameterSet(frame='phoebe',context='mesh:marching')
mesh1['delta'] = 0.1

# And some data
data = np.array([[17.758,  0.8602,  0.0206], 
                 [18.139,  0.8742,  0.0210], 
                 [18.270,  0.8769,  0.0313],
                 [18.544,  0.8668,  0.0208], 
                 [18.684,  0.8748,  0.0309], 
                 [19.032,  0.8627,  0.0300], 
                 [22.063,  0.8092,  0.0280], 
                 [22.301,  0.8110,  0.0281], 
                 [22.527,  0.8048,  0.0278], 
                 [60.439,  0.1599,  0.0074], 
                 [60.574,  0.1608,  0.0062], 
                 [61.038,  0.1472,  0.0085], 
                 [62.027,  0.1433,  0.0062], 
                 [62.148,  0.1311,  0.0059], 
                 [63.651,  0.1243,  0.0063], 
                 [63.831,  0.1226,  0.0054], 
                 [64.678,  0.1186,  0.0051], 
                 [64.816,  0.1152,  0.0052]]) 

# Body setup
# ----------
# Build the Star body
mesh = phoebe.Star(sirius,mesh1, pbdep=ifdep1, position=pos)

# Computation of observables
# --------------------------
# Set the time in the universe
mesh.set_time(0)
proj_intens1 = mesh.projected_intensity()

f1,b1,pa1,s1,p1,x1,prof1 = phoebe.ifm(mesh)

# Analysis of results
# -------------------
# Output some extra observables, such as equatorial velocity and such.
parameters = mesh.get_parameters()
print(parameters)
radi = np.sqrt((mesh.mesh['center'][:,0]**2+\
                mesh.mesh['center'][:,1]**2+\
                mesh.mesh['center'][:,2]**2))
velo = np.sqrt((mesh.mesh['velo___bol_'][:,0]**2+\
                mesh.mesh['velo___bol_'][:,1]**2+\
                mesh.mesh['velo___bol_'][:,2]**2))
print("\nRadii {} {}".format(radi.min(),radi.max()))
print("v_eq = %.3f km"%(velo.max()*8.04986111111111))
print('g_pole = %.3f [cm/s2]'%(np.log10(parameters['g_pole'])+2))
print('T_pole - T_eq = %.3f - %.3fK\n\n'%(mesh.mesh['teff'].max(),mesh.mesh['teff'].min()))

# Make a check image
phoebe.image(mesh,savefig='sirius_image.png')

"""
.. figure:: images_tut/sirius_image.png
   :scale: 50 %
   :align: center
   :alt: map to buried treasure

   Image of Sirius

"""

# And plot the visibilities

pl.figure(figsize=(11,6))
ax1 = pl.subplot(131)
pl.plot(b1,s1**2,'-',lw=2)
pl.errorbar(data[:,0],data[:,1],yerr=data[:,2],fmt='ko')
pl.xlabel('Projected baseline [m]')
pl.ylabel('Squared Visibility')
pl.xlim(0,80)
pl.ylim(0,1)
pl.grid()
pl.subplot(132,sharex=ax1)
pl.plot(b1,p1/np.pi*180,'o-')
pl.xlabel('meter')
pl.ylabel('Phase [deg]')
pl.ylim(-360,360)
pl.xlim(0,80)
pl.grid()
pl.subplot(133)
x1 = x1*1000.
pl.plot(x1,prof1/prof1.max(),lw=2)
pl.xlim(x1[0],x1[-1])
pl.ylim(0,1)
pl.grid()
pl.xlabel('Angular scale [mas]')
pl.ylabel('Normalised intensity')
pl.savefig('sirius_visibility')


"""
.. figure:: images_tut/sirius_visibility.png
   :scale: 90 %
   :align: center
   :alt: map to buried treasure

   Visibility, phases and profile of Sirius.

"""
