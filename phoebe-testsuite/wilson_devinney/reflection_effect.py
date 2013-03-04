"""
Reflection effect
=================

We start by loading a WD-type \*.active file, and change some of the parameters.
After that, we convert the WD input types to pyphoebe input types. WD needs
parameters for the binary, the light curves and radial velocity curves.
Pyphoebe needs parameters for the bodies separately, as well as a set of
binary parameters for each body. Also, we need light- and radial velocity
parameters for each body.

Next, we run pyWD with the new set of parameters, and also launch pyphoebe.
The output light curves and images are finally compared.

Initialisation
--------------

"""
# First, import necessary modules.
import time
import numpy as np
import matplotlib.pyplot as plt
import phoebe
from phoebe import wd


logger = phoebe.get_basic_logger()
c0 = time.time()

# Parameter preparation
# ---------------------
#-- Read in a WD \*.active file
ps,lc,rv = wd.lcin_to_ps('test01lcin.active',version='wd2003')

#-- Change some parameters
ps['model'] = 'unconstrained'
ps['omega'] = np.pi/2.
ps['dperdt'] = 0
ps['dpdt'] = 0.
ps['ecc'] = 0.0
ps['f1'] = 1.
ps['f2'] = 1.
ps['pot1'] = 4.75
ps['pot2'] = 4.75
ps['ifat1'] = 0
ps['ifat2'] = 0
ps['alb1'] = 1.
ps['alb2'] = 1.
ps['nref'] = 1
ps['mref'] = 2
ps['ipb'] = 0
ps['incl'] = 75.,'deg'
ps['n1'] = 60
ps['n2'] = 60
ps['teff2'] = 30000.,'K'
lc['phnorm'] = 0.0
lc['jdstrt'] = ps['hjd0']
lc['jdend'] = ps['hjd0']+ps['period']
lc['jdinc'] = 0.01*ps['period']
lc['indep_type'] = 'time (hjd)'
lc['el3'] = 0.
rv['vunit'] = 1.

# Convert to pyphoebe parameters
comp1,comp2,binary = wd.wd_to_phoebe(ps,lc,rv)
star1,lcdep1,rvdep1 = comp1
star2,lcdep2,rvdep2 = comp2
star1['irradiator'] = False

# Set the fineness of the mesh manually, there is no conversion possible between
# a WD-type and pyphoebe-type mesh.
mesh1 = phoebe.ParameterSet(frame='phoebe',context='mesh:marching')
mesh2 = phoebe.ParameterSet(frame='phoebe',context='mesh:marching')
mesh1['delta'] = 0.25
mesh2['delta'] = 0.25

# Body setup
#-----------
star1 = phoebe.BinaryRocheStar(star1,binary,mesh1,pbdep=[lcdep1,rvdep1])
star2 = phoebe.BinaryRocheStar(star2,binary,mesh2,pbdep=[lcdep2,rvdep2])

print star1
print star2
c1 = time.time()

# Computation of observables
# --------------------------
# Compute WD light curve
curve,params = wd.lc(ps,request='curve',light_curve=lc,rv_curve=rv)
image,params = wd.lc(ps,request='image',light_curve=lc,rv_curve=rv)

ps['alb1'] = 0.
ps['alb2'] = 0.
curve2,params2 = wd.lc(ps,request='curve',light_curve=lc,rv_curve=rv)
image2,params2 = wd.lc(ps,request='image',light_curve=lc,rv_curve=rv)

c2 = time.time()
# Compute pyphoebe light curve and radial velocity curves
P = binary['period']
times = np.linspace(binary['t0'],binary['t0']+P,len(curve['indeps']))
phoebe.observe(phoebe.BodyBag([star1,star2]),times,\
         subdiv_num=2,lc=True,rv=True,heating=True)

c3 = time.time()         

# Analysis of results
# -------------------
flux1 = np.array(star1.params['syn']['lcsyn'].values()[0]['flux'])
flux2 = np.array(star2.params['syn']['lcsyn'].values()[0]['flux'])
flux = flux1+flux2
rv1 = np.array(star1.params['syn']['rvsyn'].values()[0]['rv'])
rv2 = np.array(star2.params['syn']['rvsyn'].values()[0]['rv'])
rv1 = phoebe.convert('Rsol/d','km/s',rv1)
rv2 = phoebe.convert('Rsol/d','km/s',rv2)

plt.figure(figsize=(9,7))
plt.axes([0.1,0.3,0.85,0.65])
plt.plot(times,flux/flux.mean(),'ko-',label='pyphoebe')
plt.plot(curve['indeps'],curve['lc']/curve['lc'].mean(),'ro-',lw=2,label='WD')
plt.plot(curve2['indeps'],curve2['lc']/curve2['lc'].mean(),'bo-',lw=2,label='WD (no refl)')
leg = plt.legend(loc='best',fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.ylabel('Normalized flux')
plt.axes([0.1,0.1,0.85,0.20])
plt.plot(curve['indeps'],curve['lc']/curve['lc'].mean()-flux/flux.mean(),'ko-')
plt.xlabel('Phase')
plt.savefig('03_lc_comparison')

#

plt.figure(figsize=(9,7))
plt.axes([0.1,0.3,0.85,0.65])
plt.plot(times,rv1,'ko-',lw=2,label='pyphoebe(a)')
plt.plot(times,rv2,'ko--',lw=2,label='pyphoebe(b)')
plt.plot(curve['indeps'],curve['rv1'],'ro-',lw=2,label='WD(a)')
plt.plot(curve['indeps'],curve['rv2'],'ro--',lw=2,label='WD(b)')
leg = plt.legend(loc='best',fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.ylabel('Radial velocity')
plt.axes([0.1,0.1,0.85,0.20])
plt.plot(curve['indeps'],rv1-curve['rv1'],'ko-')
plt.plot(curve['indeps'],rv2-curve['rv2'],'ko--')
plt.xlabel('Phase')
plt.savefig('03_rv_comparison')

"""

+----------------------------------------------+--------------------------------------------+
| Light curve comparison                       | Radial velocity comparison                 |
+----------------------------------------------+--------------------------------------------+
| .. image:: images_tut/03_lc_comparison.png   | .. image:: images_tut/03_rv_comparison.png |
|    :height: 233px                            |    :height: 233px                          |
|    :width: 233px                             |    :width: 233px                           |
+----------------------------------------------+--------------------------------------------+

"""


c4 = time.time()

print "Finished!"
print "Initialisation:             %10.3g sec"%(c1-c0)
print "Wilson-Devinney:            %10.3g sec"%(c2-c1)
print "Pyphoebe:                   %10.3g min"%((c3-c2)/60.)
print "Analysis:                   %10.3g sec"%(c4-c3)
print "-----------------------------------------"
print "Total time:                 %10.3g min"%((c4-c0)/60.)