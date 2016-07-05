#  Testing the C++ routines for volume conservation wrapped in 
#  libphoebe
#
#  Author: Martin Horvat, June 2016

import numpy as np
import time

from math import cos, sin, pi
from libphoebe import *

# case
omega=1
Omega0=3

#
# Calculate the area and volume of the lobe
# 

start = time.time()
av = rotstar_area_volume(omega, Omega0, larea=True, lvolume=True)
end = time.time()

print "roche_area_volume, time[ms]=", 1000*(end-start)
print "orig. lobe:area=%.16e, volume=%.16e" % (av["larea"], av["lvolume"])

#
# "Volume conservation"
#
vol = av["lvolume"]
omega1 = 1.1*omega
Omega1 = rotstar_Omega_at_vol(vol, omega1, Omega0=Omega0)

print "correction:Omega0=%.16e,Omega1=%.16e" % (Omega0, Omega1)

av1 = rotstar_area_volume(omega1, Omega1, larea=True, lvolume=True)
print "corr. lobe:area=%.16e, volume=%.16e" % (av1["larea"], av1["lvolume"])


