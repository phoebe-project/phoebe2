#  Testing the C++ routines for volume conservation wrapped in 
#  libphoebe
#
#  Author: Martin Horvat, June 2016

import numpy as np
import time

from math import cos, sin, pi
from libphoebe import *

# overcontact case
#q = 0.5
#F = 0.5
#d = 1
#Omega0 = 2.65
#delta = 0.01
#choice = 0


# detacted case
#q = 1
#F = 1
#d = 1
#Omega0 = 10
#delta = 0.01
#choice = 0

# detacted case
q = 1
F = 1
d=1.0000000000000002
Omega0 = 8.992277876713667
delta = 0.01
choice = 0


#
# Calculate the area and volume of the lobe
# 

start = time.time()
av = roche_area_volume(q, F, d, Omega0, choice, larea=True, lvolume=True)
end = time.time()

print "roche_area_volume, time[ms]=", 1000*(end-start)
print "orig. lobe:area=%.16e, volume=%.16e" % (av["larea"], av["lvolume"])

#
# "Volume conservation"
#

vol = av["lvolume"]
d1 = 1.1*d
Omega1 = roche_Omega_at_vol(vol, q, F, d1, Omega0=Omega0)

print "correction:Omega0=%.16e,Omega1=%.16e" % (Omega0, Omega1)

av1 = roche_area_volume(q, F, d1, Omega1, choice, larea=True, lvolume=True)
print "corr. lobe:area=%.16e, volume=%.16e" % (av1["larea"], av1["lvolume"])


