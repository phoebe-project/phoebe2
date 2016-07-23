#  Testing the C++ routines for area and volume calcuation using
#  series approximation and integration wrapped in 
#  libphoebe
#
#  Author: Martin Horvat, July 2016

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
q = 1.
F = 1.
d = 1.
Omega0 = 10
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
