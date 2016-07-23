#  Testing the C++ routine wrapped for critical potentials in lib_phoebe_roche  
#
#  Author: Martin Horvat, June 2016

from libphoebe import *
#from meliae import scanner, loader
#import gc
#import resource

# Garbage collector
#collected = gc.collect()
#print "Garbage collector: collected %d objects." % (collected)

# Memory use 
#print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# Preparing scanning of memory use and dumping in a json file 
#scanner.dump_all_objects('big.dump')

#q = 1
#F = 1
#delta = 1

#for i in range(1, 400):
#    
#    q  = 0.01*i
#      
#    for j in range(1, 400):
#      
#      F  = 0.005*j;
#      
#      omega = critical_potential(q, F, delta)
#
#      print omega
#
#print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

#collected = gc.collect()
#print "Garbage collector: collected %d objects." % (collected)

# In a separate session do
#om = loader.load('big.dump')
#print om.summarize()

import numpy as np

q=3.00348934885e-6
F=365.25
d=1
omega = roche_critical_potential(q, F, d)

print omega

x=np.array([4.26352057697e-5,0,0]);
print "Omega at point=", roche_Omega(q,F,d,x)
