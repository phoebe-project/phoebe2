#  Testing the C++ routine wrapped in lib_phoebe_roche  
#
#  Install for memory analyzer:
#   apt-get install python-meliae
#   apt-get install runsnakerun
#
#  Run:
#   python phoebe_roche.py > /dev/null
#   runsnakemem big.dump
#
#  Tutorial:
#    http://jam-bazaar.blogspot.com/2009/11/memory-debugging-with-meliae.html

#  Author: Martin Horvat, May 2016

from lib_phoebe_roche import *
from meliae import scanner, loader
import gc
import resource

# Garbage collector
collected = gc.collect()
print "Garbage collector: collected %d objects." % (collected)

# Memory use 
print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# Preparing scanning of memory use and dumping in a json file 
scanner.dump_all_objects('big.dump')

q = 1
F = 1
delta = 1

for i in range(1, 400):
    
    q  = 0.01*i
      
    for j in range(1, 400):
      
      F  = 0.005*j;
      
      omega = critical_potential(q, F, delta)

      print omega

print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

collected = gc.collect()
print "Garbage collector: collected %d objects." % (collected)

# In a separate session do
#om = loader.load('big.dump')
#print om.summarize()
