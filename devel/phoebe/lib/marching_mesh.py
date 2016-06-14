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

#  Author: Martin Horvat, June 2016

import gc
import resource
import numpy as np
import time

from math import cos, sin, pi
from lib_phoebe_roche import *
from meliae import scanner, loader


# Garbage collector
collected = gc.collect()
print "Garbage collector: collected %d objects." % (collected)

# Memory use 
print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# Preparing scanning of memory use and dumping in a json file 
scanner.dump_all_objects('big.dump')

# overcontact case
q = 0.5
F = 0.5
d = 1
Omega0 = 2.65
delta = 0.01
choice = 0
max_triangles = 10000000 # 10^7

#
# Calculate triangulated surface
#  V - vertices  -- 2-rank numpy array 
#   Nvertice x 3D coordinate == (x,y,z)]
#  T - triangles -- 2- rank numpy arrayconnectivity matrix
#   Ntriangles x 3 integer indices

start = time.time()
res = marching_mesh(q, F, d, Omega0, delta, choice, max_triangles, vertices=True, vnormals=True, triangles=True, tnormals=True, areas=True)
end = time.time()

V = res["vertices"]
NatV = res["vnormals"];
T = res["triangles"]
NatT = res["tnormals"];
A = res["areas"]

print "marching_mesh, time[ms]=", 1000*(end-start)
print "marching_mesh, V.size= %d, T.size=%d" % (V.shape[0], T.shape[0])


#
# Calculate mesh visibility aka mask M
# M - mask - 1-rank numpu array of reals of length Ntriangles 
#

theta = 20./180*pi 
v = np.array([cos(theta),0,sin(theta)])

start = time.time()
M = mesh_visibility(v, V, T, NatT)
end = time.time()

print "triangle_mesh_visibility, time[ms]=", 1000*(end-start)

np.savetxt('py_meshingV.txt', V, fmt='%d')
np.savetxt('py_meshingNatV.txt', NatV, fmt='%d')
np.savetxt('py_meshingT.txt', T, fmt='%d')
np.savetxt('py_meshingN.txt', NatT, fmt='%.16e')
np.savetxt('py_meshingA.txt', A, fmt='%.16e')
np.savetxt('py_meshingM.txt', M, fmt='%.16e')
  
print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

collected = gc.collect()
print "Garbage collector: collected %d objects." % (collected)

# In a separate session do
#om = loader.load('big.dump')
#print om.summarize()
