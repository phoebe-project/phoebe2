#  Testing the C++ wrapper for solving radiosity problem on overcontact 
#  Roche lobes in libphoebe  
#
#  Author: Martin Horvat, July 2016


import numpy as np
import time

from math import cos, sin, pi
from libphoebe import *

# overcontact case
q = 0.5
F = 0.5
d = 1
Omega0 = 2.65
delta = 0.05
choice = 2
max_triangles = 10000000 # 10^7

#
# Calculate triangulated surface
#  V - vertices  -- 2-rank numpy array 
#   Nvertice x 3D coordinate == (x,y,z)]
#  T - triangles -- 2- rank numpy arrayconnectivity matrix
#   Ntriangles x 3 integer indices

start = time.time()
for i in xrange(100):
  res = roche_marching_mesh(q, F, d, Omega0, delta, choice, max_triangles, 
    vertices=True, vnormals=True, triangles=True, tnormals=True, 
    areas=True, area=True, volume=True)
end = time.time()

V = res["vertices"]
NatV = res["vnormals"]
Tr = res["triangles"]
NatT = res["tnormals"]
A = res["areas"]

print "marching_mesh, time[ms]=", 1000*(end-start)
print "marching_mesh, V.size= %d, Tr.size=%d, N.size=%d" % (V.shape[0], Tr.shape[0], NatT.shape[0])

np.savetxt('py_meshingV.txt', V, fmt='%.16e')
np.savetxt('py_meshingTr.txt', Tr, fmt='%d')
np.savetxt('py_meshingNatT.txt', NatT, fmt='%.16e')
np.savetxt('py_meshingA.txt', A, fmt='%.16e')

#
# Calculate the reflection effects
# 

start = time.time()

Nt = Tr.shape[0]

LDmod = [("linear", 0.3)]

LDidx = np.full(Nt, 0, dtype=np.int)
M0 = np.full(Nt, 1, dtype=np.float)
R = np.full(Nt, 0.75, dtype=np.float)

M = mesh_radiosity_Wilson_triangles(V, Tr, NatT, A, R, M0, LDmod, LDidx)

end = time.time()

print "mesh_radiosity_Wilson, time[ms]=", 1000*(end-start)

np.savetxt('py_meshingM.txt', M, fmt='%.16e')
