#  Testing pov-ray export  wrapped in libphoebe  
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
#choice = 2
#max_triangles = 10000000 # 10^7

q = 1
F = 1
d = 1
Omega0 = 10
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
for i in xrange(100):
  res = roche_marching_mesh(q, F, d, Omega0, delta, choice, max_triangles, vertices=True, vnormals=True, triangles=True, tnormals=True, areas=True, area=True, volume=True)
end = time.time()

V = res["vertices"]
NatV = res["vnormals"]
T = res["triangles"]
NatT = res["tnormals"]
A = res["areas"]

print "marching_mesh, time[ms]=", 1000*(end-start)
print "marching_mesh, V.size= %d, T.size=%d, N.size=%d" % (V.shape[0], T.shape[0], NatT.shape[0])


camera_location = np.array([0,0.5,0.5], dtype=float);
camera_look_at = np.array([0,0,0], dtype=float);
light_source = np.array([100,100,100], dtype=float);

#povray +R2 +A0.1 +J1.2 +Am2 +Q9 +H480 +W640 scene.pov
mesh_export_povray("scene.pov", V, NatV, T, camera_location, camera_look_at, light_source, plane_enable=True)
