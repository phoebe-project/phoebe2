"""
  Testing marching triangulation algorithm directly from libphoebe

"""

import numpy as np
from numpy import linalg as la
import libphoebe
import matplotlib.pyplot as plt

def potential_roche(r, q, F, d):
  r1 = np.array([r[0]- d, r[1], r[2]]);
  return 1/la.norm(r) + q*(1/la.norm(r1) - r[0]/d**2) + F**2*(1+q)*(r[0]**2 + r[1]**2)/2

def test_roche_detached(plot=False):
  
  #print "roche detached"
  
  q = 1
  F = 1
  d = 1
  Omega0 = 10
  choice = 0
   
  # determine area and volume 
  r_av = libphoebe.roche_area_volume(q, F, d, Omega0, choice, larea=True, lvolume=True)
  
  #print r_av
  
  # calculate delta
  ntriangles=1000
  delta = np.sqrt(r_av["larea"]/(np.sqrt(3)*ntriangles/4))
  
  max_triangles = int(1.5*ntriangles)
  
  # generate mesh  
  r_mesh = libphoebe.roche_marching_mesh(q, F, d, Omega0, delta, choice, max_triangles, 
                            vertices=True, vnormals=True, triangles=True,
                            tnormals=True, areas=True, area=True, volume=True, full=True)  
  
  # check the number of triagles of the mesh
  #print len(r_mesh["triangles"])
  assert(0.75*ntriangles < len(r_mesh["triangles"]) < 1.35*ntriangles)
  
  # check is the vertices are really on isosurface
  #print np.max(np.abs([potential_roche(r, q, F, d) - Omega0 for r in r_mesh["vertices"]]))
  assert(np.max(np.abs([potential_roche(r, q, F, d) - Omega0 for r in r_mesh["vertices"]])) < 1e-12)
  
  # check the area -0.00551862428183
  #print (r_mesh["area"]-r_av["larea"])/r_av["larea"]
  assert (np.abs(r_mesh["area"]-r_av["larea"])/r_av["larea"] < 1e-2)
  
  # check the volume -0.0100753493075
  #print (r_mesh["volume"]- r_av["lvolume"])/r_av["lvolume"]
  assert (np.abs(r_mesh["volume"]- r_av["lvolume"])/r_av["lvolume"] < 2e-2)

def test_roche_contact(plot=False):
  #print "roche contact"
  
  q = 0.5
  F = 0.5
  d = 1
  Omega0 = 2.65
  choice = 2
  
  # determine area and volume 
  r_av = libphoebe.roche_area_volume(q, F, d, Omega0, choice, larea=True, lvolume=True)
  
  #print r_av
  
  # calculate delta
  ntriangles=2000
  delta = np.sqrt(r_av["larea"]/(np.sqrt(3)*ntriangles/4))
  
  max_triangles = int(1.5*ntriangles)
  
  # generate mesh  
  r_mesh = libphoebe.roche_marching_mesh(q, F, d, Omega0, delta, choice, max_triangles, 
                            vertices=True, vnormals=True, triangles=True,
                            tnormals=True, areas=True, area=True, volume=True, full=True)  
  
  # check the number of triagles of the mesh
  #print len(r_mesh["triangles"])
  assert(0.75*ntriangles < len(r_mesh["triangles"]) < 1.35*ntriangles)
  
  # check is the vertices are really on isosurface
  #print np.max(np.abs([potential_roche(r, q, F, d) - Omega0 for r in r_mesh["vertices"]]))
  assert(np.max(np.abs([potential_roche(r, q, F, d) - Omega0 for r in r_mesh["vertices"]])) < 1e-12)
  
  # check the area
  #print (r_mesh["area"]-r_av["larea"])/r_av["larea"]
  assert (np.abs(r_mesh["area"]-r_av["larea"])/r_av["larea"] < 1e-2)
  
  # check the volume
  #print (r_mesh["volume"]- r_av["lvolume"])/r_av["lvolume"]
  assert (np.abs(r_mesh["volume"]- r_av["lvolume"])/r_av["lvolume"] < 1e-2)

def test_roche_semidetached(plot=False):
  
  #print "roche semi-detached"
  
  q = 1
  F = 1
  d = 1
  
  r_crit = libphoebe.roche_critical_potential(q, F, d, L1 = True, L2 = False, L3 = False)
  
  Omega0 = r_crit["L1"]
  choice = 0
   
  # determine area and volume 
  r_av = libphoebe.roche_area_volume(q, F, d, Omega0, choice, larea=True, lvolume=True)

  #print r_av
    
  # calculate delta
  ntriangles=1000
  delta = np.sqrt(r_av["larea"]/(np.sqrt(3)*ntriangles/4))
  
  max_triangles = int(1.5*ntriangles)
  
  # generate mesh  
  r_mesh = libphoebe.roche_marching_mesh(q, F, d, Omega0, delta, choice, max_triangles, 
                            vertices=True, vnormals=True, triangles=True,
                            tnormals=True, areas=True, area=True, volume=True, full=True)  
  
  # check the number of triagles of the mesh
  #print len(r_mesh["triangles"])
  assert(0.75*ntriangles < len(r_mesh["triangles"]) < 1.35*ntriangles)
  
  # check is the vertices are really on isosurface
  #print np.max(np.abs([potential_roche(r, q, F, d) - Omega0 for r in r_mesh["vertices"]]))
  assert(np.max(np.abs([potential_roche(r, q, F, d) - Omega0 for r in r_mesh["vertices"]])) < 1e-12)
  
  # check the area -0.00524575234831
  #print (r_mesh["area"]-r_av["larea"])/r_av["larea"]
  assert (np.abs(r_mesh["area"]-r_av["larea"])/r_av["larea"] < 1e-2)
  
  # check the volume -0.00985875277254
  #print (r_mesh["volume"]- r_av["lvolume"])/r_av["lvolume"]
  assert (np.abs(r_mesh["volume"]- r_av["lvolume"])/r_av["lvolume"] < 2e-2)

  
if __name__ == '__main__':
  test_roche_detached(plot=True)
  test_roche_contact(plot=True)
  test_roche_semidetached(plot=True)
