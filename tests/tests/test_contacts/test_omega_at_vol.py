"""
  Testing routine value of the potential at partial volume
"""

import numpy as np
import libphoebe

def test_contact_omega_at_vol(plot=False):
  q = 0.1
  d = 1.
  Omega0 = 1.9
  phi = np.pi/2

  neck = libphoebe.roche_contact_neck_min(phi, q, d, Omega0)

  x = neck["xmin"]       # where we cut it
  choice = 0             # 0 for left and 1 for right

  pvol = libphoebe.roche_contact_partial_area_volume(x, q, d, Omega0, choice=choice) 

  vol= pvol['lvolume']

  Omega1 = libphoebe.roche_contact_Omega_at_partial_vol(vol, phi, q, d, choice=choice)

  assert(abs(Omega0 - Omega1) < 1e-12*Omega0) 

  if plot:
    print("neck=",neck)
    print("pvol=", pvol)
    print("Omega0={}, Omega1={}".format(Omega0, Omega1))

if __name__ == '__main__':

  test_contact_omega_at_vol(plot=True)

