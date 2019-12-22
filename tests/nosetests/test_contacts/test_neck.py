"""
  Calculating the neck of the contact binary in xy and xy direction
"""
import numpy as np
import libphoebe

def test_contact_neck(plot=False):
  q=0.1
  d=1.
  Omega0 = 1.9

  neck_xy = libphoebe.roche_contact_neck_min(0., q, d, Omega0)
  neck_xz = libphoebe.roche_contact_neck_min(np.pi/2., q, d, Omega0)

  neck_xy0 = {'xmin': 0.742892957853368, 'rmin': 0.14601804638933566}
  neck_xz0 = {'xmin': 0.7383492639142092, 'rmin': 0.13255145166593718}

  assert(abs(neck_xy['xmin'] - neck_xy0['xmin']) < 1e-12*neck_xy0['xmin'])
  assert(abs(neck_xz['xmin'] - neck_xz0['xmin']) < 1e-12*neck_xz0['xmin'])

  if plot:
    print("neck_xy={}".format(neck_xy))
    print("neck_xz={}".format(neck_xz))

if __name__ == '__main__':

    test_contact_neck(plot=True)
