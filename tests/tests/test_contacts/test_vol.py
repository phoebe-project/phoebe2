"""
  Testing computing partial volume of the contact binary
"""
import libphoebe

def test_contact_vol(plot=False):

  x=0.7       # where we cut it
  choice = 0  # 0 for left and 1 for right
  q=0.1
  Omega0=1.9
  d=1.

  res = libphoebe.roche_contact_partial_area_volume(x, q, d, Omega0, choice)

  larea0 = 4.587028506379938
  lvolume0 = 0.9331872042603445

  assert(abs(res['larea'] - larea0) < 1e-10*larea0)
  assert(abs(res['lvolume']- lvolume0) < 1e-10*lvolume0)

  if plot:
    print ("results={}".format(res))


if __name__ == '__main__':

  test_contact_vol(plot=True)
