#
# The Sun-Earth system
#

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

import phoebe
from phoebe import u, c
import libphoebe


BLACKBODY = True


def initiate_sun_earth_system(pb_str):

    b = phoebe.Bundle.default_binary()

    b.add_dataset('lc', times=[0.75,], dataset='lc01', passband=pb_str)
    b.add_dataset('mesh', times=[0.0], columns=['areas', 'mus', 'visibilities', 'abs_intensities@lc01'], dataset='mesh01')

    b['pblum@primary'] = 1.*u.solLum #* 0.99 # 0.99 is bolometric correction
    b['teff@primary'] = 1.*u.solTeff
    b['requiv@primary'] = 1.*u.solRad
    b['syncpar@primary'] = 14.61

    b['period@orbit'] = 1.*u.yr
    b['q@orbit'] = 3.986004e14/1.3271244e20   # (GM)_E / (GM)_Sun
    b['sma@orbit'] = 1.*u.au

    b['teff@secondary'] = (300, 'K')
    b['requiv@secondary'] = 1.*c.R_earth
    b['syncpar@secondary'] = 365.25

    b['distance@system'] = (1, 'au')

    b.set_value_all('irrad_method', 'none')

    if BLACKBODY:
        b.set_value_all('atm', value='blackbody')
        b.set_value_all('ld_mode', 'manual')
        b.set_value_all('ld_func', value='linear')
        b.set_value_all('ld_coeffs', value=[0.0])
    else:
        b.set_value_all('atm', component='secondary', value='blackbody')
        b.set_value_all('ld_func', component='primary', value='interp')
        b.set_value_all('ld_mode', component='secondary', value='manual')
        b.set_value_all('ld_func', component='secondary', value='linear')
        b.set_value_all('ld_coeffs', component='secondary', value=[0.0])

    return b

def integrated_flux(b, pb):

  r = b['value@abs_intensities@primary']
  r *= b['areas@primary@mesh01'].get_value(unit=u.m**2)
  r *= b['value@mus@primary@mesh01']
  r *= b['value@visibilities@primary@mesh01']

  return np.nansum(r)*pb.ptf_area/b['value@distance@system']**2


def _planck(lam, Teff):
    return 2*c.h.si.value*c.c.si.value*c.c.si.value/lam**5 * 1./(np.exp(c.h.si.value*c.c.si.value/lam/c.k_B.si.value/Teff)-1)


def sun_earth_result():

  pb_str = 'Bolometric:900-40000'
  mypb = phoebe.atmospheres.passbands.get_passband(pb_str)

  # theoretical result: Planck formula + passband, no limb-darkening
  sedptf = lambda w: _planck(w, c.T_sun.si.value)*mypb.ptf(w)
  sb_flux = np.pi*integrate.quad(sedptf, mypb.ptf_table['wl'][0], mypb.ptf_table['wl'][-1])[0] # Stefan-Boltzmann flux

  # fixed point observer
  #~ xi = ((1*u.solRad).si.value/c.au.si.value)**2
  #~ iflux0 = sb_flux*xi*2/(1 + np.sqrt(1 - xi))

  # fixed direction of observation
  #~ xi = (1*u.solRad).si.value/c.au.si.value
  #~ iflux0 = sb_flux*(xi**2)*(1 + 4*xi/3 + xi**2)

  # naive fixed direction of observation
  xi = (1*u.solRad).si.value/c.au.si.value
  iflux0 = sb_flux*(xi**2)

  # phoebe result for different mesh sizes
  b = initiate_sun_earth_system(pb_str)

  res=[]
  for Nt in [5000, 10000, 20000]:
    b['ntriangles@primary'] = Nt
    b['ntriangles@secondary'] = Nt

    # we're not actually computing light curves so don't care about
    # the failing check that the earth is smaller than triangles on
    # the sun
    b.run_compute(skip_checks=True, eclipse_method='only_horizon')

    q = b['value@q@orbit']
    F = b['value@syncpar@primary']
    spin = np.array([0.,0.,1.])
    req = b['value@requiv@primary']/b['value@sma@orbit']
    V = 4*np.pi*req**3/3

    Omega0 = libphoebe.roche_misaligned_Omega_at_vol(V, q, F, 1., spin)

    area0 = libphoebe.roche_misaligned_area_volume(q, F, 1., spin, Omega0, larea=True)['larea']
    area0 *= b['value@sma@orbit']**2

    area = np.sum(b['value@areas@primary@mesh01'])
    iflux = integrated_flux(b, mypb)

    res.append([Nt, area/area0-1, iflux/iflux0-1])

  return np.array(res)


def test_sun_earth(print_results = False, save_results = False):
  res = sun_earth_result()

  if print_results:
    print(res)

  if save_results:
    np.savetxt("res.txt", res)

  assert(np.abs(res[:,1]).max() < 1e-14)
  assert(np.abs(res[:,2]).max() < 1e-3)

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')

    res = test_sun_earth(print_results = True, save_results = True)
