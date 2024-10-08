import phoebe
import libphoebe as lp
from phoebe import u
import numpy as np
import os


def test_bb_extinction_computation():
    pb = phoebe.atmospheres.passbands.Passband(
        ptf=np.vstack((np.linspace(500., 600., 101), np.ones(101))).T,
        pbset='box',
        pbname='test',
        wlunits=u.nm,
        calibrated=True
    )

    pb.compute_blackbody_intensities(include_extinction=True)
    assert 'blackbody:ext' in pb.content

    pb.save('box_test.fits')


def test_bb_extinction():
    pb = phoebe.atmospheres.passbands.Passband.load('box_test.fits')
    if 'blackbody:ext' not in pb.content:
        raise ValueError('blackbody extinction tables not found in the passband.')

    ebvs = np.linspace(0, 1, 11)
    rvs = 3.1 * np.ones_like(ebvs)
    teffs = 5772 * np.ones_like(ebvs)

    bb_sed = pb._planck(pb.wl, teffs[0])[:, None]  # (101,1)

    axbx = lp.gordon_extinction(pb.wl.reshape(-1, 1))  # (101, 2)
    ax, bx = axbx[:, 0, None], axbx[:, 1, None]
    
    Alam = 10**(-0.4 * ebvs * (rvs * ax + bx))  # (101, 11)
    iext_predicted = np.trapz(bb_sed * Alam, axis=0) / np.trapz(bb_sed, axis=0)

    query_pts = np.vstack((teffs, ebvs, rvs)).T
    iext = pb.interpolate_extinct(query_pts=query_pts, atm='blackbody', intens_weighting='photon', extrapolation_method='none').flatten()

    assert np.allclose(iext, iext_predicted, atol=2e-3, rtol=2e-3)


def test_frontend():
    phoebe.install_passband('box_test.fits')
    
    b = phoebe.default_binary()
    b.add_dataset('lc', compute_times=phoebe.linspace(0, 1, 21), passband='box:test')
    b['atm@primary'] = 'blackbody'
    b['atm@secondary'] = 'blackbody'
    b['ld_mode@primary'] = 'manual'
    b['ld_mode@secondary'] = 'manual'
    b['ld_func@primary'] = 'linear'
    b['ld_func@secondary'] = 'linear'
    b['ld_coeffs@primary'] = [0.5]
    b['ld_coeffs@secondary'] = [0.5]

    b.run_compute(model='lc_no_ext', irrad_method='none')

    b.flip_constraint('ebv', solve_for='Av')
    b['Rv'] = 3.1
    b['ebv'] = 0.3
    b.run_compute(model='lc_ext', irrad_method='none')

    assert np.mean(b['value@fluxes@lc_ext']/b['value@fluxes@lc_no_ext'])-0.785 < 1e-3

    phoebe.uninstall_passband('box:test')
    os.remove('box_test.fits')


if __name__ == '__main__':
    test_bb_extinction_computation()
    test_bb_extinction()
    test_frontend()
