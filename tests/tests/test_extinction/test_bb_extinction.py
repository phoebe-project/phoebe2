import phoebe
from phoebe.atmospheres import passbands
import libphoebe as lp
from phoebe import u
import numpy as np
# numpy 2.0+ renamed trapz to trapezoid
try:
    from numpy import trapezoid as np_trapz
except ImportError:
    from numpy import trapz as np_trapz
import os


def test_bb_extinction_computation():
    pb = passbands.Passband(
        ptf=np.vstack((np.linspace(500., 600., 101), np.ones(101))).T,
        pbset='box',
        pbname='test',
        wlunits=u.nm,
        calibrated=True
    )

    atm = phoebe.atmospheres.models.BlackbodyModelAtmosphere()
    pb.compute_intensities(atm=atm, include_mus=False, include_ld=False, include_extinction=True, verbose=False)
    assert 'blackbody:ext' in pb.content

    pb.save('box_test.fits')


def test_bb_extinction():
    pb = phoebe.atmospheres.passbands.Passband.load('box_test.fits')
    if 'blackbody:ext' not in pb.content:
        raise ValueError('blackbody extinction tables not found in the passband.')

    # E(B-V) on a regular grid, halfway between the table nodes (where the interpolation error peaks)
    # and small values (where the relative error peaks); R_V = 3.1 is not a table node either:
    ebv_nodes = pb.ndp['blackbody'].table['ext@energy']['associated_axes'][0]
    ebvs = np.concatenate((np.linspace(0, 1, 11), 0.5*(ebv_nodes[:10] + ebv_nodes[1:11]), [0.001, 0.01, 0.02]))
    rvs = 3.1 * np.ones_like(ebvs)

    axbx = lp.gordon_extinction(pb.wl.reshape(-1, 1))  # (101, 2)
    ax, bx = axbx[:, 0, None], axbx[:, 1, None]
    Alam = 10**(-0.4 * ebvs * (rvs * ax + bx))  # (101, len(ebvs))

    atm = phoebe.atmospheres.models.BlackbodyModelAtmosphere()
    for teff in (5772., 15000.):
        for intens_weighting in ('energy', 'photon'):
            bb_sed = pb._planck(pb.wl, teff)[:, None]  # (101,1)
            if intens_weighting == 'photon':
                bb_sed = bb_sed * pb.wl[:, None]
            iext_predicted = np_trapz(bb_sed * Alam, axis=0) / np_trapz(bb_sed, axis=0)

            query_cols = ['teffs', 'ebvs', 'rvs']
            query_pts = np.vstack((teff * np.ones_like(ebvs), ebvs, rvs)).T
            query = passbands.InterpQuery(cols=query_cols, pts=query_pts)

            iext = pb.interpolate_extinct(
                query=query,
                atm=atm,
                intens_weighting=intens_weighting,
                extrapolation_method='none'
            ).get_interpolated_values().flatten()

            # interpolated minus exact passband extinction, in magnitudes; the tables are interpolated
            # in log10 of the extinction factors, which is good to ~0.05 mmag here (interpolating the
            # factors themselves underestimates the extinction by up to ~12 mmag between nodes):
            dA = 2.5 * np.log10(iext_predicted / iext)
            assert np.all(np.abs(dA) < 2e-4), f'teff={teff}, {intens_weighting}: max |dA| = {np.abs(dA).max():.2e} mag'


def test_frontend():
    pb = passbands.Passband.load('box_test.fits')
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

    # extinction is gray across each (blackbody) star, so the flux ratio is the passband extinction
    # factor at E(B-V) = 0.3, R_V = 3.1:
    assert b['value@teff@primary@component'] == b['value@teff@secondary@component']
    teff = b['value@teff@primary@component']
    bb_sed = pb._planck(pb.wl, teff)
    if b.get_value(qualifier='intens_weighting', dataset='lc01') == 'photon':
        bb_sed = bb_sed * pb.wl
    ax, bx = lp.gordon_extinction(pb.wl.reshape(-1, 1)).T
    expected = np_trapz(bb_sed * 10**(-0.4 * 0.3 * (3.1 * ax + bx))) / np_trapz(bb_sed)

    ratio = b['value@fluxes@lc_ext']/b['value@fluxes@lc_no_ext']
    assert np.allclose(ratio, expected, rtol=2e-4, atol=0), f'flux ratio {ratio.mean():.6f} vs {expected:.6f}'

    phoebe.uninstall_passband('box:test')
    os.remove('box_test.fits')


if __name__ == '__main__':
    test_bb_extinction_computation()
    test_bb_extinction()
    test_frontend()
