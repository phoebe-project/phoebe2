import phoebe
from phoebe import u
from phoebe.atmospheres import models
from astropy.io import fits
from scipy import interpolate
import numpy as np
import os


path = os.path.dirname(__file__)
atms = [atm for atm in models.get_supported_atms(include_extern=False)]


def test_compute(atm_tables_available=False, atm_path=None, atms=atms):
    pb = phoebe.atmospheres.passbands.Passband(
        ptf=os.path.join(path, 'test.ptf'),
        pbset='test',
        pbname='t1',
        wlunits=u.nm,
        calibrated=True,
        reference='test passband creation, loading and saving',
        version=1.0
    )

    # generate blackbody tables:
    pb.compute_intensities(atm=models.BlackbodyModelAtmosphere(), include_mus=False, include_ld=False, verbose=True)

    if atm_tables_available:
        # generate tables:
        for atm in atms:
            if atm.name == 'blackbody':
                continue

            instance = atm.from_path(atm_path + '/' + atm.name)
            pb.compute_intensities(atm=instance, include_extinction=True, verbose=True)

    pb.save(os.path.join(path, 'test.fits'))


def test_load():
    pb = phoebe.atmospheres.passbands.Passband.load(os.path.join(path, 'test.fits'))
    assert 'blackbody:Inorm' in pb.content

    # cleanup
    os.remove(os.path.join(path, 'test.fits'))


def test_export_to_pre25(tmp_path):
    pb = phoebe.atmospheres.passbands.Passband(
        ptf=os.path.join(path, 'test.ptf'),
        pbset='test',
        pbname='t1',
        wlunits=u.nm,
        calibrated=True,
        reference='test passband export to pre-2.5 format',
        version=1.0
    )
    pb.compute_intensities(atm=models.BlackbodyModelAtmosphere(), include_mus=False, include_ld=False)

    archive = os.path.join(tmp_path, 'test_pre25.fits')
    pb.save(archive, export_to_pre25=True)

    with fits.open(archive) as hdul:
        assert hdul['primary'].header['phoebevn'] == '2.4.17'
        bb_func = hdul['bb_func'].data

    # pre-2.5 phoebe evaluates blackbody intensities as splev(teff, (t, c, 3)):
    teffs = pb.ndp['blackbody'].axes[0]
    for col, weighting in [('logi_e', 'energy'), ('logi_p', 'photon')]:
        log10ints = interpolate.splev(teffs, (bb_func['teff'], bb_func[col], 3))
        assert np.allclose(log10ints, pb.ndp['blackbody'].table[f'inorm@{weighting}']['grid'][:, 0])


def test_skip_pre25(tmp_path, capsys):
    pb = phoebe.atmospheres.passbands.Passband(
        ptf=os.path.join(path, 'test.ptf'),
        pbset='test',
        pbname='pre25',
        wlunits=u.nm,
        calibrated=True,
        reference='test skipping pre-2.5 passbands on initialization',
        version=1.0
    )
    pb.compute_intensities(atm=models.BlackbodyModelAtmosphere(), include_mus=False, include_ld=False)

    archive = os.path.join(tmp_path, 'test_pre25.fits')
    pb.save(archive, export_to_pre25=True)

    # pre-2.5 files must be skipped rather than registered (and later fail to load):
    phoebe.atmospheres.passbands._init_passband(archive)
    assert 'test:pre25' not in phoebe.atmospheres.passbands._pbtable
    assert "phoebe.download_passband('test:pre25')" in capsys.readouterr().out


if __name__ == '__main__':
    atm_path = os.path.join(path, 'tables')
    atm_tables_available = os.path.exists(atm_path)
    test_compute(atm_tables_available=atm_tables_available, atm_path=atm_path)
    test_load()
