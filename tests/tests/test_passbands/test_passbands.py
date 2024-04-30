import phoebe
from phoebe import u
from phoebe.atmospheres.passbands import supported_atms
import os


path = os.path.dirname(__file__)


def test_compute(atm_grids_available=False, atm_path=None, atms=supported_atms.keys()):
    pb = phoebe.atmospheres.passbands.Passband(
        ptf=os.path.join(path, 'test.ptf'),
        pbset='test',
        pbname='t1',
        wlunits=u.nm,
        calibrated=True,
        reference='test passband creation, loading and saving',
        version=1.0
    )

    pb.compute_blackbody_intensities(include_extinction=True)

    if atm_grids_available:
        for atm in atms:
            pb.compute_intensities(atm=atm, path=f'{atm_path}/{atm}', verbose=True)
            pb.compute_ldcoeffs(ldatm=atm)
            pb.compute_ldints(ldatm=atm)

    pb.save(os.path.join(path, 'test.fits'))


def test_load():
    pb = phoebe.atmospheres.passbands.Passband.load(os.path.join(path, 'test.fits'))
    assert 'blackbody:Inorm' in pb.content

    # cleanup
    os.remove(os.path.join(path, 'test.fits'))


if __name__ == '__main__':
    atm_path = os.path.join(path, 'tables')
    test_compute(atm_grids_available=True, atm_path=atm_path)
    test_load()
