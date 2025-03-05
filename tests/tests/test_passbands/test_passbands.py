import phoebe
from phoebe import u
from phoebe.atmospheres import models
import os


path = os.path.dirname(__file__)
atms = [atm for atm in models.ModelAtmosphere.__subclasses__() if not atm.external]


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


if __name__ == '__main__':
    atm_path = os.path.join(path, 'tables')
    atm_tables_available = os.path.exists(atm_path)
    test_compute(atm_tables_available=atm_tables_available, atm_path=atm_path)
    test_load()
