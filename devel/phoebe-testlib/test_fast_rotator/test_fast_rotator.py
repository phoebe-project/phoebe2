import numpy as np
import matplotlib.pyplot as plt
import phoebe

logger = phoebe.get_basic_logger()

def test_fast_rotator(from_main=False):
    """
    Spectrum test: compare numerical computation to analytical one
    """
    star = phoebe.ParameterSet(context='star',add_constraints=True)
    star['atm'] = 'blackbody'
    star['ld_func'] = 'linear'
    star['ld_coeffs'] = [0.6]
    star['rotperiod'] = 0.24,'d'
    star['shape'] = 'sphere'
    star['teff'] = 10000.
    star['radius'] = 1.0,'Rsol'

    mesh = phoebe.ParameterSet(context='mesh:marching',alg='c')
    mesh['delta'] = 0.05

    spdep1 = phoebe.ParameterSet(context='spdep')
    spdep1['ld_func'] = 'linear'
    spdep1['atm'] = 'blackbody'
    spdep1['ld_coeffs'] = [0.6]
    spdep1['passband'] = 'JOHNSON.V'
    spdep1['method'] = 'numerical'
    spdep1['ref'] = 'Numerical'

    spdep2 = spdep1.copy()
    spdep2['method'] = 'analytical'
    spdep2['ref'] = 'Via convolution'

    wavelengths = np.linspace(399.7, 400.3, 1000)
    spobs1 = phoebe.ParameterSet(context='spobs', ref=spdep1['ref'], wavelength=wavelengths)
    spobs2 = phoebe.ParameterSet(context='spobs', ref=spdep2['ref'], wavelength=wavelengths)

    mesh1 = phoebe.Star(star, mesh, pbdep=[spdep1, spdep2])

    mesh1.set_time(0)

    mesh1.sp(obs=spobs1)
    mesh1.sp(obs=spobs2)

    result1 = mesh1.get_synthetic(category='sp',ref=0)
    result2 = mesh1.get_synthetic(category='sp',ref=1)

    flux1 = np.array(result1['flux'][0])/np.array(result1['continuum'][0])
    flux2 = np.array(result2['flux'][0])/np.array(result2['continuum'][0])

    if not from_main:
        assert (np.all(np.abs((flux1-flux2)/flux1)<=0.00061))
    else:
	plt.plot(result1['wavelength'][0], flux1,'k-')
	plt.plot(result2['wavelength'][0], flux2,'r-')


if __name__=='__main__':
    test_fast_rotator(from_main=True)
    plt.show()
