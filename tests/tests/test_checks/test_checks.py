"""
"""

import phoebe

phoebe.logger('DEBUG')


def test_checks():
    b = phoebe.Bundle.default_binary()
    b.add_dataset('lc')

    # test overflow
    report = b.run_checks()
    assert report.passed

    b.set_value('requiv', component='primary', value=9.0)
    report = b.run_checks()
    assert not report.passed

    b.set_value('requiv', component='primary', value=1.0)

    # TODO: test overlap scenario

    # test ld_func vs ld_coeffs
    report = b.run_checks()
    assert report.passed

    b.set_value_all('ld_mode_bol', 'manual')
    b.set_value('ld_coeffs_bol', component='primary', value=[0.])
    report = b.run_checks()
    assert not report.passed

    b.set_value('ld_coeffs_bol', component='primary', value=[0.5, 0.5])
    b.set_value('ld_mode', component='primary', value='manual')
    b.set_value('ld_func', component='primary', value='logarithmic')
    b.set_value('ld_coeffs', component='primary', value=[0.])
    report = b.run_checks()
    assert not report.passed

    b.set_value('ld_coeffs', component='primary', value=[0., 0.])
    b.set_value('ld_mode', component='primary', value='interp')

    # test ld_func vs atm
    report = b.run_checks()
    assert report.passed

    b.set_value('atm', component='primary', value='blackbody')
    report = b.run_checks()
    assert not report.passed
    b.set_value('atm', component='primary', value='ck2004')

    # test gravb vs teff warning
    b.set_value('teff', component='primary', value=6000)
    b.set_value('gravb_bol', component='primary', value=1.0)
    report = b.run_checks()
    assert report.passed and len(report.items) > 0


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    test_checks()
