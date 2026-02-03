import phoebe


def test_abun_loghefrac_parameters():
    b = phoebe.default_binary()
    b.add_dataset('lc', compute_phases=phoebe.linspace(0, 1, 11))

    # by default, both stars are ck2004 and therefore have abun but not loghefrac
    assert b.get_value(qualifier='atm', component='primary') == 'ck2004'
    assert b.get_value(qualifier='atm', component='secondary') == 'ck2004'
    assert len(b.filter(qualifier='abun')) == 2
    assert len(b.filter(qualifier='loghefrac')) == 0
    assert len(b.run_checks_compute()) == 0

    # changing one to tmap_sdO should hide abun and show loghefrac
    b.set_value(qualifier='atm', component='primary', value='tmap_sdO')
    assert len(b.filter(qualifier='abun', component='primary')) == 0
    assert len(b.filter(qualifier='loghefrac', component='primary')) == 1
    # but the default value for loghefrac (0.0) is out of bounds for tmap_sdO
    assert len(b.run_checks_compute()) == 1
    assert not b.run_checks_compute().passed
    # changing value within bounds should still pass
    b.set_value(qualifier='loghefrac', component='primary', value=-1.2)
    assert len(b.run_checks_compute()) == 0
    # revert value for next test
    b.set_value(qualifier='loghefrac', component='primary', value=0.0)

    # same for tmap_DO which uses a fixed value (9.4) for loghefrac
    b.set_value(qualifier='atm', component='primary', value='tmap_DO')
    assert len(b.filter(qualifier='abun', component='primary')) == 0
    assert len(b.filter(qualifier='loghefrac', component='primary')) == 1
    assert len(b.run_checks_compute()) == 1
    assert not b.run_checks_compute().passed
    b.set_value(qualifier='loghefrac', component='primary', value=9.4)
    assert len(b.run_checks_compute()) == 0
    # revert value for next test
    b.set_value(qualifier='loghefrac', component='primary', value=0.0)

    # blackbody uses a fixed value (0.0) for abun
    b.set_value(qualifier='atm', component='primary', value='blackbody')
    b.set_value(qualifier='ld_mode', component='primary', value='manual')
    assert len(b.filter(qualifier='abun', component='primary')) == 1
    assert len(b.filter(qualifier='loghefrac', component='primary')) == 0
    assert len(b.run_checks_compute()) == 0
    b.set_value(qualifier='abun', component='primary', value=0.2)
    assert len(b.run_checks_compute()) == 1
    assert not b.run_checks_compute().passed
    # revert value for next test
    b.set_value(qualifier='abun', component='primary', value=0.0)

    # setting abun to non-zero and then switching to an atmosphere that does not
    # use abun should result in a warning
    b.set_value(qualifier='abun', component='primary', value=0.3)
    b.set_value(qualifier='atm', component='primary', value='tmap_sdO')
    b.set_value(qualifier='loghefrac', component='primary', value=-1.0)
    assert len(b.filter(qualifier='abun', component='primary')) == 0
    assert len(b.run_checks_compute()) == 1
    assert b.run_checks_compute().passed


if __name__ == '__main__':
    test_abun_loghefrac_parameters()