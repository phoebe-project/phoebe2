import phoebe


def test_abun_loghefrac_parameters():
    b = phoebe.default_binary()
    b.add_dataset('lc', compute_phases=phoebe.linspace(0, 1, 11))

    # by default, both stars are ck2004 and therefore have abun but not loghefrac
    assert b.get_value(qualifier='atm', component='primary') == 'ck2004'
    assert b.get_value(qualifier='atm', component='secondary') == 'ck2004'
    assert len(b.filter(qualifier='abun')) == 2
    assert len(b.filter(qualifier='loghefrac')) == 0

    # changing one to tmap_sdO should hide abun and show loghefrac
    b.set_value(qualifier='atm', component='primary', value='tmap_sdO')
    assert len(b.filter(qualifier='abun', component='primary')) == 0
    assert len(b.filter(qualifier='loghefrac', component='primary')) == 1

    # same for tmap_DO which uses a fixed value for loghefrac
    b.set_value(qualifier='atm', component='primary', value='tmap_DO')
    assert len(b.filter(qualifier='abun', component='primary')) == 0
    assert len(b.filter(qualifier='loghefrac', component='primary')) == 1

    # blackbody uses a fixed value for abun
    b.set_value(qualifier='atm', component='primary', value='blackbody')
    assert len(b.filter(qualifier='abun', component='primary')) == 1
    assert len(b.filter(qualifier='loghefrac', component='primary')) == 0


if __name__ == '__main__':
    test_abun_loghefrac_parameters()