import phoebe




def test_boosting(verbose=False):
    b=phoebe.default_binary()
    b.add_dataset('lc', times=0.25)
    b.run_compute(irrad_method='none')
    noboost = b.get_value(qualifier='fluxes', kind='lc', context='model')
    b.set_value_all('boosting_method','manual')
    b.set_value(qualifier='boosting_index', component='primary', value=2)
    b.run_compute(irrad_method='none')
    boost = b.get_value(qualifier='fluxes', kind='lc', context='model')
    
    assert boost > noboost



if __name__ == '__main__':
    test_boosting()

