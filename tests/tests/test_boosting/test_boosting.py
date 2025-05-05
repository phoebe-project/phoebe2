import phoebe




def test_boosting(verbose=False):
    b=phoebe.default_binary()
    b.add_dataset('lc', times=0.25)
    b.run_compute()
    noboost=b['fluxes@lc@model'].value
    b.set_value_all('boosting_method','manual')
    b.set_value(qualifier='boosting_index', component='primary', value=2)
    b.run_compute()
    boost=b['fluxes@lc@model'].value
    
    assert boost > noboost



if __name__ == '__main__':
    test_boosting()

