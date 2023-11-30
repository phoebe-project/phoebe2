import phoebe
import numpy as np
import os
import matplotlib.pyplot as plt


def _beta_vs_legacy(b, syncpar, plot=False, gen_comp=False):
    b.run_compute('phnum', model='phnumresults', overwrite=True)
    if gen_comp:
        b.run_compute('legnum', model='legnumresults', overwrite=True)
        b.filter(model='legnumresults').save('test_rm_{}.comp.model'.format(syncpar))
    else:
        b.import_model(os.path.join(os.path.dirname(__file__), 'test_rm_{}.comp.model'.format(syncpar)), model='legnumresults', overwrite=True)

    phoebe1_rv1 = b.get_value('rvs@primary@legnumresults@legnum')
    phoebe1_rv2 = b.get_value('rvs@secondary@legnumresults@legnum')

    phoebe2_rv1 = b.get_value('rvs@primary@phnumresults@phnum')
    phoebe2_rv2 = b.get_value('rvs@secondary@phnumresults@phnum')

    phoebe1_rv2[np.isnan(phoebe2_rv2)] = np.nan

    if plot:
        print("rv@primary max abs diff: {}".format(max(np.abs(phoebe1_rv1-phoebe2_rv1))))
        print("rv@secondary max abs diff: {}".format(max(np.abs(phoebe1_rv2-phoebe2_rv2))))
        plt.plot(np.abs(phoebe2_rv1-phoebe1_rv1))
        plt.plot(np.abs(phoebe2_rv2-phoebe1_rv2))
        plt.show()

    assert np.allclose(phoebe2_rv1, phoebe1_rv1, rtol=0., atol=2.0) and np.allclose(phoebe2_rv2, phoebe1_rv2, rtol=0., atol=2.0, equal_nan=True)


def test_binary(plot=False, gen_comp=False):

    b = phoebe.default_binary()

    # set equivalent radius of the secondary to be different from the primary to avoid
    # numerical artifacts during eclipse:
    b.set_value(qualifier='requiv', component='secondary', value=0.6)

    # set the mass ratio to non-unity:
    b.set_value(qualifier='q', value=0.75)

    period = b.get_value('period@orbit')
    times = np.linspace(-0.2, 1.2*period, 51)

    # turn off albedos (for comparison with legacy):
    b.set_value_all('irrad_frac_refl_bol',  0.0)

    b.add_dataset('rv', times=times, dataset='rv01', ld_mode='manual', ld_func='logarithmic', ld_coeffs=[0.5, 0.5])

    b.add_compute('phoebe', compute='phnum', ltte=False, atm='extern_planckint', rv_method='flux-weighted', irrad_method='none')
    b.add_compute('legacy', compute='legnum', ltte=False, atm='extern_planckint', rv_method='flux-weighted', refl_num=0)

    for syncpar in [1/4, 1, 4]:
        if plot:
            print("setting syncpar@primary to", syncpar)
        b.set_value('syncpar@primary', syncpar)
        _beta_vs_legacy(b, syncpar, plot=plot, gen_comp=gen_comp)

    return b


if __name__ == '__main__':
    logger = phoebe.logger()
    b = test_binary(plot=True, gen_comp=True)
