"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt
import os


def _jktebop_ext_vs_phoebe2(case, plot=False, gen_comp=False):

    b = phoebe.default_binary()
    b.add_compute(
        'jktebop',
        compute='jktebop1',
        irrad_method='none',
        rv_method='dynamical',
        distortion_method='sphere'
    )
    b.add_compute(
        compute='phoebe1',
        irrad_method='none',
        rv_method='dynamical',
        distortion_method='sphere'
    )

    if case == 1:
        jktebop_rv1 = np.loadtxt(os.path.join(os.path.dirname(__file__), 'jktebop1.rv1.out')).T
        jktebop_rv2 = np.loadtxt(os.path.join(os.path.dirname(__file__), 'jktebop1.rv2.out')).T
        atol = 1e-3

    elif case == 2:
        b['ecc@binary@component'] = 0.1
        b['per0@binary@component'] = 212
        b['sma@binary@component'] = 124
        b['period@binary@component'] = 120
        b['incl@binary@component'] = 88
        b['teff@primary@component'] = 5000
        b['teff@secondary@component'] = 5000
        b['requiv@primary@component'] = 1
        b['requiv@secondary@component'] = 3
        b['q@binary@component'] = 0.5

        jktebop_rv1 = np.loadtxt(os.path.join(os.path.dirname(__file__), 'jktebop2.rv1.out')).T
        jktebop_rv2 = np.loadtxt(os.path.join(os.path.dirname(__file__), 'jktebop2.rv2.out')).T
        atol = 3e-5


    elif case == 3:
        b['ecc@binary@component'] = 0.5
        b['sma@binary@component'] = 100
        b['period@binary@component'] = 20
        b['incl@binary@component'] = 50
        b['teff@primary@component'] = 5000
        b['teff@secondary@component'] = 5000
        b['requiv@primary@component'] = 7
        b['requiv@secondary@component'] = 7
        b['q@binary@component'] = 0.25

        jktebop_rv1 = np.loadtxt(os.path.join(os.path.dirname(__file__), 'jktebop3.rv1.out')).T
        jktebop_rv2 = np.loadtxt(os.path.join(os.path.dirname(__file__), 'jktebop3.rv2.out')).T
        atol = 2e-3


    times = np.linspace(0, b['period@binary@component'].get_value(), 100)
    rv = np.zeros_like(times)
    errs = np.ones_like(times)*0.001
    # b.add_dataset('rv', times={'primary': times}, rvs={'primary': rv}, sigmas={'primary': errs}, dataset='rv01')
    b.add_dataset('rv', times=times, dataset='rv01')


    b.set_value_all('ld_mode', value='manual')
    b.run_compute('jktebop1', model='jktebop_model1')
    b.run_compute('phoebe1', model='phoebe_model1')

    # if gen_comp:
    #     b.run_compute('legnum', model='legnumresults')
    #     b.filter(model='legnumresults').save('test_rvs_{}.comp.model'.format(ind))
    # else:
    #     b.import_model(os.path.join(os.path.dirname(__file__), 'test_rvs_{}.comp.model'.format(ind)), model='legnumresults', overwrite=True)

    if plot:
        fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]}, figsize=(6, 4))
        axs[0].plot(b['times@jktebop_model1@primary'].get_value(), b['rvs@jktebop_model1@primary'].get_value(), '+', c = 'k', label = 'jktebop backend primary')
        axs[0].plot(b['times@jktebop_model1@secondary'].get_value(), b['rvs@jktebop_model1@secondary'].get_value(), 'x', c = 'k', label = 'jktebop backend secondary')

        axs[0].plot(jktebop_rv1[0], jktebop_rv1[4], '-', c = 'r', label = 'jktebop ext. primary')
        axs[0].plot(jktebop_rv2[0], jktebop_rv2[4], '--', c = 'r', label = 'jktebop ext. secondary')
        axs[0].legend(loc='best')

        axs[0].set_ylabel('RV (km/s)')

        axs[1].plot(b['times@jktebop_model1@primary'].get_value(), b['rvs@jktebop_model1@primary'].get_value() - jktebop_rv1[4], 'o', c = 'b', label = 'jktebop primary')
        axs[1].plot(b['times@jktebop_model1@secondary'].get_value(), b['rvs@jktebop_model1@secondary'].get_value() - jktebop_rv2[4], 'o', c = 'g', label = 'jktebop primary')
        axs[1].set_ylabel('Residuals (km/s)')
        axs[0].set_ylabel('Time')

        plt.show()

    if plot:
        fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]}, figsize=(6, 4))
        axs[0].plot(b['times@phoebe_model1@primary'].get_value(), b['rvs@phoebe_model1@primary'].get_value(), '+', c = 'k', label = 'phoebe primary')
        axs[0].plot(b['times@phoebe_model1@secondary'].get_value(), b['rvs@phoebe_model1@secondary'].get_value(), 'x', c = 'k', label = 'phoebe secondary')

        axs[0].plot(jktebop_rv1[0], jktebop_rv1[4], '-', c = 'r', label = 'jktebop ext. primary')
        axs[0].plot(jktebop_rv2[0], jktebop_rv2[4], '--', c = 'r', label = 'jktebop ext. secondary')
        axs[0].legend(loc='best')

        axs[0].set_ylabel('RV (km/s)')

        axs[1].plot(b['times@phoebe_model1@primary'].get_value(), b['rvs@phoebe_model1@primary'].get_value() - jktebop_rv1[4], 'o', c = 'b', label = 'jktebop primary')
        axs[1].plot(b['times@phoebe_model1@secondary'].get_value(), b['rvs@phoebe_model1@secondary'].get_value() - jktebop_rv2[4], 'o', c = 'g', label = 'jktebop primary')
        axs[1].set_ylabel('Residuals (km/s)')
        axs[0].set_ylabel('Time')

        plt.show()


    assert(np.allclose(b['rvs@phoebe_model1@primary'].get_value(), jktebop_rv1[4], rtol=0., atol=atol))
    assert(np.allclose(b['rvs@phoebe_model1@secondary'].get_value(), jktebop_rv2[4], rtol=0., atol=atol))
    assert(np.allclose(b['rvs@jktebop_model1@primary'].get_value(), jktebop_rv1[4], rtol=0., atol=1e-8))
    assert(np.allclose(b['rvs@jktebop_model1@secondary'].get_value(), jktebop_rv2[4], rtol=0., atol=1e-8))


def test_jktebop(plot=False, gen_comp=False):

    for case in range(1,4):
        _jktebop_ext_vs_phoebe2(case, plot=plot, gen_comp=gen_comp)


if __name__ == '__main__':
    logger = phoebe.logger('debug')
    test_jktebop(plot=True, gen_comp=False)
