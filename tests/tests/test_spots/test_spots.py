"""
"""

import phoebe
import numpy as np
import os


def test_binary(plot=False, gen_comp=False):
    b = phoebe.Bundle.default_binary()

    b.add_spot(component='primary', relteff=0.9, radius=20, colat=45, long=90, feature='spot01')

    b.add_dataset('lc', times=np.linspace(0, 1, 26))
    b.add_dataset('mesh', times=[0], columns=['teffs'])
    b.add_compute('phoebe', compute='phoebe2')
    if gen_comp:
        b.add_compute('legacy', compute='phoebe1')

    # set matching atmospheres
    b.set_value_all('atm', 'extern_planckint')

    # set matching limb-darkening, both bolometric and passband
    b.set_value_all('ld_mode_bol', 'manual')
    b.set_value_all('ld_func_bol', 'logarithmic')
    b.set_value_all('ld_coeffs_bol', [0.0, 0.0])

    b.set_value_all('ld_mode', 'manual')
    b.set_value_all('ld_func', 'logarithmic')
    b.set_value_all('ld_coeffs', [0.0, 0.0])

    # turn off albedos (legacy requirement)
    b.set_value_all('irrad_frac_refl_bol',  0.0)

    print("running phoebe2 model...")
    b.run_compute(compute='phoebe2', irrad_method='none', model='phoebe2model')
    if gen_comp:
        print("running phoebe1 model...")
        b.run_compute(compute='phoebe1', refl_num=0, model='phoebe1model')
        b.filter(model='phoebe1model').save('test_spots.comp.model')
    else:
        b.import_model(os.path.join(os.path.dirname(__file__), 'test_spots.comp.model'), model='phoebe1model')

    phoebe2_val = b.get_value('fluxes@phoebe2model')
    phoebe1_val = b.get_value('fluxes@phoebe1model')

    if plot:
        print("rel: ", ((phoebe2_val-phoebe1_val)/phoebe2_val).max())
        print("abs: ", (phoebe2_val-phoebe1_val).max())

        # b.plot(dataset='mesh01', show=True)

        b.plot(dataset='lc01', legend=True, show=True)

    assert np.allclose(phoebe2_val, phoebe1_val, rtol=2e-3, atol=5e-4)


def test_single_distortions(distortion_method='sphere', plot=False, gen_comp=False):
    '''
    Test single star with a spot using different distortion methods (sphere and rotstar)
    '''
    b = phoebe.default_star()
    b.add_spot(radius=30, colat=80, long=0, relteff=0.9)
    times = np.linspace(0, 1, 101)
    b.add_dataset('lc', times=times)

    if gen_comp:
        b.run_compute(distortion_method=distortion_method, irrad_method='none', model='comp_model')
        b.filter(model='comp_model').save('test_spots_single_{}.comp.model'.format(distortion_method))
        return
    
    else:
        b.run_compute(distortion_method=distortion_method, irrad_method='none', model='test_model')

        b.import_model(os.path.join(os.path.dirname(__file__), 'test_spots_single_{}.comp.model'.format(distortion_method)), model='comp_model', overwrite=True)
        if plot:
            b.plot(show=True)
            
        comp_val = b.get_value('fluxes@comp_model')
        test_val = b.get_value('fluxes@test_model')

        assert np.allclose(comp_val, test_val, rtol=2e-2, atol=5e-3)


def test_binary_distortions(distortion_method='sphere', plot=False, gen_comp=False):
    '''
    Test binary star with a spot using different distortion methods (sphere, rotstar and roche)
    '''

    b = phoebe.default_binary()

    b.flip_constraint('mass@primary', 'sma@binary')
    b['mass@primary'].set_value(value=1.0)
    b['q'].set_value(value = 0.0001)
    b['distortion_method@primary'].set_value(value=distortion_method)
    b['distortion_method@secondary'].set_value(value='none')

    b.add_spot(component='primary', radius=30, colat=80, long=0, relteff=0.9)

    b.set_value('period@binary', 1000)

    b.flip_constraint('period@primary', 'syncpar@primary')
    b.set_value('period@primary', 10)

    times = np.linspace(0, 10, 101)
    b.add_dataset('lc', times=times)

    if gen_comp:
        b.run_compute(irrad_method='none', model='comp_model')
        b.filter(model='comp_model').save('test_spots_binary_{}.comp.model'.format(distortion_method))
        return
    
    else:
        b.run_compute(irrad_method='none', model='test_model')

        b.import_model(os.path.join(os.path.dirname(__file__), 'test_spots_binary_{}.comp.model'.format(distortion_method)), model='comp_model', overwrite=True)
        if plot:
            b.plot(show=True)
            
        comp_val = b.get_value('fluxes@comp_model')
        test_val = b.get_value('fluxes@test_model')

        assert np.allclose(comp_val, test_val, rtol=2e-2, atol=5e-3)


def test_binary_misalignment(case='case1', plot=False, gen_comp=False):
    '''
    Test binary star with a spot using different misalignment cases
    case1 - pinclination = 50
    '''
    b = phoebe.default_binary()

    b.flip_constraint('mass@primary', 'sma@binary')
    b['mass@primary'].set_value(value=1.0)
    b['q'].set_value(value = 0.0001)
    b['distortion_method@primary'].set_value(value='roche')
    b['distortion_method@secondary'].set_value(value='none')

    b.add_spot(component='primary', radius=30, colat=80, long=0, relteff=0.9)

    b.set_value('period@binary', 1000)

    b.flip_constraint('period@primary', 'syncpar@primary')
    b.set_value('period@primary', 10)

    times = np.linspace(0, 10, 101)
    b.add_dataset('lc', times=times)

    if case == 'case1':
        b.set_value(qualifier='pitch', component='primary', value=-40)
    elif case == 'case2':
        b.set_value(qualifier='pitch', component='primary', value=-120)

    if gen_comp:
        b.run_compute(irrad_method='none', model='comp_model')
        b.filter(model='comp_model').save('test_spots_misaligned_{}.comp.model'.format(case))
        return
    
    else:
        b.run_compute(irrad_method='none', model='test_model')

        b.import_model(os.path.join(os.path.dirname(__file__), 'test_spots_misaligned_{}.comp.model'.format(case)), model='comp_model', overwrite=True)
        if plot:
            b.plot(show=True)
            
        comp_val = b.get_value('fluxes@comp_model')
        test_val = b.get_value('fluxes@test_model')

        assert np.allclose(comp_val, test_val, rtol=2e-2, atol=5e-3)




if __name__ == '__main__':
    logger = phoebe.logger(clevel='DEBUG')
    test_binary(plot=True, gen_comp=True)

    test_single_distortions(distortion_method='sphere', plot=True, gen_comp=True)
    test_single_distortions(distortion_method='rotstar', plot=True, gen_comp=True)

    test_binary_distortions(distortion_method='sphere', plot=True, gen_comp=True)
    test_binary_distortions(distortion_method='rotstar', plot=True, gen_comp=True)
    test_binary_distortions(distortion_method='roche', plot=True, gen_comp=True)

    test_binary_misalignment(case='case1', plot=True, gen_comp=True)
    test_binary_misalignment(case='case2', plot=True, gen_comp=True)
