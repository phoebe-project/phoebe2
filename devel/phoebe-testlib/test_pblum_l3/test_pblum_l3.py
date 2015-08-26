import numpy as np
import matplotlib.pyplot as plt
import phoebe

#phoebe.get_basic_logger(clevel='INFO')


def test_pblum_l3():
    """
    Test scale/offset and pblum/l3 in lc.
    """
    x = phoebe.Bundle()

    x['period'] = 20.0
    x['incl'] = 90.0
    x['pot@primary'] = phoebe.compute_pot_from(x, 0.05, component=0)
    x['pot@secondary'] = phoebe.compute_pot_from(x, 0.025, component=1)
    x.set_value_all('ld_func', 'uniform')
    x.set_value_all('ld_coeffs', [0.0])
    x.set_value_all('atm', 'blackbody')

    x.lc_fromarrays(phase=np.linspace(0,1,100))
    x['pblum@lc01@secondary'] = 0.0
    x.run_compute()

    x['value@flux@lc01@lcobs'] = x['value@flux@lc01@lcsyn']


    x.lc_fromarrays(phase=np.linspace(0,1,100), flux=x['flux@lc01@lcsyn'], offset=10000)

    x.lc_fromarrays(phase=np.linspace(0,1,100), flux=x['flux@lc01@lcsyn']+10000)
    x.set_adjust('scale@lc03')
    x.set_adjust('offset@lc03')

    x.lc_fromarrays(phase=np.linspace(0,1,100), flux=0.75*x['flux@lc01@lcsyn']+10000)
    x.set_adjust('scale@lc04')
    x.set_adjust('offset@lc04')

    x.lc_fromarrays(phase=np.linspace(0,1,100), flux=x['flux@lc01@lcsyn'])
    x.lc_fromarrays(phase=np.linspace(0,1,100), flux=x['flux@lc01@lcsyn'])
    x.lc_fromarrays(phase=np.linspace(0,1,100), flux=x['flux@lc01@lcsyn'])

    x['pblum@lc02@secondary'] = 0.0
    x['pblum@lc03@secondary'] = 0.0
    x['pblum@lc04@secondary'] = 0.0
    x['pblum@lc05@secondary'] = 0.0
    x['pblum@lc06@secondary'] = 0.0
    x['pblum@lc07@secondary'] = 0.0

    x['pblum@lc05@primary'] = 4*np.pi
    x['pblum@lc06@primary'] = -1
    x['pblum@lc07@primary'] = 4*np.pi

    x['l3@lc05@primary'] = 0.0
    x['l3@lc06@primary'] = 10000.0
    x['l3@lc07@primary'] = 0.1

    x.run_compute()
    
    # Benchmark results
    results = dict()
    results['lc01'] = [0.748343497631, 1776.54192813, 2373.96587764]
    results['lc02'] = [0.951719282612, 11776.5419281, 12373.9658776]
    results['lc03'] = [0.951719282612, 11776.5419281, 12373.9658776]
    results['lc04'] = [0.961965202198, 11332.4064461, 11780.4744082]
    results['lc05'] = [0.748343497631, 0.748606948021, 1.00035204474]
    results['lc06'] = [0.951719282612, 11776.5419281, 12373.9658776]
    results['lc07'] = [0.771214041978, 0.848606948021, 1.10035204474]

    for nr in range(1,8):
        dataref = 'lc{:02d}'.format(nr)
        syn = x['{}@new_system@lcsyn'.format(dataref)]
        obs = x['{}@new_system@lcobs'.format(dataref)]
        rel_depth = syn['flux'].min() / syn['flux'].max()
        minim, maxim = syn['flux'].min() , syn['flux'].max()
        
        print('\n'+dataref)
        print('scale={}, offset={}'.format(obs['scale'], obs['offset']))
        print('pblum={}, l3={}'.format(x['pblum@'+dataref+'@primary'], x['l3@'+dataref+'@primary']))
        print("Relative eclipse depth: {}".format(rel_depth))
        print("Flux level: {} --> {}".format(minim, maxim))
        
        assert(np.abs(rel_depth-results[dataref][0])/rel_depth<0.005)
        assert(np.abs(minim-results[dataref][1])/minim<0.005)
        assert(np.abs(maxim-results[dataref][2])/maxim<0.005)
        
    
if __name__=="__main__":
    test_pblum_l3()
