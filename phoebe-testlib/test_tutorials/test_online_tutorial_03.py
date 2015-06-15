import phoebe
import nose.tools
import os
import numpy as np
import matplotlib.pyplot as plt

basedir = os.path.dirname(os.path.abspath(__file__))

def test_01():
    """
    Test online tutorial 03 (fromarrays)
    """
    eb = phoebe.Bundle()
    eb['distance'] = 10, 'pc'
    
    ucoord = np.zeros(10)
    vcoord = np.linspace(0.1,200,10)
    time = np.linspace(0, 0.5, 10)
    phase = np.linspace(0, 1, 10)
    
    eb.if_fromarrays(phase=phase, ucoord=ucoord, vcoord=vcoord)
    print(eb.summary())
    print(eb)
    print(eb['if01@primary'])
    print(eb.info('bandwidth_smearing@if01@primary'))
    print(eb.info('bandwidth_subdiv@if01@primary'))
    
    eb['bandwidth_smearing@if01@ifdep@primary'] = 'power'
    eb['bandwidth_smearing@if01@ifdep@secondary'] = 'power'
    
    eb.run_compute()
    
    print(eb['vis2@if01@ifsyn'])
    b = np.sqrt(eb['ucoord@if01@ifsyn']**2 +eb['vcoord@if01@ifsyn']**2)
    
    plt.figure()
    plt.plot(b, eb['vis2@if01@ifsyn'], 'ko-')
    
    plt.figure()
    plt.plot(eb['time@if01@ifsyn'], eb['vis2@if01@ifsyn'], 'ko-')
    
    
    plt.figure()
    eb.plot_syn('if01', fmt='bo-')
    
    plt.figure()
    eb.old_plot_syn('if01', 'ro-')
    
    #~ plt.figure()
    #~ eb.plot_syn('if01', 'ro-', xquantity='time')  # REMOVE THIS FROM TUTORIAL - WILL FIX LATER
    
    if __name__ == "__main__":
        plt.show()
        
    assert(True)


def test_02():
    """
    Test online tutorial 03 (OIFITS)
    """
    filename = os.path.join(basedir, 'example_oifits.fits')
    
    eb = phoebe.Bundle()
    eb['distance'] = 1, 'kpc'
    eb['period'] = 1e8, 'd'
    eb['t0'] = 56848.97824354 + 0.25*eb['period']
    eb['sma'] = 5, 'au'
    eb['incl'] = 0, 'deg'
    eb['pot@primary'] = 21.
    eb['pot@secondary'] = 21.
    eb['teff@primary'] = 6000.0
    eb['teff@secondary'] = 6000.0
    eb.set_value_all('atm', 'blackbody')
    eb.set_value_all('ld_func', 'uniform')
    
    eb.lc_fromarrays(phase=[0.25])
    
    #~ eb.plot_mesh(phase=0.25, dataref='lc01', label='preview')  # FIXING THIS SOON
    
    eb.if_fromfile(filename, include_closure_phase=True, passband='OPEN.BOL')
    
    eb.plot_mesh(phase=0.25, select='proj', dataref='if01', label='preview')    
    
    plt.figure()
    eb.plot_obs('if01', fmt='o', x_quantity='ucoord', y_quantity='vcoord')
    
    eb.run_compute('preview')
    
    print(eb.get_logp())
    
    plt.figure()
    eb.plot_obs('if01', fmt='ko')
    eb.plot_syn('if01', 'rx', ms=10, mew=2)
    
    plt.figure()
    eb.plot_obs('if01', fmt='ko', x_quantity='time')
    eb.plot_syn('if01', 'rx', ms=10, mew=2, x_quantity='time')
    
    plt.figure()
    eb.plot_obs('if01', fmt='ko', x_quantity='time', y_quantity='closure_phase')
    eb.plot_syn('if01', 'rx', ms=10, mew=2, x_quantity='time', y_quantity='closure_phase')
    
    
    print(eb.get_logp())
    
    observed = eb['vis2@if01@ifobs']
    simulated = eb['vis2@if01@ifsyn']
    error = eb['sigma_vis2@if01@ifobs']
    print( np.nansum((observed-simulated)**2 / error**2))
    
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close('all')
    
    assert(True)
    

if __name__ == "__main__":
    logger = phoebe.get_basic_logger(clevel='INFO')
    test_01()
    test_02()
