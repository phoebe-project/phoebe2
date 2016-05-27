"""
"""

import phoebe2
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import phoebe
import phoebeBackend as phb

def _beta_vs_legacy(b,frametitle,imagename,ltte=False,plot=False):
	
    period = b.get_value('period@orbit')
    times = np.linspace(-0.2,1.2*period,200)
    
    b.add_dataset('lc', time=times, dataset='lc01', atm='blackbody', ld_func='logarithmic', ld_coeffs = [0.5,0.5])
    b.add_dataset('rv', time=times, dataset='rv01', atm='blackbody', ld_func='logarithmic', ld_coeffs = [0.5,0.5])
    
    b.add_compute('phoebe', compute='phnum', ltte=ltte, rv_method='flux-weighted')
    b.add_compute('legacy', compute='legnum', ltte=ltte, rv_method='flux-weighted')
    
    b.run_compute('phnum', model='phnumresults')
    b.run_compute('legnum', model='legnumresults')
    
    rv1_ph2 = b['rv@primary@phnumresults@phnum'].value
    rv2_ph2 = b['rv@secondary@phnumresults@phnum'].value
    
    rv1_leg = b['rv@primary@legnumresults@legnum'].value
    rv2_leg = b['rv@secondary@legnumresults@legnum'].value
    
    if plot:
        # plot rvs
        fig = plt.figure()
        ax1 = plt.subplot2grid((4,1),(0,0), rowspan = 2)
        ax2 = plt.subplot2grid((4,1),(2,0), rowspan = 2)
        #ax3 = plt.subplot2grid((4,1),(3,0), rowspan = 1)
        
        
        ax1.plot(b['time@primary@phnumresults@phnum'].value, b['rv@primary@phnumresults@phnum'].value, 'g-', label='beta')
        ax1.plot(b['time@primary@legnumresults@legnum'].value, b['rv@primary@legnumresults@legnum'].value, 'b--', label='legacy')
    
        ax1.plot(b['time@secondary@legnumresults@legnum'].value, b['rv@secondary@legnumresults@legnum'].value, 'b--')
        ax1.plot(b['time@secondary@phnumresults@phnum'].value, b['rv@secondary@phnumresults@phnum'].value, 'g-')
    
        ax2.plot(times,b['rv@primary@phnumresults@phnum'].value-b['rv@primary@legnumresults@legnum'].value, 'k.', label = 'rv1 beta-legacy')
        #ax3.plot(times,b['rv@secondary@phnumresults@phnum'].value-b['rv@secondary@legnumresults@legnum'].value, 'k.', label = 'rv2 beta-legacy')
    
        ax1.set_xlim(-0.2,1.2*period)
        ax2.set_xlim(-0.2,1.2*period)
        #ax3.set_xlim(-0.2,1.2*period)
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        #ax3.legend(loc='best')
        
        fig.suptitle(frametitle)
        fig.savefig(imagename)
        plt.close()
    
    return times, rv1_ph2, rv2_ph2, rv1_leg, rv2_leg


def test_binary():
    
    #system = [sma (solRad), period (d)]
    logger = phoebe2.utils.get_basic_logger()
    
    system1 = [8., 3.]
    system2 = [215., 257.5] 
    system3 = [8600., 65000.]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    system = system1
    q = 1.
    ecc = 0.01
    
    b = phoebe2.Bundle.default_binary()

    for twig in b['potential@constraint'].twigs:
        b.flip_constraint(twig, 'rpole')
        
    b.set_value('sma@binary',system[0])
    b.set_value('period@binary', system[1])
    b.set_value('q', q)
    b.set_value('ecc',ecc)
    
    imagename = 'q_'+str(q)+'_sma_'+str(system[0])+'_period_'+str(system[1])+'_ecc_'+str(ecc)+'.png'
    frametitle = 'q = '+str(q)+', sma = '+str(system[0])+', period = '+str(system[1])+', ecc = '+str(ecc)
    
    times, rv1_ph2, rv2_ph2, rv1_leg, rv2_leg = _beta_vs_legacy(b,frametitle,imagename,ltte=False,plot=False)
    
    return times, rv1_ph2, rv2_ph2, rv1_leg, rv2_leg

if __name__ == '__main__':     
    logger = phoebe2.utils.get_basic_logger()
    times, rv1_ph2, rv2_ph2, rv1_leg, rv2_leg = test_binary()
    
    print ""
    if np.isnan(rv1_ph2).any() or np.isnan(rv2_ph2).any():
        print "RVS CONTAIN NANS! CALCULATIONS INCORRECT!"
    else:
        print "CALCULATIONS CORRECT!"

