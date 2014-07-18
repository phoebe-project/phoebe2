import phoebe
import nose.tools
import numpy as np
import matplotlib.pyplot as plt

def test_01():
    """
    Test online tutorial 01
    """
    eb = phoebe.Bundle('UWLMi.phoebe')

    print(eb.summary())
    print(eb.tree())
    print(eb)
    print(eb.get_ps('position'))
    print(eb['position'])
    print(eb.get_object('primary').summary())
    print(eb['primary'].summary())
    print(eb['primary'].tree())
    print(eb['primary'])
    ps = eb['primary']
    print(eb['orbit'])
    print(eb.get_parameter('incl@orbit'))
    assert(eb.get_value('incl@orbit')==80.0)
    assert(eb.get_value('incl')==80.0)
    assert(eb['incl']==80.0)  
    assert(eb.get_value('teff@primary')==eb.get_value('teff@secondary')==6000)
    assert(eb['teff@primary']==eb['teff@secondary']==6000)
    eb.set_value('teff@primary', 10000)
    eb['teff@primary'] = 10000
    eb.set_value('teff@primary', 12000, 'Far')
    eb['teff@primary'] = (12000, 'Far')
    assert(np.allclose(eb['teff@primary'],6922.03888889))
    print(eb.info('gravblaw@primary'))
    eb['gravblaw@primary'] = 'zeipel'
    print(eb.info('atm@component@primary'))
    print(eb.get_ps_dict('compute'))
    print(eb['compute'])
    print(eb['compute']['detailed'])
    for ps in eb['compute'].values():
        print(ps)
    eb['heating@legacy'] = False
    eb.run_compute('preview')
    print(eb['lcobs@UW_LMi'])
    print(eb['lcobs@UW_LMi']['APT_B'])
    print(eb['lcobs@UW_LMi']['APT_V'])
    print(eb['lcobs@UW_LMi']['APT_Ic'])
    print(eb.twigs('APT'))
    print(eb.search('APT'))
    print(eb['APT_V@lcdep@primary'])
    print(eb['APT_V@lcdep@secondary'])
    print(eb['APT_V@lcobs'])
    
    plt.figure()
    plt.plot(eb['time@APT_V@lcobs'], eb['flux@APT_V@lcobs'], 'b.')
    plt.plot(eb['time@APT_V@lcsyn'], eb['flux@APT_V@lcsyn'], 'r-')
    
    plt.figure()
    eb.plot_obs('APT_V', fmt='b.')
    eb.plot_syn('APT_V', fmt='r-')
    
    plt.figure()
    eb.plot_obs('APT_V', phased=True, fmt='b.')
    eb.plot_syn('APT_V', phased=True, fmt='r-')
    
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close('all')
    assert(True)
    
    
@nose.tools.raises(KeyError)        
def test_02():
    """
    Test online tutorial 01 (error 01)
    """
    eb = phoebe.Bundle('UWLMi.phoebe')
    ps = eb.get_ps('primary')
    
@nose.tools.raises(KeyError)            
def test_03():
    """
    Test online tutorial 01 (error 02)
    """
    eb = phoebe.Bundle('UWLMi.phoebe')
    print(eb['teff'])

if __name__ == "__main__":
    test_01()