import phoebe
import nose.tools
import numpy as np
import matplotlib.pyplot as plt

def test_01():
    """
    Test online tutorial 02
    """
    eb = phoebe.Bundle()
    print(eb.summary())
    print(eb['position'])
    print(eb['primary'])
    print(eb.tree())
    eb['teff@primary'] = 5500.0
    eb.set_value('period', 10)
    eb['teff@primary'] = 5500, 'K'
    eb.set_value('period', 10,'d')
    eb.lc_fromarrays(phase=np.linspace(-0.6, 0.6, 201))
    print(eb.summary())
    
    print(eb['lc01@lcobs'])
    print(eb['lc01@lcsyn'])
    print(eb['lc01@primary'])
    print(eb['lc01@secondary'])
    eb['flux@lc01@lcobs'] = np.random.normal(1.0, 0.1, 201)
    print(eb['compute'])
    print(eb['detailed@compute'])
    eb.run_compute('detailed')
    
    print(eb['lc01@lcsyn'])
    
    plt.figure()
    eb.plot_syn('lc01', fmt='r-')
    
    plt.figure()
    plt.plot(eb['time@lcsyn'], eb['flux@lcsyn'], 'r-')
    
    
    print(eb['lc01@lcsyn'])
    eb.clear_syn()
    print(eb['lc01@lcsyn'])
    eb.remove_data('lc01')
    print(eb.summary())
    eb.rv_fromarrays('primary', phase=np.linspace(-0.6, 0.6, 201))
    eb.rv_fromarrays('secondary', phase=np.linspace(-0.6, 0.6, 201))
    print(eb.summary())
    eb.run_compute('detailed')
    
    plt.figure()
    eb.plot_syn('rv01','r-')
    eb.plot_syn('rv02','b-')
    
    eb.disable_data('rv02')
    eb.enable_data('rv02')
     
    eb.save('my_1st_binary.phoebe')
    eb = phoebe.Bundle('my_1st_binary.phoebe')
    
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close('all')
        
    assert(True)
    

if __name__ == "__main__":
    test_01()