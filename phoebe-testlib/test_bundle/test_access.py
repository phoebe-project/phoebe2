import phoebe
import os
import numpy as np
import nose.tools

def test_access():
    """
    Bundle: setters and getters
    """
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),'defaults.phoebe')
    mybundle = phoebe.Bundle(filename)
    
    # set a new parameterSet
    #mybundle.attach_ps(phoebe.PS('reddening:interstellar'), 'Detached_1')
    
    # test get value
    assert(mybundle.get_value('vgamma@position')==-12.5)
    assert(mybundle.get_value('teff@primary')==8350)
    assert(mybundle.get_value('label@primary')=='primary')
    assert(mybundle.get_value('delta@mesh:marching@secondary')==0.0527721121857703257)
    assert(mybundle.get_value('atm@lcdep@secondary')=='kurucz')
    assert(mybundle.get_value('atm@component@secondary')=='kurucz')
    assert(mybundle.get_value('passband@reddening')=='JOHNSON.V')
    
    # test set value
    mybundle.set_value('teff@primary',9000)
    assert(mybundle.get_value('teff@primary')==9000)
    mybundle.set_value('teff@primary', 9001)
    assert(mybundle.get_value('teff@primary')==9001)
    mybundle.set_value('delta@mesh:marching@secondary', 0.5)
    assert(mybundle.get_value('delta@mesh:marching@secondary')==0.5)
    mybundle.set_value('atm@lcdep@secondary', 'blackbody')
    assert(mybundle.get_value('atm@lcdep@secondary')=='blackbody')
    mybundle.set_value('atm@component@secondary','kurucz')
    assert(mybundle.get_value('atm@component@secondary')=='kurucz')
    
    # test unit support:
    mybundle.set_value('teff@primary', 20000, 'Far')
    assert(np.abs(mybundle['value@teff@primary']-11366.4833333)<0.01)
    
    # add some data and do similar stuff
    mybundle.data_fromarrays(category='lc', dataref='mylc', time=np.linspace(0,1,100), flux=np.ones(100))
    mybundle.data_fromarrays(category='rv', dataref='myprimrv', objref='primary', time=np.linspace(0,1,100), rv=np.ones(100))
    mybundle.data_fromarrays(category='rv', dataref='mysecrv', objref='secondary', time=np.linspace(0.2,0.3,109), rv=-np.ones(109))
    #print mybundle.get_value('time@mylc')
    
    # make sure the data can be found
#    assert(len(mybundle.get('rvobs@primary'))==2)
#    assert(len(mybundle.get('rvobs@secondary'))==2)
#    assert(len(mybundle.get('lcobs@Detached_1'))==2)
    assert(len(mybundle.get('dataset'))==6)
   
    # make sure each twig is a unique twigs for itself
    for t in mybundle.twigs():
        if len(mybundle.twigs(t))!=1: print t, mybundle.twigs(t)    
        assert(len(mybundle.twigs(t))==1)

def test_dictionary():
    """
    Bundle: dictionary behaviour
    """
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),'defaults.phoebe')
    mybundle = phoebe.Bundle(filename)
    
    # this should just work, I'm not gonna test for explicit values here:
    assert(isinstance(mybundle.keys(), list))
    assert(isinstance(mybundle.keys()[0], str))
    assert(isinstance(mybundle.values(), list))
    assert(isinstance(mybundle.items(), list))
    assert(isinstance(mybundle.items()[0], tuple))
    
    # test for explicit values
    assert(mybundle.get('value@period')==5.25)
    assert(mybundle.get('non-existing-key') == None)
    assert(mybundle.get('non-existing-key', None) == None)
    assert(mybundle.get('non-existing-key', 'my_default') == 'my_default')
    
    assert(mybundle['value@period']==5.25)
    
def test_methods():
    """
    Bundle: testing that various methods don't fail
    """
    
    b = phoebe.Bundle()
    
    b.hierarchy()
    b.hierarchy(change_labels=True)
    b.get_object()
    b.get_mesh()
    
    
@nose.tools.raises(KeyError)        
def test_dictionary_err():
    """
    Bundle: dictionary behaviour (error raising)
    """
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),'defaults.phoebe')
    mybundle = phoebe.Bundle(filename)
    
    mybundle['non-existing-key']

    
@nose.tools.raises(KeyError)    
def test_error():
    """
    Bundle: setter and getters (error raising)
    """
    mybundle = phoebe.Bundle(os.path.join(os.path.dirname(os.path.abspath(__file__)),'defaults.phoebe'))
    mybundle.get_value('teff')
    
    
if __name__ == "__main__":
    test_access()
    test_error()
    test_dictionary()
    test_dictionary_err()
    
