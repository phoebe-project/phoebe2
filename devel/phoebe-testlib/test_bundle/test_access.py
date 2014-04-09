import phoebe
import os

def test_access():
    """
    Testing bundle setters and getters
    """
    mybundle = phoebe.Bundle(os.path.join(os.path.dirname(os.path.abspath(__file__)),'defaults.phoebe'))
    #print mybundle
    
    # test get value
    assert(mybundle.get_value('distance')==10)
    assert(mybundle.get_value('teff@primary')==8350)
    assert(mybundle.get_value('teff->primary')==8350)
    assert(mybundle.get_value('delta@mesh:marching@secondary')==0.0527721121857703257)
    assert(mybundle.get_value('atm@lcdep@secondary')=='kurucz')
    assert(mybundle.get_value('atm@component@secondary')=='kurucz')
    
    # test set value
    mybundle.set_value('teff@primary',9000)
    assert(mybundle.get_value('teff@primary')==9000)
    mybundle.set_value('teff->primary', 9001)
    assert(mybundle.get_value('teff->primary')==9001)
    mybundle.set_value('delta@mesh:marching@secondary', 0.5)
    assert(mybundle.get_value('delta@mesh:marching@secondary')==0.5)
    mybundle.set_value('atm@lcdep@secondary', 'blackbody')
    assert(mybundle.get_value('atm@lcdep@secondary')=='blackbody')
    mybundle.set_value('atm@component@secondary','something')
    assert(mybundle.get_value('atm@component@secondary')=='something')
    
    # test dictionairy output
    out = mybundle.get_value('teff', all=True)
    assert(out['teff@component@primary'] == 9001)
    assert(out['teff@component@secondary'] == 7780.)
    
    mybundle.set_value('teff', 8123, all=True)
    out = mybundle.get_value('teff', all=True)
    assert(out['teff@component@primary'] == 8123)
    assert(out['teff@component@secondary'] == 8123)
    
    
    
if __name__ == "__main__":
    test_access()
    