import phoebe
import os
import numpy as np
import nose.tools

def test_json():
    """
    Testing basic Bundle json input and output
    """
    mybundle = phoebe.Bundle()
    
    mybundle.set_value('period',200)
    mybundle.set_adjust('period',True)

    mybundle.save('test_io.json')

    mybundle = phoebe.Bundle('test_io.json')
    
    assert(mybundle.get_value('period')==200)
    assert(mybundle.get_adjust('period')==True)

#def test_json2():
    #"""
    #Testing customizing Bundle with json input and output
    #"""
    ## What happens with non standard parameterSets?
    #mybundle = phoebe.Bundle()
    #mybundle.attach_ps('new_system', phoebe.PS('reddening:interstellar'))
    
    #mybundle.set_value('passband@reddening', 'JOHNSON.K')
    #mybundle.save('test_io2.json')

    #mybundle = phoebe.Bundle('test_io.json')
    #assert(mybundle.get_value('passband@reddening') == 'JOHNSON.K')
    #assert(mybundle.get_value('period')==200)
    #assert(mybundle.get_adjust('period')==True)

if __name__ == "__main__":
    test_json()
    #test_json2()
    
