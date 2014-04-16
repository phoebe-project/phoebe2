import phoebe
import os
import numpy as np
import nose.tools

def test_json():
    """
    Testing bundle json input and output
    """
    mybundle = phoebe.Bundle()
    
    mybundle.set_value('period',200)
    mybundle.set_adjust('period',True)

    mybundle.save('test_io.json')

    mybundle = phoebe.Bundle('test_io.json')
    
    assert(mybundle.get_value('period')==200)
    assert(mybundle.get_adjust('period')==True)

if __name__ == "__main__":
    test_json()
    
