import phoebe
import os
import numpy as np
import nose.tools

def test_json():
    """
    Bundle: basic json input and output
    """
    mybundle = phoebe.Bundle()
    
    # attach a non-standard  parameter set
    #mybundle.attach_ps(phoebe.PS('reddening:interstellar'))
    mybundle.set_value('passband@reddening', 'JOHNSON.K')
    
    # change a value
    mybundle.set_value('period',200)
    
    # change adjust
    mybundle.set_adjust('period',True)

    # create a synthetic dataset
    mybundle.rv_fromarrays(phase=np.linspace(0,1,100),objref=['primary','secondary'],dataref='myrv_phase')
    mybundle.rv_fromarrays(time=np.linspace(0,1,50),objref='primary',dataref='myrv_time')
    mybundle.lc_fromarrays(time=np.linspace(0,1,10),dataref='mylc_time')
    
    #~ print "\n", mybundle.twigs('myrv_phase')
    #~ print "\n", mybundle.twigs('myrv_time')
    #~ print "\n", mybundle.twigs('mylc_time')
    
    assert(len(mybundle.twigs('myrv_phase'))==6)
    assert(len(mybundle.twigs('myrv_time'))==3)
    assert(len(mybundle.twigs('mylc_time'))==4)
    assert(len(mybundle['phase@myrv_phase@rvobs@primary'])==100)
    assert(len(mybundle['time@myrv_time@rvobs@primary'])==50)
    assert(len(mybundle['time@mylc_time@lcobs@new_system'])==10)
    

    mybundle.save('test_io.json')

    mybundle = phoebe.Bundle('test_io.json')
    
    # make sure the non-standard PS was restored with the changed value
    assert(mybundle.get_value('passband@reddening') == 'JOHNSON.K')
    
    # make sure the value change was loaded
    assert(mybundle.get_value('period')==200)
    
    # make sure adjustments are loaded
    assert(mybundle.get_adjust('period')==True)
    
    # make sure data_fromarrays all loaded
    #~ print "\n", mybundle.twigs('myrv_phase')
    #~ print "\n", mybundle.twigs('myrv_time')
    #~ print "\n", mybundle.twigs('mylc_time')
    
    assert(len(mybundle.twigs('myrv_phase'))==6)
    assert(len(mybundle.twigs('myrv_time'))==3)
    assert(len(mybundle.twigs('mylc_time'))==4)
    assert(len(mybundle['phase@myrv_phase@rvobs@primary'])==100)
    assert(len(mybundle['time@myrv_time@rvobs@primary'])==50)
    assert(len(mybundle['time@mylc_time@lcobs@new_system'])==10)


if __name__ == "__main__":
    test_json()
