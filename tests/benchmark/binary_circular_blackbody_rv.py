
import numpy as np
import phoebe
phoebe.progressbars_off()

b = phoebe.Bundle.default_binary()

b.add_dataset('rv', compute_times=np.linspace(0,3,101), ld_mode='manual', ld_func='logarithmic', ld_coeffs=[0,0], dataset='rv01')

b.run_compute(atm='blackbody')
