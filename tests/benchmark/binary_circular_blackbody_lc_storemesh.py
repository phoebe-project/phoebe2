
import numpy as np
import phoebe

b = phoebe.Bundle.default_binary()

b.add_dataset('lc', time=np.linspace(0,3,101), dataset='lc01')
b.add_dataset('mesh', include_times=['lc01'])

b.run_compute(atm='blackbody', ld_func='logarithmic', ld_coeffs=[0,0])
