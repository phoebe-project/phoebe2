
import numpy as np
import phoebe

b = phoebe.Bundle.default_binary()

b.add_dataset('lc', time=np.linspace(0,3,101), dataset='lc01')

b.run_compute(atm='ck2004')
