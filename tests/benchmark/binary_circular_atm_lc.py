
import numpy as np
import phoebe
phoebe.progressbars_off()

b = phoebe.Bundle.default_binary()

b.add_dataset('lc', time=np.linspace(0,3,101), dataset='lc01')

b.run_compute(atm='ck2004')
