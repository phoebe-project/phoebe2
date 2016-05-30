
import numpy as np
import phoebe

b = phoebe.Bundle.default_binary()

b.add_dataset('lc', time=np.linspace(0,3,101), dataset='lc01')

b.add_compute()
b.set_value_all('atm', 'ck2004')

b.run_compute()
