import phoebe
import numpy as np



def test_rv_compute_times():
    b = phoebe.default_binary()
    b.add_dataset('rv', compute_phases=phoebe.linspace(0,1,4), dataset='rv01')
    b.add_dataset('mesh', include_times='rv01', dataset='mesh01')
    b.run_compute()

    assert len(b.filter(context='model', kind='mesh').times) == 4

if __name__ == '__main__':
    test_rv_compute_times()
