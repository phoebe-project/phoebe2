import numpy as np
from phoebe.distortions.roche import potential2rpole as _potential2rpole
from phoebe.distortions.roche import rpole2potential as _rpole2potential

def potential2rpole(*args, **kwargs):
    """
    """
    return _potential2rpole(*args, **kwargs)

def rpole2potential(*args, **kwargs):
    """
    """
    return _rpole2potential(*args, **kwargs)


def esinw2per0(ecc, esinw):
    """
    """
    # print "*** constraints.builtin.esinw2per0", ecc, esinw

    if ecc==0.:
        return 0.
    else:
        per0 = np.arcsin(esinw/ecc)
        if np.isnan(per0):
            raise ValueError("esinw={} and ecc={} results in nan for per0, please REVERT value for esinw".format(esinw, ecc))
        return per0

def ecosw2per0(ecc, ecosw):
    """
    """
    # print "*** constraints.builtin.ecosw2per0", ecc, ecosw

    if ecc==0.:
        return 0.
    else:
        per0 = np.arccos(ecosw/ecc)
        if np.isnan(per0):
            raise ValueError("ecosw={} and ecc={} results in nan for per0, please REVERT value for ecosw".format(ecosw, ecc))
        return per0
