#!/usr/bin/env python3

import os
import sys
import numpy as np
import phoebe
from astropy import units

def test_xitau_agreement(verbose=False):

    dir_ = os.path.dirname(os.path.realpath(__file__))

    b = phoebe.default_binary()

    times, u, v, wavelengths, vises, sigmas = np.loadtxt(os.path.join(dir_, "Vis.dat"), usecols=[0, 1, 2, 3, 5, 6], unpack=True)

    b.add_dataset('vis', times=times, u=u, v=v, wavelengths=wavelengths, vises=vises, sigmas=sigmas, if_method='integrate')

    b.set_value('distance', context = 'system', value=100*units.pc)

    b.run_compute()

    f = open('twigs.txt', 'w')
    for twig in b.twigs:
      f.write("%s\n" % (twig))
    f.close()

    #print("b['times@vis01@vis@dataset'] = ", b['times@vis01@vis@dataset'])
    #print("b['u@vis01@vis@dataset'] = ", b['u@vis01@vis@dataset'])
    #print("b['v@vis01@vis@dataset'] = ", b['v@vis01@vis@dataset'])
    #print("b['wavelengths@vis01@vis@dataset'] = ", b['wavelengths@vis01@vis@dataset'])
    #print("b['vises@vis01@vis@dataset'] = ", b['vises@vis01@vis@dataset'])
    #print("")
    #print("b['vis@vis01@phoebe01@latest@vis@model'] = ", b['vis@vis01@phoebe01@latest@vis@model'])
    #print("")
    #print("b['times@vis01@phoebe01@latest@vis@model'] = ", b['times@vis01@phoebe01@latest@vis@model'])
    #print("b['u@vis01@phoebe01@latest@vis@model'] = ", b['u@vis01@phoebe01@latest@vis@model'])
    #print("b['v@vis01@phoebe01@latest@vis@model'] = ", b['v@vis01@phoebe01@latest@vis@model'])
    #print("b['wavelengths@vis01@phoebe01@latest@vis@model'] = ", b['wavelengths@vis01@phoebe01@latest@vis@model'])
    #print("b['vises@vis01@phoebe01@latest@vis@model'] = ", b['vises@vis01@phoebe01@latest@vis@model'])

    times = b['times@vis01@phoebe01@latest@vis@model'].value
    u = b['u@vis01@phoebe01@latest@vis@model'].value
    v = b['v@vis01@phoebe01@latest@vis@model'].value
    wavelengths = b['wavelengths@vis01@phoebe01@latest@vis@model'].value
    vises = b['vises@vis01@phoebe01@latest@vis@model'].value

    np.savetxt('test_xitau_agreement.out', np.c_[times, u, v, wavelengths, vises], header='times u v wavelenghts vises')

    #b.plot(show=True)
    #b.plot(x='u', marker='.', linestyle='none', show=True)
    #b.plot(x='v', marker='.', linestyle='none', show=True)

    # Note: Approximations are not the same in Xitau and Phoebe! 
    # integrate vs. simple disk, limb darkening vs. none, ...

    # cf. xitau/chi2_VIS.dat
    assert(abs(vises[0]   - 0.99996665418880559    ) < 1.0e-6)
    assert(abs(vises[329] - 7.6945976482721626E-002) < 1.0e-3)
    assert(abs(vises[330] - 0.99999882618925284    ) < 1.0e-6)
    assert(abs(vises[659] - 0.88179246639196074    ) < 2.0e-2)

if __name__ == "__main__":
    logger = phoebe.logger(clevel='INFO')

    test_xitau_agreement(verbose=True)


