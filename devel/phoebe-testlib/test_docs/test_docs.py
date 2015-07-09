import phoebe
import numpy as np
from glob import glob
import os, sys




def test_scripts(start_ind=0):
    """
    testing all testsuite example scripts
    """
    
    direc = '../../phoebe-testsuite'
    
    skip = ["../../phoebe-testsuite/pulsating_star/pulsating_star.py",          # FAILS
            "../../phoebe-testsuite/beaming/KPD1946+4340.py",                   # includes animate in compute
            "../../phoebe-testsuite/1to2comp/ph1to2comp.py",                    # FAILS
            "../../phoebe-testsuite/wilson_devinney/body_emul.py",              # missing file
            "../../phoebe-testsuite/wilson_devinney/eccentric_orbit.py",        # missing file
            "../../phoebe-testsuite/wilson_devinney/reflection_effect.py",      # missing file
            "../../phoebe-testsuite/wilson_devinney/wd_vs_phoebe.py",           # missing file
            "../../phoebe-testsuite/simple_binary/simple_binary02.py",          # missing lc.obs
            "../../phoebe-testsuite/pulsating_binary/pulsating_binary2.py",     # BUG!!!
            "../../phoebe-testsuite/sirius/sirius.py",                          # BUG!!! (in atmospheres)
            "../../phoebe-testsuite/venus/mercury.py",                          # FAILS
            "../../phoebe-testsuite/venus/venus.py",                            # FAILS
            "../../phoebe-testsuite/example_systems/example_systems.py",        # FAILS
            "../../phoebe-testsuite/vega/vega_sed.py",                          # FAILS
            "../../phoebe-testsuite/accretion_disk/T_CrB.py",                   # FAILS
            "../../phoebe-testsuite/accretion_disk/accretion_disk.py",          # FAILS
            "../../phoebe-testsuite/oblique_magnetic_dipole/oblique.py",        # FAILS
            "../../phoebe-testsuite/traditional_approximation/traditional_approximation.py", # missing file
            "../../phoebe-testsuite/misaligned_binary/misaligned_binary.py",    # SLOW (seems ok)
            "../../phoebe-testsuite/occulting_dark_sphere/transit_colors.py",   # BUG!!! (in atmospheres)
            "../../phoebe-testsuite/occulting_dark_sphere/occulting_dark_sphere.py", # BUG!!! (in atmospheres)
            "../../phoebe-testsuite/critical_rotator/critical_rotator.py",      # BUG!!! (in atmospheres)
            "../../phoebe-testsuite/contact_binary/contact_binary.py",          # missing file
            "../../phoebe-testsuite/frontend_tutorials/first_steps.py",                 # NEEDS UPDATING
            "../../phoebe-testsuite/frontend_tutorials/first_binary_from_scratch.py",   # NEEDS UPDATING
            "../../phoebe-testsuite/frontend_tutorials/interferometry.py",              # NEEDS UPDATING
            ]
    
    for i,f in enumerate(glob(os.path.join(direc, '*.py'))+glob(os.path.join(direc, '*/*.py'))):
        if f not in skip and i>=start_ind:
            print("running (making sure doesn't crash): {:d} {}".format(i,f))
            fo = open(f, 'r')
            code = fo.read()
            fo.close()
            code = code.replace('plt.show()', 'plt.savefig("tmp.png")')
            exec(code)    
            
if __name__ == "__main__":
    logger = phoebe.get_basic_logger()
    test_scripts(int(sys.argv[1]) if len(sys.argv)>1 else 0)
