import phoebeBackend as phb
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import time
import phoebe as phb2
from phoebe.io import parsers
import itertools


# Initialize Phoebe1 and Phoebe2
phb.init()
phb.configure()

# To make sure we are using the same defaults, open the same parameter file:
phb.open("default.phoebe")

phb.setpar("phoebe_lcno", 1)

Ncurves = 50
color_cycle = itertools.cycle(plt.cm.spectral(np.linspace(0,1,Ncurves)))

timings = np.zeros((Ncurves, 10))

for i in range(Ncurves):
    
    # Set random parameters in Phoebe1
    phb.setpar("phoebe_pot1", st.uniform.rvs(4.0, 1.0))
    phb.setpar("phoebe_pot2", st.uniform.rvs(5.0, 1.0))
    phb.setpar("phoebe_incl", st.uniform.rvs(80, 10))
    phb.setpar("phoebe_ecc", st.uniform.rvs(0.0, 0.0))
    phb.setpar("phoebe_perr0", st.uniform.rvs(0.0, 2*np.pi))
    phb.setpar("phoebe_rm", st.uniform.rvs(0.5, 0.5))
    phb.setpar("phoebe_teff2", st.uniform.rvs(6000, 500))

    # Set parameters in Phoebe2 to match Phoebe1
    mybundle = phb2.Bundle('default.phoebe')

    mybundle.set_value('pot@primary', phb.getpar('phoebe_pot1'))
    mybundle.set_value('pot@secondary', phb.getpar('phoebe_pot2'))
    mybundle.set_value('incl', phb.getpar('phoebe_incl'))
    mybundle.set_value('ecc', phb.getpar('phoebe_ecc'))
    mybundle.set_value('per0', phb.getpar('phoebe_perr0'))
    mybundle.set_value('q', phb.getpar('phoebe_rm'))
    mybundle.set_value('teff@secondary', phb.getpar('phoebe_teff2'))

    # Report
    print("# Qual = Phoebe1 -- Phoebe2")
    print("# pot1 = %f -- %f" % (phb.getpar("phoebe_pot1"), mybundle.get_value('pot@primary')))
    print("# pot2 = %f -- %f" % (phb.getpar("phoebe_pot2"), mybundle.get_value('pot@secondary')))
    print("# incl = %f -- %f" % (phb.getpar("phoebe_incl"), mybundle.get_value('incl')))
    print("# ecc  = %f -- %f" % (phb.getpar("phoebe_ecc"), mybundle.get_value('ecc')))
    print("# per0 = %f -- %f" % (phb.getpar("phoebe_perr0"), mybundle.get_value('per0')))
    print("# rm   = %f -- %f" % (phb.getpar("phoebe_rm"), mybundle.get_value('q')))
    print("# T2   = %f -- %f" % (phb.getpar("phoebe_teff2"), mybundle.get_value('teff@secondary')))

    # Template phases
    ph = np.linspace(-0.5, 0.5, 201)

    # Compute a phase curve with Phoebe1
    print("# Computing phoebe 1 light curve.")
    ts1 = time.time()
    lc_ph1 = phb.lc(tuple(ph.tolist()), 0)
    te1 = time.time()

    print("# Execution time: %3.3f seconds" % ((te1-ts1)))

    # Compute a phase curve with Phoebe2
    mybundle.create_syn(category='lc', phase=ph)

    ts2 = time.time()
    mybundle.run_compute(label='from_legacy')
    te2 = time.time()
    
    lc_ph2 = mybundle.get_syn(category='lc').asarray()['flux']

    print("# Execution time: %3.3f seconds" % ((te2-ts2)))

    color = color_cycle.next()
    plt.subplot(211)
    plt.plot(ph, lc_ph1/np.median(lc_ph1), 'o-', color=color)
    plt.plot(ph, lc_ph2/np.median(lc_ph2), 's-', color=color)

    plt.subplot(212)
    plt.plot(ph, lc_ph1/np.median(lc_ph1)-lc_ph2/np.median(lc_ph2), 'o-', color=color)
    
    timings[i,7] = te1-ts1
    timings[i,8] = te2-ts2
    timings[i,9] = np.std(lc_ph1/np.median(lc_ph1)-lc_ph2/np.median(lc_ph2))
    
plt.figure()
plt.subplot(221)
plt.plot(timings[:,7], timings[:,8], 'ko')
plt.subplot(222)
plt.plot(timings[:,8]/timings[:,7], 'ko-')
plt.ylabel("t(Phoebe2) / t(Phoebe1)")

plt.subplot(212)
plt.plot(timings[:,9]*1e6,'ro-')
plt.ylabel("Standard deviation of residuals [ppm]")
plt.show()
phb.quit()
