import phoebeBackend as phb
import scipy.stats as st
import numpy as np
import phoebe as phb2


# Initialize Phoebe1 and Phoebe2
phb.init()
phb.configure()
phb2.get_basic_logger()


# Set random parameters in Phoebe1
phb.setpar("phoebe_lcno", 1)

phb.setpar("phoebe_pot1", st.uniform.rvs(4.0, 3.0))
phb.setpar("phoebe_pot2", st.uniform.rvs(6.0, 3.0))
phb.setpar("phoebe_incl", st.uniform.rvs(80, 10))
phb.setpar("phoebe_ecc", st.uniform.rvs(0.0, 0.3))
phb.setpar("phoebe_perr0", st.uniform.rvs(0.0, 2*np.pi))
phb.setpar("phoebe_rm", st.uniform.rvs(0.5, 0.5))


# Set parameters in Phoebe2 to match Phoebe1
mybundle = phb2.Bundle('defaults.phoebe', remove_dataref=True)

mybundle.set_value('pot@primary', phb.getpar('phoebe_pot1'))
mybundle.set_value('pot@secondary', phb.getpar('phoebe_pot2'))
mybundle.set_value('incl', phb.getpar('phoebe_incl'))
mybundle.set_value('ecc', phb.getpar('phoebe_ecc'))
mybundle.set_value('per0', phb.getpar('phoebe_perr0'))
mybundle.set_value('q', phb.getpar('phoebe_rm'))


# Report
print("# Qual = Phoebe1 -- Phoebe2")
print("# pot1 = %f -- %f" % phb.getpar("phoebe_pot1"),mybundle.get_value('pot@primary'))
print("# pot2 = %f -- %f" % phb.getpar("phoebe_pot2"),mybundle.get_value('pot@secondary'))
print("# incl = %f -- %f" % phb.getpar("phoebe_incl"),mybundle.get_value('incl'))
print("# ecc  = %f -- %f" % phb.getpar("phoebe_ecc"),mybundle.get_value('ecc'))
print("# per0 = %f -- %f" % phb.getpar("phoebe_perr0"),mybundle.get_value('perr0'))
print("# rm   = %f -- %f" % phb.getpar("phoebe_rm"),mybundle.get_value('rm'))


# Template phases
ph = np.linspace(-0.5, 0.5, 201)


# Compute a phase curve with Phoebe1
lc_ph1 = phb.lc(tuple(ph.tolist()), 0)


# Compute a phase curve with Phoebe2
mybundle.create_syn(category='lc', phase=ph)
mybundle.run_compute(eclipse_alg='binary')
lc_ph2 = mybundle.get_syn(category='lc')['flux']


# Analyse results
for i in range(len(ph)):
    print ph[i], lc_ph1[i], lc_ph2[i]

phb.quit()
