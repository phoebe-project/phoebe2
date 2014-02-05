import phoebeBackend as phb
import scipy.stats as st
import numpy as np

phb.init()
phb.configure()

phb.setpar("phoebe_lcno", 1)

phb.setpar("phoebe_pot1", st.uniform.rvs(4.0, 3.0))
phb.setpar("phoebe_pot2", st.uniform.rvs(6.0, 3.0))
phb.setpar("phoebe_incl", st.uniform.rvs(80, 10))
phb.setpar("phoebe_ecc", st.uniform.rvs(0.0, 0.3))
phb.setpar("phoebe_perr0", st.uniform.rvs(0.0, 2*np.pi))
phb.setpar("phoebe_rm", st.uniform.rvs(0.5, 0.5))

print("# pot1 = %f" % phb.getpar("phoebe_pot1"))
print("# pot2 = %f" % phb.getpar("phoebe_pot2"))
print("# incl = %f" % phb.getpar("phoebe_incl"))
print("# ecc  = %f" % phb.getpar("phoebe_ecc"))
print("# per0 = %f" % phb.getpar("phoebe_perr0"))
print("# rm   = %f" % phb.getpar("phoebe_rm"))

ph = tuple(np.linspace(-0.5, 0.5, 201).tolist())
lc = phb.lc(ph, 0)

for i in range(len(ph)):
    print ph[i], lc[i]

phb.quit()
