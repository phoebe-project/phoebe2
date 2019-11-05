import phoebe
import numpy as np
import matplotlib.pyplot as plt
import interp

pb = phoebe.get_passband('Kepler:phoenix')

# 0) verify the simplest interpolation -- WORKS
# teffs = np.array((2900.,))
# loggs = np.array((4.0,))
# abuns = np.array((0.0,))

# req = np.vstack((teffs, loggs, abuns)).T
# axes = pb._phoenix_axes
# grid = 10**pb._phoenix_photon_grid

# interp.interp(req, axes, grid)

# 1) verify interpolation -- WORKS
# teffs = np.arange(pb._phoenix_axes[0][0], 4001., 10.)
# ints = pb.Inorm(teffs, np.ones_like(teffs)*3.0, np.zeros_like(teffs), atm='phoenix', ld_func='linear', ld_coeffs=[0.0], photon_weighted=True)
# plt.plot(pb._phoenix_axes[0][:17], 10**pb._phoenix_photon_grid[:,6,6][:17], 'bo', lw=2, label='phoenix grid points')
# plt.plot(teffs, ints, 'r-', label='phoenix interpolated')
# plt.legend(loc='upper left')
# plt.show()

teffs = np.arange(1800., 4001., 10.)

Iph = np.nan*np.ones_like(teffs)
Ibb = np.zeros_like(teffs)
Ibl = np.zeros_like(teffs)

for i, teff in enumerate(teffs):
    Ibb[i] = pb.Inorm(teff, 4.5, 0.0, atm='blackbody', ld_func='linear', ld_coeffs=[0.0], photon_weighted=True)
    Ibl[i] = pb.Inorm(teff, 4.5, 0.0, atm='blended', ld_func='linear', ld_coeffs=[0.0], photon_weighted=True)
    try:
        Iph[i] = pb.Inorm(teff, 4.5, 0.0, atm='phoenix', ld_func='linear', ld_coeffs=[0.0], photon_weighted=True)
    except:
        pass

plt.plot(teffs, Ibb, 'k-', lw=2, label='blackbody')
plt.plot(teffs, Ibl, 'g-', lw=1, label='blended')

teffs = np.arange(1800., 4001., 100.)
plt.plot(teffs, 10**pb._blended_photon_grid[15:38,9,6,0], 'bo', label='blended nodes')
plt.plot(teffs[5:], 10**pb._phoenix_photon_grid[:18,9,6,0], 'ro', label='phoenix nodes')

# plt.plot(teffs, Iph, 'r-', lw=2, label='phoenix')
# plt.plot(teffs, Ibl, 'g-', lw=1, label='blended')
plt.legend(loc='upper left')
plt.show()
