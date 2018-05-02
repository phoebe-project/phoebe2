import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

invlam, Klam55, dKdR = np.loadtxt('fm18.txt', unpack=True)

extinct_func = interpolate.splrep(invlam, Klam55, k=3, s=0)
dKdR_func = interpolate.splrep(invlam, dKdR, k=3, s=0)
extinct = lambda invwl, R: interpolate.splev(invwl, extinct_func) + 0.9934*interpolate.splev(invwl, dKdR_func) * (R-3.1)

invl = np.linspace(invlam[0], invlam[-1], 1000)

plt.xlabel(r'$1/\lambda~[1/\mu m]$')
plt.ylabel(r'$\frac{E(\lambda-55)}{E(44-55)}$')
plt.plot(invlam, Klam55, 'bo', label='FM2018')
plt.plot(invl, extinct(invl, 1.0), 'g-', lw=2, label='R=1.0')
plt.plot(invl, extinct(invl, 5.0), 'm-', lw=2, label='R=5.0')
plt.plot(invl, extinct(invl, 3.1), 'r-', lw=2, label='R=3.1')
plt.legend(loc='lower right')
plt.show()
