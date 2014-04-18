import phoebeBackend as phb
import scipy.stats as st
import numpy as np
from math import pi

def calculate_radius(M,logg):
    """
    Compute radius from mass and logg: (R/RSun) = sqrt(M/MSun*gSun/10**logg).
    Mass is in Solar masses, logg is in cgs units, the computed radius is in Solar radii.
    """
    return (27435.153*M/10**logg)**0.5

def calculate_sma(M1, M2, P):
    """
    Computes the semi-major axis of the binary (in solar radii).
    The masses M1 and M2 are in solar masses, the period is in days.
    """

    return 4.20661*((M1+M2)*P**2)**(1./3)

def calculate_pot1(r, D, q, F, l, n):
    """
    Computes surface potential of the primary star.
    """
    return 1./r + q*( (D**2+r**2-2*r*l*D)**-0.5 - r*l/D**2 ) + 0.5*F**2*(1.+q)*r**2*(1-n**2)

def calculate_pot2(r, D, q, F, l, n):
    """
    Computes surface potential of the secondary star.
    """
    q = 1./q
    pot = calculate_pot1(r, D, q, F, l, n)
    return pot/q + 0.5*(q-1)/q        

def conjunction_separation(a, e, w):
    """
    Calculates instantaneous separation at superior and inferior conjunctions.
    """

    dp = a*(1-e*e)/(1+e*np.sin(w))
    ds = a*(1-e*e)/(1-e*np.sin(w))
    return (dp, ds)

phb.init()
phb.configure()

while True:
    phb.open("default.phoebe")

    M1 = st.uniform.rvs(0.2, 4.8)
    M2 = st.uniform.rvs(0.2, 4.8)

    L1, L2 = M1**3.5, M2**3.5

    T1 = st.uniform.rvs(3500., 11500.)
    T2 = st.uniform.rvs(3500., 11500.)

    R1 = L1**0.5*(T1/5860.)**-2
    R2 = L2**0.5*(T2/5860.)**-2

    P0 = st.uniform.rvs(1, 19)

    if P0 < 1:
        ecc = st.uniform.rvs(0.0, 0.05)
    elif P0 < 10:
        ecc = st.uniform.rvs(0.0, 0.4)
    else:
        ecc = st.uniform.rvs(0.0, 0.6)

    q = st.uniform.rvs(0.2, 0.8)

    # Randomize the most important parameters
    phb.setpar('phoebe_grid_finesize1', 60)
    phb.setpar('phoebe_grid_finesize2', 60)
    phb.setpar('phoebe_period', P0)
    phb.setpar('phoebe_sma', calculate_sma(M1, M2, P0))
    phb.setpar('phoebe_rm', q)
    phb.setpar('phoebe_ecc', ecc)
    phb.setpar('phoebe_perr0', st.uniform.rvs(0, 2*pi))
    phb.setpar('phoebe_pot1', calculate_pot1(R1/phb.getpar('phoebe_sma'), 1-ecc, q, 1.0, 0, 1))
    phb.setpar('phoebe_pot2', calculate_pot1(R2/phb.getpar('phoebe_sma'), 1-ecc, q, 1.0, 0, 1))
    phb.setpar('phoebe_teff1', T1)
    phb.setpar('phoebe_teff2', T2)

    if T1 > 7500:
        phb.setpar('phoebe_grb1', 1.0)
        phb.setpar('phoebe_alb1', 1.0)
    else:
        phb.setpar('phoebe_grb1', 0.32)
        phb.setpar('phoebe_alb1', 0.6)

    if T2 > 7500:
        phb.setpar('phoebe_grb2', 1.0)
        phb.setpar('phoebe_alb2', 1.0)
    else:
        phb.setpar('phoebe_grb2', 0.32)
        phb.setpar('phoebe_alb2', 0.6)

    ph = tuple(np.linspace(0, P0, 201).tolist())
    fl = phb.lc(ph, 0)
    print ph, fl

    import matplotlib.pyplot as plt
    plt.plot(ph, fl, 'r-')
    plt.show()

    yn = input('Save?')
    if yn == 1:
        val = input("Value: ")
        phb.save('lc%02d.phoebe' % val)

phb.quit()
