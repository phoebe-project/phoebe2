"""
Verification of atmosphere tables and interpolation methods
============================================================

Compare intensities derived from the
:download:`Wilson-Devinney atmosphere grids <atmcof.dat>`, and the
:download:`Phoebe 2.0 atmosphere grids <kurucz_p00_linear_equidist_r_leastsq_teff_logg.fits>`
(click on the links to download the required files).

We create a template array for the effective temperature (between 3500 and
10000 K) and for the log surface gravities (between 1.0 and 5.0). Assuming
solar metallicities, we compute the specific intensities and compare them.

There are three diagnostic plots:

    1. the first just shows the values of the specific intensities computed with
       one or the other table.
    2. the second plot shows the residuals after subtracting Phoebe 2.0 from the
       WD intensities
    3. the third plot shows the values of the derivatives. The ones for Phoebe
       2.0 should be step functions, as the interpolation is done linearly in
       the logarithm of the tabulated specific intensities.
       
       
.. figure:: comp_logg_04_000.png
   :width: 800 px
   :align: center
   
   Varying Teff, fixed logg=4.0.


.. figure:: comp_logg_03_923.png
   :width: 800 px
   :align: center
   
   Varying Teff, fixed logg=3.923

.. figure:: comp_logg_04_123.png
   :width: 800 px
   :align: center
   
   Varying Teff, fixed logg=4.123


.. figure:: comp_teff_6000.png
   :width: 800 px
   :align: center
   
   Fixed Teff=6000K, varying logg.



Procedure
-------------

"""
import numpy as np
import matplotlib.pyplot as plt
from phoebe.atmospheres import limbdark

# Define the parameters for which to compute the LD coefficients and/or flux
atm_kwargs = dict(teff=np.linspace(3500, 10000, 1000), logg=3.75, abun=0.)
atm_kwargs = dict(teff=6000, logg=np.linspace(1.0,4.95,1000), abun=0.)

# Define the filenames of the two atmosphere tables to compare
atm_file_wd = 'atmcof.dat'
atm_file_phoebe = 'kurucz_p00_linear_equidist_r_leastsq_teff_logg.fits'

# Interpolate the intensities and LD coefficients (if available). The output
# below is an array of length Nx1000, where N=1 for WD, and N=2 for Phoebe 2.0.
# The last element of the first axis denotes the intensities, the rows before it
# denote the LD coefficients (there are none for the WD atmosphere tables).
out1 = limbdark.interp_ld_coeffs_wd(atm_file_wd, 'JOHNSON.V',
                                    atm_kwargs=atm_kwargs)
out2 = limbdark.interp_ld_coeffs(atm_file_phoebe, 'JOHNSON.V',
                                 atm_kwargs=atm_kwargs)


# Diagnostics
#---------------------
# All 'calculations' are done now, the rest of the code concerns making plots
# and computing diagnostics

# Make a figure of the true intensities, residuals and derivatives
plt.figure(figsize=(14,6))
plt.subplots_adjust(left=0.07, right=0.97, wspace=0.24)

# Which was a free parameter?
if not np.isscalar(atm_kwargs['teff']):
    x = 'teff'
    xlabel = 'Effective temperature [K]'
    xlabel_deriv = r"$d\ln(I)/dT_\mathrm{eff}$ [%]"
    ylim1 = 1e5, 2.5e7
    ylim2 = -1.5, 5
    ylim3 = -0.1, 0.3
elif not np.isscalar(atm_kwargs['logg']):
    x = 'logg'
    xlabel = 'log(Surface gravity [cm/s2]) [dex]'
    xlabel_deriv = r"$d\ln(I)/d\log g$ [%]"
    ylim1 = 3.9e6, 4.5e6
    ylim2 = -6, 1
    ylim3 = -5, 5
else:
    raise NotImplementedError

# True intensities
plt.subplot(131)
plt.title('Centre-of-disk specific intensity')
plt.plot(atm_kwargs[x], out1[0], '-', lw=2, label='WD')
plt.plot(atm_kwargs[x], out2[1], '-', lw=2, label='Phoebe')
plt.gca().set_yscale('log')
plt.ylim(ylim1)
plt.legend(loc='best').get_frame().set_alpha(0.5)
plt.xlabel(xlabel)
plt.ylabel(r'Specific intensity [erg/s/cm2/$\AA$/sr]')

# Residuals
plt.subplot(132)
plt.title('Residuals')
plt.plot(atm_kwargs[x], (out1[0]-out2[1])/out2[1]*100, 'k-', lw=2)
plt.grid()
plt.ylim(ylim2)
plt.ylabel('Relative difference ((Phoebe-WD)/WD)[%]')
plt.xlabel(xlabel)

# Derivatives
plt.subplot(133)
plt.title("Derivative")
plt.plot(atm_kwargs[x][:-1],
         100*np.diff(out1[0])/np.diff(atm_kwargs[x])/out1[0][:-1],
         label='WD', lw=2)
plt.plot(atm_kwargs[x][:-1],
         100*np.diff(out2[1])/np.diff(atm_kwargs[x])/out2[1][:-1],
         label='Phoebe', lw=2)
plt.legend(loc='best').get_frame().set_alpha(0.5)
plt.ylabel(xlabel_deriv)
plt.ylim(ylim3)
plt.xlabel(xlabel)

# That's it
plt.show()
