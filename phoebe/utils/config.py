"""
Configuration module.

This module defines all information on datasets that are useful in
various places in the code.

It also contains information on the different atmosphere files, and the parameters
in which can be interpolated (atmospheric parameters, not RV or reddening).
"""

# Nice names for the shortcuts
nice_names = {'lc':'Light curve',
              'rv':'Radial velocity curve',
              'sp':'Spectrum',
              'pl':'Spectrapolarimetry',
              'if':'Interferometry',
              'am':'Astrometry',
              'etv':'Eclipse timing variations',
              'orb':'Synthetic orbits',
              'sed':'Spectral energy distribution'}


# Atmosphere properties
atm_props = {'blackbody': ('teff',),
             'kurucz': ('teff', 'logg', 'abun'),
             'kurucz_p00': ('teff', 'logg'),
             'kurucz_m05': ('teff', 'logg'),
             'phoenix': ('teff', 'logg', 'abun'),
             'olivia.fits': ('teff', 'abun', 'eddy', 'uvflux'),
             'jorissen': ('teff', 'logg',),
             'wd':('teff','logg','abun',)}

fit_props = {'jorissen': 'equidist_mu_leastsq'}
