"""
Configuration module.

This module defines all information on datasets and lcdeps, that is useful in
various places in the code.

It also contains information on the different atmosphere files, and the parameters
in which can be interpolated (atmospheric parameters, not RV or reddening).
"""

# The list of all data types for which time is the independent variable.
indep_var_time = ['lc', 'rv', 'sp', 'pl', 'if', 'am']

# List of all data type for which time is not an independent variable (but
# rather needs to be fitted)
indep_var_other = ['etv']

# Nice names for the shortcuts
nice_names = {'lc':'Light curve',
              'rv':'Radial velocity curve',
              'sp':'Spectrum',
              'pl':'Spectrapolarimetry',
              'if':'Interferometry',
              'am':'Astrometry',
              'etv':'Eclipse timing variations',
              'orb':'Synthetic orbits'}

# Correspondence between data type and function to compute it
compute_function = {'lc':'lc',
                    'rv':'rv',
                    'sp':'sp',
                    'pl':'pl',
                    'if':'ifm',
                    'am':'am',
                    'etv':'etv',
                    'orb':'orb'}

# Correspondence between data type and pbdep (syn and obs are then similar):
pbdep_context = {'lc':'lcdep',
                 'rv':'rvdep',
                 'sp':'spdep',
                 'pl':'pldep',
                 'if':'ifdep',
                 'am':'amdep',
                 'etv':'etvdep'}

# Correspondence between data type and DataSet class:
dataset_class = {'lc':'LCDataSet',
                 'rv':'RVDataSet',
                 'sp':'SPDataSet',
                 'pl':'PLDataSet',
                 'if':'IFDataSet',
                 'am':'DataSet',
                 'etv':'DataSet',
                 'orb':'DataSet'}

# Correspondence between data type and file extension:
file_extension = {'lc':'lc',
                  'phot':'lc',
                  'spec':'sp',
                  'lprof':'sp',
                  'rv':'rv',
                  'plprof':'pl',
                  'vis2':'if',
                  'am':'am',
                  'etv':'etv'}

# Atmosphere properties
atm_props = {'blackbody': ('teff',),
             'kurucz': ('teff', 'logg', 'abun'),
             'kurucz_p00': ('teff', 'logg'),
             'phoenix': ('teff', 'logg', 'abun'),
             'olivia.fits': ('teff', 'abun', 'eddy', 'uvflux'),
             'jorissen': ('teff', 'logg',),
             'wd':('teff','logg','abun',)}

fit_props = {'jorissen': 'equidist_mu_leastsq'}
