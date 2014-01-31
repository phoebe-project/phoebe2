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

# Correspondence between data type and function to compute it
compute_function = {'lc':'lc',
                    'rv':'rv',
                    'sp':'sp',
                    'pl':'pl',
                    'if':'ifm',
                    'am':'am',
                    'etv':'etv'}

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
                 'etv':'DataSet'}

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
             'kurucz': ('teff', 'logg'),
             'phoenix': ('teff', 'logg', 'abun'),
             'olivia': ('teff', 'abun', 'eddy', 'uvflux')}
