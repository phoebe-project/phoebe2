"""
Create, transform and manage parameters

Section 1 Introduction
======================

This subpackage is responsible for creating all parameters, data, results,
priors and posteriors of a Phoebe model. It contains top level functionality
aiming at minimal user interference in creating parameters and loading data
from various types of files.

Modules:

.. autosummary::
    
    parameters
    datasets
    create
    tools

Section 2 Top level user functions
==================================

Parsing ASCII files:

.. autosummary::
    
    datasets.parse_lc
    datasets.parse_rv
    datasets.parse_phot
    datasets.parse_vis2
    datasets.parse_spec_as_lprof
    datasets.parse_header
    
Convenience functions to create common stars and systems: by default these
return (a list of) ParameterSets, but adding ``create_body=True`` builds
a default Body instead.

.. autosummary::

    create.from_library
    create.star_from_spectral_type
    create.binary_from_stars
    create.binary_from_spectroscopy
    
Adding constraints: these functions add new parameters to existing contexts,
that are completely determined by the existing parameters. Constraints are
added to derive any one of the parameters from the others, to guarantee a
consistent parameterSet. This can be particularly useful if you want to fit
for example the surface gravity and radius, instead of mass and radius.

.. autosummary::
    
    tools.add_asini
    tools.add_rotfreqcrit
    tools.add_solarosc
    tools.add_solarosc_Deltanu0
    tools.add_solarosc_numax
    tools.add_surfgrav
    tools.add_teffpolar
    tools.add_angdiam

Example usage
-------------

Load parameters of Vega, and add the fractional rotation frequency as a
parameters, setting it equal to 0.8 on the fly.

>>> vega_pars = create.from_library('vega')
>>> print(vega_pars['rotperiod'])
0.60386473
>>> tools.add_rotfreqcrit(vega_pars,0.8)
>>> print(vega_pars['rotperiod'])
0.666071098099
>>> print(vega_pars['rotfreqcrit'])
0.8

Create a star of spectral type B9V, and add the surface gravity as a parameter.
Set the ``mass`` to be the dependent parameter.

>>> star_pars = create.star_from_spectral_type('B9V')
>>> print(star_pars['mass'],star_pars['radius'])
(6.002267, 3.5792469999999996)
>>> tools.add_surfgrav(star_pars,4.0,derive='mass')
>>> print(star_pars['mass'],star_pars['radius'])
(4.6695598707231305, 3.5792469999999996)
>>> print(star_pars['surfgrav'])
4.0

Section 3 Basic functionality
=============================

Creating Parameters and ParameterSets:

.. autosummary::

    parameters.Parameter
    parameters.ParameterSet
    parameters.Distribution
    datasets.LCDataSet
    datasets.RVDataSet
    datasets.SPDataSet
    datasets.IFDataSet


    
"""
from phoebe.parameters import create
from phoebe.parameters import tools
from phoebe.parameters import datasets

if __name__=="__main__":
    import doctest
    doctest.testmod()