"""
Creating Parameters and ParameterSets:

.. autosummary::

    parameters.Parameter
    parameters.ParameterSet
    datasets.LCDataSet
    datasets.RVDataSet
    datasets.SPDataSet
    datasets.IFDataSet
    
Parsing ASCII files:

.. autosummary::
    
    datasets.parse_lc
    datasets.parse_rv
    datasets.parse_phot
    datasets.parse_vis
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
    
"""