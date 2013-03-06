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

    datasets.parse_phot
    datasets.parse_spec_as_lprof
    
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