"""
Functionality  regarding stellar atmospheres or local intensities


Limb darkening functions:

.. autosummary::

    limbdark.ld_claret
    limbdark.ld_linear
    limbdark.ld_nonlinear
    limbdark.ld_logarithmic
    limbdark.ld_quadratic
    limbdark.ld_uniform
    limbdark.ld_power
    
Specific intensities and limbdarkening coefficients:

.. autosummary::
    
    limbdark.get_specific_intensities
    limbdark.get_limbdarkening
    limbdark.fit_law
    limbdark.interp_ld_coeffs
    
Compute grids of LD coefficients:

.. autosummary::

    limbdark.compute_grid_ld_coeffs
"""