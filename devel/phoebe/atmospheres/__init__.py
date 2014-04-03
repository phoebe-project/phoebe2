"""
Functionality  regarding stellar atmospheres or local intensities

**Modules**

.. autosummary::

    phoebe.atmospheres.limbdark
    phoebe.atmospheres.passbands
    phoebe.atmospheres.pulsations
    phoebe.atmospheres.magfield
    phoebe.atmospheres.velofield
    phoebe.atmospheres.reddening
    phoebe.atmospheres.roche
    phoebe.atmospheres.sed
    phoebe.atmospheres.spectra
    phoebe.atmospheres.spots
    phoebe.atmospheres.tools

**Quick reference**

Limb darkening functions:

.. autosummary::

    limbdark.ld_claret
    limbdark.ld_linear
    limbdark.ld_nonlinear
    limbdark.ld_logarithmic
    limbdark.ld_quadratic
    limbdark.ld_square_root
    limbdark.ld_uniform
    limbdark.ld_power
    
Specific intensities, synthetic fluxes and limbdarkening coefficients:

.. autosummary::
    
    limbdark.get_specific_intensities
    limbdark.get_limbdarkening
    limbdark.fit_law
    limbdark.interp_ld_coeffs
    sed.blackbody
    sed.synthetic_flux
    
Compute grids of LD coefficients:

.. autosummary::

    limbdark.compute_grid_ld_coeffs
    
Passband functionality:

.. autosummary::

    passbands.get_response
    passbands.list_response
    passbands.eff_wave
    passbands.get_info
    
Reddening laws:

.. autosummary::
    
    reddening.get_law
    reddening.redden
    reddening.deredden
    
Various tools:

.. autosummary::

   tools.gravitational_redshift
   tools.doppler_shift
   tools.rotational_broadening

"""