r"""

.. contents:: Table of Contents
   :depth: 1
   :local:

Section 1. Package structure
============================

**Main packages**: these are most commonly used to compute models and fit parameters.

.. autosummary::    

   phoebe.frontend
   phoebe.backend
   phoebe.parameters
   phoebe.wd
   
**Utility packages**: library of functions that Phoebe relies on, but that often can be used independently too.
   
.. autosummary::

   phoebe.algorithms
   phoebe.atmospheres
   phoebe.dynamics
   phoebe.io
   phoebe.units
   phoebe.utils
   
   
   
Section 2. Operation details
============================

.. contents:: Table of Contents
   :depth: 4
   :local:

Section 2.1 Limb darkening, local intensities and absolute fluxes
------------------------------------------------------------------

Three parameters are important for setting limbdarkening options and local intensity values:
    
    - ``atm``: sets source for values of the local intensities :ref:`(more info) <label-atm-lcdep-phoebe>`
    - ``ld_coeffs``: sets (source of) the limb darkening coefficients :ref:`(more info) <label-ld_coeffs-lcdep-phoebe>`
    - ``ld_func``: sets the limb darkening function :ref:`(more info) <label-ld_func-lcdep-phoebe>`

The physically most consistent way of working is by setting ``atm`` and ``ld_coeffs``
to some grid name (e.g. kurucz, blackbody). Then local intensities and limb
darkening coefficients will be interpolated to match the local conditions
(effective temperature, gravity) on the star. Default atmosphere is blackbody,
default limb darkening law is uniform.

Sometimes, you want to specify a set of limb darkening coefficients yourself. In 
that case, supply a list with coefficients to ``ld_coeffs``.

.. warning::

    In Phoebe2, this set of three parameters occur many times. First of all,
    the main body itself (parameterSets component and star) contain them. These
    represent the *bolometric* atmospheric properties, and are mainly relevant
    for reflection/heating effects.
    
    Furthermore, these three parameters also occur in *each* passband-dependable
    parameterSet (lcdep, rvdep...). The reason for this is optimal flexibility:
    you might want to use a different atmosphere table for UV data as for
    infrared data.
    
    So to recap: you have to specify bolometric properties of the atmospheres,
    but also the passband properties. Each time you add observables in a specific
    passband, set the ``atm``, ``ld_coeffs`` and ``ld_func`` as well!


See the :ref:`limbdark <limbdark-atmospheres>` module for more information and
:py:func:`compute_grid_ld_coeffs <phoebe.atmospheres.limbdark.compute_grid_ld_coeffs>`
for more info on how to compute limb darkening tables.

See the section on :ref:`file formats <limbdark-atmospheres-fileformats>` for
more information on the file structure of limb darkening tables.

The absolute scaling of the intensity calculations is in real physical units
(W/m3). It takes into account the distance to the target,
interstellar reddening effects and reflection/heating effects. Four parameters
govern the absolute scaling:

    - ``pblum`` and ``l3``: physical scaling via fixed passband luminosity and
       third light (live in the pbdep)
    - ``scale`` and ``offset``: instrumental scaling via a linear scaling factor
       and offset term (live in the obs)

Thus, there are two ways of scaling the model fluxes to observed fluxes:

   - **physical scaling**: each passband dependent parameterSet (lcdep...)
     contains a ``pblum`` parameter. When ``pblum=-1``, the parameter is
     ignored. Otherwise, the computed fluxes will be normalised to the passband
     luminosity of the component it belongs to: a spherical star at unit
     distance (1 Rsol) without extra effects (reddening, reflection...) with
     ``pblum=1`` will have observed fluxes of :math:`(4\pi)^{-1}`. Extra
     effects can decrease (reddening, distance) or increase (reflection) it.
     
     If only the first component has a specified ``pblum``, all other fluxes
     will be scaled **relative to the primary component in the system** (which
     is the first target in the system). In WD terms, this is dubbed "coupling
     of the luminosities". If you want to decouple the luminosity computations
     from the local atmospheric quantities for one component in the system, then
     set ``pblum`` for that component. Note that the value of ``pblum`` never
     changes: if you set it to ``pblum=-1``, the code will internally compute
     the passband luminosity but will not replace the -1 with that value.
     Everything happens internally.
     
     Third light ``l3`` also works on a component-to-component basis. Its units
     are the same as ``pblum``. If ``pblum=-1``, then ``l3`` is in absolute
     physical units (i.e. W/m3). Since ``l3`` is additive, specifying it for one
     component is the same as adding it to the total system. Adding it to two
     components means adding it twice to the system.
     
   - **instrumental scaling**: many observable parameterSet (lcobs...) have
     two parameters ``scale`` and ``offset``. When you set them to be adjustable
     but do not specify any prior, a :py:func:`linear fitting <phoebe.backend.universe.compute_scale_or_offset>` will be performed to match
     the observations with the model computations. This is useful for normalised
     data, or for data for which the scaling can then be interpreted as a
     distance scaling. The offset factor can then be interpreted as
     contributions from third light. You are free to fit for only one of
     ``scale`` and ``offset``, or both. Note that in this case the model fluxes
     are not altered in situ, but only rescaled when needed (e.g. when fitting
     or plotting).
     
     The units of ``offset`` are the units of the ``scale`` factor, because
     
     .. math::
     
        \mathtt{obs} = \mathtt{scale}*\mathtt{model} + \mathtt{offset}
        
     Thus if you normalised the observations to 1 and you allow automatic scaling
     ``offset`` will be fractional. If you normalised your observations to 100,
     then ``offset`` will be in percentage.
     
**Examples:**

We initialize a wide binary Bundle, set the inclination angle to 90 degrees and
set the radius of the secondary to be half the one of the primary. We'll take
uniform disks and blackbody atmospheres. We'll set the secondary luminosity to
zero such that we have a dark object eclipsing a star. In this case, the flux
ratio :math:`F_\mathrm{ecl}` during total eclipse is 75% of the total light, since

.. math::

    F_\mathrm{ecl} = \frac{\pi R_1^2 - \pi \left(\frac{R_1}{2}\right)^2}{\pi R_1^2} \\
                   =  1 - \frac{1}{4} = 0.75

::

    # System setup
    b = phoebe.Bundle()
    b['period'] = 20.0
    b['incl'] = 90.0
    b['pot@primary'] = phoebe.compute_pot_from(b, 0.05, component=0)
    b['pot@secondary'] = phoebe.compute_pot_from(b, 0.025, component=1)
    b.set_value_all('ld_func', 'uniform')
    b.set_value_all('ld_coeffs', [0.0])
    b.set_value_all('atm', 'blackbody')

    # Addition of default data
    b.lc_fromarrays(phase=np.linspace(0,1,100))
    b['pblum@lc01@secondary'] = 0.0

We compute the light curve and use the purely physical calculations as
observations. These can then be used to auto-scale the synthetics, with different
settings of the scaling parameters::

    b.run_compute()
    obsflux = b['flux@lc01@lcsyn']
                   
To show the behaviour of ``scale`` and ``offset`` and ``pblum`` and ``l3``, we'll
add 6 different light curves on top op the original one, with different values
of the scaling parameters. Unless stated otherwise, ``pblum=-1`` (absolute fluxes),
``l3=0`` (no third light), ``scale=1`` (no scaling) and ``offset=0`` (no offset).
In each case, we list the relative eclipse depth
and the minimum and maximum flux values. The different cases are:

- ``lc01``: all default values::
        
        syn = b['lc01@lcsyn']
        obs = b['lc01@lcobs']
        rel_depth = syn['flux'].min() / syn['flux'].max()
        minim, maxim = syn['flux'].min(), syn['flux'].max()
        assert(rel_depth==0.748343497631)
        assert(minim==1776.54192813)
        assert(maxim==2373.96587764)
   
  With the default parameters, we have fluxes between 1776 and 2373 W/m3, and
  thus a relative depth of 75%. Just as expected.
        
- ``lc02``: we force an offset of 10000 units (``offset=10000``). All fluxes
   will be offset by that amount, which means that the eclipse will be shallower::

        b.lc_fromarrays(phase=np.linspace(0,1,100), offset=10000)
        b['pblum@lc02@secondary'] = 0.0
        b.run_compute()
        
        assert(rel_depth==0.951719282612)
        assert(minim==11776.5419281)
        assert(maxim==12373.9658776)
        
- ``lc03``: Artifical addition of 10000 flux units to the observations as offset, and
  automatic scaling. This results in the same fluxes as ``lc02``, but the scaling
  and offset factors are determined automatically::

        b.lc_fromarrays(phase=np.linspace(0,1,100), flux=fluxobs+10000)
        b['pblum@lc03@secondary'] = 0.0
        b.set_adjust('scale@lc03')
        b.set_adjust('offset@lc03')
        b.run_compute()

        assert(rel_depth==0.951719282612)
        assert(minim==11776.5419281)
        assert(maxim==12373.9658776)
        
        assert(b['scale@lc02@lcobs']==1.0)
        assert(b['offset@lc02@lcobs']==10000.0)
        
- ``lc04``: Like ``lc03``, but with scaling of the fluxes as well::
        
        b.lc_fromarrays(phase=np.linspace(0,1,100), flux=0.75*fluxobs+10000)
        b['pblum@lc04@secondary'] = 0.0
        b.set_adjust('scale@lc04')
        b.set_adjust('offset@lc04')
        b.run_compute()

        assert(rel_depth==0.961965202198)
        assert(minim==11332.4064461)
        assert(maxim==11780.4744082)
        
        assert(b['scale@lc02@lcobs']==0.75)
        assert(b['offset@lc02@lcobs']==10000.0)

- ``lc05``: Manual passband luminosity ``pblum=12.56`` such that the total flux
  should be normalised to 1.0::

        b.lc_fromarrays(phase=np.linspace(0,1,100), flux=obsflux)
        b['pblum@lc05@primary'] = 4*np.pi
        b.run_compute()
        
        assert(rel_depth==0.748343497631)
        assert(minim==0.748606948021)
        assert(maxim==1.00035204474)        

- ``lc06``: Manual third light::

        b.lc_fromarrays(phase=np.linspace(0,1,100), flux=obsflux)
        b['l3@lc06@primary'] = 10000.0
        b.run_compute()
        
        assert(rel_depth==0.951719282612)
        assert(minim==11776.5419281)
        assert(maxim==12373.9658776)        
        
- ``lc07``: Manual third light and passband luminosity::

        b.lc_fromarrays(phase=np.linspace(0,1,100), flux=obsflux)
        b['l3@lc07@primary'] = 0.1
        b['pblum@lc07@primary'] = 4*np.pi
        b.run_compute()
        
        assert(rel_depth==0.771214041978)
        assert(minim==0.848606948021)
        assert(maxim==1.10035204474)     
        
        
                   

Section 2.2 Reflection and heating
-----------------------------------

You can switch on/off reflection and heating effects via ``refl=True`` and
``heating=True``. You can switch off the calculation of irradiation of objects
that are too dim or too cool, by setting the parameter
``irradiator=False`` in the Body parameterSet (star, component). Note that they
will still be irradiated (and can thus reflect light or heat up), but other
bodies will not receive their radiation. If you don't want a body to take part
in anything, you will need to separate it from the others (e.g. in a
disconnected BodyBag).

The following algorithms can be chosen to compute irradiation:

    - ``irradiation_alg='point_source'`` (fast): treat the irradiating body as a
      point source. This is generally fine if the separation between the objects
      is large enough.
    - ``irradiation_alg='full'`` (slow): treat the irradiating body as an
      extended body. Physically most correct.

The following parameters are important in the reflection and heating effects:

    - ``alb``: highly reflective surface have a high albedo (e.g. snow is close
      to unity). Absorbing surfaces have a low albedo (e.g. coal). A low albedo
      implies a high heating capacity :ref:`(more info) <label-alb-lcdep-phoebe>`
    - ``redist`` heat redistribution parameter. If zero, no heat will be
      redistribution. If unity, the heat redistribution is instantaneous.
      :ref:`(more info) <label-redist-star-phoebe>`
    - ``redisth``: sets the fraction of redistributed heat to be only along lines
      of constant latitude. :ref:`(more info) <label-redisth-star-phoebe>`

With the reflection effect, the ``scattering`` phase function can additionally be
specified. The current options are:

    - ``isotropic``: Thompson scattering, e.g. in atmospheres of hot stars
    - ``henyey``: Henyey-Greenstein scattering, for either front or backwards scattering
      :py:func:`(more info) <phoebe.algorithms.reflection.henyey_greenstein>`
    - ``henyey2``: Two-term Henyey-Greenstein scattering (Jupiter-like)
      :py:func:`(more info) <phoebe.algorithms.reflection.henyey_greenstein2>`
    - ``rayleigh``: back-front symmetric scattering
      :py:func:`(more info) <phoebe.algorithms.reflection.rayleigh>`
    - ``hapke``: for regolith surfaces, or even vegetation, snow...
      :py:func:`(more info) <phoebe.algorithms.reflection.hapke>`

.. warning::

    Just as for the atmospheric parameters, Phoebe2 distinguishes *bolometric*
    (part of star, component...) and *passband* albedos (part of lcdep, rvdep...).
    The bolometric albedo is used only to govern heating effects: with a bolometric
    albedo of 0.25, 75% of the incoming light will be used to heat the object,
    the remaining 25% will be reflected. The distribution over different passbands
    is set by the *passband* albedo. Note that the passband albedo sets the
    fraction of reflected versus incoming light *within the passband*. Thus,
    grey scattering means all the passband albedos are equal. A passband albedo
    of unity means no redistribution of wavelengths. A passband albedo exceeding
    unity means that light from other wavelengths is redistributed inside the
    current passband.
    
    Bolometric albedos need to be between 0 and 1, passband albedos can exceed 1.

See the :ref:`reflection <reflection-algorithms>` module for more information.
See :py:func:`generic_projected_intensity <phoebe.universe.generic_projected_intensity>`
for implementation details.

Section 2.3 Beaming
-----------------------------------

By default, beaming effects are not calculated. You can switch it on by setting
the ``beaming_alg`` accordingly (i.e. not equal to ``none``). If you decide that
the beaming contribution of a particular component is not worth it, you can set
``beaming=False`` in the Body parameterSet (star, component). There are four
options for the beaming algorithm:

    - ``beaming_alg='none'`` (immediate): no beaming is computed.
    - ``beaming_alg='full'`` (slowest): local intensities and limb darkening
      coefficients are consistently computed via velocity shifts in the original
      specific intensity grids
    - ``beaming_alg='local'`` (moderate): beaming is computed by setting local
      beaming factors on the star
    - ``beaming_alg='simple'`` (fast): beaming  is computed by setting a
      global beaming factor.

The latter two options are particularly useful for circular orbits, as the
beaming coefficients only need to be computed once.

Section 2.4 Morphological constraints
--------------------------------------

Morphological constraints are currently only implemented for binaries. There is
a parameter :ref:`morphology <label-morphology-component-phoebe>` in the
component ParameterSet for this purpose. It can take the following values:
    
    - ``morphology='unconstrained'``: has no effect
    - ``morphology='detached'``: the component must have a potential value above
      the critical value
    - ``morphology='semi-detached'``: the component must have a potential value
      equal to the critical value
    - ``morphology='overcontact'``: the component must have a potential value
      below the critical value

Note that these values are not strictly enforced, and in principle do not
interfere with any of the computations. However, a preprocessing step is added
using :py:func:`binary_morphology <phoebe.backend.processing.binary_morphology>`,
which adjusts the limits on the potential value parameter. Then it can be
easily checked if the potential value satisfies the constraint, e.g. during
fitting.


Section 2.5 Light time travel effects
--------------------------------------

Light time travel effects are off by default. You can enable them by setting
``ltt=True``. In this case, the orbit will be precomputed
:py:func:`(barycentric orbit) <phoebe.dynamics.keplerorbit.get_barycentric_orbit>`,
such that for any given barycentric time, we know what the proper time of the
components is. Note that the relative light time travel effects *between* the
components is not taken into account, i.e. reflection/heating effects are not
treated correctly.


Section 2.6 Eclipse detection
--------------------------------------

The eclipse detection algorithm is set via the parameter ``eclipse_alg``.
Depending on the circumstances, you can choose any of the following:

    - ``only_horizon``: simples, and assumes different bodies do not overlap
      at all. If the normal of a triangle is pointed in the observer's direction,
      it is assumed to be visible.
    - ``full``: checks if any triangle is eclipsed by any other. Can also detect
      self-eclipsing parts. Very slow.
    - ``graham``: Assumes bodies have convex shapes, and uses a Graham scan and
      fast "point in hull" algorithms to label eclipsed triangles.
    - ``binary`` (default): uses Graham scan algorithm (assuming convex bodies)
      to compute eclipses during expected eclipse times, and uses ``only_horizon``
      otherwise.

Section 2.7 Oversampling in time
--------------------------------------

If the exposure time of your observations is relatively long compared to the rate at which you
expect changes in the them to happen, you should oversample the computations in time. For example, 
if you want to compute a light curve taken at 60s cadence of a star that has an eclipse that lasts
only a few minutes, the ingress and egress of the eclipse will not correctly be modelled if you do
not oversample in time. 

Oversampling can be set using the parameters ``exptime`` (exposure time in seconds) and ``samprate``
(number of time points to be computed within one exposure). Both can be single values that will be
used for every time point, or arrays with the same length as the time array of the observations.
The latter can be useful to, e.g., use a higher oversampling during eclipses when the changes in 
a light curve are more rapid.

The oversampling is done as follows: say that you want to compute observables at time 100s, using an
integration time of 200s and with an oversampling of 4, then the observation started at 0s and ended
at 200s. As a result, you get the average value of the observables at times 25s, 75s, 125s and 
175s.

The functions in which the times extract times and refs and binoversampling are done under the hood
are :py:func:`(extract_times_and_refs) <phoebe.backend.observatory.extract_times_and_refs>` and
:py:func:`(bin_oversampling) <phoebe.backend.universe.Body.bin_oversampling>`.


Section 2.8 Interstellar reddening
--------------------------------------

Interstellar reddening can be taken into account in two ways:

1. Assuming a known reddening law (Cardelli, Fitzpatrick, Seaton...),
   described in the parameterSet :ref:`(reddening:interstellar) <parlabel-phoebe-reddening:interstellar>`,
   which is a global system parameterSet.
   The extinction law and parameters (e.g. :math:`R_v`) can then be set and the
   extinction at any wavelength can be derived from the extinction at a reference
   wavelenght :math:`A_\lambda`.
2. Using custom passband extinction parameters, which should be manually added
   and given in each pbdep (see. :py:func:`add_ebv <phoebe.parameters.tools.add_ebv>`)
   
See :py:func:`generic_projected_intensity <phoebe.backend.universe.generic_projected_intensity>`
for implementation details.

See :py:mod:`phoebe.atmospheres.reddening` for more information on reddening.

Section 2.10 Pulsations
--------------------------------------

There are three implementation schemes available for adding pulsations to a Sta
or BinaryRocheStar. It is important to note that none of them take into account
the effect of stellar deformation, only the Coriolis effects can be accounted
for. This means that pulsations are not consistently taken into account, but
doing better requires *a lot* more effort [Reese2009]_. We can hope that the current
approximations are an adequate approximation for many applications. The following
schemes are available (which can be set with the parameter ``scheme`` in the
:ref:`puls <parlabel-phoebe-puls>` parameterSet):

1. ``scheme='nonrotating'``: pulsations are pure spherical harmonics
2. ``scheme='coriolis'``: first order inclusion of Coriolis forces, according to
   [Schrijvers1997]_ and [Zima2006]_.
3. ``scheme='traditional_approximation``: higher order inclusion of Coriolis forces
   following [Townsend2003]_.
   
Full information can be find in the :py:mod:`pulsations <phoebe.atmospheres.pulsations>`
module, but here are some important notes:

* The input frequency ``freq`` is the frequency in the stellar frame of reference,
  which means that the observed frequency is:
  
  .. math::
        
         f_\mathrm{obs} = f_\mathrm{input} + m (1-C_{n\ell}) f_\mathrm{rot}
  
  With :math:`C_{n\ell}` the Ledoux coefficient ([Ledoux1952]_, [Hansen1978]_)
  and :math:`f_\mathrm{rot}` the stellar rotation frequency. For a Star, this
  translates in Phoebe2-parameter-names to:
  
  .. math::
    
    f_\mathrm{obs} = \mathtt{freq} + \mathtt{m} \frac{(1-\mathtt{ledoux\_coeff})}{\mathtt{rotperiod}}
  
  while a BinaryRocheStar has
  
  .. math::
    
    f_\mathrm{obs} = \mathtt{freq} + \mathtt{m} \frac{(1-\mathtt{ledoux\_coeff})*\mathtt{period}}{\mathtt{syncpar}}

* The flux perturbation is caused by three sources: the change in apparent
  radius, the change in local effective temperature and the change in local
  surface gravity [Townsend2003]_:
      
  .. math::
  
     \frac{\Delta F}{F} = \mathrm{Re}\left[\left\{\Delta_R \mathcal{R} + \Delta_T\mathcal{T} + \Delta_g\mathcal{G}\right\}\exp{2\pi i f t}\right]
  
  The three perturbation coefficients :math:`\Delta_R`, :math:`\Delta_T` and
  :math:`\Delta_g` are complex perturbation coefficients. In Phoebe, they are
  each given by two parameters:
  
  .. math::
    
    \Delta_R = \mathtt{ampl} \exp(2\pi i\mathtt{phase})
    
    \Delta_T = \mathtt{amplteff} \exp(2\pi i (\mathtt{phaseteff} + \mathtt{phase}))
    
    \Delta_g = \mathtt{amplgrav} \exp(2\pi i (\mathtt{phasegrav} + \mathtt{phase}))
  
  Thus, the phases for the temperature and gravity are relative to the phase
  of the radius perturbation. The amplitudes are all fractional.
      
      

Section 2.11 Magnetic fields
--------------------------------------

TBD

Section 2.12 Spots
--------------------------------------

TBD

Section 2.13 Potential shapes
--------------------------------------

TBD

Section 3. Data handling
======================================

3.1 Light curves
------------------

3.2 Radial velocity curves
-----------------------------

3.3 Multicolour photometry (SEDs)
------------------------------------

3.4 Spectra
-----------------

3.5 Spectrapolarimetry
------------------------------------

3.6 Interferometry
------------------------------------

3,7 Speckle interferometry
------------------------------------

3.8 Astrometry
------------------------------------

3.9 Eclipse timing variations
------------------------------------
  

Section 4. Code structure
===============================

.. contents:: Table of Contents
   :depth: 4
   :local:

4.1 Introduction
-------------------

*Punchline: The Universe of Phoebe2 consists of fully opaque Bodies, represented by collections
of triangles (mesh) that contain all the information (by virtue of
Parameters) needed to replicate observations (velocities, intensities, positions, etc).*

Each :py:class:`Body <phoebe.backend.universe.Body>` keeps track of all the
information it needs to put itself at its location given a certain time. Bodies
can be collected in super-Bodies, also named
:py:class:`BodyBags <phoebe.backend.universe.BodyBag>`. Each Body in a BodyBag
keeps it independence (and can thus always be taken out of the BodyBag), but
BodyBags can be manipulated (translated, rotated, evolved) as a whole.

Once Bodies are setup and put in place, they can also interact through irradiation.
Separate functions are responsible for adapting the local properties given such
processes.

4.2 Code hierarchy
---------------------

The basic building block of Phoebe2 is a :py:class:`Parameter <phoebe.parameters.parameters.Parameter>`.
Sets of parameters that are logically connected (i.e. have the same *context*) are collected into
an advanced dictionary-style class dubbed a :py:class:`ParameterSet <phoebe.parameters.parameters.ParameterSet>`.
ParameterSets have two flavors: one is the base ParameterSet class for ordinary
parameters, the other is the :py:class:`DataSet <phoebe.parameters.datasets.DataSet>`,
which provides extra functionality to treat observations or synthetic calculations.
On its turn , DataSets have many different flavors, one for each of the different
type of data (light curves, radial velocity curves, spectra etc).

ParameterSets and DataSets are organised in the :envvar:`params` attribute of a
Body. This attribute is a dictionary, where the context of the ParameterSet is
the key, and the value is the ParameterSet itself. If there can be more than
one ParameterSet of the same context (e.g. pulsations), then the value is a list
of ParameterSets. The only exception to this rule are the contexts that describe
data.

Each type of observations, whether it is a light curve or something else, is described
by three different contexts, of which two need to be user-supplied, and the third
one is automatically generated:
    - a *pbdep* (passband dependable, e.g. *lcdep*): this collects all information
      that the codes needs to simulate observations. This includes passbands,
      passband albedos, passband limb darkening coefficients, the atmosphere
      tables to use in this passband, passband scattering properties etc...
    - an *obs* (observation, e.g. *lcobs*): this collects all information on the
      data that is not contained in the pbdep: times of observations, observed
      fluxes, instrumental resolution, exposure times...
    - a *syn* (synthetic, e.g. *lcsyn*): this is a mirror of the obs, where
      instead of observed fluxes the model fluxes are stored.

Each type of context has a separate entry in the :envvar:`params` attribute.
One level deeper, each pbdep, obs or syn is is stored inside yet another
dictionary with keys lcdep, lcobs, lcsyn etc. Finally, each of these dictionaries
is again a dictionary where the key is the reference of the Parameterset (:envvar:`ref`)
and the value is the ParameterSet/DataSet itself.

Thus, we can recreate a minimal version (without any boiler plate code) of a
Body in the Phoebe universe like::

    class Body(object):
        
        def __init__(self, mybodyparams, lcdep, lcobs):
            
            # Every body should remember its own proper time
            self.time = None
            
            # Create an empty mesh with columns 'triangle' and 'size'
            self.mesh = np.zeros(0, dtype=[('triangle','f8',(9,)),('size','f8',1)])
            
            # Initiate the params attribute that contains all ParameterSets
            self.params = OrderedDict()
            self.params['pbdep'] = OrderedDict()
            self.params['obs'] = OrderedDict()
            self.params['syn'] = OrderedDict()
            
            # Fill in the mybodyparams ParameterSet
            self.params['mybodyparams'] = mybodyparams
            
            # Check if the lcdep and lcobs have the same reference
            assert(lcdep['ref'] == lcobs['ref'])
            
            # Fill in the lcdep
            self.params['pbdep']['lcdep'] = OrderedDict()
            self.params['pbdep']['lcdep'][lcdep['ref']] = lcdep
            
            # Fill in the lcobs
            self.params['obs']['lcobs'] = OrderedDict()
            self.params['obs']['lcobs'][lcobs['ref']] = lcobs
            
            # Prepare a synthetic dataset to fill in later
            lcsyn = datasets.LCDataSet(ref=lcobs['ref'])
            self.params['syn']['lcsyn'] = OrderedDict()
            self.params['syn']['lcsyn'][lcsyn['ref']] = lcsyn
            


4.3 Description of base classes
---------------------------------

The next sections contain more details on the most important base classes.

4.3.1 Parameter
~~~~~~~~~~~~~~~~~~~~~

A :py:class:`Parameter <phoebe.parameters.parameters.Parameter>` is a self-contained
representation of any value, switch or option that can be accessed or changed by
the user. The most important properties that a Parameter holds are:

- the parameter's name (called :envvar:`qualifier`)
- the parameter's data type (float, integer, string, array), which is actually a *caster* (see below)
- the parameter's value
- the parameter's unit (if applicable)
- the parameter's context (e.g. *orbit*, *component*..) and frame (always *phoebe* if you use predefined parameters)
- a boolean to mark it for inclusion in fitting procedures (:envvar:`adjust`)
- the parameter's prior and posterior (if applicable)

A list of all available setters and getters is given in the documentation of the
:py:class:`Parameter <phoebe.parameters.parameters.Parameter>` class itself.

Parameters don't need to implement all properties. For example, a filename can
be a Parameter, but it doesn't make sense to implement adjust flags or priors
and posteriors for a filename. It is the responsibility of the code that deals
with Parameters to treat them correctly. Some query-functionality exists to
facilitate such coding, like :py:func:`Parameter.has_prior <phoebe.parameters.parameters.Parameter.has_prior>`
or :py:func:`Parameter.has_unit <phoebe.parameters.parameters.Parameter.has_unit>`.
In other cases, the getters return enough information. For example for the
adjust flag, you can just query :py:func:`Parameter.get_adjust <phoebe.parameters.parameters.Parameter.get_adjust>`:
if there is an adjust flag, it will return :envvar:`True` or :envvar:`False`,
if it has no such flag, it will return :envvar:`None`.

By virtue of the data type, which is actually a function that casts a value to
the correct data type, the user does not need to worry about giving integers
or floats, and parsing code can immediately pass strings as parameter values,
making writing parsers really easy.

All the parameters and parameter properties used in the code are listed in the
:py:mod:`parameters.parameters.definitions` module.

**Simple examples**:

Minimal code to construct a parameter:

>>> par = phoebe.Parameter(qualifier='myparameter')
>>> print(par)

Initiating a predefined parameter is not particularly easy because they are
not ordered in any way. This is not at all a bad thing because you never need
to initiate a single Parameter (you'll always use ParameterSets, see below),
and if you do for some custom coding, you can
always cycle over all predefined parameters and extract those with the properties
you require. All predefined Parameters are contained in the following list:

>>> predefined_parameters = phoebe.parameters.definitions.defs

This is a list of dictionaries. For example if we take the first definition:

>>> print(predefined_parameters[0])
{'description': 'Common name of the binary', 'alias': ['phoebe_name'], 'frame': ['wd'], 'repr': '%s', 'value': 'mybinary', 'cast_type': <type 'str'>, 'context': 'root', 'qualifier': 'name'}

Thus, this contains all the arguments to create a Parameter:

>>> par = phoebe.Parameter(**predefined_parameters[0])
>>> print(par)

4.3.2 ParameterSet
~~~~~~~~~~~~~~~~~~~~~

A :py:class:`ParameterSet <phoebe.parameters.parameters.ParameterSet>` is a
collection of Parameters of the same context. `Context` is a general term, it
can mean physical entity (e.g. parameters necessary to describe the properties
of a Star -- mass, radius, effective temperature...), observations (time and
observed fluxes of a light curve...), or computational options (take into
account beaming, reflection...). ParameterSets are, from a user-point perspective,
the basic building blocks to build Bodies, add physics, add observations and
specify computational options. Because they are so vital and thus need to be
easy to create and modify, ParameterSets have the look and feel of a basic Python
object, the dictionary:

>>> ps = phoebe.ParameterSet('star')
>>> ps['mass'] = 5.0
>>> print(ps['teff'])
5777.0

The dictionary interface is an interface to the Parameter *values*, not the
Parameter *objects*. If you need to get the full Parameter object, you need
to get away from the dictionary-style interface and use the ParameterSet's
methods:

>>> ps.get_parameter('radius')
<phoebe.parameters.parameters.Parameter at 0x3603050>

You rarely need to access the Parameter objects themselves to change their properties,
because a ParameterSet implements similar functions as the Parameter class.
For example to change the `adjust` flag of the radius, you could do:

>>> ps.get_parameter('radius').set_adjust(True)

but this is equivalent to

>>> ps.set_adjust('radius', True)

Contexts are Phoebe's solution to keeping the parameter names short, unique and
stable against future changes in the code. For example, as long as we're working
with binaries, it is clear that ``incl`` means orbital inclination. Adding single
Stars to the mix would require us to all of a sudden introduce ``incl_star``, in
which case we would also need to change the original definition and make it
``incl_orbit`` to avoid ambiguities. There are many other occurrences of the
inclination angle, e.g. in magnetic fields, pulsations, misalignments... Putting
``incl`` in different contexts, allows us to keep intuitive parameter names (qualifiers)
while still allowing a great deal of flexibility.

Aside from a *context*, a ParameterSet also has a *frame*. However, the latter
is usually unimportant since in the case of Phoebe2, the frame is always ``phoebe``.
An example of an predefined frame is ``wd`` for Wilson-Devinney. the only reason
for its existence is that one can apply the Parameter philosophy also to other
codes. Defining the frame then solves possible ambiguities in predefined parameters.

4.3.3 Body
~~~~~~~~~~~~~~~~~~~~~

A :py:class:`Body <phoebe.backend.universe.Body>` is the base class for all
objects in the universe. Bodies can be combined in a BodyBag, which itself is
also a Body. The philosophy of the design of the Body is that it contains all
basic methods to access and change parameters and ParameterSets, and to perform
mesh computations like :py:func:`rotations and translations <phoebe.backend.universe.Body.rotate_and_translate>`,
computation of :py:func:`triangle sizes <phoebe.backend.universe.Body.compute_sizes>`,
:py:func:`surface area <phoebe.backend.universe.Body.area>`, etc...

A Body has three basic attributes:

    - ``params``: a (nested) dictionary holding all the parameterSets.
    - ``mesh``: a numpy record array containing all information on the mesh. The
      mesh can be *virtual* (i.e. there is a ``mesh`` attribute but it is created
      on-the-fly, see BodyBag) or *real* (i.e. there is a real array linked to
      the mesh property)
    - ``time``: a float that keeps track of the Body's proper time
    
Thus, a Body contains all information on the object plus the mesh.

4.3.4 PhysicalBody
~~~~~~~~~~~~~~~~~~~~~

A :py:class:`PhysicalBody <phoebe.backend.universe.PhysicalBody>` is only a
thin wrapper around the base :py:class:`Body <phoebe.backend.universe.Body>`,
and honestly the distinction is not always completely obvious. Basically, the
PhysicalBody is a base class for any single Body (a Star, AccretionDisk,
BinaryRocheStar) that is **not a BodyBag**. It contains all methods that are shared
by actual Bodies but not BodyBags. For example, any mesh manipulation method that
alters the shape or length of the mesh, is defined in the PhysicalBody:
e.g. :py:func:`subdivision <phoebe.backend.universe.PhysicalBody.subdivide>` or
addition of columns (:py:func:`prepare_reflection <phoebe.backend.universe.PhysicalBody.prepare_reflection>` etc...) 

A Body cannot implement mesh manipulation methods, because it is possible that
the mesh is not `owned` by the Body, i.e. when it is actually a BodyBag (see below).
In that case, it can only change values within an existing mesh, such as when
rotating and translating.

4.3.5 BodyBag
~~~~~~~~~~~~~~~~~~~~~

A :py:class:`BodyBag <phoebe.backend.universe.BodyBag>` is a container for Bodies
that is itself a Body. This design holds most of the power and flexibility of
Phoebe2. So listen up!

A BodyBag has one additional basic attribute with respect to a Body: a list of
Bodies under the attribute name ``BodyBag.bodies``. Thus a BodyBag has four
basic attributes:

    - ``params``: a (nested) dictionary holding all the parameterSets
    - ``mesh``: a virtual mesh, actually a shortcut to :py:func:`get_mesh <phoebe.backend.universe.BodyBag.get_method>`
    - ``time``: a float that keeps track of the Body's proper time
    - ``bodies``: a list of Bodies

The BodyBag has many
`virtual` methods and one `virtual` attribute. The term `virtual` here means
that that particular method or attribute is not defined within the BodyBag itself,
but is derived from those of the Bodies it contains. Take for example the mesh.
If you access the ``mesh`` attribute of a BodyBag as ``BodyBag.mesh``,
the method :py:func:``get_mesh <phoebe.backend.universe.BodyBag.get_mesh>`` or
:py:func:``set_mesh <phoebe.backend.universe.BodyBag.set_mesh>`` is called. This
method actually calls the meshes of the Bodies contained in the BodyBag, and
merges them together. Thus:

>>> mymesh = myBodyBag.mesh

is equivalent to

>>> mymesh = np.hstack([mybody.mesh for mybody in myBodyBag.bodies])

The virtual methods are created whenever a BodyBag does not implement a particular
function itself. If this is the case, the call is passed on to the Bodies inside
the BodyBag. For example, a BodyBag does not know how to set the temperature
of it's members, yet still you can call:

>>> myBodyBag.temperature()

which is then equivalent to

>>> for mybody in myBodyBag.bodies:
...     mybody.temperature()

This way, BodyBags can feel just like the PhysicalBodies it contains, and the user
or programmer doesn't need to care what exactly he or she is dealing with in
most occassions. The return value will be a list of return values from the 
individual calls to the Bodies. There is one little caveat here: if a member of a BodyBag does
not implement the particular method, it is silently ignored. This also means
that if the method doesn't exist anywhere, it will raise not AttributeError!
The following function will thus pass silently:

>>> myBodyBag.i_am_pretty_sure_this_function_does_not_exist()

So beware of typos! This behaviour requires some discipline of the programmer.
If at any point this behaviour is not wanted anymore, one should change the
(short) :py:class:`CallInstruct <phoebe.backend.universe.CallInstruct>` code.

4.4 Filling the mesh / setting the time
------------------------------------------

The mesh contains the physical properties of every triangle in a Body, that are
needed to compute observables. The mesh is a numpy record array, where each
record (row) represents one triangle. Each column represents a physical property.
Any column in the mesh can be accessed as

>>> mybody.mesh['triangle']

The basic columns are:

- ``triangle``: an Nx9 array containing the coordinates of all vertices (:math:`x_1`, 
  :math:`y_1`, :math:`z_1`, :math:`x_2`, :math:`y_2`, :math:`z_2`, :math:`x_3`,
  :math:`y_3`, :math:`z_3`) (units: :math:`R_\odot`)
- ``center``: an Nx3 array containing the coordinates of the center of each
  triangle (units: :math:`R_\odot`)
- ``size``: an array of length N containing the sizes of each triangle
  (units: :math:`R_\odot^2`)
- ``normal_``: an Nx3 array containing the normal vectors on each triangle
- ``mu``: an array of length N containing the cosine of the angle between
   the line-of-sight and the the normal
- ``hidden``: an array of length N containing boolean flags to mark
  triangles as hidden from the user (eclipsed)
- ``visible`` an array of length N containing boolean flags to mark
  triangles as visible to the user
- ``partial`` an array of length N containing boolean flags to mark
  triangles as partially hidden from the user
- ``velo___bol_``: an Nx3 array containing the velocity vectors of each
  triangle (units: :math:`R_\odot/d`)
- ``ld___bol``: an Nxd array containing the limb darkening coefficients and
  normal emergent intensities of each triangle (units: :math:`W/m^3/sr`).
  The parameter :math:`d` is the number of limb darkening coefficients + 1.
  By default, :math:`d=4` which is sufficient for almost all limb darkening
  laws.

Vector quantities have a trailing underscore ``_``, which is important for the
rotation functions (vectors are translated/rotated differently from coordinates).

Building upon the minimal version of the Body class, we can add some of the
features discussed above::

    from algorithms import marching
    class PhysicalBody(Body):
    
    
        def fix_mesh(self):
            
            # Hold a list of all dtypes of the columns to add
            add_columns_dtype = []
            
            # walk over all necessary ld-columns
            for deptype in self.params['pbdep']:
                # deptype takes values 'lcdep', 'rvdep'...
                for dep in self.params['pbdep'][deptype]:
                    # dep takes as values the references of the pbdeps
                    ref = self.params['pbdep'][deptype][dep]['ref']
                    col = 'ld_' + ref
                    
                    # Remember the column if it is missing
                    if not col in self.mesh.dtype.names:
                        add_columns_dtype.append([col, 'f8', (5,)])
            
            # Define all the dtypes in the mesh, i.e. the existing and missing
            # ones
            dtypes = np.dtype(self.mesh.dtype.descr + add_columns_dtype)
            
            # Now add the missing columns to the mesh, and reset it
            self.mesh = np.zeros(N, dtype=dtypes)
            
        
        def compute_mesh(self, time=None):
            mesh = marching.cdiscretize(0.1, 100000, *self.subdivision['mesh_args'][:-1])
        
        def surface_gravity(self, time=None):
            self.mesh['logg'] = 4.4
        
        def temperature(self, time=None):
            freq = self.params['mybodyparams']['freq']
            self.mesh['teff'] = 5777.0 + 500.0*sin(2*pi*freq*time)
            
        def set_time(self, time):
            
            # Only compute the mesh if not done before, and also only check
            # if the mesh has the correct columns if the time has not set
            # before (here we do not allow adding data in the middle of
            # the computations)
            if self.time = None:
                self.compute_mesh(time)
                self.fix_mesh()
                
                            
            
            # 
                
            
            # Keep track of the time
            self.time = time
            


4.5 Synthesizing data
------------------------

Data computations involve a lot of steps and prerequisites. In principle, any
type of data follows the same pattern of computations, perhaps with slight
deviations.

4.5.1 Prerequisites
~~~~~~~~~~~~~~~~~~~~~

Before data can by synthesized, the Body or BodyBag needs to exist, need to be
set to a particular time and interactions (like reflection effects) need to be
computed. Additionally, observations need to be present. These observations are
used as a template to mirror observations (obs) in the synthetic datasets (syn).
Thus, the prerequisites are:

* Body needs to have a fully computed mesh (set to a time)
* Body needs to have observations attached

4.5.2 Synthesis
~~~~~~~~~~~~~~~~~~~~~

The base :py:class:`Body <phoebe.backend.universe.Body>` and 
:py:class:`Body <phoebe.backend.universe.PhysicalBody>` classes implement all
types of synthesis methods. The method names are the *category* of the observations:
for light curves, with datasets *lcobs*, the method name or category is *lc*.
For radial velocity curves (*rvobs*) it's *rv*, for spectra (*spobs*) it's *sp* etc..
There can be exceptions for historical reasons: the developers decided to give
interferometry (*ifobs*) the category *if*, but since that is reserved keyword
in Python, it cannot be the name of a function. Thus, the interferometry function
is called *ifm*.

The responsibility of the *category* method, which is a property of a Body, is
nothing more than calling the correct function defined in the :py:mod:`observatory <phoebe.backend.observatory>` with the correct arguments. The parameter parsing can
be a little different for BodyBags or Bodies, since sometimes the BodyBag itself
can be directly used, but sometimes the final data need to be synthesized from
the separate Bodies first. Here are two examples:

1. for light curves (:py:func:`lc <phoebe.backend.universe.PhysicalBody.lc>`), all we
   need to know are intensities and we don't care if the Body consists of
   multiple Bodies or not. We can just call a Body's or
   BodyBag's :py:func:`projected_intensity <phoebe.backend.universe.Star.projected_intensity>`,
   and we get the projected integratd flux.
2. for interferometry (:py:func:`lc <phoebe.backend.universe.Body.ifm>`), we
   cannot just create an image of a BodyBag and Fourier transform it: in the case
   of a very wide binary, it might be that the star covers only one pixel, unless
   we make an unrealistically large image consisting of nothing but blackness.
   A solution is here to descend into the BodyBag, compute visibilities of
   the separate objects in the BodyBag, and combine the interferometry afterwards
   (see doc of :py:class:`IFDataSet <phoebe.parameters.datasets.IFDataSet>`).


4.6 Recipes
-----------------

4.6.1 How to implement a new Body
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4.6.2 How to implement a new type of observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4.6.3 How to add physics
~~~~~~~~~~~~~~~~~~~~~~~~~~

"""

import os

# People shouldn't import Phoebe from the installation directory (inspired upon
# pymc warning message).
if os.getcwd().find(os.path.abspath(os.path.split(os.path.split(__file__)[0])[0]))>-1:
    # We have a clash of package name with the standard library: we implement an
    # "io" module and also they do. This means that you can import Phoebe from its
    # main source tree; then there is no difference between io from here and io
    # from the standard library. Thus, if the user loads the package from here
    # it will never work. Instead of letting Python raise the io clash (which
    # is uniformative to the unexperienced user), we raise the importError here
    # with a helpful error message
    if os.getcwd() == os.path.abspath(os.path.dirname(__file__)):
        raise ImportError('\n\tYou cannot import Phoebe from inside its main source tree.\n')
    # Anywhere else the source tree, it should be possible to import Phoebe.
    # However, it is still not advised to do that.
    else:
        print('\n\tWarning: you are importing Phoebe from inside its source tree.\n')
    

from ._version import __version__
#-- make some important classes available in the root:
from .parameters.parameters import ParameterSet,Parameter
from .parameters.parameters import ParameterSet as PS
from .parameters.datasets import DataSet,LCDataSet,IFDataSet,SPDataSet,RVDataSet
from .backend.universe import Star,BinaryRocheStar,MisalignedBinaryRocheStar,\
                              BinaryStar,BodyBag,BinaryBag,AccretionDisk,\
                              PulsatingBinaryRocheStar
from .frontend.bundle import Bundle, load, info
from .frontend.common import take_orbit_from, compute_pot_from, compute_mass_from

#-- common input and output
from .parameters.parameters import load as load_ps
from .backend.universe import load as load_body
from .backend import office
from .parameters.datasets import parse_lc,parse_phot,parse_rv,\
                                 parse_etv,parse_spec_as_lprof,parse_vis2
from .parameters import create
from .utils.utils import get_basic_logger
from phoebe.io import parsers

#-- common tasks
from .parameters.tools import add_rotfreqcrit
from .backend.observatory import compute,observe,add_bitmap,image,ifm
from .backend.fitting import run
from .atmospheres.limbdark import download_atm

#-- common modules
from .backend import observatory,universe,plotting,fitting
from .wd import wd

#-- common extras
from .units import constants
from .units.conversions import convert,Unit

