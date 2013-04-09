"""
Core interface of Phoebe

**Template class**: lists all the stuff you can do with a Body.

.. autosummary::

    universe.PhysicalBody

**Creating Bodies**

.. autosummary::
    
    universe.Star
    universe.BinaryRocheStar
    universe.BinaryStar
    universe.AccretionDisk
    universe.BinaryBag
    universe.BodyBag

**Computing observables**

.. autosummary::

    observatory.compute
    observatory.observe
    observatory.image
    observatory.make_spectrum
    observatory.stokes
    observatory.ifm

Other observables such as ``lc`` and ``rv`` are computed directly from the Body.

**Fitting parameters**

.. autosummary::
    
    fitting.run
    fitting.accept_fit
    fitting.summarize_fit
    
**Plotting data and synthetic calculations**

.. autosummary::

    plotting.plot_lcdeps_as_sed
    plotting.plot_spdep_as_profile
    plotting.plot_ifdep

"""