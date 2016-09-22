

Getting Started
==================================

The `PHOEBE 2.0-beta release <https://github.com/phoebe-project/phoebe2/releases/tag/2.0b>`_
aims to provide fully-tested functionality that
matches that of the legacy PHOEBE 1.0 (light curve and radial velocity forward
models of binary star systems) but with improved precision and a python
interface.

Although we have attempted to test as thoroughly as possible, please continue
to be skeptical of all results and
`report any issues or bugs <https://github.com/phoebe-project/phoebe2/issues>`_.

See below for installation instructions as well as a listing of all available
tutorials and example scripts.

Supported Physics (from PHOEBE 1.0)
----------------------------------------

* detached roche binaries
* keplerian orbits (including eccentric orbits with volume conservation)
* passbands/atmospheres
* limb-darkening
* gravity brightening
* reflection (heating without redistribution)
* finite integration time via oversampling
* circular spots

New Physics (not in PHOEBE 1.0)
----------------------------------------

* beaming/boosting
* single rotating stars


Unsupported Physics (from PHOEBE 1.0)
----------------------------------------

PHOEBE 2.0 can not yet handle:

* overcontact systems (in development)
* semi-detached/contact systems (planned future development)
* X-ray binaries


Unsupported Convenience Functionality
-----------------------------------------

* fitting (planned future development)
* GUI (in development)
* data in magnitudes (dropping support - convert manually)
* data in phases (dropping support - but function provided to convert during import)
* phase shifts


Planned Physics Support
------------------------------------------
More advanced physics can be found in the PHOEBE 2.0-alpha releases
and will be ported to beta as soon as they can be tested robustly.

Planned (new) future features include:

* heating (heat redistribution)
* triple and N-body systems (in development)
* N-body dynamics (in development)
* misaligned binaries
* pulsations (in development)
* bayesian (MCMC) fitting
* synthetic spectra
* synthetic eclipse timing variations (ETVs) (in development)
* synthetic interferometry


Download and Installation
===============================

Download
--------------------------------

PHOEBE 2.0 is now hosted on GitHub.  Once released, it will be packaged for installation
via pip, but for now requires manually downloading and installing through
SVN or git.

To download via the `github repository <https://github.com/phoebe-project/phoebe2/>`_:

::

   git clone https://github.com/phoebe-project/phoebe2.git


Dependencies
--------------------------------

Note for **mac users**: it is suggested to use `homebrew to install a parallel version
of python <https://joernhees.de/blog/2014/02/25/scientific-python-on-mac-os-x-10-9-with-homebrew/>`_.
PHOEBE has currently been tested to compile correctly using homebrew on El Capitan.

PHOEBE requires python 2.7+ (not yet fully tested on python 3.x) with the following packages:

* numpy (may need 1.10+)
* scipy
* astropy (1.0+)

And suggested packages (required for some optional but commonly used features):

* matplotlib (suggested for plotting)
* sympy (for safer and more flexible constraints)

And optional packages (used for less commonly used features):

* mpld3 (alternate plotting - devel version only)
* bokeh (alternate plotting - devel version only)



Installation
-------------------------------

NOTE: the beta version now builds to a python module named 'phoebe' which may
conflict with the alpha version if you have that installed (but will not
conflict with PHOEBE 0.2x, 0.3x, or 1.0).  If you do have PHOEBE 2.0-alpha
installed, please uninstall before attempting to install PHOEBE 2.0-beta.


To install without admin rights for a single-user:

::

   python setup.py build
   python setup.py install --user



or to install system-wide with admin rights:

::

   python setup.py build
   sudo python setup.py install


Testing
--------------------------------

To run all tests locally on your machine, run the following in the 'tests'
directory in the source.  Note that you may need to enabled development
mode to run all the tests (currently detected by the presence of a file
in ~/.phoebe_devel_enabled)

::

   python run_tests nosetests


Please `report any issues or bugs <https://github.com/phoebe-project/phoebe2/issues>`_.


Tutorials
===============================

Each of the following tutorials builds upon previous tutorials, so it will be
most efficient to work through them sequentially at first.  However, each should
run independently, so feel free to jump in at any point to review a specific
concept.

For more specific use-cases, see the example scripts below.

Any of these tutorials can be downloaded as an IPython Notebook or a python script.
(see the link at the top of any tutorial).  To run these you'll need PHOEBE
installed on your system as well, and for the IPython notebooks you'll also need
IPython (sudo pip install jupyter; sudo apt-get install ipython-notebook).
Then simply start the notebook service (ipython notebook [downloaded_tutorial.ipynb]).
This will allow you to interact with the tutorial - running it line-by-line
and making alterations to see how they change the output.

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :numbered:

   General Concepts<tutorials/general_concepts>
   Building a System<tutorials/building_a_system>
   Saving and Loading<tutorials/saving_and_loading>
   Constraints<tutorials/constraints>
   Datasets<tutorials/datasets>
   Computing Observables<tutorials/compute>
   Plotting<tutorials/plotting>
   Accessing and Plotting Meshes<tutorials/meshes>


Advanced Tutorials
=======================

The following set of advanced tutorials follow the same format as the tutorials
above, but cover individual advanced topics and do not depend on each other.

These all assume comfort with the tutorials listed above, but should not need to
be read in any particular order.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Advanced: Settings<tutorials/settings>
   Advanced: Animations<tutorials/animations>
   Advanced: Alternate Backends<tutorials/alternate_backends>
   Advanced: Digging into the Backend<tutorials/backend>



Datasets and Observables
===============================

The following tutorials aim to both explain the general logic that PHOEBE
uses to compute observables as well as explaining the parameters, fields, and options
for each observable type.

These aim to be quite thorough and may not be best for light-reading.  They
expect a comfortable understanding of using PHOEBE and python

.. toctree::
   :maxdepth: 1
   :titlesonly:

   How does PHOEBE compute observables<tutorials/phoebe_logic>
   Orbits (orb)<tutorials/ORB>
   Meshes (mesh)<tutorials/MESH>
   Light Curves (lc)<tutorials/LC>
   Radial Velocities (rv)<tutorials/RV>


Explanations of Individual Parameters
========================================

The following tutorials aim to explain the implementation and usage of
some of the physical effects that are incorporated in PHOEBE.  These explain
the relevant parameters and try to demonstrate how they affect the resulting
synthetic models, but expect a comfortable understanding of using PHOEBE and python

.. toctree::
   :maxdepth: 1
   :titlesonly:


   Potentials<tutorials/pot>
   Eccentricity (Volume Conservation)<tutorials/ecc>
   Apsidal Motion<tutorials/apsidal_motion>
   Systemic Velocity<tutorials/vgamma>
   Passband Luminosity<tutorials/pblum>
   Third Light<tutorials/l3>
   Distance<tutorials/distance>
   Limb Darkening (not yet fully-implemented)<tutorials/limb_darkening>
   Gravitational Redshift (RVs)<tutorials/grav_redshift>
   Reflection and Heating<tutorials/reflection_heating>
   Beaming and Boosting<tutorials/beaming_boosting>
   Eclipse Detection<tutorials/eclipse>
   Intensity Weighting<tutorials/intens_weighting>


Example Scripts
===============================

These example scripts are generally focussed to show a single advanced feature
or a specific science use-case.  They are generally less verbose than the tutorials
and assume you're comfortable with the general concepts and syntax of both
Python and PHOEBE.  Some scripts may be listed under different sections if they
fall under multiple categories.


Single Stars
------------------------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Sun (rotating single star)<examples/sun>



Detached Binary Stars
------------------------------


.. toctree::
   :maxdepth: 1
   :titlesonly:

   Minimal Example to Produce a Synthetic Light Curve<examples/minimal_synthetic>
   Complete Binary Animation<examples/animation_binary_complete>
   Rossiter-McLaughlin Effect (RVs)<examples/rossiter_mclaughlin>
   Wilson-Devinney Style Meshing<examples/mesh_wd>
   Detached Binary: Roche vs Rotstar<examples/detached_rotstar>
   Binary with Spots<examples/binary_spots>



Spots
------------------------------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Binary with Spots<examples/binary_spots>
   Comparing Spots in PHOEBE 2.0 vs PHOEBE Legacy<examples/legacy_spots>



Advanced Plotting
------------------------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Complete Binary Animation<examples/animation_binary_complete>



Alternate Backends
------------------------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Comparing PHOEBE 2.0 vs PHOEBE Legacy<examples/legacy>
   Comparing Spots in PHOEBE 2.0 vs PHOEBE Legacy<examples/legacy_spots>



Citing PHOEBE 2.0
================================

Once the paper has undergone the refereeing process and is accepted, we will
release the official 2.0 version of PHOEBE.  Until then, please consider
citing the arXiv release of the paper.

FAQ
================================


*Q: Is PHOEBE 2.0 beta backwards compatible with PHOEBE 2.0 alpha releases?*

A: Unfortunately, no.  We simply learned too much from the alpha-release that
we decided that a complete rewrite was needed.  However, many of the syntax
concepts should be very familiar if you've used the frontend in the alpha releases.

*Q: Can I speed up plotting in any way?*

A: You could try changing your backend, e.g via ``matplotlib.rcParams['backend'] = 'Agg'``
but do this before importing Phoebe.

*Q: How do I add a custom passband to PHOEBE 2?*

A: You will need a table of intensities that you can download from the PHOEBE homepage.
Then you should follow the instructions available :class:`phoebe.atmospheres.passbands.Passband`

*Q: Is Phoebe 2.x Python 3.x ready?*

A: Some effort has been done to make Phoebe 2.x Python 3.x compliant. In fact,
Phoebe should load when imported in Python 3.x. The essential dependencies
(i.e. numpy, scipy, matplotlib, pyfits...) are Python 3.x ready, as are most of
the nonessential dependencies. Syntactically, Also the C extensions are not
Python 3.x compatible as of yet.

*Q: Is it safe to use Phoebe?*

A: For the most part, yes.  If you do not have sympy installed, then constraints
will be evaluated using the 'eval' command - which could potentially be dangerous
if you blindly open a bundle from an untrusted source.  So long as you have
