

Getting Started
==================================

The beta release of PHOEBE 2.0 aims to provide fully-tested functionality that
matches that of the legacy PHOEBE 1.0 (light curve and radial velocity forward
models of binary star systems) but with improved precision and a python
interface.

That said, PHOEBE 2.0 can not yet handle:

* overcontact systems
* spots
* misaligned binaries
* TODO: anything else?

More advanced physics can be found in the PHOEBE 2.0-alpha releases
and will be ported to beta as soon as they can be tested robustly.


Download and Installation
===============================

Download
--------------------------------

This version of phoebe is currently in the devel branch of the SVN.  Once released,
it will be packaged for installation via pip, but for now requires manually
downloading through the SVN and installing.

To download the SVN branch anonymously:

::

   svn checkout svn://svn.code.sf.net/p/phoebe/code/devel/ phoebe-code



Or to download via your sourceforge account (for commit permissions), go to the
following link while logged in to sourceforge:
https://sourceforge.net/p/phoebe/code/HEAD/tree/devel/


Dependencies
--------------------------------

PHOEBE requires python with the following packages:

* numpy (may need 1.10+)
* scipy
* astropy (for units, may fork in the future to remove this dependency - needs version 1+)

And suggested packages (required for some optional but commonly used features):

* matplotlib (suggested plotting)
* sympy (for safer and more flexible constraints)

And optional packages (used for less commonly used features):

* mpld3 (alternate plotting)
* bokeh (alternate plotting)

Installation
-------------------------------

NOTE: the beta version now builds to a python module named 'phoebe' which may
conflict with the alpha version if you have that installed.


To install without admin rights for a single-user:

::

	python setup.py build
	python setup.py install --user



or to install system-wide with admin rights:

::

	python setup.py build
	sudo python setup.py install


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
   Fitting<tutorials/fitting>


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
   Advanced: Alternate Plotting Backends<tutorials/alternate_plotting>
   Advanced: Alternate Backends<tutorials/alternate_backends>
   Advanced: Detaching from Run Compute<tutorials/detach>
   Advanced: Digging into the Backend<tutorials/backend>
   Advanced (coming soon): Creating Custom Parameters<tutorials/custom_parameters>
   Advanced (coming soon): Creating Custom Constraints<tutorials/constraint_create>
   Advanced (coming soon): Time Derivatives<tutorials/time_derivatives>
   Advanced (coming soon): Undo/Redo<tutorials/undo_redo>



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
   Orbits (ORB)<tutorials/ORB>
   Meshes (MESH)<tutorials/MESH>
   Light Curves (LC)<tutorials/LC>
   Radial Velocities (RV)<tutorials/RV>
   Eclipse Timing Variations (ETV)<tutorials/ETV>


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
   Eccentricity (Volume Conservation) (not yet full-implemented)<tutorials/ecc>
   Apsidal Motion<tutorials/apsidal_motion>
   Systemic Velocity<tutorials/vgamma>
   Passband Luminosity<tutorials/pblum>
   Third Light<tutorials/l3>
   Distance<tutorials/distance>
   Limb Darkening (not yet fully-implemented)<tutorials/limb_darkening>
   Gravitational Redshift (RVs)<tutorials/grav_redshift>
   Reddening and Extinction (not yet implemented)<tutorials/reddening_extinction>
   Reflection and Heating (not yet implemented)<tutorials/reflection_heating>
   Beaming and Boosting (not yet implemented)<tutorials/beaming_boosting>

COMING SOON (differences between various t0s and phasing)


Example Scripts
===============================

These example scripts are generally focussed to show a single advanced feature
or a specific science use-case.  They are generally less verbose than the tutorials
and assume you're comfortable with the general concepts and syntax of both
Python and PHOEBE.


Single Stars
------------------------------

COMING SOON - not yet supported


Binary Stars
------------------------------


.. toctree::
   :maxdepth: 1
   :titlesonly:

   Minimal Example to Produce a Synthetic Light Curve<examples/minimal_synthetic>
   Complete Binary Animation<examples/animation_binary_complete>
   Minimal Overcontact System<examples/minimal_overcontact>
   Rossiter-McLaughlin Effect (RVs)<examples/rossiter_mclaughlin>
   Wilson-Devinney Style Meshing<examples/mesh_wd>


COMING SOON (examples from literature)


Triple Stars
-------------------------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Minimal Hierarchical Triple (TESTING - not yet supported)<examples/hierarchical_triple>
   Hierarchical Triple vs Photodynam (TESTING - not yet supported) <examples/hierarchical_triple_pd>
   LTTE ETVs in a Hierarchical Triple (TESTING - not yet supported) <examples/hierarchical_triple_etvs>


COMING SOON (examples from literature)


Advanced Constraints
------------------------------

COMING SOON (creating custom constraints, main-sequence, etc)


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
   Comparing PHOEBE 2.0 vs Photodynam (Binary)<examples/photodynam>
   Comparing PHOEBE 2.0 vs Photodynam (Hierarchical Triple)<examples/hierarchical_triple>
   Comparing PHOEBE 2.0 vs JKTEBOP <examples/jktebop>


Frontend API Docs
===============================

.. toctree::
   :maxdepth: 2

   Frontend <api/phoebe.frontend>
   Parameters <api/phoebe.parameters>


Backend (Advanced) API Docs
===============================

.. toctree::
   :maxdepth: 2

   Backend <api/phoebe.backend>
   Atmospheres <api/phoebe.atmospheres>
   Constraints <api/phoebe.constraints>
   Dynamics <api/phoebe.dynamics>
   Distortions <api/phoebe.distortions>


Development Information
================================

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Style Guidelines (coming soon)<development/style>
   Tutorials and Scripts<development/tutorials_scripts>
   API Documentation (coming soon)<development/api>
   Testing (coming soon)<development/testing>
   Benchmarking (coming soon)<development/benchmark>
   Committing Code (coming soon)<development/committing>
   Releasing a New Version (coming soon)<development/release>


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
