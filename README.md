PHOEBE 2.3
------------------------

<p align="center"><a href="http://phoebe-project.org"><img src="./images/logo_blue.svg" alt="PHOEBE logo" width="160px" align="center"/></a></p>

<pre align="center" style="text-align:center; font-family:monospace; margin: 30px">
  pip install phoebe
</pre>

<p align="center">
  <a href="https://pypi.org/project/phoebe/"><img src="https://img.shields.io/badge/pip-phoebe-blue.svg"/></a>
  <a href="http://phoebe-project.org/install"><img src="https://img.shields.io/badge/python-3.6+-blue.svg"/></a>
  <a href="https://github.com/phoebe-project/phoebe2/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-GPL3-blue.svg"/></a>
  <a href="https://travis-ci.org/phoebe-project/phoebe2"><img src="https://travis-ci.org/phoebe-project/phoebe2.svg?branch=master"/></a>
  <a href="http://phoebe-project.org/docs"><img src="https://img.shields.io/badge/docs-passing-success.svg"/></a>
<br/>
  <a href="https://ui.adsabs.harvard.edu/abs/2016ApJS..227...29P"><img src="https://img.shields.io/badge/ApJS-Prsa+2016-lightgrey.svg"/></a>
  <a href="https://ui.adsabs.harvard.edu/abs/2018ApJS..237...26H"><img src="https://img.shields.io/badge/ApJS-Horvat+2018-lightgrey.svg"/></a>
  <a href="https://ui.adsabs.harvard.edu/abs/2020ApJS..247...63J"><img src="https://img.shields.io/badge/ApJS-Jones+2020-lightgrey.svg"/></a>
  <a href="https://ui.adsabs.harvard.edu/abs/2020arXiv200616951C"><img src="https://img.shields.io/badge/ApJS-Conroy+2020-lightgrey.svg"/></a>
</p>

<p align="center">
  <a href="http://phoebe-project.org"><img src="./images/console.gif" alt="Console Animation" width="600px" align="center"/></a>
</p>


INTRODUCTION
------------

PHOEBE stands for PHysics Of Eclipsing BinariEs. PHOEBE is pronounced [fee-bee](https://www.merriam-webster.com/dictionary/phoebe?pronunciation&lang=en_us&file=phoebe01.wav).

PHOEBE 2 is a rewrite of the original PHOEBE code. For most up-to-date information please refer to the PHOEBE project webpage: [http://phoebe-project.org](http://phoebe-project.org)

PHOEBE 2.0 is described by the release paper published in the Astrophysical Journal Supplement, [Prša et al. (2016, ApJS 227, 29)](https://ui.adsabs.harvard.edu/#abs/2016ApJS..227...29P).  The addition of support for misaligned stars in version 2.1 is described in [Horvat et al. (2018, ApJS 237, 26)](https://ui.adsabs.harvard.edu/#abs/2018ApJS..237...26H).  Interstellar extinction and support for Python 3 was added in version 2.2 and described in [Jones et al. (2020, ApJS 247, 63)](https://ui.adsabs.harvard.edu/abs/2020ApJS..247...63J).  Inclusion of a general framework for solving the inverse problem as well as support for the [web and desktop clients](http://phoebe-project.org/clients) was introduced in version 2.3 as described in [Conroy et al. (2020, in press)](https://ui.adsabs.harvard.edu/abs/2020arXiv200616951C), which also removes support for Python 2.

PHOEBE 2 is released under the [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.en.html).

The source code is available for download from the [PHOEBE project homepage](http://phoebe-project.org) and from [github](https://github.com/phoebe-project/phoebe2).

The development of PHOEBE 2 is funded in part by [NSF grant #1517474](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1517474), [NSF grant #1909109](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1909109) and [NASA 17-ADAP17-68](https://ui.adsabs.harvard.edu/abs/2017adap.prop...68P).


DOWNLOAD AND INSTALLATION
-------------------------

The easiest way to download and install PHOEBE 2 is by using pip (make sure you're using the correct command for pip that points to your python3 installation - if in doubt use something like `python3 -m pip install phoebe`):

    pip install phoebe

To install it site-wide, prefix the `pip` command with `sudo` or run it as root.

To download the PHOEBE 2 source code, use git:

    git clone https://github.com/phoebe-project/phoebe2.git

To install PHOEBE 2 from the source locally, go to the `phoebe2/` directory and issue:

    python3 setup.py build
    python3 setup.py install --user

To install PHOEBE 2 from the source site-wide, go to the `phoebe2/` directory and issue:

    python3 setup.py build
    sudo python3 setup.py install

Note that as of the 2.3 release, PHOEBE requires Python 3.6 or later.  For further details on pre-requisites consult the [PHOEBE project webpage](http://phoebe-project.org/install/2.3).


GETTING STARTED
---------------

PHOEBE 2 has a fairly steep learning curve. To start PHOEBE from python, issue:

    python
    >>> import phoebe
    >>>

As of the 2.3 release, PHOEBE also includes a desktop and web client user-interface which is installed independently of the python package here.  See the [phoebe2-ui repository](https://github.com/phoebe-project/phoebe2-ui) and [phoebe-project.org/clients](http://phoebe-project.org/clients) for more details.

To understand how to use PHOEBE, please consult the [tutorials, scripts and manuals](http://phoebe-project.org/docs/2.3/) hosted on the PHOEBE webpage.


CHANGELOG
----------

### 2.3.3 - latex representation hotfix

* fix the latex representation string for `fillout_factor`, `pot`, `pot_min`,
  and `pot_max` parameters in a contact binary.

### 2.3.2 - manifest to include readme hotfix

* manually include README.md in MANIFEST.in to avoid build errors from pip

### 2.3.1 - pip install hotfix

* removes m2r as an (unlisted) build-dependency.  m2r is only required to build the submission to submit to pypi, but is not required to install or run phoebe locally.

### 2.3.0 - inverse problem feature release

* Add support for inverse problem solvers, including "estimators", "optimizers", and "samplers"
* Add support for attaching distributions (as [distl](https://github.com/kecnry/distl) objects) to parameters, including priors and posteriors.
* Add support for [web and desktop clients](http://phoebe-project.org/clients) via a light-weight built in `phoebe-server`.
* Removed support for Python 2 (now requires Python 3.6+)
* Implement optional gaussian processes for light curves
* Implement phase-masking
* Added official support for [ellc](https://github.com/pmaxted/ellc) and [jktebop](https://www.astro.keele.ac.uk/jkt/codes/jktebop.html) alternate backends
* Per-component and per-dataset RV offsets
* Fixed phasing in time-dependent systems
* Distinction between anomalous and sidereal period in apsidal motion cases
* Extinction parameters moved from per-dataset to the system-level
* Added several new optional constraints
* Overhaul of the run_checks framework
* Updated scipy dependency to 1.7+
* Numerous small bugfixes and enhancements


### 2.2.2 - kwargs hotfix

* fix overriding mesh_init_phi as kwarg to run_compute
* fix pblum computation to not require irrad_method kwarg
* fix bundle representation to exclude hidden parameters

### 2.2.1 - g++/gcc version check hotfix

* Improves the detection of g++/gcc version to compare against requirements during setup.

### 2.2.0 - extinction feature release

* Add support for interstellar extinction/reddening.
* Support for Python 3.6+ in addition to Python 2.7+.
* Overhaul of limb-darkening with new ld_mode and ld_coeffs_source parameters.
* Overhaul of passband luminosity and flux scaling with new pblum_mode parameter, including support for maintaining color relations between multiple passbands.
* Ability to provide third light in either flux or percentage units, via the new l3_mode and l3_frac parameters.
* Support for computing a model at different times than the observations, via the new compute_times or computes_phases parameter.
* Transition from pickled to FITS passband files, with automatic detection for available updates.  The tables can now also be accessed via tables.phoebe-project.org.
* DISABLED support for beaming/boosting.
* Allow flipping Kepler's thrid law constraint to solve for q.
* Require overwrite=True during add_* or run_* methods that would result in overwriting an existing label.
* Constraint for logg.
* Account for time-dependence (dpdt/dperdt) in t0 constraints.

### 2.1.17 - ignore fits passbands hotfix

* Future-proof to ignore for passband files with extensions other than ".pb"
  which may be introduced in future versions of PHOEBE.

### 2.1.16 - eccentric/misaligned irradiation hotfix

* Fixes bug where irradiation was over-optimized and not recomputed as needed for
  eccentric or misaligned orbits.  Introduced in the optimizations in 2.1.6.

### 2.1.15 - spots hotfix

* Fixes 'long' location of spots on single stars.
* Fixes treatment of spots on secondary 'half' of contact systems.
* Fixes loading legacy files with a spot that has source of 0 due to a bug in legacy.
* Fixes overriding 'ntriangles' by passing keyword argument to run_compute.

### 2.1.14 - contacts inclination RVs hotfix

* Fixes the polar rotation axis for RVs in contact systems with non-90 inclinations
  by re-enabling the alignment (pitch, yaw) constraints and enforcing them to be 0.

### 2.1.13 - constraint flip loop hotfix

* Fixes infinite loop when trying to flip esinw AND ecosw
* Adds ability to flip mass (Kepler's third law) to solve for q
* Fixes bug introduced in 2.1.9 in which out-of-limits constrained parameters in
an envelope were being raised before all constraints could resolve successfully.

### 2.1.12 - legacy ephemeris and kwargs checks hotfix

* Fixes applying t0 when importing legacy dataset which use phase.
* Fixes ignoring other compute options when running checks on kwargs during run_compute.

### 2.1.11 - legacy dataset import hotfix

* Fixes loading legacy datasets which use phase (by translating to time with the current ephemeris).
* Fixes loading legacy datasets with errors in magnitudes (by converting to errors in flux units).
* Fixes plotting RV datasets in which only one component has times (which is often the case when importing from a legacy file).

### 2.1.10 - ldint hotfix

* Removes ldint from the weights in the computations of RVs and LPs.

### 2.1.9 - limits hotfix

* Fixes a bug where parameter limits were not being checked and out-of-limits errors not raised correctly.

### 2.1.8 - mesh convergence hotfix

* Fixes a bug where certain parameters would cause the meshing algorithm to fail to converge.  With this fix, up to 4 additional attempts will be made with random initial starting locations which should converge for most cases.

### 2.1.7 - comparison operators hotfix

* Fixes a bug where comparisons between Parameters/ParameterSets and values were returning nonsensical values.
* Comparing ParameterSets with any object will now return a NotImplementedError
* Comparing Parameters will compare against the value or quantity, with default units when applicable.
* Comparing equivalence between two Parameter objects will compare the uniqueids of the Parameters, NOT the values.

### 2.1.6 - optimization hotfix

* Fixes a bug where automatic detection of eclipses was failing to properly fallback on only detecting the horizon.
* Introduces several other significant optimizations, particularly in run_compute.

### 2.1.5 - single star get_orbits and line-profile hotfix

* Fixes a bug in hierarchy.get_orbits() for a single star hierarchy which resulted in an error being raised while computing line-profiles.

### 2.1.4 - freq constraint hotfix

* This fixes the inversion of the frequency constraint when flipping to solve for period.

### 2.1.3 - overflow error for semidetached systems hotfix

* Semi-detached systems could raise an error in the backend caused by the volume being slightly over the critical value when translating between requiv in solar units to volume in unitless/roche units.  When this numerical discrepancy is detected, the critical value is now adopted and a warning is sent via the logger.

### 2.1.2 - constraints in solar units hotfix

* All constraints are now executed (by default) in solar units instead of SI.  The Kepler's third law constraint (constraining mass by default) failed to have sufficient precision in SI, resulting in inaccurate masses.  Furthermore, if the constraint was flipped, inaccurate values of sma could be passed to the backend, resulting in overflow in the semi-detached case.
* Bundles created before 2.1.2 and imported into 2.1.2+ will continue to use SI units for constraints and should function fine, but will not benefit from this update and will be incapable of changing the system hierarchy.

### 2.1.1 - MPI detection hotfix

* PHOEBE now detects if its within MPI on various different MPI installations (previously only worked for openmpi).

### 2.1.0 - misalignment feature release

* Add support for spin-orbit misalignment
* Add support for line profile (LP) datasets
* Switch parameterization from rpole/pot to requiv (including new semi-detached and contact constraints)
* Significant rewrite to plotting infrastructure to use [autofig](http://github.com/kecnry/autofig)
* Introduction of [nparray](http://github.com/kecnry/nparray) support within parameters
* Significant rewrite to mesh dataset infrastructure to allow choosing which columns are exposed
* Distinguish Roche (xyz) from Plane-of-Sky (uvw) coordinates
* Ability to toggle interactive constraints and interactive system checks independently
* Implementation of ParameterSet.tags and Parameter.tags
* General support for renaming tags/labels
* Expose pblum for contacts
* Expose per-component r and rprojs for contacts (used to be based on primary frame of reference only)
* Fix definition of vgamma (see note in 2.0.4 release below)
* Remove phshift parameter (see note in 2.0.3 release below)
* Permanently rename 'long' parameter for spots (see note in 2.0.2 release below)
* Numerous other minor bug fixes and improvements

### 2.0.11 - astropy version dependency hotfix

* Set astropy dependency to be >=1.0 and < 3.0 (as astropy 3.0 requires python 3)

### 2.0.10 - legacy import extraneous spaces hotfix

* Handle ignoring extraneous spaces when importing a PHOEBE legacy file.


### 2.0.9 - \_default Parameters hotfix

* Previously, after loading from a JSON file, new datasets were ignored by run_compute because the \_default Parameters (such as 'enabled') were not stored and loaded correctly.  This has now been fixed.
* PS.datasets/components now hides the (somewhat confusing) \_default entries.
* unicode handling in filtering is improved to make sure the copying rules from JSON are followed correctly when loaded as unicodes instead of strings.

### 2.0.8 - contacts hotfix

* Remove unused Parameters from the Bundle
* Improvement in finding the boundary between the two components of a contact system

### 2.0.7 - legacy import/export hotfix

* Handle missing parameters when importing/exporting so that a Bundle exported to a PHOEBE legacy file can successfully be reimported
* Handle importing standard weight from datasets and converting to sigma

### 2.0.6 - unit conversion hotfix

* When requesting unit conversion from the frontend, astropy will now raise an error if the units are not compatible.

### 2.0.5 - semi-detached hotfix

* Fixed bug in which importing a PHOEBE legacy file of a semi-detached system failed to set the correct potential for the star filling its roche lobe.  This only affects the importer itself.
* Implemented 'critical_rpole' and 'critical_potential' constraints.

### 2.0.4 - vgamma temporary hotfix

* The definition of vgamma in 2.0.* is in the direction of positive z rather than positive RV.  For the sake of maintaining backwards-compatibility, this will remain unchanged for 2.0.* releases but will be fixed in the 2.1 release to be in the direction of positive RV.  Until then, this bugfix handles converting to and from PHOEBE legacy correctly so that running the PHOEBE 2 and legacy backends gives consistent results.

### 2.0.3 - t0_supconj/t0_perpass hotfix

* Fixed constraint that defines the relation between t0_perpass and t0_supconj.
* Implement new 't0_ref' parameter which corresponds to legacy's 'HJD0'.
* Phasing now accepts t0='t0_supconj', 't0_perpass', 't0_ref', or a float representing the zero-point.  The 'phshift' parameter will still be supported until 2.1, at which point it will be removed.
* Inclination parameter ('incl') is now limited to the [0-180] range to maintain conventions on superior conjunction and ascending/descending nodes.
* Fixed error message in ldint.
* Fixed the ability for multiple spots to be attached to the same component.
* Raise an error if attempting to attach spots to an unsupported component.  Note: spots are currently not supported for contact systems.

### 2.0.2 - spots hotfix

* If using spots, it is important that you use 2.0.2 or later as there were several important bug fixes in this release.
* 'colon' parameter for spots has been renamed to 'long' (as its not actually colongitude).  For 2.0.X releases, the 'colon' parameter will remain as a constrained parameter to avoid breaking any existing scripts, but will be removed with the 2.1.0 release.
* Features (including spots) have been fixed to correctly save and load to file.
* Corotation of spots is now enabled: if the 'syncpar' parameter is not unity, the spots will correctly corotate with the star.  The location of the spot (defined by 'colat' and 'long' parameters) is defined such that the long=0 points to the companion star at t0.  That coordinate system then rotates with the star according to 'syncpar'.

### 2.0.1 - ptfarea/pbspan hotfix

* Definition of flux and luminosity now use ptfarea instead of pbspan.  In the bolometric case, these give the same quantity. This discrepancy was absorbed entirely by pblum scaling, so relative fluxes should not be affected, but the underlying absolute luminosities were incorrect for passbands (non-bolometric).  In addition to under-the-hood changes, the exposed mesh column for 'pbspan' is now removed and replaced with 'ptfarea', but as this is not yet a documented column, should not cause backwards-compatibility issues.  

### 2.0.0 - official release of PHOEBE 2.0

* PHOEBE 2.0 is not backwards compatible with PHOEBE 2.0-beta (although the interface has not changed appreciably) or with PHOEBE 2.0-alpha (substantial rewrite). Going forward with incremental releases, every effort will be put into backwards compatibility. The changes and important considerations of the new version will be detailed in the ChangeLog.

* If upgrading from PHOEBE 2.0-beta or PHOEBE 2.0-alpha, it is necessary to do a clean re-install (clear your build and installation directories), as the passband file format has changed and will not automatically reset unless these directories are manually cleared.  Contact us with any problems.


QUESTIONS? SUGGESTIONS? CONCERNS?
---------------------------------

Contact us! Issues and feature requests should be submitted directly through GitHub's issue tracker. Two mailing lists are dedicated for discussion, either on user level ([phoebe-discuss@lists.sourceforge.net](mailto:phoebe-discuss@lists.sourceforge.net)) or on the developer level ([phoebe-devel@lists.sourceforge.net](mailto:phoebe-devel@lists.sourceforge.net)). We are eager to hear from you, so do not hesitate to contact us!
