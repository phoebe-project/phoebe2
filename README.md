PHOEBE 2.1 RELEASE NOTES
------------------------

Hello and thank you for your interest in PHOEBE 2!


INTRODUCTION
------------

PHOEBE stands for PHysics Of Eclipsing BinariEs. PHOEBE is pronounced [fee-bee](https://www.merriam-webster.com/dictionary/phoebe?pronunciation&lang=en_us&file=phoebe01.wav).

PHOEBE 2 is a rewrite of the original PHOEBE code. For most up-to-date information please refer to the PHOEBE project webpage: [http://phoebe-project.org](http://phoebe-project.org)

PHOEBE 2.0 is described by the release paper published in the Astrophysical Journal Supplement, [Prša et al. (2016, ApJS 227, 29)](https://ui.adsabs.harvard.edu/#abs/2016ApJS..227...29P).  The addition of support for misaligned stars in version 2.1 is described in [Horvat et al. (2018, ApJS 237, 26)](https://ui.adsabs.harvard.edu/#abs/2018ApJS..237...26H).

PHOEBE 2 is released under the [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.en.html).


The source code is available for download from the [PHOEBE project homepage](http://phoebe-project.org) and from [github](https://github.com/phoebe-project/phoebe2).

The development of PHOEBE 2 is funded in part by the [NSF grant #1517474](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1517474).


DOWNLOAD AND INSTALLATION
-------------------------

The easiest way to download and install PHOEBE 2 is by using pip:

    pip install phoebe

To install it site-wide, prefix the `pip` command with `sudo` or run it as root.

To download the PHOEBE 2 source code, use git:

    git clone https://github.com/phoebe-project/phoebe2.git

To install PHOEBE 2 from the source locally, go to the `phoebe2/` directory and issue:

    python setup.py build
    python setup.py install --user

To install PHOEBE 2 from the source site-wide, go to the `phoebe2/` directory and issue:

    python setup.py build
    sudo python setup.py install

For further details on pre-requisites and minimal versions of python consult the [PHOEBE project webpage](http://phoebe-project.org).


GETTING STARTED
---------------

PHOEBE 2 has a steep learning curve. There is no graphical front-end as of yet; the front-end is now written in python. To start PHOEBE, issue:

    python
    >>> import phoebe
    >>>

To understand how to use PHOEBE, please consult the [tutorials, scripts and manuals](http://phoebe-project.org/docs/2.1/#Tutorials) hosted on the PHOEBE webpage.


CHANGELOG
----------

### 2.1.3 - overflow error for semidetached systems hotfix

* Semi-detached systems could raise an error in the backend caused by the volume being slightly over the critical value when translating between requiv in solar units to volume in unitless/roche units.  When this numerical discrepancy is detected, the critical value is now adopted and a warning is sent via the logger.

### 2.1.2 - Constraints in solar units hotfix

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
