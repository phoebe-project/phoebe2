PHOEBE 2.0 RELEASE NOTES &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; SEPTEMBER 26, 2016
------------------------------------------------------------------------

Hello and thank you for your interest in PHOEBE 2.0!


INTRODUCTION
------------

PHOEBE 2.0 is a complete rewrite of the original PHOEBE code. For most
up-to-date information please refer to the PHOEBE project webpage:

    http://phoebe-project.org

PHOEBE 2.0 is described by the release paper published in the Astrophysical
Journal Supplements

    http://adsabs.harvard.edu/abs/2016ApJS..227...29P

The source code is available for download from github:

    https://github.com/phoebe-project/phoebe2

Development of PHOEBE 2.x is funded in part by the NSF grant #1517474.


DOWNLOAD AND INSTALLATION
-------------------------

To download PHOEBE 2.0, use git:

    git clone https://github.com/phoebe-project/phoebe2.git

To install PHOEBE 2.0 locally, go to phoebe2/ directory and issue:

    python setup.py build
    python setup.py install --user

To install PHOEBE 2.0 site-wide, go to phoebe2/ directory and
issue:

    python setup.py build
    sudo python setup.py install

This will install phoebe in the current path and no further action
should be necessary. For further details on pre-requisites and minimal
versions of python consult the PHOEBE webpage.


GETTING STARTED
---------------

PHOEBE 2.0 has a steep learning curve associated with it. There is
no graphical front-end as of yet; the front-end is now written in
python. To start PHOEBE, issue:

    python
    >>> import phoebe
    >>>

To understand how to use PHOEBE, please consult the tutorials, scripts
and manuals hosted on the PHOEBE webpage:

    http://phoebe-project.org/docs/2.0b/#Tutorials


CHANGELOG
----------

### 2.0 release

* PHOEBE 2.0 is not backwards compatible with PHOEBE 2.0-beta (although the
interface has not changed much at all) or with PHOEBE 2.0-alpha (complete
rewrite).  Going forward with incremental releases, this changelog will list
any necessary considerations when upgrading to a new version.

* If upgrading from PHOEBE 2.0-beta or PHOEBE 2.0-alpha, it is necessary to
do a clean re-install (clear your build and installation directories), as the
passband file format has changed and will not automatically reset unless these
directories are manually cleared.  Contact us with any problems.


QUESTIONS? SUGGESTIONS? CONCERNS?
---------------------------------

Contact us! Issues and feature requests should be submitted directly through
GitHub's issue tracker.  Two mailing lists are dedicated for discussion, either
on user level (phoebe-discuss@lists.sourceforge.net) or on the developer level
(phoebe-devel@lists.sourceforge.net). We are eager to hear from you, so do not
hesitate to contact us!
