PHOEBE 2.0-beta RELEASE NOTES &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; SEPTEMBER 26, 2016
------------------------------------------------------------------------

Hello and thank you for your interest in PHOEBE 2.0-beta!


BEFORE YOU INSTALL AND USE PHOEBE, YOU SHOULD KNOW...
-----------------------------------------------------

PHOEBE 2.0 is released as a beta version, and is therefore still under
testing and development. We encourage everyone to try the beta version
and report any suggestions, comments, and bugs. As soon as the release
paper has been accepted for publication, we will release the official
version of PHOEBE 2.0. Until then, we do not guarantee backwards
compatibility and will try to address all suggestions and bug reports
during the refereeing process. Below are the versions we suggest using
based on your needs:

* PHOEBE 1.0 (legacy) should be used if you want to get any trustable
  science results.
* PHOEBE 2.0 alpha should be used if you want to play with support for
  new physics and observables, but should not be used for science and is
  no longer actively supported. Documentation and tutorials may be
  slightly out-of-date, but will remain available online for the time
  being.
* PHOEBE 2.0 beta should be used to learn the interface for PHOEBE going
  forward and for testing, but should be used with caution until the
  official 2.0 release. Watch here or subscribe to the phoebe
  announcements mailing list to be notified when the 2.0 version is
  officially released. If you do choose to give the 2.0b version a try,
  please contact us.


INTRODUCTION
------------

PHOEBE 2.0 is a complete rewrite of the original PHOEBE code. For most
up-to-date information please refer to the PHOEBE project webpage:

    http://phoebe-project.org

PHOEBE 2.0 is described by the release paper that is submitted to the
AAS journals:

    http://arxiv.org/abs/1609.08135

It is available for download from github:

    https://github.com/phoebe-project/phoebe2

PHOEBE 2.0-beta development is funded in part by the NSF grant #1517474.


DOWNLOAD AND INSTALLATION
-------------------------

To download PHOEBE 2.0-beta, use git:

    git clone https://github.com/phoebe-project/phoebe2.git

Once PHOEBE 2.0 is officially released, we will also provide the package
via pip and as a standalone tarball that can be downloaded from the
PHOEBE homepage.

To install PHOEBE 2.0-beta locally, go to phoebe2/ directory and issue:

    python setup.py build
    python setup.py install

Note that you will need to update your python path to reflect your local
installation directory.

To install PHOEBE 2.0-beta site-wide, go to phoebe2/ directory and
issue:

    python setup.py build
    sudo python setup.py install

This will install phoebe in the current path and no further action
should be necessary. For further details on pre-requisites and minimal
versions of python consult the PHOEBE webpage.


GETTING STARTED
---------------

PHOEBE 2.0-beta has a steep learning curve associated with it. There is
no graphical front-end as of yet; the front-end is now written in
python. To start PHOEBE, issue:

    python
    >>> import phoebe
    >>> 

To understand how to use PHOEBE, please consult the tutorials, scripts
and manuals hosted on the PHOEBE webpage:

    http://phoebe-project.org/2.0b/#Tutorials


QUESTIONS? SUGGESTIONS? CONCERNS?
---------------------------------

Contact us! Two mailing lists are dedicated for discussion, either on
user level (phoebe-discuss@lists.sourceforge.net) or on the developer
level (phoebe-devel@lists.sourceforge.net). We are eager to hear from
you, so do not hesitate to contact us!
