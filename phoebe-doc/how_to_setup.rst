How to set up Phoebe 2.x
========================

Prerequisites
-------------

Use Linux, though Mac OSX should work too. You should have installed the
following packages:

    * svn (to download the repository)
    * Python 2.7 with packages matplotlib, numpy and scipy (see :ref:`label-requirements` for recommended dependencies)

To generate the documentation, you need sphinxdoc.

Installation
------------

Download the `Phoebe SVN repository <http://phoebe.fmf.uni-lj.si/?q=node/12>`_.
You don't need the entire Phoebe repository (but it doesn't hurt) and you
don't need to install the 0.x or 1.x version of Phoebe (though nobody died
trying). The new 2.x version is completely indepedent. The minimal branch you
need to download is the devel branch.

Go to some working directory and type in a
terminal (read only access for non-developers)::

    $:> svn checkout svn://svn.code.sf.net/p/phoebe/code/devel devel
    
Developers might want read and write access, and can download Phoebe via::

    $:> svn checkout --username <username> svn+ssh://username@svn.code.sf.net/p/phoebe/code/ phoebe-code

If you want to use atmospheres different from blackbodies, see the section on
:ref:`Additional files <label-additional_files>` before going any further.

If you have ``pip``, you can do::
    
    $:> python setup.py sdist
    $:> sudo pip install dist/phoebe-2.0.tar.gz

If you don't have ``pip``, build the package with::

    $:> python setup.py build
    $:> sudo python setup.py install
    
Make sure that you have the necessary permissions for the second step.

Finally, to test your installation, go to some working directory, start a
Python shell and try to import the main Phoebe namespace::

    >>> import phoebe
    
If nothing happens: great! If something fails, check the :ref:`label-requirements`
or send a detailed (!) bug report.

Uninstalling
------------

If you installed Phoebe via ``pip``, you can simply do::
    
    $:> sudo pip uninstall phoebe
    
Otherwise, you need to manually remove the installation directory.

Specifications
==============

.. _label-requirements:

Software requirements
---------------------

*Note: not all of the version numbers are necessarily minimum requirements. If
you have earlier versions of some of these packages, try to build and see if
works. It is possible that only on specific occasions where features are used
from later versions, errors occur. If this happens to you, you can either update
your software packages, or inform a developer.*

Necessary:

    * Python 2.7
    * Numpy (1.6.2) + Scipy (0.10.1)

Recommended:

    * Matplotlib (1.1.1): required for making plots
    * pyfits (3.0.8): required for using tabulated atmosphere models
    * pymc (2.2): required for MCMC fitting with Metropolis_hastings algorithm
    * emcee (1.1.2): required for MCMC fitting with Affine Invariants
    * lmfit (0.7): required for nonlinear optimizers
    

Nice to have:

    * mayavi (4.1.0): required for making 3D plots (exclusively for debugging purposes)
    * mpi4py (1.3): required for making use of multi-processor capabilities
    * sphinxdoc (1.1.3): for documentation generation

.. note::
   
   *buntu users can install numpy, scipy, matplotlib, pyfits, mpi4py and
   mayavi from the package repository (Software Apper, Muon, apt-get)::
       
       $:> sudo apt-get install python-numpy python-scipy
       $:> sudo apt-get install python-matplotlib python-pyfits python-mpi4py mayavi2
   
   The packages pymc, emcee and lmfit can be installed through pip. If you don't
   have pip, do::
       
       $:> sudo apt-get install python-pip
   
   followed by::
       
       $:> sudo pip install pymc
       $:> sudo pip install emcee
       $:> sudo pip install lmfit


.. _label-additional_files:

Additional files
----------------

If you want to use non-blackbody atmospheres, you will have to create your
own limbdarkening tables, or use one of those provided below. Important note:
you need to download these files separately, and put them in your
``devel/phoebe/atmosphers/tables/ld_coeffs/`` directory **before** making the
distribution (with ``pip sdist``) or the ``setup.py install``.

Atmosphere files:
    
    * :download:`Kurucz, solar Z, Claret LD, fitted equidistantly in r coordinates, grid in Teff,logg <../phoebe/atmospheres/tables/ld_coeffs/kurucz_p00_claret_equidist_r_leastsq_teff_logg.fits>`.
    * :download:`Blackbody, uniform LD, grid in Teff only <../phoebe/atmospheres/tables/ld_coeffs/blackbody_uniform_none_teff.fits>`.
    
These limb darkening tables belong in ``phoebe/atmospheres/tables/ld_coeffs``. If you keep the filename as it is, it get's
automatically detected via the shortcut ``atm=kurucz`` or ``ld_coeffs=kurucz``, otherwise
you will have to replace ``kurucz`` with the actual filename.

Coding styles
-------------

1. Python
~~~~~~~~~

The basic coding style is `PEP 8 <http://www.python.org/dev/peps/pep-0008>`_.
Some highlights:

Coding:

    * Consistency with this style guide is important. Consistency within a project is more important. Consistency within one module or function is most important.
    * Use 4 spaces per indentation level.
    * Limit all lines to a maximum of 79 characters.
    * Imports should usually be on separate lines
    * Imports are always put at the top of the file, just after any module comments and docstrings, and before module globals and constants.
    * Relative imports for intra-package imports are highly discouraged. Always use the absolute package path for all imports.
    * Don't use spaces around the = sign when used to indicate a keyword argument or a default parameter value.

Naming of variables:

    * Modules should have short, all-lowercase names. Underscores can be used in the module name if it improves readability.
    * Almost without exception, class names use the CapWords convention
    * Function names should be lowercase, with words separated by underscores as necessary to improve readability.


2. C/C++
~~~~~~~~~

A C- expert should write this part...


Profiling
---------

In Python, there is an easy way to see which process cumulatively take the
longest time to run. Cumulative is quite important here, because it's
equally relevant to optimize a function that runs 0.01 s but runs a hundred
times, as to optimize a function that runs for 1.00 s but runs only one time.

Be careful though, probably the functions that take the longest are wrapper
functions, so you need to look for those that actually do some work.

As an example, you can run the ``wd_vs_phoebe.py`` script and save the
profiling information to a file called ``my.profile``::

    $:> python -m cProfile -o my.profile wd_vs_phoebe.py

This profile file can be interactively investigated::
    
    $:> python -m pstats my.profile
    sort cumulative
    stats 10

But you can also script it::

    import pstats
    p = pstats.Stats('my.profile')
    p.sort_stats('cumulative').print_stats(10)
    
Or merge several profiling output in one big file::
    
    p.add('myother.profile')
    p.dump_stats('merged.profile')

From the `Python profiles <http://docs.python.org/2/library/profile.html>`_
documentation:

Call count statistics can be used to identify bugs in code (surprising
counts), and to identify possible inline-expansion points (high call counts).
Internal time statistics can be used to identify “hot loops” that should be
carefully optimized. Cumulative time statistics should be used to identify
high level errors in the selection of algorithms. Note that the unusual
handling of cumulative times in this profiler allows statistics for recursive
implementations of algorithms to be directly compared to iterative
implementations.

There is handy visualisation tool available, called **RunSnakeRun**. You can
load a profile output file, and see the time spent in certain parts of the
code as a squaremap, where the area of each subsquare is proportional to the
execution time. It is extremely useful to sort there based on *Cum* or *Local*.

.. image:: images_tut/runsnakerun.png
   :scale: 75 %
   :align: center

    
