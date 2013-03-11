"""
.. contents:: Table of contents
   :depth: 3


How to create a custom binary system
====================================




Step 0: Preparations
---------------------

Import the Phoebe namespace and create a logger to log messages to the screen
or to a log file.

"""

import phoebe

logger = phoebe.get_basic_logger()

"""

Step 1: Set up the parameters for the components and the orbit
-----------------------------------------------------------------

There are several possibilities to define these parameters, the result of
which should each time be three ParameterSets: one for each component, and
one for the (common) orbit. You can choose to:

1. load a predefined binary :ref:`from the library <label-load_from_library>`,
2. define the components and orbits :ref:`directly in the Roche framework <label-load_directly>`,
3. :ref:`first create single stars <label-load_from_stars>`, and convert these to binary components, or,
4. load parameters from Phoebe Legacy or Wilson-Devinney and convert them to the
   Phoebe contexts

.. _label-load_from_library:

Possibility 1: Loading parameters from the library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``create`` module contains several predefined systems, which you can use
out of the box, or use as a starting point to create a new system:
"""

comp1,comp2,orbit = phoebe.create.from_library('V380_Cyg')

"""
The argument to the function needs to be either a recognised system, or a
filename. You can load parameters from a self-made ASCII file containing
components and an orbit too. This approach can be used to adjust and reload
parameters in a file-based approach. You can further refine the values of the
loaded parameters through the dictionary-style interface:
"""

orbit['vgamma'] = 52.,'km/s'

"""

.. _label-load_directly:

Possibility 2: Defining binary components directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a binary system, you can define the parameters of the components and
the orbit. The default parameters can be changed during initialisation, by
adding them as keywords, or they can be altered using the dictionary-style
interface:

"""

comp1 = phoebe.ParameterSet(context='component',syncpar=1.0,teff=(10000.,'K'))
comp2 = phoebe.ParameterSet(context='component',teff=9500.)
orbit = phoebe.ParameterSet(context='orbit',c1label=comp1['label'],
                                c2label=comp2['label'],ecc=0,incl=(89.,'deg'),
                                sma=(10,'Rsol'))
orbit['distance'] = 1.,'kpc'

"""

.. _label-load_from_stars

Possibility 3: Defining binary components via stars
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can first create the components as single stars, and place them in a
binary orbit afterwards. Via Kepler's third law, you only need to specify the
separation *or* the period of the binary system, since the total mass of the
system is the sum of the masses of the stellar components.

On their turn, stars can be initialised in two ways: either by creating a
parameterSet with the ``star`` context:
"""

star = phoebe.ParameterSet(context='star')

"""
It is possible to change the defaults using the dictionary-style interface of
parameterSets (with or without explicit specification of the units):
"""

star['radius'] = 2.,'Rsol'
star['mass'] = 1.5 # solar mass is implicit

"""
For convenience, it is also possible to immediately pass these arguments when
initialising the parameterSet. The previous three lines of code are equivalent
to:
"""

star = phoebe.ParameterSet(context='star',radius=(2.,'Rsol'),mass=1.5)

"""
Alternatively, you can choose to set the defaults to match a particular
spectral type via the convenience functions available in the ``create``
module:
"""

star = phoebe.create.star_from_spectral_type('A0V')

# Also in this case, it is possible to overwrite some parameters immediately:

star = phoebe.create.star_from_spectral_type('A0V',radius=(2.,'Rsol'),mass=1.5)

"""
To make a binary from stars, we need two stars, and some basic information on
the orbit. This can again be done via the convenience functions inside the
``create`` module, via ``binary_from_stars`` or ``binary_from_spectroscopy``,
depending on the type of information of the orbit that is available:
"""

star1 = phoebe.create.star_from_spectral_type('A0V')
star2 = phoebe.create.star_from_spectral_type('G2V')
comp1,comp2,orbit = phoebe.create.binary_from_stars(star1,star2,sma=(4.,'Rsol'))
comp1,comp2,orbit = phoebe.create.binary_from_spectroscopy(star1,star2,period=10.,ecc=0.,K1=20.)

"""
The result of the previous two operations is, again, three new parameterSets:
one for each component, and one for the orbit.

Possibility 4: Loading components from Phoebe Legacy or Wilson-Devinney
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can load the contents of a WD lcin file (:download:`download <scripts/test01lcin.active>`) to parameterSets in the WD
framework. You can afterwards change any of the parameters through the
dictionary-style interface.
"""

ps,lc,rv = phoebe.wd.lcin_to_ps('test01lcin.active',version='wd2003')
ps['n1'] = 60

# Finally, you can convert these parameterSets to the Phoebe frame:
pars1,pars2,orbit = phoebe.wd.wd_to_phoebe(ps,lc,rv)
comp1,lcdep1,rvdep1 = pars1
comp2,lcdep2,rvdep2 = pars2

# The last two lines already extract the parameterSets necessary to compute
# light and radial velocity curves (see next section). There's one for each
# component.

"""

Step 2: Define (multiple) light curves and friends
--------------------------------------------------

Defining the parameters for the components (or individual stars) is one thing,
but most often you want to compare models with observations. Observations are
always performed in a certain passband or in a certain wavelength interval,
and can be integrated across the surface, wavelength resolved, or created
via interference patterns. Each type of observables has a predefined
ParameterSet. The contexts are given by a prefix denoting the type of
observable (``lc`` for light curve, ``rv`` for radial velocity curves, ``sp``
for spectra and ``if`` for interferometry), followed by the suffix ``dep``. As
with other ParameterSets, you can change the defaults via the dictionary-style
interface and/or upon initialisation:

"""

lcdep1a = phoebe.ParameterSet(context='lcdep',passband='JOHNSON.V',\
            atm='kurucz',ld_coeffs='kurucz',ld_func='claret')
lcdep1b = phoebe.ParameterSet(context='lcdep',passband='KEPLER.V',\
            atm='blackbody',ld_coeffs=[0.5],ld_func='linear',ref='linear thing')
spdep = phoebe.ParameterSet(context='spdep')

"""
Each observable parameterSet has a ``label`` keyword, which needs to be a unique
string. If not specified, the UUID framework is used to generate a unique
label, but this is typically not human-readable. Since these labels are
automatically generated, it is not strictly necessary to specify them
manually. They are only there for convenience, in case you have a lot of
observable ParameterSets and you want to have an easy way of refering to them:
there are two ways to refer to parameterSets once the Body is created (see
below):

    * By index and subtype: e.g. the pair ``subtype='lcdep',label=1`` refers
      to the second ParameterSet of subtype 'lcdep' that is added.
    * By name: e.g. ``label='linear thing'``, in this case, also unambiguously
      refers to the second PararameterSet that was added.

This behaviour is possible because internally, the parameterSets are stored
in an OrderedDictionary. You can inconsistently mix the two approaches.

Now, you have two components in the binary system. Each of them can have
different limb darkening coefficients, or atmosphere grids etc. Therefore, you
need to define the observable parameterSets for each component separately. To
make it clear to the code that they actually belong together, you are required
to make sure that the labels match (notice that you can do that even though
a label is automatically generated):
"""

lcdep2a = phoebe.ParameterSet(context='lcdep',passband='JOHNSON.V',\
            atm='kurucz',ld_coeffs=[0.4],ld_func='linear',ref=lcdep1a['ref'])
lcdep2b = phoebe.ParameterSet(context='lcdep',passband='KEPLER.V',\
            atm='blackbody',ld_coeffs=[0.3],ld_func='linear',ref='linear thing')

"""

Notice that we do not redefine the observable set for the spectra. It is
allowed to pass the exact same parameterSet to all components (they don't even
need to be copied).

Step 3: Creating the Bodies
-------------------------------------

A binary consists of two components. The components can be of any type, but
most likely you need them to be of the class ``BinaryRocheStar``. Also, you
probably want to put them in a ``BodyBag`` to be able to call functions on
both components simultaneously (e.g. ``set_time``, ``lc`` etc...).

Possibility 1: Creating bodies from the library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to create a full blown binary system (but offering the least
amount of flexibility), is by calling the ``from_library`` function defined in
the ``create`` module as before, but setting also the keyword ``create_body``
to ``True``:
"""

system = phoebe.create.from_library('V380_Cyg',create_body=True)

"""
If you want to change parameters, you need to go through the ``params``
attribute of each Body in the BodyBag. For example, if you want to change
the temperature of the secondary component:
"""

system[1].params['component']['teff'] = 9800.,'K'

"""
It is, however, advised not to go through the interface of the Bodies once
they are created. All parameterSets are passed by reference, so it is advised
to work on the created ParameterSets immediately.

Possibility 2: Via separate components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is easy to create a BodyBag yourself, but of course you need to
create the two BinaryRocheStars first. To do that, you also need to specify
information on how to compute the mesh. This is the last parameterSet that
is required:

"""
mesh = phoebe.ParameterSet(context='mesh:marching')

# The ``BinaryRocheStars`` are then easily created and (optionally) put in
# a ``BodyBag``, after importing the universe:

star1 = phoebe.BinaryRocheStar(comp1,orbit=orbit,mesh=mesh,pbdep=[lcdep1a,lcdep1b,spdep])
star2 = phoebe.BinaryRocheStar(comp2,orbit=orbit,mesh=mesh,pbdep=[lcdep2a,lcdep2b,spdep])
system = phoebe.BodyBag([star1,star2])

"""


Possibility 3: Via hierarchical packing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the specific case of a binary, there is a class called ``BinaryBag``, which
is nothing more than a subclass of ``BodyBag``, but which is a little more
flexible and takes care of some of the things you could easily forget (like
correctly labeling the primary and secondary components). The basic idea
is that that you give it two "objects" to put in a BinarySystem. Some conventions
apply:

    * The first object you give is the primary, the second one the secondary
    
    * You can set an object equal to the ``None`` variable, in which case
      that component will not be generated.
    
    * An object can be a Body, a BodyBag, a BinaryBag or a list of Bodies.
      In the latter case, the list will first be packed in a BodyBag itself.
      This makes it easy to do *hiearchical packing*.

When creating a Body which requires an orbit to initialize, it is allowed in
this case to set it to ``None``. But only in this case!
"""

star1 = phoebe.BinaryRocheStar(comp1,mesh=mesh,pbdep=[lcdep1a,lcdep1b,spdep])
star2 = phoebe.BinaryRocheStar(comp2,mesh=mesh,pbdep=[lcdep2a,lcdep2b,spdep])
system = phoebe.BinaryBag([star1,star2],orbit=orbit)

"""

Under the hood, the latter three calls are equivalent with the
three lines before that.

.. warning:: Difference between ``BodyBag`` and ``BinaryBag``
   
   There is subtle but important difference between ``BodyBag`` and
   ``BinaryBag`` when they are called the following way::
   
       systemA = universe.BodyBag([body1,body2],orbit=orbit)
       systemB = universe.BinaryBag([body1,body2],orbit=orbit)
    
   In the first call, ``systemA`` will be put as a whole in the orbit ``orbit``.
   This could be useful when you want to put a white dwarf star and its accretion
   disk in the same orbit, e.g. both as the primary component. You need to
   set the label of ``systemA`` to be one of ``orbit['c1label']`` or ``orbit['c2label']``.
   
   The result of the second call, on the other hand, is a binary ``systemB``
   of which ``body1`` is the primary component, and ``body2`` is the secondary
   component. The ``c1label`` and ``c2label`` qualifiers in the ParameterSet
   ``orbit`` are automatically set to match the labels of ``body1`` and
   ``body2``.
   
   Reproducing the behaviour of the first call with a ``BinaryBag``
   would require something like::
    
       systemA = universe.BinaryBag([[body1,body2],None],orbit=orbit)
    
   Reproducing the behaviour of the second call with a ``BodyBag``, on the
   other hand, would require something like::
   
       orbit['c1label'] = body1.get_label()
       orbit['c2label'] = body2.get_label()
       systemB_prim = universe.BodyBag([body1],orbit=orbit)
       systemB_secn = universe.BodyBag([body2],orbit=orbit)
       systemB = universe.BodyBag([systemB_prim,systemB_secn])
  
   In which the order of the bodies in the last call is irrelevant. It's up to
   you which framework you prefer, or adapt to the situation.

Step 4: Computing observables
------------------------------------

For any Body that is created (including Stars, BinaryRocheStars, BodyBags...),
you can compute observables in two ways, *manually* and *automatically*. The
latter option again has two possibilities, for the case without observed data
and the case with observed data:

    1. *Manual computation*: you are responsible for setting the time
    (``set_time``), eclipse detection, and subdivision. You can compute fluxes,
    velocities, generate images, spectra and visibilities. You are responsible
    for gathering and structuring the results.
    
    2. *Automatic computation*: you give an array of time points, specify
    what needs to be computed at all time points (lc, rv, sp, ifm...) and the
    result of the synthetic computations are stored inside the Body (in the OrderedDictionary attribute
    ``Body.params['syn']``). You can get the results after computations
    have finished (``get_synthetic``), and analyse them.
    
        a. *Without data*, you can use ``observatory.observe``. It
        requires an array of time points to run.
        
        b. *With data*, you can use ``observatory.compute``. It requires
        no extra input, since all information is derived from the data. For
        each added dataset, the synthetics will be computed at the relevant
        time points (see Step 5).
    
The first option is covered in earlier and later tutorials, in this tutorial
the automatic computation option is explored.

We've fiddled a bit above with the creation of the Bodies. Just for clarity,
we start over quickly. We make a circular version of V380 Cyg:

"""

comp1,comp2,orbit = phoebe.create.from_library('V380_Cyg')
star1 = phoebe.BinaryRocheStar(comp1,orbit=orbit,mesh=mesh,pbdep=[lcdep1a,lcdep1b,spdep])
star2 = phoebe.BinaryRocheStar(comp2,orbit=orbit,mesh=mesh,pbdep=[lcdep2a,lcdep2b,spdep])
system = phoebe.BodyBag([star1,star2])
orbit['ecc'] = 0.


"""

.. admonition:: ParameterSets are passed by reference.

   You can (and should) always work with the ParameterSets you defined before
   creating the Bodies. They are passed by reference, so changing them outside
   of the Body will propogate into the Bodies too.


Let us define a list (or array for that matter) of times at which we want to
compute observables. The list of times does not need to be equidistant. For all I
care, they don't even need to be chronological.
"""

import numpy as np
times = np.linspace(0,orbit['period'],100)

"""
The function ``observatory.observe`` requires this list of times
as an argument. To specify want needs to be computed at each time point, set
one of the flags ``lc``, ``rv``, ``sp`` or ``im`` to ``True``. Furthermore,
you can also select the number of subdivisions you want to perform via the
keyword ``subdivide``. There are a couple of other options (see the doc of
the function), e.g. the eclipse detection algorithm to use, and whether you
want to include heating or reflection. The defaults are quite smart, so
most of the time, you won't have to worry about those.

As a little extra, there is also the possibility to give a list of ``extra_func``,
which allow you to do perform extra operations at each computed time step (e.g.
suppose you want to write the contents of the mesh during every time step to
a file).

All that is left to do now is to make the call to the function:
"""

phoebe.observe(system,times,lc=True)

"""

The results are easily retrieved and plotted as before.

"""

lightcurve = system.get_synthetic(type='lcsyn',ref=0,cumulative=True)

import matplotlib.pyplot as plt

plt.plot(lightcurve['time'],lightcurve['flux'],'ko-')
plt.show()


"""

Step 5: Adding data (NOT UP TO DATE - STILL UNDER DISCUSSION
--------------------------------------------------------------------------------

Data are treated as a kind of ParameterSet in Phoebe, with some custom
features. The easiest is probably to load the contents from a text file. How
this text file should be organized, is a topic for a later tutorial. For now
we assume that our text file called ``mylc`` is rightly formatted.

"""

lcdata = phoebe.LCDataSet(context='lcobs',filename='mylc',ref='mylc',
                              columns=['time','flux','sigma'])


# One of the features that separates ``DataSets`` from ``ParameterSets``, is
# the ``load`` and ``unload`` method. The former extract the contents of
# ``columns`` to a Parameter of the same name in the ``lcdata`` DataSet.
# The latter removes the values from those parameters again. This is handy
# to not always have all the data in memory:

lcdata.load()
lcdata.unload()

"""


Furthermore, suppose we have disentangled spectra, and we want to use the
spectrum of the primary to fit its rotational velocity, and the spectrum of
the secondary to fit the secondary's temperature.

Spectroscopic data is initialized in the same way as light curve data:

"""

spdata1 = phoebe.SPDataSet(context='spobs',filename='rotation',
                               ref='rotation',columns=['time','wavelength',
                               'flux','continuum','sigma'])
spdata2 = phoebe.SPDataSet(context='spobs',filename='temperature',
                               ref='temperature',columns=['time','wavelength',
                               'flux','continuum','sigma'])

"""
Now we need to tell the code that the light curve belongs to the whole system,
while the spectroscopic data only pertains to the separate components. You
could do that after initializing the Bodies, but since this is not advised
(you are recommended to initialize the Bodies after creating all ParameterSets
-- and I do mean *all* ParameterSets, also the DataSets), we do it upon
initialization here.

Every Body accepts the keyword ``data``. You should pass the data that belongs
to it. The spectra belong to each component, while the light curve belongs to
the combination of both (i.e. the BodyBag). Note that you are not required
to add data to all Bodies.

"""

star1 = phoebe.BinaryRocheStar(comp1,orbit=orbit,mesh=mesh,
                                 pbdep=[lcdep1a,lcdep1b,spdep],obs=[spdata1])
star2 = phoebe.BinaryRocheStar(comp2,orbit,mesh,
                                 pbdep=[lcdep2a,lcdep2b,spdep],obs=[spdata2])
system = phoebe.BodyBag([star1,star2],obs=[lcdata])

"""

Step 6: Computing observables (reprise)
---------------------------------------

The function ``compute_observables`` in ``observatory`` is meant as a
convenience function for unfirom, one-time calculations. In the case of
fitting data, this a bit cumbersome, because you light curves and spectra
are most of the time not taken simultaneously. Thus, you should iterate
over that function, for different time arrays for different observables.
Moreover, in that case of the data, you shouldn't need to manually specify
the time array because it can be readily derived from the data themselves.

For that reason, the function ``auto_compute`` was devised. The nice thing
about it is that you can MPI it. All arguments to this function need to be
given in parametersets. These specify how many subdivisions you want,
how many nodes you want to use etc...

"""

mpi = phoebe.ParameterSet(context='mpi')
params = phoebe.ParameterSet(context='compute')
phoebe.compute(system,params=params,mpi=mpi)

"""

Step 7: Fitting data
-------------------------------------

The previous step brings us in a nice position to start a fitting process: we
can automatically generate simulated  data that should perfectly reproduce 
the observations, there are only two things that stand in our way:

    1. We need to find the optimal value for the parameters
    2. We need to assess the quality of the simulated data with respect to the
       observations (i.e. the goodness-of-fit).

The first problem is solved by running Monte Carlo Markov Chains. The parameters
for these chains are also given by a ParameterSet:
"""
mcmc_params = phoebe.ParameterSet(context='fitting:mcmc',iter=3000)

"""

It is impossible to fit all the parameters that are defined, we better specify
which parameters we want to fit. This we do by setting the ``adjust`` flag
of all the parameters you want to fit. For the MCMC, you also need to set
the prior information you have on the parameter. If you don't know much, it
is better to set a reasonably wide uniform distribution:

"""

orbit.set_adjust(True,'incl')
orbit.get_parameter('incl').set_prior(distribution='uniform',lower=70.,upper=90.)

# Then we can run the sampler:

mc = phoebe.run(system,params=params,fitparams=mcmc_params,mpi=mpi)

# Finally, we can choose to use the parameters from MCMC run.

system.set_parameters_from_posterior()

# And perhaps compute the system again to generate the final accepted fit:
phoebe.compute(system,params=params,mpi=mpi)

# Save everything for future analysis
system.save('mybinary.phoebe')


# And we're done! I hope you had a good ride...
"""

.. tip:: Helpful hints

   1. First define all your ParameterSets, then create the ``Bodies`` and
   ``BodyBags``
   
   2. If you want to change parameters after creating the Bodies, do so in the
   original ParameterSets. It is not always straightforward e.g. to change the
   eccentricity of a particular orbit in a hierarchical system.
   
   3. If you change parameters after ``set_time`` has been called, it is not
   guarenteed that the system reflects those changes -- after all, ``set_time``
   tries to be smart and does not recomputing everything each time, so it
   could be that the parameter simply isn't used (e.g. the radius for a ``Star``).
   If you want the changes to propagate correctly, call ``body.reset()``
   but be aware that you then append any future results to the existing
   results. If you don't want that behaviour, make an additional call to ``body.clear_synthetic()``.

"""


