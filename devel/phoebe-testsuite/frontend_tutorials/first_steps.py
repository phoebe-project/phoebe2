"""
First steps with PHOEBE 2.0-alpha
=================================

Last updated: ***time***

In this tutorial we will take a look at how to manipulate an already existing parameter file that is dear to your heart from PHOEBE 1.0 (or 0.3x). 
If you are starting afresh and have no PHOEBE parameter files lying around, you can just skim through this part and continue with part 2 where we 
build a binary from scratch; alternatively, grab our `demo parameter file for UW LMi <http://www.phoebe-project.org/files/UWLMi.tgz>`_ and you can type along with everyone else. Make sure you run python 
from a directory other than the distribution directory (otherwise importing phoebe will fail), and make sure that your (or our) parameter files are in that directory.

Let us start with the description of a bundle. Imagine all parameters, any data files, any generated plots, any auxiliary data, any notes, any 
interim results, any fit solutions, etc, bundled into a single collection. That's a bundle. It is a convenience structure that holds... well, everything 
related to your system. And best of all: old parameter sets can easily be transformed into a fresh and shiny new bundle.

So let us do that. Let us fire up python and import a parameter file as a bundle:
"""

import phoebe
eb = phoebe.Bundle('UWLMi.phoebe')

"""
That will import all parameter settings from the phoebe file and pack them nice and neat into the bundle called 'eb'. You can see a summary of the contents by issuing:
"""

print eb.summary()

"""
If you are interested in full detail of the bundle contents, you can use the tree() method:
"""

print eb.tree()

"""
or simply:
"""

print eb

"""
Don't be scared of the long output. While it likely looks daunting right now, printing bundles will become your everyday soon (at least while you are working interactively). But let us start with the summary view.

The first line of the summary is the name of your star system (UW_LMi [#f1]_ ), and next to it is its container type (BodyBag). BodyBags are collections of bodies (stars, accretion disks, UFOs, Ferraris, ...) in the system.

The following four lines list ParameterSets that are attached to the system. ParameterSets are collections of parameters grouped together according to their attributes. 
In particular, 'position' is a ParameterSet that collects all positional parameters (RA, Dec, proper motion, etc), 'orbit' collects all orbital parameters (semi-major axis, eccentricity, period, etc),
'lcobs' collects light curve data parameters (light curve timestamps, fluxes, a filter, etc.), and 'lcsyn' collects synthetic light curve parameters. To access the contents of the Parameter Set, i.e. 'position', you can issue:
"""

print eb.get_ps('position')

"""
or simply:
"""

print eb['position']

"""
That will list the parameters (e.g. 'ra'), their values ('0.0'), their units ('deg'), and their human-readable description ('Right ascension'). Those of you with attention to detail may have noticed
a single-character space between the units and the description. That is the parameter adjustment bit. If nothing is displayed (such as for 'ra'), then the parameter is not adjustable. If '-' is displayed
(such as for 'distance'), the parameter is adjustable but not marked for adjustment. If 'x' is displayed, the parameter is marked for adjustment ('vgamma' in our case). Also, if you noticed any of the 
parameters rendered in green in the tree() output, that has the same meaning. We will get back to that in a little while.

Under the parameter sets you can see all the bodies in the BodyBag. In this case, the two bodies are 'primary' and 'secondary'. Their container types (BinaryRocheStar) are given in the parentheses.
The BinaryRocheStar container implies that Roche geometry applies to the components. There are other containers, i.e. for single stars, accretion disks, Ferraris, UFOs, but we will deal with
those later. Each Body in the BodyBag has its own collection of ParameterSets. Thus, if you issue:
"""

print eb.get_object('primary').summary()

"""
You will see a summary output similar to that of the bundle. If you want to see more detail, just replace the .summary() part with .tree(). For all the gory detail, omit .summary() and .tree() altogether.
Again, these printouts appear overwhelming, but trust us, you will get used to them for interactive work, and you won't really care for them when scripting.

By now you may have guessed that you can achieve the same with the abbreviated statements:
"""

print eb['primary'].summary()
print eb['primary'].tree()
print eb['primary']

"""
A quick digression before we move on. You may have found the use of dictionary-style [#f2]_ manipulation of the bundle structure so convenient that using actual methods such as get_object() or get_ps() 
might seem redundant. In general, you would be correct, however what the method-style approach buys you is an added level of error handling. For example, doing something like this:
"""

ps = eb['primary']

"""
will work just fine, if assigning the Body to the variable 'ps' is what you wanted to do. If, however, you wanted to attach a parameter set that belongs to the 'primary', then the method-based call:
"""

ps = eb.get_ps('primary')

"""
would fail since 'primary' is not a Parameter Set. Hence, if you are good at keeping track of what is what, the dictionary-style approach is probably your best choice; on the other hand, 
if you prefer an added level of error-checking, you might be better off with the method-style approach.

With this digression out of the way, let us take a look at the parameters. Issue:
"""

print eb['orbit']

"""
Focus, for example, on the 'incl' parameter. You can get its description by running:
"""

print eb.get_parameter('incl@orbit')

"""
and its value by running:
"""

print eb.get_value('incl@orbit')

"""
Here we used the '@' sign to identify a Parameter Set to which a parameter belongs. In this case, 'incl' belongs to 'orbit', thus the parameter is identified as 'incl@orbit'. Now check this out: 
if the parameter is unique, such as orbital inclination in a binary, you can omit the Parameter Set, so you can do just this:
"""

print eb.get_value('incl')

"""
or even simpler:
"""

print eb['incl']

"""
So how would that work for, say, effective temperature? Each component in a binary has its own effective temperature, so running:

::

    print eb['teff']


won't work (try it!). The interpreter returns an error message and lists all parameters that contain 'teff', in this case two. But what the heck? Those parameters don't look anything like teff@primary, but rather:

teff@component@primary@system@Bundle
teff@component@secondary@system@Bundle

Each parameter is thus fully qualified with its complete ancestry: 'teff' belongs to the 'component' ParameterSet, that is a part of the 'primary' Body, that is in turn a part of the 'system' 
that lies in the Bundle. This full ancestry is what we call a twig, and twigs uniquely determine parameters. Embrace twigs. They are your friends.

Obviously you will not be typing whole twigs to identify parameters, but you need the smallest unique sequence (hence the name twig in the first place). You can omit most of the ancestry, e.g.
'teff@primary' and 'teff@secondary' would uniquely qualify your parameters. Thus, to get the temperatures, you would issue:
"""

print eb.get_value('teff@primary'), eb.get_value('teff@secondary')

"""
or:
"""

print eb['teff@primary'], eb['teff@secondary']

"""
If you are running a native python interpreter (as opposed to, say, ipython), you'll be happy to hear that TAB completion works on twigs. So if you type:

::

    print eb.get_value('tef



and then hit TAB, it will auto-complete the twig to 'teff@component@'; if you hit TAB twice, it will list all available options; typing 'p' or 's' and hitting TAB 
one last time will complete the twig. But hey, you've used TAB completion before, so we'll stop wasting your time. This works for both method-style and dictionary-style approach, so you're in luck either way.

Now let us change the values of the parameters. As you can imagine, both of the following approaches work:
"""

eb.set_value('teff@primary', 10000)
eb['teff@primary'] = 10000

"""
Another thing that you might find handy at times is automatic units conversion. If, say, you wanted to pass the temperature in Fahrenheit (please don't -- we're just trying to make a point here, not encourage evil units), you could do:
"""

eb.set_value('teff@primary', 12000, 'Far')
eb['teff@primary'] = (12000, 'Far')
print eb['teff@primary']

"""
All supported units and their abbreviations are listed in the :ref:`units module documentation <phoebe.units>`.

Well this was a piece of cake; but sometimes the parameters are something other than single numeric values. For example, there are several supported treatments of gravity brightening in PHOEBE, 
and you can only select one of the predefined ones. If you do:
"""

print eb.info('gravblaw@primary')

"""
you will get all the information on that parameter. The 'Allowed values' line lists all available options, in this case 'zeipel', 'espinosa' and 'claret'. If you wanted to set the value, you would do:
"""

eb['gravblaw@primary'] = 'zeipel'

"""
If you are particularly bad at typing (or lazy like us), you could assign only 'zei' or even 'z', for as long as it is unique, and it will still work.

Yet other parameters can take either keywords or filenames. For example, model atmospheres can either be taken from internal tables provided with PHOEBE, or user-provided. Issuing:
"""

print eb.info('atm@component@primary')

"""
will describe the parameter and explain what values are allowed/expected, and you can see what you can pass to it and make it out in one piece.

Alright, we beat around the bush long enough. Let's now compute a light curve and plot it. We will presume that your parameter file already defines at least one light curve and that those data are available;
if you are using our UW LMi example, there are three light curves and two radial velocity curves, so you are all set.

The parameters that determine the computation of the model reside in the top-level bundle ParameterSets called 'compute'. Note that the plural in ParameterSets is not a typo: looking at the summary() output,
you can see that 'compute' has 3 children: 'detailed', 'preview', and 'legacy'. Each of these is an individual ParameterSet that contains parameters for the three computation schemes (we'll come to that in a minute).
Thus, any branch in the summary() output that has sub-branches (such as 'lcobs', 'lcdep' and 'compute') is not a ParameterSet itself but a collection of ParameterSets. To make this distinction clear, all collections
of ParameterSets are italicized in the summary() output (that is, if you're cool enough to be running an ANSI-capable terminal).

In PHOEBE, these sets are held in a dictionary structure. You can access them with:
"""

print eb.get_ps_dict('compute')

"""
or
"""

print eb['compute']

"""
Since these are dictionaries, you can print out an individual ParameterSet, say 'detailed', by:
""" 

print eb['compute']['detailed']

"""
or, if you wanted to list them all (you are familiar with python, right?):
"""

for ps in eb['compute'].values():
    print ps
    
"""
This will also give you an idea of what the 'detailed', 'preview' and 'legacy' ParameterSets are all about. The 'detailed' ParameterSet will compute theoretical curves in full detail,
with heating, reflection, light time travel effect, beaming and subdivision all turned on. The 'preview' ParameterSet will compute theoretical curves in a cursory manner, with all of
the aforementioned effects turned off. Finally, the 'legacy' ParameterSet matches the legacy functionality as closely as possible: options that are available in PHOEBE 0.3 and 1 are turned on,
and previously unsupported options are turned off. This achieves maximum compatibility with the legacy version.

Sounds scary? No reason to fret, twigs come to the rescue! To set any of the parameter values in any of these ParameterSets, use exactly the same syntax as before. For example, if you
want to turn heating off in the 'legacy' ParameterSet, you would do:
"""

eb['heating@legacy'] = False

"""
With that out of the way, let us compute [#f3]_ a light curve using the 'preview' ParameterSet. To do that, we use run_compute():
"""

eb.run_compute('preview')

"""
You may find it strange that there is no output (assuming you have the required atmosphere files -- otherwise PHOEBE will yell at you); this method clears any synthetic data from before, and computes a synthetic light curve.

So where are the data? At what timestamps have the fluxes been computed? How to find them? To explain this, we need to discuss three structures that deal with data and computations: 'lcobs', 'lcdep' and 'lcsyn'. 
And what better way of doing that than on an example?

Much like we listed the 'compute' options before, we can list all light curve data defined in the system:
"""

print eb['lcobs@UW_LMi']
print eb['lcobs@UW_LMi']['APT_B']
print eb['lcobs@UW_LMi']['APT_V']
print eb['lcobs@UW_LMi']['APT_Ic']

"""
You could of course deduce this from the summary() output, but we figured it may be useful to reiterate how you traverse ParameterSet dictionaries. You can see that we have three light 
curves, 'APT_B', 'APT_V' and 'APT_Ic'. In case you were wondering, these names are user-defined, named from the observations acquired by the Automated Photoelectric Telescope (APT) in Johnson B, V and Cousins
I passbands. You are of course free to change them to whatever you like. These references are used to connect the observed data parameters stored in 'lcobs' to passband-dependent model parameters stored in 'lcdep' 
and to synthetic data parameters stored in 'lcsyn'. Thus, these references are defined for each of these structures, and you can see them if you run the following:
"""

print eb.twigs('APT')

"""
A tiny digression, since we haven't introduced the twigs() method before. Calling it with no arguments will list all defined twigs within the bundle. Passing an argument will list all the 
twigs that start with the passed "twiglet". (Yes. We're nerds. We know.) The "twiglet" is the upper-most part of the twig, i.e. everything till the first '@' sign. While on the subject, if you want
to search through the twigs, use the search() method, where you can pass any string and the method will list all the twigs that contain the search string, i.e.:
"""

print eb.search('APT')

"""
Back to the topic at hand; let us focus on a single lightcurve, say 'APT_V'. Its reference connects the relevant 'lcdep', 'lcobs' and 'lcsyn' structures. Let us inspect their contents,
starting with 'lcdep'. This is where all passband-dependent parameters are stored. Let's print them.
"""

print eb['APT_V@lcdep@primary']
print eb['APT_V@lcdep@secondary']


"""
These are model parameters that provide PHOEBE with necessary information to compute the light curve. For example, the 'atm' parameter sets the atmosphere model, 'ld_func' sets the limb darkening
modeling function, 'ld_coeffs' contains the corresponding limb darkening coefficients, etc.

Next, 'lcobs'. Issue:
"""

print eb['APT_V@lcobs']

"""
This will display a complete ParameterSet for your observations. The timestamps are in the 'time' parameter, fluxes in 'flux', and sigmas in 'sigma'. We will defer the discussion of other
fields to another document that deals specifically with data handling, so the take-home message here is that 'APT_V@lcobs' stores all your observed data.

On to 'lcsyn'. This structure contains all computed parameters. The 'time' parameter will by default be copied from the 'lcobs' structure, so the light curve will by default be computed at
the same timestamps. If you'd prefer to pass your own array of times, the best way would be to generate a new 'lcobs' ParameterSet, but let's not get ahead of ourselves -- we will show you how 
to do that in the :doc:`next tutorial <tutorial.first_binary_from_scratch>`. The 'flux' parameter contains the computed fluxes.

This should have given you just about enough information to whip out your matplotlib skills and make your first plot.
"""

import matplotlib.pyplot as plt
plt.plot(eb['time@APT_V@lcobs'], eb['flux@APT_V@lcobs'], 'b.')
plt.plot(eb['time@APT_V@lcsyn'], eb['flux@APT_V@lcsyn'], 'r-')
plt.show()

"""
Yep, it truly is as easy as that. PHOEBE comes with several helper functions dedicated for plotting, and they work in conjunction with matplotlib. To achieve the same thing as above, you would do:
"""

eb.plot_obs('APT_V', fmt='b.')
eb.plot_syn('APT_V', fmt='r-')
plt.show()

"""
You may wonder how you would plot phased curves; very easily, actually -- by providing a phased keyword:
"""

eb.plot_obs('APT_V', phased=True, fmt='b.')
eb.plot_syn('APT_V', phased=True, fmt='r-')
plt.show()

"""
So cmon, take PHOEBE for a whirl! Change some parameters, compute light curves, see how close you can get by manually modifying the parameters.

Congratulations, you survived your first PHOEBE 2.0-alpha tutorial! You can now catch a break, poke around PHOEBE with your newly acquired skills, or if you are particularly insatiable, you can continue onto 
the next tutorial, :doc:`Building my first binary from scratch with PHOEBE 2.0-alpha <tutorial.first_binary_from_scratch>`.

.. rubric::  Footnotes

.. [#f1] It is considered bad practice to use spaces in the system name; that is why all whitespaces (spaces, tabs, carriage returns) are changed into underscores.
.. [#f2] Just to clarify: eb['primary'] is a dictionary-style approach and eb.get_object('primary') is a method-style approach.
.. [#f3] To compute a light curve, you need to have model atmosphere files in place. If you followed our installation instructions, you already have them; if not, start another terminal, run python as root (sudo python), import phoebe and issue phoebe.download_atm(). That will download and set up the atmosphere files for you. 
"""
