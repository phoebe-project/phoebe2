"""
How to find information in PHOEBE
=================================

Last updated: ***time***

In this tutorial, we want to take a second to show you how to access information properly in PHOEBE. To start, let's get a plain old 
binary fired up:
"""

import phoebe
eb = phoebe.Bundle()

"""
Instinctually, you might want to just try something like this:
"""

print eb

"""
We see that this produces a structured tree of everything contained in the bundle. Another way of doing this same thing would be:
"""

print eb.tree()

"""
For a bit more of an overview of what's going on in our Bundle, we would issue:
"""

print eb.summary()

"""
This seems much more aesthetically appealing, althought it doesn't divulge as much information as the previous method.
However, it stands to illustrate the format of the twig structure. Twigs are the components of the path that is used to 
access information in bundles. For example, if we want to know what parameters define the primary star, we would issue:
"""

print eb['component@primary@system']

"""
Tracing this back from the output of eb.summary(), we see that component belongs to both primary and secondary, so we have to clarify
which ParameterSet we want to access (we picked primary), and finally we see it's located in system. 
Within 'component@primary@system', we see a ton of parameters used to compute our model. To access one of these, just toss in on the 
font of your existing twig!
"""

print eb['teff@component@primary@system']

"""
This gives us a bit more information about the particular parameter we're interested in. To get to the value of our parameter we have
two options:
"""

print eb['value@teff@component@primary']

"""
or
"""

print eb.get_value('teff@component@primary')

"""
They both return the same thing, so you can pick and choose which you prefer.

Now if you were really paying attention earlier, you might have noticed a few parameters that exist in a multiple places. 

TO DO:
- Force all values
- set limits
"""

