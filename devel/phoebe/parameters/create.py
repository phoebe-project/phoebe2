"""
Library containing parameters of known stars, systems and objects.

This module contains one-liners to create default parameterSets and Bodies
representing well-known or often-used objects such as the Sun, Vega, Sirius,
or specific binary systems. It also provides L{spectral_type} function for
easy creation of generic stars of a given spectral type and luminosity class.

By default, these functions return C{ParameterSet} instances. However, most of
them also accept an extra keyword argument C{create_body}. When set to C{True},
a C{Body} instance will be returned, probably also containing a C{mesh}
ParameterSet and observables (C{lcdep, spdep...}). You can change the defaults
of some of these extra parameterSets by providing extra keyword arguments.

Some functions can be used to validate ParameterSet (specifically for binary
systems), and some functions can create Binaries from the information provided
by single stars, plus extra information (e.g. spectroscopic observables in
L{binary_from_spectroscopy} or a separation via L{binary_from_stars}).

**General functions**

.. autosummary::

    from_library
    star_from_spectral_type
    binary_from_stars
    binary_from_spectroscopy
    dep_from_object
    
**Generic systems**

.. autosummary::

    binary
    overcontact
    hierarchical_triple
    pulsating_star
    binary_pulsating_primary
    binary_pulsating_secondary
    close_beta_cephei
    
**Specific targets**    
    
.. autosummary::    

    mu_Cas
    T_CrB
    KOI126
    vega_monnier2012
    
    
    

    

**Example usage:**

Only involving ParameterSets:

>>> the_sun_pars = from_library('sun')
>>> print(the_sun_pars)
      teff 5779.5747            K - phoebe Effective temperature
    radius 1.0               Rsol - phoebe Radius
      mass 1.0               Msol - phoebe Stellar mass
       atm kurucz              --   phoebe Bolometric Atmosphere model
 rotperiod 22.0                 d - phoebe Polar rotation period
   diffrot 0.0                  d - phoebe (Eq - Polar) rotation period (<0 is solar-like)
     gravb 1.0                 -- - phoebe Bolometric gravity brightening
  gravblaw zeipel              --   phoebe Gravity brightening law
      incl 82.75              deg - phoebe Inclination angle
      long 0.0                deg - phoebe Orientation on the sky (East of North)
  distance 4.84813681108e-06   pc - phoebe Distance to the star
     shape sphere              --   phoebe Shape of surface
    vgamma 0.0               km/s - phoebe Systemic velocity
       alb 1.0                 -- - phoebe Bolometric albedo (alb heating, 1-alb reflected)
    redist 0.0                 -- - phoebe Global redist par (1-redist) local heating, redist global heating
irradiator False               --   phoebe Treat body as irradiator of other objects
      abun 0.0                 --   phoebe Metallicity
     label Sun                 --   phoebe Name of the body
   ld_func claret              --   phoebe Bolometric limb darkening model
 ld_coeffs kurucz              --   phoebe Bolometric limb darkening coefficients
  surfgrav 274.351532944      n/a   constr constants.GG*{mass}/{radius}**2
  
Upon creation, you can override the values of the parameters directly via
extra keyword arguments:
  
>>> myAstar_pars = star_from_spectral_type('A0V',label='myA0V')
>>> print(myAstar_pars)
      teff 9549.92586       K - phoebe Effective temperature
    radius 2.843097      Rsol - phoebe Radius
      mass 4.202585      Msol - phoebe Stellar mass
       atm kurucz          --   phoebe Bolometric Atmosphere model
 rotperiod 0.821716         d - phoebe Polar rotation period
   diffrot 0.0              d - phoebe (Eq - Polar) rotation period (<0 is solar-like)
     gravb 1.0             -- - phoebe Bolometric gravity brightening
  gravblaw zeipel          --   phoebe Gravity brightening law
      incl 90.0           deg - phoebe Inclination angle
      long 0.0            deg - phoebe Orientation on the sky (East of North)
  distance 10.0            pc - phoebe Distance to the star
     shape equipot         --   phoebe Shape of surface
    vgamma 0.0           km/s - phoebe Systemic velocity
       alb 1.0             -- - phoebe Bolometric albedo (1-alb heating, alb reflected)
    redist 0.0             -- - phoebe Global redist par (1-redist) local heating, redist global heating
irradiator False           --   phoebe Treat body as irradiator of other objects
      abun 0.0             --   phoebe Metallicity
     label myA0V           --   phoebe Name of the body
   ld_func claret          --   phoebe Bolometric limb darkening model
 ld_coeffs kurucz          --   phoebe Bolometric limb darkening coefficients
  surfgrav 142.639741491  n/a   constr constants.GG*{mass}/{radius}**2

When creating binaries from stars, you can set the extra arguments also directly,
but since there are three parameterSets to be created, you need to give them
in three dictionaries, names ``kwargs1`` (for the primary), ``kwargs2`` (for
the secondary) and ``orbitkwargs`` (for the orbit). The only exception are
the semi major axis ``sma`` and period ``period``, one of which is required
(but not both) to create the system.
  
>>> Astar = star_from_spectral_type('A0V', label='myA0V')
>>> Bstar = star_from_spectral_type('B3V', label='myB3V')
>>> comp1,comp2,orbit = binary_from_stars(Astar, Bstar, sma=(10,'Rsol'), 
...                                       orbitkwargs=dict(label='myorbit'))
>>> print(comp1)
       atm kurucz         --   phoebe Bolometric Atmosphere model
       alb 1.0            -- - phoebe Bolometric albedo (alb heating, 1-alb reflected)
    redist 0.0            -- - phoebe Global redist par (1-redist) local heating, redist global heating
   syncpar 1.30628493224  -- - phoebe Synchronicity parameter
     gravb 1.0            -- - phoebe Bolometric gravity brightening
       pot 5.21866195357  -- - phoebe Roche potential value
      teff 9549.92586      K - phoebe Mean effective temperature
irradiator False          --   phoebe Treat body as irradiator of other objects
      abun 0.0            --   phoebe Metallicity
     label myA0V          --   phoebe Name of the body
   ld_func claret         --   phoebe Bolometric limb darkening model
 ld_coeffs kurucz         --   phoebe Bolometric limb darkening coefficients
>>> print(comp2)
       atm kurucz         --   phoebe Bolometric Atmosphere model
       alb 1.0            -- - phoebe Bolometric albedo (alb heating, 1-alb reflected)
    redist 0.0            -- - phoebe Global redist par (1-redist) local heating, redist global heating
   syncpar 1.03283179352  -- - phoebe Synchronicity parameter
     gravb 1.0            -- - phoebe Bolometric gravity brightening
       pot 4.84468313898  -- - phoebe Roche potential value
      teff 19054.60718     K - phoebe Mean effective temperature
irradiator False          --   phoebe Treat body as irradiator of other objects
      abun 0.0            --   phoebe Metallicity
     label myB3V          --   phoebe Name of the body
   ld_func claret         --   phoebe Bolometric limb darkening model
 ld_coeffs kurucz         --   phoebe Bolometric limb darkening coefficients
>>> print(orbit)
     dpdt 0.0                  s/yr - phoebe Period change
   dperdt 0.0                deg/yr - phoebe Periastron change
      ecc 0.0                    -- - phoebe Eccentricity
       t0 0.0                    JD - phoebe Zeropoint date
   t0type periastron passage     --   phoebe Interpretation of zeropoint date
     incl 90.0                  deg - phoebe Inclination angle
    label myorbit                --   phoebe Name of the system
   period 1.07339522938           d - phoebe Period of the system
     per0 90.0                  deg - phoebe Periastron
  phshift 0.0                    -- - phoebe Phase shift
        q 1.76879729976          -- - phoebe Mass ratio
   vgamma 0.0                  km/s - phoebe Systemic velocity
      sma 10.0                 Rsol - phoebe Semi major axis
  long_an 0.0                   deg - phoebe Longitude of ascending node
  c1label myA0V                  --   phoebe ParameterSet connected to the primary component
  c2label myB3V                  --   phoebe ParameterSet connected to the secondary component
 distance 10.0                   pc - phoebe Distance to the binary system
     sma1 4443130136.21         n/a   constr {sma} / (1.0 + 1.0/{q})
     sma2 2511949863.79         n/a   constr {sma} / (1.0 + {q})
totalmass 2.3138943678e+31      n/a   constr 4*pi**2 * {sma}**3 / {period}**2 / constants.GG
    mass1 8.35703779398e+30     n/a   constr 4*pi**2 * {sma}**3 / {period}**2 / constants.GG / (1.0 + {q})
    mass2 1.4781905884e+31      n/a   constr 4*pi**2 * {sma}**3 / {period}**2 / constants.GG / (1.0 + 1.0/{q})
    asini 6955080000.0          n/a   constr {sma} * sin({incl})
      com 4443130136.21         n/a   constr {q}/(1.0+{q})*{sma}
       q1 1.76879729976         n/a   constr {q}
       q2 0.565355906036        n/a   constr 1.0/{q}

Creating Bodies:

>>> the_sun_body = from_library('sun', create_body=True)
>>> myAstar_body = star_from_spectral_type('A0V', create_body=True)
>>> the_system = binary_from_stars(Astar, Bstar, period=(20., 'd'), create_body=True)

"""
import logging
import os
import glob
import functools
import inspect
import re
import numpy as np
from phoebe.parameters import parameters
from phoebe.parameters import tools
from phoebe.backend import universe
from phoebe.dynamics import keplerorbit
from phoebe.atmospheres import roche
from phoebe.units import conversions
from phoebe.units import constants
from phoebe.io import ascii
from phoebe.io import parsers


logger = logging.getLogger('PARS.CREATE')
logger.addHandler(logging.NullHandler())


class GenericBody(object):
    """
    Automatically build the right body from the input.
    
    Try to derive which is Body needs to be instantiated from the given set of
    arguments, and return it.
    
    This class tries to be a smart as possible, and should return useful
    information if no body can be created:
    
        - the list of arguments could be not unique. Then, the list of ambiguous
          bodies will be returned, and you are required to create them explicitly
        - the list of arguments is not enough to create any Body: the closest
          match will be returned, stating which arguments were given, which are
          missing and which are given but are not allowed.
    
    """
    def __new__(self, *args, **kwargs):
        #-- check which contexts are given
        pbdep = kwargs.pop('pbdep',[])
        if kwargs:
            raise NotImplementedError("GenericBody cannot handle given keyword arguments yet")
        contexts = []
        contexts_args = []
        other_arguments = []
        for arg in args:
            if hasattr(arg,'context') and arg.context in ['spdep','ifdep','lcdep','rvdep']:
                pbdep.append(arg)
            elif hasattr(arg,'context'):
                contexts.append(arg.context)
                contexts_args.append(arg)
            else:
                other_arguments.append(arg)
        
        #-- run over all things defined in the Universe, and check which ones
        #   are actually a class
        info = {}
        match = []
        
        # List all possible Body classes:
        body_classes = []
        for body_class in dir(universe):
            try:
                if issubclass(getattr(universe, body_class), universe.Body) \
                     and not body_class in ['Body', 'BodyBag', 'BinaryBag']:
                    body_classes.append(body_class)
            except TypeError:
                continue
    
        
        for body_class in body_classes:
            thing = getattr(universe, body_class)
            info[body_class] = {'required':[],'missing':[],'rest':[],'optional':{}}
            #-- if we have a class, check what arguments it needs
            argspec = inspect.getargspec(thing.__init__)
            arg_names = argspec[0][1:] # we can forget about the self arg
            arg_defaults = argspec[3]
            #-- split the arguments into two categories: those that are
            #   required, and those that are not.
            if arg_defaults:
                arg_required = list(arg_names[:-len(arg_defaults)])
                #============ >PYTHON 2.7 ===============================
                arg_optional = {name:default for name,default in zip(arg_names[-len(arg_defaults):],arg_defaults)}
                #============ <PYTHON 2.7 ===============================
                #arg_optional = {}
                #for name,default in zip(arg_names[-len(arg_defaults):],arg_defaults):
                #    arg_optional[name] = default
                #========================================================
            else:
                arg_required = arg_names
                arg_optional = {}
            if pbdep:
                info[body_class]['optional']['pbdep'] = pbdep
            #-- absorb given keyword arguments into optional arguments
            for key in kwargs:
                if key in arg_optional:
                    arg_optional[key] = kwargs[key]
            #-- then check which contexts are given. Make a local copy of
            #   the contexts that can be exhausted. For all arguments that
            #   are required, we check which ones are missing, which ones
            #   are given, and in the end which ones were given but not
            #   required.
            contexts_ = [context for context in contexts] 
            for par in arg_required:
                #-- if the required but missing argument is a mesh,
                #   create one!
                if not par in contexts and par=='mesh':    
                    info[body_class]['required'].append(parameters.ParameterSet(context='mesh:marching'))
                #-- else, it's just missing and we don't make any assumptions
                elif not par in contexts:
                    info[body_class]['missing'].append(par)
                else:
                    par = contexts_.pop(contexts_.index(par))
                    info[body_class]['required'].append(contexts_args[contexts.index(par)])
            #-- for optional arguments, we only need to check if they are
            #   in the contextsreturn star
    
    
    
            for par in arg_optional:
                if par=='mesh' and not par in contexts:
                    info[body_class]['optional']['mesh'] = parameters.ParameterSet(context='mesh:marching')
                if par in contexts:
                    par_ = contexts_.pop(contexts_.index(par))
                    info[body_class]['optional'][par] = contexts_args[contexts.index(par_)]
            info[body_class]['rest'] = contexts_
            if not info[body_class]['missing'] and not info[body_class]['rest'] and len(contexts_)==0:
                match.append(body_class)
        #-- Evaluate and return a Body if possible. Else, try to give some
        #   information about what went wrong
        if len(match)==1:
            logger.info('Input matches {}: creating {} body'.format(match[0],match[0]))
            #-- merge given keyword arguments with derived keyword arguments
            return getattr(universe,match[0])(*info[match[0]]['required'],**info[match[0]]['optional'])
        elif len(match)>1:
            logger.debug("Given set of arguments is ambiguous: could match any Body of {}".format(", ".join(match)))
            if 'Star' in match:
                mymatch = 'Star'
            elif 'BinaryRocheStar' in match:
                mymatch = 'BinaryRocheStar'
            else:
                mymatch = match[0]
            logger.info("Creating Body of type {}".format(mymatch))
            return getattr(universe,mymatch)(*info[mymatch]['required'],**info[mymatch]['optional'])
        else:
            best_match = '<no close match found>'
            nr_match = 0
            for body in info:
                if len(info[body]['required'])>nr_match:
                    best_match = body
                    nr_match = len(info[body]['required'])
            message = "No Body found matching the input arguments. Best match: {}".format(best_match)
            m1 = [arg.context for arg in info[best_match]['required']]
            m2 = info[best_match]['missing']
            m3 = info[best_match]['rest']
            if m1: message = message + " -- required and given args: {}".format(", ".join(m1)) 
            if m2: message = message + " -- required but missing args: {}".format(", ".join(m2))
            if m3: message = message + " -- does not accept extra args: {}".format(", ".join(m3))
            raise ValueError(message)


def make_body_from_parametersets(ps):
    if not isinstance(ps,tuple) and not isinstance(ps,list):
        ps = list([ps])
    elif isinstance(ps, tuple):
        ps = list(ps)
    contexts = [ips.context for ips in ps]
    
    # Remove positional ps to add later on
    pos_ps = ps.pop(contexts.index('position')) if 'position' in contexts else None
    if pos_ps is not None:
        contexts.remove('position')
    
    #-- let's create two bodies and put them in a BodyBag
    if len(ps)==3 and contexts.count('component')==2 and contexts.count('orbit')==1:
        orbit = ps[contexts.index('orbit')]
        comp1 = ps[contexts.index('component')]
        comp2 = ps[::-1][contexts[::-1].index('component')]
        #-- find out which body needs to be made
        body1 = GenericBody(comp1,orbit)
        body2 = GenericBody(comp2,orbit)
        #-- if there are irradiators, we need to prepare for reflection
        #   results
        if comp1['irradiator'] or comp2['irradiator']:
            body1.prepare_reflection()
            body2.prepare_reflection()
        #-- put the two Bodies in a BodyBag and return it to the user.
        body = universe.BodyBag([body1,body2],label=orbit['label'], position=pos_ps)
    #-- OK, if it's no binary, we assume we can deal with it as is
    else:
        if pos_ps is not None:
            ps = ps + [pos_ps]
        body = GenericBody(*ps)
    
    return body


def body_from_string(body_string):
    """
    Create a Body from a string representation.
    
    Possibilities:
    
        - string represents file name from the library (or local file)
        - string represents a predefined system from this module
        - string represents a spectral type
        
    """
    
    # First try to see if the system is recognised by the library, or if it's
    # predefined in this module
    try:
        body = from_library(body_string, create_body=True)
        return body
    except NotImplementedError:
        pass
    
    # Then try to create a spectral type
    try:
        body = star_from_spectral_type(body_string, create_body=True)
        return body
    except ValueError:
        pass
    
    


def from_library(name, create_body=False):
    """
    Create stars and objects from the library.
    
    In case of a single star, this function returns a single parameterSet.
    In case of a double star, this function returns the primary, secondary and
    orbital parameters.
    
    In case C{name} is a more complex system, it is best to not automatically
    create a body or to call one of the dedicated functions.
    
    Example usage::
    
        >>> vega = from_library('Vega')
        
    Extra keyword arguments will be passed to the creation of the parameterSets,
    e.g.::
    
        >>> vega = from_library('Vega', rotperiod=(5., 'd'))
    
    @param name: name of the object or system
    @type name: str
    @param create_body: flag to automatically create a Body from the parameters
    @type create_body: bool
    @return: parameterSets or Body
    @rtype: parameterSets of Body
    """
    # if the argument is not a filename, try to retrieve the star from the
    # local library
    if not os.path.isfile(name):
        files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'library','*.par'))
        filesl = [os.path.basename(ff).lower() for ff in files]
        if name.lower()+'.par' in filesl:
            name = files[filesl.index(name.lower()+'.par')]
    if not os.path.isfile(name) and name in globals():
        return globals()[name](create_body=create_body)
    if not os.path.isfile(name):
        raise NotImplementedError("system not in library")
    ps = parameters.load_ascii(name)
    logger.info("Loaded {} from library".format(name))
    
    if create_body:
        ps = make_body_from_parametersets(ps)
        return ps
    elif len(ps) == 1:
    #-- we have a few special cases here:
    #-- perhaps we're not creating bodies. Then, if the ASCII files contains
    #   only one parameterSet, we just return that one
        return ps[0]
    #-- else, we return the list
    else:
        return ps

 
def star_from_spectral_type(spectype, create_body=False, **kwargs):
    """
    Create a star with a specific spectral type.
    
    Spectral type information is taken from [Pickles1998]_ and [Habets1981]_.
    
    Rotational velocities are taken from [McNally1965]_.
        
    This function is not meant as a calibration of some sorts, it only returns
    approximate parameters for a star of a given spectral type. I mean
    approximate in the B{really} broad sense.
    
    If the spectral type is not recognised, a list of allowed spectral types
    will be returned.
    
    >>> Astar = star_from_spectral_type('A0V')
    >>> print(Astar['teff'])
    9549.92586
    >>> print(Astar['mass'])
    4.202585
    >>> print(Astar['radius'])
    2.843097
    >>> print(Astar['rotperiod'])
    0.821716
    
    **Tables of recognised spectral types and parameters**
    
    **Dwarfs**::
    
        SpType           teff         lumi      rad     mass     Mbol   rotperiod
        O5V         39810.717   288404.435   11.319   35.670   -8.900       3.013
        O9V         35481.339   104713.321    8.586   23.257   -7.800       2.286
        B0V         28183.829    18197.090    5.673   12.244   -5.900       1.435
        B1V         22387.211     6025.623    5.174   10.617   -4.700       1.308
        B3V         19054.607     1995.271    4.110    7.434   -3.500       1.039
        B57V        14125.375      549.543    3.925    6.922   -2.100       0.992
        B8V         11748.976      239.884    3.748    6.446   -1.200       0.948
        B9V         10715.193      151.357    3.579    6.002   -0.700       0.905
        A0V          9549.926       60.256    2.843    4.203    0.300       0.822
        A2V          8912.509       28.840    2.258    2.943    1.100       0.653
        A3V          8790.225       23.988    2.117    2.663    1.300       0.612
        A5V          8491.805       16.596    1.887    2.228    1.700       0.545
        A7V          8053.784       11.482    1.745    1.974    2.100       0.504
        F0V          7211.075        5.012    1.438    1.463    3.000       1.212
        F2V          6776.415        3.802    1.418    1.432    3.300       1.196
        F5V          6531.306        3.162    1.392    1.392    3.500       1.174
        wF5V         6606.934        2.399    1.185    1.085    3.800       0.999
        F6V          6280.584        2.399    1.312    1.269    3.800       1.106
        rF6V         6109.420        3.802    1.745    1.974    3.300       1.471
        F8V          6039.486        1.995    1.294    1.242    4.000       1.090
        wF8V         6137.620        1.514    1.091    0.954    4.300       0.920
        rF8V         5767.665        2.630    1.629    1.774    3.700       1.373
        G0V          5807.644        1.660    1.276    1.216    4.200       5.377
        wG0V         5847.901        1.047    1.000    0.833    4.700       4.213
        rG0V         5636.377        2.399    1.629    1.774    3.800       6.864
        G2V          5636.377        1.148    1.127    1.003    4.600       4.749
        G5V          5584.702        1.047    1.096    0.961    4.700       4.619
        wG5V         5754.399        0.794    0.899    0.707    5.000       3.789
        rG5V         5597.576        2.188    1.577    1.687    3.900       6.646
        G8V          5333.349        0.724    1.000    0.833    5.100       4.213
        K0V          5188.000        0.550    0.920    0.733    5.400       9.307
        rK0V         5236.004        1.047    1.247    1.173    4.700      12.612
        K2V          4886.524        0.347    0.824    0.618    5.900       8.333
        K3V          4497.799        0.219    0.772    0.559    6.400       7.812
        K4V          4345.102        0.166    0.721    0.502    6.700       7.291
        K5V          4187.936        0.126    0.676    0.455    7.000       6.836
        K7V          3999.447        0.087    0.616    0.394    7.400       6.234
        M0V          3801.894        0.079    0.651    0.429    7.500      16.471
        M1V          3681.290        0.060    0.605    0.383    7.800      15.301
        M2V          3548.134        0.046    0.567    0.347    8.100      14.346
        M2.5V        3443.499        0.035    0.525    0.307    8.400      13.266
        M3V          3303.695        0.026    0.496    0.282    8.700      12.552
        M4V          3111.716        0.018    0.465    0.255    9.100      11.769
        M5V          2951.209        0.013    0.430    0.226    9.500      10.883
        M6V          2697.739        0.008    0.409    0.209   10.000      10.345

    **Subgiants**::
    
        SpType           teff         lumi       rad     mass     Mbol   rotperiod
        O8III       31622.777   239884.361    16.360   84.642   -8.700     220.873
        B12III      19952.623    18197.090    11.319   40.512   -5.900     151.145
        B3III       16982.437     5495.433     8.586   23.312   -4.600     114.361
        B5III       14791.084     1995.271     6.820   14.709   -3.500      90.668
        B9III       11091.748      263.028     4.403    6.132   -1.300      59.481
        A0III        9571.941      114.816     3.907    4.826   -0.400      53.213
        A3III        8974.288       60.256     3.220    3.278    0.300      44.886
        A5III        8452.788       50.119     3.310    3.464    0.500      47.241
        A7III        8053.784       31.623     2.896    2.652    1.000      42.978
        F0III        7585.776       28.840     3.117    3.073    1.100      48.527
        F2III        6839.116       18.197     3.046    2.935    1.600      58.421
        F5III        6531.306       18.197     3.340    3.528    1.600      75.544
        G0III        5610.480       12.589     3.765    4.483    2.000      95.220
        G5III        5164.164        4.169     2.557    2.068    3.200      64.674
        wG5III       5188.000        6.026     3.046    2.935    2.800      77.042
        rG5III       5116.818        3.467     2.376    1.785    3.400      60.080
        G8III        5011.872        4.571     2.843    2.556    3.100      71.900
        wG8III       5023.426        6.607     3.402    3.661    2.700      86.046
        K0III        4852.885        4.571     3.032    2.500    3.100      76.688
        wK0III       4977.371        7.943     3.800    2.500    2.500      96.101
        rK0III       4819.478        3.802     2.804    2.500    3.300      70.914
        K1III        4655.861       15.136     5.995    2.500    1.800     151.611
        wK1III       4819.478       26.303     7.376    2.500    1.200     186.522
        rK1III       4591.980        4.169     3.234    2.500    3.200      81.796
        K2III        4456.562       34.674     9.904    2.500    0.900     250.455
        wK2III       4634.469       41.687    10.041    2.500    0.700     253.939
        rK2III       4446.313        7.943     4.762    2.500    2.500     120.429
        K3III        4365.158       50.119    12.411    2.500    0.500     313.856
        wK3III       4415.704       95.500    16.741    2.500   -0.200     423.380
        rK3III       4375.221       18.197     7.444    2.500    1.600     188.248
        K4III        4226.686      125.893    20.979    2.500   -0.500     530.554
        wK4III       4246.196      199.527    26.169    2.500   -1.000     661.805
        rK4III       4226.686       45.709    12.641    2.500    0.600     319.691
        K5III        4008.667      263.028    33.713    2.500   -1.300     852.570
        rK5III       3990.249      104.713    21.468    2.500   -0.300     542.913
        M0III        3819.443      457.090    48.954    2.500   -1.900    1238.027
        M1III        3775.722      549.543    54.928    2.500   -2.100    1389.089
        M2III        3706.807      660.696    62.487    2.500   -2.300    1580.265
        M3III        3630.781      794.332    71.415    2.500   -2.500    1806.051
        M4III        3556.313     1047.133    85.466    2.500   -2.800    2161.374
        M5III        3419.794     1258.931   101.343    2.500   -3.000    2562.890
        M6III        3250.873     1995.271   141.186    2.500   -3.500    3570.507
        M7III        3126.079     2630.280   175.304    2.500   -3.800    4433.329
        M8III        2890.680     3801.911   246.486    2.500   -4.200    6233.471
        M9III        2666.859     5495.433   348.170    2.500   -4.600    8805.012
        M10III       2500.345     7943.318   476.203    2.500   -5.000   12042.868

    **Giants**::

        SpType           teff        lumi       rad    mass     Mbol   rotperiod
        B2II        15995.580   19952.712    18.441   1.000   -6.000     932.739
        B5II        12589.254    7943.318    18.784   1.000   -5.000     950.079
        F0II         7943.282     794.332    14.921   1.000   -2.500     754.675
        F2II         7328.245     724.439    16.741   1.000   -2.400     846.759
        G5II         5248.075     870.967    35.792   1.000   -2.600    1810.339
        K01II        5011.872    1047.133    43.032   1.000   -2.800    2176.506
        K34II        4255.984    1380.390    68.516   1.000   -3.100    3465.452
        M3II         3411.929    3467.384   168.963   1.000   -4.100    8545.942

    **Supergiants**::

        SpType           teff         lumi        rad    mass     Mbol   rotperiod
        B0I         26001.596   288404.435     26.533   1.500   -8.900    1342.023
        B1I         20701.413   165959.430     31.754   1.500   -8.300    1606.053
        B3I         15595.525    87096.747     40.531   1.500   -7.600    2050.029
        B5I         13396.767    60256.227     45.687   1.500   -7.200    2310.787
        B8I         11194.379    45709.023     56.989   1.500   -6.900    2882.438
        A0I          9727.472    38019.109     68.832   1.500   -6.700    3481.448
        A2I          9078.205    38019.109     79.030   1.500   -6.700    3997.237
        F0I          7691.304    34673.840    105.146   1.500   -6.600    5318.142
        F5I          6637.431    34673.840    141.186   1.500   -6.600    7141.015
        F8I          6095.369    34673.840    167.414   1.500   -6.600    8467.592
        G0I          5508.077    34673.840    205.018   1.500   -6.600   10369.551
        G2I          5296.634    31622.917    211.735   1.500   -6.500   10709.272
        G5I          5046.613    31622.917    233.234   1.500   -6.500   11796.684
        G8I          4591.980    31622.917    281.703   1.500   -6.500   14248.198
        K2I          4255.984    34673.840    343.393   1.500   -6.600   17368.405
        K3I          4130.475    34673.840    364.579   1.500   -6.600   18439.958
        K4I          3990.249    38019.109    409.064   1.500   -6.700   20689.974
        M2I          3451.437   165959.430   1142.330   1.500   -8.300   57777.659
        
    @param spectype: spectral type
    @type spectype: str
    """
    #-- stellar parameters that are equal for all stars
    star = parameters.ParameterSet(frame='phoebe',context='star',add_constraints=True)
    star['incl'] = 90.,'deg'
    star['shape'] = 'equipot'
    star['ld_func'] = 'claret'
    star['atm'] = 'kurucz'
    star['ld_coeffs'] = 'kurucz'
    star['label'] = spectype+'_'+star['label']
    
    #-- stellar parameter dependent on spectral type: temperature, radius, mass
    #   and rotation period (luminosity follows from these)
    
    #-- first get information
    data = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),'catalogs','spectypes.dat'),dtype=np.str)
    sptypes = np.rec.fromarrays(data.T,dtype=np.dtype([('SpType', '|S9'),\
         ('teff', '<f8'), ('lumi', '<f8'), ('rad', '<f8'), ('mass', '<f8'),\
         ('Mbol', '<f8'), ('rotperiod', '<f8')]))
    sptype_names = [spt.strip() for spt in sptypes['SpType']]
    if not spectype in sptype_names:
        raise ValueError("SpType {} not one of {}".format(spectype,sptype_names))
    index = sptype_names.index(spectype)
    #-- then fill in the values
    star['teff'] = sptypes[index]['teff']
    star['radius'] = sptypes[index]['rad'],'Rsol'
    star['mass'] = sptypes[index]['mass'],'Msol'
    star['rotperiod'] = sptypes[index]['rotperiod'],'d'
    
    for key in kwargs:
        star[key] = kwargs[key]
    
    logger.info('Created star of spectral type {}'.format(spectype))
    
    if create_body:
        star = make_body_from_parametersets(star)
    
    return star    


def binary_from_spectral_types(spectype1, spectype2, sma=None, period=None,
                               kwargs1=None, kwargs2=None, orbitkwargs=None,
                               create_body=False, **kwargs):
    """
    Create a binary consisting of two stars with a given spectral type.
    
    See :py:func:`star_from_spectral_type` for details on the spectral type
    definitions.
    
    See :py:func:`binary_from_stars` for details on parameters and extra
    keyword arguments.
    """
    star1 = star_from_spectral_type(spectype1)
    star2 = star_from_spectral_type(spectype2)
    return binary_from_stars(star1, star2, sma=sma, period=period,\
                      kwargs1=kwargs1, kwargs2=kwargs2, orbitkwargs=orbitkwargs,\
                      create_body=create_body, **kwargs)
    
    

def binary_from_deb2011(name,create_body=False,**kwargs):
    """
    Create a binary from the catalog [deb2011]_
    """
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),'catalogs','deb2011.dat')
    data = ascii.read2recarray(filename,delimiter='\t')
    star_names = [iname.strip() for iname in data['Name']]
    star_types = [iname.strip() for iname in data['Type']]
    index = star_names.index(name)
    
    star1 = parameters.ParameterSet(context='star')
    star2 = parameters.ParameterSet(context='star')
    
    star1['mass'] = data['M1'][index],'Msol'
    star2['mass'] = data['M2'][index],'Msol'
    star1['radius'] = data['R1'][index],'Rsol'
    star2['radius'] = data['R2'][index],'Rsol'
    star1['teff'] = data['T1'][index],'K'
    star2['teff'] = data['T2'][index],'K'
    
    period = data['PME'][index]
    ecc = 0.
    K2 = data['K1'][index]
    K1 = data['K2'][index]
    sma = data['a'][index]
    mybinary = binary_from_spectroscopy(star1,star2,period,ecc,K1,K2=K2)
    mybinary[2]['sma'] = sma
    mybinary[2]['c1label'] = star1['label']
    mybinary[2]['c2label'] = star2['label']
    if star_types[index] == ['ED']:
        validate_orbit(mybinary[2])
    else:
        mybinary = [mybinary[0],mybinary[2]]
    return mybinary


def dep_from_object(myobject,context,**kwargs):
    """
    Initialise an observable parameterSet with defaults from an object.
    
    This is useful to initialize a consistent set of observables, i.e.
    where all the atmosphere models and limb darkening coefficients are
    compliant with the bolometric ones.
    
    Extra kwargs override initialisation.
    """
    obsdep = parameters.ParameterSet(context=context,**kwargs)
    for key in ['atm','ld_func','ld_coeffs','alb']:
        if not key in kwargs:
            obsdep[key] = myobject[key]
    return obsdep


def binary_from_stars(star1, star2, sma=None, period=None,\
                      kwargs1=None, kwargs2=None, orbitkwargs=None,\
                      create_body=False):
    """
    Create a binary system from two separate stars.
    
    Extra information given is the separation (C{sma}) between the two objects
    or the orbital period. You should give one of them (and only one) as a
    tuple (value,unit). The other one will be derived using the masses of the
    individual stars and
    :py:func:`Kepler's third law <phoebe.dynamics.keplerorbit.third_law>`.
    
    Usually, you'd want to give period rather than semi-major axis, since
    the period is usually much better constrained.
    
    The inclination angle and synchronicity parameters will by default be
    derived from the inclination angles from the stars, and their rotation period
    with respect to the orbital period.
    
    Extra ``kwargs1`` override defaults in the creation of the primary
    :ref:`component <parlabel-phoebe-component>`, extra
    ``kwargs2`` for the secondary :ref:`component <parlabel-phoebe-component>`,
    and ``orbitkwargs`` does the same for the creation of the
    :ref:`orbit <parlabel-phoebe-orbit>`.
    
    @param star1: ParameterSet of context 'star' representing star 1
    @type star1: ParameterSet
    @param star2: ParameterSet of context 'star' representing star 2
    @type star2: ParameterSet
    @param sma: sma between the two components, and its unit
    @type sma: (float,string)
    @param period: period of the system
    @type period: (float, string)
    @param create_body: create a BodyBag containing both objects
    @type create_body: bool
    @return: parameterSets component 1, component 2 and orbit, or BodyBag instance
    @rtype: (ParameterSet, ParameterSet, ParameterSet) or BodyBag
    """
    if kwargs1 is None:
        kwargs1 = {}
    if kwargs2 is None:
        kwargs2 = {}
    if orbitkwargs is None:
        orbitkwargs = {}
    if sma is not None and period is not None:
        raise ValueError("Give only sma or period, not both")
    
    if star1['incl'] == star2['incl'] and not 'incl' in orbitkwargs:
        orbitkwargs.setdefault('incl', star1['incl'])
    elif not 'incl' in orbitkwargs:
        logger.warning(("Cannot derive orbital inclination from stars and "
                        "'incl' is not given in orbitkwargs, "
                        "taking default value."))
    if star1['long'] == star2['long'] and not 'long_an' in orbitkwargs:
        orbitkwargs.setdefault('long_an', star1['long'])
    elif not 'long_an' in orbitkwargs:
        logger.warning(("Cannot derive orbital longitude orientation from "
                        "stars and 'long_an' is not given in orbitkwargs, "
                        "taking default value."))
        
    comp1 = parameters.ParameterSet(context='component',**kwargs1)
    comp2 = parameters.ParameterSet(context='component',**kwargs2)
    orbit = parameters.ParameterSet(context='orbit',**orbitkwargs)
    
    # Separation at perioastron passage
    d = 1.0 - orbit['ecc']
    
    #-- get value on total mass
    M = star1.get_value('mass','Msol')+star2.get_value('mass','Msol')
    #-- get information on semi-major axis
    if sma is not None:
        try:
            sma = conversions.convert(sma[1],'au',sma[0])
        except TypeError:
            raise ValueError("'sma' needs to be a tuple (value,'unit')")
        
        period = keplerorbit.third_law(totalmass=M,sma=sma)
        orbit['sma'] = sma,'au'
        orbit['period'] = period,'d'
    elif period is not None:
        try:
            period = conversions.convert(period[1],'d',period[0])
        except TypeError:
            raise ValueError("'period' needs to be a tuple (value,'unit')")
        
        sma = keplerorbit.third_law(totalmass=M,period=period)
        orbit['sma'] = sma,'au'
        orbit['period'] = period,'d'
    
    orbit['q'] = star2.get_value('mass','Msol')/star1.get_value('mass','Msol')
    
    logger.info("Creating binary: sma={:.3g}Rsol, period={:.3g}d, ecc={:.3g}, q={:.3g}".format(orbit['sma'],orbit['period'],orbit['ecc'],orbit['q']))
    
    if not 'syncpar' in kwargs1:
        comp1['syncpar'] = orbit.request_value('period','d')/star1.request_value('rotperiod','d')
    if not 'syncpar' in kwargs2:
        comp2['syncpar'] = orbit.request_value('period','d')/star2.request_value('rotperiod','d')
    
    logger.info('Creating binary: syncpar1={:.3g}, syncpar2={:.3g}'.format(comp1['syncpar'],comp2['syncpar']))
    
    R1 = star1.get_value('radius','au')
    R2 = star2.get_value('radius','au')
    
    R1crit = roche.calculate_critical_radius(orbit['q'],component=1,d=d,F=comp1['syncpar'])*sma
    R2crit = roche.calculate_critical_radius(orbit['q'],component=2,d=d,F=comp2['syncpar'])*sma
    #if R1>R1crit: raise ValueError("Star1 radius exceeds critical radius: R/Rcrit={}".format(R1/R1crit))
    #if R2>R2crit: raise ValueError("Star2 radius exceeds critical radius: R/Rcrit={}".format(R2/R2crit))
    logger.info('Primary   radius estimate (periastron): {:.6f} Rcrit'.format(R1/R1crit))
    logger.info('Secondary radius estimate (periastron): {:.6f} Rcrit'.format(R2/R2crit))
    if orbit['ecc']>0:
        R1crit = roche.calculate_critical_radius(orbit['q'],component=1,d=1+orbit['ecc'],F=comp1['syncpar'])*sma
        R2crit = roche.calculate_critical_radius(orbit['q'],component=2,d=1+orbit['ecc'],F=comp2['syncpar'])*sma
        logger.info('Primary   radius estimate (apastron): {:.6f} Rcrit'.format(R1/R1crit))
        logger.info('Secondary radius estimate (apastron): {:.6f} Rcrit'.format(R2/R2crit))
    
    comp1['pot'] = roche.radius2potential(R1/sma,orbit['q'],d=d,component=1,F=comp1['syncpar'])
    comp2['pot'] = roche.radius2potential(R2/sma,orbit['q'],d=d,component=2,F=comp2['syncpar'])
    
    for key in ['teff']:
        if not key in kwargs1:
            comp1[key] = star1.get_value_with_unit(key)
        if not key in kwargs2:
            comp2[key] = star2.get_value_with_unit(key)
    for key in ['atm','label','ld_func','ld_coeffs','gravb','irradiator','alb','redist','abun','gravblaw']:
        if not key in kwargs1:
            comp1[key] = star1[key]
        if not key in kwargs2:
            comp2[key] = star2[key]
    orbit['c1label'] = comp1['label']
    orbit['c2label'] = comp2['label']
    logger.info('Creating binary: pot1={:.3g}, pot2={:.3g}'.format(comp1['pot'],comp2['pot']))
    
    
    if create_body:
        system = make_body_from_parametersets((comp1, comp2, orbit))
        return system
    else:   
        return comp1, comp2, orbit    

def stars_from_binary(comp1, comp2, orbit):
    """
    Create two stars starting from a binary system.
    """
    sma = orbit.request_value('sma','au')
    period = orbit.request_value('period','d')
    q = orbit.request_value('q')
    pot1 = comp1['pot']
    pot2 = comp2['pot']
    ecc = orbit['ecc']
    f1 = comp1['syncpar']
    f2 = comp2['syncpar']
    
    # Derive masses
    totalmass = keplerorbit.third_law(sma=sma,period=period)
    M1 = totalmass / (1.0 + q)
    M2 = q * M1
    
    # Derive radii at periastron passage
    R1 = roche.potential2radius(pot1,q,1-ecc,f1,component=1,sma=sma)
    R2 = roche.potential2radius(pot2,q,1-ecc,f2,component=2,sma=sma)
    
    # Derive rotation period
    rotperiod1 = period/f1
    rotperiod2 = period/f2
    
    # Build the stars with the basic parameters
    star1 = parameters.ParameterSet(context='star', mass=(M1,'Msol'),
                                    radius=(R1,'au'), rotperiod=(rotperiod1,'d'))
    star2 = parameters.ParameterSet(context='star', mass=(M2,'Msol'),
                                    radius=(R2,'au'), rotperiod=(rotperiod2,'d'))
    # And copy some values straight from the components
    for key in ['teff']:
        star1[key] = comp1.get_value_with_unit(key)
        star2[key] = comp2.get_value_with_unit(key)
    for key in ['atm','label','ld_func','ld_coeffs','gravb','irradiator','alb']:
        star1[key] = comp1[key]
        star2[key] = comp2[key]
    star1['label'] = comp1['label']
    star2['label'] = comp2['label']
    
    return star1, star2
    



def binary_from_spectroscopy(star1,star2,period,ecc,K1,K2=None,\
             vsini1=None,vsini2=None,\
             logg1=None,logg2=None,\
             syncpar1=1.,syncpar2=1.,\
             eclipse_duration=None):
    """
    Initiate a consistent set of parameters for a binary system from spectroscopic parameters.
    
    Set one of the star's radius to 0 if you give the eclipse duration,
    the second radius will be calculated then.
    
    Set one of the star's masses to 0 if you give K2.
    
    The parameters that you want to derive, need to be adjustable!
    
    @param star1: parameterSet representing primary
    @type star1: parameterSet
    @param star2: parameterSet representing secondary
    @type star2: parameterSet
    @param period: period in days
    @type period: float
    @param ecc: eccentricity
    @type ecc: float
    @param K1: semi-amplitude in km/s
    @type K1: float
    @param K2: semi-amplitude in km/s (optional)
    @type K2: float
    """
    logger.info("Creating binary from spectroscopy")
    #-- take care of logg parameters
    for i,(star,logg) in enumerate(zip([star1,star2],[logg1,logg2])):
        if logg is not None:
            tools.add_surfgrav(star,logg,derive='mass')   
    
    #-- period, ecc and q
    comp1 = parameters.ParameterSet(context='component', label=star1['label'])
    comp2 = parameters.ParameterSet(context='component', label=star2['label'])
    orbit = parameters.ParameterSet(context='orbit',c1label=comp1['label'],c2label=comp2['label'],add_constraints=True)
    orbit['period'] = period,'d'
    orbit['ecc'] = ecc
    #-- take over some values
    for key in ['teff','ld_func','ld_coeffs','label','atm','irradiator','gravb']:
        comp1[key] = star1[key]
        comp2[key] = star2[key]
    
    period_sec = orbit.request_value('period','s')
    K1_ms = conversions.convert('km/s','m/s',K1)
    
    #-- if we have K2, we can derive the mass ratio and the system semi-major
    #   axis. Otherwise, we use the masses from the stars
    if K2 is not None:
        q = abs(K1/K2)
        K2_ms = conversions.convert('km/s','m/s',K2)
        star2['mass'] = q*star1['mass']
        logger.info("Calculated q from K1/K2: q={}".format(q))
        logger.info('Calculated secondary mass from K2 and q: M2={} Msol'.format(star2.request_value('mass','Msol')))
    else:
        q = star2['mass']/star1['mass']
        K2_ms = K1_ms/q
        logger.info('Calculated q from stellar masses: q = {}'.format(q))
        logger.info("Calculated K2 from q and K1: K2={} km/s".format(K2_ms/1000.))
    
    # Calculate system semi major axis from period and masses
    sma = keplerorbit.third_law(period=orbit.request_value('period', 'd'),
                                totalmass=star1.request_value('mass','Msol')+\
                                          star2.request_value('mass','Msol'))
    sma = conversions.convert('au', 'Rsol', sma)
    orbit['sma'] = sma, 'Rsol'
    
    logger.info("Calculated system sma from period and total mass: sma={} Rsol".format(sma))
    
    orbit['q'] = q
    asini = conversions.convert('m','Rsol',keplerorbit.calculate_asini(period_sec,ecc,K1=K1_ms,K2=K2_ms))
    
    # Possibly, the binary system cannot exist. Suggest a possible solution.
    if asini > sma:
        #-- calculate the mass function
        fm = keplerorbit.calculate_mass_function(period_sec,ecc,K1_ms)
        new_mass1 = conversions.convert('kg','Msol',fm*(1+orbit['q'])**2/orbit['q']**3)
        new_mass2 = q*new_mass1
        raise ValueError(("Binary system cannot exist: asini={} > sma={} Rsol.\n"
                         "Possible solution: decrease total mass or period.\n"
                         "If you want to keep the mass ratio, you might set "
                         "M1={} Msol, M2={} Msol").format(asini, sma, new_mass1+1e-10, new_mass2+1e-10))
    
    logger.info('Calculated system asini from period, ecc and semi-amplitudes: asini={} Rsol'.format(asini))
    tools.add_asini(orbit,asini,derive='incl',unit='Rsol')
    logger.info("Calculated incl: {} deg".format(orbit.request_value('incl','deg')))
        
    #-- when the vsini is fixed, we can only adjust the radius or the synchronicity!
    for i,(comp,star,vsini) in enumerate(zip([comp1,comp2],[star1,star2],[vsini1,vsini2])):
        if vsini is None:
            continue
        radius = star.get_value_with_unit('radius')
        #-- adjust the radius if possible
        if star.get_adjust('radius'):
            comp.add(parameters.Parameter(qualifier='radius',value=radius[0],unit=radius[1],description='Approximate stellar radius',adjust=star.get_adjust('radius')))
            comp.add(parameters.Parameter(qualifier='vsini',value=vsini,unit='km/s',description='Approximate projected equatorial velocity',adjust=(vsini is None)))
            orbit.add_constraint('{{c{:d}pars.radius}} = {{period}}*{{c{:d}pars[vsini]}}/(2*np.pi*np.sin({{incl}}))'.format(i+1,i+1))
            orbit.run_constraints()
            logger.info('Star {:d}: radius will be derived from vsini={:.3g} {:s} (+ period and incl)'.format(i+1,*comp.get_value_with_unit('vsini')))
        #-- or else adjust the synchronicity
        else:
            radius_km = conversions.convert(radius[1],'km',radius[0])
            syncpar = period_sec*vsini/(2*np.pi*radius_km*np.sin(incl[0]))
            comp['syncpar'] = syncpar
            logger.info('Star {:d}: Synchronicity parameters is set to synpar={:.4f} match vsini={:.3g} km/s'.format(i+1,syncpar,vsini))        
    
    #if 'radius' in orbit['c1pars']:
    #    logger.info('Radius 1 = {} {}'.format(*orbit['c1pars'].get_value_with_unit('radius')))
    #    star1['radius'] = orbit['c1pars'].get_value_with_unit('radius')
    #if 'radius' in orbit['c2pars']:
    #    logger.info('Radius 2 = {} {}'.format(*orbit['c2pars'].get_value_with_unit('radius')))
    #    star1['radius'] = orbit['c2pars'].get_value_with_unit('radius')
    
    R1 = star1.get_value('radius','au')
    R2 = star2.get_value('radius','au')
    sma = orbit.get_value('sma','au'),'au'
    #-- we can also derive something on the radii from the eclipse duration, but
    #   only in the case for cicular orbits
    if eclipse_duration is not None and ecc==0.:
        total_radius = keplerorbit.calculate_total_radius_from_eclipse_duration(eclipse_duration,orbit.get_value('incl','rad'))
        total_radius*= sma[0]
        logger.info('Eclipse duration: {}'.format(eclipse_duration))
        logger.info('Total radius: {} Rsol'.format(total_radius*constants.au/constants.Rsol))
        #-- we can derive the total radius. Thus, if one of the radii is fixed,
        #   we can derive the other one. If both are unknown, the eclipse
        #   duration doesn't give any immediate constraints, but the user should
        #   have given radii which are compatible.
        if star1.get_adjust('radius'):
            R1 = total_radius-R2
            star1['radius'] = R1,'au'
            logger.info('Radius 1 = {} {}'.format(*star1.get_value_with_unit('radius')))
        elif star2.get_adjust('radius'):
            R2 = total_radius-R1
            star2['radius'] = R2,'au'
            logger.info('Radius 2 = {} {}'.format(*star2.get_value_with_unit('radius')))
        else:
            if total_radius!=(R1+R2):
                raise ValueError("Sum of radii not compatible with eclipse duration and inclination angle")
    
    #-- check if eclipsing, but only if the system is circular
    if ecc==0:
        incl = orbit.request_value('incl','rad')
        logger.info("The circular system is{}eclipsing".format((np.sin(np.pi/2-incl)<=((R1+R2)/sma[0])) and ' ' or ' not '))
    
    #-- fill in Roche potentials
    R1crit = roche.calculate_critical_radius(orbit['q'],component=1)*sma[0]
    R2crit = roche.calculate_critical_radius(orbit['q'],component=2)*sma[0]
    R1crit = conversions.convert(sma[1],star1.get_unit('radius'),R1crit)
    R2crit = conversions.convert(sma[1],star1.get_unit('radius'),R2crit)
    logger.info('Primary critical radius = {} Rsol'.format(R1crit))
    logger.info('Secondary critical radius = {} Rsol'.format(R2crit))
    if R1crit<R1:
        raise ValueError("Primary exceeds its Roche lobe")
    if R2crit<R2:
        raise ValueError("Secondary exceeds its Roche lobe")
    try:
        comp1['pot'] = roche.radius2potential(R1/sma[0],orbit['q'],component=1)
        comp2['pot'] = roche.radius2potential(R2/sma[0],orbit['q'],component=2)
    except:
        raise ValueError("One of the star has an impossible Roche geometry")

    return comp1,comp2,orbit


#}  

#{ More complicated systems

def T_CrB(create_body=True,**kwargs):
    """
    T Coronae Borealis - The Blaze star
    """
    orbit,red_giant,white_dwarf,disk = from_library('T_CrB_system',create_body=False,**kwargs)
    disk['label'] = white_dwarf['label']
    if create_body:
        mesh = parameters.ParameterSet(context='mesh:marching')
        body1 = universe.BinaryRocheStar(red_giant,orbit=orbit,mesh=mesh)
        body2 = universe.BinaryRocheStar(white_dwarf,orbit=orbit,mesh=mesh.copy())
        body3 = universe.AccretionDisk(disk)
        body3 = universe.BodyBag([body2, body3], label='white_dwarf_system', orbit=orbit)
        #-- if there are irradiators, we need to prepare for reflection
        #   results
        if red_giant['irradiator'] or white_dwarf['irradiator']:
            body1.prepare_reflection()
            body2.prepare_reflection()
            body3.prepare_reflection()
        #-- put the two Bodies in a BodyBag and return it to the user.
        output = universe.BodyBag([body1,body3])
    else:
        output = red_giant,white_dwarf,disk,orbit
    
    return output
    

def KOI126(create_body=True,**kwargs):
    """
    Kepler Object of Interest 126 - a hierarchical version of the triple system
    """
    T0 = 2455170.5
    starA = parameters.ParameterSet(frame='phoebe',context='star',add_constraints=True)
    starA['teff'] = 5875,'K'
    starA['mass'] = 1.347,'Msol'
    starA['shape'] = 'sphere'
    starA['radius'] = 2.0254,'Rsol'
    starA['rotperiod'] = np.inf
    starA['ld_func'] = 'claret'
    starA['atm'] = 'kurucz'
    starA['ld_coeffs'] = 'kurucz'
    starA['label'] = 'starA'
    
    starB = parameters.ParameterSet(frame='phoebe',context='star',add_constraints=True)
    starB['teff'] = 3000,'K'
    starB['mass'] = 0.2413,'Msol'
    starB['shape'] = 'sphere'
    starB['radius'] = 0.2543,'Rsol'
    starB['rotperiod'] = np.inf
    starB['ld_func'] = 'linear'
    starB['atm'] = 'blackbody'
    starB['ld_coeffs'] = [0.5]
    starB['label'] = 'starB'
    
    starC = parameters.ParameterSet(frame='phoebe',context='star',add_constraints=True)
    starC['teff'] = 2800,'K'
    starC['mass'] = 0.2127,'Msol'
    starC['shape'] = 'sphere'
    starC['radius'] = 0.2318,'Rsol'
    starC['rotperiod'] = np.inf
    starB['ld_func'] = 'linear'
    starB['atm'] = 'blackbody'
    starB['ld_coeffs'] = [0.5]
    starC['label'] = 'starC'
    
    starA['incl'] = 90.
    starB['incl'] = 90.
    starC['incl'] = 90.
    
    #-- inner binary
    orbitBC = parameters.ParameterSet(frame='phoebe',context='orbit',add_constraints=True,\
              c1label=starB['label'],c2label=starC['label'])
    orbitBC['period'] = 1.76713,'d'
    orbitBC['sma'] = 0.021986,'au'
    orbitBC['ecc'] = 0.02334
    orbitBC['per0'] = 89.52,'deg'
    orbitBC['incl'] = 94.807,'deg'#96.907-92.100,'deg'
    orbitBC['long_an'] = 8.012,'deg'
    orbitBC['label'] = 'systemBC'
    
    #-- outer binary
    orbitA_BC = parameters.ParameterSet(frame='phoebe',context='orbit',add_constraints=True,\
              c1label=starA['label'],c2label=orbitBC['label'])
    orbitA_BC['period'] = 33.9214,'d'
    orbitA_BC['sma'] = 0.2495,'au'
    orbitA_BC['ecc'] = 0.3043
    orbitA_BC['per0'] = 52.88,'deg'
    orbitA_BC['incl'] = 92.100,'deg'
    orbitA_BC['label'] = 'systemABC'
    
    
    mean_anomaly = 19.87/180.*np.pi
    orbitA_BC['t0'] = T0-keplerorbit.mean_anomaly_to_time(mean_anomaly,starA.request_value('mass','kg')+\
                          starB.request_value('mass','kg')+starC.request_value('mass','kg'),\
                          orbitA_BC.request_value('sma','m'))/(3600*24.)
    mean_anomaly = 355.66/180.*np.pi
    orbitBC['t0'] = T0-keplerorbit.mean_anomaly_to_time(mean_anomaly,starB.request_value('mass','kg')+\
                          starC.request_value('mass','kg'),\
                          orbitBC.request_value('sma','m'))/(3600*24.)    
    
    meshA = parameters.ParameterSet(frame='phoebe',context='mesh:marching', alg='c')
    meshB = parameters.ParameterSet(frame='phoebe',context='mesh:marching', alg='c')
    meshC = parameters.ParameterSet(frame='phoebe',context='mesh:marching', alg='c')
    
    #-- light curve
    lcdep1 = parameters.ParameterSet(frame='phoebe',context='lcdep')
    lcdep1['ld_func'] = 'claret'
    lcdep1['ld_coeffs'] = 'kurucz'
    lcdep1['atm'] = 'kurucz'
    lcdep1['passband'] = 'JOHNSON.V'
    lcdep1['ref'] = 'light curve'
    lcdep2 = lcdep1.copy()
    lcdep2['ld_func'] = 'linear'
    lcdep2['ld_coeffs'] = [0.5]
    lcdep2['atm'] = 'blackbody'
    lcdep3 = lcdep2.copy()
    
    
    starB = universe.BinaryStar(starB,orbitBC,meshB,pbdep=[lcdep2])
    starC = universe.BinaryStar(starC,orbitBC,meshC,pbdep=[lcdep3])
    starA = universe.BinaryStar(starA,orbitA_BC,meshA,pbdep=[lcdep1])
    systemBC = universe.BodyBag([starB,starC],orbit=orbitA_BC,label='systemBC')
    
    system = universe.BodyBag([starA,systemBC], label='KOI126')
    return system
    
def KOI126_alternate(create_body=True,**kwargs):
    """
    Kepler Object of Interest 126 - a hierarchical version of the triple system
    """
    T0 = 2455170.5
    starA = parameters.ParameterSet(frame='phoebe',context='star',add_constraints=True)
    starA['teff'] = 5875,'K'
    starA['mass'] = 1.347,'Msol'
    starA['shape'] = 'sphere'
    starA['radius'] = 2.0254,'Rsol'
    starA['rotperiod'] = np.inf
    starA['ld_func'] = 'claret'
    starA['atm'] = 'kurucz'
    starA['ld_coeffs'] = 'kurucz'
    
    starB = parameters.ParameterSet(frame='phoebe',context='star',add_constraints=True)
    starB['teff'] = 3000,'K'
    starB['mass'] = 0.2413,'Msol'
    starB['shape'] = 'sphere'
    starB['radius'] = 0.2543,'Rsol'
    starB['rotperiod'] = np.inf
    starB['ld_func'] = 'linear'
    starB['atm'] = 'blackbody'
    starB['ld_coeffs'] = [0.5]
    
    starC = parameters.ParameterSet(frame='phoebe',context='star',add_constraints=True)
    starC['teff'] = 2800,'K'
    starC['mass'] = 0.2127,'Msol'
    starC['shape'] = 'sphere'
    starC['radius'] = 0.2318,'Rsol'
    starC['rotperiod'] = np.inf
    starB['ld_func'] = 'linear'
    starB['atm'] = 'blackbody'
    starB['ld_coeffs'] = [0.5]
    
    #-- inner binary
    orbitBC = parameters.ParameterSet(frame='phoebe',context='orbit',add_constraints=True)
    orbitBC['period'] = 1.76713,'d'
    orbitBC['sma'] = 0.021986,'au'
    orbitBC['ecc'] = 0.02334
    orbitBC['per0'] = 89.52,'deg'
    orbitBC['incl'] = 96.907-92.100,'deg'
    orbitBC['long_an'] = 8.012,'deg'
    
    #-- outer binary
    orbitA_BC = parameters.ParameterSet(frame='phoebe',context='orbit',add_constraints=True)
    orbitA_BC['period'] = 33.9214,'d'
    orbitA_BC['sma'] = 0.2495,'au'
    orbitA_BC['ecc'] = 0.3043
    orbitA_BC['per0'] = 52.88,'deg'
    orbitA_BC['incl'] = 92.100,'deg'
    orbitA_BC['label'] = 'KOI126'
    
    
    mean_anomaly = 19.87/180.*np.pi
    orbitA_BC['t0'] = T0-keplerorbit.mean_anomaly_to_time(mean_anomaly,starA.request_value('mass','kg')+\
                          starB.request_value('mass','kg')+starC.request_value('mass','kg'),\
                          orbitA_BC.request_value('sma','m'))/(3600*24.)
    mean_anomaly = 355.66/180.*np.pi
    orbitBC['t0'] = T0-keplerorbit.mean_anomaly_to_time(mean_anomaly,starB.request_value('mass','kg')+\
                          starC.request_value('mass','kg'),\
                          orbitBC.request_value('sma','m'))/(3600*24.)    
    
    meshA = parameters.ParameterSet(frame='phoebe',context='mesh:marching')
    meshB = parameters.ParameterSet(frame='phoebe',context='mesh:marching')
    meshC = parameters.ParameterSet(frame='phoebe',context='mesh:marching')
    
    #-- light curve
    lcdep1 = parameters.ParameterSet(frame='phoebe',context='lcdep')
    lcdep1['ld_func'] = 'claret'
    lcdep1['ld_coeffs'] = 'kurucz'
    lcdep1['atm'] = 'kurucz'
    lcdep1['passband'] = 'JOHNSON.V'
    lcdep1['ref'] = 'light curve'
    lcdep2 = lcdep1.copy()
    lcdep2['ld_func'] = 'linear'
    lcdep2['ld_coeffs'] = [0.5]
    lcdep2['atm'] = 'blackbody'
    lcdep3 = lcdep2.copy()
    
    
    starA = universe.BinaryStar(starA,None,meshA,pbdep=[lcdep1])
    starB = universe.BinaryStar(starB,None,meshB,pbdep=[lcdep2])
    starC = universe.BinaryStar(starC,None,meshC,pbdep=[lcdep3])
    
    systemBC = universe.BinaryBag([starB,starC],orbit=orbitBC)
    systemA_BC = universe.BinaryBag([starA,systemBC],orbit=orbitA_BC, label='KOI126')
    
    return systemA_BC


def vega_aufdenberg2006(create_body=True):
    """
    Vega model from Aufdenberg (2006).
    """
    #-- basic parameterSet
    star = from_library('vega2006')
    tools.add_teffpolar(star,10150.)
    tools.add_rotfreqcrit(star,0.91)
    tools.add_parallax(star,128.93,unit='mas')
    tools.add_surfgrav(star,derive='mass')
    #-- add priors
    star.get_parameter('rotfreqcrit').set_prior(distribution='normal', sigma=0.03)
    star.get_parameter('parallax').set_prior(distribution='normal', sigma=0.55)
    star.get_parameter('radius').set_prior(distribution='normal', sigma=0.07)
    star.get_parameter('teffpolar').set_prior(distribution='normal', sigma=100.)
    star.get_parameter('mass').set_prior(distribution='normal', sigma=0.2)
    star.get_parameter('incl').set_prior(distribution='normal', sigma=0.3)
    star.get_parameter('rotperiod').set_prior(distribution='normal', sigma=0.03)
    star.get_parameter('surfgrav').set_prior(distribution='normal', sigma=0.1)
    
    star = universe.Star(star)
    
    return star


def vega_hill2010():
    """
    Vega model from Hill (2010).
    """
    #-- basic parameterSet
    star = from_library('vega2010')
    tools.add_teffpolar(star,10000)
    tools.add_rotfreqcrit(star,0.81)
    tools.add_parallax(star,130.23,unit='mas')
    tools.add_surfgrav(star,derive='mass')
    #-- add priors
    star.get_parameter('rotfreqcrit').set_prior(distribution='normal', sigma=0.02)
    star.get_parameter('parallax').set_prior(distribution='normal', sigma=0.36)
    star.get_parameter('radius').set_prior(distribution='normal', sigma=0.01)
    star.get_parameter('teffpolar').set_prior(distribution='normal', sigma=30.)
    star.get_parameter('mass').set_prior(distribution='normal', sigma=0.2)
    star.get_parameter('incl').set_prior(distribution='normal', sigma=0.1)
    star.get_parameter('rotperiod').set_prior(distribution='normal', sigma=0.01)
    star.get_parameter('surfgrav').set_prior(distribution='normal', sigma=0.25)
    
    return star


def vega_yoon2010():
    """
    Vega model from Yoon (2010).
    """
    #-- basic parameterSet
    star = from_library('vega2010')
    tools.add_teffpolar(star,10059.)
    tools.add_rotfreqcrit(star,0.8760)
    tools.add_parallax(star,130.23,unit='mas')
    tools.add_surfgrav(star,derive='mass')
    #-- add priors
    star.get_parameter('rotfreqcrit').set_prior(distribution='normal',sigma=0.0057)
    star.get_parameter('parallax').set_prior(distribution='normal',sigma=0.36)
    star.get_parameter('radius').set_prior(distribution='normal',sigma=0.012)
    star.get_parameter('teffpolar').set_prior(distribution='normal',sigma=13.)
    star.get_parameter('mass').set_prior(distribution='normal',sigma=0.074)
    star.get_parameter('incl').set_prior(distribution='normal',sigma=0.081)
    star.get_parameter('rotperiod').set_prior(distribution='normal',sigma=0.0084)
    
    return star


def vega_monnier2012():
    """
    Vega model from Monnier (2012).
    """
    #-- basic parameterSet
    star = from_library('vega2012')
    tools.add_teffpolar(star,10070.)
    tools.add_rotfreqcrit(star,0.774)
    tools.add_parallax(star,130.23,unit='mas')
    tools.add_surfgrav(star,derive='mass')
    #-- add priors
    star.get_parameter('rotfreqcrit').set_prior(distribution='normal',sigma=0.012)
    star.get_parameter('parallax').set_prior(distribution='normal',sigma=0.36)
    star.get_parameter('radius').set_prior(distribution='normal',sigma=0.012)
    star.get_parameter('teffpolar').set_prior(distribution='normal',sigma=90.)
    star.get_parameter('mass').set_prior(distribution='normal',sigma=0.13)
    star.get_parameter('incl').set_prior(distribution='normal',sigma=0.4)
    star.get_parameter('rotperiod').set_prior(distribution='normal',sigma=0.02)
    
    return star


def binary(create_body=True):
    """
    Default Phoebe2 binary system
    """
    component1 = parameters.ParameterSet('component', label='primary')
    component2 = parameters.ParameterSet('component', label='secondary')
    orbit = parameters.ParameterSet('orbit', c1label='primary', c2label='secondary', label='new_system')
    comp1 = universe.BinaryRocheStar(component=component1, orbit=orbit)
    comp2 = universe.BinaryRocheStar(component=component2, orbit=orbit)
    position = parameters.ParameterSet('position', distance=(10.,'pc'))
    reddening = parameters.ParameterSet('reddening:interstellar')
    
    return universe.BodyBag([comp1, comp2], label='new_system', reddening=reddening,
                            position=position)
    

def overcontact(create_body=True):
    """
    Overcontact system
    """
    component1 = parameters.ParameterSet('component', label='overcontact')
    component1['pot'] = 2.75
    component1['morphology'] = 'overcontact'
    orbit = parameters.ParameterSet('orbit', c1label='overcontact', c2label='<unknown>', label='new_system')
    orbit['q'] = 0.5
    position = parameters.ParameterSet('position', distance=(10.,'pc'))
    reddening = parameters.ParameterSet('reddening:interstellar')
    
    comp1 = universe.BinaryRocheStar(component=component1, orbit=orbit)
    
    return universe.BodyBag([comp1], label='new_system', reddening=reddening,
                            position=position)


def GD2938(create_body=True):
    """
    Pulsating white dwarf with a disk
    
    Be sure to run the computations with ``irradiation_alg='full'``.
    """
    starpars = parameters.ParameterSet('star', irradiator=True,
                                       incl=(70.,'deg'),teff=20000,
                                       rotperiod=np.inf,
                                       label='white_dwarf')
    diskpars = parameters.ParameterSet('accretion_disk', Rout=(11.,'Rsol'),
                               height=(2e-2, 'Rsol'), dmdt=(1e-6,'Msol/yr'),
                               incl=(70.,'deg'), Rin=(2.,'Rsol'),
                               label='accretion_disk')

    puls = parameters.ParameterSet('puls', freq=0.5, l=1, m=1,
                                   amplteff=2.0, label='f01')
    mesh = parameters.ParameterSet(context='mesh:marching', delta=0.25)
    meshdisk = parameters.ParameterSet(context='mesh:disk')
    star = universe.Star(starpars, mesh=mesh, puls=[puls])
    disk = universe.AccretionDisk(diskpars, mesh=meshdisk)

    system = universe.BodyBag([star, disk], label='GD2938')
    
    return system


def close_beta_cephei(create_body=True):
    """
    Binary system with pulsating oblique magnetic star and fast rotating star with a disk
    """
    long = 211.6

    # Define the parameters of the star. We're inspired by [Donati1997]_,
    # [Morel2006]_ and [Nieva2002]_.

    star = parameters.ParameterSet('star',label='beta Cephei')
    star['atm'] = 'blackbody'
    star['ld_func'] = 'linear'
    star['ld_coeffs'] = [0.33]
    star['abun'] = -0.2
    star['teff'] = 27000.,'K'
    star['rotperiod'] =  12.001663246,'d'
    star['incl'] = 100.,'deg'
    star['long'] = long, 'deg'
    star['mass'] = 14.,'Msol'
    star['radius'] = 6.4,'Rsol'
    #star['vgamma'] = 7.632287,'km/s'
    star['label'] = 'betacep'
    
    # Just for fun, also add the parallax, the surface gravity and vsini (and define
    # a mesh).
    tools.add_parallax(star,parallax=4.76,unit='mas')
    tools.add_surfgrav(star,3.70,derive='radius')
    
    mesh_betacep = parameters.ParameterSet('mesh:marching',delta=0.05, alg='c')

    # For the parameters of the pulsation mode, we're inspired by [Telting1997]_:
    freq1 = parameters.ParameterSet('puls', freq= 5.24965427519, phase=0.122545,
                                    ampl=0.10525/50, l=3, m=2, amplteff=0.2,
                                    label='f01')

    # For the parameters of the magnetic dipole, we take guesses from our own
    # research:
    mag_field1 = parameters.ParameterSet('magnetic_field:dipole')
    mag_field1['Bpolar'] = 276.01,'G'
    mag_field1['beta'] = 61.29,'deg'
    mag_field1['phi0'] = 80.522,'deg'
    
    #-- secondary information from Wheelwright 2009
    be_star = parameters.ParameterSet('star',gravblaw='zeipel',label='Be-star')
    be_star['mass'] = 4.4,"Msol"
    be_star.get_parameter('mass').set_prior(distribution='normal',sigma=(0.7,'Msol'))
    be_star['teff'] = 17000.
    be_star['incl'] = 100.,'deg'
    be_star['long'] = long, 'deg'
    be_star['rotperiod'] = 0.7,'d'
    be_star.get_parameter('teff').set_prior(distribution='normal',sigma=1000.)
    tools.add_surfgrav(be_star,4.1,derive='radius',unit='[cm/s2]')
    
    spot = parameters.ParameterSet('circ_spot', long=0.0, colat=140., angrad=15., teffratio=0.85)

    diskpars = parameters.ParameterSet(context='accretion_disk')
    diskpars['Rin'] = 6.
    diskpars['Rout'] = 10.,'Rsol'
    diskpars['height'] = 1e-2,'Rsol'
    diskpars['dmdt'] = 1e-6,'Msol/yr'
    diskpars['label'] = 'accretion_disk'
    diskpars['incl'] = 100.
    diskpars['long'] = long
    
    mag_field2 = parameters.ParameterSet('magnetic_field:dipole',Bpolar=0., beta=0., phi0=0.)
    mesh_bestar = parameters.ParameterSet('mesh:marching',delta=0.1,alg='c')
    mesh_disk = parameters.ParameterSet('mesh:disk')

    comp1,comp2,orbit = binary_from_stars(star, be_star, period=(10.,'d'))

    orbit['ecc'] = 0.60
    orbit['per0'] = 20.,'deg'
    orbit['t0'] = conversions.convert('CD','JD',(1914.,8,6.15)) #1914.6
    orbit['long_an'] = long,'deg'
    orbit['incl'] = 100,'deg'
    orbit['c1label'] = 'betacep'
    orbit['c2label'] = 'Be-star_system'
    
    
    betacep = universe.BinaryStar(star, mesh=mesh_betacep, puls=[freq1],
                                magnetic_field=mag_field1, orbit=orbit)
    bestar = universe.Star(be_star, mesh_bestar, magnetic_field=mag_field2)#, circ_spot=[spot])
    mydisk = universe.AccretionDisk(diskpars, mesh=mesh_disk)
    bestar = universe.BodyBag([bestar, mydisk], orbit=orbit, label='Be-star_system')
    system = universe.BodyBag([betacep,bestar], label='system')
    
    return system

def hierarchical_triple(create_body=True):
    """
    Hierarchical triple system
    """    
    # Create two binary systems, we'll use the first as the inner system, and
    # the second as the outer system
    library_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'library/')
    system = os.path.join(library_dir, 'defaults.phoebe')
    bodybag1, compute = parsers.legacy_to_phoebe2(system)
    bodybag2, compute = parsers.legacy_to_phoebe2(system)
    
    # Inner system: set some labels to clarify that this is the inner system,
    # and make it compact and short-period. Remove the positional parameterSet,
    # that will be the responsibility of the outer system
    bodybag1.set_label("outer_secondary")
    bodybag1[0].set_label("inner_primary")
    bodybag1[1].set_label("inner_secondary")
    bodybag1.params.pop('position')
    bodybag1[0].params['orbit']['sma'] = 5.0
    bodybag1[0].params['orbit']['period'] = 1.54
    
    # Outer system: also set some labels and make it wider and longer-period
    bodybag2.set_label("outer_system")
    bodybag2[0].set_label("outer_primary")
    bodybag2[1].set_label("outer_secondary")
    bodybag2[0].params['orbit']['sma'] = 15.0
    bodybag2[0].params['orbit']['period'] = 10.123
    
    # Make the inner system into a new BodyBag with an orbit
    bodybag1 = universe.BodyBag(bodybag1.bodies, orbit=bodybag2[1].params['orbit'],
                                label='outer_secondary')
    
    # And put the outer primary and inner system in a BodyBag
    system = universe.BodyBag([bodybag2[0], bodybag1])
    
    # That's it!
    return system
        


def pulsating_star(create_body=True):
    """
    Pulsating single star
    """
    star = from_library('Sun', create_body=True)
    star.set_params(parameters.ParameterSet('puls', ampl=0.1, amplteff=0.05, label='puls01'))
    
    return star

def binary_pulsating_primary(create_body=True, npuls=1):
    """
    Binary system with a pulsating primary component
    """
    library_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'library/')
    system = os.path.join(library_dir, 'defaults.phoebe')
    bodybag, compute = parsers.legacy_to_phoebe2(system)
    for i in range(npuls):
        bodybag[0].set_params(parameters.ParameterSet('puls', label='puls{:02d}'.format(i+1)))
    return bodybag


def binary_pulsating_secondary(create_body=True, npuls=1):
    """
    Binary system with a pulsating secondary component
    """
    library_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'library/')
    system = os.path.join(library_dir, 'defaults.phoebe')
    bodybag, compute = parsers.legacy_to_phoebe2(system)
    for i in range(npuls):
        bodybag[1].set_params(parameters.ParameterSet('puls', label='puls{:02d}'.format(i+1)))
    return bodybag
    

#}

if __name__=="__main__":
    import doctest
    doctest.testmod()
