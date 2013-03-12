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

.. autosummary::

    from_library
    star_from_spectral_type
    binary_from_stars
    binary_from_spectroscopy
    T_CrB
    KOI126
    dep_from_object
    

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
      incl 82.75              deg - phoebe Inclination angle
      long 0.0                deg - phoebe Orientation on the sky
  distance 4.84813681108e-06   pc - phoebe Distance to the star
     shape equipot             --   phoebe Shape of surface
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
      incl 90.0           deg - phoebe Inclination angle
      long 0.0            deg - phoebe Orientation on the sky
  distance 10.0            pc - phoebe Distance to the star
     shape equipot         --   phoebe Shape of surface
       alb 1.0             -- - phoebe Bolometric albedo (alb heating, 1-alb reflected)
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
  
>>> Astar = star_from_spectral_type('A0V',label='myA0V')
>>> Bstar = star_from_spectral_type('B3V',label='myB3V')
>>> comp1,comp2,orbit = binary_from_stars(Astar,Bstar,sma=(10,'Rsol'),orbitkwargs=dict(label='myorbit'))
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
     dpdt 0.0                 s/yr - phoebe Period change
   dperdt 0.0               deg/yr - phoebe Periastron change
      ecc 0.0                   -- - phoebe Eccentricity
       t0 0.0                   JD - phoebe Zeropoint date
     incl 90.0                 deg - phoebe Inclination angle
    label myorbit               --   phoebe Name of the system
   period 1.07339522938          d - phoebe Period of the system
     per0 90.0                 deg - phoebe Periastron
  phshift 0.0                   -- - phoebe Phase shift
        q 1.76879729976         -- - phoebe Mass ratio
   vgamma 0.0                 km/s - phoebe Systemic velocity
      sma 10.0                Rsol - phoebe Semi major axis
  long_an 0.0                  deg - phoebe Longitude of ascending node
  c1label myA0V                 --   phoebe ParameterSet connected to the primary component
  c2label myB3V                 --   phoebe ParameterSet connected to the secondary component
 distance 10.0                  pc   phoebe Distance to the binary system
     sma1 4443130136.21        n/a   constr {sma} / (1.0 + 1.0/{q})
     sma2 2511949863.79        n/a   constr {sma} / (1.0 + {q})
totalmass 2.3138943678e+31     n/a   constr 4*pi**2 * {sma}**3 / {period}**2 / constants.GG
    mass1 8.35703779398e+30    n/a   constr 4*pi**2 * {sma}**3 / {period}**2 / constants.GG / (1.0 + {q})
    mass2 1.4781905884e+31     n/a   constr 4*pi**2 * {sma}**3 / {period}**2 / constants.GG / (1.0 + 1.0/{q})
    asini 6955080000.0         n/a   constr {sma} * sin({incl})
      com 4443130136.21        n/a   constr {q}/(1.0+{q})*{sma}
       q1 1.76879729976        n/a   constr {q}
       q2 0.565355906036       n/a   constr 1.0/{q}

Creating Bodies:

>>> the_sun_body = from_library('sun',create_body=True)
>>> myAstar_body = star_from_spectral_type('A0V',create_body=True)
>>> the_system = binary_from_stars(Astar,Bstar,separation='well-detached',create_body=True)

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


logger = logging.getLogger('PARS.CREATE')

#{ General purpose

def make_body(fctn):
    """
    Immediately make a Body if desired (otherwise just apply C{fctn}).
    
    *args are passed to fctn
    **kwargs are passed to get_obs
    """
    @functools.wraps(fctn)
    def make_body_decorator(*args,**kwargs):
        #-- first check if a body needs to be created
        create_body = kwargs.pop('create_body',False)
        #-- then make the parameterSets
        ps = fctn(*args,**kwargs)
        #-- if needed, we create bodies from the returned ParameterSets
        if create_body:
            if not isinstance(ps,tuple) and not isinstance(ps,list):
                ps = ps,
            contexts = [ips.context for ips in ps]
            if args[0] in globals():
                ps = globals()[args[0]](*args[1:],**kwargs)
            #-- let's create two bodies and put them in a BodyBag
            elif len(ps)==3 and contexts.count('component')==2 and contexts.count('orbit')==1:
                myobs1 = observables(**kwargs)
                myobs2 = [myobs.copy() for myobs in myobs1]
                orbit = ps[contexts.index('orbit')]
                comp1 = ps[contexts.index('component')]
                comp2 = ps[::-1][contexts[::-1].index('component')]
                #-- find out which body needs to be made
                body1 = GenericBody(comp1,orbit,pbdep=myobs1)
                body2 = GenericBody(comp2,orbit,pbdep=myobs2)
                #-- if there are irradiators, we need to prepare for reflection
                #   results
                if comp1['irradiator'] or comp2['irradiator']:
                    body1.prepare_reflection()
                    body2.prepare_reflection()
                #-- put the two Bodies in a BodyBag and return it to the user.
                ps = universe.BodyBag([body1,body2])
            #-- OK, if it's no binary, we assume we can deal with it as is
            else:
                myobs = observables(**kwargs)
                ps = GenericBody(*ps,pbdep=myobs)
        return ps
    return make_body_decorator

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
    def __new__(self,*args,**kwargs):
        #-- check which contexts are given
        pbdep = kwargs.pop('pbdep',[])
        if kwargs:
            print kwargs
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
        for attr in dir(universe):
            info[attr] = {'required':[],'missing':[],'rest':[],'optional':{}}
            thing = getattr(universe,attr)
            if inspect.isclass(thing) and hasattr(thing,'__init__'):
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
                    info[attr]['optional']['pbdep'] = pbdep
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
                        info[attr]['required'].append(parameters.ParameterSet(context='mesh:marching'))
                    #-- else, it's just missing and we don't make any assumptions
                    elif not par in contexts:
                        info[attr]['missing'].append(par)
                    else:
                        par = contexts_.pop(contexts_.index(par))
                        info[attr]['required'].append(contexts_args[contexts.index(par)])
                #-- for optional arguments, we only need to check if they are
                #   in the contexts
                for par in arg_optional:
                    if par=='mesh' and not par in contexts:
                        info[attr]['optional']['mesh'] = parameters.ParameterSet(context='mesh:marching')
                    if par in contexts:
                        par_ = contexts_.pop(contexts_.index(par))
                        info[attr]['optional'][par] = contexts_args[contexts.index(par_)]
                info[attr]['rest'] = contexts_
                if not info[attr]['missing'] and not info[attr]['rest'] and len(contexts_)==0:
                    match.append(attr)
        #-- Evaluate and return a Body if possible. Else, try to give some
        #   information about what went wrong
        if len(match)==1:
            logger.info('Input matches {}: creating {} body'.format(match[0],match[0]))
            #-- merge given keyword arguments with derived keyword arguments
            return getattr(universe,match[0])(*info[match[0]]['required'],**info[match[0]]['optional'])
        elif len(match)>1:
            match = match[-1:]
            logger.warning("Given set of arguments is ambiguous: could match any Body of {}".format(", ".join(match)))
            return getattr(universe,match[0])(*info[match[0]]['required'],**info[match[0]]['optional'])
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
        

def observables(pbdep=('lcdep','spdep','ifdep','rvdep'),**kwargs):
    """
    Create a homogeneous set of observables.
    
    Extra keyword arguments are set simultaneously to all observables, resulting
    in a uniform set of observables. E.g., if give an extra keyword argument
    C{passband='JOHNSON.V'}, all observables will have that passband.
    """
    #-- set some defaults
    kwargs.setdefault('ld_func','claret')
    kwargs.setdefault('ld_coeffs','kurucz')
    kwargs.setdefault('atm','kurucz')
    kwargs.setdefault('passband','JOHNSON.V')
    kwargs.setdefault('method','numerical')
    #-- collect observables: it's also possible that observable sets were given
    #   in the obs tuple, directly add them if this is the case
    pbdep_ = []
    for obstype in pbdep:
        if obstype=='lcdep':
            #-- light curve
            lcdep = parameters.ParameterSet(frame='phoebe',context='lcdep')
            pbdep_.append(lcdep)
        elif obstype=='spdep':
            #-- spectrum
            spdep = parameters.ParameterSet(frame='phoebe',context='spdep')
            pbdep_.append(spdep)
        elif obstype=='ifdep':
            #-- interferometry
            ifdep = parameters.ParameterSet(frame='phoebe',context='ifdep')
            pbdep_.append(ifdep)
        elif obstype=='rvdep':
            #-- radial velocity
            rvdep = parameters.ParameterSet(frame='phoebe',context='rvdep')
            pbdep_.append(rvdep)
        elif isinstance(obstype,parameters.ParameterSet):
            pbdep_.append(obstype)
    #-- set extra keyword arguments in the observables ParameterSet, but only
    #   if applicable
    for key in kwargs:
        for iobs in pbdep_:
            if key in iobs:
                iobs[key] = kwargs[key]
        
    return pbdep_    
    
@make_body
def from_library(name,create_body=False,**kwargs):
    """
    Create stars and objects from the library.
    
    In case of a single star, this function returns a single parameterSet.
    In case of a double star, this function returns the primary, secondary and
    orbital parameters.
    
    In case C{name} is a more complex system, it is best to not automatically
    create a body or to call one of the dedicated functions.
    
    @param name: name of the object or system
    @type name: str
    """
    # if the argument is not a filename, try to retrieve the star from the
    # local library
    if not os.path.isfile(name):
        files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'library','*.par'))
        filesl = [os.path.basename(ff).lower() for ff in files]
        if name.lower()+'.par' in filesl:
            name = files[filesl.index(name.lower()+'.par')]
    if not os.path.isfile(name) and name in globals():
        return globals()[name](create_body=create_body,**kwargs)
    if not os.path.isfile(name):
        raise NotImplementedError("system not in library")
    ps = parameters.load_ascii(name)
    logger.info("Loaded {} from library".format(name))
    for pset in ps:
        for kw in kwargs:
            if kw in pset:
                pset[kw] = kwargs[kw]
    
    #-- we have a few special cases here:
    #-- perhaps we're not creating bodies. Then, if the ASCII files contains
    #   only one parameterSet, we just return that one
    if len(ps)==1:
        return ps[0]
    #-- else, we return the list
    else:
        return ps

 
@make_body
def star_from_spectral_type(spectype,create_body=False,**kwargs):
    """
    Create a star with a specific spectral type.
    
    Spectral type information from Pickles 1998 and Habets 1981.
    Rotational velocities from http://adsabs.harvard.edu/abs/1965Obs....85..166M
    
    This function is not meant as a calibration of some sorts, it only returns
    approximate parameters for a star of a given spectral type. I mean
    approximate in the B{really} wide sense.
    
    If the spectral type is not recognised, a list of allowed spectral types
    will be returned.
    
    >>> Astar = spectral_type('A0V')
    >>> print(Astar['teff'])
    9549.92586
    >>> print(Astar['mass'])
    4.202585
    >>> print(Astar['radius'])
    2.843097
    >>> print(Astar['rotperiod'])
    0.821716
    
    @param spectype: spectral type
    @type spectype: str
    """
    #-- stellar parameters that are equal for all stars
    star = parameters.ParameterSet(frame='phoebe',context='star',add_constraints=True)
    star['incl'] = 90.,'deg'
    star['shape'] = 'equipot'
    star['distance'] = 10.,'pc'
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
    
    #-- overwrite with values from the keyword dictionary if possible
    for key in kwargs:
        if key in star:
            star[key] = kwargs[key]
    
    logger.info('Created star of spectral type {}'.format(spectype))
    
    return star    
    


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

@make_body
def binary_from_stars(star1,star2,sma=None,period=None,\
                      kwargs1=None,kwargs2=None,orbitkwargs=None,\
                      create_body=False,**kwargs):
    """
    Create a binary system from two separate stars.
    
    Extra information given is the separation (C{sma}) between the two objects
    or the orbital period. You should give one of them (and only one) as a
    tuple (value,unit).
    
    Extra ``kwargs1`` override defaults in the creation of the primary, extra
    ``kwargs2`` for the secondary, and ``orbitkwargs`` does the same for the
    creation of the orbit.
    
    Extra kwargs are used to create observables if C{create_body=True}. Else,
    they are ignored.
    
    @param star1: ParameterSet representing star 1
    @type star1: ParameterSet
    @param star2: ParameterSet representing star 2
    @type star2: ParameterSet
    @param sma: sma between the two components, and its unit
    @type sma: (float,string)
    @param period: period of the system
    @type period: float
    @param create_body: create a BodyBag containing both objects
    @type create_body: bool
    @return: parameterSets component 1, component 2 and orbit, or BodyBag instance
    @rtype: ParameterSet,ParameterSet,ParameterSet or BodyBag
    """
    if kwargs1 is None:
        kwargs1 = {}
    if kwargs2 is None:
        kwargs2 = {}
    if orbitkwargs is None:
        orbitkwargs = {}
    if sma is not None and period is not None:
        raise ValueError("Give only sma or period, not both")
    
    comp1 = parameters.ParameterSet(context='component',**kwargs1)
    comp2 = parameters.ParameterSet(context='component',**kwargs2)
    orbit = parameters.ParameterSet(context='orbit',**orbitkwargs)
    
    #-- get value on total mass
    M = star1.get_value('mass','Msol')+star2.get_value('mass','Msol')
    #-- get information on semi-major axis
    if sma is not None:
        sma = conversions.convert(sma[1],'au',sma[0])
        period = keplerorbit.third_law(totalmass=M,sma=sma)
        orbit['sma'] = sma,'au'
        orbit['period'] = period,'d'
    elif period is not None:
        period = conversions.convert(period[1],'d',period[0])
        sma = keplerorbit.third_law(totalmass=M,period=period)
        orbit['sma'] = sma,'au'
        orbit['period'] = period,'d'
    
    orbit['q'] = star2.get_value('mass','Msol')/star1.get_value('mass','Msol')
    
    logger.info("Creating binary: sma={:.3g}au, period={:.3g}d, ecc={:.3g}, q={:.3g}".format(orbit['sma'],orbit['period'],orbit['ecc'],orbit['q']))
    
    if not 'syncpar' in kwargs1:
        comp1['syncpar'] = orbit.request_value('period','d')/star1.request_value('rotperiod','d')
    if not 'syncpar' in kwargs2:
        comp2['syncpar'] = orbit.request_value('period','d')/star2.request_value('rotperiod','d')
    
    logger.info('Creating binary: syncpar1={:.3g}, syncpar2={:.3g}'.format(comp1['syncpar'],comp2['syncpar']))
    
    R1 = star1.get_value('radius','au')
    R2 = star2.get_value('radius','au')
    R1crit = roche.calculate_critical_radius(orbit['q'],component=1,F=comp1['syncpar'])*sma
    R2crit = roche.calculate_critical_radius(orbit['q'],component=2,F=comp2['syncpar'])*sma
    #if R1>R1crit: raise ValueError("Star1 radius exceeds critical radius: R/Rcrit={}".format(R1/R1crit))
    #if R2>R2crit: raise ValueError("Star2 radius exceeds critical radius: R/Rcrit={}".format(R2/R2crit))
    comp1['pot'] = roche.radius2potential(R1/sma,orbit['q'],component=1,F=comp1['syncpar'])
    comp2['pot'] = roche.radius2potential(R2/sma,orbit['q'],component=2,F=comp2['syncpar'])
    
    for key in ['teff']:
        comp1[key] = star1.get_value_with_unit(key)
        comp2[key] = star2.get_value_with_unit(key)
    for key in ['atm','label','ld_func','ld_coeffs']:
        comp1[key] = star1[key]
        comp2[key] = star2[key]
    orbit['c1label'] = comp1['label']
    orbit['c2label'] = comp2['label']
    logger.info('Creating binary: pot1={:.3g}, pot2={:.3g}'.format(comp1['pot'],comp2['pot']))
    return comp1,comp2,orbit    


@make_body
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
    
    @param period: period in days
    @param K1: semi-amplitude in km/s
    """
    logger.info("Creating binary from spectroscopy")
    #-- take care of logg parameters
    for i,(star,logg) in enumerate(zip([star1,star2],[logg1,logg2])):
        if logg is not None:
            tools.add_surfgrav(star,logg,derive='mass')   
    
    #-- period, ecc and q
    comp1 = parameters.ParameterSet(context='component')
    comp2 = parameters.ParameterSet(context='component')
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
    else:
        q = star2['mass']/star1['mass']
        K2_ms = K1_ms/q
    
    orbit['q'] = q
    logger.info('Mass ratio q = {}'.format(q))
    asini = conversions.convert('m','Rsol',keplerorbit.calculate_asini(period_sec,ecc,K1=K1_ms,K2=K2_ms))
    tools.add_asini(orbit,asini,derive='incl',unit='Rsol')        
    
    #-- calculate the mass function
    fm = keplerorbit.calculate_mass_function(period_sec,ecc,K1_ms)
    
    #-- inclination angle from masses and mass function, but make sure it's
    #   an allowed value
    for i in range(2):
        M1 = star1.request_value('mass','Msol')
        M2 = star2.request_value('mass','Msol')
        orbit['sma'] = keplerorbit.third_law(totalmass=M1+M2,period=orbit['period']),'au'
        if np.isnan(orbit['incl']):
            logger.warning("Invalid masses: sma={} but asini={}".format(orbit['sma'],orbit['asini']))
            star1['mass'] = fm*(1+orbit['q'])**2/orbit['q']**3,'kg'
            star2['mass'] = q*star1['mass']
        else:
            break
    
    logger.info("Total mass of the system = {} Msol".format(star1['mass']+star2['mass']))
    logger.info("Inclination of the system = {} deg".format(orbit['incl']))
    
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
    
    sma = orbit.get_value_with_unit('sma')
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
    orbit,red_giant,white_dwarf,disk = from_library('T_CrB',create_body=False,**kwargs)
    disk['label'] = white_dwarf['label']
    if create_body:
        myobs1 = observables(**kwargs)
        myobs2 = [myobs.copy() for myobs in myobs1]
        myobs3 = [myobs.copy() for myobs in myobs2]
        mesh = parameters.ParameterSet(context='mesh:marching')
        body1 = universe.BinaryRocheStar(red_giant,orbit=orbit,mesh=mesh,pbdep=myobs1)
        body2 = universe.BinaryRocheStar(white_dwarf,orbit=orbit,mesh=mesh.copy(),pbdep=myobs2)
        body3 = universe.AccretionDisk(disk,pbdep=myobs3)
        body3 = universe.BodyBag([body3],orbit=orbit.copy())#,label=white_dwarf['label'])
        #-- if there are irradiators, we need to prepare for reflection
        #   results
        if red_giant['irradiator'] or white_dwarf['irradiator']:
            body1.prepare_reflection()
            body2.prepare_reflection()
            body3.prepare_reflection()
        #-- put the two Bodies in a BodyBag and return it to the user.
        output = universe.BodyBag([body1,body2,body3],solve_problems=True)
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
    
    #-- inner binary
    orbitBC = parameters.ParameterSet(frame='phoebe',context='orbit',add_constraints=True,\
              c1label=starB['label'],c2label=starC['label'])
    orbitBC['period'] = 1.76713,'d'
    orbitBC['sma'] = 0.021986,'au'
    orbitBC['ecc'] = 0.02334
    orbitBC['per0'] = 89.52,'deg'
    orbitBC['incl'] = 96.907-92.100,'deg'
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
    
    
    starB = universe.BinaryStar(starB,orbitBC,meshB,pbdep=[lcdep2])
    starC = universe.BinaryStar(starC,orbitBC,meshC,pbdep=[lcdep3])
    starA = universe.BinaryStar(starA,orbitA_BC,meshA,pbdep=[lcdep1])
    systemBC = universe.BodyBag([starB,starC],orbit=orbitA_BC,label='systemBC')
    
    system = universe.BodyBag([starA,systemBC])
    return system
    
def KOI126_alternate(create_body=True,**kwargs):
    """
    Kepler Object of Interest 126 - a hierarchical version of the triple system
    """
    T0 = 2455170.5
    starA = parameters.ParameterSet(frame='phoebe',context='star',add_constraints=True)
    starA['teff'] = 5875,'K'
    starA['mass'] = 1.347,'Msol'
    starA['surface'] = 'sphere'
    starA['radius'] = 2.0254,'Rsol'
    starA['rotperiod'] = np.inf
    starA['ld_model'] = 'claret'
    starA['atm'] = 'kurucz'
    starA['cl'] = 'kurucz'
    
    starB = parameters.ParameterSet(frame='phoebe',context='star',add_constraints=True)
    starB['teff'] = 3000,'K'
    starB['mass'] = 0.2413,'Msol'
    starB['surface'] = 'sphere'
    starB['radius'] = 0.2543,'Rsol'
    starB['rotperiod'] = np.inf
    starB['ld_model'] = 'linear'
    starB['atm'] = 'blackbody'
    starB['cl'] = [0.5]
    
    starC = parameters.ParameterSet(frame='phoebe',context='star',add_constraints=True)
    starC['teff'] = 2800,'K'
    starC['mass'] = 0.2127,'Msol'
    starC['surface'] = 'sphere'
    starC['radius'] = 0.2318,'Rsol'
    starC['rotperiod'] = np.inf
    starB['ld_model'] = 'linear'
    starB['atm'] = 'blackbody'
    starB['cl'] = [0.5]
    
    #-- inner binary
    orbitBC = parameters.ParameterSet(frame='phoebe',context='orbit',add_constraints=True)
    orbitBC['period'] = 1.76713,'d'
    orbitBC['sma'] = 0.021986,'au'
    orbitBC['ecc'] = 0.02334
    orbitBC['per0'] = 89.52,'deg'
    orbitBC['incl'] = 96.907-92.100,'deg'
    orbitBC['long'] = 8.012,'deg'
    
    #-- outer binary
    orbitA_BC = parameters.ParameterSet(frame='phoebe',context='orbit',add_constraints=True)
    orbitA_BC['period'] = 33.9214,'d'
    orbitA_BC['sma'] = 0.2495,'au'
    orbitA_BC['ecc'] = 0.3043
    orbitA_BC['per0'] = 52.88,'deg'
    orbitA_BC['incl'] = 92.100,'deg'
    
    
    
    mean_anomaly = 19.87/180.*np.pi
    orbitA_BC['t0'] = T0-keplerorbit.mean_anomaly_to_time(mean_anomaly,starA.request_value('mass','kg')+\
                          starB.request_value('mass','kg')+starC.request_value('mass','kg'),\
                          orbitA_BC.request_value('sma','m'))/(3600*24.)
    mean_anomaly = 355.66/180.*np.pi
    orbitBC['t0'] = T0-keplerorbit.mean_anomaly_to_time(mean_anomaly,starB.request_value('mass','kg')+\
                          starC.request_value('mass','kg'),\
                          orbitBC.request_value('sma','m'))/(3600*24.)    
    
    meshA = parameters.ParameterSet(frame='phoebe',context='mesh')
    meshB = parameters.ParameterSet(frame='phoebe',context='mesh')
    meshC = parameters.ParameterSet(frame='phoebe',context='mesh')
    
    #-- light curve
    lcdep1 = parameters.ParameterSet(frame='phoebe',context='lcdep')
    lcdep1['ld_model'] = 'claret'
    lcdep1['cl'] = 'kurucz'
    lcdep1['atm'] = 'kurucz'
    lcdep1['passband'] = 'JOHNSON.V'
    lcdep1['ref'] = 'light curve'
    lcdep2 = lcdep1.copy()
    lcdep2['ld_model'] = 'linear'
    lcdep2['cl'] = [0.5]
    lcdep2['atm'] = 'blackbody'
    lcdep3 = lcdep2.copy()
    
    
    starA = universe.BinaryStar(starA,None,meshA,obs=[lcdep1])
    starB = universe.BinaryStar(starB,None,meshB,obs=[lcdep2])
    starC = universe.BinaryStar(starC,None,meshC,obs=[lcdep3])
    
    systemBC = universe.BinaryBag([starB,starC],orbit=orbitBC)
    systemA_BC = universe.BinaryBag([starA,systemBC],orbit=orbitA_BC)
    
    return systemA_BC

#}

if __name__=="__main__":
    import doctest
    doctest.testmod()