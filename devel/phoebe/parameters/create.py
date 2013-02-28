"""
Library containing parameters of known stars, systems and objects.

THis module contains one-liners to create default parameterSets and Bodies
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

Example usage:

Only involving ParameterSets:

>>> the_sun_pars = from_library('sun')
>>> myAstar_pars = star_from_spectral_type('A0V')


>>> Astar = star_from_spectral_type('A0V')
>>> Bstar = star_from_spectral_type('B3V')
>>> comp1,comp2,orbit = binary_from_stars(Astar,Bstar,separation='well-detached')

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
from phoebe.backend import universe


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
            raise ValueError("Given set of arguments is ambiguous: could match any Body of {}".format(", ".join(match)))
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
    star['surface'] = 'roche'
    star['distance'] = 10.,'pc'
    star['ld_func'] = 'claret'
    star['atm'] = 'kurucz'
    star['ld_coeffs'] = 'kurucz'
    star['label'] = spectype+'_'+star['label']
    
    #-- stellar parameter dependent on spectral type: temperature, radius, mass
    #   and rotation period (luminosity follows from these)
    
    #-- first get information
    data = utils.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),'catalogs','spectypes.dat'),dtype=np.str)
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