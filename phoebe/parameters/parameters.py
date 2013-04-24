"""
Definitions of classes representing parameters.

Summary
=======

.. autosummary::

    Parameter
    ParameterSet
    Distribution

Section 1. Parameter preparation
================================

Section 1.1 Basic usage
-----------------------

Section 1.1.1 Constructs
~~~~~~~~~~~~~~~~~~~~~~~~

The parsing of parameters is done via a L{ParameterSet}, which can be
regarded as a modified (nested) ordered dictionary. Default parameters can be
loaded (the following two lines are equivalent and show the default behaviour):

>>> ps = ParameterSet()
>>> ps = ParameterSet(frame='phoebe',context='star')

The context of parameters is set by the keyword C{context}, which defaults to
C{phoebe}. The different contexts represent different types of L{ParameterSet}s:
parameters describing binary systems as a whole (C{orbt}, i.e., the period,
eccentricity...) are different than the ones describing the a light curve
(C{lcdep}, i.e. passband, limb darkening coefficients, atmosphere tables...).

A L{ParameterSet} acts like a dictionary, i.e. you can ask for its keys, cycle
through it, ask and change values:

>>> print(ps.keys())
['teff', 'radius', 'mass', 'atm', 'rotperiod', 'diffrot', 'gravb', 'incl', 'long', 'distance', 'shape', 'alb', 'redist', 'irradiator', 'abun', 'label', 'ld_func', 'ld_coeffs']
>>> print(ps['radius'])
1.0
>>> ps['radius'] = 11.7
>>> print(ps['radius'])
11.7

Upon creation, you can immediately set some of the keywords to override the
default values:

>>> mylc = ParameterSet(frame='phoebe',context='lcdep',passband='GENEVA.V')

You can, if you wish, nest L{ParameterSets}:

>>> ps.add(Parameter(qualifier='my_lc_curve',value=mylc))

you then intuitively access all keywords via a nested structure:

>>> ps['my_lc_curve']['passband'] = 'JOHNSON.B'

and check your results have been passed on via

>>> print(ps['my_lc_curve']['passband'])
JOHNSON.B

To remove a parameter,simply do:

>>> out = ps.pop('my_lc_curve')

The L{ParameterSet}s can then be passed on to different codes. Different codes
usually mean different frames.

Section 2.1.2 Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to put a constraints on L{Parameter}s, but only in the context of
a L{ParameterSet}, since constraints are in general given with respect to the
other L{Parameter}s. As an example, we construct an orbit L{ParameterSet} and
add a new L{Parameter} called C{asini},

>>> ps_constr = ParameterSet(context='orbit')
>>> ps_constr.add(Parameter(qualifier='asini',value=12.1,unit='Rsol'))

which represent the value of the semi-major axis times the sine of the
inclination. This could originate e.g. from radial velocity data. If you only
want to change C{asini} and C{incl}, and desire that C{sma} changes
accordingly, you can do:

>>> ps_constr.add_constraint('{sma} = {asini}/sin({incl})')

Parameter qualifiers must be given within curly brackets, The expression on the
right hand side will be evaluated as if it were pure Python code, but with basic
mathematical functions readily available (though all numpy functions are
available via the prefix C{np.}). You can include numbers also, but make sure
they are in B{SI units} regardless of the frame or context. You can also include
physical constants via C{constants.GG} for the gravitational constant etc.

If you change either C{asini} or C{incl}, C{sma} will automatically be updated.
Changing C{sma} is no longer possible: the L{ParameterSet} will execute all
defined constraints when a variable is set, so after setting C{sma}, the
L{ParameterSet} will reset C{sma} to satisfy the constraint. To remove (and
return) the constraint on a qualifier, simply do

>>> out = ps_constr.remove_constraint('sma')

From this syntax it is clear that you cannot put different constraints on one
qualifier, and if you add a new contraint on a Parameter that was constrained
before, the old one will be overwritten in favour of the new one. This
satisfies the principle of least surprise.

Note that the constraints are only run when a parameter is I{set} (i.e. via
L{ParameterSet.__setitem__}, and I{not when asking for a value}. This is to 
save time, so that the constraints are not executed each time you retrieve a
value while nothing has changed.

It is not mandatory to have the right hand side of the constraint defined as
a parameter. You could simply do:

>>> ps_constr = ParameterSet(frame='phoebe',context='orbit')
>>> ps_constr.add_constraint('{asini} = {sma}*sin({incl})')

And now C{asini} can be queried via:

>>> ps_constr.get_constraint('asini','Rsol')
200.0

If you do not specify the units, they will be SI. The parameter C{asini} is not
visible when accessing L{ParameterSet.keys()} and it is not listed in the string
representation of the ParameterSet.


Section 2.1.3 Input and output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contents of a ParameterSet can be displayed for viewing purposes via
the command

>>> #print(ps)

    
A ParameterSet can easily be saved and loaded to and from a binary file:

>>> ps.save('mytest.par')
>>> ps2 = load('mytest.par')
>>> print(str(ps)==str(ps2))
True
>>> os.unlink('mytest.par') # clean up

A limited interface to ASCII files is also provided. Only the keywords and
values are actually written to the file, i.e. all information on units,
flags, descriptions etc is ommitted. Any change made to the ASCII file can
be read in using the L{load_ascii} function, but again with limited functionality.
If units are disputed, one can always have a look at the context/framework
of the parameterSet, and then look up the definitions in L{definitions.py}. It
is impossible to read and write ParameterSets that are not predefined:

>>> ps.save_ascii('mytest2.par')
>>> os.unlink('mytest2.par') # clean up

You can also read and write a bunch of parameterSets to one file:

>>> ps2 = ps.copy()
>>> ps2['label'] = 'bla'
>>> save_ascii('myfile.par',ps,ps2)
>>> ps,ps2 = load_ascii('myfile.par')
>>> os.unlink('myfile.par') # clean up

Section 2.1.4 Units
~~~~~~~~~~~~~~~~~~~

Parameters are smart in the sense that they can have units and know about them.
When converting from one frame to another, these conversions are taken into
account, so the user shouldn't worry about them and work in the predefined
units framework given by each frame (see L{parameter_definitions}). However, if
the user really wants to give a parameter in other units, a parameter in any given
units can be passed as a tuple (value,units) instead of only a value:

E.g., if you desperately want to give period in seconds and argument of
periastron in radians:

>>> bps = ParameterSet(context='orbit')
>>> bps['period'] = 23.4*86400.,'s'
>>> bps['per0'] = 3.14,'rad'

Internally, the value is converted to whatever units were defined in this module's
framework:

>>> print bps['period']
23.4
>>> print bps['per0']
179.908747671

However, again if you're desparate, you can get a parameter in whatever units
you like:

>>> print bps.get_value('period','s')
2021760.0
>>> print bps.get_value('per0','rad')
3.14

Section 2.2 Fitting
-------------------

Section 2.2.1 Setting parameters to be fitted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, none of the parameters are marked for fitting. This made clear to
the code via the attribute C{adjust}. If C{adjust=False}, the parameter will
not be adjusted in the fit. Otherwise it will be.

>>> bps.get_parameter('period').set_adjust(True)

is equivalent to

>>> bps.set_adjust(True,'period')


Section 2.2.2 Setting limits and priors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two distinct interfaces to set a valid range for a parameter value.
One is via C{limits}, the other via C{priors}. These are different concepts,
and their values I{can} be equal but they don't have to be.

Limits are real limits on the values of the parameters: e.g. it doesn't make
sense to set a negative effective temperature, or to set an inclination angle
of 1000 degrees. Trying to set a value to a parameter outside of these limits,
will raise a warning and reset it to be equal to the closest limit. This
information can be useful also for GUI purposes.

>>> bps.get_parameter('period').set_limits(1.,100.)

Priors are B{distributions}, not B{limits}. If the distribution is uniform,
then the distribution limits can be equal to C{limits}, but they don't have
to be. For example, although a physical limit for the effective temperature
is 0 Kelvin, you know that the temperature of a star will be way higher than
that, and for example set the prior to a uniform distribution with limits
3000K and 10000K.

>>> bps.get_parameter('period').set_prior(distribution='uniform',lower=1,upper=100.)



"""
#-- load standard modules
import os
import re
import copy
import pickle
import functools
import uuid
import logging
import json
import inspect
import difflib
from collections import OrderedDict
#-- load extra 3rd party modules
from numpy import sin,cos,sqrt,log10,pi,tan,exp
import numpy as np
from scipy.stats import distributions
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Soft warning: matplotlib could not be found on your system, 2D plotting is disabled, as well as IF functionality")
#-- load self defined modules
from phoebe.units import conversions
from phoebe.units import constants
from phoebe.utils import decorators
from phoebe.parameters import definitions as defs
try:
    import pymc
except ImportError:
    print("Unable to load pymc: restricted fitting facilities")
  
logger = logging.getLogger("PARAMETERS")  

pattern = re.compile('[\W_]+')
default_frame = 'phoebe'
default_context = 'star'

#{ Internal helper functions

def parse_plotkwargs(fctn):
    """
    """
    @functools.wraps(fctn)
    def parse(self,*args,**kwargs):
        """
        default if no plotset is given, any addition kwargs should override given plotset or defaults
        """
        plotset = args[0] if len(args) else kwargs.pop('plotset',ParameterSet(context='plotting'))
        #if linestyle has not been set, make decision based on type
        if plotset['linestyle'] is 'default':
            plotset['linestyle'] = 'None' if self.context[-3:] == 'obs' else '-' 
        if 'ref' in list(plotset.keys()):
            ref = plotset.pop('ref')
        for key in kwargs:
            plotset[key]=kwargs[key]
        return fctn(self,plotset=plotset,**kwargs)
    return parse

@decorators.memoized
def get_frames_and_contexts(defs):
    frames = []
    contexts = []
    for idef in defs:
        frames += (idef['frame'] if isinstance(idef['frame'],list) else [idef['frame']])
        contexts += (idef['context'] if isinstance(idef['context'],list) else [idef['context']])
    
    return set(frames),set(contexts)

def attach_signal(self,funcname,callback,*args):
    """
    Attach a signal to a function
    """
    if not hasattr(self,'signals'):
        self.signals = {}
    if funcname not in self.signals:
        self.signals[funcname] = []
    self.signals[funcname].append((callback,args))
    
def callback(fctn):
    """
    Provides a callback to any function after the function is executed.
    """
    @functools.wraps(fctn)
    def add_callback(self,*args,**kwargs):
        output = fctn(self,*args,**kwargs)
        #-- possibly the "self" has no signals: then do nothing but
        #   return the stuff
        if not hasattr(self,'signals'):
            return output
        #-- possibly the "self" has signals but the called function
        #   has no signals attached to it. If so, Just return the output
        if not fctn.__name__ in self.signals:
            return output
        #-- else, we need to execute all the functions in the callback list
        for func_name,func_args in self.signals[fctn.__name__]:
            func_name(self,*func_args)
        #-- we need to return the output anyway
        return output
    return add_callback


            
#}

#{ Base classes

class Parameter(object):
    """
    Class Parameter represent a parameter and its properties.
    
    The parameter can describe a binary system, observational data or
    computational specifications.
    
    B{Section 0. Overview}
    
    **Retrieve Parameter information**
    
    .. autosummary::
    
       Parameter.get_value
       Parameter.get_qualifier
       Parameter.get_description
       Parameter.get_unit
       Parameter.get_adjust
       Parameter.get_limits
       Parameter.get_step
       Parameter.get_choices
       Parameter.get_cast_type
       Parameter.get_context
       Parameter.get_prior
       Parameter.get_posterior
       Parameter.get_value_from_prior
       Parameter.get_value_from_posterior
       Parameter.get_unique_label
       
    .. autosummary::
    
        Parameter.has_unit
        Parameter.has_prior
        Parameter.has_posterior
        Parameter.has_limits
        Parameter.is_lim
        Parameter.list_available_units
        
    **Set existing Parameter information**
    
    .. autosummary::
    
        Parameter.set_value
        Parameter.set_value_from_prior
        Parameter.set_value_from_posterior
        Parameter.set_unit
        Parameter.set_adjust
        Parameter.set_limits
        Parameter.set_step
        Parameter.set_prior
        Parameter.set_posterior
    
    **Add/remove Parameter information**
    
    .. autosummary::
    
        Parameter.add_limits
        Parameter.add_choice
        Parameter.remove_choice
    
    B{Section 1. Implementation details}
    
    B{Section 1.1 Casting}
    
    A Parameter accepts a (raw) value from a user and a user can ask the value
    back. In the process of asking the value of the Parameter, a B{casting}
    function can be added. That is, upon initalization, a keyword C{cast_type}
    can be given. In general, the value should be a function. In its simplest
    form, this can be the C{float} or C{int} function, which guarentees that
    the user-given value is of the correct type. Thus, the user can input a
    string containing a float, but when the L{get_value} function is called,
    the right casting is done and programs like Phoebe or pyWD really receive
    a float instead of a string.
    
    More sophisticated examples could be a read function, which assumes that
    C{value} is a filename, and cast the filename to a data array. Another
    common casting mechanism is casting a string to a flag, such as for the
    filter names in the WD code. Specifically for those situations, I also
    allowed for a shortcut C{index} or C{indexf} string as a cast_type (which
    are not functions). In that case, the casting algorithm knows it should take
    the Python- or Fortran-style index of C{value} out of a list of C{choices}.
    
    In summary, C{cast_type} guarantees the value of parameter being of a
    specific type.
    
    B{Section 1.2 Units}
    
    Some parameters have units. When applicable, a Parameter instance has a
    default unit, and if the user gives a value, it is silently assumed that the
    value is the same as the default unit. However, when the Parameter value is
    set by the user and an extra argument (the unit as a string) is given,
    a conversion from the given unit to the default unit is performed before
    storing the value. Thus, it is impossible to retrieve afterwards in which
    unit the user has given the value of the parameter originally.
    
    B{Section 1.3 Priors and posteriors}
    
    Some parameters have priors. Usually, also posteriors can be computed from
    them using MCMC algorithms.
    
    B{Section 2. Example usage}
    
    We define one parameter but for two different frames. The C{filter} parameter
    needs to be cast to an integer in the WD code, but in general it is just
    cast to the string representation (so it actually doesn't change the input
    value).
    
    >>> def1 = dict(qualifier='filter',description='Filter name',repr='%s',cast_type=str,value='JOHNSON.V',frame=["main"])
    >>> def2 = dict(qualifier='filter',description='Filter name',choices=['STROMGREN.U','stromgren.v','stromgren.b','stromgren.y',
    ...                                                             'johnson.U','johnson.B','JOHNSON.V','johnson.R','johnson.I','johnson.J','johnson.K','johnson.L','johnson.M','johnson.N',
    ...                                                             'bessell.RC','bessell.IC',
    ...                                                             'kallrath.230','kallrath.250','kallrath.270','kallrath.290','kallrath.310','kallrath.330',
    ...                                                             'tycho.B','tycho.V','hipparcos.hp','COROT.EXO','COROT.SIS','JOHNSON.H',
    ...                                                             'GENEVA.U','GENEVA.B','GENEVA.B1','GENEVA.B2','GENEVA.V','GENEVA.V1','GENEVA.G',
    ...                                                             'Kepler.V','SDSS.U'],repr='%s',cast_type='indexf',value='johnson.V',frame=['wd'])
    >>> par1 = Parameter(**def1)
    >>> print par1
    Name:           filter
    Description:    Filter name
    Value:          JOHNSON.V
    Raw value:      JOHNSON.V
    Type:           <type 'str'>
    Frame:          main
    
    For convenience, it is also possible to immediately use the predefined
    parameters from L{defs.defs}:
    
    >>> par1 = Parameter(qualifier='filter',frame='main')
    
    You can change or display the value:
    
    >>> par1.set_value('COROT.EXO')
    >>> print par1.get_value()
    COROT.EXO
    
    The string interpretation is pretty smart when the value is limited to a
    list of choices (key C{choices} in the parameter dictionary). Under the hood,
    it will only compare lower case strings, stripped from all non-alphanumeric
    characeters. In this framework, 'xra' is equivalent with 'X-ray binary' as
    long as there is no ambiguity with other parameters.
    
    A second example includes units and speaks for itself.
    
    >>> def3 = dict(qualifier="period",description="Orbital period in days",
    ...           repr= "%14.6f", llim=  0.0, ulim=  1E10, step= 0.0001,
    ...           adjust=False, cast_type=float, unit='d',value= 22.1891087,
    ...           frame=["main","wd","jktebop"],alias=['p','phoebe_period.val'])
    >>> par2 = Parameter(**def3)
    >>> par2.set_value(21)
    >>> print par2.get_value()
    21.0
    >>> par2.set_value(0.1,'yr')
    >>> print par2.get_value()
    36.525
    
    You can define your own casting functions. The C{cast_type} should then be:
        
        1. A function existing in the current name space
        2. A function's name (str) available in the global name space of the
        C{meb} module.
        
    The function's call signature should be (value,*args), thus containing
    optional positional arguments which a user can add when calling the
    L{set_value} function. The function is entirely free in the type of the
    return value. An example casting function is L{filename2data}.
    
    
    
    """
    #{ General methods
    def __init__(self,qualifier=None,**props):
        """
        Specify parameter properties via the dictionary C{props}. It has to
        contain the follliming fields:
        
            1. C{qualifier}: name of the parameter
            2. C{description}: a description of the parameter
            3. C{repr}: string representation of the parameter
            4. C{cast_type}: type to cast the value to (str,float,int,list...)
            5. C{value}: value of the parameter
        
        Optionally, you can give parameter ranges, steps, or constrain the user
        input to a specific set of choices, via:
            
            6. C{llim}: lower limit on the parameter
            7. C{ulim}: upper limit on the parameter
            8. C{step}: step size of the parameter
            9. C{choices}: list of possible values
        
        The keys in C{props} will be set as attributes of this Parameter class.
        
        There are several ways one can define a parameter: one can use the default
        definitions given in L{defs.defs}:
        
        >>> par = Parameter(qualifier='filter')
        >>> par = Parameter(qualifier='filter',frame='wd',context='lc')
        >>> par = Parameter(qualifier='wd.lc.filter')
        
        or one can explicitly list all the properties:
        
        >>> par = Parameter(qualifier='filter',description='Filter name',
        ...    repr='%s',cast_type=str,value='JOHNSON.V',frame=["main"])
        
        @parameter props: all information on the parameter
        @type props: dictionary
        """
        #-- remember what I am
        frame = props.pop('frame','main')
        context = props.pop('context',None)
        prior = props.pop('prior',None)
        #-- if no qualifier is given, we don't know what to do...
        if qualifier is None:
            raise ValueError('Parameter instance needs at least a qualifier as an argument')
        #-- maybe the qualifier is given as 'wd_lc_ld', i.e. 'frame_context_qualifier'
        if '.' in qualifier:
            frame,context,qualifier = qualifier.split('.')
            if context.lower()=='none':
                context = None
        #-- if only a qualifier (and optionally the frame) is given, look up
        #   the name or alias in the existing parameter definitions
        elif not props:
            for idef in defs.defs:
                #-- is the given qualifier equal to this definitions' qualifier
                #   or alias?
                if not qualifier== idef['qualifier'] or (hasattr(idef,'alias') and not qualifier in idef['alias']):
                    continue
                #-- is the frame correct?
                if not frame in idef['frame']: continue
                #-- is the context correct?
                if context and context!=idef['context']: continue
                #-- else, this is the correct one
                props = idef
                break
        props['qualifier'] = qualifier
        props['frame'] = isinstance(frame,str) and [frame] or frame
        props['context'] = isinstance(context,str) and [context] or context
        #-- a Parameter must at least implement these properties
        props.setdefault('description','No description available')
        props.setdefault('repr','%s')
        props.setdefault('cast_type',return_self)
        props.setdefault('value',0)
        if prior is not None:
            self.set_prior(**prior)
        
        #-- set a unique label, if parameter is 'label' but no string is given
        if props['qualifier'][-5:]=='label' and not props['value']:
            props['value'] = uuid.uuid4()
        #-- set a unique label, if parameter is 'ref' but no string is given
        if props['qualifier'][-3:]=='ref' and not props['value']:
            props['value'] = uuid.uuid4()
        
        #-- remember initial settings
        self._initial = props.copy()
        #-- keep a flag if it's opaque or not
        #self.opaque = False
        #-- attach all keys to the class instance
        self.reset()    
        #-- what is the default behaviour for when limits our outside of bounds?
        self._force_inside_limits = True
    
    def reset(self):
        """
        Reset the parameter values to its initial values.
        """
        for key in self._initial:
            setattr(self,key,self._initial[key])
    
    def clear(self):
        """
        Strip this instance from its properties
        """
        for key in self._initial:
            delattr(self,key)
    
    #}    
    #{ Get parameter properties
    def get_value(self,*args):
        """
        Cast the value of a parameter to the right type and return it.
        
        Accepted values for C{cast_type} are:
            
            1. 'indexf' (str): Fortran-style index of C{value} in C{choices}
            will be returned.
            2. 'index' (str): Python-style index of C{value} in C{choices} will
            be returned
            3. 'list' (str): a list will be returned, perhaps containing only
            one element.
            4. other string: the string will be interpreted as the name of a
            function in this module's global name space. C{value} and C{*args}
            will be parsed to that function.
            5. function: C{value} and C{*args} will be parsed to C{function}.
        
        In all other cases, it will be assumed that C{value} is a physical
        quantity (maybe we want change this and explicitly give a unit?), so
        units can be given as a second positional argument to L{get_value}.
        @return: value
        @rtype: any
        """
        #-- compile the pattern that will remove non-alphanumeric characters
        #   this will only be used if C{self.value} is a string
        try:
            #-- if C{self.value} is a string and a (Fortran) index should be
            #   returned, get the index of the corresponding value in C{self.choices}.
            #   Remove nonalphanumeric characters and check for minimal correspondence
            #   (i.e. if 'X-ray binary' is a choice, 'xra' should be enough to
            #   identify it).
            if isinstance(self.cast_type,str) and ('index' in self.cast_type or 'choose' in self.cast_type) and isinstance(self.value,str):
                index = match_string(self.value,self.choices)
                if args: raise TypeError('conversion not possible for this type')
                #-- if not found, be desparate: maybe the string is the index?
                if index is None:
                    return int(float(self.value))
                elif 'index' in self.cast_type:
                    return index + (self.cast_type=='indexf' and 1 or 0)
                elif 'choose' in self.cast_type:
                    return self.choices[index]
                else:
                    raise ValueError('cannot cast')
            #-- may be an index needs to returned, but an index is already given.
            #   It's the user responsibility to be sure it exists!
            elif isinstance(self.cast_type,str) and  'index' in self.cast_type:
                if args: raise TypeError('conversion not possible for this type')
                return int(self.value)
            elif isinstance(self.cast_type,str) and self.cast_type=='list':
                if args: raise TypeError('conversion not possible for this type')
                #-- the user gave one value, but it should actually be a list
                retval = self.value
                if isinstance(self.value,str):
                    retval = json.loads(retval)
                if not hasattr(retval,'__iter__'):
                    return [retval]
                else:
                    return [float(i) for i in list(retval)]
            #-- maybe the cast_type is a string representing a function's name
            #   in the local namespace.
            elif isinstance(self.cast_type,str):
                return globals()[self.cast_type](self.value,*args)
            #-- maybe we can easily cast the value to the right type (i.e. the
            #   'cast_type' is a function
            else:
                #-- in this case, we can try to convert something to the right
                #   units
                casted_value = self.cast_type(self.value)
                if args:
                    casted_value = conversions.convert(self.unit,args[0],casted_value)
                return casted_value
        #-- catch *ANY* exception, but throw it back out there with some extra
        #   information
        except Exception as msg:
            raise TypeError("qualifier '{}': cannot cast {} to {} (original message: {})".format(self.qualifier,self.value,self.cast_type,msg))
    
    def get_input(self,*args):
        """
        Return value given by user
        
        @return: value
        @rtype: any
        """
        return self.value
    
    def get_description(self):
        """
        Get the description.
        
        @return: the description
        @rtype: str
        """
        return self.description
    
    def get_unit(self):
        """
        Get the unit of a parameter.
        
        @return: the unit
        @rtype: str
        """
        if hasattr(self,'unit'):
            return self.unit
        else:
            raise ValueError('Parameter {0} has no units'.format(self.qualifier))
    
    def get_adjust(self):
        """
        See if a parameter is adjustable.
        
        A parameter is not adjustable if:
        
            - its adjust parameters is False
            - it has no adjust parameter (then None is returned)
        
        
        @return: the adjust parameter
        @rtype: bool/None
        """
        if hasattr(self,'adjust'):
            return self.adjust
    
    def get_qualifier(self,alias=None):
        """
        Return the qualifier or check if it has a certain alias.
        
        Aliases are B{not} case sensitive.
        
        Returns the qualifier if 'alias' is the qualifier itself or in the
        aliases list. Else, it returns None
        
        @parameter alias: alias of the qualifier
        @type alias: str
        @return: name of the qualifier
        @rtype: str
        """
        if alias is None:
            return self.qualifier
        if self.qualifier == alias.lower():
            return self.qualifier
        if hasattr(self,'alias') and alias.lower() in self.alias:
            return self.qualifier
    
    def get_limits(self):
        """
        Return lower and upper bounds on this variable.
        
        Sets None to those limits that are not available. If none are available,
        this function returns "None,None"
        
        @return: lower limit, upper limit
        @rtype: tuple/(None,None)
        """
        llim = self.llim if hasattr(self,'llim') else None
        ulim = self.ulim if hasattr(self,'ulim') else None
        return llim,ulim
    
    def get_choices(self):
        """
        Return the allowed choices if available.
        
        If not available, this function returns None.
        
        @return: list of choices if available, otherwise None
        @rtype: list/None
        """
        if hasattr(self,'choices'):
            return self.choices
    
    def get_cast_type(self):
        """
        Return the cast type if available.
        
        If not available, this function returns None.
        
        @return: cast type
        @rtype: function/None
        """
        if hasattr(self,'cast_type'):
            return self.cast_type
    
    def get_context(self):
        """
        Return the context.
        
        @return: context
        @rtype: str/None
        """
        if hasattr(self,'context'):
            return self.context
    
    def get_step(self):
        """
        Returns the step size on this variable.
                
        @return: step size of the variable
        @rtype: float/None
        """
        step = self.step if hasattr(self,'step') else None
        return step
    
    def get_prior(self,fitter=None,**kwargs):
        """
        Construct a prior to feed an MCMC sampler.
        
        Only pyMC implemented for the moment.
        
        @param fitter: retrieve the prior in a form suitable to
        pass to a certain fitter
        @type fitter: str, one of ('pymc') or None
        @return: prior information
        @rtype: prior
        """
        if fitter is None:
            return self.prior
        else:
            prior = self.prior.get_distribution(distr_type=fitter,**kwargs)
        return prior
    
    def get_value_from_prior(self,size=1):
        """
        Get a random value from the prior.
        
        @param size: number of values to generate
        @type size: int
        @return: random value from the prior
        @rtype: array[C{size}]
        """
        if not hasattr(self,'prior'):
            raise ValueError("Parameter '{}' (context={}) has no prior".format(self.qualifier,self.get_context()))
        return self.prior.draw(size=size)
    
    def get_value_from_posterior(self,size=1):
        """
        Get a random value from the prior.
        
        @param size: number of values to generate
        @type size: int
        @return: random value from the prior
        @rtype: array[C{size}]
        """
        if not hasattr(self,'posterior'):
            raise ValueError("Parameter '{}' (context={}) has no posterior".format(self.qualifier,self.get_context()))
        return self.posterior.draw(size=size)
        
    
    def get_posterior(self,burn=0,thin=1):
        """
        Return the posterior trace.
        
        @param burn: burn-in number
        @type burn: int
        @param thin: thinning factor
        @type thin: int
        @return: trace
        @rtype: array[n]
        """
        if hasattr(self,'posterior'):
            return np.array(self.posterior)[burn::thin]
    
    def get_unique_label(self):
        """
        Retrieve a unique label for this parameter.
        
        If there is no label yet, it will be created on the fly.
        
        @return: unique string
        @rtype: str
        """
        if not hasattr(self,'_unique_label'):
            self._unique_label = str(uuid.uuid4())
        return self._unique_label
    
    #}
    #{ Set parameter properties
    def set_qualifier(self,new_qualifier):
        """
        Change the name of the parameter.
        
        @param new_qualifier: new name
        @type new_qualifier: str
        """
        self.qualifier = new_qualifier
        
    def set_value(self,value,*args):
        """
        Change a parameter value.
        
        Extra positional arguments are interpreted as the units of the value.
        
        @parameter value: whatever value
        @type value: whatever value
        """
        #clear_memoization(self)
        old_value = self.value
        if args:
            if not hasattr(self,'unit'):
                raise ValueError('Parameter {0} has no units'.format(self.qualifier))
            try:
                value = conversions.convert(args[0],self.unit,self.cast_type(value))
            except:
                #-- if something went wrong, try to give as much information
                #   as possible: first, we try to find out what type of unit
                #   the user has given:
                try:
                    given_type = conversions.get_type(args[0])
                except:
                    given_type = 'not understood'
                #-- in any case, we give a list of allowed units. Possibly, some
                #   of them might not make sense...
                utype,loau = self.list_available_units()
                default_unit = self.get_unit()
                if default_unit in loau:
                    loau.remove(default_unit)
                raise ValueError("Given unit type '{0}' is {1}: {2} must be '{3}' (default) or one of {4} or equivalent. Or perpaps there was not enough information.".format(args[0],given_type,utype,default_unit,loau))
        self.value = value
        #-- check for inside limits
        has_limits = self.has_limits()
        inside_limits = self.is_inside_limits()
        if has_limits and not inside_limits and self._force_inside_limits:
            if self.value<self.llim:
                self.value = self.llim
            elif self.value> self.ulim:
                self.value = self.ulim
            #self.value = old_value
            logger.error('value {0} for {1} is outside of range [{2},{3}]: set to {4}'.format(value,self.qualifier,self.llim,self.ulim,self.value))
        elif has_limits and not inside_limits:
            logger.error('value {0} for {1} is outside of range [{2},{3}] (ignored)'.format(value,self.qualifier,self.llim,self.ulim))
    
    def set_value_from_prior(self):
        """
        Set a random value from the prior.
        """
        try:
            value = self.get_value_from_prior()[0]
        except ValueError:
            return None
        self.set_value(value)
        logger.info("Set value of {} to {}".format(self.get_qualifier(),value))
    
    def set_value_from_posterior(self):
        """
        Change a parameter value to the mean of the posterior distribution.
        
        Only done if the parameter has a posterior, otherwise the call to
        this function is silently ignored.
        """
        if hasattr(self,'posterior'):
            new_value = self.posterior.draw()[0]
            self.set_value(new_value)
            logger.info("Set value from posterior")
        
    
    def set_unit(self,unit):
        """
        Change the unit of a parameter.
        
        The values, lower and upper limits, step sizes and priors are also
        changed accordingly.
        
        @parameter unit: a physical unit
        @type unit: str, interpretable by L{conversions.convert}
        """
        #-- are we lazy? Possibly, somebody just set 'SI' or some other convention
        #   as a unit... in this case we have to derive the convention's version
        #   of the current unit... bastard!:
        if unit in conversions._conventions:
            unit = conversions.change_convention(unit,self.unit)
        logger.info("Converting parameter {} from {} to {}".format(self.qualifier,self.unit,unit))
        #-- the prior
        if hasattr(self,'prior'):
            self.prior.convert(self.unit,unit)
        #-- the value
        new_value = conversions.convert(self.unit,unit,self.get_value())
        if hasattr(self,'llim'):
            new_llim = conversions.convert(self.unit,unit,self.cast_type(self.llim))
            self.llim = new_llim
        if hasattr(self,'ulim'):
            new_ulim = conversions.convert(self.unit,unit,self.cast_type(self.ulim))
            self.ulim = new_ulim
        if hasattr(self,'step'):
            new_step = conversions.convert(self.unit,unit,self.cast_type(self.step))
            self.step = new_step
        self.unit = unit
        self.value = new_value
        
    
    def set_adjust(self,adjust):
        """
        Lock (C{adjust}=False) or release (C{adjust}=True) a parameter value.
        
        @parameter adjust: flag to set/release/change adjust
        @type adjust: boolean
        """
        if hasattr(self,'adjust'):
            self.adjust = adjust
            logger.debug('set_adjust {0} to {1}'.format(self.qualifier,adjust))
        #else:
        #    raise AttributeError,"Parameter '%s' cannot be %s"%(self.qualifier,(adjust and 'released (adjustable)' or 'locked (not adjustable)'))
    
    def set_limits(self,llim=None,ulim=None,force=True):
        """
        Set lower and upper bounds on this variable.
        """
        if hasattr(self,'llim') and llim is not None:
            self.llim = llim
        if hasattr(self,'ulim') and ulim is not None:
            self.ulim = ulim
        self._force_inside_limits = force

    def set_step(self,step):
        """
        Set step size on this variable.
        @parameter step: step size of the variable
        @type step: float
        """
        if hasattr(self,'step'):
            self.step = step
            logger.debug('set_step {0} to {1}'.format(self.qualifier,step))
    
    def set_prior(self,**kwargs):
        """
        Set the distribution of the parameter's prior.
        
        If no previous prior existed, it will be created.
        If a previous prior existed, it's values will be overwritten.
        
        Example for use with pymc:
        
        >>> np.random.seed(100)
        >>> mypar = Parameter(qualifier='bla')
        >>> mypar.set_prior(distribution='uniform',lower=-1,upper=0.)
        >>> prior = mypar.get_prior(distr_type='pymc',name='I am uniform')
        >>> print(prior)
        {'upper': 0.0, 'distribution': 'uniform', 'lower': -1}
        
        Or you can change the prior information later on:
        
        >>> mypar.set_prior(distribution='normal',mu=5,sigma=1.)
        
        """
        #-- do unit conversions if necessary:
        for key in kwargs:
            if isinstance(kwargs[key],tuple):
                value,unit = kwargs[key]
                kwargs[key] = conversions.convert(unit,self.unit,value)
        #-- if only sigma is set for normal priors, take the mu equal to the
        #   current value of the parameter
        if 'distribution' in kwargs and kwargs['distribution']=='normal' and not 'mu' in kwargs:
            kwargs['mu'] = self.get_value()
        if not hasattr(self,'prior') or 'distribution' in kwargs:
            self.prior = Distribution(**kwargs)
        else:
            self.prior.update_distribution_parameters(**kwargs)
            
    
    def set_posterior(self,**kwargs):
        """
        Set the posterior distribution.
        
        """
        if not hasattr(self,'posterior') or 'distribution' in kwargs:
            self.posterior = Distribution(**kwargs)
        else:
            self.posterior.update_distribution_parameters(**kwargs)
    #}
    
    #{ Add/remove parameter properties
    def add_limits(self,llim=None,ulim=None):
        """
        Add lower and upper bounds on this variable.
        """
        self.llim = llim
        self.ulim = ulim
    
    def add_choice(self,choice):
        """
        Add this choice to the list of allowed values.
        
        If this parameter does not support choices, the call will be
        silently ignored.
        
        @param choice: new choice to add
        @type choice: str
        """
        if hasattr(self,'choices'):
            self.choices.append(choice)
    
    def remove_choice(self,choice):
        """
        Remove a choice from the list of allowed values.
        
        If this parameter does not support choices or the choice is not
        present in the list of allowed values, the call will be silently
        ignored.
        
        This function returns the old choice if the choice was successfully
        removed, else it returns None
        
        @param choice: old choice to remove
        @type choice: str
        @return: the removed choice (if succeeded)
        @rtype: str or None
        """
        if hasattr(self,'choices'):
            if choice in self.choices:
                index = self.choices.index(choice)
                return self.choices.pop(index)
    
    def remove_limits(self):
        """
        Remove the limits from the parameter
        """
        if hasattr(self,'llim'): del self.llim
        if hasattr(self,'ulim'): del self.ulim
    
    #}
    #{ Check for parameter properties
    
    def has_unit(self):
        """
        Return True if this parameter has units
        
        @return: C{True} if it has units, otherwise C{False}
        @rtype: bool
        """
        if hasattr(self,'unit'):
            return True
        else:
            return False
    
    def has_step(self):
        """
        Return True if this parameter has a step
        
        @return: C{True} if it has a step, otherwise C{False}
        @rtype: bool
        """
        if hasattr(self,'step'):
            return True
        else:
            return False
        
    
    def has_prior(self):
        """
        Return True if a parameter has a prior.
        
        @return: C{True} if it has a prior, otherwise C{False}
        @rtype: bool
        """
        if hasattr(self,'prior'):
            return True
        else:
            return False
    
    def has_posterior(self):
        """
        Return True if a parameter has a posterior.
        
        @return: C{True} if it has a posterior, otherwise C{False}
        @rtype: bool
        """
        if hasattr(self,'posterior'):
            return True
        else:
            return False
    
    def has_limits(self):
        """
        Return True if a parameter has limits.
        
        @return: C{True} if it has limits
        @rtype: bool
        """
        if hasattr(self,'llim') and hasattr(self,'ulim'):
            return True
        else:
            return False
    
    def is_lim(self):
        """
        Return True if the value equals one of the limits.
        
        @return: C{True} if the value equals the lower or upper limit
        @rtype: bool
        """
        if self.has_limits() and (self.get_value()==self.llim or self.get_value()==self.ulim):
            return True
        else:
            return False
    
    def is_inside_limits(self):
        """
        Return True if value is inside of limits.
        
        @return: C{True} if inside of limits
        @rtype: bool
        """
        if hasattr(self,'llim'):
            value = self.get_value()
            if not (self.llim<=value<=self.ulim):
                return False
        return True
        
    #}
    
    #{ Other convenience functions
    
    def list_available_units(self):
        """
        Return an approximate list of available units.
        
        If default unit is set to 'K', for example, you get:
        
        ('temperature', ['K', 'Far', 'Cel', 'Tsol'])
        
        @return: unit type, list of available units
        @rtype: str,list
        """
        unit_type = conversions.get_type(self.get_unit())
        allowed = []
        for fac in conversions._factors:
            if conversions._factors[fac][2]==unit_type:
                allowed.append(fac)
        return unit_type,allowed
    
    def transform_to_unbounded(self,from_='limits'):
        r"""
        Transform a bounded parameter to an unbounded version.
    
        This can be helpful for inclusion in fitting algorithms that cannot handle
        bounds.
        
        The transformation of a parameter :math:`P` with upper limit :math:`U`
        and :math:`L` to an unbounded parameter :math:`P'` is given by:
        
        .. math::
        
            P' = \left(\frac{\arctan(P)}{\pi} + \frac{1}{2}\right) (U-L) + L
            
            P = \tan\left(\pi\left(\frac{P'-L}{U-L}-\frac{1}{2}\right)\right)
        
        For constraints reasons, we only unbound parameters in SI units.
        
        We also need to transform the prior accordingly.
        
        The upper and lower limits can be chosen from the limits on the
        parameter, or from the prior.
        """
        self.set_unit('SI')
        if from_=='limits':
            L,U = self.get_limits()
        elif from_=='prior':
            L,U = self.get_prior().get_limits()
        else:
            raise ValueError("do not understand {}".format(from_))
        # set limits to be unbounded
        self.set_limits(-np.inf,+np.inf)
        # change the prior
        self.get_prior().transform_to_unbounded(L,U)
        new_value = transform_to_unbounded(self.get_value(),L,U)
        self.set_value(new_value)
        return L,U
    
    def copy(self):
        """
        Return a copy of the instance.
        
        @return: a copy of the instance
        @rtype: Parameter
        """
        return copy.deepcopy(self)    
    
    #}
    
    #{Input/output
    
    def as_string(self):
        """
        Return a string representation of this parameter.
        
        @return: a string representation of the parameter
        @rtype: str
        """
        return self.repr % self.get_value()
    
    def to_dict(self):
        """
        Return a dictionary representation of this parameter.
        
        @return: a dictionary representation of the parameter
        @rtype: dict
        """
        out_dict = {}
        for attrname in dir(self):
            if attrname[:2]=='__': continue
            attrinst = getattr(self,attrname)
            if inspect.ismethod(attrinst): continue
            out_dict[attrname] = attrinst
        return out_dict
        
    #}
    
    #{ Overloaders    
    def __str__(self):
        """
        Return a string representation of the parameter
        
        @return: the string representation
        @rtype: str
        """
        #-- obligatory properties
        value_str = self.repr % self.get_value()
        rawvl_str = str(self.value)
        if '\n' in value_str: value_str = '\n'+value_str
        if '\n' in rawvl_str: rawvl_str = '\n'+rawvl_str
        info  = "Name:           %-s\n" % self.qualifier
        info += "Description:    %-s\n" % self.description
        info += "Value:          %-s\n" % value_str
        info += "Raw value:      %-s\n" % rawvl_str
        info += "Type:           %-s\n" % self.cast_type
        if hasattr(self,'unit'):
            info += 'Unit:           %-s\n' % self.unit
        if hasattr(self,'frame'):
            info += 'Frame:          %-s\n' % ', '.join(self.frame)
        if hasattr(self,'choices'):
            info += 'Allowed values: %-s\n' % ' -- '.join(self.choices)
        if hasattr(self,'llim') and hasattr(self,'ulim') and hasattr(self,'step'):
            info += 'llim/ulim/step:   %-s / %-s / %-s\n'%(self.repr%self.llim,self.repr%self.ulim,self.repr%self.step)
        if hasattr(self,'choices') and self.cast_type=='indexf':
            info += " (%s)\n" % self.repr % self.choices[self.get_value()-1]
        elif hasattr(self,'choices') and self.cast_type=='index':
            info += " (%s)\n" % self.repr % self.choices[self.get_value()]
        info = info.strip()
        return info
        
    def __getitem__(self,item):
        return getattr(self,item)
    
    def __eq__(self,other):
        """
        Test equality of Parameters.
        
        @return: boolean
        @rtype: bool
        """
        #-- figure out if parameter has units
        try:
            unit0 = self.get_unit()
            unit1 = other.get_unit()
        except ValueError:
            unit0 = None
            unit1 = None
            
        #-- if there are units, get the value in SI units
        if unit0 is not None and unit1 is not None:
            try:
                return np.all(self.get_value('SI')==other.get_value('SI'))
            except TypeError: # sometimes there are empty arrays
                return np.all(self.get_value()==other.get_value())
        #-- otherwise retrieve raw values
        else:
            return self.get_value()==other.get_value()
    
    def __hash__(self):
        """
        Overriding __eq__ blocks inheritance of __hash__ in 3.x.
        
        We could need this feature if we want parameters to be keys in a
        dictionary.
        
        So we define __hash__ ourselves.
        """
        return id(self)
    
    #}


class ParameterSet(object):
    """
    Class holding a list of parameters.
    
    B{Section 0. Overview}
    
    **Retrieve Parameter information**
    
    .. autosummary::
    
       get_parameter
       get_adjust
       get_value
       get_unit
       get_value_with_unit
       get_description
       get_context
       get_constraint
       pop_constraint
       request_value
       has_prior
       has_qualifier
       has_unit
       keys
       items
       values
       pop
       
    **Set Parameter information**
    
    .. autosummary::   
    
       load_defaults
       reset
       set_adjust
       set_value
       set_value_from_prior
       set_value_from_posterior
       set_default_units
       set_prior
    
    **Modify the ParameterSet**
    
    .. autosummary::
    
       add
       add_constraint
       remove
       remove_constraint
       run_constraints
       set_convention
       copy
       save
       save_ascii
       to_string

    
    
    B{Section 1. Description}
    
    The parameters can be accessed and changed dictionary-wise via their
    qualifier or one of the aliases. When accessed, they will automatically be
    cast to the right type (e.g. for input in a code).
    
    ParameterSets can be nested. To do this, set the value of a Parameter to be
    a ParameterSet and add that Parameter to the parent ParameterSet.
    
    Initialise a ParameterSet for a binary and print the default model. In the
    next example, we take the first few parameters from the parameter definition
    module, and put them in a ParameterSet:
    
    >>> bps = ParameterSet(definitions=defs.defs[:3],frame='wd',context='root')
    
    Access the values via a qualifier or alias:
    
    >>> print(bps['model'])
    2
    
    Is equal to
    
    >>> print(bps['mode'])
    2
    
    The parameter itself can be accessed via L{get_parameter}. E.g., to print all
    information on a specific parameter to the screen:
    
    >>> print(bps.get_parameter('model'))
    Name:           model
    Description:    Morphological constraints
    Value:          2
    Raw value:      Unconstrained binary system
    Type:           indexf
    Frame:          wd
    Allowed values: X-ray binary -- Unconstrained binary system -- Overcontact binary of the W UMa type -- Detached binary -- Overcontact binary not in thermal contact -- Semi-detached binary, primary star fills Roche lobe -- Semi-detached binary, secondary star fills Roche lobe -- Double contact binary
     (Unconstrained binary system)
     
    Change the model to X-ray binary, and print out both the parameter and the
    entire parameter set.
     
    >>> bps['model'] = 'xra'
    >>> print(bps.get_parameter('model'))
    Name:           model
    Description:    Morphological constraints
    Value:          1
    Raw value:      xra
    Type:           indexf
    Frame:          wd
    Allowed values: X-ray binary -- Unconstrained binary system -- Overcontact binary of the W UMa type -- Detached binary -- Overcontact binary not in thermal contact -- Semi-detached binary, primary star fills Roche lobe -- Semi-detached binary, secondary star fills Roche lobe -- Double contact binary
     (X-ray binary)
    >>> print bps
     name mybinary     --       wd Common name of the binary
    model 1            --       wd Morphological constraints
     hjd0 55124.89703 HJD -     wd Origin of time
    
    Because a ParameterSet is meant to mimic the behaviour of an ordered
    dictionary, you can cycle through it or ask the available keys:
    
    >>> for key in bps:
    ...     print key
    name
    model
    hjd0
    >>> print bps.keys()
    ['name', 'model', 'hjd0']
    
    Now change the name, but use the phoebe qualifier.
    
    >>> bps['phoebe_name'] = 'Still My Binary'
    >>> print bps
     name Still My Binary  --       wd Common name of the binary
    model 1                --       wd Morphological constraints
     hjd0 55124.89703     HJD -     wd Origin of time
    
    
    """
    
    __marker = object() # for pop behaviour (copied from dict)
    
    #{ Initialisation    
    
    def __init__(self,context=None,frame=None,definitions='default',add_constraints=True,**kwargs):
        """
        Initialisation of the WD parameter set class.
        
        It loads value or custom definitions given by C{definitions} into an
        an ordered (iterable) dictionary of WD parameters (WdPar).
        """
        frame = frame if frame is not None else default_frame
        context = context if context is not None else default_context
        #-- this class is iterable
        self.index = 0
        #-- it behaves like an Ordered dictionary
        self.container = OrderedDict()
        #-- remember the frame which is currently set
        self.frame = frame
        #-- useful nomenclature to track the context of nesting.
        self.context = context
        #-- there could be constraints
        self.constraints = OrderedDict()
        
        self.__default_units = None
        
        #-- by default, only load main parameter definitions
        if isinstance(definitions,str) and 'default' in definitions.lower():
            definitions = defs.defs
        
        #-- add the parameter definitions to the class instance
        if definitions is not None:
            #-- we need to deepcopy the list of definitions, otherwise we
            #   could have references to the same default variables lying
            #   around. In particular, if there is a Parameter with an empty
            #   list as a default value, this is **the same list** for **all**
            #   initiated parameterSets. We don't want that: appending a value
            #   to one light curve **MUST NOT* append it to all light curves!
            definitions = copy.deepcopy(definitions)
            #-- select only those from the given frame
            definitions = [idef for idef in definitions if frame in (idef['frame'] if isinstance(idef['frame'],list) else [idef['frame']])]
            #-- subselect only those for the given context
            definitions = [idef for idef in definitions if context in (idef['context'] if isinstance(idef['context'],list) else [idef['context']])]
            for idef in definitions:
                self.add(idef)
            #-- else, see if any parameters are given as kwargs
            #self.container['body1'] = kwargs['body1'].get_parameter('label')
            #-- else, add nothing
            frames,contexts = get_frames_and_contexts(definitions)
            if not self.container and frame not in frames or context not in contexts:
                raise ValueError("frame '{0}' or context '{1}' not predefined".format(frame,context))
        
        #-- maybe already override some of the values
        while kwargs:
            self.__setitem__(*kwargs.popitem())
        #-- add some default constraints upon request
        if add_constraints and frame in defs.constraints:
            if context in defs.constraints[frame]:
                for constr in defs.constraints[frame][context]:
                    self.add_constraint(constr)
        
        #-- give a warning when an empty set was created: it is certainly
        #   possible to start from an empty parameterSet, but probably one
        #   wants to start from an existing set of definitions
        if not self.container:
            logger.warning('Created empty ParameterSet with unknown context {}'.format(context))
        
        
        
    def load_defaults(self,frame,context='root'):
        """
        Load all default parameters for a given frame and context of nesting.
        
        @parameter frame: frame name to load parameters from
        @type frame: str
        @parameter context: nested context name
        @type context: str
        """
        definitions = [idef for idef in defs.defs if frame in idef['frame']]
        definitions = [idef for idef in definitions if idef['context']==context]
        for idef in definitions:
            self.add(idef)
        self.frame = frame
    
    #}
    #{ Accessibility functions to the Parameters
    
    def add(self,parameter):
        """
        Add a parameter to the class instance.
        
        If the parameter qualifier is already in the container, it will be
        silently overwritten.
        
        C{parameter} can either be a Parameter instance, or a dictionary with
        stuff like C{qualifier}, C{value} etc.
        """
        #-- maybe we gave a dictionary with the properties instead of a Parameter.
        #   in that case, convert the dict to a Parameter object first
        if not isinstance(parameter,Parameter):
            parameter = Parameter(**parameter)
        #-- now add it to the container
        if parameter.qualifier in self.container:
            raise KeyError('{0} already exists'.format(parameter.qualifier) )
        self.container[parameter.qualifier] = parameter
        #self.__dict__[parameter.qualifier] = parameter.get_value()
    
    def nestParameterSet(self,**kwargs):
        """
        Add a ParameterSet as parameter to the existing parameterSet.
        
        Kwargs are passed to ParameterSet.__init__()
        """
        new_ps = ParameterSet(**kwargs)
        parameter = Parameter(qualifier=kwargs.pop('context'),value=new_ps)
        self.add(parameter)
        
    
    def remove(self,qualifier,*args):
        """
        Remove a parameter from the class instance.
        """
        self.pop(qualifier,*args)
    
    def reset(self,qualifier):
        """
        Reset a Parameter.
        """
        self.get_parameter(qualifier).reset()
        
    def get_parameter(self,qualifier):
        """
        Return a parameter via its qualifier.
        
        @parameter qualifier: name or alias of the variable
        @type qualifier: str
        @return: Parameter corresponding to the qualifier
        @rtype: Parameter
        """
        #-- use the qualifier
        if qualifier in self.container:
            return self.container[qualifier]
        #--  or check any aliases
        else:
            for qual in self.container:
                if qual == self.container[qual].get_qualifier(qualifier):
                    return self.container[qual]
        #-- perhaps nested?
        if '.' in qualifier:
            qual1,pnt,qual2 = qualifier.partition('.')
            return self[qual1].get_parameter(qual2)
    
    def set_adjust(self,value,*qualifiers):
        """
        Adjust the value of the parameter
        """
        #clear_memoization(self)
        if not qualifiers:
            qualifiers = self.container
        for qualifier in qualifiers:
            qualifier_ = self.alias2qualifier(qualifier)
            if qualifier_ in self.container:
                self.container[qualifier_].set_adjust(value)
            else:
                possibilities = list(self.keys())
                close = difflib.get_close_matches(qualifier,possibilities)
                raise KeyError('Did you mean any of "{}"? "{}" not in ParameterSet ({}, {}): not any of {}'.format('", "'.join(close),qualifier,self.frame,self.context,", ".join(list(self.keys()))) )
    
    def get_adjust(self,*qualifiers):
        """
        Get the adjust value of the parameters.
        """
        adjusts = []
        for qualifier in qualifiers:
            qualifier = self.alias2qualifier(qualifier)
            adjusts.append(self.container[qualifier].get_adjust())
        if len(adjusts)==1:
            return adjusts[0]
        else:
            return adjusts
    
    def has_prior(self,*qualifiers):
        """
        Return True if a parameter has a prior.
        """
        priors = []
        for qualifier in qualifiers:
            qualifier = self.alias2qualifier(qualifier)
            priors.append(self.container[qualifier].has_prior())
        if len(priors)==1:
            return priors[0]
        else:
            return priors
    
    
    #{ Accessibility to the values and units
    
    def set_value(self,qualifier,value,*args):
        """
        Set parameter value, perhaps in different units
        
        The units can be specified as optional arguments.
        
        @parameter qualifier: name or alias of the variable
        @type qualifier: string
        @parameter value: value to set
        @type value: dependent on qualifier
        """
        #clear_memoization(self)
        self.get_parameter(qualifier).set_value(value,*args)
        self.run_constraints()
    
    def set_value_from_posterior(self,qualifier):
        """
        Set the value of parameter from its posterior.
        
        If C{qualifier} is a list of qualifiers and the posterior is a trace
        (i.e. an array), then a value is chosen from that trace, but at the
        same index for all parameters. This guarentees proper correlations
        between all the parameters.
        """
        if isinstance(qualifier,list):
            index = None
            for qual in qualifier:
                param = self.get_parameter(qual)
                if not param.posterior.distribution=='sample':
                    param.set_value_from_posterior()
                    continue
                sample = param.posterior.distr_pars['sample']
                if index is None:
                    index = int(np.random.uniform(high=len(sample)))
                param.set_value(sample[index])
        else:
            self.get_parameter(qualifier).set_value_from_posterior()
        self.run_constraints()
    
    def set_value_from_prior(self,qualifier):
        """
        Set the value of parameter from its prior.
        """
        self.get_parameter(qualifier).set_value_from_prior()
        self.run_constraints()
    
    def set_prior(self,qualifier,**kwargs):
        """
        Set or update prior information.
        """
        self.get_parameter(qualifier).set_prior(qualifier,**kwargs)
    
    def get_value(self,qualifier,*args):
        """
        Return a parameter value, perhaps in different units
        
        Optionally, you can specify the units (str) in which you need the
        value of the parameter returned.
        
        @parameter qualifier: name or alias of the variable
        @type qualifier: string
        @return: value of the Parameter corresponding to the qualifier
        @rtype: anything
        """
        try:
            return self.get_parameter(qualifier).get_value(*args)
        except AttributeError:
            raise AttributeError("ParameterSet '{}' has no keyword '{}'".format(self.context,qualifier))
    
    def get(self,args,default):
        """
        Return a parameter value if it exists, otherwise return default.
        
        @param args: argument tuple passed on to L{get_value}.
        @type args: tuple
        @param default: default return statement
        @type default: anything
        @return: value of a Parameter or default
        @rtype: anything
        """
        try:
            self.get_value(*args)
        except AttributeError:
            return default
    
    def get_unit(self,qualifier):
        """
        Retrieve the unit from a qualifier.
        
        raises ValueError when no unit is available
        
        @parameter qualifier: name or alias of the variable
        @type qualifier: string
        @return: unit of the Parameter corresponding to the qualifier
        @rtype: str
        """
        return self.get_parameter(qualifier).get_unit()
        
    def get_value_with_unit(self,qualifier):
        """
        Get the value and unit of a qualifier
        
        raises ValueError when no unit is available
        """
        value = self.get_value(qualifier)
        unit = self.get_unit(qualifier)
        return value,unit
        
    def get_description(self,qualifier):
        """
        Get the description of a qualifier
        """
        return self.get_parameter(qualifier).get_description()
    
    def get_context(self):
        """
        Get the context
        
        @return: context
        @rtype: str or None
        """
        if hasattr(self,'context'):
            return self.context
    #}
    #{ Accessibility to constraints
        
    def add_constraint(self,constraint,include_as_parameter=False):
        """
        Add explicit constraint to the parameter set.
        
        B{Example:} First we add a new parameter named C{asini}.
        
        >>> ps = ParameterSet(frame='phoebe',context='orbit',add_constraints=False)
        >>> ps.add(Parameter(qualifier='asini',value=12.1,unit='Rsol'))
        
        These are the values before the hook is added:
        
        >>> print ps['sma'],ps['incl'],ps['asini']
        200.0 90.0 12.1
        
        We want to add the contraint that C{sma} is always equal to {asini/sini}.
        
        >>> ps.add_constraint('{sma} = {asini}/sin({incl})')
        
        Now check the values:
        
        >>> print ps['sma'],ps['incl'],ps['asini']
        12.1 90.0 12.1
        
        Changing the value of C{incl} or C{asini} changes C{sma}:
        
        >>> ps['incl'] = 45.
        >>> print ps['sma'],ps['incl'],ps['asini']
        17.1119841047 45.0 12.1
        >>> ps['asini'] = 12.5
        >>> print ps['sma'],ps['incl'],ps['asini']
        17.6776695297 45.0 12.5
        
        B{Warning!} Any numbers contained in C{constraint} must be in SI units!!!
        B{Warning!} Changing C{sma} is now impossible:
        
        >>> ps['sma'] = 10.
        >>> print ps['sma'],ps['incl'],ps['asini']
        17.6776695297 45.0 12.5
        """
        #clear_memoization(self)
        #-- clean up the contraint given by the user
        splitted = constraint.split('=')
        qualifier,expression = splitted[0],'='.join(splitted[1:])
        qualifier = qualifier.split('{')[1].split('}')[0].strip()
        self.constraints[qualifier] = expression
        if include_as_parameter==True:
            self.add(Parameter(qualifier=qualifier,value=0.))
        elif isinstance(include_as_parameter,str):
            self.add(Parameter(qualifier=qualifier,value=0.,unit=include_as_parameter))
        self.run_constraints()
    
    def remove_constraint(self,constraint):
        """
        Remove a constraint.
        """
        return self.constraints.pop(constraint)
    
    def pop_constraint(self,constraint,default=__marker):
        """
        Pop a constraint
        """
        if constraint in self.constraints:
            return self.constraints.pop(constraint,default=default)
        elif default is self.__marker:
            raise KeyError(constraint)
        else:
            return default
    
    def get_constraint(self,qualifier,unit=None):
        """
        Get the value from a constraint.
        
        C{qualifier} doesn't need to be defined in ParameterSet, as long as the
        qualifiers from the right hands side of the constraint definition are!
        
        Return value of qualifier in SI units unless given otherwise.
        """
        #-- don't bother if it's not necessary!
        if not qualifier in self.constraints: raise ValueError('{0} not constrained'.format(qualifier))
        #-- to calculate with the values, we need to convert everything that has
        #   a unit to SI
        #_self = self.copy()
        #_self.set_convention('SI')
        names = [i.split('}')[0] for i in self.constraints[qualifier].split('{') if '}' in i]
        values = {}
        for name in names:
            par = self.get_parameter(name)
            if par is None and name in self:
                values = self[name]
            elif hasattr(par,'unit'):
                values[name] = par.get_value('SI')
            elif par is not None:
                values[name] = par.get_value()
        #-- now evaluate all the constraints, but convert the final values back
        #   to the original unit. We cannot use the "ps['qualifier'] = bla"
        #   method because we call L{run_constraints} in L{_setitem_}, causing
        #   infinite recursion.
        #-- also, the qualifier from the left hand side of the constraint doesn't
        #   need to be defined in the ParameterSet, so if it doesn't exist,
        #   just skip it
        #value = eval(_self.constraints[qualifier].format(**_self))
        if '.' in qualifier: return None
        #self.set_default_units('SI')
        value = eval(self.constraints[qualifier].format(**values))
        if unit:
            value = conversions.convert('SI',unit,value)
        #self.set_default_units(None)
        return value
        
    
    def run_constraints(self):
        """
        Run constraints on the ParameterSet
        """
        #-- don't bother if it's not necessary!
        if not self.constraints: return None
        #-- to calculate with the values, we need to convert everything that has
        #   a unit to SI
        #_self = self.copy()
        #_self.set_convention('SI')
        #-- now evaluate all the constraints, but convert the final values back
        #   to the original unit. We cannot use the "ps['qualifier'] = bla"
        #   method because we call L{run_constraints} in L{_setitem_}, causing
        #   infinite recursion.
        #-- also, the qualifier from the left hand side of the constraint doesn't
        #   need to be defined in the ParameterSet, so if it doesn't exist,
        #   just skip it
        self.set_default_units('SI')
        for qualifier in self.constraints:
            #-- it's possible that the qualifier is not in the ParameterSet,
            #   in this case we have a virtual constraint which is only
            #   accessible through the L{get_constraint} function.
            param = self.get_parameter(qualifier)
            if param is None:
                continue
            try:
                value = eval(self.constraints[qualifier].format(**self))
            except:
                print(qualifier)
                print((self.constraints[qualifier].format(**self)))
                raise
            if hasattr(param,'unit'):
                param.set_value(value,'SI')
            else:
                param.set_value(value)
        self.set_default_units(None)
    
    #}
    #{ Interface to container and constraints
    
    def has_qualifier(self,qualifier):
        """
        Check if a qualifier is in the container or constraints.
        """
        if self.alias2qualifier(qualifier):
            return True
        elif qualifier in self.constraints:
            return True
        return False
    
    def has_unit(self,qualifier):
        """
        Check if a parameter has a unit.
        """
        return self.get_parameter(qualifier).has_unit()
    
    def is_constrained(self,qualifier):
        """
        Check if a parameter is constrained.
        """
        qualifier = self.alias2qualifier(qualifier)
        if qualifier in self.constraints:
            return True
        else:
            return None
    
    def request_value(self,qualifier,*args):
        """
        Request the value of a qualifier, regardless of whether it is a
        constraint or a real parameter.
        
        When requesting a value, you should give
        units wherever possible!.
        """
        try:
            return self.get_value(qualifier,*args)
        except:
            return self.get_constraint(qualifier,*args)
        
    
    
    
    #}
    #{ Changing frames, units...        
        
    def propagate(self,frame):
        """
        Propagate all parameters from the present frame to another.
        
        See corresponding definition in class Parameter.
        
        Deprecated!!!
        """
        #-- first convert existing parameters
        for qualifier in self.container:
            if hasattr(self.container[qualifier],'frame'):
                self.container[qualifier].propagate(frame)
        #-- then add nonexisting parameters (put the defaults)
        for idef in defs.defs:
            if idef['qualifier'] not in self and (frame in idef['frame']) and idef['context']==self.context:
                self.add(idef)
        self.frame = frame
    
    def set_convention(self,convention):
        """
        Set all units to comply to a certain convention (i.e. SI or CGS...)
        
        @parameter convention: name of the convention, interpretatble by
        L{conversions.convert}
        @type convention: str
        """
        for qual in self:
            par = self.get_parameter(qual)
            if hasattr(par,'unit'):
                par.set_unit(convention)
    
    #}
    #{ Arithmic overloaders
    
    def __eq__(self,other):
        """
        CHeck if two parameterSets are equal.
        """
        mybool = True
        for key in self.container:
            if not key in other.container:
                mybool = False
                break
            if not(self.container[key]==other.container[key]):
                mybool = False
                break
        return mybool
    
    def __hash__(self):
        """
        Overriding __eq__ blocks inheritance of __hash__ in 3.x.
        
        We could need this feature if we want parameters to be keys in a
        dictionary.
        
        So we define __hash__ ourselves.
        """
        return id(self)
    
    def __ior__(self,other):
        """
        Union of two ParameterSets.
        """
        for parameter in other:
            if parameter in self: continue
            self.add(other.get_parameter(parameter))
        return self
    
    def __or__(self,other):
        """
        Union of two ParameterSets.
        """
        mergeParSet = self.copy()
        for parameter in other:
            if parameter in self: continue
            mergeParSet.add(other.get_parameter(parameter))
        return mergeParSet
    
    def __iand__(self,other):
        """
        Intersection of two ParameterSets.
        """
        for parameter in other:
            if not (parameter in self) or not (parameter in other):
                self.remove(parameter)
        return self
    
    def __and__(self,other):
        """
        Intersection of two ParameterSets.
        """
        interParSet = self.copy()
        for parameter in other:
            if not (parameter in self) or not (parameter in other):
                interParSet.remove(parameter)
        return interParSet
    
    def __isub__(self,other):
        """
        Difference of two ParameterSets.
        """
        for parameter in other:
            if (parameter in self):
                self.remove(parameter)
        return self
    
    def __sub__(self,other):
        """
        Difference of two ParameterSets.
        
        >>> ps1 = ParameterSet(frame='wd',context='root')
        >>> ps2 = ParameterSet(frame='phoebe',context='orbit')
        >>> diffps = (ps1-ps2)
        """
        diffParSet = self.copy()
        for parameter in other:
            if (parameter in diffParSet):
                diffParSet.remove(parameter)
        return diffParSet
   
        
    #}
    #{ Smart Ordered Dictionary-style behaviour
    
    def alias2qualifier(self,alias):
        """
        Convert an alias to the 'official' qualifier.
        
        Returns None of not existing.
        
        @return: the official qualifier
        @rtype: str
        """
        #-- possibly alias is already an official qualifier
        if alias in self.container:
            return alias
        #-- or it's an alias
        for qualifier in self.container:
            if qualifier==self.container[qualifier].get_qualifier(alias):
                return qualifier
        #-- or it doesn't exist, we might raise an error here.
    
    def keys(self):
        """
        Return all the qualifiers.
        
        @return: list of all the qualifiers
        @rtype: list of strings
        """
        return list(self.container.keys())
    
    def items(self):
        """
        Return tuples (qualifier,value).
        
        @return: list of (qualifier,value)
        @rtype: list of tuples
        """
        return list(zip(list(self.container.keys()),[self.container[key].get_value() for key in self.container]))
    
    def values(self):
        """
        Return values for all the qualifiers
        
        @return: list of all values
        @rtype: list
        """
        return [self.container[key].get_value() for key in self.container]
    
    def pop(self,qualifier,default=__marker):
        """
        Pop a parameter from the class instance
        
        @return: the parameter
        @rtype: Parameter
        """
        qualifier = self.alias2qualifier(qualifier)
        parameter = self.get_parameter(qualifier)
        
        #-- remove it from the container
        if qualifier in self.container:
            return self.container.pop(parameter.qualifier)
        elif default is self.__marker:
            raise KeyError(qualifier)
        else:
            return default
    
    def set_default_units(self,units=None):
        self.__default_units = units
        for qualifier in self:
            if hasattr(self[qualifier],'set_default_units'):
                self[qualifier].set_default_units(units)
    
    def __getitem__(self, qualifier):
        """
        Return a parameter via its qualifier
        
        @rtype: anything
        """
        if '.' in qualifier:
            qual1,pnt,qual2 = qualifier.partition('.')
            return self[qual1][qual2]
        #-- just remember the incoming value to be able to complain later
        qualifier_in_ = qualifier
        if qualifier=='__root__':
            return self
        if not qualifier in self.container:
            qualifier = self.alias2qualifier(qualifier)
        #-- method 1
        if not qualifier in self.container and qualifier_in_ in self.constraints:
            return self.get_constraint(qualifier_in_)
        elif not qualifier in self.container:
            raise KeyError("parameter '{}' not available in context '{}')".format(qualifier_in_,self.context))
            #possibilities = list(self.keys())
            #print possibilities
            #close = difflib.get_close_matches(qualifier,possibilities)
            #suggestions = ("Did you mean "+' or '.join(close)+'? ') if close else ''
            #raise KeyError('"{}" not in ParameterSet ({}, {}). {}It is not any of {}'.format(qualifier_in_,self.frame,self.context,suggestions,", ".join(list(self.keys()))) )
        #-- method 2
        #if not qualifier in self.container:
        #    raise KeyError,'%s'%(qualifier_in_)
        if self.__default_units and hasattr(self.container[qualifier],'unit'):
            return self.container[qualifier].get_value(self.__default_units)
        else:
            return self.container[qualifier].get_value()
    
    def __setitem__(self, qualifier, value):
        """
        Set the value of a parameter via its qualifier
        """
        if not qualifier in self:
            possibilities = list(self.keys())
            close = difflib.get_close_matches(qualifier,possibilities)
            suggestions = ("Did you mean "+' or '.join(close)+'? ') if close else ''
            raise KeyError('"{}" not in ParameterSet ({}, {}). {}It is not any of {}'.format(qualifier,self.frame,self.context,suggestions,", ".join(list(self.keys()))) )
        qualifier = self.alias2qualifier(qualifier)
        if isinstance(value,Parameter):
            self.container[qualifier] = value
        elif isinstance(value,tuple) and len(value)==2 and isinstance(value[1],str):
            self.container[qualifier].set_value(*value)
        else:
            self.container[qualifier].set_value(value)\
        
        self.run_constraints()
        #-- update the value in the dot-styled access
        #self.__dict__[qualifier] = self[qualifier]
        
    def __iter__(self):
        """
        Make the class iterable
        """
        return self
    
    def next(self):
        """
        Return the next parameter qualifier in the class when iterating.
        """
        if self.index>=len(self.container):
            self.index = 0
            raise StopIteration
        else:
            self.index += 1
            #return self.container[self.container.keys()[self.index-1]]
            return list(self.container.keys())[self.index-1]
    
    def __contains__(self,qualifier):
        """
        Check if a parameter is in the class instance.
        """
        #-- qualifier available?
        if qualifier in self.container:
            return True
        #-- is it an alias?
        elif self.alias2qualifier(qualifier) in self.container:
            return True
    
    def __call__(self,*lookups):
        retvalue = self
        for key in lookups:
            retvalue = retvalue[key]
        return retvalue
    #}
    #{ Other convenience functions
   
    def copy(self):
        """
        Return a copy of the instance.
        
        @rtype: ParameterSet
        """
        return copy.deepcopy(self)
    
    
    def save(self,filename):
        """
        Save to pickle.
        
        >>> bps = ParameterSet()
        >>> bps.save('mytest.par')
        >>> bps2 = load('mytest.par')
        >>> str(bps)==str(bps2)
        True
        >>> os.unlink('mytest.par')
        
        """
        ff = open(filename,'w')
        pickle.dump(self,ff)
        ff.close()        
    
    def save_ascii(self,fileobj,label=None,mode='w'):
        """
        Save a parameterSet to an ascii file.
        """
        sep = '_/_'
        if label is None and 'label' in self:
            label = self.get_value('label')
        elif label is None:
            label = uuid.uuid4()
        label = '{0}--{1}--{2}'.format(self.frame,self.context,label)
        
        if isinstance(fileobj,str):
            outfile = open(fileobj,mode)
        else:
            outfile = fileobj
        #-- write main header
        outfile.write('[{0}]\n'.format(label))
        #-- first write out all things that are not ParameterSets
        for param in self.container:
            value = self.get_value(param)
            try:
                unit = self.get_unit(param)
            except ValueError:
                unit = None
            if not isinstance(value,ParameterSet):
                comment = ''
                comment+= (unit is not None) and '({}) '.format(unit) or ' '
                comment+= self.get_description(param)
                outfile.write("{0:10s} = {1:10s} # {2}\n".format(str(param),str(value),comment))
        
        #-- then write out all the constraints
        outfile.write('\n')
        for param in self.constraints:
            if not isinstance(self.request_value(param),ParameterSet):
                outfile.write("{0} ~ {1}\n".format(param,self.constraints[param]))
        
        #-- then write out all the ParameterSets
        outfile.write('\n')
        for param in self.container:
            value = self.get_value(param)
            if isinstance(value,ParameterSet):
                value.save_ascii(outfile,label=label+sep+param+sep)
    
    def to_dict(self):
        """
        Convert parameterSet to dictionary
        """
        out_dict = {}
        out_dict['container'] = {}
        for param in self.container:
            out_dict['container'][param] = self.container[param].to_dict()
        for param in self.constraints:
            out_dict['constraints'][param] = self.constraints[param]
        for attrname in dir(self):
            if attrname[:2]=='__': continue
            if attrname in ['container','constraints']: continue
            attrinst = getattr(self,attrname)
            if inspect.ismethod(attrinst): continue
            out_dict[attrname] = attrinst
        return out_dict
        
            
    
    def __str__(self):
        """
        String representation of the class instance
        
        For root parameters, this will output:
        
        parameter_name casted_value  units  frame description
        
        For light curve parameters, the lines will be indented
        
        @rtype: str
        """
        return self.to_string()
    
    def to_string(self,only_adjustable=False):
        """
        String representation of the class instance with extra options.
        
        @rtype: str
        """
        #-- first derive what the maximum lengths is for all the values. For the
        #   width of the value column, we split also over the '\n' characters. We
        #   also check if there are constraints give, they will be appended in
        #   the back
        def shortstr(par):
            gvalue = par#.get_value()
            if hasattr(gvalue,'__iter__') and not isinstance(gvalue,str) \
                and not isinstance(gvalue,ParameterSet) and not isinstance(gvalue,dict):
                    if (hasattr(gvalue,'shape') and not gvalue.shape) or len(gvalue)<=2: # for unsized arrays
                        gvalue = str(gvalue)
                    else:
                        gvalue = '[{} ... {}]'.format(gvalue[0],gvalue[-1])
            elif isinstance(gvalue,dict):
                gvalue = '{'+":,".join(list(gvalue.keys()))+':}'
            else:
                gvalue = str(gvalue)
            return gvalue
        
        try:
            col_width_value = max([len(mystr) for par in self for mystr in shortstr(self.container[par].get_value()).split('\n') if (not only_adjustable or self.container[par].get_adjust())] +\
                                [len(str(self.get_constraint(cnt))) for cnt in self.constraints if not only_adjustable])
        except ValueError:
            return "<empty ParameterSet>"
        col_width_qualf = max([len(str(self.container[par].qualifier)) for par in self] +\
                              [len(str(cnt)) for cnt in self.constraints])
        col_width_unit = max([hasattr(self.container[par],'unit') and len(str(self.container[par].unit)) or 2 for par in self]+[3])
        col_width_frame = max([len(self.frame),len('constr')])
        #-- build the string representation, cycling over all parameters
        mystr = []
        for par in self:
            par = self.container[par]
            qualifier = par.qualifier
            unit = hasattr(par,'unit') and par.unit or '--'
            adjust = hasattr(par,'adjust') and (par.adjust and 'x' or '-') or ' '
            if only_adjustable and not par.get_adjust():
                continue
            frame = self.frame
            description = hasattr(par,'description') and par.description or '--'            
            value = '{0:<{1}}'.format(shortstr(par.get_value()),col_width_value)
            str_qual = '{0:>{1}}'.format(qualifier,col_width_qualf)
            str_unit = '{0:>{1}}'.format(unit,col_width_unit)
            str_frame = '{0:>{1}}'.format(frame,col_width_frame)
            if '\n' in value:
                value = ('\n'+(col_width_qualf+1)*' ').join(['{0:<{1}}'.format(line,col_width_value) for line in value.split('\n')])
            mystr.append(" ".join([str_qual,value,str_unit,adjust,str_frame,str(description)]))    
        #--- add the constraints
        for constraint in self.constraints:
            if only_adjustable: continue
            str_qual = '{0:>{1}}'.format(constraint,col_width_qualf)
            str_valu = '{0:<{1}}'.format(self.get_constraint(constraint),col_width_value)
            str_frame = '{0:>{1}}'.format('constr',col_width_frame)
            str_unit = '{0:>{1}}'.format('n/a',col_width_unit)
            mystr.append(' '.join([str_qual,str_valu,str_unit,' ',str_frame,str(self.constraints[constraint]).strip()]))
        return '\n'.join(mystr)
    
    def __config__(self):
        """
        @rtype: str
        """
        nestings = []
        mystr = '[%s]\n'%(self.context)
        for key in self:
            par = self.container[key]
            if isinstance(par.get_value(),ParameterSet):
                nestings.append('%s'%(par.get_value().__config__()))
            else:
                mystr += "%s = %s\n"%(par.qualifier,par.get_value())
        mystr += '\n'.join(nestings)
        return mystr
                    
        
    #}        


class Distribution(object):
    """
    Class representing a distribution.
    
    Attempts at providing a uniform interface to different codes or routines.
    
    Recognised distributions:
    
    >>> d = Distribution('normal',mu=0,sigma=1)
    >>> d = Distribution('uniform',lower=0,upper=1)
    >>> d = Distribution('histogram',bins=array1,prob=array2,discrete=True)
    >>> d = Distribution('histogram',bins=array1,prob=array2,discrete=False)
    >>> d = Distribution('sample',sample=array3,discrete=False)
    >>> d = Distribution('discrete',values=array4)
    
    If C{distribution='histogram'} and C{discrete=False}, the bins are interpreted
    as the edges of the bins, and so there is one more bin than probabilities.
    
    If C{distribution='histogram'} and C{discrete=True}, the bins are
    interpreted as the set of discrete values, and so there are equally many
    bins as probabilities. It is possible to use ``valuess`` as an alias for
    bins here, then internally that parameter will be converted to ``bins``. 
    
    """
    def __init__(self,distribution,**distribution_parameters):
        self.distribution = distribution.lower()
        self.distr_pars = distribution_parameters
        
        #-- discrete can be equivalent to histogram
        if distribution=='discrete':
            distribution = 'histogram'
            distribution_parameters['discrete'] = True
            distribution_parameters['prob'] = np.ones(len(distribution_parameters['values']))
            if 'values' in distribution_parameters:
                distribution_parameters['bins'] = distribution_parameters.pop('values')
        
        #-- check contents:
        given_keys = set(list(distribution_parameters.keys()))
        
        if   distribution=='histogram':
            required_keys = set(['bins','prob','discrete'])
        elif distribution=='uniform':
            required_keys = set(['lower','upper'])
        elif distribution=='normal':
            required_keys = set(['mu','sigma'])
        elif distribution=='sample':
            required_keys = set(['sample','discrete'])
        
        if (given_keys - required_keys) or (required_keys - given_keys):
            raise ValueError('Distribution {} needs keys {}'.format(distribution,", ".join(list(required_keys))))
        
        #-- make sure the histogram is properly normalised
        if distribution=='histogram':
            self.distr_pars['prob'] = self.distr_pars['prob']/float(np.sum(self.distr_pars['prob']))
    
    def update_distribution_parameters(self,**distribution_parameters):
        """
        Update the distribution parameters.
        
        When a sample is updated, the values get appended to the previous ones
        In all other cases, the keys are overwritten.
        """
        for key in distribution_parameters:
            if not key in self.distr_pars:
                raise ValueError('Distribution {} does not accept key {}'.format(self.distribution,key))
            elif self.distribution=='sample' and key=='sample':
                self.distr_pars[key] = np.hstack([self.distr_pars[key],distribution_parameters[key]])
            else:
                self.distr_pars[key] = distribution_parameters[key]
                
    
    def get_distribution(self,distr_type=None,**kwargs):
        """
        Return the distribution in a specific form.
        
        When distr_type=='pymc', you need to supply the keyword C{name} in the
        kwargs.
        """
        #-- plain
        if distr_type is None:
            return self.distribution,self.distr_pars
        #-- pymc
        elif distr_type=='pymc':
            if self.distribution=='normal':
                kwargs['mu'] = self.distr_pars['mu']
                kwargs['tau'] = 1./self.distr_pars['sigma']**2
            else:
                for key in self.distr_pars:
                    kwargs.setdefault(key,self.distr_pars[key])
            return getattr(pymc,self.distribution.title())(**kwargs)
        else:
            L,U = self.get_limits()
            if self.distribution=='normal':
                x = kwargs.get('x',np.linspace(L,U,1000))
                return x,getattr(distributions.norm(loc=self.distr_pars['mu'],scale=self.distr_pars['sigma']),distr_type)(x)
            elif self.distribution=='uniform':
                x = kwargs.get('x',np.linspace(L,U,1000))
                return x,getattr(distributions.uniform(loc=self.distr_pars['lower'],scale=self.distr_pars['upper']),distr_type)(x)
            elif self.distribution=='histogram':
                bins = self.distr_pars['bins']
                bins = bins[:-1] + np.diff(bins)
                return bins,self.distr_pars['prob']
            elif self.distribution=='sample' or self.distribution=='histogram':
                return self.distr_pars['bins'],self.distr_pars['prob']
            
    def convert(self,from_,to_,*args):
        """
        Convert a distribution to different units.
        
        We cannot `simply` convert a distribution, because the conversion can
        be nonlinear (e.g. from linear to log scale). Therefore, we first get
        the PDF, and convert that one. Unfortunately, this also means that
        you can only convert continuous distributions. Extension to discrete
        ones should be fairly straightforward though (just an
        ``if==distr_pars['discrete']``) thingy. But I don't feel like it
        today.
        """
        logger.info('Converting {} distr. in {} to histogram distr. in {}'.format(self.distribution,from_,to_))
        old_x,old_y = self.pdf()
        old_dx = np.diff(old_x)
        x_shifted = np.hstack([old_x[:-1] - np.diff(old_x)/2.,old_x[-2:] + np.diff(old_x)[-1]/2.])
        #-- we definitely cannot go over the limits!
        x_shifted[0] = old_x[0]
        x_shifted[-1] = old_x[-1]
        old_dx = np.diff(x_shifted)
        
        if from_=='bounded':
            #old_x[old_x<args[0]] = args[0]
            #old_x[old_x>args[1]] = args[1]
            #x_shifted[x_shifted<args[0]] = args[0]
            #x_shifted[x_shifted>args[1]] = args[1]
            new_x = transform_to_unbounded(old_x,*args)
            new_x_shifted = transform_to_unbounded(x_shifted,*args)
        else:
            new_x = conversions.convert(from_,to_,old_x)
            new_x_shifted = conversions.convert(from_,to_,x_shifted)
        new_dx = np.diff(new_x_shifted)
        new_y = old_dx/new_dx*old_y
        norm = np.trapz(new_y,x=new_x)
        new_y = new_y/norm
        self.distribution = 'histogram'
        self.distr_pars = dict(discrete=False,bins=new_x_shifted,prob=new_y)
    
    
    def pdf(self,**kwargs):
        """
        Return the probability density function.
        """
        return self.get_distribution(distr_type='pdf',**kwargs)
    
    def cdf(self,**kwargs):
        """
        Return the cumulative density function.
        """
        return self.get_distribution(distr_type='cdf',**kwargs)
    
    def get_limits(self):
        if self.distribution=='uniform':
            lower = self.distr_pars['lower']
            upper = self.distr_pars['upper']
        elif self.distribution=='normal':
            lower = self.distr_pars['mu']-3*self.distr_pars['sigma']
            upper = self.distr_pars['mu']+3*self.distr_pars['sigma']
        elif self.distribution=='sample' or self.distribution=='histogram':
            lower = self.distr_pars['bins'].min()
            upper = self.distr_pars['bins'].max()
        else:
            raise NotImplementedError
        return lower,upper
    
    def draw(self,size=1):
        """
        Draw a (set of) random value(s) from the distribution.
        
        @param size: number of values to generate
        @type size: int
        @return: random value from the distribution
        @rtype: array[C{size}]
        """
        if self.distribution=='uniform':
            values = np.random.uniform(size=size,low=self.distr_pars['lower'],
                                 high=self.distr_pars['upper'])
        elif self.distribution=='normal':
            values = np.random.normal(size=size,loc=self.distr_pars['mu'],
                                 scale=self.distr_pars['sigma'])
        elif self.distribution=='histogram' or self.distribution=='sample':
            #-- when the Distribution is actually a sample, we need to make
            #   the histogram first ourselves.
            if self.distribution=='sample':
                myhist = np.histogram(self.distr_pars['sample'],normed=True)
                if self.distr_pars['discrete']:
                    bins = np.unique(self.distr_pars['sample'])
                else:
                    bins = myhist[1]
                prob = myhist[0]
            #-- else they are readily available
            else:
                bins,prob = self.distr_pars['bins'],self.distr_pars['prob']
            #-- draw random uniform values from the cumulative distribution
            cumul = np.cumsum(prob*np.diff(bins))#*np.diff(bins)
            cumul /= cumul.max()
            locs = np.random.uniform(size=size)
            
            #plt.figure()
            #plt.plot(bins[:-1],cumul,'ko-')
            #plt.show()
            
            #-- little different for discrete or continuous distributions
            if self.distr_pars['discrete']:
                values = bins[indices-1]
            else:
                values = np.interp(locs,cumul,bins[1:])

        else:
            raise NotImplementedError
            
        return values
    
    def transform_to_unbounded(self,low,high):
        """
        Transform the distribution to be bounded.
        
        - Uniform distribution set bounds between -100 and +100
        - Normal distribution set mu=0.0, sigma=2.0
        - Sample distribution transformed to unbounded
        - Histogram distribution transformed to unbounded
        """
        #-- problems with conversions to SI
        #sample = self.draw(size=1000)
        #transf_sample = transform_to_unbounded(sample,low,high)
        #self.distribution = 'sample'
        #self.distr_pars = dict(sample=transf_sample,discrete=False)
        
        if self.distribution=='uniform':
            self.distr_pars['lower'] = -5
            self.distr_pars['upper'] = +5
        elif self.distribution=='normal':
            self.distr_pars['mu'] = 0.
            self.distr_pars['sigma'] = 2.
        elif self.distribution=='sample' or self.distribution=='histogram':
            self.convert('bounded','unbounded',low,high)
            #self.distr_pars['bins'] = transform_to_unbounded(self.distr_pars['bins'],low-1e-3*low,high+1e-3*high)
        else:
            raise NotImplementedError
            
        
        
    def __str__(self):
        """
        String representation of class Distribution.
        """
        name = self.distribution.title()
        pars = ", ".join(['{}={}'.format(key,self.distr_pars[key]) for key in sorted(list(self.distr_pars.keys()))])
        return "{}({})".format(name,pars)
    
    

#}

#{ Input/output

def transform_to_unbounded(par,low,high):
    return np.tan(np.pi*((par-low)/(high-low)-0.5))

def transform_to_bounded(par,low,high):
    return (np.arctan(par)/np.pi+0.5)*(high-low)+low

def load(filename):
    """
    Load and assign a saved ParameterSet.
    
    Construct a BPS:
    
    >>> bps = ParameterSet()
    
    Save to a file
    
    >>> bps.save('mytest.par')
    
    Load 
    
    >>> bps2 = load('mytest.par')
    >>> print str(bps)==str(bps2)
    True
    >>> os.unlink('mytest.par') # clean up
    """
    ff = open(filename,'r')
    inst = pickle.load(ff)
    ff.close()
    return inst
        

def save_ascii(filename,*args):
    """
    Save a bunch of ParameterSets to one single file.
    """
    for i,parset in enumerate(args):
        parset.save_ascii(filename,mode=i and 'a' or 'w')
    
def load_ascii(filename):
    """
    Load and assign saved ParameterSets in ascii format.
    
    @param filename: name of the file containing one or more ParameterSets in
    ASCII format
    @type filename: str
    @return: list of ParameterSets
    @rtype: list of ParameterSets
    """
    parsets = OrderedDict()
    failed = ''
    with open(filename,'r') as infile:
        logger.info("Loading contents from file {}".format(filename))
        for line in infile.readlines():
            line = line.strip()
            if not line: continue # empty line
            if line[0]=='#': continue # comment line
            if line[0]=='[': # header line
                prev_label = None
                #-- ParameterSets can be nested: this information is encoded
                #   with the "_/_" sign, similar (but not equal) to directory
                #   naming in Unix.
                header = line[1:-1].split('_/_')
                curparset = parsets
                
                #-- if the ParameterSet is not nested, we can append it to
                #   the main container.
                if len(header)==1:
                    frame,context,label = header[0].split('--')
                    logger.info('Loading set: frame={}, context={} (label={})'.format(frame,context,label))
                    parsets[label] = ParameterSet(frame=frame,context=context)
                    continue
                
                for i,head in enumerate(header):
                    #-- the header is encoded as "frame--context--label". We
                    #   need to access this information
                    head = head.split('--')
                    if not prev_label is None:
                        frame,context,label = head
                        #-- Nested ParameterSets: first create a new
                        #   parameterSet with this frame and context, and then
                        #   make a new Parameter with the label as qualifier
                        #   and this new ParameterSet as value
                        new_ps = ParameterSet(frame=frame,context=context)
                        logger.info('Loading nested set: frame={}, context={} (label={})'.format(frame,context,label))
                        if not prev_label in curparset:
                            curparset.add(Parameter(qualifier=prev_label,value=new_ps))
                        else:
                            curparset[prev_label] = new_ps
                        label = prev_label
                        prev_label = None
                        continue
                    
                    if len(head)==1:
                        prev_label, = head
                        continue
                    
                    frame,context,label = head
                    if not label in curparset:
                        curparset[label] = ParameterSet(frame=frame,context=context)
                    else:
                        curparset = curparset[label]
            #-- in the "Body" of the ASCII file, there are real Parameters and
            #   there are constraints. They are discriminated with the '=' and
            #   '~' sign.
            else:
                #-- real parameter
                if '=' in line:
                    linesplit = line.split('=')
                    #-- there could be a '=' in the description
                    if len(linesplit)>2:
                        linesplit = linesplit[0],'='.join(linesplit[1:])
                    #-- there could be a '=' after a # but not before
                    if '#' in linesplit[0]:
                        logger.warning("Skipping line '{}'".format(line))
                        continue
                    name,value = linesplit
                    name = name.strip()
                    #-- in parameterSet
                    if name in curparset[label]:
                        curparset[label][name] = value.split('#')[0].strip()
                    #-- else create a new one
                    else:
                        logger.warning("Cannot load parameter '{}'".format(name))
                        #if '#' in value:
                            #description = value.split('#')[1].strip()
                        #else:
                            #description = 'No description available'
                        #value = value.split('#')[0].strip()
                        #custom_par = Parameter(qualifier=name,description=description,unit='mas',value=value)
                        #curparset[label].add(custom_par)
                #-- constraint
                elif ' ~ ' in line:
                    name,value = line.split('~')
                    curparset[label].add_constraint('{{{}}} = {}'.format(name.strip(),value.split('#')[0].strip()))
                elif line:
                    #-- we don't want to repeat the same statement over and over again:
                    if name!=failed:
                        logger.warning("Parameter '{}' probably not correctly loaded".format(name))
                    failed = name
                    
    return list(parsets.values())
    
#}    

#{ Casters
    
def return_self(x):
    """
    Dummy function.
    
    This is needed for pickling a parameter cast type which does nothing. Purely
    from a functionality point of view, one could choose C{lambda x:x} as a cast
    type, but a lambda function cannot be pickled.
    """
    return x

def return_string_or_list(x):
    """
    Return a list if the value can be converted to a list, otherwise return a
    string.
    """
    try:
        ret_value = x.strip()
        if ret_value[0]=='[' and ret_value[-1]==']':
            ret_value = eval(ret_value)
        else:
            ret_value = x
    except:
        ret_value = x
    return ret_value
            
def return_list_of_strings(x):
    """
    Return a list of strings if x is a string, or otherwise assume it is a list.
    """
    if isinstance(x,str):
        x = x.strip()
        x = x[1:-1]
        x = x.replace("'",'')
        x = x.replace('"','')
        return x.split(', ')
    else:
        return x

def make_bool(value):
    """
    Make something into a boolean, but also return string 'True' as True and
    'False' as False.
    """
    if isinstance(value,str):
        if value.strip().lower()=='false':
            return False
    #-- everything else is always true (all nonempty strings are true)
    return bool(value)
    
def make_upper(value):
    return str(value).upper()
    

def match_string(choice,possible_values):
    """
    Matches a string stripped from all nonalphanumeric characters and converted
    to lower case.
    
    A TypeError will be raised when the match fails.
    
    Examples:
    
    >>> choices = ['X-ray binary','Unconstrained binary system','Overcontact binary of the W UMa type','Detached binary','Overcontact binary not in thermal contact','Semi-detached binary, primary star fills Roche lobe','Semi-detached binary, secondary star fills Roche lobe','Double contact binary']
    >>> print(match_string('double',choices))
    7
    >>> print(match_string('xra',choices))
    0
    >>> print(match_string('^^U--nc*o&n',choices))
    1
    >>> print(match_string('weird system',choices))
    None
    >>> try:
    ...     print(match_string('Overcontact',choices))
    ... except ValueError:
    ...     print("fail")
    fail
    
    @parameter choice: the value that needs to be tested for membership in C{possible_values}.
    @type choice: str
    @parameter possible_values: the string/strings with wich to compare C{choice} to
    @type possible_values: str or list of str
    @rtype: int
    @return: index of C{choice} in C{possible_values} (or None when not available)
    """
    choice = pattern.sub('',choice)
    matches = []
    if isinstance(possible_values,str):
        possible_values = [possible_values]
    for i,value in enumerate(possible_values):
        value = pattern.sub('',value)
        if choice.lower()==value[:len(choice)].lower():
            matches.append(i)
    if len(matches)==1:
        return matches[0]
    elif len(matches)>1:
        raise ValueError('ambiguous identification of parameter %s'%(choice))


#}


if __name__=="__main__":
    import doctest
    fails,tests = doctest.testmod()
    if not fails:
        print(("All {0} tests succeeded".format(tests)))
    else:
        print(("{0}/{1} tests failed".format(fails,tests)))