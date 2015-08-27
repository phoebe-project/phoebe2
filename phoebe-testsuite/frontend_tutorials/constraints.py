"""
Making use of constraints in PHOEBE 2.0 [EXPERIMENTAL]
===========================================================

Last updated: ***time***

This tutorial will cover the basic steps of utilizing constraints to 
derive parameters not used in the model, re-parameterizing the model,
and constraining existing parameters.

These "constraints" should not be confused with priors - they should not
constrain the limits of a particular parameter for fitting, but rather 
should constrain a parameter to be some function of other parameters.
"""

"""

Example with existing parameters
------------------------------------

Let's start with the default binary
"""

import phoebe

logger = phoebe.get_basic_logger()
b = phoebe.Bundle()
print b.list

"""
Output::

    new_system (BodyBag)
    |
    +----------> primary (BinaryRocheStar)
    |
    +----------> secondary (BinaryRocheStar)

"""

"""
by default this binary consists of two components ('primary' and 'secondary'),
each with independent temperatures ('teff@primary' and 'teff@secondary').
"""

print b.get_value('teff@primary'), b.get_value('teff@secondary')

"""
Output::
    
    6000.0 6000.0

"""


"""
If you were fitting a model to data and knew something about these values,
you'd create a prior.  But let's just say that for some reason you always
wanted a model in which the primary is twice the temperature of the secondary.

Let's constrain teff@primary to always be 2*teff@secondary.  We can do that
as simply as passing an expression with twigs surrounded by curly braces to
add_constraint.
"""

b.add_constraint('{teff@primary} = 2*{teff@secondary}')


"""
By doing this, teff@primary will become a read-only parameter - you will no longer be able
to change its value directly, but it will automatically be updated when you change teff@secondary.

You'll see that the constraint is already in place.
"""

print b.get_value('teff@primary'), b.get_value('teff@secondary')

"""
Output::

    12000.0 6000.0
    
"""

"""
Now let's change the value of teff@secondary
"""

b['teff@secondary'] = 5000

"""
And see that the value for teff@primary has changed for us.
"""

print b.get_value('teff@primary'), b.get_value('teff@secondary')

"""
Output::

    10000.0 5000.0
    
"""


"""
If you try to set the value of teff@primary, an error will be raised.

If you'd rather be able to set the value of teff@primary and have teff@secondary
derived... well you should have set the expression to be {teff@secondary} = 0.5*{teff@primary}.
But it's not too late, you can easily choose to solve for any of the parameters in the constraint (so long
as you have the python module sympy available on your system - if you don't, you'll need to provide
the expression for any parameter you wish to solve for).


Let's look at the ParameterSet that add_constraint added to the bundle.
As with anything else, these are easily found through the twig dictionary-access
"""

print b['constraint']

"""
Output::

    {'constraint01': <Constraint: teff@primary(teff@secondary)>}

"""

"""
Here we'll see that the constraint was automatically given the label 'constraint01'.
We could have given it a more memorable label like 'tratio' by passing label to the
add_constraint call.

Let's look at that ParameterSet
"""

print b['constraint01@constraint']

"""
Output::

                label constraint01        --   label                               
            solve_for teff@primary        --   which parameter should be derived   
                                               and become read-only                
      eq:teff@primary 2*{teff@secondary}  --   equation to use when deriving       
                                               teff@primary                        
    eq:teff@secondary {teff@primary}/2    --   equation to use when deriving       
                                               teff@secondary   

"""

"""
The ParameterSet itself is quite simple, it only consists of the label,
the expressions to solve for each parameter (again, if you have sympy), and
solve_for (which parameter is derived from all the others).

If you don't have sympy, you don't need to start over, you just need to provide
the corresponding expression first (note that you don't provide the {teff@secondary} = ):
"""

b['eq:teff@secondary@constraint01'] = '0.5*{teff@primary}'

"""
Now, let's change this constraint to solve for the secondary temperature instead.
"""

b['solve_for@constraint01'] = 'teff@secondary'

"""
Now we're able to change teff@primary manually and see that teff@secondary changes
"""

b['teff@primary'] = 11000

print b.get_value('teff@primary'), b.get_value('teff@secondary')

"""
Output::

    11000.0 5500.0

"""

"""
Those are the basics of constraints.  But let's say that instead of fixing
the temperature ratio to be exactly 2 always, we want to let that be a parameter
as well.  To do that, we need to create a new parameter, let's call it teffratio,
and create a constraint between teffratio, teff@primary, and teff@secondary.

To do that, let's start from scratch with a new bundle.

"""

"""

Example with new parameters
----------------------------------

"""

b = phoebe.Bundle()

"""
Next, we'll create the teffratio parameter and attach it to the orbit@new_system ParameterSet
(since it doesn't really belong in either component).

We just need to provide the new twig (which will be the new parameter name followed
by the twig required to reach the ParameterSet we want to place it in) and a few other
options.
"""

b.add_parameter('teffratio@orbit@new_system', value=0.0, description='temperature ratio teff@primary/teff@secondary')

"""
So far we have a new parameter, but it doesn't do anything.  It isn't connected
to the temperatures and therefore has absolutely no influence on our model.
"""

print b.get_value('teffratio'), b.get_value('teff@primary'), b.get_value('teff@secondary')

"""
Output::

    0.0 6000.0 6000.0

"""

"""
So let's create a constraint
"""

b.add_constraint('{teffratio} = {teff@primary}/{teff@secondary}', label='tratio')

"""
We can now see that our new parameter is derived by the individual temperatures
"""

print b.get_value('teffratio'), b.get_value('teff@primary'), b.get_value('teff@secondary')

"""
Output::

    1.0 6000.0 6000.0

"""

"""
and updates immediately when we change either temperature
"""

b['teff@primary'] = 6500

print b.get_value('teffratio'), b.get_value('teff@primary'), b.get_value('teff@secondary')

"""
Output::

    1.08333333333 6500.0 6000.0

"""

b['teff@secondary'] = 6500

print b.get_value('teffratio'), b.get_value('teff@primary'), b.get_value('teff@secondary')

"""
Output::

    1.0 6500.0 6500.0

"""

"""
but, again, since teffratio is being derived, we'll get an error if we try
to change its value.

Let's say we want to set teffratio and teff@primary and have teff@secondary 
computed for us.

Once again, we just need to change the solve_for parameter (first providing the expression
if you don't have sympy available).
"""

b['solve_for@tratio@constraint'] = 'teff@secondary'

print b.get_value('teffratio'), b.get_value('teff@primary'), b.get_value('teff@secondary')

"""
Output::

    1.0 6500.0 6500.0

"""

"""
Example with built-in constraints
------------------------

Some constraints and expressions can get rather complex - and why go through
the hassle when we can handle all of that for you.

Let's say you want to derive the mass for each component in your system.
You could create a new parameter called mass and setup a constraint that uses
Kepler's third law with the period, semi-major axis, and mass-ratio of the
parent orbit of that component.  Or you could just use the built-in constraint.

"""

b = phoebe.Bundle()
b.add_constraint(phoebe.constraints.mass_from_parent_orbit(b, 'primary'))

"""
or in a slightly-simpler syntax:
"""

b.add_constraint(phoebe.constraints.mass_from_parent_orbit, 'secondary')

print b['constraint01']

"""
Output::

        label constraint01                     --   label                          
    solve_for mass                             --   which parameter should be      
                                                    derived and become read-only   
      eq:mass 4.0*{sma}**3*{constants.pi}**2/  --   equation to use when deriving  
              (1.0*{q}*{period}**2*{constants       mass                           
              .GG} +                                                               
              1.0*{period}**2*{constants.GG})                                      
       eq:sma 1.0*(0.25*{q}*{period}**2*{cons  --   equation to use when deriving  
              tants.GG}*{mass}/{constants.pi}       sma                            
              **2 + 0.25*{period}**2*{constan                                      
              ts.GG}*{mass}/{constants.pi}**2                                      
              )**(1/3)                                                             
    eq:period 2.0*({sma}**3*{constants.pi}**2  --   equation to use when deriving  
              /({constants.GG}*{mass}*({q} +        period                         
              1)))**(1/2)                                                          
         eq:q -1.0*(1.0*{period}**2*{constant  --   equation to use when deriving q
              s.GG}*{mass} - 4.0*{sma}**3*{co                                      
              nstants.pi}**2)/({period}**2*{c                                      
              onstants.GG}*{mass})                                   

"""

print b['constraint02']

"""
Output::

                label constraint02                 --   label                      
            solve_for mass@secondary               --   which parameter should be  
                                                        derived and become read-   
                                                        only                       
    eq:mass@secondary 4.0*{q}*{sma}**3*{constants  --   equation to use when       
                      .pi}**2/(1.0*{q}*{period}**       deriving mass@secondary    
                      2*{constants.GG} + 1.0*{per                                  
                      iod}**2*{constants.GG})                                      
               eq:sma -0.5*(0.25*{period}**2*{con  --   equation to use when       
                      stants.GG}*{mass@secondary}       deriving sma               
                      /{constants.pi}**2 + 0.25*{                                  
                      period}**2*{constants.GG}*{                                  
                      mass@secondary}/({q}*{const                                  
                      ants.pi}**2))**(1/3) - 0.5*                                  
                      3**(1/2)*I*(0.25*{period}**                                  
                      2*{constants.GG}*{mass@seco                                  
                      ndary}/{constants.pi}**2 +                                   
                      0.25*{period}**2*{constants                                  
                      .GG}*{mass@secondary}/({q}*                                  
                      {constants.pi}**2))**(1/3)                                   
            eq:period 2.0*({q}*{sma}**3*{constant  --   equation to use when       
                      s.pi}**2/({constants.GG}*{m       deriving period            
                      ass@secondary}*({q} +                                        
                      1)))**(1/2)                                                  
                 eq:q -1.0*{period}**2*{constants  --   equation to use when       
                      .GG}*{mass@secondary}/(1.0*       deriving q                 
                      {period}**2*{constants.GG}*                                  
                      {mass@secondary} - 4.0*{sma                                  
                      }**3*{constants.pi}**2)  

"""

print b.get_value('mass@primary'), b.get_value('mass@secondary')

"""
Output::

    6.70782467318 6.70782467318
    
"""

"""
look at the :py:mod:`phoebe.frontend.constraints` module to see the available built-in 
constraints.  All constraints take the bundle (if using the first syntax) and
labels of the necessary objects as inputs.  Please see the documentation
for each function for descriptions of the necessary inputs and examples.
"""

"""

Dealing with constants
-------------------------------

For any constants built into phoebe, you can pass those as well with curly braces
and they will be computed at run-time.

As an example, let's create a constraint to compute luminosity using Stefan-Boltzmann law.
We'll do this for both components in the system, so we need to create two new parameters,
and then a constraint for each of them.
"""

b = phoebe.Bundle()
b.add_parameter('lum@component@primary', value=0, unit='J/s', description='luminosity of the component from Stefan-Boltzmann')
b.add_parameter('lum@component@secondary', value=0, unit='J/s', description='luminosity of the component from Stefan-Boltzmann')

"""
COMING SOON - needs to be 4*pi*R**2... but I dont' have R, yet.  So for now this is just a fake example to show constants.
"""

b.add_constraint('{lum@primary} = {constants.sigma} * {teff@primary}**4')

"""
this is a good point to note that all equations, regardless of default-units
are computed using SI units.  If creating your own expression, it is important
that you write it to use SI units (or radians) for all quantities.

For any constants that aren't in :py:mod:`phoebe.units.constants`, you should insert the
actual value into the equation.  Even though sigma is available, for the
second constraint we'll hardcode it in.
"""

b.add_constraint('{lum@secondary} = %f * {teff@secondary}**4' % (phoebe.constants.sigma))

"""
Note that even though the "{}".format(value) syntax may be preferred, this
will cause problems with the curly-brace syntax for expressions, so you'll
need to format the string some other way.

Now let's compare the difference between these two methods:
"""

print b['constraint01']

"""
Output::

    COMING SOON
    
"""

print b['constraint02']

"""
Output::

    COMING SOON

"""

"""
And lastly, see that the values are filled
"""

print b.get_value('lum@primary'), b.get_value('lum@secondary')

"""
Output::

    COMING SOON

"""


"""

Utilizing built-in functions
-------------------------------

COMING SOON (explain which functions are available and how to use them)

"""

"""
Sympy vs manually providing expressions
----------------------------------------

COMING SOON (need to explain the difference, options to manually provide
all expressions, try_sympy, providing multiple expressions to add_compute,
and how to deal with failed sympy solutions)

"""

"""

Time-dependent constraints
-----------------------------

COMING SOON (maybe)


"""


