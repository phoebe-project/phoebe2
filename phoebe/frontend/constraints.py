"""

"""

import logging
from phoebe.parameters.parameters import ParameterSet, Parameter
from phoebe.units import constants

import re
import types
from collections import OrderedDict

logger = logging.getLogger("FRONT.CONSTRAINTS")

try:
    import sympy
except ImportError:
    _use_sympy = False
else:
    _use_sympy = True



def asini(b, orbit):
    """
    [FUTURE]
    
    add constraint for asini to the orbit.  If the parameter {asini} does not exist, it will be created.
    
    Required parameters:
    - sma@[orbit]
    - incl@[orbit]
    
    Example use:
    >>> b = phoebe.Bundle()
    >>> b.add_constraint(phoebe.constraints.asini(b, 'my_orbit_label'), label='constraint_label')

    Or slightly more simply:
    >>> b = phoebe.Bundle()
    >>> b.add_constraint(phoebe.constraints.asini, 'my_orbit_label', label='constraint_label')
        
    If you know or want to fit asini (ie from radial velocities) and constraint a:
    >>> b = phoebe.Bundle()
    >>> b.add_constraint(phoebe.constraints.asini, 'my_orbit_label', solve_for='sma', label='constraint_label')
    
    or you can always change solve_for after creating the constraint
    >>> b['solve_for@constraint_label'] = 'incl'
        
    @param orbit: label of the orbit
    @type orbit: str
    @return: the constraint expression to be passed to bundle.add_constraint
    @rtype: str
    """
    ti = b._get_by_search(kind='Parameter', label=orbit, ignore_errors=True, return_trunk_item=True, qualifier='asini')
    if ti is None:
        logger.warning("asini parameter not found, adding to system")
        b.add_parameter("asini@orbit@{}".format(orbit), unit='Rsol', value=0., description='Projected semi major axis', cast_type=float)
        ti = b._get_by_search(kind='Parameter', label=orbit, ignore_errors=True, return_trunk_item=True, qualifier='asini')
    asini = "{%s}" % b.least_unique_twig(ti['twig'])
    
    ti = b._get_by_search(kind='Parameter', label=orbit, ignore_errors=True, return_trunk_item=True, qualifier='sma')
    a = "{%s}" % b.least_unique_twig(ti['twig'])
    
    ti = b._get_by_search(kind='Parameter', label=orbit, ignore_errors=True, return_trunk_item=True, qualifier='incl')
    i = "{%s}" % b.least_unique_twig(ti['twig'])
    
    
    return "%s = %s * sin(%s)" % (asini, a, i)

def mass_from_parent_orbit(b, component):
    """
    [FUTURE]
    
    Compute the mass of a BinaryRocheStar from its parent orbit (period, sma, q) using Kepler's third law
    
    Example use:
    >>> b = phoebe.Bundle()
    >>> b.add_constraint(phoebe.constraints.mass_from_parent_orbit(b, 'my_component_label'), label='constraint_label')

    Or slightly more simply:
    >>> b = phoebe.Bundle()
    >>> b.add_constraint(phoebe.constraints.asini, 'my_component_label', label='constraint_label')
    
    @param component: label of component
    @type component: str
    @return: the constraint expression to be passed to bundle.add_constraint
    @rtype: str
    """
    orbit = b.get_parent(component).get_label()

    # TODO require component to be a BRS? - or if not then allow deriving (and therefore creating) whatever param is missing

    ti = b._get_by_search(kind='Parameter', label=component, ignore_errors=True, return_trunk_item=True, qualifier='mass')
    if ti is None:
        logger.warning("mass parameter not found, adding to system")
        b.add_parameter("mass@component@{}".format(component), unit='Msol', value=0., description='Mass of component', cast_type=float)
        ti = b._get_by_search(kind='Parameter', label=component, ignore_errors=True, return_trunk_item=True, qualifier='mass')
    mass = "{%s}" % b.least_unique_twig(ti['twig'])
    
    ti = b._get_by_search(kind='Parameter', label=orbit, ignore_errors=True, return_trunk_item=True, qualifier='sma')
    a = "{%s}" % b.least_unique_twig(ti['twig'])
    
    ti = b._get_by_search(kind='Parameter', label=orbit, ignore_errors=True, return_trunk_item=True, qualifier='period')
    period = "{%s}" % b.least_unique_twig(ti['twig'])
    
    ti = b._get_by_search(kind='Parameter', label=orbit, ignore_errors=True, return_trunk_item=True, qualifier='q')
    if b.get_object(component).get_component()==0:
        # then primary component so we want q
        q_or_inv_q = "{%s}" % b.least_unique_twig(ti['twig'])
    else:
        q_or_inv_q = "1.0/{%s}" % b.least_unique_twig(ti['twig'])
    
    return "%s = 4*{constants.pi}**2 * %s**3 / %s**2 / {constants.GG} / (1.0 + %s)" % (mass, a, period, q_or_inv_q)
    
def keplers_third_law_hierarchical(b, outerorbit, innerorbit):
    """
    [FUTURE]
    
    Example use:
    >>> b = phoebe.Bundle()
    >>> b.add_constraint(phoebe.constraints.keplers_third_law_hierarchical(b, 'inner_orbit_label', 'outer_orbit_label'), label='constraint_label')

    Or slightly more simply:
    >>> b = phoebe.Bundle()
    >>> b.add_constraint(phoebe.constraints.keplers_third_law_hierarchical, 'inner_orbit_label', 'outer_orbit_label' label='constraint_label')
    
    @param outerorbit: label of the outerorbit in the triple hierarchy
    @type outerorbit: str
    @param innerorbit: label of the innerorbit in the triple hierarchy
    @type innerorbit: str
    @return: the constraint expression to be passed to bundle.add_constraint
    @rtype: str
    """
    
    # TODO: check to make sure innerorbit is a child of outerorbit?
    
    ti = b._get_by_search(kind='Parameter', label=outerorbit, ignore_errors=True, return_trunk_item=True, qualifier='sma')
    a_outer = "{%s}" % b.least_unique_twig(ti['twig'])
    
    ti = b._get_by_search(kind='Parameter', label=innerorbit, ignore_errors=True, return_trunk_item=True, qualifier='sma')
    a_inner = "{%s}" % b.least_unique_twig(ti['twig'])
    
    ti = b._get_by_search(kind='Parameter', label=outerorbit, ignore_errors=True, return_trunk_item=True, qualifier='q')
    if b.get_object(innerorbit).get_component()==0:
        q_or_inv_q_outer = "{%s}" % b.least_unique_twig(ti['twig'])
    else:
        q_or_inv_q_outer = "1.0/{%s}" % b.least_unique_twig(ti['twig'])
    
    ti = b._get_by_search(kind='Parameter', label=outerorbit, ignore_errors=True, return_trunk_item=True, qualifier='period')
    period_outer = "{%s}" % b.least_unique_twig(ti['twig'])
    
    ti = b._get_by_search(kind='Parameter', label=innerorbit, ignore_errors=True, return_trunk_item=True, qualifier='period')
    period_inner = "{%s}" % b.least_unique_twig(ti['twig'])

    return '%s**3 = %s**3 * (1+%s) * %s**2/%s**2' % (a_outer, a_inner, q_or_inv_q_outer, period_outer, period_inner)

    
    
def logg(b, component):
    """
    [FUTURE]
    
    Example use:
    >>> b = phoebe.Bundle()
    >>> b.add_constraint(phoebe.constraints.logg(b, 'my_component_label'), label='constraint_label')

    Or slightly more simply:
    >>> b = phoebe.Bundle()
    >>> b.add_constraint(phoebe.constraints.logg, 'my_component_label', label='constraint_label')
    
    @return: the constraint expression to be passed to bundle.add_constraint
    @rtype: str
    """
    ti = b._get_by_search(kind='Parameter', label=component, ignore_errors=True, return_trunk_item=True, qualifier='logg')
    if ti is None:
        #~ b.add_parameter('logg@component@{}'.format(component) )
        raise KeyError("logg parameter does not exist, create it before adding constraint")
    else:
        logg = "{%s}" % b.least_unique_twig(ti['twig'])
    
    ti = b._get_by_search(kind='Parameter', label=component, ignore_errors=True, return_trunk_item=True, qualifier='mass')
    if ti is None:
        # TODO: need to include derivation of mass (from parent orbit) - can actually call mass_from_parent_orbit constraint
        raise NotImplementedError("logg needs mass parameter")
    else:
        m = "{%s}" % b.least_unique_twig(ti['twig'])
        
    ti = b._get_by_search(kind='Parameter', label=component, ignore_errors=True, return_trunk_item=True, qualifier='radius')
    if ti is None:
        # TODO: need to include derivation of radius
        raise NotImplementedError("logg needs radius parameter")
    else:
        r = "{%s}" % b.least_unique_twig(ti['twig'])
    
    return "%s = log10({constants.GG} * %s / (%s**2))" % (logg, m, r)



class Var(object):
    """
    Internal object to hold variables (parameters and constants) that are used
    in constraint equations.  There should be no need for a user to create or
    interact with these objects at all.
    
    Please see :py:func:`Bundle.add_constraint`
    """
    def __init__(self, b, label):
        """
        @param b: the bundle
        @type b: Bundle
        @param label: unique label or twig pointing to the parameter
        @type label: str
        """
        self._user_label = label

        if b.get_system().find_parameter_by_unique_label(label):
            self._unique_label = label
        else:
            param = b._get_by_search(kind='Parameter', twig=label, ignore_errors=True)
            if param is not None:
                self._unique_label = param.get_unique_label()

        if self.is_param:
            self._safe_label = self._unique_label.replace('@', '_').replace('-', '_').replace('+', '_')
            self.update_user_label(b)  # call _set_curly_label()
        else:
            # handle _safe label for constants
            self._safe_label = self._user_label.replace('.', '_')
            self._value = None
            self._set_curly_label()


            
    def _set_curly_label(self):
        """
        sets curly label based on user label
        """
        self._curly_label = '{'+self.user_label+'}'
            
    def update_user_label(self, b):
        """
        finds this parameter and gets the least_unique_twig from the bundle
        
        in order for
        """
        tw = b._get_by_search(unique_label=self.unique_label, kind='Parameter', return_key='twig')
        self._user_label = b.least_unique_twig(tw)
        self._set_curly_label()
            
    @property
    def is_param(self):
        """
        is this variable a parameter of the system (vs a constant)
        """
        return hasattr(self, '_unique_label')
        
    @property
    def is_constant(self):
        """
        is this variable a constant (vs a parameter of the system)
        """
        return hasattr(self, '_value')
    
    def get_parameter(self, system=None):
        """
        get the parameter object from the system for this var
        
        needs to be backend safe (not passing or storing bundle)
        """
        if system:
            param = system.find_parameter_by_unique_label(self.unique_label)
            self._param = param
        return self._param
        
    def get_value(self, system, units=None):
        """
        get the value (either of the constant or from the parameter) for this var
        
        needs to be backend safe (not passing or storing bundle)
        """
        if self.is_constant:
            return self._value

        param = self.get_parameter(system)
        if units is not None and hasattr(param, 'unit'):
            return self.get_parameter(system).get_value(units)
        return self.get_parameter(system).get_value()
        
    @property
    def unique_label(self):
        """
        unique_label corresponding to parameter.get_unique_label
        call get_parameter(system) to retrieve the parameter itself
        
        
        needs to be backend safe (not passing or storing bundle)
        """
        return self._unique_label
        
    @property
    def user_label(self):
        """
        label as the user provided it
        
        needs to be backend safe (not passing or storing bundle)
        """
        return self._user_label

    @property
    def curly_label(self):
        """
        label with {} brackets - used for the user's view of the equation
        
        needs to be backend safe (not passing or storing bundle)
        """
        return self._curly_label

    @property
    def safe_label(self):
        """
        label that is safe to be passed through sympy - with escapes from any mathetmatical symbols
        
        needs to be backend safe (not passing or storing bundle)
        """
        return self._safe_label
        
        
class Constraint(ParameterSet):
    """
    This class builds and runs cross-parameterset constraints using the frontend,
    but in a way such that the backend can run them without the presence of the bundle (ie during fitting).
    
    Although these can be used independently, generally the user will interact with
    constraints through :py:func:`Bundle.add_constraint`, :py:func:`Bundle.get_constraint`, and :py:func:`Bundle.remove_constraint`
    """
    def __init__(self, b, *args, **kwargs):
        """
        @param b: the bundle
        @type b: Bundle
        @param expr: mathematical expression for the constraint, using unique labels for parameter values (SI only) or <<time>>
        @type expr: str
        @param solve_for: which variable (parameter, not <<time>>) in expr should be derived (and therefore read-only)
        @type solve_for: str
        @param label: label of the constraint PS
        @type label: str
        @param run: whether to run the constraint on creation (default: True)
        @type run: bool
        @param time: time to pass to the constraint (if run==True and applicable)
        @type time: float
        """
        super(Constraint, self).__init__(context='constraint')

        # TODO: make this JSON-serializable ready

        solve_for = kwargs.get('solve_for', None)
        label = kwargs.get('label', 'constraint{:02d}'.format(len(b.sections['constraint'])+1))
        run = kwargs.get('run', True)
        time = kwargs.get('time', None)

        # TODO: change from using expr to taking kwargs (but might cause problems with @'s in the strings)... how about *args and split on the =?
      
        system = b.get_system()
        
        self._vars = []

        # we have to do this in the beginning so label is available for any warnings
        self.set_value('label', label)
        
        eqs = {}
        for i,expr in enumerate(args):
            lhs, rhs = self._parse_expr(b, expr)

            if _use_sympy:
                if i > 0:
                    raise ValueError("only pass 1 expression if using sympy")
                else:
                        
                    for var in self._params:
                        for v in self._vars:
                            sympy.var(v.safe_label)

                        eq_user = self._get_eq(lhs, rhs)
                        eq_safe = self._eq_user_to_safe(eq_user)
                        try:
                        #~ if True:
                            eq = sympy.solve(eq_safe, var.safe_label)[0]
                        except:
                            logger.error("could not solve {} for {}".format(eq_user, var.user_label))
                        else:
                            eqs[var.user_label] = self._eq_safe_to_user(str(eq))
                       
                        #~ self.get_parameter(var.user_label)._sympy = eq
            else:
                var = self._get_var(lhs.replace('{', '').replace('}', '').strip())
                eqs[var.user_label] = rhs
                
        
        for var in self._params:
            self.add(Parameter(qualifier='eq:{}'.format(var.user_label), 
                                description='equation to use when deriving {}'.format(var.user_label), 
                                cast_type=str, value=eqs.get(var.user_label, '')))

        
        self.get_parameter('solve_for').choices = [var.user_label for var in self._params]
        self.set_value('solve_for', solve_for if solve_for is not None else self._params[0].user_label)
        
        # we need to hook into the solve_for and label parameter's set_value  
        def _solve_for_post_set_value(param, solve_for, *args):
            # after solve_for.set_value() is called, this function will run
            # retrieve this constraint object (self) and loop through self._params
            self = param.get_parameterset()
            for v in self._params:
                p = v.get_parameter() # NOTE: we don't have system here so are depending on it being set - this might cause problems with loading from ascii
                #~ print "setting {} _is_constraint_input={}, _is_constraint_output={}".format(v.user_label, v.user_label!=solve_for, v.user_label==solve_for)
                p._is_constraint_input = v.user_label != solve_for
                p._is_constraint_output = v.user_label == solve_for
                
                if v.user_label == solve_for and p.get_adjust():
                    # constraint outputs can't be set to adjust
                    p.set_adjust(False)
                    logger.warning("constraint outputs can't be adjustable, setting adjust to False for {}".format(v.user_label))
        
        def _label_post_set_value(param, label, *args):
            self = param.get_parameterset()
            for v in self._params:
                p = v.get_parameter() # NOTE: we don't have system here so are depending on it being set - this might cause problems with loading from ascii
                p._constraint_label = label

        # we need to attach this function as a method type of this instance (see comment in parameters.py:__init__() for more examples            
        solve_for_param = self.get_parameter('solve_for')
        solve_for_param._set_value_post = [types.MethodType(_solve_for_post_set_value, solve_for_param)]

        label_param = self.get_parameter('label')
        label_param._set_value_post = [types.MethodType(_label_post_set_value, label_param)]
        
        # we have to do this again so that we set all parameters._constraint_label
        self.set_value('label', label)
        
        if run:
            self.run(system, time=time)
            
    def __repr__(self):
        solve_for = self.get_value('solve_for')
        return "<Constraint: {}({})>".format(solve_for, ", ".join([var.user_label for var in self._params if var.user_label!=solve_for]))
        
    @property
    def label(self):
        return self.get_value('label')
        
    @property
    def _params(self):
        """
        retrieve all vars that are parameters of the system (vs constants)
        """
        return [v for v in self._vars if v.is_param]

    def _check_and_add_var(self, var, lbl, system):
        """
        """
        
        # check to make sure parameter was found and is adjustable (float or int)
        if var.is_param and hasattr(var.get_parameter(system), 'adjust'):
            # make parameter read-only
            param = var.get_parameter(system)
            param._constraint_label = self.get_value('label')
            self._vars.append(var)
        else:
            if lbl.split('.')[0]=='constants' and hasattr(constants, lbl.split('.')[1]):
                # then we'll set the value now and retrieve it from the val when running
                var._value = getattr(constants, lbl.split('.')[1])
                self._vars.append(var)
            # we need to see if this could still be interpreted by sympy
            elif not len(lbl) or lbl in dir(sympy.functions) or lbl in ['<<time>>']:
                # then this should be able to be handled by sympy
                pass
            else:
                try:
                    float(lbl)
                    # then this is just a number
                except:
                    raise KeyError("could not parse {} in expression to a valid parameter".format(lbl))
                    return

    def _parse_expr(self, b, expr):
        """
        """
        system = b.get_system()
        

                
        # we need to get all the unique labels
        lbls = re.findall(r'\{.[^{}]*\}', expr)
        
        for lbl in lbls:
            # try to find parameter
            lbl = lbl.replace('{', '').replace('}', '')
            var = Var(b, lbl)
            self._check_and_add_var(var, lbl, system)
            if var.is_param:
                # we have change the labels to be least_unique_twigs,
                # we need to do the same in the expression
                expr = expr.replace('{'+lbl+'}', var.curly_label)
                   
        lhs, rhs = expr.split('=')
                        
        return lhs, rhs
        
    def _eq_safe_to_user(self, eq):
        """
        convert sympy-safe equation to user version
        """
        # loop through from longest to shortest string so we don't overwrite substrings
        for var in reversed(sorted(self._vars, key=lambda v: len(v.safe_label))):
            #~ print "_eq_safe_to_user: ", var.safe_label, var.curly_label
            eq = eq.replace(var.safe_label, var.curly_label)
        return eq
        
    def _eq_user_to_safe(self, eq):
        """
        convert user-provided equation to sympy-safe version
        """
        # loop through from longest to shortest string so we don't overwrite substrings
        for var in reversed(sorted(self._vars, key=lambda v: len(v.curly_label))):
            #~ print "_eq_user_to_safe: ", var.curly_label, var.safe_label
            eq = eq.replace(var.curly_label, var.safe_label)
        eq = eq.replace('{', '').replace('}', '')
        return eq

    def _get_eq(self, lhs, rhs):
        """
        format the equation to be equal to 0 (ie lhs - rhs = 0) to prepare
        to solve for any parameter by sympy
        """
        eq = '({})-({})'.format(lhs, rhs)
        return eq
            
    def _get_var(self, label):
        """
        retrieve a var by user_label, safe_label, or unique_label
        """
        for var in self._vars:
            if label in [var.user_label, var.safe_label, var.unique_label]:
                return var
        return None
        
    def _get_param_value(self, system, time, label):
        """
        get the value given a user_label, safe_label, or unique_label
        """
        if label == '<<time>>':
            return time
        else:
            var = self._get_var(label)
            return system.find_parameter_by_unique_label(var.unique_label).get_value()

    
    @property
    def expr(self):
        """
        retrieve the current expression (user equation for the solve_for variable)
        """
        return self.get_value('eq:{}'.format(self.get_value('solve_for')))
    
    def run(self, system, time=None, set_value=True):
        """
        run the constraint
        
        @param system: the system
        @type system: Body or BodyBag
        @param time: time to pass to the constraint (if run==True and applicable)
        @type time: float
        @param set_value: whether to update the value of the parameter (default=True)
        @type set_value: bool
        @return: value of the parameter (in SI)
        @rtype: float
        """
        
        # get values by searching by unique id
        
        solve_for = self.get_value('solve_for')

        param = self._get_var(solve_for).get_parameter(system)
        
        eq = self.expr
        
        if not len(eq):
            logger.error("equation not provided to solve '{}' constraint for '{}'".format(self.get_value('label'), solve_for))
            return None
        
        if _use_sympy:
            values = {var.safe_label: var.get_value(system, 'SI') for var in self._vars}
            # just to be safe, let's reinitialize the sympy vars
            for v in self._vars:
                #~ print "creating sympy var: ", v.safe_label
                sympy.var(v.safe_label)

            eq = sympy.N(self._eq_user_to_safe(eq))
            value = eq.subs(values).evalf()
        else:
            values = {var.user_label: var.get_value(system, 'SI') for var in self._vars}
            value = float(eval(eq.format(**values)))
            
        
        if set_value:
            if hasattr(param,'unit'):
                param.set_value(value,'SI')
            else:
                param.set_value(value)
            # now send a logger message, here we'll use param.get_value() so we show with default units
            logger.info("setting value of {} to {} because of '{}' constraint".format(self._get_var(solve_for).user_label, param.get_value(), self.label))

        return value
    
    
# TODO: write tutorial (tag as experimental for now)

# TODO: handle when a label or unique_label in the system changes - need to loop over all vars, expressions, solve_for.choices, and parameters in all constraints, update, and re-set to new values

# TODO: allow solve_for (at least during initialization) to match with least_unique_twig (ie can provide full twig instead of needing to provide exact match to user_label)

# TODO: if the user changes one of the eqs - need to check to make sure no new vars need to be created
# TODO: forbid constraints to depend on read-only values (perhaps from another constraint), and forbid parameters in a new constraint that are already read-only - or fix so that order can be guessed or system of eqs is solved
# TODO: enable and test time dependency (will need to hook into set_time, which means also passed through run_compute)
# TODO: allow enabling/disabling constraints?


# TODO: make sure this is json-serializable-loadable-ready (can we load and rebuild this, including vars) given the bundle and the string version of the PS?)
