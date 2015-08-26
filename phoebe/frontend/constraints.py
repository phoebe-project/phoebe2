"""

"""

import logging
from phoebe.parameters.parameters import ParameterSet, Parameter
from phoebe.units import constants

import re
from collections import OrderedDict

logger = logging.getLogger("FRONT.CONSTRAINTS")

try:
    import sympy
except ImportError:
    _use_sympy = False
else:
    _use_sympy = True

class Var(object):
    def __init__(self, b, label):
        """
        @param b: the bundle
        @type b: Bundle
        @param label: unique label or twig pointing to the parameter
        @type label: str
        """
        self._user_label = label
        self._curly_label = '{'+label+'}'
        
        if b.get_system().find_parameter_by_unique_label(label):
            self._unique_label = label
        else:
            param = b._get_by_search(kind='Parameter', twig=label, ignore_errors=True)
            if param is not None:
                self._unique_label = param.get_unique_label()

        if self.is_param:
            self._safe_label = self._unique_label.replace('@', '_').replace('-', '_').replace('+', '_')
        else:
            # handle _safe label for constants
            self._safe_label = self._user_label.replace('.', '_')
            self._value = None
            
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
    
    def get_parameter(self, system):
        """
        get the parameter object from the system for this var
        
        needs to be backend safe (not passing or storing bundle)
        """
        return system.find_parameter_by_unique_label(self.unique_label)
        
        
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
        label = kwargs.get('label', None)
        run = kwargs.get('run', True)
        time = kwargs.get('time', None)

        # TODO: change from using expr to taking kwargs (but might cause problems with @'s in the strings)... how about *args and split on the =?

        if label is not None:
            self.set_value('label', label)
        
        system = b.get_system()
        
        eqs = {}
        self._vars = []
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

        
        # TODO: handle hooks into set_value
        # TODO: handle updating twigs or unique labels (update self._vars, self.expr, self.eq)
        
        if run:
            self.run(system, time=time)
            
    def __repr__(self):
        solve_for = self.get_value('solve_for')
        return "<Constraint: {}({})>".format(solve_for, ", ".join([var.user_label for var in self._params if var.user_label!=solve_for]))
         
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
        
        lhs, rhs = expr.split('=')
                
        # we need to get all the unique labels
        lbls = re.findall(r'\{.[^{}]*\}', expr)
        
        for lbl in lbls:
            # try to find parameter
            lbl = lbl.replace('{', '').replace('}', '')
            var = Var(b, lbl)
            self._check_and_add_var(var, lbl, system)

                        
        return lhs, rhs
        
    def _eq_safe_to_user(self, eq):
        """
        convert sympy-safe equation to user version
        """
        # loop through from longest to shortest string so we don't overwrite substrings
        for var in reversed(sorted(self._vars, key=lambda v: len(v.safe_label))):
            eq = eq.replace(var.safe_label, var.curly_label)
        return eq
        
    def _eq_user_to_safe(self, eq):
        """
        convert user-provided equation to sympy-safe version
        """
        # loop through from longest to shortest string so we don't overwrite substrings
        for var in reversed(sorted(self._vars, key=lambda v: len(v.user_label))):
            eq = eq.replace(var.user_label, var.safe_label)
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
            logger.error("equation not provided to solve constraint '{}' for '{}'".format(self.get_value('label'), solve_for))
            return None
        
        if _use_sympy:
            values = {var.safe_label: var.get_value(system, 'SI') for var in self._vars}
            # just to be safe, let's reinitialize the sympy vars
            for v in self._vars:
                sympy.var(v.safe_label)
            eq = sympy.N(self._eq_user_to_safe(eq))
            value = eq.subs(values).evalf()
        else:
            values = {var.user_label: var.get_value(system, 'SI') for var in self._vars}
            value = eval(eq.format(**values))
        
        if set_value:
            if hasattr(param,'unit'):
                param.set_value(value,'SI')
            else:
                param.set_value(value)

        return value
        

def mass_from_parent_orbit(b, component, orbit):
    """
    @param component: label of component
    @type component: str
    @param orbit: label of orbit
    @type orbit: str
    """
    # TODO: determine component number (does this need bundle or system???)
    # TODO: build constraint string
    # TODO: make this work as: bundle.add_constraint(constraints.mass_from_parent_orbit, 'starA', 'orbitAB') - or similar
    raise NotImplementedError
    return None
    
def keplers_third_law_hierarchical(b, innerorbit, outerorbit):
    """
    """
    raise NotImplementedError
    return None


# TODO: some var names that are currently considered safe, aren't actually.  Particularly random unique_labels, perhaps that start with a digit?
# TODO: smart auto-label if not provided
# TODO: set constrained parameters to read-only with error when trying to set_value
# TODO: instead of showing user_labels let's show twig_labels which are (currently unique)... this means creating a function on the bundle to retrieve the smallest possible unique twig for a given entry.  
# TODO: handle when a label or unique_label in the system changes - need to loop over all vars and parameters in all constraints, update, and re-set to system
# TODO: if the user changes one of the eqs - need to check to make sure no new vars need to be created
# TODO: forbid constraints to depend on read-only values (perhaps from another constraint) - or fix so that order can be guessed or system of eqs is solved
# TODO: enable and test time dependency (will need to hook into set_time, which means also passed through run_compute)
# TODO: allow enabling/disabling constraints?
# TODO: create library of "pre-set" constraints similar to those in backend... will need to be careful about being hierarchy-generic
