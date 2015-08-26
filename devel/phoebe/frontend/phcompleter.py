import __main__

class Completer:
    """
    [FUTURE]
    """
    def __init__(self, namespace = None):
        """Create a new completer for the command line.

        Completer([namespace]) -> completer instance.

        Completer instances should be used as the completion mechanism of
        readline via the set_completer() call:

        readline.set_completer(Completer(my_namespace).complete)
        """

        if namespace and not isinstance(namespace, dict):
            raise TypeError('namespace must be a dictionary')

        # Don't bind to namespace quite yet, but flag whether the user wants a
        # specific namespace or to use __main__.__dict__. This will allow us
        # to bind to __main__.__dict__ at completion time, not now.
        if namespace is None:
            self.use_main_ns = 1
        else:
            self.use_main_ns = 0
            self.namespace = namespace

    def complete(self, text, state):
        """Return the next possible completion for 'text'.

        This is called successively with state == 0, 1, 2, ... until it
        returns None.  The completion should begin with 'text'.

        """
        if self.use_main_ns:
            self.namespace = __main__.__dict__
        if state == 0:
            self.matches = self.attr_matches(text)
        try:
            return self.matches[state]
        except IndexError:
            return None

    def attr_matches(self, text):
        """Compute matches when text contains a dot.

        Assuming the text is of the form NAME.NAME....[NAME], and is
        evaluable in self.namespace, it will be evaluated and its attributes
        (as revealed by dir()) are used as possible completions.  (For class
        instances, class members are also considered.)

        WARNING: this can still invoke arbitrary C code, if an object
        with a __getattr__ hook is evaluated.

        """
        tb_compl_commands = {'.get_parameter(': {'kind': 'Parameter', 'method': ['match', 'robust']},
                        '.info(': {'kind': 'Parameter', 'method': ['match', 'robust']},
                        '.get_value(': {'kind': ['Parameter','Parameter:value'], 'method': ['match', 'robust']},
                        '.get_value_all(': {'kind': ['Parameter','Parameter:value'], 'method': ['match', 'robust']},
                        '.set_value(': {'kind': ['Parameter','Parameter:value'], 'method': ['match', 'robust']},
                        '.set_value_all(': {'kind': ['Parameter','Parameter:value'], 'method': ['match', 'robust']},
                        '.get_adjust(': {'kind': ['Parameter','Parameter:adjust'], 'method': ['match', 'robust']},
                        '.set_adjust(': {'kind': ['Parameter','Parameter:adjust'], 'method': ['match', 'robust']},
                        '.set_adjust_all(': {'kind': ['Parameter','Parameter:adjust'], 'method': ['match', 'robust']},
                        '.get_prior(': {'kind': ['Parameter','Parameter:prior'], 'method': ['match', 'robust']},
                        '.set_prior(': {'kind': ['Parameter','Parameter:prior'], 'method': ['match', 'robust']},
                        '.get_posterior(': {'kind': ['Parameter', 'Parameter:posterior'], 'method': ['match', 'robust']},
                        '.remove_posterior(': {'kind': ['Parameter', 'Parameter:posterior'], 'method': ['match', 'robust']},
                        '.get_ps(': {'kind': 'ParameterSet', 'method': ['robust_notfirst', 'robust']},
                        '.get_ps_dict(': {'kind': 'OrderedDict', 'method': ['match', 'robust']},
                        '.get(': {'method': ['match', 'robust']},
                        '.twigs(': {'method': ['match', 'robust']},
                        '.search(': {'method': ['match', 'robust']},
                        '[': {'method': ['match', 'robust']},
                        '.get_constraint(': {'kind': 'ParameterSet', 'section': 'constraint', 'method': ['match', 'robust']},
                        '.run_constraint(': {'kind': 'ParameterSet', 'section': 'constraint', 'method': ['match', 'robust']},
                        '.remove_constraint(': {'kind': 'ParameterSet', 'section': 'constraint', 'method': ['match', 'robust']},
                        '.get_compute(': {'kind': 'ParameterSet', 'section': 'compute', 'method': ['match', 'robust']},
                        '.run_compute(': {'kind': 'ParameterSet', 'section': 'compute', 'method': ['match', 'robust']},
                        '.remove_compute(': {'kind': 'ParameterSet', 'section': 'compute', 'method': ['match', 'robust']},
                        '.get_fitting(': {'kind': 'ParameterSet', 'section': 'fitting', 'method': ['match', 'robust']},
                        '.run_fitting(': {'kind': 'ParameterSet', 'section': 'fitting', 'method': ['match', 'robust']},
                        '.remove_fitting(': {'kind': 'ParameterSet', 'section': 'fitting', 'method': ['match', 'robust']},
                        '.get_object(': {'kind': 'Body', 'method': ['robust_notfirst', 'robust']},
                        '.get_children(': {'kind': 'Body', 'method': ['robust_notfirst', 'robust']},
                        '.get_parent(': {'kind': 'Body', 'method': ['robust_notfirst', 'robust']},
                        '.set_enabled(': {'kind': 'OrderedDict', 'section': 'dataset', 'context': 'None', 'method': ['robust_notfirst', 'robust']},
                        '.enable_data(': {'kind': 'OrderedDict', 'section': 'dataset', 'context': 'None', 'method': ['robust_notfirst', 'robust']},
                        '.disable_data(': {'kind': 'OrderedDict', 'section': 'dataset', 'context': 'None', 'method': ['robust_notfirst', 'robust']},
                        '.get_obs(': {'context': '*obs', 'class_name': '*DataSet', 'method': ['robust_notfirst', 'robust']},
                        '.reload_obs(': {'context': '*obs', 'class_name': '*DataSet', 'method': ['robust_notfirst', 'robust']},
                        '.plot_obs(': {'context': '*obs', 'class_name': '*DataSet', 'method': ['robust_notfirst', 'robust']},
                        '.plot_residuals(': {'context': '*obs', 'class_name': '*DataSet', 'method': ['robust_notfirst', 'robust']},
                        '.get_syn(': {'context': '*syn', 'class_name': '*DataSet', 'method': ['robust_notfirst', 'robust']},
                        '.plot_syn(': {'context': '*syn', 'class_name': '*DataSet', 'method': ['robust_notfirst', 'robust']},
                        '.get_dep(': {'context': '*dep', 'kind': 'ParameterSet', 'method': ['robust_notfirst', 'robust']},
                        
                        }
        
        expr = None
        for cmd,kwargs in tb_compl_commands.items():
            if cmd in text:
                expr, attr = text.rsplit(cmd, 1)
                search_kwargs = kwargs
                if len(attr)==0:
                    return []
                elif attr[0] not in ["'",'"']:
                    return []
                else:
                    stringchar = attr[0]
                    attr = attr[1:]
                    break
        
        if expr is None:
            # then we haven't found a match
            return []
        
        try:
            thisobject = eval(expr, self.namespace)
        except Exception:
            return []

        # get the content of the object, except __builtins__
        words = thisobject.twigs(attr,ignore_exact=True,**search_kwargs)
        matches = []
        n = len(attr)
        for word in words:
            matches.append('{}{}{}{}'.format(expr,cmd,stringchar,word))
        return matches

