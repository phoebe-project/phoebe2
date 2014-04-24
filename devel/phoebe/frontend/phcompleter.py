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
            raise TypeError,'namespace must be a dictionary'

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
        tb_compl_commands = {'.get_parameter(': 'Parameter',
                        '.info(': 'Parameter',
                        '.get_value(': ['Parameter','Parameter:value'],
                        '.get_value_all(': ['Parameter','Parameter:value'],
                        '.set_value(': ['Parameter','Parameter:value'],
                        '.set_value_all(': ['Parameter','Parameter:value'],
                        '.get_adjust(': ['Parameter','Parameter:adjust'],
                        '.set_adjust(': ['Parameter','Parameter:adjust'],
                        '.set_adjust_all(': ['Parameter','Parameter:adjust'],
                        '.get_prior(': ['Parameter','Parameter:prior'],
                        '.set_prior(': ['Parameter','Parameter:prior'],
                        '.get_ps(': 'ParameterSet',
                        '.get_ps_dict(': 'OrderedDict',
                        '.get(': None,
                        '.twigs(': None,
                        '.search(': None,
                        '[': None,
                        }
        
        this_kind = None
        for cmd,kind in tb_compl_commands.items():
            if cmd in text:
                expr, attr = text.rsplit(cmd, 1)
                this_kind = kind
                if len(attr)==0:
                    return []
                elif attr[0] not in ["'",'"']:
                    return []
                else:
                    stringchar = attr[0]
                    attr = attr[1:]
                    break
                    
                
        else:
            return []
        
        try:
            thisobject = eval(expr, self.namespace)
        except Exception:
            return []

        # get the content of the object, except __builtins__
        words = thisobject.twigs(attr,kind=this_kind)
        matches = []
        n = len(attr)
        for word in words:
            matches.append('{}{}{}{}'.format(expr,cmd,stringchar,word))
        return matches

