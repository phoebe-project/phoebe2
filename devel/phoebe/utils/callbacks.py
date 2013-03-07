"""
Add callbacks to class methods.

This can be useful for the GUI or for any other implementation that needs to
know if a class method has been called or not.

Example usage: we'll attach a callback to the C{set_value} method of a
C{Parameter}.

    >>> from pyphoebe.parameters import parameters
    >>> par = parameters.Parameter(qualifier='teff',value=1000.)
    
We'll assume we have a dictionary holding the screen value of the Parameter,
and whenever the C{set_value} method is called, we'd want to update that
screen value:

    >>> widget = dict(screen_value=par.get_value())

So each time C{set_value} is set, we want to call the following function:
    
    >>> def update_GUI_widget(par,widget):
    ...    widget['screen_value'] = par.get_value()
    

It is now easy to attach that function to the parameter:
    
    >>> attach_signal(par,'set_value',update_GUI_widget,widget)

Now if you call C{set_value}:
    
    >>> par.set_value(2000.)

You can see that both the value and the widget are updated:

    >>> print(par.get_value())
    2000.0
    >>> print(widget['screen_value'])
    2000.0

It is possible to remove a signal again:
    
    >>> remove_signal(par,'set_value',update_GUI_widget)
    
Indeed, when setting the value again, the screen value is not updated:
    
    >>> par.set_value(3000.)
    >>> print(par.get_value())
    3000.0
    >>> print(widget['screen_value'])
    2000.0
    
Adding decorators to the functions removes their ability to be pickled.
To make them pickable again, you need to purge the signals (which is stronger
than removing them: the latter only removes the callback functions from the
list of signals):

    >>> purge_signals(par)
    >>> import cPickle
    >>> pickle_str = cPickle.dumps(par)
    >>> par.set_value(700.)
    >>> par.get_value()
    700.0
    >>> par = cPickle.loads(pickle_str)
    >>> par.get_value()
    3000.0
    
"""
import functools


def modify_class(cls,name):
    """
    Modify a class attribute/method.
    
    This decorator factory returns a decorator which modifies on the fly
    (i.e. monkeypatches) the class "cls" by adding or replacing the attribute
    (typically method) indicated by the variable "name" (which defaults to the
    name of the wrapper function) with the result  of the decorated function,
    to which is passed as only parameter the old attribute with the same name.
     
    @param cls: the class to modify
    @type cls: class object (not class instance or class name!)
    @param name: the name of the attribute/method to modify in the class
    @type name: str
    @return: a decorator
    @rtype: python function
    """
    def wrapper(self,fn):
        """
        The actual decorator returned by modify_class, which actually modifies
        the class.
        
        @param fn: the function to decorate
        @return: the argument function.
        """
        original_method = getattr(cls,name)
        new_method = fn(self,original_method)
        #-- and override the original method
        setattr(cls,name,new_method)
        return fn
    return wrapper

    
def callback(self,fctn):
    """
    Provides a callback to any function after the function is executed.
    
    @param self: the instance to add a callback to
    @type self: class instance
    @param fctn: function to add callback to
    @type fctn: python function
    @return: function wrapped with callback
    @rtype: python function
    """
    @functools.wraps(fctn)
    def add_callback(*args,**kwargs):
        output = fctn(*args,**kwargs)
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



def attach_signal(self,funcname,callbackfunc,*args):
    """
    Attach a signal to a function
    
    The calling ID of callbackfunc needs to be::
    
        callbackfunc(self,*args)
    
    where C{self} is the class instance. callbackfunc shouldn't return
    anything.
    
    @param self: the object to attach something to
    @type self: some class
    @param funcname: name of the class's method to add a callback to
    @type funcname: str
    @param callbackfunc: the callback function
    @type callbackfunc: callable function
    """
    #-- only add decorators when there is no decorator attached, by
    #   default, we'll assume there is already a decorator there.
    add_decorator = False
    #-- the list of functions that are called are collected in the "signals"
    #   attribute of a class. If no decorator was added to any method of the
    #   class, it does not exist yet.
    if not hasattr(self,'signals'):
        self.signals = {}
    #-- if no callbacks have been added to this function, reserve a place for
    #   it in the signal attribute. If it's not already in there, we know we
    #   have to add the callback decorator. Else, we can just append the
    #   callback function to the list.
    if funcname not in self.signals:
        self.signals[funcname] = []
        add_decorator = True
    #-- so add the callback function to the list
    self.signals[funcname].append((callbackfunc,args))
    #-- if necessary add a decorator
    if add_decorator:
        modify_class(self,funcname)(self,callback)

def remove_signal(self,funcname,callbackfunc):
    """
    Remove and attached signal.
    
    @param self: the object to attach something to
    @type self: some class
    @param funcname: name of the class's method to add a callback to
    @type funcname: str
    @param callbackfunc: the callback function
    @type callbackfunc: callable function
    """
    #-- look for the signal (we can't do "index" because the function and
    #   its initial arguments are stored)
    for index in range(len(self.signals[funcname])):
        if self.signals[funcname][index][0]==callbackfunc:
            self.signals[funcname].pop(index)
            

def purge_signals(cls):
    """
    Remove previously attached signals from a class.
    
    Can be useful if you want to pickle a class instance.
    
    @param cls: the class to modify
    @type cls: class object (not class instance or class name!)
    """
    if hasattr(cls,'signals'):
        for funcname in cls.signals:
            delattr(cls,funcname)
        delattr(cls,'signals')
        

    
if __name__=="__main__":
    import doctest
    doctest.testmod()