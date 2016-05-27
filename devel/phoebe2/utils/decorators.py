"""
List of common all-purpose decorators.
"""
import pickle
import functools
import inspect
import subprocess
import tempfile
import os

memory = {}

#{ Common tools

def memoized(*setting_args, **setting_kwargs):
    """
    Cache a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """
    no_args = False
    if len(setting_args) == 1 and not setting_kwargs \
        and callable(setting_args[0]):
        # We were called without args
        fctn = setting_args[0]
        no_args = True
    
    clear_when_different = setting_kwargs.get('clear_when_different', False)
    def outer(fctn):
        @functools.wraps(fctn)
        def memo(*args,**kwargs):
            haxh = pickle.dumps((fctn.__name__, args, sorted(kwargs.items())))
            modname = fctn.__module__
            if not (modname in memory):
                memory[modname] = {}
            if not (haxh in memory[modname]) and not clear_when_different:
                memory[modname][haxh] = fctn(*args,**kwargs)
            elif not (haxh in memory[modname]):
                memory[modname] = {haxh:fctn(*args,**kwargs)}
                
            return memory[modname][haxh]
        if memo.__doc__:
            memo.__doc__ = "\n".join([memo.__doc__,"This function is memoized."])
        return memo
    
    if no_args:
        return outer(fctn)
    else:
        return outer


def clear_memoization(keys=None):
    """
    Clear contents of memory
    """
    if keys is None:
        keys = list(memory.keys())
    for key in keys:
        if key in memory:
            riddens = [memory[key].pop(ikey) for ikey in list(memory[key].keys())[:]]