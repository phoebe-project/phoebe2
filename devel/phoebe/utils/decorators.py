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

def memoized(fctn):
    """
    Cache a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """
    @functools.wraps(fctn)
    def memo(*args,**kwargs):
        haxh = pickle.dumps((fctn.__name__, args, sorted(kwargs.items())))
        modname = fctn.__module__
        if not (modname in memory):
            memory[modname] = {}
        if not (haxh in memory[modname]):
            memory[modname][haxh] = fctn(*args,**kwargs)
        return memory[modname][haxh]
    if memo.__doc__:
        memo.__doc__ = "\n".join([memo.__doc__,"This function is memoized."])
    return memo

def clear_memoization(keys=None):
    """
    Clear contents of memory
    """
    if keys is None:
        keys = list(memory.keys())
    for key in keys:
        if key in memory:
            riddens = [memory[key].pop(ikey) for ikey in list(memory[key].keys())[:]]