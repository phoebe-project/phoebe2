"""
Plotting facilities for observations and synthetic computations.

.. autosummary::
    
   plot_lcsyn
   plot_lcobs
   plot_lcres
   plot_lcsyn_as_sed
   plot_lcobs_as_sed
   plot_lcres_as_sed
   
   plot_rvsyn
   plot_rvobs
   plot_rvres
   
   
   
   
"""

import logging
import itertools
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from phoebe.atmospheres import passbands
from phoebe.atmospheres import roche
from phoebe.atmospheres import tools
from phoebe.algorithms import marching
from phoebe.units import conversions
from phoebe.utils import utils
from phoebe.parameters import parameters

logger = logging.getLogger("BE.PLOT")

#{ Atomic plotting functions



def plot_lcsyn(system, *args, **kwargs):
    """
    Plot lcsyn as a light curve.
    
    All args and kwargs are passed on to matplotlib's `plot <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`_, except:
    
        - ``ref=0``: the reference of the lc to plot
        - ``scale='obs'``: correct synthetics for ``pblum`` and ``l3`` from
          the observations
        - ``repeat=0``: handy if you are actually fitting a phase curve, and you
          want to repeat the phase curve a couple of times.
        - ``period=None``: period of repetition. If not given, the last time point
          will be used
        - ``phased=False``: decide whether to phase the data according to
         ``period`` or not.
          
    **Example usage:**
    
    >>> artists, syn, pblum, l3 = plot_lcsyn(system,'r-',lw=2)
    
    The plotted data can then be reproduced with:
    
    >>> p, = plt.plot(syn['time'], syn['flux'] * pblum + l3, 'r-', lw=2)
        
    Returns the matplotlib objects, the synthetic parameterSet and the used ``pblum``
    and ``l3`` values
    
    The synthetic parameterSet is 'array-ified', which means that all the columns
    are arrays instead of lists.
    """
    # Get some default parameters
    ref = kwargs.pop('ref', 0)
    scale = kwargs.pop('scale', 'obs')
    repeat = kwargs.pop('repeat', 0)
    period = kwargs.pop('period', None)
    phased = kwargs.pop('phased', False)
    t0 = kwargs.pop('t0', 0.0)
    
    # Get parameterSets and set a default label if none is given
    syn = system.get_synthetic(category='lc', ref=ref)
    kwargs.setdefault('label', syn['ref'] + ' (syn)')
    
    # Get axes
    ax = kwargs.pop('ax',plt.gca())
    
    # Load synthetics: they need to be here
    loaded = syn.load(force=False)
    
    # Try to get the observations. They don't need to be loaded, we just need
    # the pblum and l3 values.
    # We can scale the synthetic light curve using the observations
    pblum = 1.0
    l3 = 0.0
    if scale == 'obs':
        try:
            obs = system.get_obs(category='lc', ref=ref)
            pblum = obs['pblum']
            l3 = obs['l3']
        except ValueError:
            pass
        #    raise ValueError("No observations in this system or component, so no scalings available: set keyword `scale=None`")
    # or using the synthetic computations    
    elif scale=='syn':
        pblum = syn['pblum']
        l3 = syn['l3']
    # else we don't scale

    
    # Now take third light and passband luminosity contributions into account
    time = np.array(syn['time'])
    flux = np.array(syn['flux'])
    flux = flux * pblum + l3
    
    # Get the period to repeat the LC with
    if period is None:
        period = max(time)
    
    # Plot model: either in phase or in time.
    artists = []
    if not phased:
        for n in range(repeat+1):
            p, = ax.plot(time+n*period, flux, *args, **kwargs)
            artists.append(p)
    else:
        time = ((time-t0) % period) / period
        sa = np.argsort(time)
        time, flux = time[sa], flux[sa]
        for n in range(repeat+1):
            if n>=1:
                kwargs['label'] = '_nolegend_'
            p, = ax.plot(time+n, flux, *args, **kwargs)
            artists.append(p)
    
    # Return the synthetic computations as an array
    ret_syn = syn.asarray()
    
    # Unload if loaded
    if loaded:
        syn.unload()
    
    # That's it!
    return artists, ret_syn, pblum, l3


def plot_lcobs(system,errorbars=True,**kwargs):
    """
    Plot lcobs as a light curve.
    
    All kwargs are passed on to matplotlib's `errorbar <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.errorbar>`_, except:
    
        - ``ref=0``: the reference of the lc to plot
        - ``repeat=0``: handy if you are actually fitting a phase curve, and you
          want to repeat the phase curve a couple of times.
        - ``period=None``: period of repetition. If not given, the last time point
          will be used
        - ``phased=False``: decide whether to phase the data according to
         ``period`` or not.
    
    **Example usage:**
    
    >>> artists, obs = plot_lcobs(system,fmt='ko-')
    
    Returns the matplotlib objects and the observed parameterSet
    """
    ref = kwargs.pop('ref',0)
    repeat = kwargs.pop('repeat',0)
    period = kwargs.pop('period', None)
    phased = kwargs.pop('phased', False)
    t0 = kwargs.pop('t0', 0.0)
    ax = kwargs.pop('ax',plt.gca())

    #-- get parameterSets
    obs = system.get_obs(category='lc',ref=ref)
    kwargs.setdefault('label', obs['ref'] + ' (obs)')
    
    #-- load observations: they need to be here
    loaded = obs.load(force=False)
    
    #-- take care of phased data
    if not 'time' in obs['columns'] and 'phase' in obs['columns']:
        myperiod = system[0].params['orbit']['period']
        myt0 = system[0].params['orbit']['t0']
        myphshift = system[0].params['orbit']['phshift']
        time = obs['phase'] * myperiod + myt0 #+ myphshift * myperiod
    elif 'time' in obs['columns']:
        time = obs['time']
    else:
        raise IOError("No times or phases defined")
    
    flux = obs['flux']
    sigm = obs['sigma']
    
    if not len(sigm):
        raise ValueError("Did not find uncertainties")
    
    #-- get the period to repeat the LC with
    if period is None:
        period = max(time)
    
    #-- plot model
    artists = []
    if not phased:
        for n in range(repeat+1):
            if n>=1:
                kwargs['label'] = '_nolegend_'
            if errorbars:
                p = ax.errorbar(time+n*period,flux,yerr=sigm,**kwargs)
            else:
                p = ax.plot(time+n*period,flux,**kwargs)
            artists.append(p)
    else:
        time = ((time-t0) % period) / period
        # need to sort by time (if using lines)
        o = time.argsort()
        time, flux = time[o], flux[o]
        for n in range(repeat+1):
            if n>=1:
                kwargs['label'] = '_nolegend_'
            if errorbars:
                p = ax.errorbar(time+n,flux,yerr=sigm,**kwargs)
            else:
                p = ax.plot(time+n,flux,**kwargs)

    if loaded: obs.unload()
    
    return artists,obs


def plot_lcres(system,*args,**kwargs):
    """
    Plot lcsyn and lcobs as a residual light curve.
    
    All kwargs are passed on to matplotlib's `errorbar <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.errorbar>`_, except:
    
        - ``ref=0``: the reference of the lc to plot
        - ``scale='obs'``: correct synthetics for ``pblum`` and ``l3`` from
          the observations
        - ``repeat=0``: handy if you are actually fitting a phase curve, and you
          want to repeat the phase curve a couple of times.
        - ``period=None``: period of repetition. If not given, the last time point
          will be used
    
    **Example usage:**
    
    >>> artists, obs, syn, pblum, l3 = plot_lcres(system,fmt='ko-')
    
        
    Returns the matplotlib objects, the observed and synthetic parameterSet, and the used ``pblum`` and ``l3``
    """
    ref = kwargs.pop('ref',0)
    scale = kwargs.pop('scale','obs')
    repeat = kwargs.pop('repeat',0)
    period = kwargs.pop('period',None)
    phased = kwargs.pop('phased',False)
    ax = kwargs.pop('ax',plt.gca())
    
    #-- get parameterSets
    obs = system.get_obs(category='lc',ref=ref)
    syn = system.get_synthetic(category='lc',ref=ref)
    kwargs.setdefault('label', obs['ref'])
    
    #-- load observations: they need to be here
    loaded_obs = obs.load(force=False)
    loaded_syn = syn.load(force=False)
    
    
    #-- try to get the observations. They don't need to be loaded, we just need
    #   the pblum and l3 values.
    if scale=='obs':
        pblum = obs['pblum']
        l3 = obs['l3']
    elif scale=='syn':
        pblum = syn['pblum']
        l3 = syn['l3']
    else:
        pblum = 1.
        l3 = 0.
    
    #-- take third light and passband luminosity contributions into account
    syn_time = np.array(syn['time'])
    syn_flux = np.array(syn['flux'])
    syn_flux = syn_flux*pblum + l3
    
    obs_time = obs['time']
    obs_flux = obs['flux']
    obs_sigm = obs['sigma']
    
    #-- get the period to repeat the LC with
    if period is None:
        period = max(obs_time)
    
    #-- plot model
    artists = []
    if not phased:
        for n in range(repeat+1):
            p = ax.errorbar(syn_time+n*period,(obs_flux-syn_flux)/obs_sigm,yerr=np.ones_like(obs_sigm),**kwargs)
            artists.append(p)
    else:
        syn_time = (syn_time % period) / period
        for n in range(repeat+1):
            p = ax.errorbar(syn_time+n,(obs_flux-syn_flux)/obs_sigm,yerr=np.ones_like(obs_sigm),**kwargs)


    if loaded_obs: obs.unload()
    if loaded_syn: syn.unload()
    
    return artists, obs, syn, pblum, l3
    
    


def plot_rvsyn(system,*args,**kwargs):
    """
    Plot rvsyn as a radial velocity curve.
    
    All args and kwargs are passed on to matplotlib's `plot <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`_, except:
    
        - ``ref=0``: the reference of the lc to plot
        - ``scale='obs'``: correct synthetics for ``pblum`` and ``l3`` from
        the observations
        - ``repeat=0``: handy if you are actually fitting a phase curve, and you
        want to repeat the phase curve a couple of times.
        - ``period=None``: period of repetition. If not given, the last time point
        will be used
    
    **Example usage:**
    
    >>> artists, syn, pblum, l3 = plot_rvsyn(system,'r-',lw=2)
        
    Returns the matplotlib objects, the plotted data and the ``pblum`` and
    ``l3`` values
    """
    ref = kwargs.pop('ref',0)
    scale = kwargs.pop('scale','obs')
    repeat = kwargs.pop('repeat',0)
    period = kwargs.pop('period',None)
    phased = kwargs.pop('phased',False)
    ax = kwargs.pop('ax',plt.gca())
    

    #-- get parameterSets
    syn = system.get_synthetic(category='rv',ref=ref)
    kwargs.setdefault('label', syn['ref'])
    
    #-- load synthetics: they need to be here
    loaded = syn.load(force=False)
    
    #-- try to get the observations. They don't need to be loaded, we just need
    #   the pblum and l3 values.
    if scale=='obs':
        obs = system.get_obs(category='rv',ref=ref)
        l3 = obs['l3']
    elif scale=='syn':
        l3 = syn['l3']
    else:
        l3 = 0.
    
    #-- take third light and passband luminosity contributions into account
    time = np.array(syn['time'])
    rv = np.array(syn['rv'])
    rv = rv + l3
    
    #-- get the period to repeat the RV with
    if period is None:
        period = max(time)
    
    #-- plot model
    artists = []
    if not phased:
        for n in range(repeat+1):
            p, = ax.plot(time+n*period, conversions.convert('Rsol/d','km/s',rv), *args,**kwargs)
            artists.append(p)
    else:
        time = (time % period) / period
        # need to sort by time (if using lines)
        o = time.argsort()
        time, rv = time[o], rv[o]
        for n in range(repeat+1):
            p, = ax.plot(time+n, conversions.convert('Rsol/d','km/s',rv), *args,**kwargs)
            artists.append(p)

    if loaded: syn.unload()
    
    return artists,syn,l3


def plot_rvobs(system,errorbars=True,**kwargs):
    """
    Plot rvobs as a radial velocity curve.
    
    All kwargs are passed on to matplotlib's `errorbar <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.errorbar>`_, except:
    
        - ``ref=0``: the reference of the lc to plot
        - ``repeat=0``: handy if you are actually fitting a phase curve, and you
          want to repeat the phase curve a couple of times.
        - ``period=None``: period of repetition. If not given, the last time point
          will be used
    
    **Example usage:**
    
    >>> artists, obs = plot_rvobs(system,fmt='ko-')
    
    Returns the matplotlib objects and the observed parameterSet
    """
    ref = kwargs.pop('ref',0)
    repeat = kwargs.pop('repeat',0)
    period = kwargs.pop('period',None)
    phased = kwargs.pop('phased',False)
    ax = kwargs.pop('ax',plt.gca())

    #-- get parameterSets
    obs = system.get_obs(category='rv',ref=ref)
    kwargs.setdefault('label', obs['ref'])
    
    #-- load observations: they need to be here
    loaded = obs.load(force=False)
    
    time = obs['time']
    rv = obs['rv']
    sigm = obs['sigma'] if 'sigma' in obs.keys() else [0]*len(rv) # or can we handle weights instead of sigma?
    
    #-- get the period to repeat the RV with
    if period is None:
        period = max(time)
    
    #-- plot model
    artists = []
    if not phased:
        for n in range(repeat+1):
            if errorbars:
                p = ax.errorbar(time+n*period,rv,yerr=sigm,**kwargs)
            else:
                p = ax.plot(time+n*period,rv, **kwargs)
            artists.append(p)
    else:
        time = (time % period) / period
        # need to sort by time (if using lines)
        o = time.argsort()
        time, rv = time[o], rv[o]
        for n in range(repeat+1):
            if errorbars:
                p = ax.errorbar(time+n,rv,yerr=sigm,**kwargs)
            else:
                p = ax.plot(time+n,rv,**kwargs)
            artists.append(p)

    if loaded: obs.unload()
    
    return artists,obs


def plot_rvres(system,*args,**kwargs):
    """
    Plot rvsyn and rvobs as a residual radial velocity curve.
    
    All kwargs are passed on to matplotlib's `errorbar <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.errorbar>`_, except:
    
        - ``ref=0``: the reference of the lc to plot
        - ``scale='obs'``: correct synthetics for ``l3`` from the observations
        - ``repeat=0``: handy if you are actually fitting a phase curve, and you
          want to repeat the phase curve a couple of times.
        - ``period=None``: period of repetition. If not given, the last time point
          will be used
    
    **Example usage:**
    
    >>> artists, obs, syn, l3 = plot_rvres(system,fmt='ko-')
    
        
    Returns the matplotlib objects, the observed and synthetic parameterSet, and the used ``l3``
    """
    ref = kwargs.pop('ref',0)
    scale = kwargs.pop('scale','obs')
    repeat = kwargs.pop('repeat',0)
    period = kwargs.pop('period',None)
    phased = kwargs.pop('phased',False)
    ax = kwargs.pop('ax',plt.gca())
    
    #-- get parameterSets
    obs = system.get_obs(category='rv',ref=ref)
    syn = system.get_synthetic(category='rv',ref=ref)
    kwargs.setdefault('label', obs['ref'])
    
    #-- load observations: they need to be here
    loaded_obs = obs.load(force=False)
    loaded_syn = syn.load(force=False)
    
    #-- try to get the observations. They don't need to be loaded, we just need
    #   the l3 values.
    if scale=='obs':
        l3 = obs['l3']
    elif scale=='syn':
        l3 = syn['l3']
    else:
        l3 = 0.
    
    #-- take third light and passband luminosity contributions into account
    syn_time = np.array(syn['time'])
    syn_rv = np.array(syn['rv'])
    syn_rv = syn_rv + l3
    
    obs_time = obs['time']
    obs_rv = obs['rv']
    obs_sigm = obs['sigma']
    
    #-- get the period to repeat the LC with
    if period is None:
        period = max(obs_time)
    
    #-- plot model
    artists = []
    syn_rv = conversions.convert('Rsol/d','km/s',syn_rv)
    
    if not phased:
        for n in range(repeat+1):
            p = ax.errorbar(syn_time+n*period,(obs_rv-syn_rv)/obs_sigm,yerr=np.ones_like(obs_sigm),**kwargs)
            artists.append(p)
    else:
        syn_time = (syn_time % period) / period
        # need to sort by time (if using lines)
        o = syn_time.argsort()
        syn_time_, syn_rv_ = syn_time[o], syn_rv[o]
        obs_rv_ = obs_rv[o]
        obs_sigm_ = obs_sigm[o]
        for n in range(repeat+1):
            p = ax.errorbar(syn_time_+n,(obs_rv_-syn_rv_)/obs_sigm_,yerr=np.ones_like(obs_sigm),**kwargs)
            artists.append(p)



    if loaded_obs: obs.unload()
    if loaded_syn: syn.unload()
    
    return artists, obs, syn, l3
    
def plot_etvsyn(system,*args,**kwargs):
    """
    Plot etvsyn as an etv curve
    
    All args and kwargs are passed on to matplotlib's `plot <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`_, except:
    
        - ``ref=0``: the reference of the lc to plot
        - ``repeat=0``: handy if you are actually fitting a phase curve, and you
        want to repeat the phase curve a couple of times.
        - ``period=None``: period of repetition. If not given, the last time point
        will be used
    
    **Example usage:**
    
    >>> artists, syn = plot_etvsyn(system,'r-',lw=2)
        
    Returns the matplotlib objects, the plotted data and the ``pblum`` and
    ``l3`` values
    """
    ref = kwargs.pop('ref',0)
    scale = kwargs.pop('scale','obs')
    repeat = kwargs.pop('repeat',0)
    period = kwargs.pop('period',None)
    phased = kwargs.pop('phased',False)
    ax = kwargs.pop('ax',plt.gca())

    #-- get parameterSets
    syn = system.get_synthetic(category='etv',ref=ref)
    
    #-- load synthetics: they need to be here
    loaded = syn.load(force=False)
    
    #-- take third light and passband luminosity contributions into account
    time = np.array(syn['time'])
    etv = np.array(syn['etv'])
    
    #-- get the period to repeat the RV with
    if period is None:
        period = max(time)
    
    #-- plot model
    artists = []
    if not phased:
        for n in range(repeat+1):
            p, = ax.plot(time+n*period, conversions.convert('d','s',etv), *args,**kwargs)
            artists.append(p)
    else:
        time = (time % period) / period
        # need to sort by time (if using lines)
        o = time.argsort()
        time, etv = time[o], etv[o]
        for n in range(repeat+1):
            p, = ax.plot(time+n, conversions.convert('d','s',etv), *args,**kwargs)
            artists.append(p)

    if loaded: syn.unload()
    
    return artists,syn
    
def plot_etvobs(system,errorbars=True,**kwargs):
    """
    Plot etvobs as a etv curve.
    
    All kwargs are passed on to matplotlib's `errorbar <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.errorbar>`_, except:
    
        - ``ref=0``: the reference of the lc to plot
        - ``repeat=0``: handy if you are actually fitting a phase curve, and you
          want to repeat the phase curve a couple of times.
        - ``period=None``: period of repetition. If not given, the last time point
          will be used
    
    **Example usage:**
    
    >>> artists, obs = plot_etvobs(system,fmt='ko-')
    
    Returns the matplotlib objects and the observed parameterSet
    """
    ref = kwargs.pop('ref',0)
    repeat = kwargs.pop('repeat',0)
    period = kwargs.pop('period',None)
    phased = kwargs.pop('phased',False)
    ax = kwargs.pop('ax',plt.gca())

    #-- get parameterSets
    obs = system.get_obs(category='etv',ref=ref)
    
    #-- load observations: they need to be here
    loaded = obs.load(force=False)
    
    time = obs['time']
    etv = obs['etv']
    sigm = obs['sigma']
    
    #-- get the period to repeat the ETV with
    if period is None:
        period = max(time)
    
    #-- plot model
    artists = []
    if not phased:
        for n in range(repeat+1):
            if errorbars:
                p = ax.errorbar(time+n*period,conversions.convert('d','s',etv),yerr=sigm,**kwargs)
            else:
                p = ax.plot(time+n*period, conversions.convert('d','s',etv), **kwargs)
            artists.append(p)
    else:
        time = (time % period) / period
        # need to sort by time (if using lines)
        o = time.argsort()
        time, etv = time[o], etv[o]
        for n in range(repeat+1):
            if errorbars:
                p = ax.errorbar(time+n,conversions.convert('d','s',etv),yerr=sigm,**kwargs)
            else:
                p = ax.plot(time+n,conversions.convert('d','s',etv),**kwargs)
            artists.append(p)

    if loaded: obs.unload()
    
    return artists,obs


#}

def plot_lcsyn_as_sed(system, *args, **kwargs):
    """
    Plot all lcsyns as an SED.
    """
    cmap = kwargs.pop('cmap', plt.cm.spectral)
    time = kwargs.pop('time', 0.)
    include_label = kwargs.pop('label', True)
    scale = kwargs.pop('scale', 'obs')
    # We'll need to plot all the observations of the LC category
    all_lc_refs = system.get_refs(category='lc')
    
    to_plot = OrderedDict()
    
    # Collect the points per passband system, not per passband
    for j,ref in enumerate(all_lc_refs):
        # Get the pbdep (for info) and the synthetics
        dep, ref = system.get_parset(type='pbdep',ref=ref)
        syn = system.get_synthetic(category='lc',ref=ref).asarray()
        
        # Try to get the observations. They don't need to be loaded, we just need
        # the pblum and l3 values.
        # We can scale the synthetic light curve using the observations
        if scale == 'obs':
            try:
                obs = system.get_obs(category='lc', ref=ref)
                pblum = obs['pblum']
                l3 = obs['l3']
            except:
                raise ValueError("No observations in this system or component, so no scalings available: set keyword `scale=None`")
        # or using the synthetic computations    
        elif scale=='syn':
            pblum = syn['pblum']
            l3 = syn['l3']
        # else we don't scale
        else:
            pblum = 1.
            l3 = 0.
        
        passband = dep['passband']
        pass_sys = os.path.splitext(passband)[0]
        
        if not pass_sys in to_plot:
            to_plot[pass_sys] = dict(x=[], y=[])
        
        # An SED means we need the effective wavelength of the passbands
        right_time = (syn['time'] == time)
        wave = passbands.get_info([passband])['eff_wave']
        wave = list(wave) * len(syn['flux'][right_time])
        
        to_plot[pass_sys]['x'].append(wave)
        to_plot[pass_sys]['y'].append(syn['flux'][right_time] * pblum + l3)
    
    # Decide on the colors
    color_cycle = itertools.cycle(cmap(np.linspace(0, 1, len(list(to_plot.keys())))))
    
    # And finally plot the points
    for key in to_plot.keys():
        kwargs_ = kwargs.copy()
    
        # Get data
        x = np.hstack(to_plot[key]['x'])
        y = np.hstack(to_plot[key]['y'])
        
        # Plot data
        if include_label:
            kwargs_['label'] = key
        kwargs_.setdefault('color',color_cycle.next())
        plt.plot(x, y, *args, **kwargs_)


def plot_lcobs_as_sed(system, *args, **kwargs):
    """
    Plot all lcobs as an SED.
    
    After plotting all photometry and showing the legend, you might find out
    that the legend is a big part of the figure. Either make it a figure
    legend, or try something like (it sets the text size small, and makes
    the legend frame semi-transparent)::
        
        >>> leg = plt.legend(loc='best', prop=dict(size='small'))
        >>> leg.get_frame().set_alpha(0.5)
    """
    cmap = kwargs.pop('cmap', plt.cm.spectral)
    time = kwargs.pop('time', 0.)
    include_label = kwargs.pop('label', True)
    # We'll need to plot all the observations of the LC category
    all_lc_refs = system.get_refs(category='lc')
    
    to_plot = OrderedDict()
    
    # Collect the points per passband system, not per passband
    for j,ref in enumerate(all_lc_refs):
        # Get the pbdep (for info) and the synthetics
        dep, ref = system.get_parset(type='pbdep', ref=ref)
        obs, ref = system.get_parset(type='obs', ref=ref)
        
        passband = dep['passband']
        pass_sys = os.path.splitext(passband)[0]
        
        
        # An SED means we need the effective wavelength of the passbands
        right_time = (obs['time'] == time)
        if not np.any(right_time):
            continue
        wave = passbands.get_info([passband])['eff_wave']
        
        wave = list(wave) * len(obs['flux'][right_time])
        
        if not pass_sys in to_plot:
            to_plot[pass_sys] = dict(x=[], y=[], e_y=[])
        
        
        to_plot[pass_sys]['x'].append(wave)
        to_plot[pass_sys]['y'].append(obs['flux'][right_time])
        to_plot[pass_sys]['e_y'].append(obs['sigma'][right_time])
    
    # Decide on the colors
    color_cycle = itertools.cycle(cmap(np.linspace(0, 1, len(list(to_plot.keys())))))
    
    # And finally plot the points
    for key in to_plot.keys():
        kwargs_ = kwargs.copy()
        # Get data
        x = np.hstack(to_plot[key]['x'])
        y = np.hstack(to_plot[key]['y'])
        e_y = np.hstack(to_plot[key]['e_y'])
        
        # Plot data
        if include_label:
            kwargs_['label'] = key
        kwargs_.setdefault('color',color_cycle.next())
        plt.errorbar(x, y, yerr=e_y, **kwargs_)


def plot_lcres_as_sed(system, *args, **kwargs):
    """
    Plot all lc residuals as an SED.
    """
    cmap = kwargs.pop('cmap', plt.cm.spectral)
    time = kwargs.pop('time', 0.)
    scale = kwargs.pop('scale', 'obs')
    units = kwargs.pop('units', 'sigma')
    include_label = kwargs.pop('label', True)
    # We'll need to plot all the observations of the LC category
    all_lc_refs = system.get_refs(category='lc')
    
    to_plot = OrderedDict()
    
    # Collect the points per passband system, not per passband
    for j,ref in enumerate(all_lc_refs):
        # Get the pbdep (for info) and the synthetics
        dep, ref = system.get_parset(type='pbdep', ref=ref)
        obs, ref = system.get_parset(type='obs', ref=ref)
        syn = system.get_synthetic(category='lc', ref=ref).asarray()
        
        
        # Try to get the observations. They don't need to be loaded, we just need
        # the pblum and l3 values.
        # We can scale the synthetic light curve using the observations
        if scale == 'obs':
            try:
                obs = system.get_obs(category='lc', ref=ref)
                pblum = obs['pblum']
                l3 = obs['l3']
            except:
                raise ValueError("No observations in this system or component, so no scalings available: set keyword `scale=None`")
        # or using the synthetic computations    
        elif scale=='syn':
            pblum = syn['pblum']
            l3 = syn['l3']
        # else we don't scale
        else:
            pblum = 1.
            l3 = 0.
        
        passband = dep['passband']
        pass_sys = os.path.splitext(passband)[0]
        
        if not pass_sys in to_plot:
            to_plot[pass_sys] = dict(x=[], y=[], e_y=[])
        
        # An SED means we need the effective wavelength of the passbands
        right_time = (obs['time'] == time)
        if not sum(right_time):
            continue
        wave = passbands.get_info([passband])['eff_wave']
        wave = list(wave) * len(obs['flux'][right_time])
        
        to_plot[pass_sys]['x'].append(wave)
        if units == 'sigma':
            to_plot[pass_sys]['y'].append(((obs['flux']-(syn['flux']*pblum+l3)) / obs['sigma'])[right_time])
            to_plot[pass_sys]['e_y'].append(np.ones(len(obs['flux'][right_time])))
        elif units == 'real':
            to_plot[pass_sys]['y'].append((obs['flux']-(syn['flux']*pblum+l3))[right_time])
            to_plot[pass_sys]['e_y'].append(obs['sigma'][right_time])
        elif units == 'relative':
            to_plot[pass_sys]['y'].append((obs['flux']/(syn['flux']*pblum+l3))[right_time])
            to_plot[pass_sys]['e_y'].append((obs['sigma']/(syn['flux']*pblum+l3))[right_time])
        else:
            raise NotImplementedError("units = {}".format(units))
        
    
    # Decide on the colors
    color_cycle = itertools.cycle(cmap(np.linspace(0, 1, len(list(to_plot.keys())))))
    
    
    # And finally plot the points
    for key in to_plot.keys():
        kwargs_ = kwargs.copy()
        # Get data
        if not len(to_plot[key]['x']):
            continue
        x = np.hstack(to_plot[key]['x'])
        y = np.hstack(to_plot[key]['y'])
        e_y = np.hstack(to_plot[key]['e_y'])
        
        # Plot data
        if include_label:
            kwargs_['label'] = key
        kwargs_.setdefault('color',color_cycle.next())
        plt.errorbar(x, y, yerr=e_y, **kwargs_)


def plot_spsyn_as_profile(system, *args, **kwargs):
    """
    Plot spsyn as a spectroscopic line profile.
    """
    scale = kwargs.pop('scale', 'obs')
    ref = kwargs.pop('ref', 0)
    index = kwargs.pop('index', 0)
    ax = kwargs.pop('ax',plt.gca())
    velocity = kwargs.pop('velocity',None)
    
    syn, ref = system.get_parset(category='sp', type='syn', ref=ref)
    loaded = syn.load(force=False)
    
    kwargs.setdefault('label', syn['ref'] + ' (syn)')
    
    
    x = syn['wavelength'][index]
    y = syn['flux'][index] / syn['continuum'][index]

    pblum = 1.0
    l3 = 0.0
    if scale == 'obs':
        try:
            obs = system.get_obs(category='sp', ref=ref)
            pblum = obs['pblum']
            l3 = obs['l3']
        
            # Shift the synthetic wavelengths if necessary
            if 'vgamma' in obs and obs['vgamma']!=0:
                x = tools.doppler_shift(x, -obs.get_value('vgamma', 'km/s'))    
        except ValueError:
            pass
            #raise ValueError(("No observations in this system or component, "
            #             "so no scalings available: set keyword `scale=None`"))
    
    if velocity is not None:
        x = conversions.convert('nm','km/s', x, wave=velocity)
    
    y = y * pblum + l3
    
    p, = ax.plot(x, y, *args, **kwargs)
    
    if loaded:
        syn.unload()
    
    return [p], syn, pblum, l3
    
    
def plot_spobs_as_profile(system, *args, **kwargs):
    """
    Plot spobs as a spectroscopic line profile.
    """
    ref = kwargs.pop('ref', 0)
    index = kwargs.pop('index', 0)
    
    
    obs, ref = system.get_parset(category='sp', type='obs', ref=ref)
    loaded = obs.load(force=False)
    
    kwargs.setdefault('label', obs['ref'] + ' (obs)')
    
    x = obs['wavelength'][index]
    y = obs['flux'][index] / obs['continuum'][index]
    e_y = obs['sigma'][index] / obs['continuum'][index]
    
    plt.errorbar(x, y, yerr=e_y, **kwargs)
    
    if loaded:
        obs.unload()    
    
    
def plot_spres_as_profile(system, *args, **kwargs):
    """
    Plot spobs as a spectroscopic line profile.
    """
    ref = kwargs.pop('ref', 0)
    index = kwargs.pop('index', 0)
    scale = kwargs.pop('scale', 'obs')
    
    obs, ref = system.get_parset(category='sp', type='obs', ref=ref)
    syn, ref = system.get_parset(category='sp', type='syn', ref=ref)
    
    loaded_obs = obs.load(force=False)
    loaded_syn = syn.load(force=False)
    
    try:
        pblum = obs['pblum']
        l3 = obs['l3']
        y2 = syn['flux'][index] / syn['continuum'][index] * pblum + l3
    except:
        raise ValueError(("No observations in this system or component, "
                         "so no scalings available: set keyword `scale=None`"))
    
    
    
    x = obs['wavelength'][index]
    y1 = obs['flux'][index] / obs['continuum'][index]
    e_y1 = obs['sigma'][index] / obs['continuum'][index]
    y = (y1 - y2) / e_y1
    e_y = np.ones(len(y))
    
    plt.errorbar(x, y, yerr=e_y, **kwargs)
    
    if loaded_obs:
        obs.unload()        
    if loaded_syn:
        syn.unload()        
    
    



def plot_lcdeps_as_sed(system,residual=False,
                       kwargs_obs=None,kwargs_syn=None,
                       kwargs_residual=None):
    """
    Plot lcdeps as a spectral energy distribution.
    
    This function will draw to the current active axes, and will set the
    axis labels. If C{residual=False}, both axis will be logscaled.
    
    @param system: system to plot
    @type system: Body
    @param residual: plot residuals or computed model and observations
    @type residual: bool
    @param kwargs_obs: extra matplotlib kwargs for plotting observations (errorbar)
    @type kwargs_obs: dict
    @param kwargs_syn: extra matplotlib kwargs for plotting synthetics (plot)
    @type kwargs_syn: dict
    @param kwargs_residual: extra matplotlib kwargs for plotting residuals (plot)
    @type kwargs_residual: dict
    """
    #-- get plotting options
    if kwargs_obs is None:
        kwargs_obs = dict(fmt='ko',ecolor='0.5',capsize=7)
    if kwargs_syn is None:
        kwargs_syn = dict(color='r',mew=2,marker='x',linestyle='',ms=10)
    if kwargs_residual is None:
        kwargs_residual = dict(fmt='ko',capsize=7)
    #-- we'll need to plot all the observations/synthetics of the lc category
    all_lc_refs = system.get_refs(category='lc')
    for j,ref in enumerate(all_lc_refs):
        #-- get the pbdep (for info), the synthetics and the observations
        dep,ref = system.get_parset(type='pbdep',ref=ref)
        syn,ref = system.get_parset(type='syn',ref=ref)
        obs,ref = system.get_parset(type='obs',ref=ref)
        #-- an SED means we need the effective wavelength of the passbands
        wave = passbands.get_info([dep['passband']])['eff_wave']
        wave = list(wave)*len(obs['flux'])
        #-- don't clutter the plot with labels
        label1 = 'Observed' if j==0 else '__nolegend__'
        label2 = 'Computed' if j==0 else '__nolegend__'
        #-- plot residuals or data
        if residual:
            plt.errorbar(wave,(obs['flux']-syn['flux'])/obs['sigma'],yerr=np.ones(len(wave)),label=label1,**kwargs_residual)
        else:    
            plt.errorbar(wave,obs['flux'],yerr=obs['sigma'],label=label1,**kwargs_obs)
            plt.plot(wave,syn['flux'],label=label2,**kwargs_syn)
            plt.legend(loc='best',numpoints=1).get_frame().set_alpha(0.5)
        plt.grid()
    if residual:
        plt.gca().set_xscale('log',nonposx='clip')
        plt.xlabel("Wavelength [$\AA$]")
        plt.ylabel("$\Delta$ Flux/$\sigma$")
    else:
        plt.gca().set_yscale('log',nonposy='clip')
        plt.gca().set_xscale('log',nonposx='clip')
        plt.xlabel("Wavelength [$\AA$]")
        plt.ylabel("Flux [erg/s/cm$^2$/$\AA$]")


def plot_spobs(system, *args, **kwargs):
    """
    Plot an observed spectrum.
    """
    ref = kwargs.pop('ref', 0)
    index = kwargs.pop('index', 0)
    ax = kwargs.pop('ax', plt.gca())
    normalised = kwargs.pop('normalised', True)
    
    # Get ParameterSets
    obs = system.get_obs(category='sp', ref=ref)
    kwargs.setdefault('label', obs['ref'] + ' (obs)')
    
    # Load observations, they need to be here
    loaded = obs.load(force=False)
    
    
    wavelength = np.ravel(np.array(obs['wavelength']))
    wavelength = wavelength.reshape(-1,len(obs['flux'][0]))
    wavelength = wavelength[min(index,wavelength.shape[0]-1)]
    
    # shift the observed wavelengths if necessary
    if 'vgamma' in obs and obs['vgamma']!=0:
        wavelength = tools.doppler_shift(wavelength, obs.get_value('vgamma','km/s'))
    
    flux = obs['flux'][index]
    cont = obs['continuum'][index]
    sigm = obs['sigma'][index]
    
    if normalised:
        flux = flux / cont
        sigm = sigm / cont
    
    artists = []
    p = ax.errorbar(wavelength, flux, yerr=sigm, **kwargs)
    artists.append(p)
    
    if loaded: obs.unload()
    
    return artists, obs
    
    
    
def plot_spsyn(system, *args, **kwargs):
    """
    Plot an observed spectrum.
    """
    ref = kwargs.pop('ref', 0)
    index = kwargs.pop('index', 0)
    ax = kwargs.pop('ax', plt.gca())
    normalised = kwargs.pop('normalised', True)
    
    # Get ParameterSets
    try:
        obs = system.get_obs(category='sp', ref=ref)
        pblum = obs['pblum']
        l3 = obs['l3']
    except:
        pblum = 1.0
        l3 = 0.0
        
    syn = system.get_synthetic(category='sp', ref=ref)
    kwargs.setdefault('label', syn['ref'] + ' (syn)')
    
    # Load observations, they need to be here
    loaded = syn.load(force=False)
    
    wavelength = np.ravel(np.array(syn['wavelength']))
    wavelength = wavelength.reshape(-1,len(syn['flux'][0]))
    wavelength = wavelength[min(index,wavelength.shape[0]-1)]
    
    flux = syn['flux'][index]
    cont = syn['continuum'][index]
    
    if normalised:
        flux = flux / cont
        
    flux = flux*pblum + l3
    
    artists = []
    p = ax.plot(wavelength, flux, *args, **kwargs)
    artists.append(p)
    
    if loaded: syn.unload()
    
    return artists, syn

    
def plot_spdep_as_profile(system,index=0,ref=0,residual=False,
                       kwargs_obs=None,kwargs_syn=None,
                       kwargs_residual=None):
    """
    Plot an entry in an spdep as a spectral profile.
    
    This function will draw to the current active axes, and will set the
    axis labels.
    
    @param system: system to plot
    @type system: Body
    @param index: for spdeps that are time dependent, take the index-th spectrum
    @type index: int
    @param ref: reference of the spectrum to be plotted
    @type ref: str
    @param residual: plot residuals or computed model and observations
    @type residual: bool
    @param kwargs_obs: extra matplotlib kwargs for plotting observations (errorbar)
    @type kwargs_obs: dict
    @param kwargs_syn: extra matplotlib kwargs for plotting synthetics (plot)
    @type kwargs_syn: dict
    @param kwargs_residual: extra matplotlib kwargs for plotting residuals (plot)
    @type kwargs_residual: dict
    """
    #-- get plotting options
    if kwargs_obs is None:
        kwargs_obs = dict(fmt='ko-',ecolor='0.5')
    if kwargs_syn is None:
        kwargs_syn = dict(color='r',linestyle='-',lw=2)
    if kwargs_residual is None:
        kwargs_residual = dict(fmt='ko',ecolor='0.5')
    #-- get parameterSets
    dep,ref = system.get_parset(category='sp',type='pbdep',ref=ref)
    syn = system.get_synthetic(category='sp',ref=ref)
    obs,ref = system.get_parset(category='sp',type='obs',ref=ref)
    
    loaded_obs = obs.load(force=False)
    try:
        loaded_syn = syn.load(force=False)
    except IOError:
        loaded_syn = None
    
    #-- correct synthetic flux for corrections of third light and passband
    #   luminosity
    obs_wave = obs['wavelength'].ravel()
    obs_flux = obs['flux'][index]
    if 'sigma' in obs:
        obs_sigm = obs['sigma'][index]
    else:
        obs_sigm = np.zeros(len(obs_flux))
    
    #-- shift the observed wavelengths if necessary
    if 'vgamma' in obs and obs['vgamma']!=0:
        obs_wave = tools.doppler_shift(obs_wave,obs.get_value('vgamma','km/s'))
    
    #-- normalise the spectrum and take third light and passband luminosity
    #   contributions into account
    if loaded_syn is not None:
        syn_flux = np.array(syn['flux'][index])/np.array(syn['continuum'][index])
        syn_flux = syn_flux*obs['pblum'] + obs['l3']
    
    #-- plot residuals or data + model
    if residual:
        plt.errorbar(obs_wave,(obs_flux-syn_flux)/obs_sigm,yerr=np.ones(len(obs_sigm)),**kwargs_obs)
        plt.ylabel('$\Delta$ normalised flux')
    else:
        plt.errorbar(obs_wave,obs_flux,yerr=obs_sigm,**kwargs_obs)
        if loaded_syn is not None:
            plt.plot(syn['wavelength'][index],syn_flux,**kwargs_syn)
        plt.ylabel('Normalised flux')
    
    plt.xlabel("Wavelength [$\AA$]")
    plt.title(ref)
    
    if loaded_obs: obs.unload()
    if loaded_syn: syn.unload()
    
def plot_ifsyn(system, *args, **kwargs):
    """
    Plot ifsyn.
    
    Parameter ``x`` can be any of:
    
        - ``'baseline'``: plot visibilities wrt baseline.
        - ``'time'``: plot visibilities wrt time
    
    Parameter ``y`` can be any of:
    
        - ``'vis'``: Visibilities
        - ``'vis2'``: Squared visibilities
        - ``'phase'``: Phases
    
    """
    # Get some default parameters
    ref = kwargs.pop('ref', 0)
    x = kwargs.pop('x', 'baseline')
    y = kwargs.pop('y', 'vis2')
    
    # Get parameterSets and set a default label if none is given
    syn = system.get_synthetic(category='if', ref=ref).asarray()
    kwargs.setdefault('label', syn['ref'] + ' (syn)')
    
    # Load synthetics: they need to be here
    loaded = syn.load(force=False)
    
    time = syn['time']
    
    if x == 'baseline':
        plot_x = np.sqrt(syn['ucoord']**2 + syn['vcoord']**2)
    else:
        plot_x = syn[x]
    
    if y == 'vis':
        plot_y = np.sqrt(syn['vis2'])
    else:
        plot_y = syn[y]
    
    plt.plot(plot_x, plot_y, *args, **kwargs)
   
    if loaded:
        syn.unload()


def plot_ifobs(system, *args, **kwargs):
    """
    Plot ifobs.
    
    Parameter ``x`` can be any of:
    
        - ``'baseline'``: plot visibilities wrt baseline.
        - ``'time'``: plot visibilities wrt time
    
    Parameter ``y`` can be any of:
    
        - ``'vis'``: Visibilities
        - ``'vis2'``: Squared visibilities
        - ``'phase'``: Phases
    
    """
    # Get some default parameters
    ref = kwargs.pop('ref', 0)
    x = kwargs.pop('x', 'baseline')
    y = kwargs.pop('y', 'vis2')
    
    # Get parameterSets and set a default label if none is given
    obs = system.get_obs(category='if', ref=ref)
    kwargs.setdefault('label', obs['ref'] + ' (obs)')
    
    # Load observations: they need to be here
    loaded = obs.load(force=False)
    
    time = obs['time']
    
    # Collect X-data for plotting
    if x == 'baseline':
        plot_x = np.sqrt(obs['ucoord']**2 + obs['vcoord']**2)
    else:
        plot_x = obs[x]
    
    # Are there uncertainties on X?
    if 'sigma_'+x in obs:
        plot_e_x = obs['sigma_'+x]
    else:
        plot_e_x = None
    
    # Collect Y-data for plotting
    if y == 'vis':
        plot_y = np.sqrt(obs['vis2'])
    else:
        plot_y = obs[y]
    
    # Are there uncertainties on Y?
    if 'sigma_'+y in obs:
        plot_e_y = obs['sigma_'+y]
    else:
        plot_e_y = None
    
    
    plt.errorbar(plot_x, plot_y, xerr=plot_e_y, yerr=plot_e_y, *args, **kwargs)
   
    if loaded:
        obs.unload()


def plot_ifres(system, *args, **kwargs):
    """
    Plot ifres.
    
    Parameter ``x`` can be any of:
    
        - ``'baseline'``: plot visibilities wrt baseline.
        - ``'time'``: plot visibilities wrt time
    
    Parameter ``y`` can be any of:
    
        - ``'vis'``: Visibilities
        - ``'vis2'``: Squared visibilities
        - ``'phase'``: Phases
    
    """
    # Get some default parameters
    ref = kwargs.pop('ref', 0)
    x = kwargs.pop('x', 'baseline')
    y = kwargs.pop('y', 'vis2')
    
    # Get parameterSets and set a default label if none is given
    obs = system.get_obs(category='if', ref=ref)
    syn = system.get_synthetic(category='if', ref=ref)
    kwargs.setdefault('label', obs['ref'])
    
    # Load observations: they need to be here
    loaded_obs = obs.load(force=False)
    loaded_syn = syn.load(force=False)
    
    time = obs['time']
    
    # Collect X-data for plotting
    if x == 'baseline':
        plot_x = np.sqrt(obs['ucoord']**2 + obs['vcoord']**2)
    else:
        plot_x = obs[x]
    
    # Are there uncertainties on X?
    if 'sigma_'+x in obs:
        plot_e_x = obs['sigma_'+x]
    else:
        plot_e_x = None
    
    # Collect Y-data for plotting
    if y == 'vis':
        plot_y_obs = np.sqrt(obs['vis2'])
        plot_y_syn = np.sqrt(syn['vis2'])
    else:
        plot_y_obs = obs[y]
        plot_y_syn = syn[y]
    
    # Are there uncertainties on Y?
    if 'sigma_'+y in obs:
        plot_e_y = obs['sigma_'+y]
    else:
        plot_e_y = np.ones(len(plot_y_syn))
    
    
    plt.errorbar(plot_x, (plot_y_obs-plot_y_syn)/plot_e_y, xerr=plot_e_y,
                                   yerr=np.ones_like(plot_e_y), *args, **kwargs)
   
    if loaded_obs:
        obs.unload()
    if loaded_syn:
        syn.unload()
    
    
def plot_pldep_as_profile(system,index=0,ref=0,stokes='I',residual=False,
                          velocity=False, factor=1.0,
                          kwargs_obs=None,kwargs_syn=None,
                          kwargs_residual=None):
    """
    Plot an entry in an pldep as a Stokes profile.
    
    This function will draw to the current active axes, and will set the
    axis labels.
    
    @param system: system to plot
    @type system: Body
    @param index: for spdeps that are time dependent, take the index-th spectrum
    @type index: int
    @param ref: reference of the spectrum to be plotted
    @type ref: str
    @param residual: plot residuals or computed model and observations
    @type residual: bool
    @param kwargs_obs: extra matplotlib kwargs for plotting observations (errorbar)
    @type kwargs_obs: dict
    @param kwargs_syn: extra matplotlib kwargs for plotting synthetics (plot)
    @type kwargs_syn: dict
    @param kwargs_residual: extra matplotlib kwargs for plotting residuals (plot)
    @type kwargs_residual: dict
    """
    #-- get plotting options
    if kwargs_obs is None:
        kwargs_obs = dict(fmt='ko-',ecolor='0.5')
    if kwargs_syn is None:
        kwargs_syn = dict(color='r',linestyle='-',lw=2)
    if kwargs_residual is None:
        kwargs_residual = dict(fmt='ko',ecolor='0.5')
    #-- get parameterSets
    dep,ref = system.get_parset(category='pl',type='pbdep',ref=ref)
    syn,ref = system.get_parset(category='pl',type='syn',ref=ref)
    obs,ref = system.get_parset(category='pl',type='obs',ref=ref)
    
    loaded_obs = obs.load(force=False)
    loaded_syn = syn.load(force=False)
    
    #-- correct synthetic flux for corrections of third light and passband
    #   luminosity
    if stokes=='I':
        y,yerr = 'flux','sigma'
    else:
        y,yerr = stokes,'sigma_'+stokes
        
    obs_flux = obs[y][index]
    if yerr in obs:
        obs_sigm = obs[yerr][index]
    else:
        obs_sigm = np.zeros(len(obs_flux))
    
    ##-- shift the observed wavelengths if necessary
    ## NONONO! This is taken care of during computation of the spectra!
    #if 'vgamma' in obs and obs['vgamma']!=0:
        #obs_wave = tools.doppler_shift(obs_wave,obs.get_value('vgamma','km/s'))
    
    #-- normalise the spectrum and take third light and passband luminosity
    #   contributions into account
    syn_flux = np.array(syn[y][index])/np.array(syn['continuum'][index])
    if stokes=='I':
        syn_flux = syn_flux*obs['pblum'] + obs['l3']
    else:
        syn_flux = syn_flux*obs['pblum']
    
    x_obs = obs['wavelength']
    x_syn = syn['wavelength'][index]
    
    if velocity:
        clambda = (x_obs[0]+x_obs[-1])/2.0
        unit = obs.get_parameter('wavelength').get_unit()
        x_obs = conversions.convert(unit, 'km/s', x_obs, wave=(clambda,unit))
        x_syn = conversions.convert(unit, 'km/s', x_syn, wave=(clambda,unit))
    
    #-- plot residuals or data + model
    if residual:
        plt.errorbar(x_obs,(obs_flux-syn_flux)/obs_sigm,yerr=np.ones(len(obs_sigm)),**kwargs_obs)
        plt.ylabel('$\Delta$ normalised Stokes {}'.format(stokes))
    else:
        plt.errorbar(x_obs,obs_flux*factor,yerr=obs_sigm*factor,**kwargs_obs)
        plt.plot(x_syn,syn_flux*factor,**kwargs_syn)
        plt.ylabel('Normalised Stokes {}'.format(stokes))
    
    plt.xlabel("Wavelength [$\AA$]")
    plt.title(ref)
    
    #plt.figure()
    #mu, sigma, model = system.get_model()
    #retvalue = (model - mu) / sigma
    #plt.figure()
    #plt.errorbar(range(len(mu)),mu,yerr=sigma,fmt='ko-')
    #plt.plot(range(len(mu)),model,'r-',lw=2)
    #plt.annotate('$\chi^2 = {:.3f}$'.format((retvalue**2).mean()),(0.9,0.9),ha='right',xycoords='axes fraction')      
    
    if loaded_obs: obs.unload()
    if loaded_syn: syn.unload()


def contour_BinaryRoche(orbit, height=0, comp=None, vmin=None, vmax=None,
                               res=200, ax=None, **kwargs):
    """
    Plot BinaryRoche potentials.
    
    If you give a component, then that component will be used to draw a line.
    
    Else, it'll be a bunch of contours!
    """
    if orbit['ecc'] != 0:
        raise NotImplementedError("Non-zero eccentricity needs a time point, or at least a default time point")
    
    # plot onto a given axes if necessary
    if ax is None:
        ax = plt.gca()
    # plot only one level if a component is given.
    if comp is not None:
        syncpar = comp['syncpar']
    else:
        syncpar = 1.0
        
    # Create a mesh grid in cartesian coordinates
    x_ = np.linspace(-1.5, 2.5, res)
    y_ = np.linspace(-1.5, 1.5, res)
    z = np.zeros((len(x_)*len(y_)))
    x,y = np.meshgrid(x_,y_)

    # Calculate all the potentials
    for i,(ix,iy) in enumerate(zip(np.ravel(x), np.ravel(y))):
        z[i] = marching.BinaryRoche([ix, iy, height], 1.0, orbit['q'], syncpar)
    
    # Make in logscale, that's nicer to plot
    z = np.log10(np.abs(z).reshape((len(x_),len(y_))))
    
    # Derive the levels to plot
    if comp is None:
        if vmin is None:
            vmin = z.min()
        if vmax is None:
            vmax = z.max()
        levels = kwargs.pop('levels',np.linspace(vmin, vmax, 100))
    else:
        levels = kwargs.pop('levels',[np.log10(comp['pot'])])
    
    # Make the actual plot
    artist = plt.contour(x*orbit['sma'], y*orbit['sma'], z, levels=levels, **kwargs)
    return artist, (x*orbit['sma'],y*orbit['sma'],z)


    
class Axes(object):
    """
    Class representing a collection of plot commands for a single axes
    """
    def __init__(self,**kwargs):
        """
        Initialize an axes

        all kwargs will be added to the plotting:axes ParameterSet
        it is suggested to at least initialize with a category (lc,rv,etc) and title
        """
        #~ super(Axes, self).__init__()
        
        self.axesoptions = parameters.ParameterSet(context="plotting:axes")
        self.plots = []
        #~ self.context = 'Axes'
        
        for key in kwargs.keys():
            self.set_value(key, kwargs[key])
            
    def keys(self):
        return self.axesoptions.keys()
        
    def values(self):
        return self.axesoptions.values()
        
    def add_plot(self,plotoptions=None,**kwargs):
        """
        Add a new plot command to this axes
        
        kwargs will be applied to plotoptions ParameterSet
        
        @param plotoptions: options for the new plot
        @type plotoptions: ParameterSet
        """
        if plotoptions is None:
            plotoptions = parameters.ParameterSet(context="plotting:plot")
        for key in kwargs.keys():
            plotoptions.set_value(key, kwargs[key])
        self.plots.append(plotoptions)
        
    def remove_plot(self,i):
        """
        Remove (permanently) the plot at the given index from the axes
        To deactivate the plot but keep the settings use get_plot(i).set_value('active',False)
        
        @param i: index of plot
        @type i: int
        @return: the remove plotoptions
        @rtype: ParameterSet
        """
        return self.plots.pop(i)
        
    def get_plot(self,i=None):
        """
        Return a given plot by index
        
        @param i: index of plot, will return list if None given
        @type i: int
        @return: the desired plotoptions
        @rtype: ParameterSet        
        """
        if i is None:
            return self.plots
        else:
            return self.plots[i]
            
    def get_dataset(self,plot,system):
        """
        return the dataset attached to a given plotting parameterset
        
        @param plot: the plotoptions or index of the plot
        @type plot: plotoptions or int
        @param system: the system
        @type system: BodyBag
        @return: the dataset attached to the plotoptions
        @rtype: ParameterSet
        """
        if plot in self.plots: #then we already have the plot
            plotoptions = plot
        else:
            plotoptions = self.get_plot(plot)
            
        obj = self._get_object(plotoptions['objref'],system)
        if plotoptions['type'][-3:]=='syn' and hasattr(obj,'bodies'):
            #~ dataset = obj.get_synthetic(ref=plotoptions['dataref'],cumulative=True if plotoptions['type'][:-3]=='lc' else False)
            dataset = obj.get_synthetic(ref=plotoptions['dataref'])
        else:
            dataset,ref = obj.get_parset(type=plotoptions['type'][-3:], context=plotoptions['type'], ref=plotoptions['dataref'])
        return dataset
        
    def get_value(self,key):
        """
        Get a value from the axesoptions
        Same as axes.axesoptions.get_value()
        
        @param key: the name of the parameter
        @type key: str
        @return: the parameter
        """
        return self.axesoptions.get_value(key)
    
    def set_value(self,key,value):
        """
        Set a value in the axesoptions
        Same as axes.axesoptions.set_value()
        
        @param key: the name of the parameter
        @type key: str
        @param value: the new value
        """
        self.axesoptions.set_value(key,value)
        
    def _get_object(self,objectname,bodybag):
        """
        copied functionality from bundle.get_object to avoid import
        """
        if objectname=='auto':
            return bodybag
        for item in bodybag.get_bodies():
            # check to see if we want the item
            for comp_key in ['component','star']:
                if comp_key in item.params.keys() and item.params[comp_key]['label'] == objectname:
                    return item
            
            # check to see if we want the itembag
            if 'orbit' in item.params.keys() and item.params['orbit']['label'] == objectname:
                return bodybag
        return None
        
    def _get_orbit(self,objectname,bodybag):
        for path,item in bodybag.walk_all():
            if path[-1] == 'orbit' and item['label']==objectname:
                return item
        return None
            
    def plot(self,system,mplfig=None,mplaxes=None,location=None,*args,**kwargs):
        """
        Plot all the children plots on a single axes
        
        @parameter system: the phoebe system
        @type system: System
        @parameter mplfig: the matplotlib figure to add the axes to, if none is give one will be created
        @type mplfig: plt.Figure()
        @parameter mplaxes: the matplotlib axes to plot to (overrides mplfig, axesoptions will not apply)
        @type mplaxes: plt.axes.Axes()
        @parameter location: the location on the figure to add the axes
        @type location: str or tuple        
        """
        
        # get options for axes
        ao = {}
        for key in self.axesoptions.keys():
            if key not in ['location', 'active', 'category', 'xaxis', 'yaxis']:
                ao[key] = self.axesoptions.get_value(key)
                
        # override anything set from kwargs
        for key in kwargs:
            ao[key]=kwargs[key]

            
        # control auto options
        xaxis, yaxis = self.axesoptions['xaxis'], self.axesoptions['yaxis']
        if xaxis == 'auto':
            xaxis = 'time'
        if yaxis == 'auto':
            if self.axesoptions['category'] == 'lc':
                yaxis = 'flux'
            elif self.axesoptions['category'] == 'rv':
                yaxis = 'rv'
            elif self.axesoptions['category'] == 'sp':
                yaxis = 'wavelength'
        if ao['xlabel'] == 'auto':
            ao['xlabel'] = xaxis
        if ao['ylabel'] == 'auto':
            ao['ylabel'] = yaxis
        if ao['xlim'] == (None, None):
            ao.pop('xlim')
        if ao['ylim'] == (None, None):
            ao.pop('ylim')
        if ao['xticks'] == ['auto']:
            ao.pop('xticks')
        if ao['yticks'] == ['auto']:
            ao.pop('yticks')
        if ao['xticklabels'] == ['auto']:
            ao.pop('xticklabels')
        if ao['yticklabels'] == ['auto']:
            ao.pop('yticklabels')
            
        # add the axes to the figure
        if location is None:
            location = self.axesoptions['location'] # location not added in ao
        
        if mplfig is None:
            if location == 'auto':  # then just plot to an axes
                if mplaxes is None: # no axes provided
                    axes = plt.axes(**ao)
                else: # use provided axes
                    axes = mplaxes
            else:
                mplfig = plt.Figure()
            
        if location != 'auto':
            if isinstance(location, str):
                axes = mplfig.add_subplot(location,**ao)
            else:
                axes = mplfig.add_subplot(location[0],location[1],location[2],**ao)
        else:
            if mplfig is not None:
                axes = mplfig.add_subplot(111,**ao)
                
        # get phasing information
        xaxis = self.axesoptions.get_value('xaxis')
        phased = xaxis.split(':')[0]=='phase'
        

                
        # now loop through individual plot commands
        for plotoptions in self.plots:
            #~ print "***", plotoptions['dataref'], plotoptions['objref'], plotoptions['type'], plotoptions['active']
            if not plotoptions['active']:
                continue

            # copied functionality from bundle.get_object (since we don't want to import bundle)
            obj = self._get_object(plotoptions['objref'],system) # will return system if objref=='auto'
            
            dataset = self.get_dataset(plotoptions,system)
            if len(dataset['time'])==0: # then empty dataset
                continue
                
            po = {}
            for key in plotoptions.keys():
                if key == 'errorbars':
                    errorbars = plotoptions.get_value(key)
                if key not in ['dataref', 'objref', 'type', 'active','errorbars']:
                    po[key] = plotoptions.get_value(key)
                    
            #if linestyle has not been set, make decision based on type
            if po['linestyle'] == 'auto':
                po['linestyle'] = 'None' if plotoptions['type'][-3:] == 'obs' else '-'
            if po['marker'] == 'auto':
                po['marker'] = 'None' if plotoptions['type'][-3:] == 'syn' else '.'
            #if color has not been set, make decision based on type
            if po['color'] == 'auto':
                po['color'] = 'k' if plotoptions['type'][-3:] == 'obs' else 'r' 
                
            # remove other autos
            for key in po.keys():
                if po[key]=='auto':
                    po.pop(key)
           
            if phased and len(xaxis.split(':')) > 1:
                orbit = self._get_orbit(xaxis.split(':')[1],system) 
            elif hasattr(obj,'params') and 'orbit' in obj.params.keys():
                orbit = obj.params['orbit']
            else:
                orbit = self._get_orbit(plotoptions['objref'],system)
            period = orbit.get_value('period')
            #~ print "** period", period
            #~ print "** phased", phased     
            #~ print "** type", plotoptions['type']

            # call appropriate plotting command
            if plotoptions['type']=='lcobs':
                artists,obs = plot_lcobs(obj, ref=plotoptions['dataref'], ax=axes, errorbars=errorbars, phased=phased, period=period, **po)
            elif plotoptions['type']=='lcsyn':
                artists,obs,pblum,l3 = plot_lcsyn(obj, ref=plotoptions['dataref'], ax=axes, phased=phased, period=period, **po)
            elif plotoptions['type']=='rvobs':
                artists,obs = plot_rvobs(obj, ref=plotoptions['dataref'], ax=axes, errorbars=errorbars, phased=phased, period=period, **po)
            elif plotoptions['type']=='rvsyn':
                artists,obs,l3 = plot_rvsyn(obj, ref=plotoptions['dataref'], ax=axes, phased=phased, period=period, **po)
            else:
                artists,obs = [],[]
