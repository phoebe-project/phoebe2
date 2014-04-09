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
    
    Optional keyword :envvar:`y_unit` can be any of ``ppt``, ``ppm``, ``relative``
    or an actual physical unit.
    
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
    y_unit = kwargs.pop('y_unit', None)
    
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
        except TypeError:
            pass
        #    raise ValueError("No observations in this system or component, so no scalings available: set keyword `scale=None`")
    # or using the synthetic computations    
    elif scale=='syn':
        pblum = syn['pblum']
        l3 = syn['l3']
    # else we don't scale
    
    if y_unit is not None:
        l3 = l3*pblum
        pblum = 1

    # Now take third light and passband luminosity contributions into account
    time = np.array(syn['time'])
    flux = np.array(syn['flux'])
    flux = flux * pblum + l3
    
    if y_unit is not None:
        if y_unit == 'pph':
            flux = (flux / np.median(flux) - 1 )*100
        elif y_unit == 'ppt':
            flux = (flux / np.median(flux) - 1 )*1000
        elif y_unit == 'ppm':
            flux = (flux / np.median(flux) - 1 )*1e6
        elif y_unit == 'relative':
            flux = flux / np.median(flux)
        else:
            lcdep,ref = system.get_parset(category='lc', ref=ref)
            flux = conversions.convert('erg/s/cm2/AA', y_unit, flux, passband=lcdep['passband'])
    
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
    
    All kwargs are passed on to matplotlib's `errorbar <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.errorbar>`_,
    except:
    
        - ``ref=0``: the reference of the lc to plot (index (int) or reference (str))
        - ``repeat=0``: handy if you are actually fitting a phase curve, and you
          want to repeat the phase curve a couple of times in the plot.
        - ``period=None``: period of repetition. If not given and :envvar:`phased=True`,
          the last time point will be used for the period.
        - ``phased=False``: decide whether to phase the data according to
          ``period`` or not.
        - ``t0``: arbitrary offset for phase plots (defaults to 0)
        - ``ax``: axis to plot on. If not given, the plot will be made on the
          current active axis (i.e. the return value of ``plt.gca()``)
    
    The following behaviour of matplotlib's ``errorbar`` function is overwritten:
        - ``label``: if you give no label, the default label is ``<ref> (obs)``.
          If you want no label at all, you need to set this to ``_nolegend_``.
    
    **Example usage:**
    
    >>> artists, obs = plot_lcobs(system, fmt='ko-')
    
    Returns the matplotlib objects and the observed parameterSet
    """
    ref = kwargs.pop('ref', 0)
    repeat = kwargs.pop('repeat', 0)
    period = kwargs.pop('period', None)
    phased = kwargs.pop('phased', False)
    t0 = kwargs.pop('t0', 0.0)
    ax = kwargs.pop('ax',plt.gca())

    # Get parameterSets
    obs = system.get_obs(category='lc', ref=ref)
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
    if errorbars:
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
                p = ax.errorbar(time+n,flux,yerr=sigm[o],**kwargs)
            else:
                p = ax.plot(time+n,flux,**kwargs)

    if loaded: obs.unload()
    
    return artists, obs


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
        period = max(syn_time)
    
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
    ``l3`` values.
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
    
    # Return the synthetic computations as an array
    ret_syn = syn.asarray()
    
    if loaded: syn.unload()
    
    return artists, ret_syn, l3


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
    ``l3`` values.
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
    
    To group together all the lcsyns which you want to plot, you can either
    give:
    
        - :envvar:`time`: a single point in time
        - :envvar:`group`: a group name
        - :envvar:`ref_pattern`: a reference pattern contained in all references
    """
    cmap = kwargs.pop('cmap', plt.cm.spectral)
    time = kwargs.pop('time', 0.)
    group = kwargs.pop('group', None)
    ref_pattern = kwargs.pop('ref_pattern', None)
    include_label = kwargs.pop('label', True)
    scale = kwargs.pop('scale', 'obs')
    
    
    # We'll need to plot all the observations of the LC category
    all_lc_refs = system.get_refs(category='lc')
    to_plot = OrderedDict()
    
    if ref_pattern is not None:
        all_lc_refs = [ref for ref in all_lc_refs if ref_pattern in ref]
    
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
        if group is None and ref_pattern is None:
            right_time = (syn['time'] == time)
        else:
            right_time = np.ones(len(syn['time']), bool)
        
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
    group = kwargs.pop('group', None)
    ref_pattern = kwargs.pop('ref_pattern', None)
    include_label = kwargs.pop('label', True)
    
    # We'll need to plot all the observations of the LC category
    all_lc_refs = system.get_refs(category='lc')
    if ref_pattern is not None:
        all_lc_refs = [ref for ref in all_lc_refs if ref_pattern in ref]
    
    to_plot = OrderedDict()
    
    # Collect the points per passband system, not per passband
    for j,ref in enumerate(all_lc_refs):
        # Get the pbdep (for info) and the synthetics
        dep, ref = system.get_parset(type='pbdep', ref=ref)
        obs, ref = system.get_parset(type='obs', ref=ref)
        
        passband = dep['passband']
        pass_sys = os.path.splitext(passband)[0]
        
        
        # An SED means we need the effective wavelength of the passbands
        if group is None and ref_pattern is None:
            right_time = (obs['time'] == time)    
        else:
            right_time = np.ones(len(obs['time']), bool)
            
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
    group = kwargs.pop('group', None)
    ref_pattern = kwargs.pop('ref_pattern', None)
    scale = kwargs.pop('scale', 'obs')
    units = kwargs.pop('units', 'sigma')
    include_label = kwargs.pop('label', True)
    
    
    # We'll need to plot all the observations of the LC category
    all_lc_refs = system.get_refs(category='lc')
    if ref_pattern is not None:
        all_lc_refs = [ref for ref in all_lc_refs if ref_pattern in ref]
        
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
        if group is None and ref_pattern is None:
            right_time = (obs['time'] == time)
        else:
            right_time = np.ones(len(obs['time']), bool)
            
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
    offset = kwargs.pop('offset', 0)
    ax = kwargs.pop('ax',plt.gca())
    x_unit = kwargs.pop('x_unit', 'nm')
    
    syn = system.get_synthetic(category='sp', ref=ref)
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
            #if 'vgamma' in obs and obs['vgamma']!=0:
            #    x = tools.doppler_shift(x, -obs.get_value('vgamma', 'km/s'))    
        except ValueError:
            pass
            #raise ValueError(("No observations in this system or component, "
            #             "so no scalings available: set keyword `scale=None`"))
    
    if x_unit is not 'nm':
        x = conversions.convert('nm',x_unit, x, wave=(np.median(x),'nm'))
    
    y = y * pblum + l3
    
    p, = ax.plot(x, y+offset, *args, **kwargs)
    
    if loaded:
        syn.unload()
    
    return [p], syn, pblum, l3
    
    
def plot_spobs_as_profile(system, *args, **kwargs):
    """
    Plot spobs as a spectroscopic line profile.
    """
    ref = kwargs.pop('ref', 0)
    index = kwargs.pop('index', 0)
    offset = kwargs.pop('offset', 0)
    
    
    obs, ref = system.get_parset(category='sp', type='obs', ref=ref)
    loaded = obs.load(force=False)
    
    kwargs.setdefault('label', obs['ref'] + ' (obs)')
    
    x = obs['wavelength']
    # only if wavelengths are different for every observation do we need to
    # select the correct one
    if len(x.shape)==2:
        x = x[index]
    y = obs['flux'][index] / obs['continuum'][index]
    e_y = obs['sigma'][index] / obs['continuum'][index]
    
    p = plt.errorbar(x, y+offset, yerr=e_y, **kwargs)
    
    if loaded:
        obs.unload()    
    
    return p, obs
    
    
def plot_spres_as_profile(system, *args, **kwargs):
    """
    Plot spobs as a spectroscopic line profile.
    """
    ref = kwargs.pop('ref', 0)
    index = kwargs.pop('index', 0)
    scale = kwargs.pop('scale', 'obs')
    offset = kwargs.pop('offset', 0)
    
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
    
    
    x = obs['wavelength']
    # only if wavelengths are different for every observation do we need to
    # select the correct one
    if len(x.shape)==2:
        x = x[index]
    y1 = obs['flux'][index] / obs['continuum'][index]
    e_y1 = obs['sigma'][index] / obs['continuum'][index]
    y = (y1 - y2) / e_y1
    e_y = np.ones(len(y))
    
    p = plt.errorbar(x, y+offset, yerr=e_y, **kwargs)
    
    if loaded_obs:
        obs.unload()        
    if loaded_syn:
        syn.unload()
    
    return p, obs, syn, pblum, l3
    
    


def plot_plsyn_as_profile(system, *args, **kwargs):
    """
    Plot plsyn as a spectroscopic line profile.
    """
    scale = kwargs.pop('scale', 'obs')
    ref = kwargs.pop('ref', 0)
    index = kwargs.pop('index', 0)
    ax = kwargs.pop('ax',plt.gca())
    velocity = kwargs.pop('velocity',None)
    stokes = kwargs.pop('stokes','I')
    
    if stokes == 'I':
        column = 'flux'
    else:
        column = stokes
    
    syn, ref = system.get_parset(category='pl', type='syn', ref=ref)
    loaded = syn.load(force=False)
    
    kwargs.setdefault('label', syn['ref'] + ' (syn)')
    
    x = syn['wavelength'][index]
    y = syn[column][index] / syn['continuum'][index]

    pblum = 1.0
    l3 = 0.0
    if scale == 'obs':
        try:
            obs = system.get_obs(category='pl', ref=ref)
            pblum = obs['pblum']
            l3 = obs['l3']
        except ValueError:
            pass
            #raise ValueError(("No observations in this system or component, "
            #             "so no scalings available: set keyword `scale=None`"))
    
    if velocity is not None:
        x = conversions.convert('nm','km/s', x, wave=velocity)
    
    if stokes == 'I':
        y = y * pblum + l3
    else:
        y = y * pblum
    
    p, = ax.plot(x, y, *args, **kwargs)
    
    if loaded:
        syn.unload()
    
    return [p], syn, pblum, l3
    
    
def plot_plobs_as_profile(system, *args, **kwargs):
    """
    Plot plobs as a spectroscopic line profile.
    """
    ref = kwargs.pop('ref', 0)
    index = kwargs.pop('index', 0)
    stokes = kwargs.pop('stokes','I')
    
    if stokes == 'I':
        column = 'flux'
        e_column = 'sigma'
    else:
        column = stokes
        e_column = 'sigma_' + stokes
    
    
    obs, ref = system.get_parset(category='pl', type='obs', ref=ref)
    loaded = obs.load(force=False)
    
    kwargs.setdefault('label', obs['ref'] + ' (obs)')
    
    obs = obs.asarray()
    if len(obs['wavelength'].shape)==2:
        x = obs['wavelength'][index]
    else:
        x = obs['wavelength']
    y = obs[column][index] / obs['continuum'][index]
    e_y = obs[e_column][index] / obs['continuum'][index]
    
    plt.errorbar(x, y, yerr=e_y, **kwargs)
    
    if loaded:
        obs.unload()    
    
    
def plot_plres_as_profile(system, *args, **kwargs):
    """
    Plot plobs as a spectroscopic line profile.
    """
    ref = kwargs.pop('ref', 0)
    index = kwargs.pop('index', 0)
    scale = kwargs.pop('scale', 'obs')
    stokes = kwargs.pop('stokes','I')
    
    if stokes == 'I':
        column = 'flux'
        e_column = 'sigma'
    else:
        column = stokes
        e_column = 'sigma_' + stokes
    
    obs, ref = system.get_parset(category='pl', type='obs', ref=ref)
    syn, ref = system.get_parset(category='pl', type='syn', ref=ref)
    
    loaded_obs = obs.load(force=False)
    loaded_syn = syn.load(force=False)
    
    kwargs.setdefault('label', syn['ref'] + ' (syn)')
    
    x = syn['wavelength'][index]
    y = syn[column][index] / syn['continuum'][index]
    z = obs[column][index] / obs['continuum'][index]
    e_z = obs[e_column][index] / obs['continuum'][index]
    
    pblum = 1.0
    l3 = 0.0
    if scale == 'obs':
        try:
            obs = system.get_obs(category='pl', ref=ref)
            pblum = obs['pblum']
            l3 = obs['l3']
        except ValueError:
            pass
            #raise ValueError(("No observations in this system or component, "
            #             "so no scalings available: set keyword `scale=None`"))    
    
    if stokes == 'I':
        y = y * pblum + l3
    else:
        y = y * pblum
    
    plt.errorbar(x, (z-y)/e_z, yerr=np.ones_like(x), **kwargs)
    
    if loaded_obs:
        obs.unload()        
    if loaded_syn:
        syn.unload()
    
    return x, y, (z, e_z)
    
    


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
    
    Keyword :envvar:`x` can be any of:
    
        - ``'baseline'``: plot visibilities wrt baseline.
        - ``'time'``: plot visibilities wrt time
    
    Keyword :envvar:`y` can be any of:
    
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
    elif y == 'vis2':
        plot_y = syn[y]
    else:
        raise NotImplementedError("y=phase not implemented in plot_ifsyn. Stick to vis or vis2")
    
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

