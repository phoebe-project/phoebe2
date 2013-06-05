"""
Plotting facilities for observations and synthetic computations.

.. autosummary::
    
   plot_lcsyn
   plot_lcobs
   plot_lcres
   plot_rvsyn
   plot_rvobs
   plot_rvres
   
   
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from phoebe.atmospheres import passbands
from phoebe.atmospheres import tools
from phoebe.units import conversions
from phoebe.parameters import parameters

logger = logging.getLogger("BE.PLOT")

#{ Atomic plotting functions

def plot_lcsyn(system,*args,**kwargs):
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
          
    **Example usage:**
    
    >>> artists, syn, pblum, l3 = plot_lcsyn(system,'r-',lw=2)
        
    Returns the matplotlib objects, the synthetic parameterSet and the used ``pblum``
    and ``l3`` values
    """
    ref = kwargs.pop('ref',0)
    scale = kwargs.pop('scale','obs')
    repeat = kwargs.pop('repeat',0)
    period = kwargs.pop('period',None)
    #-- get parameterSets
    syn = system.get_synthetic(category='lc',ref=ref)
    
    #-- load synthetics: they need to be here
    loaded = syn.load(force=False)
    
    #-- try to get the observations. They don't need to be loaded, we just need
    #   the pblum and l3 values.
    if scale=='obs':
        try:
            obs = system.get_obs(category='lc',ref=ref)
            pblum = obs['pblum']
            l3 = obs['l3']
        except:
            raise ValueError("No observations in this system or component, so no scalings available: set keyword `scale=None`")
        
    elif scale=='syn':
        pblum = syn['pblum']
        l3 = syn['l3']
    else:
        pblum = 1.
        l3 = 0.
    
    #-- take third light and passband luminosity contributions into account
    time = np.array(syn['time'])
    flux = np.array(syn['flux'])
    flux = flux*pblum + l3
    
    #-- get the period to repeat the LC with
    if period is None:
        period = max(time)
    
    #-- plot model
    artists = []
    for n in range(repeat+1):
        p, = plt.plot(time+n*period, flux, *args, **kwargs)
        artists.append(p)

    if loaded: syn.unload()
    
    return artists,syn,pblum,l3


def plot_lcobs(system,**kwargs):
    """
    Plot lcobs as a light curve.
    
    All kwargs are passed on to matplotlib's `errorbar <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.errorbar>`_, except:
    
        - ``ref=0``: the reference of the lc to plot
        - ``repeat=0``: handy if you are actually fitting a phase curve, and you
          want to repeat the phase curve a couple of times.
        - ``period=None``: period of repetition. If not given, the last time point
          will be used
    
    **Example usage:**
    
    >>> artists, obs = plot_lcobs(system,fmt='ko-')
    
    Returns the matplotlib objects and the observed parameterSet
    """
    ref = kwargs.pop('ref',0)
    repeat = kwargs.pop('repeat',0)
    period = kwargs.pop('period',None)
    #-- get parameterSets
    obs = system.get_obs(category='lc',ref=ref)
    
    #-- load observations: they need to be here
    loaded = obs.load(force=False)
    
    time = obs['time']
    flux = obs['flux']
    sigm = obs['sigma']
    
    #-- get the period to repeat the LC with
    if period is None:
        period = max(time)
    
    #-- plot model
    artists = []
    for n in range(repeat+1):
        p = plt.errorbar(time+n*period,flux,yerr=sigm,**kwargs)
        artists.append(p)

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
    
    #-- get parameterSets
    obs = system.get_obs(category='lc',ref=ref)
    syn = system.get_synthetic(category='lc',ref=ref)
    
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
    for n in range(repeat+1):
        p = plt.errorbar(syn_time+n*period,(obs_flux-syn_flux)/obs_sigm,yerr=np.ones_like(obs_sigm),**kwargs)
        artists.append(p)

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
    #-- get parameterSets
    syn = system.get_synthetic(category='rv',ref=ref)
    
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
    for n in range(repeat+1):
        p, = plt.plot(time+n*period, conversions.convert('Rsol/d','km/s',rv), *args,**kwargs)
        artists.append(p)

    if loaded: syn.unload()
    
    return artists,syn,l3


def plot_rvobs(system,**kwargs):
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
    #-- get parameterSets
    obs = system.get_obs(category='rv',ref=ref)
    
    #-- load observations: they need to be here
    loaded = obs.load(force=False)
    
    time = obs['time']
    rv = obs['rv']
    sigm = obs['sigma']
    
    #-- get the period to repeat the RV with
    if period is None:
        period = max(time)
    
    #-- plot model
    artists = []
    for n in range(repeat+1):
        p = plt.errorbar(time+n*period,rv,yerr=sigm,**kwargs)
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
    
    #-- get parameterSets
    obs = system.get_obs(category='rv',ref=ref)
    syn = system.get_synthetic(category='rv',ref=ref)
    
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
    for n in range(repeat+1):
        p = plt.errorbar(syn_time+n*period,(obs_rv-syn_rv)/obs_sigm,yerr=np.ones_like(obs_sigm),**kwargs)
        artists.append(p)

    if loaded_obs: obs.unload()
    if loaded_syn: syn.unload()
    
    return artists, obs, syn, l3

#}


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

def plot_lcdep_as_lc(system,ref=0,residual=False,
                       kwargs_obs=None,kwargs_syn=None,
                       kwargs_residual=None,repeat=False):
    """
    Plot an entry in an lcdep as a light curve.
    
    This function will draw to the current active axes, and will set the
    axis labels.
    
    @param system: system to plot
    @type system: Body
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
        kwargs_residual = dict(fmt='ko-',ecolor='0.5')
    #-- get parameterSets
    dep,ref = system.get_parset(category='lc',type='pbdep',ref=ref)
    syn = system.get_synthetic(category='lc',ref=ref)
    obs = system.get_obs(category='lc',ref=ref)
    
    loaded_obs = obs.load(force=False)
    try:
        loaded_syn = syn.load(force=False)
    except IOError:
        loaded_syn = None
    
    #-- correct synthetic flux for corrections of third light and passband
    #   luminosity
    obs_time = obs['time']
    obs_flux = obs['flux']
    if 'sigma' in obs:
        obs_sigm = obs['sigma']
    else:
        obs_sigm = np.zeros(len(obs_flux))
    
    #-- take third light and passband luminosity contributions into account
    if loaded_syn is not None:
        syn_flux = np.array(syn['flux'])
        syn_flux = syn_flux*obs['pblum'] + obs['l3']
    
    #-- plot residuals or data + model
    if residual:
        plt.errorbar(obs_time,(obs_flux-syn_flux)/obs_sigm,yerr=np.ones(len(obs_sigm)),**kwargs_residual)
        if repeat:
            plt.errorbar(obs_time+obs_time[-1],(obs_flux-syn_flux)/obs_sigm,yerr=np.ones(len(obs_sigm)),**kwargs_residual)
        plt.ylabel(r'$\Delta$ Flux/$\sigma$')
    else:
        plt.errorbar(obs_time,obs_flux,yerr=obs_sigm,**kwargs_obs)
        if loaded_syn is not None:
            plt.plot(syn['time'],syn_flux,**kwargs_syn)
        
        if repeat:
            plt.errorbar(obs_time+obs_time[-1],obs_flux,yerr=obs_sigm,**kwargs_obs)
            if loaded_syn is not None:
                plt.plot(syn['time']+syn['time'][-1],syn_flux,**kwargs_syn)
            
            
        plt.ylabel('Flux')
    
    plt.xlabel("Time [days]")
    plt.title(ref)
    
    if loaded_obs: obs.unload()
    if loaded_syn: syn.unload()

    
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
    syn,ref = system.get_parset(category='sp',type='syn',ref=ref)
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
    




    
def plot_ifdep(system,ref=0,residual=False,select='vis2',
                       kwargs_obs=None,kwargs_syn=None,
                       kwargs_residual=None):
    """
    Plot squared visibilities or their phases.
    
    This function will draw to the current active axes, and will set the
    axis labels.
    
    @param system: system to plot
    @type system: Body
    @param ref: reference of the visibilities to be plotted
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
        kwargs_obs = dict(fmt='ko',ecolor='0.5',capsize=7)
    if kwargs_syn is None:
        kwargs_syn = dict(color='r',mew=2,marker='x',linestyle='',ms=10)
    if kwargs_residual is None:
        kwargs_residual = dict(fmt='ko',ecolor='0.5',capsize=7)
    #-- get parameterSets
    dep,ref = system.get_parset(category='if',type='pbdep',ref=ref)
    syn,ref = system.get_parset(category='if',type='syn',ref=ref)
    obs,ref = system.get_parset(category='if',type='obs',ref=ref)
    
    loaded_obs = obs.load(force=False)
    loaded_syn = syn.load(force=False)
    
    if 'sigma_vis2' in obs:
        obs_sigma_vis2 = obs['sigma_vis2']
    else:
        obs_sigma_vis2 = np.zeros(len(obs_flux))
    
    obs_baselines = np.sqrt(obs['ucoord']**2+obs['vcoord']**2)
    syn_baselines = np.sqrt(np.array(syn['ucoord'])**2+np.array(syn['vcoord'])**2)
    obs_vis2 = obs['vis2']
    syn_vis2 = np.array(syn['vis2'])
    #-- plot residuals or data + model
    if residual:
        plt.errorbar(obs_baselines,(obs_vis2-syn_vis2)/obs_sigma_vis2,yerr=np.ones(len(obs_sigma_vis2)),**kwargs_residual)
        plt.ylabel('$\Delta V^2$')
    else:
        plt.errorbar(obs_baselines,obs_vis2,yerr=obs_sigma_vis2,**kwargs_obs)
        plt.plot(syn_baselines,syn_vis2,**kwargs_syn)
        plt.ylabel('$V^2$')
        plt.gca().set_yscale('log',nonposy='clip')
    
    plt.xlabel("Baseline [m]")
    plt.grid()
    
    if loaded_obs: obs.unload()
    if loaded_syn: syn.unload()
    
    
def plot_pldep_as_profile(system,index=0,ref=0,stokes='I',residual=False,
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
    
    #-- shift the observed wavelengths if necessary
    if 'vgamma' in obs and obs['vgamma']!=0:
        obs_wave = tools.doppler_shift(obs_wave,obs.get_value('vgamma','km/s'))
    
    #-- normalise the spectrum and take third light and passband luminosity
    #   contributions into account
    syn_flux = np.array(syn[y][index])/np.array(syn['continuum'][index])
    if stokes=='I':
        syn_flux = syn_flux*obs['pblum'] + obs['l3']
    else:
        syn_flux = syn_flux*obs['pblum']
    
    #-- plot residuals or data + model
    if residual:
        plt.errorbar(obs['wavelength'],(obs_flux-syn_flux)/obs_sigm,yerr=np.ones(len(obs_sigm)),**kwargs_obs)
        plt.ylabel('$\Delta$ normalised Stokes {}'.format(stokes))
    else:
        plt.errorbar(obs['wavelength'],obs_flux,yerr=obs_sigm,**kwargs_obs)
        plt.plot(syn['wavelength'][index],syn_flux,**kwargs_syn)
        plt.ylabel('Normalised Stokes {}'.format(stokes))
    
    plt.xlabel("Wavelength [$\AA$]")
    plt.title(ref)
    
    if loaded_obs: obs.unload()
    if loaded_syn: syn.unload()
    
class Axes(parameters.ParameterSet):
    """
    Class representing a collection of plot commands for a single axes
    """
    def __init__(self,**kwargs):
        """
        Initialize an axes

        all kwargs will be added to the plotting:axes ParameterSet
        it is suggested to at least initialize with a category (lc,rv,etc) and title
        """
        self.axesoptions = parameters.ParameterSet(context="plotting:axes")
        self.plots = []
        
        for key in kwargs.keys():
            self.set_value(key, kwargs[key])
        
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
            if self.axesoptions['category'] == 'rv':
                yaxis = 'rv'
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
                
        # now loop through individual plot commands
        for plotoptions in self.plots:
            if not plotoptions['active']:
                continue
            if plotoptions['objref']=='auto':
                obj = system
            else:
                # copied functionality from bundle.get_object
                for path,item in system.walk_all():
                    if path[-1] == plotoptions['objref']:
                        obj = item
                    if path[-1] == 'orbit' and item['label'] == plotoptions['objref'] and len(path) == 3:
                        obj = system
                
            dataset,ref = obj.get_parset(type=plotoptions['type'][-3:], context=plotoptions['type'], ref=plotoptions['dataref'])
            
            if dataset is None:
                logger.error("dataset {} failed to load for objects {}".format(plotoptions['dataref'],plotoptions['objref']))
                return
                
            loaded = dataset.load(force=False) 
                
            po = {}
            for key in plotoptions.keys():
                if key not in ['dataref', 'objref', 'type', 'active']:
                    po[key] = plotoptions.get_value(key)
                    
            #if linestyle has not been set, make decision based on type
            if po['linestyle'] == 'auto':
                po['linestyle'] = 'None' if plotoptions['type'][-3:] == 'obs' else '-'
            #if color has not been set, make decision based on type
            if po['color'] == 'auto':
                po['color'] = 'k' if plotoptions['type'][-3:] == 'obs' else 'r' 
                
            # remove other autos
            for key in po.keys():
                if po[key]=='auto':
                    po.pop(key)

            # call mpl plot command
            
            # include an option to draw error bars, and if type is obs and errorbar on call axes.errorbar()?
            
            axes.plot(dataset[xaxis],dataset[yaxis],**po)
                
            # return data to its original loaded/unloaded state
            if loaded: dataset.unload()
