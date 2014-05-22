"""
Aminations and reporting.

Animate Phoebe computations interactively, or save them to a movie file.
"""
import numpy as np
import matplotlib.pyplot as plt
import phoebe
from phoebe.utils import coordinates

def summarize(system, time=None, filename=None):
    """
    Summarize a system's parameters in human readable form.
    
    """
    text = []
    system = system.copy()
    
    #for loc, thing in system.walk_all():
        #if isinstance(thing, parameters.ParameterSet) and 'extinction' in thing:
            #thing['extinction'] = 0.0
            #system.set_time(0.)
    if not len(system.mesh):
        system.set_time(0.)
    
            
    params = {}
    for loc, thing in phoebe.BodyBag([system]).walk_all():
        class_name = thing.__class__.__name__
        error = False
        if class_name in ['Star', 'BinaryStar', 'BinaryRocheStar',
                          'PulsatingBinaryRocheStar',
                          'AccretionDisk']:
            
            # Compute the luminosity of the object
            lum = phoebe.convert('erg/s', 'W', thing.luminosity()), 'W'
            
            # If this thing is a star, we can immediately derive mass, radius,
            # and effective temperature
            if class_name in ['Star', 'BinaryStar']:
                mass = thing.params['star'].get_value('mass', 'kg'), 'kg'
                radi = thing.params['star'].get_value('radius', 'm'), 'm'
                teff = thing.params['star'].get_value('teff', 'K'), 'K'
            
            # Actually we can also do that for BinaryRocheStars
            elif class_name in ['BinaryRocheStar', 'PulsatingBinaryRocheStar']:
                
                pot = thing.params['component']['pot']
                q = thing.params['orbit']['q']
                F = thing.params['component']['syncpar']
                sma = thing.params['orbit'].get_value('sma', 'm')
                
                comp = thing.get_component()
                mass = thing.params['orbit'].request_value('mass{:d}'.format(comp+1), 'kg'), 'kg'
                radi = phoebe.atmospheres.roche.potential2radius(pot, q, d=1,
                            F=F, component=comp+1, sma=sma, loc='pole',
                            tol=1e-10, maxiter=50), 'm'
                teff = thing.params['component'].get_value('teff', 'K'), 'K'
            
            # Else, we can't do it.
            else:
                error = True
            
            
            title = "Body '{}' (t={})".format(thing.get_label(), thing.time)
            text.append("\n\n\033[32m{}\n{}\033[m".format(title,'='*len(title)))
            text.append("\nSystem parameters\n==================\n")
            text.append(thing.list(summary='physical'))
            
            if not error:
                M = phoebe.convert(mass[1],'Msol', mass[0])
                L = phoebe.convert(lum[1],'Lsol', lum[0])
                R = phoebe.convert(radi[1],'Rsol', radi[0])
                T = phoebe.convert(teff[1],'K', teff[0])
                G = phoebe.convert('m/s2', '[cm/s2]', phoebe.constants.GG*mass[0]/radi[0]**2)
                
                teff_min, teff_max = thing.mesh['teff'].min(), thing.mesh['teff'].max()
                logg_min, logg_max = thing.mesh['logg'].min(), thing.mesh['logg'].max()
                radii = coordinates.norm(thing.mesh['_o_center'], axis=1)
                radi_min, radi_max = radii.min(), radii.max()
                
                text.append("")
                text.append("mass          = {:.6f} Msol".format(M))
                text.append("luminosity    = {:.6f} Lsol".format(L))
                text.append("radius        = {:.6f} Rsol  [{:.6f} - {:.6f}]".format(R, radi_min, radi_max))
                text.append("Teff          = {:.6f} K     [{:.6f} - {:.6f}]".format(T, teff_min, teff_max))
                text.append("logg          = {:.6f} [cgs] [{:.6f} - {:.6f}]".format(G, logg_min, logg_max))
                
                params[thing.get_label()] = dict(mass=M, luminosity=L, radius=R, teff=T, logg=G)
            
                ld_columns = [name[3:] for name in thing.mesh.dtype.names if (name[:3] == 'ld_')]
                
            text.append("\nPassband parameters\n===================\n")
            
            for ref in ld_columns:
                lbl = 'ld_{}'.format(ref)
                text.append("\n\033[33mPassband '{}':\n-------------------------\033[m".format(ref))
                
                parset = thing.get_parset(ref=ref)
                text.append(str(parset[0]))
                
                proj_int = thing.projected_intensity(ref=ref)
                mesh = thing.mesh
                wflux = mesh['proj_'+ref]
                wsize = mesh['size']
                weights = wflux*wsize
                
                if weights.sum()==0:
                    continue
                
                pteff = np.average(mesh['teff'], weights=weights, axis=0)
                plogg = np.log10(np.average(10**mesh['logg'], weights=weights, axis=0))
                
                dlnI_dlnT = np.log(mesh[lbl][:,-1]).ptp()/np.log(mesh['teff']).ptp()
                dlnI_dlng = np.log(mesh[lbl][:,-1]).ptp()/np.log(10**mesh['logg']).ptp()
                dlnT_dlng = np.log(mesh['teff']).ptp()/np.log(10**mesh['logg']).ptp()
                passband_gravb = dlnI_dlng + dlnT_dlng*dlnI_dlnT
                
                
                a0 = np.average(mesh[lbl][:,0], weights=weights), np.std(mesh[lbl][:,0])
                a1 = np.average(mesh[lbl][:,1], weights=weights), np.std(mesh[lbl][:,1])
                a2 = np.average(mesh[lbl][:,2], weights=weights), np.std(mesh[lbl][:,2])
                a3 = np.average(mesh[lbl][:,3], weights=weights), np.std(mesh[lbl][:,3])
                
                text.append("Projected intensity = {:.6e} erg/s/cm2/AA".format(proj_int))
                if ref=='__bol':
                    passband = 'BOL'
                    app_mag = phoebe.convert('erg/s/cm2/AA','mag',
                                                    proj_int, passband='OPEN.BOL')
                else:
                    passband = passband=parset[0]['passband']
                    app_mag = phoebe.convert('erg/s/cm2/AA','mag',
                                                    proj_int, passband=parset[0]['passband'])
                if not np.isnan(app_mag):
                    text.append("Apparent mag({}) = {:.3f}".format(passband, app_mag))
                text.append("Passband LD(a0) = {:8.4f} +/- {:8.4e}".format(*a0))
                text.append("Passband LD(a1) = {:8.4f} +/- {:8.4e}".format(*a1)) 
                text.append("Passband LD(a2) = {:8.4f} +/- {:8.4e}".format(*a2)) 
                text.append("Passband LD(a3) = {:8.4f} +/- {:8.4e}".format(*a3)) 
                text.append('Passband Teff = {:.3f} K'.format(pteff))
                text.append('Passband logg = {:.3f} [cgs]'.format(plogg))
                text.append("Passband dlnI/dlnTeff = {:.4f}".format(dlnI_dlnT))
                text.append("Passband dlnI/dlng = {:.4f}".format(dlnI_dlng))
                text.append("Passband gravity darkening = {:.4f}".format(passband_gravb))
                
    if filename is None:
        return "\n".join(text)
    else:
        with open(filename,'w') as ff:
            ff.write("\n".join(text))




class Animation(object):
    """
    Base class for on-the-fly animations
    
    anim.save('animation.mp4', fps=30, 
          extra_args=['-vcodec', 'h264', 
                      '-pix-fmt', 'yuv420p'])
    """
    
    def __init__(self):
        self.system = None
        self.axes = []
        self.draw_funcs = []
        self.draw_args = []
        self.draw_kwargs = []
        self.close_after_finish = True
        self.repeat = False
        self.save = None
        return None
    
    def draw(self):
        """
        Update all defined axes.
        """
        artists = []
        iterator = zip(self.axes, self.draw_funcs,
                        self.draw_args, self.draw_kwargs)
        for ax, draw_func, args, kwargs in iterator:
            if not self.initialized:
                kwargs['do_init'] = True
            else:
                kwargs['do_init'] = False
                
            kwargs['ax'] = ax
            if hasattr(ax, '_arts_to_update'):
                for art in ax._arts_to_update:
                    art.remove()
            output = draw_func(*args, **kwargs)
            ax._arts_to_update = output[0]
        self.initialized = True
        
        return artists
    
    def init_func(self):
        return None

        
def image(*args, **kwargs):
    """
    Draw an image.
    """
    xlims = kwargs.pop('xlims', None)
    ylims = kwargs.pop('ylims', None)
    do_init = kwargs.pop('do_init', False)
    ax = kwargs.pop('ax')
    system = args[0]
    
    figdec, artdec, p = system.plot2D(ax=ax, with_partial_as_half=False,
                                      **kwargs)
    
    if do_init:
        plt.xlabel("On sky coordinate X ($R_\odot$)")
        plt.ylabel("On sky coordinate Y ($R_\odot$)")
    if xlims is None and not do_init:
        xlims = list(ax.get_xlim())
        if xlims[0] > figdec['xlim'][0]:
            xlims[0] = figdec['xlim'][0]
        if xlims[1] < figdec['xlim'][1]:
            xlims[1] = figdec['xlim'][1]
    elif xlims is None:
        xlims = figdec['xlim']
    
    if ylims is None and not do_init:
        ylims = list(ax.get_ylim())
        if ylims[0] > figdec['ylim'][0]:
            ylims[0] = figdec['ylim'][0]
        if ylims[1] < figdec['ylim'][1]:
            ylims[1] = figdec['ylim'][1]
    elif ylims is None:
        ylims = figdec['ylim']

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    
    return [p],
    
def image(*args, **kwargs):
    """
    Draw an image.
    """
    xlims = kwargs.pop('xlims', None)
    ylims = kwargs.pop('ylims', None)
    do_init = kwargs.pop('do_init', False)
    ax = kwargs.pop('ax')
    incremental_axes = kwargs.pop('incremental_axes',True)
    system = args[0]
    
    figdec, artdec, p = system.plot2D(ax=ax, with_partial_as_half=False,
                                      **kwargs)
    
    if do_init:
        plt.xlabel("On sky coordinate X ($R_\odot$)")
        plt.ylabel("On sky coordinate Y ($R_\odot$)")
    
    if xlims is None and not incremental_axes:
        xlims = figdec['xlim']
    elif xlims is None and not do_init:
        xlims = list(ax.get_xlim())
        if xlims[0] > figdec['xlim'][0]:
            xlims[0] = figdec['xlim'][0]
        if xlims[1] < figdec['xlim'][1]:
            xlims[1] = figdec['xlim'][1]
    elif xlims is None:
        xlims = figdec['xlim']
    
    if ylims is None and not incremental_axes:
        ylims = figdec['ylim']
    elif ylims is None and not do_init:
        ylims = list(ax.get_ylim())
        if ylims[0] > figdec['ylim'][0]:
            ylims[0] = figdec['ylim'][0]
        if ylims[1] < figdec['ylim'][1]:
            ylims[1] = figdec['ylim'][1]
    elif ylims is None:
        ylims = figdec['ylim']

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    
    return [p],        



def plot_lcsyn(*args, **kwargs):
    """
    Draw a light curve.
    """
    xlims = kwargs.pop('xlims', None)
    ylims = kwargs.pop('ylims', None)
    do_init = kwargs.pop('do_init', False)
    ax = kwargs.pop('ax')
    system = args[0]
    
    out = phoebe.plotting.plot_lcsyn(system, ax=ax,**kwargs)
    if xlims is None:
        xlims = out[1]['time'].min(),out[1]['time'].max()
    #xlims_ = plt.xlim()
    #ax.set_xlim(min(xlims_[0], xlims[0]), max(xlims_[1], xlims[1]))
    ax.set_xlim(xlims)
    
    if ylims is None:
        ylims = out[1]['flux'].min(),out[1]['flux'].max()
    #ylims_ = plt.ylim()
    #ax.set_ylim(min(ylims_[0], ylims[0]), max(ylims_[1], ylims[1]))
    ax.set_ylim(ylims)
    
    if do_init:
        plt.xlabel("Time")
        plt.ylabel("Flux")
    
    return out[0],

def plot_rvsyn(*args, **kwargs):
    """
    Draw a light curve.
    """
    xlims = kwargs.pop('xlims', None)
    ylims = kwargs.pop('ylims', None)
    do_init = kwargs.pop('do_init', False)
    ax = kwargs.pop('ax')
    system = args[0]
    
    kwargs['color'] = 'k'
    out1 = phoebe.plotting.plot_rvsyn(system[0], ax=ax,**kwargs)
    kwargs['color'] = 'r'
    out2 = phoebe.plotting.plot_rvsyn(system[1], ax=ax,**kwargs)
    if xlims is None:
        xlims = out1[1]['time'].min(),out1[1]['time'].max()
    #xlims_ = plt.xlim()
    #ax.set_xlim(min(xlims_[0], xlims[0]), max(xlims_[1], xlims[1]))
    ax.set_xlim(xlims)
    
    #if ylims is None:
    #    ylims = out1[1]['rv'].min(),out1[1]['rv'].max()
    #ylims_ = plt.ylim()
    #ax.set_ylim(min(ylims_[0], ylims[0]), max(ylims_[1], ylims[1]))
    #ax.set_ylim(ylims)
    
    if do_init:
        plt.xlabel("Time")
        plt.ylabel("Radial velocity")
    
    return out1[0] + out2[0],

def plot_spsyn(*args, **kwargs):
    """
    Draw a light curve.
    """
    xlims = kwargs.pop('xlims', None)
    ylims = kwargs.pop('ylims', None)
    do_init = kwargs.pop('do_init', False)
    ax = kwargs.pop('ax')
    system = args[0]
    
    out = phoebe.plotting.plot_spsyn_as_profile(system, ax=ax, index=-1, **kwargs)
    
    if xlims is None:
        xlims = out[1]['wavelength'][0][0],out[1]['wavelength'][0][-1]
    xlims_ = plt.xlim()
    ax.set_xlim(min(xlims_[0], xlims[0]), max(xlims_[1], xlims[1]))
    
    if ylims is None:
        flux = out[1]['flux'][-1]/out[1]['continuum'][-1]
        ylims = flux.min(),flux.max()
    ylims_ = plt.ylim()
    ax.set_ylim(min(ylims_[0], ylims[0]), max(ylims_[1], ylims[1]))
    
    if do_init:
        plt.xlabel("Wavelength")
        plt.ylabel("Flux")
        
    return out[0],




    
class Animation1(Animation):
    """
    Only image
    """
    def __init__(self, system, **kwargs):
        self.system = system
        self.repeat = kwargs.pop('repeat', False)
        self.save = kwargs.pop('save', None)
        self.close_after_finish = kwargs.pop('close_after_finish',True)
        ax1 = plt.subplot(111)
        self.axes = [ax1]
        self.draw_funcs = [image]
        self.draw_args = [(system,)]
        self.draw_kwargs = [kwargs]
        self.initialized = False
        
        


class AnimationImLC(Animation):
    """
    Image and light curve
    """
    def __init__(self, system, kwargs1=None, kwargs2=None, **kwargs):
        if kwargs1 is None:
            kwargs1 = dict()
        if kwargs2 is None:
            kwargs2 = dict()
        
        self.system = system
        self.repeat = kwargs.pop('repeat', False)
        self.save = kwargs.pop('save', None)
        self.close_after_finish = kwargs.pop('close_after_finish',True)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        self.axes = [ax1, ax2]
        self.draw_funcs = [image, plot_lcsyn]
        self.draw_args = [(system,),(system,)]
        self.draw_kwargs = [kwargs1, kwargs2]
        self.initialized = False
        
class AnimationImRV(Animation):
    """
    Image and radial velocity curve
    """
    def __init__(self, system, kwargs1=None, kwargs2=None, **kwargs):
        if kwargs1 is None:
            kwargs1 = dict()
        if kwargs2 is None:
            kwargs2 = dict()
        
        #kwargs1.setdefault('context','rvdep')
        #kwargs1.setdefault('ref',0)
            
        self.system = system
        self.repeat = kwargs.pop('repeat', False)
        self.save = kwargs.pop('save', None)
        self.close_after_finish = kwargs.pop('close_after_finish',True)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        self.axes = [ax1, ax2, ax2]
        self.draw_funcs = [image, plot_rvsyn]
        self.draw_args = [(system,),(system,)]
        self.draw_kwargs = [kwargs1, kwargs2]
        self.initialized = False
        


class AnimationImSP(AnimationImLC):
    """
    Image and spectrum
    """
    def __init__(self, system, kwargs1=None, kwargs2=None, **kwargs):
        super(AnimationImSP,self).__init__(system, kwargs1=kwargs1, kwargs2=kwargs2, **kwargs)
        self.draw_funcs = [image, plot_spsyn]
    
    

class Animation4(Animation):
    """
    Separated images.
    """
    def __init__(self, system, **kwargs):
        self.system = system
        self.repeat = kwargs.pop('repeat', False)
        self.save = kwargs.pop('save', None)
        kwargs['incremental_axes'] = False
        self.close_after_finish = kwargs.pop('close_after_finish',True)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        self.axes = [ax1, ax2]
        self.draw_funcs = [image, image]
        self.draw_args = [(system[0],),(system[1],)]
        self.draw_kwargs = [kwargs, kwargs]
        self.initialized = False


    