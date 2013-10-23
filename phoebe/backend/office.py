"""
Animate Phoebe computations interactively, or save them to a movie file.
"""
import numpy as np
import matplotlib.pyplot as plt
import phoebe

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
        
        


class Animation2(Animation):
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


class Animation3(Animation2):
    """
    Image and spectrum
    """
    def __init__(self, *args, **kwargs):
        super(Animation3,self).__init__(*args,**kwargs)
        self.draw_funcs = [image, plot_spsyn]
    
    

class Animation4(Animation):
    """
    Separated images.
    """
    def __init__(self, system, **kwargs):
        self.system = system
        self.repeat = kwargs.pop('repeat', False)
        self.save = kwargs.pop('save', None)
        self.close_after_finish = kwargs.pop('close_after_finish',True)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        self.axes = [ax1, ax2]
        self.draw_funcs = [image, image]
        self.draw_args = [(system[0],),(system[1],)]
        self.draw_kwargs = [kwargs, kwargs]
        self.initialized = False


    