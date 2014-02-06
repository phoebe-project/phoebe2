"""
Frontend plotting functionality
"""
from collections import OrderedDict
from phoebe.parameters import parameters
from phoebe.backend import plotting
import matplotlib.pyplot as plt
import numpy as np

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
        
        self.settings = OrderedDict()
        
        self.settings['axes'] = parameters.ParameterSet(context="plotting:axes")
        self.settings['selector'] = parameters.ParameterSet(context="plotting:selector")
        self.settings['plots'] = []
        
        self.phased = False
        self.period = None
        
        for key in kwargs.keys():
            self.set_value(key, kwargs[key])
            
    def __str__(self):
        return self.to_string()
        
    def to_string(self):
        txt = ""
        for section in self.settings.keys():
            if isinstance(self.settings[section],list):
                # then assume plots (if add another section with list
                # then we'll need to make this more general like it is
                # in usersettings
                for label,ps in self.get_plot().items():
                    if ps is not None:
                        txt += "\n============ {}:{} ============\n".format(section,label)
                        txt += ps.to_string()
            else:
                ps = self.settings[section]
                if ps is not None:
                    txt += "\n============ {} ============\n".format(section)
                    txt += self.settings[section].to_string()
                
        return txt
            
    def keys(self):
        return self.settings['axes'].keys()
        
    def values(self):
        return self.settings['axes'].values()
        
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
        self.settings['plots'].append(plotoptions)
        
    def remove_plot(self,i):
        """
        Remove (permanently) the plot at the given index from the axes
        To deactivate the plot but keep the settings use get_plot(i).set_value('active',False)
        
        @param i: index of plot
        @type i: int
        @return: the remove plotoptions
        @rtype: ParameterSet
        """
        plot = self.get_plot(i)
        
        return self.settings['plots'].pop(self.settings['plots'].index(plot))
        
    def get_plot(self,ident=None):
        """
        Return a given plot by index
        
        @param ident: index of plot or objref:dataref:type
        @type ident: int or str
        @return: the desired plotoptions
        @rtype: ParameterSet        
        """
        plots = OrderedDict([('{}@{}@{}'.format(pl.get_value('dataref'),pl.get_value('type'),pl.get_value('objref')), pl) for pl in self.settings['plots']])
        
        if ident is None:
            return plots
        elif isinstance(ident,str):
            return plots[ident]
        else:
            return plots.values()[ident]
            
    def get_selector(self):
        """
        """
        return self.settings['selector']
            
    def get_dataset(self,plot,bundle):
        """
        return the dataset attached to a given plotting parameterset
        
        @param plot: the plotoptions or index of the plot
        @type plot: plotoptions or int
        @param bundle: the bundle
        @type bundle: Bundle
        @return: the dataset attached to the plotoptions
        @rtype: ParameterSet
        """
        if plot in self.settings['plots']: #then we already have the plot
            plotoptions = plot
        else:
            plotoptions = self.get_plot(plot)
            
        obj = bundle.get_object(plotoptions['objref'])
        if plotoptions['type'][-3:]=='syn' and hasattr(obj,'bodies'):
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
        return self.settings['axes'].get_value(key)
    
    def set_value(self,key,value):
        """
        Set a value in the axesoptions
        Same as axes.axesoptions.set_value()
        
        @param key: the name of the parameter
        @type key: str
        @param value: the new value
        """
        self.settings['axes'].set_value(key,value)
        
    def set_zoom(self,xlim,ylim):
        """
        set the xlim and ylim
        Same as set_value('xlim',xlim); set_value('ylim',ylim)
        
        @param xlim: xlimits (xmin,xmax)
        @type xlim: tuple
        @param ylim: ylimits (ymin,ymax)
        @type ylim: tuple
        """
        self.set_value('xlim',xlim)
        self.set_value('ylim',ylim)
        
    def get_zoom(self):
        """
        returns the xlim and ylim
        
        @return: ((xmin,xmax),(ymin,ymax))
        @rtype: 2 tuples
        """
        return self.get_value('xlim'),self.get_value('ylim')

    def savefig(self,bundle,fname):
        """
        Save the plot to a file
        
        @parameter system: the phoebe system
        @type system: System
        @parameter fname: filename of output image
        @type fname: str
        """
        
        self.plot(bundle)
        plt.savefig(fname)
        return fname
            
    def plot(self,bundle,mplfig=None,mplaxes=None,location=None,*args,**kwargs):
        """
        Plot all the children plots on a single axes
        
        @parameter bundle: the phoebe bundle
        @type bundle: Bundle
        @parameter mplfig: the matplotlib figure to add the axes to, if none is give one will be created
        @type mplfig: plt.Figure()
        @parameter mplaxes: the matplotlib figure to plot to (overrides mplfig)
        @type mpaxes: plt.Axes()
        @parameter location: the location on the figure to add the axes
        @type location: str or tuple        
        """
        
        system = bundle.get_system()
        
        # get options for axes
        ao = {}
        for key in self.settings['axes'].keys():
            if key not in ['location', 'active', 'category', 'xaxis', 'yaxis']:
                ao[key] = self.settings['axes'].get_value(key)
                
        # override anything set from kwargs
        for key in kwargs:
            ao[key]=kwargs[key]

            
        # control auto options
        xaxis, yaxis = self.settings['axes']['xaxis'], self.settings['axes']['yaxis']
        if xaxis == 'auto':
            xaxis = 'time'
        if yaxis == 'auto':
            if self.settings['axes']['category'] == 'lc':
                yaxis = 'flux'
            elif self.settings['axes']['category'] == 'rv':
                yaxis = 'rv'
            elif self.settings['axes']['category'] == 'sp':
                yaxis = 'wavelength'
            elif self.settings['axes']['category'] == 'etv':
                yaxis = 'ETV'
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
            location = self.settings['axes']['location'] # location not added in ao
            
        # of course we may be trying to plot to a different figure
        # so we'll override this later if that is the case 
        # just know that self.mplaxes is just to predict which axes
        # to use and not necessarily the correct axes
        
        if mplaxes is not None:
            axes = mplaxes
            mplfig = plt.gcf()
        elif mplfig is None:
            axes = plt.axes(**ao)
            mplfig = plt.gcf()
        else:
            if location != 'auto':
                if isinstance(location, str):
                    axes = mplfig.add_subplot(location,**ao)
                else:
                    axes = mplfig.add_subplot(location[0],location[1],location[2],**ao)
            else:
                axes = mplfig.add_subplot(111,**ao)
        
        # get phasing information
        xaxis = self.settings['axes'].get_value('xaxis')
        phased = xaxis.split(':')[0]=='phase'
        
        # now loop through individual plot commands
        for plotoptions in self.settings['plots']:
            #~ print "***", plotoptions['dataref'], plotoptions['objref'], plotoptions['type'], plotoptions['active']
            if not plotoptions['active']:
                continue

            # copied functionality from bundle.get_object (since we don't want to import bundle)
            obj = bundle.get_object(plotoptions['objref'])
            
            if plotoptions['type'][-3:]=='obs':
                dataset = bundle.get_obs(objref=plotoptions['objref'],dataref=plotoptions['dataref'])
            else:
                dataset = bundle.get_syn(objref=plotoptions['objref'],dataref=plotoptions['dataref'])
                
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
                orbit = bundle.get_orbitps(xaxis.split(':')[1])
            elif hasattr(obj,'params') and 'orbit' in obj.params.keys():
                orbit = obj.params['orbit']
            else:
                orbit = bundle.get_orbitps(plotoptions['objref'])
            period = orbit.get_value('period')
            self.period = period
            self.phased = phased
            
            # call appropriate plotting command
            if plotoptions['type']=='lcobs':
                artists,obs = plotting.plot_lcobs(obj, ref=plotoptions['dataref'], ax=axes, errorbars=errorbars, phased=phased, period=period, **po)
            elif plotoptions['type']=='lcsyn':
                artists,obs,pblum,l3 = plotting.plot_lcsyn(obj, ref=plotoptions['dataref'], ax=axes, phased=phased, period=period, **po)
            elif plotoptions['type']=='rvobs':
                artists,obs = plotting.plot_rvobs(obj, ref=plotoptions['dataref'], ax=axes, errorbars=errorbars, phased=phased, period=period, **po)
            elif plotoptions['type']=='rvsyn':
                artists,obs,l3 = plotting.plot_rvsyn(obj, ref=plotoptions['dataref'], ax=axes, phased=phased, period=period, **po)
            elif plotoptions['type']=='etvobs':
                artists,obs = plotting.plot_etvobs(obj, ref=plotoptions['dataref'], ax=axes, errorbars=errorbars, phased=phased, period=period, **po)
            elif plotoptions['type']=='etvsyn':
                artists,obs = plotting.plot_etvsyn(obj, ref=plotoptions['dataref'], ax=axes, phased=phased, period=period, **po)
            else:
                artists,obs = [],[]
        
        if mplfig is not None:
            mplfig.tight_layout()
            mplfig.data_axes = axes
            mplfig.sel_axes = axes.twinx()
            #~ mplfig.sel_axes.sharey(axes)
            mplfig.sel_axes.get_xaxis().set_visible(False)
            mplfig.sel_axes.get_yaxis().set_visible(False)

            self.plot_select_time(bundle,bundle.select_time,mplfig=mplfig)
                
    def plot_select_time(self,bundle,time,mplfig):
        mplaxes_sel = mplfig.sel_axes
        mplaxes_sel.cla()
        mplaxes_sel.set_yticks([])
        #~ mplaxes_sel.set_xticks([])
            
        if time is not None:
            if self.phased:
                time = (time % self.period) / self.period
            
            xlim = mplfig.data_axes.get_xlim()
            if time < xlim[1] and time > xlim[0]:
                # change command here based on self.settings['selector'] ps
                so = self.get_selector()
                if so.get_value('type')=='axvline':
                    mplaxes_sel.axvline(time, color=so.get_value('color'), alpha=so.get_value('alpha'))
                elif so.get_value('type')=='axvspan':
                    width_perc = so.get_value('size')
                    width_time = abs(xlim[1]-xlim[0])*width_perc/100.
                    mplaxes_sel.axvspan(time-width_time/2.,time+width_time/2., color=so.get_value('color'), alpha=so.get_value('alpha'))
                elif so.get_value('type')=='interp':
                    for plot in self.get_plot().values():
                        if plot.get_value('type')[-3:]=='syn': #then we want to plot interpolated value
                            # interpolate necessary value and make call to plot
                            ds = self.get_dataset(plot,bundle).asarray()
                            times, signal = ds['time'], ds['flux']
                            mplaxes_sel.plot([time],np.interp(np.array([time]),times,signal), marker=so.get_value('marker'), color=so.get_value('color'), markersize=so.get_value('size'), alpha=so.get_value('alpha'))
                else:
                    logger.warning("did not recognize type {}".format(so.get_value('type')))
                    
        mplaxes_sel.set_ylim(mplfig.data_axes.get_ylim())
