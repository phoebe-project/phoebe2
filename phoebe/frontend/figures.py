"""
Frontend plotting functionality
"""
import matplotlib.pyplot as plt
from phoebe.backend import plotting

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
        #~ self.mplaxes = {} # mplfig as keys, data axes as values
        #~ self.mplaxes_sel = {} # mplfig as keys, overlay as values
        self.plots = []
        
        self.phased = False
        self.period = None
        
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
        if plot in self.plots: #then we already have the plot
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

    def savefig(self,system,fname):
        """
        Save the plot to a file
        
        @parameter system: the phoebe system
        @type system: System
        @parameter fname: filename of output image
        @type fname: str
        """
        if self.mplaxes is None:
            self.plot(system)
        self.mplaxes.savefig(fname)
        return
            
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
            elif self.axesoptions['category'] == 'etv':
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
            location = self.axesoptions['location'] # location not added in ao
            
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
        xaxis = self.axesoptions.get_value('xaxis')
        phased = xaxis.split(':')[0]=='phase'
        
        # now loop through individual plot commands
        for plotoptions in self.plots:
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
                #~ orbit = self._get_orbit(xaxis.split(':')[1],system) 
                orbit = bundle.get_orbit(xaxis.split(':')[1])
            elif hasattr(obj,'params') and 'orbit' in obj.params.keys():
                orbit = obj.params['orbit']
            else:
                #~ orbit = self._get_orbit(plotoptions['objref'],system)
                orbit = bundle.get_orbit(plotoptions['objref'])
            period = orbit.get_value('period')
            self.period = period
            self.phased = phased
            #~ print "** period", period
            #~ print "** phased", phased     
            #~ print "** type", plotoptions['type']
            
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
            mplfig.sel_axes.get_xaxis().set_visible(False)
            mplfig.sel_axes.get_yaxis().set_visible(False)

            self.plot_select_time(bundle.select_time,mplfig=mplfig)
                
    def plot_select_time(self,time,mplfig):
        mplaxes_sel = mplfig.sel_axes
        mplaxes_sel.cla()
        mplaxes_sel.set_yticks([])
            
        if time is not None:
            if self.phased:
                time = (time % self.period) / self.period
            
            xlim = mplfig.data_axes.get_xlim()
            if time < xlim[1] and time > xlim[0]:
                mplaxes_sel.axvline(time, color='r')
