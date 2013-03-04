"""
Classes and functions handling synthetic or observed data.
"""
import logging
import os
import numpy as np
from phoebe.parameters import parameters
import matplotlib.pyplot as plt

logger = logging.getLogger("PARS.DATA")


                    
class DataSet(parameters.ParameterSet):
    """
    ParameterSet representing a generic data set.
    
    A L{DataSet} is similar to a L{ParameterSet}, with additional features:
    
        1. It can :py:func:`load` columns of data (arrays) from a C{filename} to
           C{Parameters} in the L{DataSet}. This eliminates the need for parameters
           to be always loaded into memory. The L{DataSet} needs to contain the
           C{Parameter} C{filename} specifying the location of the file and
           C{columns} specifying the columns in the file.
        
        2. It can :py:func:`unload` C{columns} from the C{DataSet}. This removes them from
           memory, but they can always be reloaded from the file again. All in-place
           changes to these arrays since loading from the file will be lost.
        
        3. It can :py:func:`save` data from C{columns} in C{filename}.
        
        4. ``DataSets`` can be plotted
        
        5. It is possible to perform basic arithmic operations on DataSets,
           e.g. to take the difference between two DataSets.
    
    B{Example usage:} Create two C{Parameters} 
    
    >>> filename = Parameter(qualifier='filename',value='sample.lc')
    >>> columns = Parameter(qualifier='columns',value=['date','flux'])
    
    And put these in a L{DataSet}.
    
    >>> ds = DataSet(frame='test',definitions=None)
    >>> ds.add(filename)
    >>> ds.add(columns)
    >>> print(ds)
    filename sample.lc         --     test No description available
     columns ['date', 'flux']  --     test No description available
    
    At this point, only the C{filename} and C{columns} exist, but there is
    no way to access the data itself directly. For this, we need to hit the
    :py:func:`load` function:
    
    >>> ds.load()
    >>> print(ds)
    filename sample.lc                     --     test No description available
     columns ['date', 'flux']              --     test No description available
        date [0.0 ... 4.0]                 --     test <loaded from file sample.lc>
        flux [0.0 ... -1.22464679915e-15]  --     test <loaded from file sample.lc>
    
    To remove these arrays from memory, hit py:func:`unload`.
    
    >>> ds.unload()
    >>> print(ds)
    filename sample.lc         --     test No description available
     columns ['date', 'flux']  --     test No description available
     
    """
    def load(self,force=True):
        """
        Load the contents of a data file.
        
        This should be more sofisticated. We could autodect columns/units and
        descriptions from the file header. We should also be able to handle
        FITS and ASCII. This load function should be different for spectra,
        light curve etc.
        """
        if self.has_qualifier('filename') and os.path.isfile(self.get_value('filename')):
            filename = self.get_value('filename')
            columns = self.get_value('columns')
            #-- check if the data is already in here, and only reload when
            #   it is not, or when force is True
            if not force and len(self[self['columns'][0]])>0:
                return False
            data_columns = np.loadtxt(filename).T
            for i,col in enumerate(data_columns):
                if not columns[i] in self:
                    self.add(dict(qualifier=columns[i],value=col,description='<loaded from file {}>'.format(filename)))
                else:
                    self[columns[i]] = col
           # logger.info("Loaded contents of {}".format(filename))
        elif self.has_qualifier('filename') and self['filename']:
            raise IOError("File {} does not exist".format(self.get_value('filename')))
        elif self.has_qualifier('filename'):
            return False
        else:
            logger.info("No file to reload")
        return True
            
    def unload(self):
        """
        Remove arrays from the parameterSet to save memory.
        """
        if self.has_qualifier('filename'):
            filename = self.get_value('filename')
            columns = self.get_value('columns')
            for col in columns:
                if col in self: self.remove(col)
    
    def save(self,filename=None):
        """
        Save the contents of C{columns} to C{filename}.
        
        We should offer the possibility here to write FITS files too.
        Because I'm astronomically lazy, it ain't gonna happen today!
        """
        #-- take 'filename' from the parameterset if nothing is given, but
        #   check if we have the permission to write the file.
        #if filename is None and not self.get_adjust('filename'):
        #    filename = self['filename']
        #    raise ValueError("Cannot overwrite file {} (safety is ON)".format(filename))
        if filename is None:
            filename = self['filename']
        if not filename:
            filename = self['ref']
        #-- possibly we need to redefine the columns here:
        for col in self['columns']:
            if not len(self[col]):
                self[col] = np.zeros(len(self['time']))
        np.savetxt(filename,np.column_stack([self[col] for col in self['columns']]))
        logger.info('Wrote file {} with columns {} from dataset {}:{}'.format(filename,', '.join(self['columns']),self.context,self['ref']))
    
    def estimate_noise(self,from_col='flux',to_col='sigma'):
        """
        Estimate the noise by differencing.
        """
        did_load = False
        if not len(self[from_col]):
            self.load()
            did_load = True
        noise = np.diff(self[from_col])/np.sqrt(2)
        self[to_col] = np.hstack([noise[0],noise])
        if did_load:
            self.unload()
    
    def __add__(self,other):
        """
        Add two DataSets together.
        
        This means summing up every column except for the "time" and  "weights"
        columns, which will be kept from the original one. The "sigma" columns
        will be added with error propagation. If more complicated stuff needs
        to be done, you need to subclass DataSet and reimplement addition.
        
        Note that we should actually check here first if we can add columns
        (i.e. they have the same shape), and if the "date" columns are every
        where equal.
        """
        result = self.copy()
        columns = result.get_value('columns')
        for col in columns:
            if col in ['time','weights','flag']: continue
            this_col = np.array(result[col])
            other_col = np.array(other[col])
            if col in ['sigma']:
                result[col] = list(np.sqrt( this_col**2 + other_col**2))
            else:
                result[col] = list(this_col + other_col)
        return result
    
    def __radd__(self,other):
        if other==0:
            return self.copy()
        else:
            return self.__add__(other)
    
class LCDataSet(DataSet):
    def __init__(self,**kwargs):
        kwargs.setdefault('context','lcobs')
        super(LCDataSet,self).__init__(**kwargs)
    
    def plot(self,ax=None,**plotoptions):
        """
        Plot spectra.
        """
        if ax is None:
            ax = plt.gca()
        loaded = self.load(force=False)
        if self.context[-3:]=='obs':
            plotoptions.setdefault('color','k')
            plotoptions.setdefault('linestyle','None')
            plotoptions.setdefault('marker','o')
        else:
            plotoptions.setdefault('color','r')
            plotoptions.setdefault('linestyle','-')
            plotoptions.setdefault('marker','None')
            plotoptions.setdefault('linewidth',2)
        plotoptions.setdefault('mode','time series')
        
        if plotoptions['mode']=='time series':
            x = np.array(self['time'])
        
        flux = np.array(self['flux'])
        
        if 'sigma' in self:
            e_flux = np.array(self['sigma'])
        else:
            e_flux = None
            
        #-- and then plot!
        if e_flux is not None:
            ax.errorbar(x,flux,e_flux,**plotoptions)
        else:   
            ax.plot(x,flux,**plotoptions)
        ax.set_xlim(x.min(),x.max())
        ax.set_xlabel('Time')
        ax.set_ylabel("Flux")
        if loaded:
            self.unload()
        
class RVDataSet(DataSet):
    def __init__(self,**kwargs):
        kwargs.setdefault('context','rvobs')
        super(RVDataSet,self).__init__(**kwargs)

class SPDataSet(DataSet):
    """
    DataSet representing a spectrum.
    
    There is not a one-to-one correspondence to load and unload, since we
    assume the wavelengths to be the same for each spectrum. What to do with
    this? If we allow them to be different, it becomes a pain in the *** to
    save them to a file (unless we start using HDF5). We can also load the
    same wavelength array for each saved spectrum (I think we'd want to allow
    the user to define different wavelength ranges for each computation,
    it's only automatic saving that becomes difficult).
    """
    def __init__(self,**kwargs):
        kwargs.setdefault('context','spobs')
        super(SPDataSet,self).__init__(**kwargs)
        
    def load(self,force=True):
        """
        Load the contents of a spectrum file.
        
        Returns True or False if data was reloaded.
        """
        if self.has_qualifier('filename') and os.path.isfile(self.get_value('filename')):
            filename = self.get_value('filename')
            #-- check if the data is already in here, and only reload when
            #   it is not, or when force is True
            if not force and len(self[self['columns'][0]])>0:
                return False
            #-- read the comments: they can contain other qualifiers
            with open(filename,'r') as ff:
                for line in ff.readlines():
                    line = line.strip()
                    if line[0]=='#' and not line[:3]=='#//':
                        splitted = line[1:].split('=')
                        self[splitted[0].strip()] = splitted[1].strip()
            columns = self.get_value('columns')
            data = np.loadtxt(filename)
            N = len(columns)-2
            for i,col in enumerate(columns):
                if col=='time':
                    value = data[1:,0]
                elif col=='wavelength':
                    value = data[0,1::N]
                else:
                    index = columns.index(col)-1
                    value = data[1:,index::N]
                if not col in self:
                    self.add(dict(qualifier=col,value=value,description='<loaded from file {}>'.format(filename)))
                else:
                    self[col] = value
            #logger.info("Loaded contents of {} to spectrum".format(filename))
        elif self.has_qualifier('filename') and len(self[self['columns'][0]])==0:
            raise IOError("File {} does not exist".format(self.get_value('filename')))
        else:
            logger.info("No file to reload")
        return True
                
                
    def save(self,filename=None):
        """
        Save the contents of C{columns} to C{filename}.
        """
        #-- take 'filename' from the parameterset if nothing is given, but
        #   check if we have the permission to write the file.
        #if filename is None:# and not self.get_adjust('filename'):
            #filename = self['filename']
            #raise ValueError("Cannot overwrite file {} (safety is ON)".format(filename))
        if filename is None:
            filename = self['filename']
        if not filename:
            filename = self['label']
        columns = self['columns']
        #-- perhaps the user gave only one spectrum, and didn't give it as a
        #   list
        for col in columns:
            if col=='time': continue
            if not hasattr(self[col][0],'__len__'):
                self[col] = [self[col]]
        #-- if so, the user probably also forgot to give a time:
        if not len(self['time']):
            self['time'] = np.zeros(len(self['wavelength']))
        #-- if no continuum is give, assume the spectrum is normalised
        if not len(self['continuum']):
            self['continuum'] = np.ones_like(self['flux'])
        N = len(columns[2:])
        wavelength = np.array(self['wavelength'])[0] # we assume they're all the same
        line0 = np.column_stack(N*[wavelength]).ravel()
        out_data = [[np.array(self[col])[:,i] for col in columns[2:]] for i in range(len(self['flux'][0]))]
        out_data = np.array(out_data).ravel().reshape((N*len(self['flux'][0]),-1)).T
        out_data = np.vstack([line0,out_data])        
        times = np.hstack([np.nan,self['time']])
        out_data = np.column_stack([times,out_data])
        #-- before numpy version 1.7, we need to do some tricks:
        header = ''
        for qualifier in self:
            if not qualifier in columns:
                par = self.get_parameter(qualifier)
                header += '# {} = {}\n'.format(qualifier,par.as_string())
        np.savetxt(filename+'temp',out_data)
        with open(filename,'w') as ff:
            with open(filename+'temp','r') as gg:
                ff.write(header)
                ff.write("".join(gg.readlines()))
        os.unlink(filename+'temp')
        logger.info('Wrote file {} as spectrum ({}:{})'.format(filename,self.context,self['ref']))
    
    def estimate_noise(self,from_col='flux',to_col='sigma'):
        """
        Estimate the noise by differencing.
        """
        if not to_col in self['columns']:
            self['columns'].append(to_col)
        did_load = False
        if not len(self[from_col]):
            self.load()
            did_load = True
        for i in range(len(self[from_col])):
            noise = np.diff(self[from_col][i])/np.sqrt(2)
            self[to_col].append(np.ones(len(self[from_col][i]))*noise.std())#np.hstack([noise[0],noise]))
        if did_load:
            self.unload()
    
    def plot(self,ax=None,**plotoptions):
        """
        Plot spectra.
        """
        if ax is None:
            ax = plt.gca()
        loaded = self.load(force=False)
        if self.context[-3:]=='obs':
            plotoptions.setdefault('color','k')
            plotoptions.setdefault('linestyle','None')
            plotoptions.setdefault('marker','o')
        else:
            plotoptions.setdefault('color','r')
            plotoptions.setdefault('linestyle','-')
            plotoptions.setdefault('marker','None')
            plotoptions.setdefault('linewidth',2)
        wave = np.array(self['wavelength'])
        flux = np.array(self['flux'])/np.array(self['continuum'])
        
        if 'sigma' in self:
            e_flux = np.array(self['sigma'])
        else:
            e_flux = None
        #-- some magic to be sure of the shapes:
        wave = wave.ravel()
        fluxshape = flux.shape
        flux = flux.ravel()
        if e_flux is not None:
            e_flux = e_flux.ravel()
        if len(wave)<len(flux):
            flux = flux.reshape(fluxshape)[0]
        if e_flux is not None and len(wave)<len(e_flux):
            e_flux = e_flux.reshape(fluxshape)[0]
        #-- and then plot!
        if e_flux is not None:
            ax.errorbar(wave,flux,e_flux,**plotoptions)
        else:   
            ax.plot(wave,flux,**plotoptions)
        ax.set_xlim(wave.min(),wave.max())
        ax.set_xlabel('Wavelength [AA]')
        ax.set_ylabel("Normalised flux")
        if loaded:
            self.unload()


class IFDataSet(DataSet):
    def __init__(self,**kwargs):
        kwargs.setdefault('context','ifobs')
        super(IFDataSet,self).__init__(**kwargs)
    
