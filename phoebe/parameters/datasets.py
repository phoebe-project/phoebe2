"""
Classes and functions handling synthetic or observed data.

Introduction
============

The following datasets are defined:

.. autosummary::

    LCDataSet
    RVDataSet
    SPDataSet
    IFDataSet
    PLDataSet

The following ASCII parsers translate ASCII files to lists of DataSets.

.. autosummary::

    parse_lc
    parse_rv
    parse_phot
    parse_vis2
    parse_spec_as_lprof
    parse_plprof
    
These helper functions help parsing/processing header and file contents,
regardless of the type:

    parse_header
    process_header
    process_file


"""

import logging
import os
import glob
import numpy as np
from collections import OrderedDict
from phoebe.parameters import parameters
from phoebe.parameters import tools as ptools
from phoebe.units import conversions
from phoebe.units import constants
from phoebe.io import oifits
from phoebe.io import ascii
from phoebe.atmospheres import spectra
import matplotlib.pyplot as plt

logger = logging.getLogger("PARS.DATA")
logger.addHandler(logging.NullHandler())

#{
                    
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
    
    def __init__(self,*args,**kwargs):
        # if the user specified her/his own set of columns, we need to remove
        # the ones that are not given
        if 'columns' in kwargs:
            columns = kwargs.pop('columns')
        else:
            columns = None
        # normal init
        super(DataSet,self).__init__(*args,**kwargs)
        # now check the columns
        if columns is not None:
            default_columns = self['columns']
            for col in default_columns:
                if not col in columns and col in self:
                    thrash = self.pop(col)
            self['columns'] = columns
        
    def pop_column(self, qualifier):
        """
        Pop a column.
        """
        columns = self['columns']
        columns.remove(qualifier)
        self['columns'] = columns
        return super(DataSet,self).pop(qualifier)
        
    
    
    def load(self, force=True):
        """
        Load the contents of a data file.
        
        This should be more sofisticated. We could autodect columns/units and
        descriptions from the file header. We should also be able to handle
        FITS and ASCII. This load function should be different for spectra,
        light curve etc.
        """
        # File exists
        if self.has_qualifier('filename') and os.path.isfile(self.get_value('filename')):
            
            # First load data
            filename = self.get_value('filename')
            columns = self.get_value('columns')
            #-- check if the data is already in here, and only reload when
            #   it is not, or when force is True
            if not force and (self['columns'][0] in self and len(self[self['columns'][0]])>0):
                return False
            data_columns = np.loadtxt(filename).T
            for i,col in enumerate(data_columns):
                if not columns[i] in self:
                    self.add(dict(qualifier=columns[i],value=col,description='<loaded from file {}>'.format(filename)))
                else:
                    if 'user_units' in self and columns[i] in self['user_units']:
                        to_unit = self.get_parameter(columns[i]).get_unit()
                        from_unit = self['user_units'][columns[i]]
                        self[columns[i]] = conversions.convert(from_unit, to_unit, col)
                    else:
                        self[columns[i]] = col
           
            # Then try to load header:
            with open(filename, 'r') as ff:
                while True:
                    line = ff.readline()
                    
                    # End of file
                    if not line:
                        break
                   
                    # End of header
                    elif not line[0] == '#':
                        break
                   
                    # Separator
                    elif line[:4] == '#---':
                        continue
                    
                    # Some other line
                    elif not '=' in line:
                        continue
                    
                    key, val = line[1:].split('=')
                    key = key.strip()
                    val = val.strip()                    
                   
                    if key in self.keys():
                        self[key] = val
           
           
        # File does not exist
        elif self.has_qualifier('filename') and self['filename']:
            
            if not force and (self['columns'][0] in self and len(self[self['columns'][0]])>0):
                logger.debug("File {} does not exist, but observations are already loaded. Force reload won't work!".format(self.get_value('filename')))
                return False
            else:
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
                if col in self:
                    self[col] = []
    
    def save(self, filename=None, pretty_header=False):
        """
        Save the contents of C{columns} to C{filename}.
        
        We should offer the possibility here to write FITS files too.
        Because I'm astronomically lazy, it ain't gonna happen today!
        
        Later statement: perhaps no FITS option in this function, but add 
        a separate one. But again: it ain't gonna happen today!
        
        @param filename: filename or file writing stream
        @type filename: str or stream
        @param pretty_header: append a pretty header in the comments
        @type pretty_header: bool
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
        
        #-- prepare to write a pretty header.
        header = '#NAME '   
        #-- possibly we need to redefine the columns here:
        for col in self['columns']:
            if not len(self[col]):
                self[col] = np.zeros(len(self['time']))
            header += ' {} '.format(col)
        separator = '#' + '-'*(len(header)-1) + '\n'
        header = header + '\n' + separator
                
        #-- open the filename or the stream
        if isinstance(filename,str):
            ff = open(filename,'a')
        else:
            ff = filename
        
        #-- write the parameters which are not columns
        if pretty_header:
            ff.write(separator)
        
        for key in self:
            if not key in self['columns']:
                par = self.get_parameter(key)
                ff.write('# {:s} = {:s}\n'.format(par.get_qualifier(),par.to_str()))
        
        if pretty_header:
            ff.write(header)
        
        #-- write the data
        np.savetxt(ff,np.column_stack([self[col] for col in self['columns']]))
        
        #-- clean up
        if isinstance(filename,str):
            ff.close()
        
        
        logger.info('Wrote file {} with columns {} from dataset {}:{}'.format(filename,', '.join(self['columns']),self.context,self['ref']))
    
    def estimate_sigma(self, from_col='flux', to_col='sigma', force=True):
        """
        Estimate the sigma.
        """
        did_load = False
        
        if from_col is not None:
            if not len(self[from_col]):
                self.load()
                did_load = True
            sigma = np.diff(self[from_col])/np.sqrt(2)
            sigma = np.ones(len(self)) * np.std(sigma) 
        else:
            sigma = -1*np.ones(len(self))
        
        # contruct parameter if not present
        if not to_col in self:
            if 'sigma_'+str(from_col) in self:
                to_col = 'sigma_'+str(from_col)
            
            
            #else:
                
            #context = self.get_context()
            #par = parameters.Parameter(to_col, value=[-1]*len(self), context=context, frame='phoebe')
            #self.add(par)
        
        if to_col in self and (force or not len(self[to_col])==len(self)):
            # Fill with the value
            self[to_col] = sigma #np.hstack([noise[0],noise])
        
        # Make sure the column is in the columns
        #if not to_col in self['columns']:
        #    self['columns'].append(to_col)
        
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
            if col in ['time','weights','flag','wavelength','samprate','used_samprate']:
                continue
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
    
    def __getitem__(self, item):
        """
        Access information in a dataset.
        
        Suppose you initialized a dataset:
        
        >>> time = np.array([1,2,3.,4])
        >>> flux = np.array([-1,1,-2.,0])
        >>> lcobs = DataSet(context='lcobs', columns=['time','flux'], time=time, flux=flux, pblum=1.0, l3=0.35, ref='mylc')
        >>> print(lcobs)
          filename                            -- - phoebe Name of the file containing the data
               ref mylc                       --   phoebe Name of the data structure
              time [1.0 ... 4.0]              JD   phoebe Timepoints of the data
              flux [-1.0 ... 0.0]   erg/s/cm2/AA   phoebe Observed signal
            weight []                         --   phoebe Signal weight
        fittransfo linear                     --   phoebe Transform variable in fit
           columns ['time', 'flux']           --   phoebe Data columns
                l3 0.35                       -- - phoebe Third light or intercept constant
             pblum 1.0                        -- - phoebe Passband luminosity
        statweight 1.0                        -- - phoebe Statistical weight in overall fitting
        
        Then the different ways to access elements are:
        
        Access *header* information and *array* information
        
        >>> lcobs['l3']
        0.35
        >>> lcobs['flux']
        array([-1.,  1., -2.,  0.])
        
        Index and select rows in the arrays via index arrays, and return
        a copy of the whole dataset:
        
        >>> keep = lcobs['flux']>=0
        >>> lcobs2 = lcobs[keep]
        >>> lcobs2['flux']
        array([ 1.,  0.])
        >>> lcobs2['time']
        array([ 2.,  4.])
        
        Index and select rows in the arrays via slices:
        
        >>> lcobs2 = lcobs[::2]
        array([-1., -2.])

        """
        if np.isscalar(item):
            return super(DataSet,self).__getitem__(item)
        else:
            self_copy = self.copy()
            for col in self_copy['columns']:
                if col=='wavelength':
                    continue
                self_copy[col] = self_copy[col][item]
            return self_copy
    
    def take(self, index):
        """
        Take elements from the data arrays.
        
        This function does the same thing as fancy indexing (indexing arrays
        using arrays).
        """
        for col in self['columns']:
            self[col] = self[col][index]
    
    def overwrite_values_from(self, other_ds):
        """
        Overwrite parameters values in this dataset with those from another.
        
        @param other_ds: other dataset to take the values from
        @type other_ds: DataSet
        """
        for key in other_ds:
            self[key] = other_ds[key]
    
    def get_dtype(self):
        """
        Get the numpy data dtype for all the data columns.
        
        Example usage:
        
        >>> print(ds.dtype.names)
        ('time', 'flux', 'sigma')
        >>> print(ds['columns'])
        ['time', 'flux', 'sigma']
        """
        dtypes = np.dtype([(col,self[col].dtype) for col in self['columns']])
    
        return dtypes
    
    def get_shape(self):
        return (len(self),)    
    
    def __len__(self):
        # if there is time in this parameterSet, we'll take the length of that
        if 'time' in self['columns']:
            return len(self['time'])
        # otherwise we guess the firt column
        else:
            return len(self[self['columns'][0]])
    
    def asarray(self):
        self_copy = self.copy()
        for col in self['columns']:
            if col in self:
                self_copy[col] = np.array(self[col])
        return self_copy
    
    def clear(self):
        # Reset all parameter to the initial ones
        #self.reset()
        # But clear the results
        for col in self['columns']:
            self[col] = []
        
    
    
    dtype = property(get_dtype)
    shape = property(get_shape)
    
    
class LCDataSet(DataSet):
    """
    DataSet representing a light curve or photometry
    """
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
    
    def bin_oversampling(self):
        """
        Bin synthetics according to the desired oversampling rate.
        """
        new_flux = []
        new_time = []
        new_samprate = []
        used_samprate = []
        #used_exptime = []
        
        old_flux = np.array(self['flux'])
        old_time = np.array(self['time'])
        old_samprate = np.array(self['samprate'])
        #old_exptime = np.array(self['used_exptime'])
        sa = np.argsort(old_time)
        old_flux = old_flux[sa]
        old_time = old_time[sa]
        old_samprate = old_samprate[sa]
        #old_exptime = old_exptime[sa]
        
        
        seek = 0
        while seek<len(old_flux):
            samprate = old_samprate[seek]
            #exptime = old_exptime[seek]
            new_flux.append(np.mean(old_flux[seek:seek+samprate], axis=0))
            new_time.append(np.mean(old_time[seek:seek+samprate], axis=0))
            used_samprate.append(samprate)
            #used_exptime.append(exptime)
            new_samprate.append(1)
            seek += samprate
        self['flux'] = new_flux
        self['time'] = new_time
        self['samprate'] = new_samprate
        self['used_samprate'] = used_samprate
        #self['used_exptime'] = used_exptime
        #logger.info("Binned data according to oversampling rate")
    
        
class RVDataSet(DataSet):
    """
    DataSet reprensenting radial velocity measurements
    """
    def __init__(self,**kwargs):
        kwargs.setdefault('context','rvobs')
        super(RVDataSet,self).__init__(**kwargs)
    
    def bin_oversampling(self):
        return bin_oversampling(self, x='time', y='rv')

class SPDataSet(DataSet):
    """
    DataSet representing a spectrum
    
    There is not a one-to-one correspondence to load and unload, since we
    assume the wavelengths to be the same for each spectrum. What to do with
    this? If we allow them to be different, it becomes a pain in the to
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
            if not force and (self['columns'][0] in self and len(self[self['columns'][0]])>0):
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
            columns_without_time = columns[:]
            columns_without_time.pop(columns.index('time'))
            for i,col in enumerate(columns):
                if col=='time':
                    value = data[1:,0]
                elif col=='wavelength':
                    value = data[0,1::N]
                else:
                    index = columns_without_time.index(col)#-1
                    value = data[1:,index::N]
                if not col in self:
                    self.add(dict(qualifier=col,value=value,description='<loaded from file {}>'.format(filename)))
                else:
                    self[col] = value
            #logger.info("Loaded contents of {} to spectrum".format(filename))
        elif self.has_qualifier('filename') and self['filename']:
            
            if not force and (self['columns'][0] in self and len(self[self['columns'][0]])>0):
                logger.debug("File {} does not exist, but observations are already loaded. Force reload won't work!".format(self.get_value('filename')))
                return False
            else:
                raise IOError("File {} does not exist".format(self.get_value('filename')))
        elif self.has_qualifier('filename'):
            return False
        else:
            logger.info("No file to reload")
        return True
                
                
    def save(self,filename=None):
        """
        Save the contents of C{columns} to C{filename}.
        """
        # Take 'filename' from the parameterset if nothing is given, but check
        # if we have the permission to write the file.
        #if filename is None:# and not self.get_adjust('filename'):
            #filename = self['filename']
            #raise ValueError("Cannot overwrite file {} (safety is ON)".format(filename))
        
        if filename is None:
            filename = self['filename']
        if not filename:
            filename = self['label']
        columns = self['columns']
        
        # Perhaps the user gave only one spectrum, and didn't give it as a list
        for col in columns:
            if col == 'time':
                continue
            if not hasattr(self[col][0], '__len__'):
                self[col] = [self[col]]
                
        # If so, the user probably also forgot to give a time:
        if not len(self['time']):
            self['time'] = np.zeros(len(self['wavelength']))
        
        # If no continuum is give, assume the spectrum is normalised
        if not len(self['continuum']):
            self['continuum'] = np.ones_like(self['flux'])
        
        N = len(columns) - 2
        
        # We assume all wavelength arrays are the same
        wavelength = np.array(self['wavelength'])[0]
        line0 = np.column_stack(N*[wavelength]).ravel()
        cols_for_out_data = list(columns)
        
        if 'time' in cols_for_out_data:
            cols_for_out_data.remove('time')
        if 'wavelength' in cols_for_out_data:
            cols_for_out_data.remove('wavelength')
        
        out_data = [[np.array(self[col])[:,i] for col in cols_for_out_data] \
                                           for i in range(len(self['flux'][0]))]
                                       
        out_data = np.array(out_data).ravel().reshape((N*len(self['flux'][0]), -1)).T
        out_data = np.vstack([line0, out_data])        
        
        times = np.hstack([np.nan, self['time'].ravel()])
        out_data = np.column_stack([times, out_data])
        
        # Before numpy version 1.7, we need to do some tricks:
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
    
    def join(self,other_list):
        """
        Join with a list of other SPDataSets.
        
        Assumes wavelengths are the same.
        """
        self_loaded = self.load(force=False)
        
        for iother in other_list:
            loaded = iother.load(force=False)
            
            if not np.all(iother['wavelength'] == self['wavelength']):
                raise ValueError("DataSets with different wavelength arrays cannot be joinded")
            
            for col in self['columns']:
                if col == 'wavelength':
                    continue
                
                if len(self[col].shape) == 2:
                    self[col] = np.vstack([self[col], iother[col]])
                else:
                    self[col] = np.hstack([self[col], iother[col]])
            
            if loaded:
                iother.unload()
        
        self.save()
        
        if not self_loaded:
            self.unload()        
                    
                
    
    def estimate_sigma(self,from_col='flux',to_col='sigma', force=True):
        """
        Estimate the noise by differencing.
        """
        return None
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
    
    def bin_oversampling(self):
        """
        Bin synthetics according to the desired oversampling rate.
        """
        return None
        new_flux = []
        new_time = []
        new_cont = []
        new_samprate = []
        
        old_flux = np.array(self['flux'])
        old_cont = np.array(self['continuum'])
        old_time = np.array(self['time'])
        old_wavelength = np.array(self['wavelength'])
        old_samprate = np.array(self['samprate'])
        sa = np.argsort(old_time)
        old_flux = old_flux[sa]
        old_cont = old_cont[sa]
        old_time = old_time[sa]
        old_samprate = old_samprate[sa]
        old_wavelength = old_wavelength[sa]
        
        seek = 0
        while seek<len(old_flux):
            samprate = old_samprate[seek]
            new_flux.append(np.mean(old_flux[seek:seek+samprate]))
            new_time.append(np.mean(old_time[seek:seek+samprate]))
            new_cont.append(np.mean(old_cont[seek:seek+samprate]))
            new_samprate.append(1)
            seek += samprate
        self['flux'] = new_flux
        self['continuum'] = new_continuum
        self['wavelength'] = old_wavelength[seek]
        self['time'] = new_time
        self['samprate'] = new_samprate
        #logger.info("Binned data according to oversampling rate")
    
    def convert_to_fourier(self, freq=None):
        """
        Experimental feature. Really Experimental.
        """
        try:
            from ivs.timeseries import freqanalyse
        except ImportError:
            logger.warning("Can't do this yet, sorry")
            return None
        
        time = np.asarray(self['time'])
        wavelength = np.asarray(self['wavelength'])
        if not len(wavelength.shape)==2:
            wavelength = wavelength.reshape((1,-1))
        flux = np.asarray(self['flux'])
        cont = np.asarray(self['continuum'])
        flux = flux/cont
        
        # Make sure everything is interpolated onto the same wavelength grid
        if len(wavelength)>1:
            flux_ = np.zeros((len(time), len(wavelength[0])))
            for i in range(len(time)):
                flux_[i] = np.interp(wavelength[0], wavelength[i], flux[i])
            flux = flux_
                 
        T = time.ptp()
        nyquist = 0.5 / np.median(np.diff(time))
        if freq is None:
            f0 = 0.1/T
            fn = nyquist
            df = 0.1/T
        else:
            f0 = freq
            fn = freq
            df = 0
            
        output = freqanalyse.spectrum_2D(time, wavelength[0],
                                         flux, f0=f0, fn=fn, df=df, threads=6,
                                         method='scargle', subs_av=True,
                                         full_output=True, harmonics=0)
        
        const, e_const = output['pars']['const'], output['pars']['e_const']
        ampl, e_ampl = output['pars']['ampl'], output['pars']['e_ampl']
        freq, e_freq = output['pars']['freq'], output['pars']['e_freq']
        phase, e_phase = output['pars']['phase'], output['pars']['e_phase']
        
        self.add(parameters.Parameter(qualifier='const', value=const, cast_type=np.array))
        self.add(parameters.Parameter(qualifier='sigma_const', value=e_const, cast_type=np.array))
        self.add(parameters.Parameter(qualifier='ampl', value=ampl, cast_type=np.array))
        self.add(parameters.Parameter(qualifier='sigma_ampl', value=e_ampl, cast_type=np.array))
        self.add(parameters.Parameter(qualifier='freq', value=freq, cast_type=np.array))
        self.add(parameters.Parameter(qualifier='sigma_freq', value=e_freq, cast_type=np.array))
        self.add(parameters.Parameter(qualifier='phase', value=phase, cast_type=np.array))
        self.add(parameters.Parameter(qualifier='sigma_phase', value=e_phase, cast_type=np.array))
        self.add(parameters.Parameter(qualifier='avprof', value=output['avprof'], cast_type=np.array))
        self.add(parameters.Parameter(qualifier='model', value=output['model'], cast_type=np.array))
        
    def convert_to_moments(self, max_moment=3):
        """
        Convert spectral timeseries to their moments
        """
        
        time = np.asarray(self['time'])
        wavelength = np.asarray(self['wavelength'])
        if not len(wavelength.shape)==2:
            wavelength = wavelength.reshape((1,-1))
        flux = np.asarray(self['flux'])
        cont = np.asarray(self['continuum'])
        flux = flux/cont
        
        # Compute small unnormalised moments
        mymoments = np.zeros((len(time), max_moment+1, 2))
        wc = np.median(wavelength[0])
        SNR = 200.0
        for i in range(len(time)):
            velo = conversions.convert('nm', 'km/s', wavelength[i], wave=(wc,'nm'))
            moms, e_moms = spectra.moments(velo, flux[i], SNR, max_moment=max_moment)
            mymoments[i,:,0] = moms
            mymoments[i,:,1] = e_moms
        
        # observed unnormalised moments
        x0 = np.median(mymoments[:,1,0]/mymoments[:,0,0])
        for i in range(len(time)):
            velo = conversions.convert('nm', 'km/s', wavelength[i], wave=(wc,'nm'))
            moms, e_moms = spectra.moments(velo-x0, flux[i], SNR, max_moment=max_moment)
            mymoments[i,:,0] = moms
            mymoments[i,:,1] = e_moms
        
        # observed normalised moments
        mymoments[:,1:,0] = mymoments[:,1:,0]/mymoments[:,0,0][:,None]
        
        # add to parameterSet
        for i in range(max_moment+1):
            self.add(parameters.Parameter(qualifier='mom{}'.format(i), value=mymoments[:,i,0], cast_type=np.array))
            self.add(parameters.Parameter(qualifier='sigma_mom{}'.format(i), value=mymoments[:,i,1], cast_type=np.array))
        
        
        
        
        
        
    

class PLDataSet(SPDataSet):
    """
    DataSet representing spectropolarimetry
    """
    def __init__(self,**kwargs):
        kwargs.setdefault('context','plobs')
        super(SPDataSet,self).__init__(**kwargs)
    
    def longitudinal_field(self,vmin=-500,vmax=500):
        r"""
        Compute the longitudinal component of the magnetic field for each profile in the DataSet.
        
        See, e.g., [Donati1997]_:
        
        .. math::
            
            B_\ell = -\frac{1}{C\lambda_c^2 g_L c} \frac{\int \lambda V(\lambda)d\lambda}{\int (1-I(\lambda))d\lambda}
        
        with :math:`g_L` the average Lande factor and C the Lorentz unit, i.e.
        
        .. math::
        
            C = \frac{e}{4\pi m c^2} = 4.67 \times 10^{-13} \AA^{-1}
        
        
        The latter is really annoying because it is in Gaussian CGS units, and
        who uses that? Also, we actually need C in nm, and can then plug in
        the speed of light in cgs units. That seems to work out, but it's a
        little fuzzy to me... Anyway, we're only interested in the final value of
        :math:`B_\ell`.
        
        In velocity units (we'll use that):
        
        .. math::
        
            B_\ell = \frac{2.14\times 10^{11}}{\lambda_c g_L c} \frac{\int v V(v)dv}{\int (1-I(v))dv}
        
        with :math:`\lambda_c` in nm and c in km/s, then you'll get
        :math:`B_\ell` in Gauss.
        
        You can check the value of :math:`B_\ell` that you get by computing
        the flux-weighted line-of-sight component (:math:`B_z`) of the
        magnetic field:
        
        >>> Bz = -star.mesh['B_'][:,2]
        >>> weights = star.mesh['proj_ref'] * star.mesh['size']
        >>> B_ell = np.average(Bz,weights=weights)
        
        @return: longitudinal component of the magnetic field
        @rtype: array        
        """
        loaded = self.load(force=False)
        lams = np.array(self['wavelength'])
        cont = np.array(self['continuum'])
        flux = (np.array(self['flux'])/cont)
        V = (np.array(self['V'])/cont)
        glande = self.get('glande',1.2)
        clambda = lams.mean()
        #-- two ways to compute the longitudinal field: with and without
        #   error
        Bl = np.zeros(len(flux))
        e_Bl = np.zeros(len(flux))
        
        if len(lams.shape)==2:
            lams = lams[0]
        
        #-- convert to velocity
        v = conversions.convert('nm','km/s',lams,wave=(clambda,'nm'))
        factor = 2.14e11/(clambda*glande*constants.cc/1000.)
        
        #-- and cut arrays
        keep = (vmin<=v) & (v<=vmax)
        v = v[keep]
        
        if not "sigma_V" in self['columns']:
            for i,(iflux,iV) in enumerate(zip(flux,V)):
                iflux = iflux[keep]
                iV = iV[keep]
                numerator = np.trapz(iV*v,x=v)
                denominator = np.trapz((1-iflux),x=v)
                Bl[i] = factor * numerator/denominator
        #-- else integrate with errors
        else:
            sigma_V = np.array(self['sigma_V'])
            sigma = np.array(self['sigma'])
            for i,(iflux,iV,sig,sigV) in enumerate(zip(flux,V,sigma,sigma_V)):
                iV = iV[keep]
                sig = sig[keep]
                iflux = iflux[keep]
                sigV = sigV[keep]
                integrand1 = iV*v
                integrand2 = (1-iflux)
                numerator = np.trapz(integrand1,x=v)
                e_numerator = np.sqrt(np.sum((np.diff(v)*sigV[1:]*v[1:])**2))
                denominator = np.trapz(integrand2,x=v) 
                e_denominator = np.sqrt(np.sum((np.diff(v)*sig[1:])**2))
                Bl[i] = factor * numerator/denominator
                e_Bl[i] = Bl[i]*np.sqrt( (e_numerator/numerator)**2 + (e_denominator/denominator)**2)
        if loaded:
            self.unload()
        return Bl,e_Bl

class IFDataSet(DataSet):
    """
    DataSet representing interferometry
    """
    def __init__(self,**kwargs):
        kwargs.setdefault('context','ifobs')
        
        # Allow the user to give baselines and position angles, but convert
        # them immediately to u and v coordinates. If only baselines are given,
        # asume the position angles are zero.
        if 'baseline' in kwargs:
            baseline = kwargs.pop('baseline')
            posangle = kwargs.pop('posangle', np.zeros_like(baseline))
            kwargs['ucoord'] = baseline * np.cos(posangle)
            kwargs['vcoord'] = baseline * np.sin(posangle)
            logger.info("IFOBS: Converted baseline and position angle data to u and v coordinates")
        
        super(IFDataSet,self).__init__(**kwargs)
    
    
    def save2oifits(self,filename,like):
        """
        Save contents of an IFDataSet to an OIFITS file.
        """
        self.load()
        blank = np.zeros(len(self['time']))
        newobj = oifits.open(like)
        newobj.allvis2['mjd'] = self['time']
        newobj.allvis2['ucoord'] = self['ucoord']
        newobj.allvis2['vcoord'] = self['vcoord']
        newobj.allvis2['vis2data'] = np.array(self['vis'])**2
        #newobj.allvis2['vis2err'] = self['sigma_vis'] # wrong!
        newobj.save(filename)
#}

#{ ASCII parsers

def parse_header(filename,ext=None):
    """
    Parse only the header of an ASCII file.
    
    Parsing the headers means deriving the column descriptors in the file, and
    extracting keywords.
    
    A header is defined as that part of the ASCII file coming before the data,
    where the lines are preceded with comment character '#'.
    
    There are two parts in a header: an unprocessed part and a processed part.
    The processed part needs to be enclosed with separator "#---" (you can put
    in more dashes).
    
    The processed part can contain two types of information: information on
    parameters in pbdep and obs parameterSets, and description of the data
    columns. The column descriptors are preceded with a word in upper case
    letters, the parameters are in lower case letters and are followed by an
    equal sign ('=')::
    
        # This part of the header is completely ignored.
        # Completely, I tell you! You can write whatever
        # rubbish here. I'm an egg, I'm an egg, I tell you, 
        # I'm an egg!
        #--------------------------
        # atm = kurucz
        # passband = JOHNSON.B
        #NAME time flux sigma
        #UNIT d mag mag
        #TYPE f8 f8 f8
        #COMP Vega Vega Vega
        #--------------------------
        2345667. 19.  0.1
        2345668. 18.5 0.05
        
    Everything is optional, i.e. there can also be no header. You don't need to
    give keywords *and* column descriptors, and if you give column descriptors,
    you don't need to give them all.
        
    If there are data, then the first line is also read, to determine the
    number of columns present in the data.
    
    You can give a component's name also by supplying a line ``# label = mylabel``
    in the header. If column names are found, the output ``components`` will
    be a list of the same length, with all elements set to ``mylabel``. If
    there are no column names found, I cannot figure out how many columns there
    are, so only a string ``mylabel`` is returned, instead of a list.
    
    The type of data is determined via the keyword ``ext``. If it is not given,
    an attempt is made to determine the filetype from the extension.
        
    @param filename: input file
    @type filename: str
    @param ext: file type, one of ``lc``, ``phot``, ``rv``, ``spec``, ``lprof``, ``vis2``, ``plprof``
    @type ext: str
    @return: (columns, components, units, data types, ncol), (pbdep, dataset)
    @rtype: (list/None, list/str/None, dict/None, dict/None, int), (ParameterSet, DataSet)
    """
    
    # Create a default pbdep and DataSet
    contexts = dict(rv='rvdep',
                    phot='lcdep', lc='lcdep',
                    spec='spdep', lprof='spdep',
                    vis2='ifdep', plprof='pldep')
    dataset_classes = dict(rv=RVDataSet,
                           phot=LCDataSet, lc=LCDataSet,
                           spec=SPDataSet, lprof=SPDataSet,
                           vis2=IFDataSet, plprof=PLDataSet)
    
    # It is possible to automatically detect the type of data from the
    # extension
    ext = filename.split('.')[-1] if ext is None else ext
    if not ext in contexts:
        context = ext + 'dep'
    else:
        context = contexts[ext]
        
    if not ext in dataset_classes:
        dataset_class = DataSet
    else:
        dataset_class = dataset_classes[ext]
    
    pb = parameters.ParameterSet(context=context)
    ds = dataset_class(context=context[:-3]+'obs')
    
    # They belong together, so they should have the same reference
    ds['ref'] = pb['ref']
    
    # Open the file and start reading the lines
    n_columns = -1
    with open(filename, 'rb') as ff:
        
        # We can only avoid reading in the whole file by first going through
        # it line by line, and collect the comment lines. We need to be able
        # to look ahead to detect where the header ends.
        all_lines = []
        for line in ff.xreadlines():
            line = line.strip()
            if not line:
                continue
            elif line[0] == '#':
                
                # Break when we reached the end!
                all_lines.append(line[1:])
            
            # Perhaps the user did not give a '----', is this safe?
            elif n_columns < 0:
                # Count the number of columns by taking all the characters
                # before the comment character, stripping whitespace in the
                # beginning and at the end of the line, and then splitting by
                # whitespace.
                n_columns = len(line.split('#')[0].strip().split())
                break
                
    # Prepare some output and helper variables
    header_length = len(all_lines)
    components = None
    columns = None
    units = None
    dtypes = None
    
    # Now iterate over the header lines
    inside_col_descr = False
    col_descr = ['NAME', 'UNIT', 'COMP', 'TYPE']
    for iline, line in enumerate(all_lines):
        # Remove white space at the beginning of the line
        line = line.strip()
        
        # Toggle inside/outside col descr
        is_separator = line[:3] == '---'
        if is_separator:
            inside_col_descr = not inside_col_descr
            continue
        
        # Everything inside the column descripter part needs to be thoroughly
        # analysed
        if inside_col_descr:
            
            # Comment lines can contain qualifiers from the DataSet, we
            # recognise them by the presence of the equal "=" sign.
            split = line.split("=")
        
            # If they do, they consist of "qualifier = value", with "qualifier"
            # an all-lower-case string. Careful, perhaps there are more than 1
            # "=" signs in the value (e.g. the reference contains a "="). There
            # are never "=" signs in the qualifier.
            if len(split) > 1 and split[0] not in col_descr:
                
                # Qualifier is for sure the first element, remove whitespace
                # surrounding it
                qualifier = split[0].strip()
            
                # If this qualifier exists, in either the RVDataSet or pbdep,
                # add it. Text-to-value parsing is done at the ParameterSet
                # level, so we don't need to worry about it
                qualifier_consumed = False
                if qualifier in ds:
                    ds[qualifier] = "=".join(split[1:]).strip()
                    qualifier_consumed = True
                if qualifier in pb:
                    pb[qualifier] = "=".join(split[1:]).strip()
                    qualifier_consumed = True
                if qualifier == 'label':
                    components = "=".join(split[1:]).strip()
                    qualifier_consumed = True
                    
                if not qualifier_consumed:
                    raise ValueError("Cannot interpret parameter {}".format(qualifier))
            
            # It is also possible that the line contains the column names: they
            # should then contain at least the required columns! We recognise
            # the column headers as that line which is followed by a line
            # containing '#---' at least. Or the line after that one; in the
            # latter case, also the components are given
            elif line.split()[0] in col_descr:
                descr = line[:4]
                contents = line.split()[1:]
                if descr == 'NAME':
                    columns = contents
                elif descr == 'UNIT':
                    units = contents
                elif descr == 'COMP':
                    components = contents
                elif descr == 'TYPE':
                    dtypes = contents
                else:
                    raise ValueError("Unrecognized column descriptor {}".format(descr))
            
            # Else I don't know what to do!
            else:
                raise ValueError("I don't know what to do")
        
    # Some post processing to put stuff in the right format:
    if isinstance(components, str) and columns is not None:
        components = [components] * len(columns)
    if 'filename' in ds:
        ds['filename'] = filename
    if units is not None:
        units = {colname: unit for colname, unit in zip(columns, units)}
    if dtypes is not None:
        dtypes = {colname: dtype for colname, dtype in zip(columns, dtypes)}
    
    # That's it!
    return (columns, components, units, dtypes, n_columns), (pb,ds)    




def process_header(info, sets, default_column_order, required=2, columns=None,
                   components=None, dtypes=None, units=None):
    """
    Process header information.
    
    This assumes that the header is parsed, i.e. the first two arguments come
    directly from the output of :py:func:`parse_header`.
    
    Additional information is the default order of the columns, and the number
    of required columns.
    
    The user can override the auto-detection of ``columns`` and ``components``::
    
        >>> columns = ['rv', 'sigma', 'time']
        >>> components = ['compA', 'compA', None]
    
    The user can override the default data types (``dtypes``) of the columns, 
    and/or can specify custom ones for non-recognised columns. If the dtype
    of a column is not given, it needs to be a standard, recognised column::
    
        >>> dtypes = dict(flux=int, my_new_column=float)
    
    Finally, also units can be given. If not given, they are assumed to be the
    default ones. Otherwise, unit conversions are made to convert them to the
    given units to the default ones.::
    
        >>> units = dict(flux='W/m2/nm')
        
    @param info: information from :py:func:`parse_header`
    @type info: tuple
    @param sets: datasets from :py:func:`parse_header`
    @type sets: tuple
    @param default_column_order: list of default column names
    @type default_column_order: list of str
    @param required: number of required columns
    @type required: int
    @param columns: list of user-defined columns (overrides defaults)
    @type columns: list of str
    @param components: list of user-defined components (overrides defaults)
    @type components: list of str
    @param dtypes: data types of the columns (overrides defaults)
    @type dtypes: dict with keys one or more column names, and values a data type
    @param units: units of the columns (converts to defaults)
    @type units: dict with keys one or more column names, and values a unit string
    @return: (info on column names, dtypes and units), (obs and dep ParameterSet)
    @rtype: tuple, tuple
    """
    (columns_in_file, components_in_file, units_in_file, \
                                   dtypes_in_file, ncol), (pb, ds) = info, sets
    
    # Check minimum number of columns
    if ncol < required:
        raise ValueError(("You need to give "
                         "at least {} columns ({})").format(required,\
                                    ", ".join(default_column_order[:required])))
    
    # Set units as an empty dictionary by default, that's easier
    if units is None and units_in_file is not None:
        units = units_in_file
    elif units is None:
        units = dict()
    
    # Check if dtypes are given anywhere
    if dtypes is None and dtypes_in_file is not None:
        dtypes = dtypes_in_file
    
    # These are the default columns
    default_column_order = default_column_order[:ncol]
    
    # If the user did not give any columns, and there are no column
    # specifications, we need to assume that they are given in default order
    if columns is None and not columns_in_file:
        columns_in_file = default_column_order
        
    # If the user has given columns manually, we'll use those
    elif columns is not None:
        columns_in_file = columns
    
    # What data types are in the columns? We know about a few commonly used ones
    # a priori, but allow to user to add/override any via ``dtypes``.
    columns_required = default_column_order[:required]
    columns_specs = dict(time=float, flux=float, sigma=float, flag=float,
                         phase=float, rv=float, exptime=float, samprate=int,
                         ucoord=float, vcoord=float, sigma_vis2=float,
                         sigma_phase=float, eff_wave=float, vis2=float,
                         unit=str)
    
    if dtypes is not None:
        for col in dtypes:
            if not col in columns_specs:
                logger.warning(("Unrecognized column '{}' in file. Added but "
                                "ignored in the rest of the code").format(col))
            columns_specs[col] = dtypes[col]
    
    # Perhaps the user gave enough columns, but some of the required columns
    # are missing
    missing_columns = set(columns_required) - set(columns_in_file)
    
    # However, the user is allowed to give phases instead of time
    if 'time' in missing_columns and 'phase' in columns_in_file:
        logger.warning('You have given phased data')
        __ = missing_columns.remove('time')
    if 'flux' in missing_columns and 'mag' in columns_in_file:
        logger.warning(("You have given flux in magnitude, this will be "
            "converted to flux automatically"))
        index = columns_in_file.index('mag')
        columns_in_file[index] = 'flux'
        units['flux'] = 'mag'
        __ = missing_columns.remove('flux')
    
    # If there are still missing columns, report to the user.
    if len(missing_columns) > 0:
        raise ValueError(("Missing columns in "
                          "file: {}").format(", ".join(missing_columns)))
    
    # Alright, we now know all the columns: add them to the DataSet
    ds['columns'] = columns_in_file
    
    # Make sure to add unrecognized columns
    for col in columns_in_file:
        # Add a Parameter for column names that are missing
        if not col in ds:
            ds.add(dict(qualifier=col, value=[],
                        description='User defined column', cast_type=np.array))
        # And if by now we don't know the dtype, use a float
        if not col in columns_specs:
            columns_specs[col] = float
            
    return (columns_in_file, columns_specs, units), (pb, ds)


def process_file(filename, default_column_order, ext, columns, components,\
                 dtypes, units, **kwargs):
    """
    Process the basic parts of an input file.
    
    This shouldn't be used manually.
    """
    
    # Which columns are present in the input file, and which columns are
    # possible in the DataSet? The columns that go into the DataSet is the
    # intersection of the two. The columns that are in the file but not in the
    # DataSet are probably used for the pbdeps (e.g. passband) or for other
    # purposes (e.g. label or unit).
    #
    # Parse the header
    parsed = parse_header(filename, ext=ext)
    (columns_in_file, components_in_file, units_in_file, dtypes_in_file, ncol),\
                   (pb, ds) = parsed
    # Remember user-supplied arguments and keyword arguments for this parse
    # function
    # add columns, components, dtypes, units, full_output    
    ds['user_columns'] = columns
    ds['user_components'] = components
    ds['user_dtypes'] = dtypes
    ds['user_units'] = units
    
    # Process the header: get the dtypes (column_specs), figure out which
    # columns are actually in the file, which ones are missing etc...
    processed = process_header((columns_in_file,components_in_file,
                                units_in_file, dtypes_in_file, ncol),
                             (pb, ds), default_column_order, required=2, 
                             columns=columns, components=components,
                             dtypes=dtypes, units=units)
    (columns_in_file, columns_specs, units), (pb, ds) = processed
        
    # Prepare output dictionaries. The first level will be the label key
    # of the Body. The second level will be, for each Body, the pbdeps or
    # datasets.
    output = OrderedDict()
    ncol = len(columns_in_file)
    
    # Collect all data
    data = []

    # Open the file and start reading the lines: skip empty and comment lines
    with open(filename, 'rb') as ff:
        for line in ff.readlines():
            line = line.strip()
            if not line:
                continue
            if line[0] == '#':
                continue
            data.append(tuple(line.split()[:ncol]))

    # We have the information from header now, but only use that if it is not
    # overriden
    if components is None and components_in_file is None:
        components = ['__nolabel__'] * len(columns_in_file)
    elif components is None and isinstance(components_in_file, str):
        components = [components_in_file] * len(columns_in_file)
    elif components is None:
        components = components_in_file

    # Make sure all the components are strings
    components = [str(c) for c in components]
    
    # We need unique names for the columns in the record array
    columns_in_data = ["".join([col, name]) for col, name in \
                                         zip(columns_in_file, components)]
    
    # Add these to an existing dataset, or a new one. Also create pbdep to go
    # with it!
    # Numpy records to allow for arrays of mixed types. We do some numpy magic
    # here because we cannot efficiently predefine the length of the strings in
    # the file: therefore, we let numpy first cast everything to strings:
    data = np.core.records.fromrecords(data, names=columns_in_data)
    
    # And then say that it can keep those string arrays, but it needs to cast
    # everything else to the column specificer (i.e. the right type)
    descr = data.dtype.descr
    descr = [descr[i] if columns_specs[columns_in_file[i]] == str \
                          else (descr[i][0], columns_specs[columns_in_file[i]])\
                              for i in range(len(descr))]
    dtype = np.dtype(descr)
    data = np.array(data, dtype=dtype)
    
    # For each component, create two lists to contain the LCDataSets or pbdeps    
    for label in set(components):
        if label.lower() == 'none':
            continue
        output[label] = [[ds.copy()], [pb.copy()]]
    
    for col, coldat, label in zip(columns_in_file, columns_in_data, components):
        if label.lower() == 'none':
            # Add each column
            for lbl in output:
                output[lbl][0][-1][col] = data[coldat]
            continue
        
        output[label][0][-1][col] = data[coldat]
        
        # Override values already there with extra kwarg values, for both
        # obs and pbdep parameterSets.
        for key in kwargs:
            if key in output[label][0][-1]:
                output[label][0][-1][key] = kwargs[key]
            if key in output[label][1][-1]:
                output[label][1][-1][key] = kwargs[key]
    
    # lastly, loop over all data columns with units, and convert them
    for unit in units:
        if units[unit].lower() in ['none','mag']:
            continue
        for label in output:
            from_units = units[unit]
            to_units = output[label][0][-1].get_parameter(unit).get_unit()
            output[label][0][-1][unit] = conversions.convert(from_units,
                                  to_units, output[label][0][-1][unit])
    
    return output, (columns_in_file, columns_specs, units), (pb, ds)


def parse_lc(filename, columns=None, components=None, dtypes=None, units=None,
             full_output=False, **kwargs):
    """
    Parse LC files to LCDataSets and lcdeps.
    
    **File format description**
    
        1. Simple text files
        2. Advanced text files
    
    *Simple text files*
    
    The minimal structure of an LC file is::
    
        2455453.0       1.   
        2455453.1       1.01 
        2455453.2       1.02 
        2455453.3       0.96 
    
    Which can be read in with one of the following equivalent lines of code::
    
    >>> obs, pbdep = parse_lc('myfile.lc')
    >>> obs, pbdep = parse_lc('myfile.lc', columns=['time', 'flux'])
    
    Extra columns may be given. They can be given either in the default order:
    
        1. Time (``time``)
        2. Flux (``flux``)
        3. Uncertainty (``sigma``)
        4. Flag (``flag``)
        5. Exposure time (``exptime``)
        6. Sampling rate (``samprate``)
        
    Or the order can be specified with the ``columns`` keyword. In the later
    case, you can e.g. give exposure times without giving a flag column.
    
    In the case phases are given instead of time, you need to specify the order
    of the columns, and set ``phase`` as the label instead of ``time``. E.g. a
    file with contents::
    
        0.0       1.   
        0.1       1.01 
        0.2       1.02 
        0.3       0.96 
        
    should not be read in without extra keyword arguments, since you need to
    specify that the first column contains phases rather than time::
    
    >>> obs, pbdep = parse_lc('myfile.lc', columns=['phase', 'flux'])
    
    Additionally, you can substitute ``mag`` for ``flux``. In contrast to the
    ``phase`` array, which will be retained as is in the observational
    ParameterSet (and converted to time during computations only), the ``mag``
    column will be converted to fluxes during parsing::
    
    >>> obs, pbdep = parse_lc('myfile.lc', columns=['time', 'mag'])
    
    But the original contents may always be retrieved via::
    
    >>> mymag = obs.get_value('flux', 'mag')
    
    You can comment out data lines if you don't want to use them.
    
    *Advanced text files*
    
    The information on the order and names of the columns, as well as
    additional information on the light curve, can be given inside the
    text file in the form of comments. An example advanced LC file is::
        
        # This comment is ignored
        #----------------------
        # passband = JOHNSON.V
        # atm = kurucz
        # ref = april2011
        #NAME time mag sigma
        #----------------------
        2455453.0       1.     0.01    
        2455453.1       1.01   0.015    
        2455453.2       1.02   0.011    
        2455453.3       0.96   0.009    
    
    Which can be read in with one of the following equivalent lines of code::
    
    >>> obs, pbdep = parse_lc('myfile.lc')
    >>> obs, pbdep = parse_lc('myfile.lc', columns=['time', 'mag', 'sigma'], passband='JOHNSON.V', atm='kurucz', ref='april2011')
    
    An attempt will be made to interpret the comment lines (i.e. those lines
    where the first character equals '#') inside the ``#---`` separators as
    qualifier/value for either the :ref:`lcobs <parlabel-phoebe-lcobs>` or
    :ref:`lcdep <parlabel-phoebe-lcdep>`. Upon failure, a ``ValueError`` is
    raised.
    
    
    **Uncertainties**
    
    If no ``sigma`` column is given, a column will be added with all-equal
    values.
    
    **Units**
    
    If you want to pass flux units different from the default ones
    (``erg/s/cm2/AA``), you need to specify those via the ``units`` keyword, or
    specify them in the column descriptors.
    
    Assume the following file with times and magnitudes instead fluxes::
    
        0.0     1.               
        0.1     1.01              
        0.2     1.02              
        0.3     0.96          
    
    You need to read this in via::
    
        >>> obs, pbdep = parse_lc('myfile.lc', units=dict(flux='mag'))
    
    Or specify the units in the file::
    
        #---------------------------
        #UNIT d  mag     mag 
        #---------------------------
        0.0     1.       0.1    
        0.1     1.01     0.2         
        0.2     1.02     0.1         
        0.3     0.96     0.3
        
    
    Though for your convenience, we also recognize the column name ``mag``,
    which will be internally converted to fluxes. A file::
    
        #---------------------------
        #NAME time  mag     sigma 
        #---------------------------
        0.0     1.       0.1    
        0.1     1.01     0.2         
        0.2     1.02     0.1         
        0.3     0.96     0.3
        
    Can be read in via::
    
        >>> obs, pbdep = parse_lc('myfile.lc')
        >>> print(obs[0]['flux'])
        [  3.98107171e-08   3.94457302e-08   3.90840896e-08   4.13047502e-08]
        >>> print(obs[0]['sigma'])
        [  3.66670255e-09   7.26617203e-09   3.59977768e-09   1.14129242e-08]
    
    You can retrieve the original fluxes again via::
    
        >>> print(obs[0].get_value('flux','mag'))
        [1.0 1.01 1.02 0.96]
        
    But for the uncertainties, you need to remember that they are relative
    with respect to the fluxes, so you need to be careful:
    
        >>> mag, e_mag = phoebe.convert('erg/s/cm2/AA', 'mag', obs[0]['flux'], obs[0]['sigma'])
        >>> print(e_mag)
        [ 0.1  0.2  0.1  0.3]
    
    **Input and output**
    
    The output can be readily appended to the C{obs} and C{pbdep} keywords in
    upon initialization of a ``Body``.
    
    If C{full_output=True}, you will get a consistent output, regardless what
    the input file looks like. This can be useful for automatic parsing. In
    this case, the output is an OrderedDict, with the keys at the first level
    the key of the component (if no labels are given, this will be C{__nolabel__}).
    The value for each key is a list of two lists: the first list contains the
    LCDataSets, the second list the corresponding pbdeps.
    
    **Example usage**
    
    Assume the first example in the *Advanced test file section* is saved to
    a file called ``myfile.lc``. Then you can do (the following lines are
    equivalent):
    
    >>> obs, pbdep = parse_lc('myfile.lc')
    >>> obs, pbdep = parse_lc('myfile.lc', columns=['time', 'flux', 'sigma'])
    
    Which is in this case equivalent to:
    
    >>> output = parse_lc('myfile.lc', full_output=True)
    >>> obs,pbdeps = output['__nolabel__']
    
    or 
    
    >>> obs,pbdeps = output.values()[0]
    
    The output can then be given to any Body:
    
    >>> starpars = parameters.ParameterSet(context='star')
    >>> meshpars = parameters.ParameterSet(context='mesh:marching')
    >>> star = Star(starpars, mesh=meshpars, pbdep=[pbdeps], obs=[obs])
    
    The first example explicitly contains a label, so an OrderedDict will
    always be returned, regardless the value of ``full_output``.
    
    @param filename: filename
    @type filename: string
    @param columns: columns in the file. If not given, they will be automatically detected or should be the default ones.
    @type columns: None or list of strings
    @param components: list of components for each column in the file. If not given, they will be automatically detected.
    @type components: None or list of strings
    @param dtypes: data types, with dictionary keys the column names, and value the numpy dtype
    @type dtypes: dictionary
    @param units: unit strings, with dictionary keys the column names, and value the numpy dtype
    @type units: dictionary
    @param full_output: if False and there are no labels in the file, only the data from the first component will be returned, instead of the OrderedDict
    @type full_output: bool
    @return: :ref:`lcobs <parlabel-phoebe-lcobs>`, :ref:`lcdep <parlabel-phoebe-lcdep>`) or OrderedDict with the keys the labels of the objects, and then the lists of lcobs and lcdeps.
    """
    default_column_order = ['time', 'flux', 'sigma', 'flag', 'exptime',
                            'samprate']
    
    # Process the header and body of the file
    output, \
        (columns_in_file, columns_specs, units), \
        (pb, ds) = process_file(filename, default_column_order, 'lc', columns, \
                                            components, dtypes, units, **kwargs)
            
    # Add sigma if not available:
    myds = output.values()[0][0][0]
    mypb = output.values()[0][1][0]
    if not 'sigma' in myds['columns']:
        myds.estimate_sigma(from_col='flux', to_col='sigma')
        logger.info("Obs {}: sigma estimated (not available)".format(myds['ref']))
    
    # Convert to right units
    for col in units:
        if col == 'flux':
            unit_from = units[col]
            unit_to = myds.get_parameter(col).get_unit()
            
            if unit_from != unit_to:
                passband = mypb['passband']
                logger.info("Obs {}: flux and sigma converted from {} to {} ({})".format(myds['ref'], unit_from, unit_to, passband))
                f, e_f = conversions.convert(unit_from, unit_to,
                                         myds['flux'], myds['sigma'], passband=passband)
                myds['flux'] = f
                myds['sigma'] = e_f
    
    # If the user didn't provide any labels (either as an argument or in the
    # file), we don't bother the user with it:
    if not full_output:
        return output.values()[0][0][0], output.values()[0][1][0]
    else:
        return output



def parse_rv(filename, columns=None, components=None,
             full_output=False, dtypes=None, units=None, **kwargs):
    """
    Parse RV files to RVDataSets and rvdeps.
    
    See :py:func:`parse_lc` for more information.
    
    @param filename: filename
    @type filename: string
    @param columns: columns in the file. If not given, they will be
      automatically detected or should be the default ones.
    @type columns: None or list of strings
    @param components: list of components for each column in the file. If not
      given, they will be automatically detected.
    @type components: None or list of strings
    @param full_output: if False and there are no labels in the file, only the
      data from the first component will be returned, instead of the OrderedDict
    @type full_output: bool
    @return: (list of :ref:`rvobs <parlabel-phoebe-rvobs>`, list of :ref:`rvdep <parlabel-phoebe-rvdep>`) or OrderedDict with the keys the labels of the objects, and then the lists of rvobs and rvdeps.
    """
    default_column_order = ['time','rv','sigma', 'flag', 'exptime', 'samprate']
    
    # Process the header and body of the file
    output, \
        (columns_in_file, columns_specs, units), \
        (pb, ds) = process_file(filename, default_column_order, 'rv', columns, \
                                            components, dtypes, units, **kwargs)
    
    # Add sigma if not available:
    myds = output.values()[0][0][-1]
    if not 'sigma' in myds['columns']:
        myds.estimate_sigma(from_col='rv', to_col='sigma')
    
    # Convert to right units
    for col in units:
        if col == 'rv':
            # if sigma units and normal units are not the same, we need to
            # probably first convert rv to sigma units, then rv&sigma to correct
            # units, just to be safe (probably more important for mag-flux
            # conversions)
            f, e_f = myds['rv'], myds['sigma']
            if units[col] == 'Rsol/d' and myds.get_parameter(col).get_unit()=='km/s':
                f = f*8.04986111111111
                e_f = e_f*8.04986111111111
            elif units[col] != myds.get_parameter(col).get_unit():
                f, e_f = conversions.convert(units[col],
                                         myds.get_parameter(col).get_unit(), 
                                         myds['rv'], myds['sigma'])
            
            myds['rv'] = f
            myds['sigma'] = e_f
    
    # Remove nans:
    for comp in output.keys():
        for ds in output[comp][0]:
            columns = ds['columns']
            keep = np.ones(len(ds[columns[0]]), bool)
            # First look for nans in all columns
            for col in columns:
                keep = keep & -np.isnan(ds[col])
            # Then throw the rows with nans out
            for col in columns:
                ds[col] = ds[col][keep]
    
            # Sort according to time
            ds = ds[np.argsort(ds['time'])]
    
    
    # Remember user-supplied arguments and keyword arguments for this parse
    # function
    # add columns, components, dtypes, units, full_output
    
    # NOT IMPLEMENTED YET
    
    # If the user didn't provide any labels (either as an argument or in the
    # file), we don't bother the user with it:
    if not full_output:
        return output.values()[0][0][0], output.values()[0][1][0]
    else:
        return output
        
def parse_etv(filename, columns=None, components=None,
             full_output=False, dtypes=None, units=None, **kwargs):
    """
    Parse ETV files to ETVDataSets and etvdeps.
    
    See :py:func:`parse_lc` for more information.
    
    @param filename: filename
    @type filename: string
    @param columns: columns in the file. If not given, they will be
      automatically detected or should be the default ones.
    @type columns: None or list of strings
    @param components: list of components for each column in the file. If not
      given, they will be automatically detected.
    @type components: None or list of strings
    @param full_output: if False and there are no labels in the file, only the
      data from the first component will be returned, instead of the OrderedDict
    @type full_output: bool
    @return: (list of :ref:`etvobs <parlabel-phoebe-etvobs>`, list of :ref:`etvdep <parlabel-phoebe-etvdep>`) or OrderedDict with the keys the labels of the objects, and then the lists of etvobs and etvdeps.
    """
    default_column_order = ['time','etv','sigma']
    
    # Process the header and body of the file
    output, \
        (columns_in_file, columns_specs, units), \
        (pb, ds) = process_file(filename, default_column_order, 'etv', columns, \
                                            components, dtypes, units, **kwargs)
    
    
    # Add sigma if not available:
    myds = output.values()[0][0][-1]
    if not 'sigma' in myds['columns']:
        myds.estimate_sigma(from_col='etv', to_col='sigma')
    
    # Convert to right units
    for col in units:
        if col == 'etv':
            f, e_f = conversions.convert(units[col],
                                         myds.get_parameter(col).get_unit(), 
                                         myds['etv'], myds['sigma'])
            myds['etv'] = f
            myds['sigma'] = e_f
    
    # Remove nans:
    for comp in output.keys():
        for ds in output[comp][0]:
            columns = ds['columns']
            keep = np.ones(len(ds[columns[0]]), bool)
            # First look for nans in all columns
            for col in columns:
                keep = keep & -np.isnan(ds[col])
            # Then throw the rows with nans out
            for col in columns:
                ds[col] = ds[col][keep]
    
    # Remember user-supplied arguments and keyword arguments for this parse
    # function
    # add columns, components, dtypes, units, full_output
    
    # NOT IMPLEMENTED YET
    
    # If the user didn't provide any labels (either as an argument or in the
    # file), we don't bother the user with it:
    if not full_output:
        return output.values()[0][0][0], output.values()[0][1][0]
    else:
        return output

def parse_phot(filenames, columns=None, full_output=False, group=None,
               group_kwargs=None, **kwargs):
    """
    Parse PHOT files to LCDataSets and lcdeps.
    
    **File format description**
    
    The generic structure of a PHOT file is::
        
        # atm = kurucz
        # fittransfo = log
        STROMGREN.U    7.43   0.01   mag   0.
        STROMGREN.B    7.13   0.02   mag   0.
        GENEVA.U       7.2    0.001  mag   0.
        IRAC.45        2.     0.1    Jy    0.
    
    An attempt will be made to interpret the comment lines (i.e. those lines
    where the first character equals '#') as qualifier/value for either the
    :ref:`lcobs <parlabel-phoebe-lcobs>` or :ref:`lcdep <parlabel-phoebe-lcdep>`.
    Failures are silently ignored, so you are allowed to put whatever comments
    in there (though with caution), or comment out data lines if you don't
    want to use them. The comments are optional. So this is also allowed::
    
        STROMGREN.U    7.43   0.01   mag   0.
        STROMGREN.B    7.13   0.02   mag   0.
        GENEVA.U       7.2    0.001  mag   0.
        IRAC.45        2.     0.1    Jy    0.
        
    The only way in which you are allowed to deviate from this structure, is
    by specifying column names, followed by a comment line of dashes (there
    needs to be at least one dash, no spaces)::
    
        # atm = kurucz
        # fittransfo = log
        # flux     passband    time  sigma  unit
        #---------------------------------------
        7.43       STROMGREN.U 0.     0.01  mag
        7.13       STROMGREN.B 0.     0.02  mag
        7.2        GENEVA.U    0.     0.001 mag
        2.         IRAC.45     0.     0.1   Jy
    
    In the latter case, you are allowed to omit any column except for ``flux``
    ``sigma`` and ``passband``, which are required. If not given, the default
    ``unit`` is ``erg/s/cm2/AA`` and ``time`` is zero.
        
    Finally, there is also an option to merge data of different Bodies in
    one PHOT file. In that case, an extra column is required containing the
    label of the Body where the data needs to be attached to::
    
        # atm = kurucz
        # fittransfo = log
        # flux     passband    time  sigma  unit label
        #---------------------------------------------
        7.43       STROMGREN.U 0.     0.01  mag  starA
        7.13       STROMGREN.B 0.     0.02  mag  starA
        7.2        GENEVA.U    0.     0.001 mag  starA
        2.         IRAC.45     0.     0.1   Jy   starB
    
    .. warning::
    
       You are not allowed to have missing values. Each column must
       have a value for every observation.
    
    **Input and output**
    
    The output can be readily appended to the C{obs} and C{pbdep} keywords in
    upon initialization of a ``Body``.
    
    Extra keyword arguments are passed to output LCDataSets or pbdeps,
    wherever they exist and override the contents of the comment lines in the
    phot file.
    
    If C{full_output=True}, you will get a consistent output, regardless what
    the input file looks like. This can be useful for automatic parsing. In
    this case, the output is an OrderedDict, with the keys at the first level
    the key of the component (if no labels are given, this will be C{__nolabel__}).
    The value for each key is a list of two lists: the first list contains the
    LCDataSets, the second list the corresponding pbdeps.
    
    If C{full_output=False}, you will get the same output as described in the
    previous paragraph only if labels are given in the file. Else, the output
    consists of only one component, which is probably a bit confusing for the
    user (at least me). So if there are no labels given and C{full_output=False},
    the two lists are immediately returned.
    
    **Grouping**
    
    You can :py:func:`group <phoebe.parameters.tools.group>` observations, e.g.
    such to :py:func:`adjust the distance to scale the SED <phoebe.backend.processing.sed_scale_to_distance>`.
    
    **Example usage**
    
    Assume that any of the first three examples is saved in 
    file called ``myfile.phot``, you can do (the following lines are equivalent):
    
    >>> obs,pbdeps = parse_phot('myfile.phot')
    >>> obs,pbdeps = parse_phot('myfile.phot',columns=['passband','flux','sigma','unit','time'])
    
    Which is in this case equivalent to:
    
    >>> output = parse_phot('myfile.phot',full_output=True)
    >>> obs,pbdeps = output['__nolabel__']
    
    or 
    
    >>> obs,pbdeps = output.values()[0]
    
    The output can then be given to any Body:
    
    >>> starpars = parameters.ParameterSet(context='star')
    >>> meshpars = parameters.ParameterSet(context='mesh:marching')
    >>> star = Star(starpars,mesh=meshpars,pbdep=pbdeps,obs=obs)
    
    The last example contains labels, so the full output is always given.
    Assume the contents of the last file is stored in ``myfile2.phot``:
    
    >>> output = parse_phot('myfile2.phot')
    >>> obs1,pbdeps1 = output['starA']
    >>> obs2,pbdeps2 = output['starB']
    
    @param filenames: list of filename or a filename glob pattern. If you give a list of filenames, they need to all have the same structure!
    @type filenames: list or string
    @param columns: columns in the file. If not given, they will be automatically detected or should be the default ones.
    @type columns: None or list of strings
    @param full_output: if False and there are no labels in the file, only the data from the first component will be returned, instead of the OrderedDict
    @type full_output: bool
    @return: (list of :ref:`lcobs <parlabel-phoebe-lcobs>`, list of :ref:`lcdep <parlabel-phoebe-lcdep>`) or OrderedDict with the keys the labels of the objects, and then the lists of lcobs and lcdeps.
    """
    #-- which columns are present in the input file, and which columns are
    #   possible in the LCDataSet? The columns that go into the LCDataSet
    #   is the intersection of the two. The columns that are in the file but
    #   not in the LCDataSet are probably used for the pbdeps (e.g. passband)
    #   or for other purposes (e.g. label or unit).
    if columns is None:
        columns_in_file = ['passband','flux','sigma','unit','time']
    else:
        columns_in_file = columns
    columns_required = ['passband','flux','sigma']
    columns_specs = dict(passband=str,flux=float,sigma=float,time=float,unit=str,label=str)
    
    missing_columns = set(columns_required) - set(columns_in_file)
    if len(missing_columns)>0:
        raise ValueError("Missing columns in PHOT file: {}".format(", ".join(missing_columns)))
    
    #-- prepare output dictionaries. The first level will be the label key
    #   of the Body. The second level will be, for each Body, the pbdeps or
    #   datasets.
    components = OrderedDict()
    ncol = len(columns_in_file)
    
    #-- you are allowed to give a list of filenames. If you give a string,
    #   it will be interpreted as a glob pattern
    if not isinstance(filenames,list):
        filenames = sorted(glob.glob(filenames))
    
    if not len(filenames):
        raise IOError("PHOT file does not exist")
    
    for filename in filenames:
        #-- create a default LCDataSet and pbdep
        ds = LCDataSet(columns=['time','flux','sigma'])
        pb = parameters.ParameterSet(context='lcdep')
        #-- collect all data
        data = []
        #-- open the file and start reading the lines
        with open(filename,'rb') as ff:
            all_lines = ff.readlines()
            for iline,line in enumerate(all_lines):
                line = line.strip()
                if not line: continue
                #-- coment lines: can contain qualifiers from the LCDataSet,
                #   we recognise them by the presence of the equal "=" sign.
                if line[0]=='#':
                    split = line[1:].split("=")
                    # if they do, they consist of "qualifier = value". Careful,
                    # perhaps there are more than 1 "=" signs in the value
                    # (e.g. the reference contains a "="). There are never
                    # "=" signs in the qualifier.
                    if len(split)>1:
                        #-- qualifier is for sure the first element
                        qualifier = split[0].strip()
                        #-- if this qualifier exists, in either the LCDataSet
                        #   or pbdep, add it. Text-to-value parsing is done
                        #   at the ParameterSet level
                        if qualifier in ds:
                            ds[qualifier] = "=".join(split[1:]).strip()
                        if qualifier in pb:
                            pb[qualifier] = "=".join(split[1:]).strip()
                    #-- it is also possible that the line contains the column
                    #   names: they should then contain at least the required
                    #   columns! We recognise the column headers as that line
                    #   which is followed by a line containing '#-' at least.
                    #   We cannot safely automatically detect headers otherwise
                    #   since it does not necessarily need to be the last
                    #   comment line, since also the first (or any) data point
                    #   can be commented out. We don't look for columns if
                    #   they are explicitly given.
                    elif columns is None:
                        if len(all_lines)>(iline+1) and all_lines[iline+1][:2]=='#-':
                            columns_in_file = line[1:].split()
                            ncol = len(columns_in_file)
                            logger.info("Auto detecting columns in PHOT {}: {}".format(filename,", ".join(columns_in_file)))
                            missing_columns = set(columns_required) - set(columns_in_file)
                            if len(missing_columns)>0:
                                raise ValueError("Missing columns in PHOT file: {}".format(", ".join(missing_columns)))\
                #-- data lines:
                else:
                    data.append(tuple(line.split()[:ncol]))
            #-- add these to an existing dataset, or a new one.
            #   also create pbdep to go with it!
            #-- numpy records to allow for arrays of mixed types. We do some
            #   numpy magic here because we cannot efficiently predefine the
            #   length of the strings in the file: therefore, we let numpy
            #   first cast everything to strings:
            data = np.core.records.fromrecords(data,names=columns_in_file)
            #-- and then say that it can keep those string arrays, but it needs
            #   to cast everything else to the column specificer (i.e. the right type)
            descr = [idescr if columns_specs[idescr[0]]==str else (idescr[0],columns_specs[idescr[0]]) for idescr in data.dtype.descr]
            dtype = np.dtype(descr)
            data = np.array(data,dtype=dtype)
            #-- some columns were not required, they will be created
            #   with default values:
            auto_columns = []
            auto_columns_names = []
            if not 'time' in columns_in_file:
                auto_columns_names.append('time')
                auto_columns.append(np.zeros(len(data)))
            if not 'unit' in columns_in_file:
                auto_columns_names.append('unit')
                auto_columns.append(np.array(['erg/s/cm2/AA']*len(data)))
            if not 'label' in columns_in_file:
                auto_columns_names.append('label')
                auto_columns.append(np.array(['__nolabel__']*len(data)))
            if len(auto_columns):
                data = plt.mlab.rec_append_fields(data,auto_columns_names,auto_columns)
            #-- now, make sure each flux value is in the right units:
            flux,sigma = conversions.nconvert(data['unit'],'erg/s/cm2/AA',data['flux'],data['sigma'],passband=data['passband'])
            data['flux'] = flux
            data['sigma'] = sigma
            #-- now create datasets for each component (be sure to add them
            #   in the order they appear in the file
            labels_indexes = np.unique(data['label'],return_index=True)[1]
            labels = [data['label'][index] for index in sorted(labels_indexes)]
            for label in labels:
                #-- for each component, select those entries that go along
                #   with it
                selection = data[data['label']==label]
                passbands_indexes = np.unique(selection['passband'],return_index=True)[1]
                passbands = [selection['passband'][index] for index in sorted(passbands_indexes)]
                #-- for each component, create two lists to contain the
                #   LCDataSets or pbdeps
                components[label] = [[],[]]
                #-- for each component, make separate datasets for each
                #   passband, and add that dataset to the master OrderedDicts
                for passband in passbands:
                    subselection = selection[selection['passband']==passband]
                    ref = "{}-{}".format(passband,filename)
                    components[label][0].append(ds.copy())
                    components[label][1].append(pb.copy())
                    #-- fill in the actual data
                    for col in ds['columns']:
                        components[label][0][-1][col] = subselection[col] 
                    #-- fill in other keys from the parameterSets
                    components[label][0][-1]['ref'] = ref
                    components[label][1][-1]['ref'] = ref
                    components[label][1][-1]['passband'] = passband
                    #-- override values already there with extra kwarg values
                    for key in kwargs:
                        if key in components[label][0][-1]:
                            components[label][0][-1][key] = kwargs[key]
                        if key in components[label][1][-1]:
                            components[label][1][-1][key] = kwargs[key]
    
    if group is not None:
        if group_kwargs is None:
            group_kwargs = dict()
        ptools.group(components.values()[0][0], group, **group_kwargs)
    
    #-- If the user didn't provide any labels (either as an argument or in the
    #   file), we don't bother the user with it:
    if not 'label' in columns_in_file and not full_output:
        return components.values()[0]
    else:
        return components
            

def parse_spec_timeseries(timefile, clambda=None, columns=None,
                          components=None, dtypes=None, units=None,
                          full_output=False, fixed_wavelength=True, **kwargs):
    """
    Parse a timeseries of spectrum files.
    """
    default_column_order = ['wavelength', 'flux', 'sigma', 'continuum']
    
    # read in the time file:
    timedata = np.loadtxt(timefile, dtype=str, usecols=(0,1), unpack=True)
    basedir = os.path.dirname(timefile)
    filenames = [os.path.join(basedir, filename) for filename in timedata[0]]
    time = np.array(timedata[1],float)
    
    # Process the header and body of the file
    for i, filename in enumerate(filenames):
        output_, \
        (columns_in_file, columns_specs, units), \
        (pb, ds) = process_file(filename, default_column_order, 'lprof', columns, \
                                            components, dtypes, units, **kwargs)
        
        if i==0:
            output = output_
            myds = output.values()[0][0][-1]
            myds['filename'] = timefile
        else:
            for col in columns_in_file:
                if col != 'wavelength' or not fixed_wavelength:
                    myds[col] = np.vstack([myds[col], output_.values()[0][0][-1][col]])
        
    # Convert to right units
    for i, col in enumerate(myds['columns']):
        if units is None or i>=len(units):
            continue
            
        if col == 'wavelength':
            if conversions.get_type(units[i])=='velocity' and clambda is None:
                raise ValueError(("If the wavelength is given in velocity "
                                  "units, you need to specify the central "
                                  "wavelength 'clambda'"))
            myds['wavelength'] = conversions.convert(units[i],
                                         myds.get_parameter(col).get_unit(), 
                                         myds['wavelength'], wave=clambda)
    
    # Add sigma if not available
    if not 'sigma' in myds['columns']:
        myds.estimate_sigma(from_col='flux', to_col='sigma')
        logger.info("No uncertainties available in data --> estimated")
    
    # Add continuum if not available
    if not 'continuum' in myds['columns']:
        myds['continuum'] = np.ones_like(myds['wavelength'])
        myds['columns'] = myds['columns'] + ['sigma']
    
    
    # Add time
    myds['time'] = time
    myds['columns'] = ['time'] + myds['columns']
    
    
    # If the user didn't provide any labels (either as an argument or in the
    # file), we don't bother the user with it:
    if not full_output:
        return output.values()[0][0][0], output.values()[0][1][0]
    else:
        return output

def parse_spec_as_lprof(filename, clambda=None, wrange=None, time=0.0, 
                        columns=None, components=None, dtypes=None, units=None,
                        full_output=False, **kwargs):
    """
    Parse a SPEC file as an LPROF file to an SPDataSet and a spdep.
    
    This effectively extracts a line profile from a full spectrum, or reads in
    a full spectrum.
    
    The structure of a SPEC file is::
    
        3697.204  5.3284e-01
        3697.227  2.8641e-01
        3697.251  2.1201e-01
        3697.274  2.7707e-01
        
    @param filenames: list of filename or a filename glob pattern
    @type filenames: list or string
    @param clambda: central wavelength of profile (AA)
    @type clambda: float
    @param wrange: entire wavelength range (AA)
    @type wrange: float
    @return: list of :ref:`lcobs <parlabel-phoebe-lcobs>`, list of :ref:`lcdep <parlabel-phoebe-lcdep>`
    """
    default_column_order = ['wavelength', 'flux', 'sigma', 'continuum']
    
    # Process the header and body of the file
    output, \
        (columns_in_file, columns_specs, units), \
        (pb, ds) = process_file(filename, default_column_order, 'lprof', columns, \
                                            components, dtypes, units, **kwargs)
    
    myds = output.values()[0][0][-1]
    
    # Correct columns to be a list of arrays instead of arrays
    for col in myds['columns']:
        myds[col] = np.array([myds[col]])
    
    # Convert to right units
    for col in units:
        if col == 'wavelength':
            myds['wavelength'] = conversions.convert(units[col],
                                         myds.get_parameter(col).get_unit(), 
                                         myds['wavelength'], wave=clambda)
    
    # Cut out the line profile
    if clambda is not None and wrange is not None:
        clambda = conversions.convert(clambda[1],\
                        myds.get_parameter('wavelength').get_unit(), clambda[0])
        wrange = conversions.convert(wrange[1],\
                        myds.get_parameter('wavelength').get_unit(), wrange[0])
        
        keep = np.abs(myds['wavelength'][0]-clambda) < (wrange/2.0)
        for col in myds['columns']:
            myds[col] = myds[col][:,keep]
    
    
    # Add sigma if not available
    if not 'sigma' in myds['columns']:
        myds.estimate_sigma(from_col='flux', to_col='sigma')
        logger.info("No uncertainties available in data --> estimated")
    
    # Add continuum if not available
    if not 'continuum' in myds['columns']:
        myds['continuum'] = np.ones_like(myds['wavelength'])
        myds['columns'] = myds['columns'] + ['sigma']
    
    
    # Add time
    myds['time'] = np.array([time])
    myds['columns'] = ['time'] + myds['columns']
    
    
    # If the user didn't provide any labels (either as an argument or in the
    # file), we don't bother the user with it:
    if not full_output:
        return output.values()[0][0][0], output.values()[0][1][0]
    else:
        return output
    

    
def parse_vis2(filename, columns=None, components=None, dtypes=None, units=None,
               full_output=False, **kwargs):
    """
    Parse VIS2 files to IFDataSets and ifdeps.
    
    **File format description**
    
        1. Simple text files
        2. Advanced text files
    
    *Simple text files*
    
    The minimal structure of a VIS2 file is::
        
        -7.779     7.980  0.932 0.047   56295.0550
        -14.185    0.440  0.808 0.040   56295.0550
        -29.093  -15.734  0.358 0.018   56295.0551
         -6.406   -7.546  0.957 0.048   56295.0552
        -21.314  -23.720  0.598 0.030   56295.0534
    
    The order of the columns is:
    
        1. :math:`u`-coordinate (``ucoord``)
        2. :math:`v`-coordinate (``vcoord``)
        3. Squared visibility (``vis2``)
        4. Uncertainty on squared visibility (``sigma_vis2``)
        5. Time (``time``)
    
    An attempt will be made to interpret the comment lines (i.e. those lines
    where the first character equals '#') as qualifier/value for either the
    :ref:`ifobs <parlabel-phoebe-ifobs>` or :ref:`ifdep <parlabel-phoebe-ifdep>`.    
    
    .. warning::
    
       You are not allowed to have missing values. Each column must
       have a value for every observation.
    
    See :py:func:`parse_lc` for more information.
    
    @param filename: filename
    @type filename: string
    @param columns: columns in the file. If not given, they will be automatically detected or should be the default ones.
    @type columns: None or list of strings
    @param components: list of components for each column in the file. If not given, they will be automatically detected.
    @type components: None or list of strings
    @param dtypes: data types, with dictionary keys the column names, and value the numpy dtype
    @type dtypes: dictionary
    @param units: unit strings, with dictionary keys the column names, and value the numpy dtype
    @type units: dictionary
    @param full_output: if False and there are no labels in the file, only the data from the first component will be returned, instead of the OrderedDict
    @type full_output: bool
    @return: :ref:`ifobs <parlabel-phoebe-ifobs>`, :ref:`ifdep <parlabel-phoebe-ifdep>`) or OrderedDict with the keys the labels of the objects, and then the lists of ifobs and ifdeps.
    """
    default_column_order = ['ucoord', 'vcoord', 'vis2', 'sigma_vis2', 'time']
    
    # Which columns are present in the input file, and which columns are
    # possible in the IFDataSet? The columns that go into the IFDataSet
    # is the intersection of the two. The columns that are in the file but
    # not in the IFDataSet are probably used for the pbdeps (e.g. passband)
    # or for other purposes (e.g. label or unit).
    # Parse the header
    parsed = parse_header(filename, ext='vis2')
    (columns_in_file, components_in_file, units_in_file, dtypes_in_file, ncol),\
                   (pb, ds) = parsed
    
    # Remember user-supplied arguments and keyword arguments for this parse
    # function
    # add columns, components, dtypes, units, full_output
    
    ds['user_columns'] = columns
    ds['user_components'] = components
    ds['user_dtypes'] = dtypes
    ds['user_units'] = units
    
    # Process the header: get the dtypes (column_specs), figure out which
    # columns are actually in the file, which ones are missing etc...
    processed = process_header((columns_in_file,components_in_file,
                                units_in_file, dtypes_in_file, ncol),
                             (pb, ds), default_column_order, required=2, 
                             columns=columns, components=components,
                             dtypes=dtypes, units=units)
    (columns_in_file, columns_specs, units), (pb, ds) = processed
        
    # Prepare output dictionaries. The first level will be the label key
    # of the Body. The second level will be, for each Body, the pbdeps or
    # datasets.
    output = OrderedDict()
    ncol = len(columns_in_file)
    
    # Collect all data
    data = []

    # Open the file and start reading the lines: skip empty and comment lines
    with open(filename, 'rb') as ff:
        for line in ff.readlines():
            line = line.strip()
            if not line:
                continue
            if line[0] == '#':
                continue
            data.append(tuple(line.split()[:ncol]))

    # We have the information from header now, but only use that if it is not
    # overriden
    if components is None and components_in_file is None:
        components = ['__nolabel__'] * len(columns_in_file)
    elif components is None and isinstance(components_in_file, str):
        components = [components_in_file] * len(columns_in_file)
    elif components is None:
        components = components_in_file

    # Make sure all the components are strings
    components = [str(c) for c in components]
    
    # We need unique names for the columns in the record array
    columns_in_data = ["".join([col, name]) for col, name in \
                                         zip(columns_in_file, components)]
    
    # Add these to an existing dataset, or a new one. Also create pbdep to go
    # with it!
    # Numpy records to allow for arrays of mixed types. We do some numpy magic
    # here because we cannot efficiently predefine the length of the strings in
    # the file: therefore, we let numpy first cast everything to strings:
    data = np.core.records.fromrecords(data, names=columns_in_data)
    
    # And then say that it can keep those string arrays, but it needs to cast
    # everything else to the column specificer (i.e. the right type)
    descr = data.dtype.descr
    descr = [descr[i] if columns_specs[columns_in_file[i]] == str \
                          else (descr[i][0], columns_specs[columns_in_file[i]])\
                              for i in range(len(descr))]
    dtype = np.dtype(descr)
    data = np.array(data, dtype=dtype)
    
    # For each component, create two lists to contain the LCDataSets or pbdeps    
    for label in set(components):
        if label.lower() == 'none':
            continue
        output[label] = [[ds.copy()], [pb.copy()]]
    for col, coldat, label in zip(columns_in_file, columns_in_data, components):
        if label.lower() == 'none':
            # Add each column
            for lbl in output:
                output[lbl][0][-1][col] = data[coldat]
            continue
        
        output[label][0][-1][col] = data[coldat]
        
        # Override values already there with extra kwarg values, for both
        # obs and pbdep parameterSets.
        for key in kwargs:
            if key in output[label][0][-1]:
                output[label][0][-1][key] = kwargs[key]
            if key in output[label][1][-1]:
                output[label][1][-1][key] = kwargs[key]
                
    # If the user didn't provide any labels (either as an argument or in the
    # file), we don't bother the user with it:
    if not full_output:
        return output.values()[0][0][0], output.values()[0][1][0]
    else:
        return output
        
    

def parse_plprof(filename, clambda=None, wrange=None, time=0.0, columns=None,
                 components=None, dtypes=None, units=None, full_output=False,
                 **kwargs):
    """
    Parse a PLPROF file to an PLDataSet and a pldep.
    
    @param filenames: list of filename or a filename glob pattern
    @type filenames: list or string
    @param clambda: central wavelength of profile (AA)
    @type clambda: float
    @return: list of :ref:`plobs <parlabel-phoebe-plobs>`, list of :ref:`pldep <parlabel-phoebe-pldep>`
    """   
    default_column_order = ['wavelength', 'flux', 'sigma',
                            'V', 'sigma_V', 'Q', 'sigma_Q', 'U', 'sigma_U']
    
    # Process the header and body of the file
    output, \
        (columns_in_file, columns_specs, units), \
        (pb, ds) = process_file(filename, default_column_order, 'plprof', columns, \
                                            components, dtypes, units, **kwargs)
    
    myds = output.values()[0][0][-1]
    
    # Correct columns to be a list of arrays instead of arrays
    for col in myds['columns']:
        myds[col] = np.array([myds[col]])
    
    # Convert to right units
    for col in units:
        if col == 'wavelength':
            try:
                myds['wavelength'] = conversions.convert(units[col],
                                         myds.get_parameter(col).get_unit(), 
                                         myds['wavelength'], wave=clambda)
            except TypeError:
                raise TypeError(("Cannot convert wavelength column from {} to {}. "
                                 "Perhaps missing 'clambda'?").format(units[col],
                                 myds.get_parameter(col).get_unit()))
    
    # Cut out the line profile
    if clambda is not None and wrange is not None:
        clambda = conversions.convert(clambda[1],\
                        myds.get_parameter('wavelength').get_unit(), clambda[0])
        wrange = conversions.convert(wrange[1],\
                        myds.get_parameter('wavelength').get_unit(), wrange[0])
        
        keep = np.abs(myds['wavelength'][0]-clambda) < (wrange/2.0)
        for col in myds['columns']:
            myds[col] = myds[col][:,keep]
    
    
    # Add sigma if not available
    if not 'sigma' in myds['columns']:
        myds.estimate_sigma(from_col='flux', to_col='sigma')
        logger.info("No uncertainties available in data --> estimated")
    
    # Add continuum if not available
    if not 'continuum' in myds['columns']:
        myds['continuum'] = np.ones_like(myds['wavelength'])
        myds['columns'] = myds['columns'] + ['continuum']
    
    # Add time
    myds['time'] = np.array([time])
    myds['columns'] = ['time'] + myds['columns']
    
    # Change filename (makes sure the original file is not overwritten on a
    # "save" call)
    myds['filename'] = myds['filename'] + '.plobs'
    
    # If the user didn't provide any labels (either as an argument or in the
    # file), we don't bother the user with it:
    if not full_output:
        return output.values()[0][0][0], output.values()[0][1][0]
    else:
        return output
   

def parse_lsd_as_plprof(filename,columns=None,components=None,full_output=False,skiprows=0,**kwargs):
    """
    Parse an LSD file to an PLDataSet and a pldep.
    
    The structure of an LSD file is::
    
        # atm = kurucz
        # ld_func = claret
        -141.9  0.995078    0.000767859 2.591441e-05    6.5304e-05  1.295886e-05    6.5304e-05
        -140.1  0.995345    0.000767691 1.547061e-05    6.52591e-05 6.91076e-06 6.52591e-05
        -138.3  0.995209    0.000767578 2.679006e-05    6.5256e-05  7.54846e-06 6.52559e-05
    
    An attempt will be made to interpret the comment lines as qualifier/value
    for either the :ref:`plobs <parlabel-phoebe-plobs>` or :ref:`pldep <parlabel-phoebe-pldep>`.
    Failures are silently ignored, so you are allowed to put whatever comments
    in there (though with caution), or comment out data lines if you don't
    want to use them.
    
    The output can be readily appended to the C{obs} and C{pbdep} keywords in
    upon initialization of a ``Body``.
    
    Extra keyword arguments are passed to output PLDataSets or pldeps,
    wherever they exist and override the contents of the comment lines in the
    phot file.
    
    Example usage:
    
    >>> obs,pbdeps = parse_plprof('myfile.plprof')
    >>> starpars = parameters.ParameterSet(context='star')
    >>> meshpars = parameters.ParameterSet(context='mesh:marching')
    >>> star = Star(starpars,mesh=meshpars,pbdep=pbdeps,obs=obs)
    
    @param filenames: list of filename or a filename glob pattern
    @type filenames: list or string
    @param clambda: central wavelength of profile (AA)
    @type clambda: float
    @return: list of :ref:`plobs <parlabel-phoebe-plobs>`, list of :ref:`pldep <parlabel-phoebe-pldep>`
    """
    #-- parse the header
    (columns_in_file,components_in_file),(pb,ds) = parse_header(filename,ext='plprof')

    if columns is None and columns_in_file is None:
        columns_in_file = ['velocity','flux']
    elif columns is not None:
        columns_in_file = columns
    columns_required = ['velocity','flux']
    columns_specs = dict(time=float,wavelength=float,flux=float,sigma=float,
                                                     V=float,sigma_V=float,
                                                     Q=float,sigma_Q=float,
                                                     U=float,sigma_U=float)
    
    missing_columns = set(columns_required) - set(columns_in_file)
    if len(missing_columns)>0:
        raise ValueError("Missing columns in PLPROF file: {}".format(", ".join(missing_columns)))
    
    if 'velocity' in columns_in_file:
        index = columns_in_file.index('velocity')
        columns_in_file[index] = 'wavelength'
    
    #-- prepare output dictionaries. The first level will be the label key
    #   of the Body. The second level will be, for each Body, the pbdeps or
    #   datasets.
    output = OrderedDict()
    ncol = len(columns_in_file)
    
    #-- collect all data
    data = []
    #-- read the comments of the file
    with open(filename,'rb') as ff:
        for linenr,line in enumerate(ff.readlines()):
            if linenr<skiprows: continue
            line = line.strip()
            if not line: continue
            if line[0]=='#': continue
            data.append(tuple(line.split()[:ncol]))
    #-- we have the information from header now, but only use that
    #   if it is not overriden
    if components is None and components_in_file is None:
        components = ['__nolabel__']*len(columns_in_file)
    elif components is None and isinstance(components_in_file,str):
        components = [components_in_file]*len(columns_in_file)
    elif components is None:
        components = components_in_file
    #-- make sure all the components are strings
    components = [str(c) for c in components]
    #-- we need unique names for the columns in the record array
    columns_in_data = ["".join([col,name]) for col,name in zip(columns_in_file,components)]
    #-- add these to an existing dataset, or a new one.
    #   also create pbdep to go with it!
    #-- numpy records to allow for arrays of mixed types. We do some
    #   numpy magic here because we cannot efficiently predefine the
    #   length of the strings in the file: therefore, we let numpy
    #   first cast everything to strings:
    data = np.core.records.fromrecords(data,names=columns_in_data)
    #-- and then say that it can keep those string arrays, but it needs
    #   to cast everything else to the column specificer (i.e. the right type)
    descr = data.dtype.descr
    descr = [descr[i] if columns_specs[columns_in_file[i]]==str else (descr[i][0],columns_specs[columns_in_file[i]]) for i in range(len(descr))]
    dtype = np.dtype(descr)
    data = np.array(data,dtype=dtype)
    
    #-- for each component, create two lists to contain the
    #   SPDataSets or pbdeps    
    for label in set(components):
        if label.lower()=='none':
            continue
        output[label] = [[ds.copy()],[pb.copy()]]
    for col,coldat,label in zip(columns_in_file,columns_in_data,components):
        if label.lower()=='none':
            for lbl in output:
                output[lbl][0][-1][col] = data[coldat]
            continue
        output[label][0][-1][col] = data[coldat]
        #-- override values already there with extra kwarg values
        for key in kwargs:
            if key in output[label][0][-1]:
                output[label][0][-1][key] = kwargs[key]
            if key in output[label][1][-1]:
                output[label][1][-1][key] = kwargs[key]
    
    for label in output.keys():
        for ds,pb in zip(*output[label]):
            #-- save to the dataset
            ref = kwargs.get('ref',"{}".format(filename))
            ds['continuum'] = np.ones(len(ds['wavelength'])).reshape((1,-1))
            ds['wavelength'] = conversions.convert('km/s','nm',ds['wavelength'],wave=(4000.,'AA')).reshape((1,-1))
            ds['flux'] = ds['flux'].reshape((1,-1))
            ds['sigma'] = ds['sigma'].reshape((1,-1))
            ds['ref'] = ref
            #ds.estimate_sigma()
            ds['filename'] = ref+'.plobs'
            ds['columns'] = columns_in_file + ['continuum']
            if not 'time' in columns_in_file:
                ds['time'] = np.zeros(len(ds['flux']))
            if not 'time' in ds['columns']:
                ds['columns'] = ds['columns'] + ['time']
            for stokes_type in ['V','Q','U']:
                if 'sigma_{}'.format(stokes_type) in columns_in_file:
                    ds['sigma_{}'.format(stokes_type)] = ds['sigma_{}'.format(stokes_type)].reshape((1,-1))
                if stokes_type in columns_in_file:
                    ds[stokes_type] = ds[stokes_type].reshape((1,-1))
            pb['ref'] = ref
            #-- override with extra kwarg values
            for key in kwargs:
                if key in pb: pb[key] = kwargs[key]
                if key in ds: ds[key] = kwargs[key]
            ds.save()
            ds.unload()
            
    #-- If the user didn't provide any labels (either as an argument or in the
    #   file), we don't bother the user with it:
    if '__nolabel__' in output and not full_output:
        return output.values()[0]
    else:
        return output
    

#}

#{ High-end wrappers

def hstack(tup):
    """
    Stack datasets in sequence horizontally (column wise).
    
    Take a sequence of datasets and stack them horizontally to make
    a single array.
    
    Parameters
    ----------
    tup : sequence of datasets
        All datasets must have the same shape
    
    Returns
    -------
    stacked : dataset
        The dataset formed by stacking the given datasets.
    """
    out = tup[0].copy()
    for col in out['columns']:
        out[col] = np.hstack([out[col]] + [dset[col] for dset in tup[1:]])
    return out


def oifits2vis2(filename,wtol=1.,ttol=1e-6,**kwargs):
    """
    Convert an OIFITS file to Phoebe's VIS2 format.
    
    @param filename: OIFITS file location
    @type filename: str
    @return: VIS file location
    @rtype: str
    """
    #-- prepare output contexts
    ref = os.path.splitext(filename)[0]
    ifmdep = parameters.ParameterSet(context='ifdep',ref=ref)
    ifmobs = IFDataSet(context='ifobs',ref=ref,columns=['ucoord','vcoord','vis2','sigma_vis2','time'])
    for key in kwargs:
        if key in ifmdep:
            ifmdep[key] = kwargs[key]
        if key in ifmobs:
            ifmobs[key] = kwargs[key]
    #-- read in the visibilities
    templatedata = oifits.open(filename)
    allvis2 = templatedata.allvis2 
    vis = allvis2['vis2data']
    vis_sigma = allvis2['vis2err']
    ucoord = allvis2['ucoord']
    vcoord = allvis2['vcoord']
    time = allvis2['mjd']
    eff_wave = conversions.convert('m','AA',allvis2['eff_wave'])
    
    skip_columns = ifmobs['columns']
    all_keys = list(set(list(ifmobs.keys())) | set(list(ifmdep.keys())))
    all_keys = [key for key in all_keys if not key in skip_columns]
    if 'filename' in all_keys:
        all_keys.pop(all_keys.index('filename'))
    comments = ['# {} = {}'.format(key,ifmdep[key]) if key in ifmdep else '# {} = {}'.format(key,ifmobs[key]) for key in all_keys]
    comments+= ['# ucoord vcoord vis2 sigma_vis2 time eff_wave']
    comments+= ['#------------------------------------------']
    output_filename = os.path.splitext(filename)[0]+'.vis2'
    ascii.write_array(np.column_stack([ucoord,vcoord,vis,vis_sigma,time,eff_wave]),
              output_filename,comments=comments)
    return output_filename


def esprit2plprof(filename,**kwargs):
    """
    Convert an ESPRIT file to Phoebe's LPROF format.
    
    @param fileanem: ESPRIT file location
    @type filename: str
    """
    clambda = kwargs.get('clambda',5000.)
    if not hasattr(clambda,'__iter__'):
        clambda = (clambda,'AA')
    #-- prepare output contexts
    ref = os.path.splitext(filename)[0]
    pldep = parameters.ParameterSet(context='pldep',ref=ref)
    plobs = PLDataSet(context='plobs',ref=ref,columns=['wavelength','flux','sigma','V','sigma_V','continuum'])
    for key in kwargs:
        if key in pldep:
            pldep[key] = kwargs[key]
        if key in plobs:
            plobs[key] = kwargs[key]
    #-- read in the Stokes profiles
    data = np.loadtxt(filename,skiprows=2).T
    wavelength = conversions.convert('km/s','AA',data[0],wave=clambda)
    
    skip_columns = plobs['columns']
    all_keys = list(set(list(plobs.keys())) | set(list(pldep.keys())))
    all_keys = [key for key in all_keys if not key in skip_columns]
    if 'filename' in all_keys:
        all_keys.pop(all_keys.index('filename'))
    comments = ['# {} = {}'.format(key,pldep[key]) if key in pldep else '# {} = {}'.format(key,plobs[key]) for key in all_keys]
    comments+= ['# wavelength flux sigma V sigma_V']
    comments+= ['#-------------------------------']
    output_filename = os.path.splitext(filename)[0]+'.plprof'
    ascii.write_array(np.column_stack([wavelength,data[1],data[2],data[3],data[4]]),
              output_filename,comments=comments)
    return output_filename
    
    
    
def bin_oversampling(system, x='time', y='flux'):
    """
    Bin synthetics according to the desired oversampling rate.
    """
    new_flux = []
    new_time = []
    new_samprate = []
    used_samprate = []
    #used_exptime = []
    
    old_flux = np.array(system[y])
    old_time = np.array(system[x])
    old_samprate = np.array(system['samprate'])
        
    #old_exptime = np.array(self['used_exptime'])
    sa = np.argsort(old_time)
    old_flux = old_flux[sa]
    old_time = old_time[sa]
    old_samprate = old_samprate[sa]
    #old_exptime = old_exptime[sa]
    
    
    seek = 0
    while seek<len(old_flux):
        samprate = old_samprate[seek]
        #exptime = old_exptime[seek]
        new_flux.append(np.mean(old_flux[seek:seek+samprate], axis=0))
        new_time.append(np.mean(old_time[seek:seek+samprate], axis=0))
        used_samprate.append(samprate)
        #used_exptime.append(exptime)
        new_samprate.append(1)
        seek += samprate
    system[y] = new_flux
    system['time'] = new_time
    system['samprate'] = new_samprate
    system['used_samprate'] = used_samprate
    #self['used_exptime'] = used_exptime
    #logger.info("Binned data according to oversampling rate")
    


#}
