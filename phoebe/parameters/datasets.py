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

The following ASCII parsers translate ASCII files to lists of DataSets. The
string following ``parse_`` in the name of the function refers to the extension
the file should have.

.. autosummary::

    parse_rv
    parse_phot
    parse_vis
    parse_spec_as_lprof

"""

import logging
import os
import glob
import numpy as np
from collections import OrderedDict
from phoebe.parameters import parameters
from phoebe.units import conversions
from phoebe.io import oifits
from phoebe.io import ascii
import matplotlib.pyplot as plt

logger = logging.getLogger("PARS.DATA")

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
                if not col in columns:
                    thrash = self.pop(col)
            self['columns'] = columns
    
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
            if not force and (self['columns'][0] in self and len(self[self['columns'][0]])>0):
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
        
class RVDataSet(DataSet):
    """
    DataSet reprensenting radial velocity measurements
    """
    def __init__(self,**kwargs):
        kwargs.setdefault('context','rvobs')
        super(RVDataSet,self).__init__(**kwargs)

class SPDataSet(DataSet):
    """
    DataSet representing a spectrum
    
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
            raise IOError("File {} does not exist or no calculations were performed".format(self.get_value('filename')))
        else:
            logger.info("No file to reload")
            return False
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
        cols_for_out_data = list(columns)
        if 'time' in cols_for_out_data: cols_for_out_data.remove('time')
        if 'wavelength' in cols_for_out_data: cols_for_out_data.remove('wavelength')
        out_data = [[np.array(self[col])[:,i] for col in cols_for_out_data] for i in range(len(self['flux'][0]))]
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

class PLDataSet(SPDataSet):
    pass


class IFDataSet(DataSet):
    """
    DataSet representing interferometry
    """
    def __init__(self,**kwargs):
        kwargs.setdefault('context','ifobs')
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

def parse_header(filename):
    """
    Parse only the header of an ASCII file.
    
    The columns and components are returned as well as a pbdep and dataset
    where the keywords in the header are parsed.
    
    If there are no columns defined, ``columns`` will be ``None``. If there
    are no components defined, ``components`` will be ``None``.
    
    You cannot define components without defining the columns, but you can
    define columns without defining components.
    
    When only columns are defined, they need to be in the second-to-last
    line, followed by a separator ``#---``.
    
    When both columns and components are defined, the columns need to be in
    the third-to-last line, the components in the second-to-last, followed
    by a separator ``#---``.
    
    After the separator, it is assumed the data begin.
    
    You can give a component's name also by supplying a line ``# label = mylabel``
    in the header. If column names are found, the output ``components`` will
    be a list of the same length, with all elements set to ``mylabel``. If
    there are no column names found, I cannot figure out how many columns there
    are, so only a string ``mylabel`` is returned, instead of a list.
    
    These are possible headers with their output::
    
        # atm = kurucz
        # fittransfo = log
    
    >>> info, ps = parse_header('example_file.phot')
    >>> print(info)
    (None, None)
    >>> print(ps)
    (<phoebe.parameters.parameters.ParameterSet object at 0x837ccd0>, <phoebe.parameters.datasets.LCDataSet object at 0x7f0e046ed6d0>
    
    Another one::
      
        # passband = JOHNSON.V
        # atm = kurucz
        # ref = SiII_rvs
        # label = componentA
        
    >>> info, ps = parse_header('example_file.rv')
    >>> print(info)
    (None, 'componentA')
    >>> print(ps)
    (<phoebe.parameters.parameters.ParameterSet object at 0x837ccd0>, <phoebe.parameters.datasets.RVDataSet object at 0x7f0e046ed6d0>
    
    Another one::
      
        # atm = kurucz
        # fittransfo = log
        # flux     passband    time  sigma  unit
        #---------------------------------------
        
    >>> info, ps = parse_header('example_file.phot')
    >>> print(info)
    (['flux', 'passband', 'time', 'sigma', 'unit'], None)
    >>> print(ps)
    (<phoebe.parameters.parameters.ParameterSet object at 0x837ccd0>, <phoebe.parameters.datasets.LCDataSet object at 0x7f0e046ed6d0>
    
    Another one::
      
        # passband = JOHNSON.V
        # atm = kurucz
        # ref = SiII_rvs
        # label = componentA
        # rv       time           sigma
        #---------------------------------------
        
    >>> info, ps = parse_header('example_file.rv')
    >>> print(info)
    (['rv', 'time', 'sigma'], ['componentA', 'componentA', 'componentA'])
    >>> print(ps)
    (<phoebe.parameters.parameters.ParameterSet object at 0x837ccd0>, <phoebe.parameters.datasets.RVDataSet object at 0x7f0e046ed6d0>
    
    And the last example::
    
        # passband = JOHNSON.V
        # atm = kurucz
        # ref = SiII_rvs
        # rv       time           sigma   sigma   rv
        # starA    None           starA   starB   starB
        #------------------------------------------------
    
    >>> info, ps = parse_header('example_file.rv')
    >>> print(info)
    (['rv', 'time', 'sigma', 'sigma', 'rv'], ['starA', 'None', 'starA', 'starB', 'starB'])
    >>> print(ps)
    (<phoebe.parameters.parameters.ParameterSet object at 0x837ccd0>, <phoebe.parameters.datasets.RVDataSet object at 0x7f0e046ed6d0>
        
    @param filename: input file
    @type filename: str
    @return: (columns, components), (pbdep, dataset)
    @rtype: (list/None, list/str/None), (ParameterSet, DataSet)
    """
    #-- create a default pbdep and DataSet
    contexts = dict(rv='rvdep',
                    phot='lcdep',lc='lcdep',
                    spec='spdep',lprof='spdep',
                    vis='ifdep')
    dataset_classes = dict(rv=RVDataSet,
                           phot=LCDataSet,lc=LCDataSet,
                           spec=SPDataSet,lprof=SPDataSet,
                           vis=IFDataSet)
    ext = filename.split('.')[-1]
    pb = parameters.ParameterSet(context=contexts[ext])
    ds = dataset_classes[ext]()
    #-- they belong together, so they should have the same reference
    ds['ref'] = pb['ref']
    #-- open the file and start reading the lines
    with open(filename,'r') as ff:
        # We can only avoid reading in the whole file by first going through
        # it line by line, and collect the comment lines. We need to be able
        # to look ahead to detect where the headers ends.
        all_lines = []
        for line in ff.xreadlines():
            line = line.strip()
            if not line: continue
            elif line[0]=='#':
                #-- break when we reached the end!
                if line[1:4]=='---':
                    break
                all_lines.append(line[1:])
            #-- perhaps the user did not give a '----', is this safe?
            else:
                break
                
    #-- prepare some output and helper variables
    header_length = len(all_lines)
    components = None
    columns = None
    #-- now iterate over the header lines
    for iline,line in enumerate(all_lines):
        #-- comment lines can contain qualifiers from the RVDataSet,
        #   we recognise them by the presence of the equal "=" sign.
        split = line[1:].split("=")
        # if they do, they consist of "qualifier = value". Careful,
        # perhaps there are more than 1 "=" signs in the value
        # (e.g. the reference contains a "="). There are never
        # "=" signs in the qualifier.
        if len(split)>1:
            #-- qualifier is for sure the first element
            qualifier = split[0].strip()
            #-- if this qualifier exists, in either the RVDataSet
            #   or pbdep, add it. Text-to-value parsing is done
            #   at the ParameterSet level
            if qualifier in ds:
                ds[qualifier] = "=".join(split[1:]).strip()
            if qualifier in pb:
                pb[qualifier] = "=".join(split[1:]).strip()
            if qualifier=='label':
                components = "=".join(split[1:]).strip()
        #-- it is also possible that the line contains the column
        #   names: they should then contain at least the required
        #   columns! We recognise the column headers as that line
        #   which is followed by a line containing '#---' at least.
        #   Or the line after that one; in the latter case, also
        #   the components are given
        elif iline==(header_length-2):
            columns = line.split()
            components = all_lines[iline+1].split()
            break
        #-- now we only have column names
        elif iline==(header_length-1):
            columns = line.split()
    #-- some post processing:
    if isinstance(components,str) and columns is not None:
        components = [components]*len(columns)
    #-- that's it!
    return (columns, components), (pb,ds)    

def parse(file_pattern,full_output=False,**kwargs):
    """
    Parse a list of files.
    """
    if isinstance(file_pattern,str):
        file_pattern = sorted(glob.glob(file_pattern))
    
    output = OrderedDict()
    for ff in file_pattern:
        ext = os.path.splitext(ff)[1][1:]
        ioutput = globals()['parse_{}'.format(ext)](ff,full_output=True,**kwargs)
        for label in ioutput:
            if not label in output.keys():
                output[label] = [[],[]]
            output[label][0] += ioutput[label][0]
            output[label][1] += ioutput[label][1]
        
    if '__nolabel__' in output and not full_output:
        return output.values()[0]
    else:
        return output
    


def parse_lc(filename,columns=None,components=None,full_output=False,**kwargs):
    """
    Parse LC files to LCDataSets and lcdeps.
    
    **File format description**
    
    The filename **must** have the extension ``.lc``.
    
    The generic structure of an LC file is::
        
        # passband = JOHNSON.V
        # atm = kurucz
        # ref = april2011
        # label = capella
        2455453.0       1.     0.01    
        2455453.1       1.01   0.015    
        2455453.2       1.02   0.011    
        2455453.3       0.96   0.009    
    
    In this structure, you can only give LCs of one component in one
    file. An attempt will be made to interpret the comment lines (i.e. those lines
    where the first character equals '#') as qualifier/value for either the
    :ref:`lcobs <parlabel-phoebe-lcobs>` or :ref:`lcdep <parlabel-phoebe-lcdep>`.
    Failures are silently ignored, so you are allowed to put whatever comments
    in there (though with caution), or comment out data lines if you don't
    want to use them. The comments are optional. So this is also allowed::
    
        2455453.0       1.     0.01    
        2455453.1       1.01   0.015    
        2455453.2       1.02   0.011    
        2455453.3       0.96   0.009    
    
    In this case all the defaults from the :ref:`lcobs <parlabel-phoebe-lcobs>`,
    and :ref:`lcdep <parlabel-phoebe-lcdep>` ParameterSets will be kept, but can
    be possibly overriden by the extra kwargs. If, in this last example, you
    want to specify that different columns belong to different components,
    you need to give a list of those component labels, and set ``None`` wherever
    a column does not belong to a component (e.g. the time).
    
    The only way in which you are allowed to deviate from this structure, is
    by specifying column names, followed by a comment line of dashes (there
    needs to be at least one dash, no spaces)::
    
        # passband = JOHNSON.V
        # atm = kurucz
        # ref = april2011
        # label = capella
        # flux     time           sigma
        #---------------------------------------
        1.         2455453.0       0.01     
        1.01       2455453.1       0.015     
        1.02       2455453.2       0.011     
        0.96       2455453.3       0.009 
    
    In the latter case, you are allowed to omit any column except for ``time``
    ``lc`` and ``sigma``, which are required.
    
    .. tip::
 
       You can give missing values by setting a value to ``nan``
    
    Flux needs to be in erg/s/cm2/AA.
    
    **Input and output**
    
    The output can be readily appended to the C{obs} and C{pbdep} keywords in
    upon initialization of a ``Body``.
    
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
    
    **Example usage**
    
    Assume that any of the **second** example is saved in 
    file called ``myfile.lc``, you can do (the following lines are equivalent):
    
    >>> obs,pbdeps = parse_lc('myfile.lc')
    >>> obs,pbdeps = parse_lc('myfile.lc',columns=['time','flux','sigma'])
    
    Which is in this case equivalent to:
    
    >>> output = parse_lc('myfile.lc',full_output=True)
    >>> obs,pbdeps = output['__nolabel__']
    
    or 
    
    >>> obs,pbdeps = output.values()[0]
    
    The output can then be given to any Body:
    
    >>> starpars = parameters.ParameterSet(context='star')
    >>> meshpars = parameters.ParameterSet(context='mesh:marching')
    >>> star = Star(starpars,mesh=meshpars,pbdep=pbdeps,obs=obs)
    
    The first example explicitly contains a label, so an OrderedDict will
    always be returned, regardless the value of ``full_output``.
    
    The last example contains labels, so the full output is always given.
    Assume the contents of the last file is stored in ``myfile2.lc``:
    
    >>> output = parse_lc('myfile2.lc')
    >>> obs1,pbdeps1 = output['starA']
    >>> obs2,pbdeps2 = output['starB']
    
    @param filename: filename
    @type filename: string
    @param columns: columns in the file. If not given, they will be automatically detected or should be the default ones.
    @type columns: None or list of strings
    @param components: list of components for each column in the file. If not given, they will be automatically detected.
    @type components: None or list of strings
    @param full_output: if False and there are no labels in the file, only the data from the first component will be returned, instead of the OrderedDict
    @type full_output: bool
    @return: (list of :ref:`lcobs <parlabel-phoebe-lcobs>`, list of :ref:`lcdep <parlabel-phoebe-lcdep>`) or OrderedDict with the keys the labels of the objects, and then the lists of rvobs and rvdeps.
    """
    #-- which columns are present in the input file, and which columns are
    #   possible in the RVDataSet? The columns that go into the RVDataSet
    #   is the intersection of the two. The columns that are in the file but
    #   not in the RVDataSet are probably used for the pbdeps (e.g. passband)
    #   or for other purposes (e.g. label or unit).
    #-- parse the header
    (columns_in_file,components_in_file),(pb,ds) = parse_header(filename)
    
    if columns is None and columns_in_file is None:
        columns_in_file = ['time','flux','sigma']
    elif columns is not None:
        columns_in_file = columns
    columns_required = ['time','flux','sigma']
    columns_specs = dict(time=float,flux=float,sigma=float)
    
    missing_columns = set(columns_required) - set(columns_in_file)
    if len(missing_columns)>0:
        raise ValueError("Missing columns in LC file: {}".format(", ".join(missing_columns)))
    
    #-- prepare output dictionaries. The first level will be the label key
    #   of the Body. The second level will be, for each Body, the pbdeps or
    #   datasets.
    output = OrderedDict()
    Ncol = len(columns_in_file)
    
    #-- collect all data
    data = []
    #-- open the file and start reading the lines    
    with open(filename,'r') as ff:
        for line in ff.readlines():
            line = line.strip()
            if not line: continue
            if line[0]=='#': continue
            data.append(tuple(line.split()[:Ncol]))
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
    #   LCDataSets or pbdeps    
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
                
    #-- If the user didn't provide any labels (either as an argument or in the
    #   file), we don't bother the user with it:
    if '__nolabel__' in output and not full_output:
        return output.values()[0]
    else:
        return output

def parse_rv(filename,columns=None,components=None,full_output=False,**kwargs):
    """
    Parse RV files to RVDataSets and rvdeps.
    
    **File format description**
    
    The generic structure of an RV file is::
        
        # passband = JOHNSON.V
        # atm = kurucz
        # ref = SiII_rvs
        # label = componentA
        2455453.0     -10.   0.1    
        2455453.1      -5.   0.15    
        2455453.2       2.   0.11    
        2455453.3       6.   0.09    
    
    In this structure, you can only give RVs of one component in one
    file. An attempt will be made to interpret the comment lines (i.e. those lines
    where the first character equals '#') as qualifier/value for either the
    :ref:`rvobs <parlabel-phoebe-rvobs>` or :ref:`rvdep <parlabel-phoebe-rvdep>`.
    Failures are silently ignored, so you are allowed to put whatever comments
    in there (though with caution), or comment out data lines if you don't
    want to use them. The comments are optional. So this is also allowed::
    
        2455453.0     -10.   0.1    
        2455453.1      -5.   0.15    
        2455453.2       2.   0.11    
        2455453.3       6.   0.09
    
    In this case all the defaults from the :ref:`rvobs <parlabel-phoebe-rvobs>`,
    and :ref:`rvdep <parlabel-phoebe-rvdep>` ParameterSets will be kept, but can
    be possibly overriden by the extra kwargs. If, in this last example, you
    want to specify that different columns belong to different components,
    you need to give a list of those component labels, and set ``None`` wherever
    a column does not belong to a component (e.g. the time).
    
    The only way in which you are allowed to deviate from this structure, is
    by specifying column names, followed by a comment line of dashes (there
    needs to be at least one dash, no spaces)::
    
        # passband = JOHNSON.V
        # atm = kurucz
        # ref = SiII_rvs
        # label = componentA
        # rv       time           sigma
        #---------------------------------------
        -10.       2455453.0       0.1    
         -5.       2455453.1       0.15    
          2.       2455453.2       0.11    
          6.       2455453.3       0.09
    
    In the latter case, you are allowed to omit any column except for ``time``
    ``rv`` and ``sigma``, which are required.
    
    The final option that you have is to supply information on multiple
    components in one file. In that case, you need to specify the column names
    **and** the components, in two consecutive lines (first column names, then
    component labels), or pass ``columns`` and ``components`` arguments during
    the function call::
    
        # passband = JOHNSON.V
        # atm = kurucz
        # ref = SiII_rvs
        # rv       time           sigma   sigma   rv
        # starA    None           StarA   starB   starB
        #------------------------------------------------
        -10.       2455453.0       0.1   0.05     11.
         -5.       2455453.1       0.15  0.12      6.
          2.       2455453.2       0.11  nan      nan
          6.       2455453.3       0.09  0.2     -12.
    
    .. tip::
 
       You can give missing values by setting a value to ``nan``
    
    Radial velocities need to be in km/s.
    
    **Input and output**
    
    The output can be readily appended to the C{obs} and C{pbdep} keywords in
    upon initialization of a ``Body``.
    
    Extra keyword arguments are passed to output RVDataSets or pbdeps,
    wherever they exist and override the contents of the comment lines in the
    phot file. For example, compare these two files::
    
        # passband = JOHNSON.V
        # atm = kurucz
        # ref = SiII_rvs
        # rv       time           sigma   sigma   rv
        # starA    None           starA   starB   starB
        #------------------------------------------------
        -10.       2455453.0       0.1   0.05     11.
         -5.       2455453.1       0.15  0.12      6.
          2.       2455453.2       0.11  nan      nan
          6.       2455453.3       0.09  0.2     -12.

    and::
    
        -10.       2455453.0       0.1   0.05     11.
         -5.       2455453.1       0.15  0.12      6.
          2.       2455453.2       0.11  nan      nan
          6.       2455453.3       0.09  0.2     -12.
    
    Since all the information is available in the first example, you can
    readily do:
    
    >>> output = parse_rv('myfile.rv')
    
    While you need to do supply more information the second case:
    
    >>> output = parse_rv('myfile.rv',columns=['rv','time','sigma','sigma','rv'],
    ...     components=['starA',None,'StarA','starB','starB'])
    
    
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
    
    **Example usage**
    
    Assume that any of the **second** example is saved in 
    file called ``myfile.phot``, you can do (the following lines are equivalent):
    
    >>> obs,pbdeps = parse_phot('myfile.rv')
    >>> obs,pbdeps = parse_phot('myfile.rv',columns=['time','rv','sigma'])
    
    Which is in this case equivalent to:
    
    >>> output = parse_phot('myfile.rv',full_output=True)
    >>> obs,pbdeps = output['__nolabel__']
    
    or 
    
    >>> obs,pbdeps = output.values()[0]
    
    The output can then be given to any Body:
    
    >>> starpars = parameters.ParameterSet(context='star')
    >>> meshpars = parameters.ParameterSet(context='mesh:marching')
    >>> star = Star(starpars,mesh=meshpars,pbdep=pbdeps,obs=obs)
    
    The first example explicitly contains a label, so an OrderedDict will
    always be returned, regardless the value of ``full_output``.
    
    The last example contains labels, so the full output is always given.
    Assume the contents of the last file is stored in ``myfile2.rv``:
    
    >>> output = parse_phot('myfile2.rv')
    >>> obs1,pbdeps1 = output['starA']
    >>> obs2,pbdeps2 = output['starB']
    
    @param filename: filename
    @type filename: string
    @param columns: columns in the file. If not given, they will be automatically detected or should be the default ones.
    @type columns: None or list of strings
    @param components: list of components for each column in the file. If not given, they will be automatically detected.
    @type components: None or list of strings
    @param full_output: if False and there are no labels in the file, only the data from the first component will be returned, instead of the OrderedDict
    @type full_output: bool
    @return: (list of :ref:`rvobs <parlabel-phoebe-rvobs>`, list of :ref:`rvdep <parlabel-phoebe-rvdep>`) or OrderedDict with the keys the labels of the objects, and then the lists of rvobs and rvdeps.
    """
    #-- which columns are present in the input file, and which columns are
    #   possible in the RVDataSet? The columns that go into the RVDataSet
    #   is the intersection of the two. The columns that are in the file but
    #   not in the RVDataSet are probably used for the pbdeps (e.g. passband)
    #   or for other purposes (e.g. label or unit).
    #-- parse the header
    (columns_in_file,components_in_file),(pb,ds) = parse_header(filename)
    
    if columns is None and columns_in_file is None:
        columns_in_file = ['time','rv','sigma']
    elif columns is not None:
        columns_in_file = columns
    columns_required = ['time','rv','sigma']
    columns_specs = dict(time=float,rv=float,sigma=float)
    
    missing_columns = set(columns_required) - set(columns_in_file)
    if len(missing_columns)>0:
        raise ValueError("Missing columns in RV file: {}".format(", ".join(missing_columns)))
    
    #-- prepare output dictionaries. The first level will be the label key
    #   of the Body. The second level will be, for each Body, the pbdeps or
    #   datasets.
    output = OrderedDict()
    Ncol = len(columns_in_file)
    
    #-- collect all data
    data = []
    #-- open the file and start reading the lines    
    with open(filename,'r') as ff:
        for line in ff.readlines():
            line = line.strip()
            if not line: continue
            if line[0]=='#': continue
            data.append(tuple(line.split()[:Ncol]))
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
    #   RVDataSets or pbdeps    
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
                
    #-- If the user didn't provide any labels (either as an argument or in the
    #   file), we don't bother the user with it:
    if '__nolabel__' in output and not full_output:
        return output.values()[0]
    else:
        return output

def parse_phot(filenames,columns=None,full_output=False,**kwargs):
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
    Ncol = len(columns_in_file)
    
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
        with open(filename,'r') as ff:
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
                            Ncol = len(columns_in_file)
                            logger.info("Auto detecting columns in PHOT {}: {}".format(filename,", ".join(columns_in_file)))
                            missing_columns = set(columns_required) - set(columns_in_file)
                            if len(missing_columns)>0:
                                raise ValueError("Missing columns in PHOT file: {}".format(", ".join(missing_columns)))\
                #-- data lines:
                else:
                    data.append(tuple(line.split()[:Ncol]))
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
                        if key in components[label][0][ref]:
                            components[label][0][-1][key] = kwargs[key]
                        if key in components[label][1][ref]:
                            components[label][1][-1][key] = kwargs[key]
                        
    #-- If the user didn't provide any labels (either as an argument or in the
    #   file), we don't bother the user with it:
    if not 'label' in columns_in_file and not full_output:
        return components.values()[0]
    else:
        return components
            

def parse_lprof(filenames):
    raise NotImplementedError

def parse_spec_as_lprof(filename,line_name,clambda,wrange,columns=None,components=None,full_output=False,**kwargs):
    """
    Parse a SPEC file as an LPROF file to an SPDataSet and a spdep.
    
    This effectively extracts a line profile from a full spectrum.
    
    The structure of a SPEC file is::
    
        # atm = kurucz
        # ld_func = claret
        3697.204  5.3284e-01
        3697.227  2.8641e-01
        3697.251  2.1201e-01
        3697.274  2.7707e-01
    
    An attempt will be made to interpret the comment lines as qualifier/value
    for either the :ref:`spobs <parlabel-phoebe-spobs>` or :ref:`spdep <parlabel-phoebe-spdep>`.
    Failures are silently ignored, so you are allowed to put whatever comments
    in there (though with caution), or comment out data lines if you don't
    want to use them.
    
    The output can be readily appended to the C{obs} and C{pbdep} keywords in
    upon initialization of a ``Body``.
    
    Extra keyword arguments are passed to output SPDataSets or pbdeps,
    wherever they exist and override the contents of the comment lines in the
    phot file.
    
    Example usage:
    
    >>> obs,pbdeps = parse_spec_as_lprof('myfile.spec')
    >>> starpars = parameters.ParameterSet(context='star')
    >>> meshpars = parameters.ParameterSet(context='mesh:marching')
    >>> star = Star(starpars,mesh=meshpars,pbdep=pbdeps,obs=obs)
    
    @param filenames: list of filename or a filename glob pattern
    @type filenames: list or string
    @param clambda: central wavelength of profile (AA)
    @type clambda: float
    @param wrange: entire wavelength range (AA)
    @type wrange: float
    @return: list of :ref:`lcobs <parlabel-phoebe-lcobs>`, list of :ref:`lcdep <parlabel-phoebe-lcdep>`
    """
    #-- parse the header
    (columns_in_file,components_in_file),(pb,ds) = parse_header(filename)

    if columns is None and columns_in_file is None:
        columns_in_file = ['wavelength','flux']
    elif columns is not None:
        columns_in_file = columns
    columns_required = ['wavelength','flux']
    columns_specs = dict(time=float,wavelength=float,flux=float,sigma=float)
    
    missing_columns = set(columns_required) - set(columns_in_file)
    if len(missing_columns)>0:
        raise ValueError("Missing columns in LC file: {}".format(", ".join(missing_columns)))
    
    #-- prepare output dictionaries. The first level will be the label key
    #   of the Body. The second level will be, for each Body, the pbdeps or
    #   datasets.
    output = OrderedDict()
    Ncol = len(columns_in_file)
    
    #-- collect all data
    data = []
    #-- read the comments of the file
    with open(filename,'r') as ff:
        for line in ff.readlines():
            line = line.strip()
            if not line: continue
            if line[0]=='#': continue
            data.append(tuple(line.split()[:Ncol]))
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
            #-- cut out the line profile
            start = np.searchsorted(ds['wavelength'],clambda-wrange/2.0)
            end = np.searchsorted(ds['wavelength'],clambda+wrange/2.0)-1
            if (end-start)>10000:
                raise ValueError('Spectral window too big')
            #-- save to the dataset
            ref = "{}-{}".format(line_name,filename)
            ds['continuum'] = np.ones(len(ds['wavelength'][start:end])).reshape((1,-1))
            ds['wavelength'] = ds['wavelength'][start:end].reshape((1,-1))
            ds['flux'] = ds['flux'][start:end].reshape((1,-1))
            ds['ref'] = ref
            ds['filename'] = ref+'.spobs'
            ds.estimate_noise()
            if not 'time' in columns_in_file:
                ds['time'] = np.zeros(len(ds['flux']))
                #ds['columns'].append('time')
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
    
def parse_vis(filename,columns=None,full_output=False,**kwargs):
    """
    Parse VIS files to IFDataSets and ifdeps.
    
    **File format description**
    
    The generic structure of a VIS file is::
        
        # atm = kurucz
        # passband = 2MASS.KS
        -7.779     7.980  0.932 0.047   56295.0550
        -14.185    0.440  0.808 0.040   56295.0550
        -29.093  -15.734  0.358 0.018   56295.0551
         -6.406   -7.546  0.957 0.048   56295.0552
        -21.314  -23.720  0.598 0.030   56295.0534

    
    The columns represent respectively the **U-coordinate, V-coordinate, vis, sigma_vis, time**.
    An attempt will be made to interpret the comment lines (i.e. those lines
    where the first character equals '#') as qualifier/value for either the
    :ref:`ifobs <parlabel-phoebe-ifobs>` or :ref:`ifdep <parlabel-phoebe-ifdep>`.
    Failures are silently ignored, so you are allowed to put whatever comments
    in there (though with caution), or comment out data lines if you don't
    want to use them. The comments are optional. So this is also allowed::
    
        -7.779     7.980  0.932 0.047  56295.0550
        -14.185    0.440  0.808 0.040  56295.0550
        -29.093  -15.734  0.358 0.018  56295.0551
         -6.406   -7.546  0.957 0.048  56295.0552
        -21.314  -23.720  0.598 0.030  56295.0534
        
    The only way in which you are allowed to deviate from this structure, is
    by specifying column names, followed by a comment line of dashes (there
    needs to be at least one dash, no spaces)::
    
        # atm = kurucz
        # fittransfo = log
        # passband = 2MASS.KS
        # ucoord  vcoord  vis  sigma_vis time
        #---------------------------------------------
        -7.779     7.980  0.932 0.047 56295.0550
        -14.185    0.440  0.808 0.040 56295.0550
        -29.093  -15.734  0.358 0.018 56295.0551
         -6.406   -7.546  0.957 0.048 56295.0552
        -21.314  -23.720  0.598 0.030 56295.0534
    
    In the latter case, you are allowed to omit any column except for ``ucoord``
    ``vcoord``, ``vis`` and ``sigma_vis``, which are required.
    If not given, the default ``time`` is zero.
        
    
    .. warning::
    
       You are not allowed to have missing values. Each column must
       have a value for every observation.
    
    **Input and output**
    
    The output can be readily appended to the C{obs} and C{pbdep} keywords in
    upon initialization of a ``Body``.
    
    Extra keyword arguments are passed to output IFDataSets or pbdeps,
    wherever they exist and override the contents of the comment lines in the
    phot file.
    
    If C{full_output=True}, you will get a consistent output, regardless what
    the input file looks like. This can be useful for automatic parsing. In
    this case, the output is an OrderedDict, with the keys at the first level
    the key of the component (if no labels are given, this will be C{__nolabel__}).
    The value for each key is a list of two lists: the first list contains the
    IFDataSets, the second list the corresponding pbdeps.
    
    If C{full_output=False}, you will get the same output as described in the
    previous paragraph only if labels are given in the file. Else, the output
    consists of only one component, which is probably a bit confusing for the
    user (at least me). So if there are no labels given and C{full_output=False},
    the two lists are immediately returned.
    
    **Example usage**
    
    Assume that any of the first three examples is saved in 
    file called ``myfile.vis``, you can do (the following lines are equivalent):
    
    >>> obs,pbdeps = parse_vis('myfile.vis')
    >>> obs,pbdeps = parse_vis('myfile.vis',columns=['ucoord','vcoord','vis','sigma_vis','time'])
    
    Which is in this case equivalent to:
    
    >>> output = parse_vis('myfile.vis',full_output=True)
    >>> obs,pbdeps = output['__nolabel__']
    
    or 
    
    >>> obs,pbdeps = output.values()[0]
    
    The output can then be given to any Body:
    
    >>> starpars = parameters.ParameterSet(context='star')
    >>> meshpars = parameters.ParameterSet(context='mesh:marching')
    >>> star = Star(starpars,mesh=meshpars,pbdep=pbdeps,obs=obs)
    
    @param filenames: list of filename or a filename glob pattern. If you give a list of filenames, they need to all have the same structure!
    @type filenames: list or string
    @param columns: columns in the file. If not given, they will be automatically detected or should be the default ones.
    @type columns: None or list of strings
    @param full_output: if False and there are no labels in the file, only the data from the first component will be returned, instead of the OrderedDict
    @type full_output: bool
    @return: (list of :ref:`lcobs <parlabel-phoebe-ifobs>`, list of :ref:`ifdep <parlabel-phoebe-ifdep>`) or OrderedDict with the keys the labels of the objects, and then the lists of lcobs and lcdeps.
    """
    #-- which columns are present in the input file, and which columns are
    #   possible in the LCDataSet? The columns that go into the LCDataSet
    #   is the intersection of the two. The columns that are in the file but
    #   not in the LCDataSet are probably used for the pbdeps (e.g. passband)
    #   or for other purposes (e.g. label or unit).
    (columns_in_file,components_in_file),(pb,ds) = parse_header(filename)
    
    if columns is None:
        columns_in_file = ['ucoord','vcoord','vis','sigma_vis','time']
    else:
        columns_in_file = columns
    columns_required = ['ucoord','vcoord','vis','sigma_vis']
    columns_specs = dict(ucoord=float,vcoord=float,sigma_vis=float,
                         time=float,unit=str,vis=float,phase=float,sigma_phase=float)
    
    missing_columns = set(columns_required) - set(columns_in_file)
    if len(missing_columns)>0:
        raise ValueError("Missing columns in VIS file: {}".format(", ".join(missing_columns)))
    
    #-- prepare output dictionaries. The first level will be the label key
    #   of the Body. The second level will be, for each Body, the pbdeps or
    #   datasets.
    output = OrderedDict()
    Ncol = len(columns_in_file)
    
    #-- collect all data
    data = []
    #-- open the file and start reading the lines    
    with open(filename,'r') as ff:
        for line in ff.readlines():
            line = line.strip()
            if not line: continue
            if line[0]=='#': continue
            data.append(tuple(line.split()[:Ncol]))
    #-- we have the information from header now, but only use that
    #   if it is not overriden
    #-- we don't allow information on components
    components = ['__nolabel__']*len(columns_in_file)
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
    #   IFDataSets or pbdeps    
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
                
    #-- If the user didn't provide any labels (either as an argument or in the
    #   file), we don't bother the user with it:
    if '__nolabel__' in output and not full_output:
        return output.values()[0]
    else:
        return output
#}

#{ High-end wrappers

def oifits2vis(filename,wtol=1.,ttol=1e-6,**kwargs):
    """
    Convert an OIFITS file to Phoebe's VIS format.
    
    @param filename: OIFITS file location
    @type filename: str
    @return: VIS file location
    @rtype: str
    """
    #-- prepare output contexts
    ref = os.path.splitext(filename)[0]
    ifmdep = parameters.ParameterSet(context='ifdep',ref=ref)
    ifmobs = IFDataSet(context='ifobs',ref=ref,columns=['ucoord','vcoord','vis','sigma_vis','time'])
    for key in kwargs:
        if key in ifmdep:
            ifmdep[key] = kwargs[key]
        if key in ifmobs:
            ifmobs[key] = kwargs[key]
    #-- read in the visibilities
    templatedata = oifits.open(filename)
    allvis2 = templatedata.allvis2 
    vis = np.sqrt(allvis2['vis2data'])
    vis_sigma = 0.5*allvis2['vis2err']/allvis2['vis2data']
    ucoord = allvis2['ucoord']
    vcoord = allvis2['vcoord']
    time = allvis2['mjd']
    
    skip_columns = ifmobs['columns']
    all_keys = list(set(list(ifmobs.keys())) | set(list(ifmdep.keys())))
    all_keys = [key for key in all_keys if not key in skip_columns]
    if 'filename' in all_keys:
        all_keys.pop(all_keys.index('filename'))
    comments = ['# {} = {}'.format(key,ifmdep[key]) if key in ifmdep else '# {} = {}'.format(key,ifmobs[key]) for key in all_keys]
    comments+= ['# ucoord vcoord vis sigma_vis time']
    comments+= ['#---------------------------------']
    output_filename = os.path.splitext(filename)[0]+'.vis'
    ascii.write_array(np.column_stack([ucoord,vcoord,vis,vis_sigma,time]),
              output_filename,comments=comments)
    return output_filename
#}