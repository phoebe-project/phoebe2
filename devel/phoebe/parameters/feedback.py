"""
Definitions of fitting feedback and helper functions.
"""
import logging
import os
import numpy as np
import scipy.stats as st
import scipy.special as sp
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import copy
from phoebe.utils import decorators
from phoebe.units import conversions
from phoebe.parameters import parameters

logger = logging.getLogger("PAR.FB")

green = lambda x: "\033[32m" + x + '\033[m'

def gauss(x,mu,sig):
    return 1./np.sqrt(2*np.pi)/sig*np.exp(-(x-mu)**2/sig/sig*0.5)

def matrixify(x,y,z,Nx,Ny,limits=False,border=10.0):
    if limits == False:
        xmin = min(x)
        ymin = min(y)
        xmax = max(x)
        ymax = max(y)
        
        xmin -= (xmax-xmin)/border
        xmax += (xmax-xmin)/border
        ymin -= (ymax-ymin)/border
        ymax += (ymax-ymin)/border
        
    else:
        xmin,ymin,xmax,ymax = limits

    m = np.zeros((Nx,Ny))
    c = np.zeros((Nx,Ny),dtype=int)

    xstep = xmax - xmin
    ystep = ymax - ymin
    for i in zip(x,y,z):
        try:
            xi = int((i[0]-xmin)/xstep*Nx)
            yi = int((i[1]-ymin)/ystep*Ny)
            m[yi][xi] += i[2]
            c[yi][xi] += 1
        except IndexError:
            pass
    
    return c,xmin,xmax,ymin,ymax

class Feedback(object):
    """
    Feedback Baseclass.
    
    """
    def __init__(self):
        # The parameters that where used to fit
        self._parameters = None
        # The initial values
        self._init_parameters = None
        # Fit statistics (chi2, lnprob...)
        self._fit_stats = None
        # Fit history (traces etc)
        self._traces = None
        # Correlation matrix
        self._cormat = None
        self._converged = None
        # Fitting parameterSet
        self._fit_parset = None
        # Optional dictionary of unique identifiers to readable str
        self._translation = dict()
        self._info = ''
        self.fitting = None
        self.compute = None
        return None
    
    
    def get_parameters(self):
        """
        Return the parameters.
        
        They have qualifiers, values...
        """
        return None
    
    def get_label(self):
        
        if self.fitting is not None:
            return self.fitting['label']
    
    def get_computelabel(self):
        if self.compute is not None:
            return self.compute['label']
    
    def apply_to(self, system, revert=False):
        """
        Set values of the adjustable parameters in the system from the fitting results.
        """
        adjustable_parameters = self._parameters
        ids = [par.get_unique_label() for par in adjustable_parameters]
        
        for parset in system.walk():
            for par in parset:
                system_par = parset.get_parameter(par)
                # Try to find this parameter in the list of adjustable
                # parameters. If it's not there, don't bother
                try:
                    index = ids.index(system_par.get_unique_label())
                    this_par = adjustable_parameters[index]
                except ValueError:
                    continue
                
                if revert:
                    system_par.reset()
                else:
                    # Set priors, posteriors and values
                    system_par.set_value(this_par.get_value())
                    system_par.prior = this_par.get_prior()
                    system_par.posterior = this_par.get_posterior()
    
    def draw_from_posteriors(self, size=1):
        pass
    
    def to_string(self, color=True):
        mystring = [self._info]
        
        def valid(text):
            return text if not color else "\033[32m" + text + '\033[m'
        
        def invalid(text):
            return text if not color else "\033[31m" + text + '\033[m'
        
        mystring.append("\nParameter values\n=================")
        
        start_line = len(mystring)
        
        table = [['name', 'unit', 'value(POST)',\
                  'loc(POST)', 'scale(POST)', 'dist(POST)'],
                  ['', '', 'value(PRIOR)', 'loc(PRIOR)', 'scale(PRIOR)', 'dist(PRIOR)']]
        
        # First report all the values
        for i, param in enumerate(self._parameters):
            unique_id = param.get_unique_label()
            row = param.as_string_table()
            row[unique_id][0] = self._translate(unique_id)
            
            table.append(row[unique_id][:6])
            table.append(['']*2 + row[unique_id][6:])
            
            # What can go wrong?
            # 1. the init value can be equal to the to posterior value
            # 2. the uncertainty can be zero
            # 3. the posterior best value can be unlikely given the prior
            # 4. the posterior best value can be at the edge of the prior
        
        column_widths = []
        for col in range(len(table[0])):
            column_widths.append(max([len(table[row][col]) for row in range(len(table))])+1)
        row_template = "|".join(['{{:<{:d}s}}'.format(column_widths[0])] + ['{{:>{:d}s}}'.format(w) for w in column_widths[1:]])
        
        for row in table:
            mystring.append(row_template.format(*tuple(row)))
        
        separator = "+".join(['-'*w for w in column_widths])
        mystring.insert(start_line, separator)
        mystring.insert(start_line+3, separator)
        mystring.append(separator)
        
        mystring.append("\nCorrelations\n=============")
        
        mystring += ['({:2d}) {}'.format(i, self._translate(param.get_unique_label())) \
                           for i, param in enumerate(self._parameters)]
        mystring.append('       '+' '.join(['  ({:2d})'.format(i) for i in range(len(self._parameters))]))
        
        old_settings = np.get_printoptions()
        
        fmt = lambda x: green( '{:+6.3f}'.format(x)) if np.abs(x)>0.5 else ('{:+6.3f}'.format(x) if -np.isnan(x) else '   -  ')
        
        np.set_printoptions(precision=3, linewidth=200, suppress=True, formatter=dict(float=fmt))
        cormat = self._cormat.copy()
        cormat[np.tril_indices(cormat.shape[0])] = np.nan
        cor_string = str(cormat).split('\n')
        cor_string = ['({:2d}) '.format(i) + line + ' ({:2d})'.format(i) for i,line in enumerate(cor_string)]
        mystring += cor_string
        # Then report the most significant correlations
        np.set_printoptions(**old_settings)
        
        # Should we test distributions? E.g.
        #scipy.stats.kstest(np.random.uniform(size=1000), 'uniform')[1]

        return "\n".join(mystring)
    
    def _translate(self, unique_id):
        if unique_id in self._translation:
            return self._translation[unique_id]
        else:
            return unique_id
    
    def set_translation(self, translation):
        self._translation = translation
    
    def get_parameter(self, unique_id):
        for par in self._parameters:
            if par.get_unique_label() == unique_id:
                break
        else:
            return None
        
        return par
            
    
    def __str__(self):
        return self.to_string()
    
    
    
class FeedbackLmfit(Feedback):
    """
    Feedback from the lmfit fitting package.
    """
    def __init__(self, lmfit_result, lmfit_pars, extra_info, init, fitting=None,
                 compute=None):
        
        self._translation = dict()
        self._info = ''
        
        if extra_info is not None:
            self._traces = extra_info['traces']
            self._fit_stats = extra_info['redchis']
        
        # Retrieve the (initial) parameters
        init_phoebe_pars = self.retrieve_parameters(init)
        
        # The parameters that were used to fit (but their initial state)
        self._parameters = copy.deepcopy(init_phoebe_pars)
        for par in init_phoebe_pars:
            par.remember()
        
        # Build correlation matrix and set posteriors at the same time
        parlabels = [lmfit_pars[par].name for par in lmfit_pars]
        correls = np.zeros((len(init_phoebe_pars), len(init_phoebe_pars)))
        for i, ipar in enumerate(lmfit_pars):
            for j, jpar in enumerate(lmfit_pars):
                if i==j:
                    correls[i,j] = 1.0
                    self._parameters[i].set_value(lmfit_pars[ipar].value)
                    self._parameters[i].set_posterior(distribution='normal',
                                                      mu=lmfit_pars[ipar].value,
                                                      sigma=lmfit_pars[ipar].stderr)
                else:
                    if lmfit_pars[ipar].correl:
                        correls[i,j] = lmfit_pars[ipar].correl[jpar]
        self._cormat = correls    
        self.fitting = fitting
        self.compute = compute
    
    def retrieve_parameters(self, init):
        """
        Retrieve parameters from whatever is given.
        
        init can be a list of adjustable parameters
        init can be a system
        init can be a Bundle
        """
        if isinstance(init, list):
            return init
        
        # is this a bundle?
        if hasattr(init, 'twigs'):
            system = init.get_system()
            
        # else it is a system
        else:
            system = init
        
        adjustables = system.get_adjustable_parameters()
                
        # if a Bundle is given, set the translations
        if hasattr(init, 'twigs'):
            translations = dict()
            trunk = init.trunk
            adj_ids = [par.get_unique_label() for par in adjustables]
            adj_names = []
            
            for item in trunk:
                # if it is not a parameter, don't bother
                if not item['class_name'] == 'Parameter':
                    continue
                # if we can't match the unique label, don't bother
                try:
                    index = adj_ids.index(item['unique_label'])
                except ValueError:
                    continue
                
                # We got it!
                translations[adj_ids[index]] = item['twig']
            
            # Now strip the common parts of the twig, in order not too make the
            # name too long
            if len(translations)>1:
                # Cycle over the keys in the same order always
                keys = list(translations.keys())                
                while True:
                    # what's the current last twig entry?
                    try:
                        last_entry = translations[keys[0]].rsplit('@', 1)[1]
                    # If we can't even split it, we've gone far enough
                    except IndexError:
                        break
                    # Is it the same for the other ones? if it is, strip it
                    # from all entries and try again until exhaustion
                    for key in keys[1:]:
                        if translations[key].rsplit('@', 1)[1] != last_entry:
                            break
                    else:
                        for key in keys:
                            translations[key] = translations[key].rsplit('@',1)[0]
                        continue
                    break                                                        
        
            self.set_translation(translations)
        return adjustables
    
    

        
        
class FeedbackEmcee(Feedback):
    """
    Feedback from the emcee fitting package.
    """
    def __init__(self, emcee_file, init=None, lnproblim=-np.inf,
                 burnin=0, thin=1, fitting=None, compute=None):
        
        self._emcee_file = emcee_file
        self._translation = dict()
        
        # keep track of filesize to see if anything changed
        self._checkfilesize = os.stat(emcee_file).st_size
        
        # Retrieve the (initial) parameters
        init_phoebe_pars = self.retrieve_parameters(init)
        
        # Set the parameters and remember their state
        self._parameters = copy.deepcopy(init_phoebe_pars)
        
        for par in self._parameters:
            par.remember()
        
        self.do_reload(lnproblim, burnin, thin)
        self.fitting = fitting
        self.compute = compute
        # Reshape the array in a convenient format
        #data = data.reshape( len(data) / nwalkers, nwalkers, -1)
    
    def check_file(self, raise_error=True):
        """
        Check if the emcee file is still the same one.
        
        We check this by checking if the filesize is still the same.
        """
        samefile = os.stat(self._emcee_file).st_size == self._checkfilesize
        
        if not samefile and raise_error:
            raise IOError("Emcee file {} has changed, cannot reload".format(self._emcee_file))
        
        return samefile
            
    
    def retrieve_parameters(self, init):
        """
        Retrieve parameters from whatever is given.
        
        init can be a list of adjustable parameters
        init can be a system
        init can be a Bundle
        """
        # If is in None, allow building the feedback with generic parameters
        if init is None:
            with open(self._emcee_file, 'r') as ff:
                while True:
                    line = ff.readline()
                    if not line:
                        raise IOError("Invalid file")
                    if line[0] == '#':
                        continue
                    n_pars = len(line.strip().split())-2
                    break
            
            adjustables = []
            for i in range(n_pars):
                adjustables.append(parameters.Parameter(qualifier=str(i), cast_type=float, value=0))
                adjustables[-1]._unique_label = str(i)
            return adjustables                                                                                                       
                    
                
        
        if isinstance(init, list):
            return init
        
        # is this a bundle?
        if hasattr(init, 'twigs'):
            system = init.get_system()
            
        # else it is a system
        else:
            system = init
        
        adjustables = system.get_adjustable_parameters()
        
        # Check if the adjustable's unique ID is conform what is in the chain file
        with open(self._emcee_file, 'r') as open_file:
            header = open_file.readline()[1:].strip().split()[1:-1]
        for adj, uid in zip(adjustables, header):
            if adj.get_unique_label() != uid:
                logger.warning("Parameter {} in emcee file has different unique label".format(adj.get_qualifier()))
        
        # if a Bundle is given, set the translations for readable labels
        if hasattr(init, 'twigs'):
            translations = dict()
            trunk = init.trunk
            adj_ids = [par.get_unique_label() for par in adjustables]
            adj_names = []
            
            for item in trunk:
                # if it is not a parameter, don't bother
                if not item['class_name'] == 'Parameter':
                    continue
                # if we can't match the unique label, don't bother
                try:
                    index = adj_ids.index(item['unique_label'])
                except ValueError:
                    continue
                
                # We got it!
                translations[adj_ids[index]] = item['twig']
            
            # Now strip the common parts of the twig, in order not too make the
            # name too long
            if len(translations)>1:
                # Cycle over the keys in the same order always
                keys = list(translations.keys())                
                while True:
                    # what's the current last twig entry?
                    try:
                        last_entry = translations[keys[0]].rsplit('@', 1)[1]
                    # If we can't even split it, we've gone far enough
                    except IndexError:
                        break
                    # Is it the same for the other ones? if it is, strip it
                    # from all entries and try again until exhaustion
                    for key in keys[1:]:
                        if translations[key].rsplit('@', 1)[1] != last_entry:
                            break
                    else:
                        for key in keys:
                            translations[key] = translations[key].rsplit('@',1)[0]
                        continue
                    break                                                        
        
            self.set_translation(translations)
        return adjustables
    
    
    def do_reload(self, lnproblim=-np.inf, burnin=0, thin=1):
        """
        Reload the data from the file.
        
        """
        # Load the emcee file and remember some properties            
        data = np.loadtxt(self._emcee_file)
        self._lnproblim = lnproblim
        self._burnin = burnin
        self._thin = thin
        
        walkers, data, logp = data[burnin::thin,0], data[burnin::thin,1:-1],\
                              data[burnin::thin,-1]
        nwalkers = int(walkers.max() + 1)
        niterations = int(data.shape[0] / nwalkers)
        npars = data.shape[1]
        
        myinfo = 'EMCEE {}'.format(self._emcee_file)
        myinfo += '\n' + len(myinfo)*'=' + '\n'
        myinfo += ("Number of walkers: {}".format(nwalkers)) + '\n'
        myinfo += ("Number of iterations: {}".format(niterations)) + '\n'
        myinfo += ("Number of parameters: {}".format(npars)) + '\n'
        myinfo += ("Maximum lnprob: {}".format(logp.max())) + '\n'
        myinfo += ("Cut-off lnprob: {}".format(lnproblim)) + '\n'
        myinfo += ("Burn-in: {}".format(burnin)) + '\n'
        myinfo += ("Thinning: {}".format(thin)) + '\n'
                   
        self._info = myinfo
        
        select = (logp >= lnproblim)
        data = data[select]
        walkers = walkers[select]
        logp = logp[select]
        
        sigmas = np.std(data, axis=0)
        correl = np.zeros((npars, npars))
        
        for i, ipar in enumerate(self._parameters):
            for j, jpar in enumerate(self._parameters):
                if i==j:
                    correl[i,j] = 1.0
                    self._parameters[i].set_value(np.median(data[:,i]))
                    self._parameters[i].set_posterior(distribution='trace',
                                                      trace=data[:,i])
                else:
                    prs = st.spearmanr(data[:, i], data[:, j])[0]
                    correl[i,j] = prs #* sigmas[i] * sigmas[j]
        
        self._cormat = correl

        
    
    def to_string(self, *args, **kwargs):
        mystring = super(FeedbackEmcee, self).to_string(*args, **kwargs)
        
        # Check if distributions are normal or uniform
        #scipy.stats.kstest(np.random.uniform(size=1000), 'uniform')[1]
        return mystring
        
    def set_values(self, system):
        pass
    
    @decorators.memoized(clear_when_different=True)
    def get_data(self,  lnproblim=None, burnin=None, thin=None, reshape=True):
        """
        Read in the data file.
        """
        if lnproblim is None:
            lnproblim = self._lnproblim
        if burnin is None:
            burnin = self._burnin
        if thin is None:
            thin = self._thin
        
        self.check_file()
        data = np.loadtxt(self._emcee_file)
        walkers, data, logp = data[burnin::thin,0], data[burnin::thin,1:-1],\
                              data[burnin::thin,-1]
        nwalkers = int(walkers.max() + 1)
        niterations = int(data.shape[0] / nwalkers)
        npars = data.shape[1]
        
        # mask out low probability models
        data[logp<=lnproblim] = np.nan
        
        if reshape:
            # Reshape the array in a convenient format
            data = data.reshape( len(data) / nwalkers, nwalkers, -1)
            logp = logp.reshape(-1, nwalkers)
            walkers = walkers.reshape(-1, nwalkers)
        
        return (walkers, data, logp), (nwalkers, niterations, npars)
            
        
        
    def plot_logp(self, lnproblim=None, burnin=None, thin=None, ax=None):
        """
        Plot the history of logp.
        """
        if ax is None:
            ax = plt.gca()
            
        (walkers, data, logp), (nwalkers, niterations, npars) = self.get_data(lnproblim=None, burnin=None, thin=None)
        
        for i in range(nwalkers):
            ax.plot(logp[:,i], alpha=0.2)
    
    def plot_history(self, qualifier=None, ax=None):
        (walkers, data, logp), (nwalkers, niterations, npars) = self.get_data(lnproblim=None, burnin=None, thin=None)
        
        if qualifier is None:
            pars = self._parameters
            indices = range(len(pars))
        else:
            pars = [self.get_parameter(qualifier)]
            indices = [i for i in range(len(self._parameters)) if self_parameters[i]==pars[0]]
        
        for i, par in zip(indices, pars):
            if ax is None:
                plt.figure()
                ax_ = plt.gca()
            else:
                ax_ = ax
                
            ylabel = self._translate(par.get_unique_label())
            if par.has_unit():
                ylabel += ' ({})'.format(conversions.unit2texlabel(par.get_unit()))
            ax_.set_ylabel(ylabel)
            ax_.set_xlabel('Iteration')
            for w in range(nwalkers):
                ax_.plot(data[:, w, i], alpha=0.2)
            
    
    def plot_summary(self, lnproblim=None, burnin=None, thin=None, bins=20):
        cbins = 20
        fontsize = 8
        (walkers, data, logp), (nwalkers, niterations, npars) = self.get_data(lnproblim=lnproblim, burnin=burnin, thin=thin)
        #npars = 2
        # range of contour levels
        lrange = np.arange(2.5,0.0,-0.5)
        
        # labels
        labels = [self._translate(par.get_unique_label()) for par in self._parameters]
        
        #plot
        f = plt.figure(figsize=(12,12))
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.95,
                            wspace=0.0, hspace=0.0)
        
        # First make a grid of the diagonal axes, they control the axes limits
        axs = []
        for row in range(npars):
            axs.append([])
            for col in range(npars):
                axs[-1].append(plt.subplot(npars, npars, row*npars + col + 1))
                axs[-1][-1].set_autoscale_on(False)
        
        # Then fill the axes
        for row in range(npars):
            for col in range(npars):
                ax = axs[row][col]
                # Get the axes, and share the axes with right diagonal axes
                smpls_x = data[:, :, row].ravel()
                smpls_y = data[:, :, col].ravel()
                keep = -np.isnan(smpls_x) & -np.isnan(smpls_y)
                smpls_x = smpls_x[keep]
                smpls_y = smpls_y[keep]
                this_logp = logp.ravel()[keep]
            
                if row > col:
                    m,xmin,xmax,ymin,ymax = matrixify(smpls_y, smpls_x, this_logp, cbins, cbins)
                    mx = np.max(m)
                    levels = [int(mx*(1-sp.erf(k/np.sqrt(2.0)))) for k in lrange]
                    if levels[-1] < mx: levels.append(mx)
                    
                    #modify this if you don't want smoothing (mp = m)
                    mp = nd.zoom(nd.gaussian_filter(m,1),3)
                    
                    ax.contourf(mp,origin='lower',interpolation='gaussian',extent=(xmin,xmax,ymin,ymax),aspect=(xmax-xmin)/(ymax-ymin),cmap=plt.cm.gray_r,levels=levels)
                    ax.contour(mp,origin='lower',interpolation='gaussian',extent=(xmin,xmax,ymin,ymax),aspect=(xmax-xmin)/(ymax-ymin),colors='k',levels=levels)
                    
                    #if priors is not None and i<len(priors):
                        #axs[ai].set_xlim(priors[i])
                    
                elif col > row:
                    prs = self._cormat[row, col]
                    ax.annotate(r'${:+.2f}$'.format(prs),(0.5,0.5),
                                  xycoords='axes fraction', ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                elif row == col:
                    
                    avg, std = np.average(smpls_x), np.std(smpls_x)
                    out = ax.hist(smpls_x, histtype='step', bins=bins, normed=True)
                    #ax.plo
                    ax.autoscale() # doesn't work very well
                    ax.set_ylim(0, out[0].max()+0.1*out[0].ptp())
                    
                    
                    
                    
                
                    
                if row == npars-1:
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(fontsize) 
                        tick.label.set_rotation('vertical')
                    ax.set_xlabel(labels[col], fontsize=fontsize)
                    ax.get_xaxis().get_major_formatter().set_useOffset(False)
                elif col <= row:
                    plt.setp(ax.get_xticklabels(), visible=False)
                    ax.get_xaxis().get_major_formatter().set_useOffset(False)
                    
                if col == 0:
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label.set_fontsize(fontsize)
                    ax.get_yaxis().get_major_formatter().set_useOffset(False)
                    ax.set_ylabel(labels[row], fontsize=fontsize)
                elif col <= row:
                    plt.setp(ax.get_yticklabels(), visible=False)
                    ax.get_yaxis().get_major_formatter().set_useOffset(False)
                        
                if row == 0:
                    ax.set_title(r'$%.3g\pm%.3g$' % (np.average(smpls_y),np.std(smpls_y)),fontsize=fontsize)
                
        for row in range(npars):
            for col in range(npars):
                axs[row][col].set_autoscale_on(False)
                if row < col:
                    continue
                elif row == col:
                    continue
                else:
                    axs[row][col].set_xlim(axs[col][col].get_xlim())
                    axs[row][col].set_ylim(axs[row][row].get_xlim())
                    #ax.get_xaxis().get_major_formatter().set_useOffset(False)
                    #ax.get_yaxis().get_major_formatter().set_useOffset(False)
                    
        return axs        