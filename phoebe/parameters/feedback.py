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
        if self.fitting is not None:
            return self.fitting['computelabel']
    
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
                    this_par = copy.deepcopy(adjustable_parameters[index])
                except ValueError:
                    continue
                
                if revert:
                    this_par.reset()
                
                # Set priors, posteriors and values
                system_par.set_value(this_par.get_value())
                if this_par.has_prior():
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
        
        
class FeedbackDc(Feedback):
    """
    Feedback from the dc fitting routine
    """
    def __init__(self, dc_result, init, fitting=None,
            compute=None, ongoing=False):

        self._translation = dict()
        self._info = ''
                
        self.fitting = fitting
        self.compute = compute
        self.results = dc_result
        
        # Retrieve the (initial) parameters
        init_phoebe_pars = self.retrieve_parameters(init)
        
        # Set the parameters and remember their state
        self._parameters = copy.deepcopy(init_phoebe_pars)
        for par in self._parameters:
            par.remember()


        for i,value in enumerate(dc_result):
            self._parameters[i].set_value(value)
            mu = value
            sigma = 1.0   # TODO: fix
            self._parameters[i].set_posterior(distribution='normal',
                                              mu=mu, sigma=sigma)
            

        self._cormat = np.zeros((len(init_phoebe_pars), len(init_phoebe_pars)))  # TODO: fix
    

    
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
    
    
class FeedbackLmfit(Feedback):
    """
    Feedback from the lmfit fitting package.
    """
    def __init__(self, lmfit_result, lmfit_pars, extra_info, init, fitting=None,
                 compute=None, ongoing=False):
        
        self._translation = dict()
        self._info = ''
        
        if extra_info is not None:
            self._traces = extra_info['traces']
            self._fit_stats = extra_info['redchis']
        
        # Retrieve the (initial) parameters
        init_phoebe_pars = self.retrieve_parameters(init)
        
        # The parameters that were used to fit (but their initial state)
        self._parameters = copy.deepcopy(init_phoebe_pars)
        for par in self._parameters:
            par.remember()
        
        # Build correlation matrix and set posteriors at the same time
        parlabels = [lmfit_pars[par].name for par in lmfit_pars]
        correls = np.zeros((len(init_phoebe_pars), len(init_phoebe_pars)))
        for i, ipar in enumerate(lmfit_pars):
            for j, jpar in enumerate(lmfit_pars):
                if i==j:
                    correls[i,j] = 1.0
                    value = lmfit_pars[ipar].value
                    mu = lmfit_pars[ipar].value
                    sigma = lmfit_pars[ipar].stderr
                    # Standard error when not available is incredibly small
                    if sigma is None or np.isnan(sigma):
                        sigma = 1e-8*mu
                    self._parameters[i].set_value(value)
                    self._parameters[i].set_posterior(distribution='normal',
                                                      mu=mu, sigma=sigma)
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
                 burnin=0, thin=1, fitting=None, compute=None, ongoing=False):
        
        self._emcee_file = os.path.abspath(emcee_file)
        self._translation = dict()
        self._nwalkers = 0
        self._niterations = 0
        
        # keep track of filesize to see if anything changed
        self._checkfilesize = os.stat(emcee_file).st_size if not ongoing else None
        
        # Retrieve the (initial) parameters
        init_phoebe_pars = self.retrieve_parameters(init)
        
        # Set the parameters and remember their state
        self._parameters = copy.deepcopy(init_phoebe_pars)
        
        for par in self._parameters:
            par.remember()
            
        self._lnproblim = lnproblim
        self._burnin = burnin
        self._thin = thin
        
        self.do_reload()
        self.fitting = fitting
        self.compute = compute
        # Reshape the array in a convenient format
        #data = data.reshape( len(data) / nwalkers, nwalkers, -1)
    
    def check_file(self, raise_error=True):
        """
        Check if the emcee file is still the same one.
        
        We check this by checking if the filesize is still the same.
        """
        # Try to locate the file in this working directory if it does not exist
        if not os.path.isfile(self._emcee_file) and os.path.isfile(os.path.basename(self._emcee_file)):
            self._emcee_file = os.path.basename(self._emcee_file)
            
        samefile = (os.stat(self._emcee_file).st_size == self._checkfilesize) or (self._checkfilesize is None)
        
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
        
        adjustables = system.get_adjustable_parameters(with_priors=True)
        adjustables+= system.get_adjustable_parameters(with_priors=False)
        adjustables+= system.get_parameters_with_priors(is_adjust=False, is_derived=True)
        
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
                translations[adj_ids[index]] = '@'.join(item['twig'].split('@')[:-1])
            
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
        
    def set_lnproblim(self, lnproblim):
        """
        [FUTURE]
        
        change the limit on lnprob.  This will force a reload from the
        chain file, so make sure that file still exists
        
        """
        # make sure lnproblim is negative
        lnproblim = -1 * abs(lnproblim)
        
        logger.info("setting lnproblim to {}".format(lnproblim))
        
        # set value and do reload
        self._lnproblim = lnproblim
        if self.check_file():
            self.do_reload()
    
    
    def do_reload(self):
        """
        Reload the data from the file.
        
        """
        # Load the emcee file and remember some properties            
        data = np.loadtxt(self._emcee_file)
        lnproblim = self._lnproblim
        burnin = self._burnin
        thin = self._thin
        nwalkers = int(data[:,0].max() + 1)
        walkers, data, acc, logp = data[burnin*nwalkers::thin,0],\
                                   data[burnin*nwalkers::thin,1:-2],\
                                   data[burnin*nwalkers::thin,-2],\
                                   data[burnin*nwalkers::thin,-1]
        
        niterations = int(data.shape[0] / nwalkers)
        npars = data.shape[1]
        
        self._nwalkers = nwalkers
        self._niterations = niterations
        
        myinfo = 'EMCEE {}'.format(self._emcee_file)
        myinfo += '\n' + len(myinfo)*'=' + '\n'
        myinfo += ("Number of walkers: {}".format(nwalkers)) + '\n'
        myinfo += ("Number of iterations: {}".format(niterations)) + '\n'
        myinfo += ("Number of parameters: {}".format(npars)) + '\n'
        myinfo += ("Maximum lnprob: {}".format(logp.max())) + '\n'
        myinfo += ("Cut-off lnprob: {}".format(lnproblim)) + '\n'
        myinfo += ("Burn-in: {}".format(burnin)) + '\n'
        myinfo += ("Thinning: {}".format(thin)) + '\n'
        myinfo += ("Median acceptance fraction: {}".format(np.median(acc))) + '\n'
                   
        self._info = myinfo
        
        select = (logp >= lnproblim)
        data = data[select]
        walkers = walkers[select]
        logp = logp[select]
        acc = acc[select]
        index = np.argmax(logp)
        
        sigmas = np.std(data, axis=0)
        correl = np.zeros((npars, npars))
        
        for i, ipar in enumerate(self._parameters):
            for j, jpar in enumerate(self._parameters):
                if i==j:
                    correl[i,j] = 1.0
                    #keep = -np.isnan(data[:,i]) & -np.isinf(data[:,i])
                    self._parameters[i].set_value(data[:,i][index])
                    self._parameters[i].set_posterior(distribution='trace',
                                                      trace=data[:,i])
                else:
                    keepi = -np.isnan(data[:,i]) & -np.isinf(data[:,i])
                    keepj = -np.isnan(data[:,j]) & -np.isinf(data[:,j])
                    keep = keepi & keepj
                    prs = st.spearmanr(data[keep, i], data[keep, j])[0]
                    correl[i,j] = prs #* sigmas[i] * sigmas[j]
        
        self._cormat = correl

        
    
    def to_string(self, *args, **kwargs):
        mystring = super(FeedbackEmcee, self).to_string(*args, **kwargs)
        
        # Check if distributions are normal or uniform
        #scipy.stats.kstest(np.random.uniform(size=1000), 'uniform')[1]
        return mystring
        
    def set_values(self, system):
        pass
    
    def modify_chain(self, lnproblim=None, burnin=None, thin=None):
        if lnproblim is not None:
            self._lnproblim = lnproblim
        if burnin is not None:
            self._burnin = burnin
        if thin is not None:
            self._thin = thin
        decorators.clear_memoization(keys=['phoebe.parameters.feedback'])
        self.do_reload()
    
    @decorators.memoized(clear_when_different=True)
    def get_data(self,  reshape=True):
        """
        Read in the data file.
        """
        lnproblim = self._lnproblim
        burnin = self._burnin
        thin = self._thin
        
        self.check_file()
        data = np.loadtxt(self._emcee_file)
        logger.info("Loaded {}".format(self._emcee_file))
        nwalkers = int(data[:,0].max() + 1)
        walkers, data, acc, logp = data[nwalkers*burnin::thin,0],\
                                   data[nwalkers*burnin::thin,1:-2],\
                                   data[nwalkers*burnin::thin,-2],\
                                   data[nwalkers*burnin::thin,-1]
        niterations = int(data.shape[0] / nwalkers)
        npars = data.shape[1]
        
        # mask out low probability models
        mask = logp<=lnproblim
        data[mask] = np.nan
        logp[mask] = np.nan
        acc[mask] = np.nan
        
        if reshape:
            # Reshape the array in a convenient format
            data = data.reshape( len(data) / nwalkers, nwalkers, -1)
            logp = logp.reshape(-1, nwalkers)
            acc = acc.reshape(-1, nwalkers)
            walkers = walkers.reshape(-1, nwalkers)
            
        
        return (walkers, data, acc, logp), (nwalkers, niterations, npars)
            
        
        
    def plot_logp(self, ax=None):
        """
        Plot the history of logp.
        """
        if ax is None:
            ax = plt.gca()
            
        (walkers, data, acc, logp), (nwalkers, niterations, npars) = self.get_data()
        
        for i in range(nwalkers):
            ax.plot(logp[:,i], alpha=0.2)
        
        plt.xlabel("Iteration number")
        plt.ylabel("log(Probability) [dex]")
        plt.title("Probability history")
    
    def plot_acceptance_fraction(self, ax=None):
        """
        Plot the acceptance fraction vs iteration number.
        """
        if ax is None:
            ax = plt.gca()
            
        (walkers, data, acc, logp), (nwalkers, niterations, npars) = self.get_data()
        
        for i in range(nwalkers):
            ax.plot(acc[:,i], alpha=0.3)
        
        plt.axhspan(0.2, 0.5, color='g', alpha=0.1)
        
        plt.xlabel("Iteration number")
        plt.ylabel("Acceptance fraction")
        plt.title("Acceptance fraction")
        
    
    def plot_history(self, qualifier=None, ax=None):
        (walkers, data, acc, logp), (nwalkers, niterations, npars) = self.get_data()
        
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
            
            if par.has_prior():
                par.get_prior().plot(on_axis='y', alpha=0.1, color='r')
                
            ylabel = self._translate(par.get_unique_label())
            if par.has_unit():
                ylabel += ' ({})'.format(conversions.unit2texlabel(par.get_unit()))
            ax_.set_ylabel(ylabel)
            ax_.set_xlabel('Iteration')
            for w in range(nwalkers):
                ax_.plot(data[:, w, i], alpha=0.2)
            
    
    def plot_summary(self, axes=None, figsize=(12,12), fontsize=8, 
                    color_priors=True, color_correlations=True, label_posterior=True,
                    twig_labels=True, label_offset=None, skip_inds=[]):
        """
        
        :param axes: matplotlib axes to use (or None to create automatically)
        :type axes: mpl axes
        :param figsize: size for the matplotlib figure, defaults to (12,12)
        :type figsize: tuple
        :param fontsize: size to use for axes labels, etc
        :type fontsize: int
        :param color_priors: whether to show red shading on diagonal elements to show limits of priors (this set to true will force the xlimits to extend to the width of the prior)
        :type color_priors: bool
        :param color_correlations: whether to show green shades for strong correlations
        :type color_correlations: bool
        :param label_posterior: whether to show the posterior (value and error) at the top of the figure
        :type label_posterior: bool
        :param twig_labels: True to create labels from twigs, list to provide labels (must be same length as ALL parameters, include labels for skipped parameters), False to show index number
        :type twig_labels: bool or list
        :param label_offset: offset to apply to the axes labels, in axes units (probably negative), or None to set automatically
        :type label_offset: int or None
        :param skip_inds: indices of parameters to skip
        :type skip_inds: list of ints
        """
        cbins = 20
        (walkers, data, acc, logp), (nwalkers, niterations, npars) = self.get_data()
        #npars = 2
        # range of contour levels
        lrange = np.arange(2.5,0.0,-0.5)

        rows, cols = range(npars), range(npars)

        # labels
        if isinstance(twig_labels, list) and len(twig_labels) == len(rows):
            labels = twig_labels
        elif twig_labels:
            labels = [self._translate(par.get_unique_label()) for par in self._parameters]
        else:
            labels = ['( {} )'.format(i) for i in range(len(self._parameters))]
        
        # handle skipping parameters
        npars = npars - len(skip_inds)
        skip_inds.sort(reverse=True)
        for skip_ind in skip_inds:
            rows.remove(skip_ind)
            cols.remove(skip_ind)

        #plot
        f = plt.figure(figsize=figsize)
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.95,
                            wspace=0.0, hspace=0.0)
        
        
       
        # First make a grid of the diagonal axes, they control the axes limits
        if not axes:
            axs = []
            for row_i, row in enumerate(rows):
                axs.append([])
                for col_i, col in enumerate(cols):
                    axs[-1].append(plt.subplot(npars, npars, row_i*npars + col_i + 1))
                    axs[-1][-1].set_autoscale_on(False)
        else:
            axs = axes
        
        # Then fill the axes
        for row_i, row in enumerate(rows):
            for col_i, col in enumerate(cols):
                ax = axs[row_i][col_i]
                # Get the axes, and share the axes with right diagonal axes
                smpls_x = self._parameters[row].get_posterior().get_distribution()[1]['trace']#data[:, :, row].ravel()
                smpls_y = self._parameters[col].get_posterior().get_distribution()[1]['trace']#data[:, :, col].ravel()
                keep = -np.isnan(smpls_x) & -np.isnan(smpls_y)
                smpls_x = smpls_x[keep]
                smpls_y = smpls_y[keep]
                
                # Skip if parameter did not vary over run
                do_skip = False
                if smpls_x.std()==0 or smpls_y.std()==0:
                    do_skip = True
                
                this_logp = logp.ravel()[keep]            
                if row > col and not do_skip:
                    m,xmin,xmax,ymin,ymax = matrixify(smpls_y, smpls_x, this_logp, cbins, cbins)
                    mx = np.max(m)
                    levels = [int(mx*(1-sp.erf(k/np.sqrt(2.0)))) for k in lrange]
                    if levels[-1] < mx: levels.append(mx)
                    
                    #modify this if you don't want smoothing (mp = m)
                    mp = nd.zoom(nd.gaussian_filter(m,1),3)
                    
                    ax.contourf(mp,origin='lower',interpolation='gaussian',extent=(xmin,xmax,ymin,ymax),aspect=(xmax-xmin)/(ymax-ymin),cmap=plt.cm.gray_r,levels=levels)
                    ax.contour(mp,origin='lower',interpolation='gaussian',extent=(xmin,xmax,ymin,ymax),aspect=(xmax-xmin)/(ymax-ymin),colors='k',levels=levels)
                    
                    prs = self._cormat[col, row]
                    if color_correlations:
                        if np.abs(prs)>=0.75:
                            ax.set_axis_bgcolor((0.6,1.0,0.6))
                        elif np.abs(prs)>=0.50:
                            ax.set_axis_bgcolor((0.9,1.0,0.9))
                    
                    #if priors is not None and i<len(priors):
                        #axs[ai].set_xlim(priors[i])
                    
                elif col > row:
                    prs = self._cormat[row, col]
                    ax.annotate(r'${:+.2f}$'.format(prs),(0.5,0.5),
                                  xycoords='axes fraction', ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                elif row == col and not do_skip:
                    
                    avg, std = np.average(smpls_x), np.std(smpls_x)
                    out = np.histogram(smpls_x, density=True, bins=32)
                    
                    # Add posterior information
                    self._parameters[row].get_posterior().plot(ax=ax, color='b')
                    
                    # Add prior information
                    if self._parameters[row].has_prior() and color_priors:
                        self._parameters[row].get_prior().plot(color='r', alpha=0.2, ax=ax)
                    
                    # Set axes
                    ax.autoscale() # doesn't work very well
                    ax.set_ylim(0, out[0].max()+0.1*out[0].ptp())
                
                if row_i == npars-1:
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(fontsize) 
                        tick.label.set_rotation('vertical')
                    ax.set_xlabel(labels[col], fontsize=fontsize)
                    if label_offset:
                        ax.xaxis.set_label_coords(0.5, label_offset)
                    ax.get_xaxis().get_major_formatter().set_useOffset(False)
                elif col <= row:
                    plt.setp(ax.get_xticklabels(), visible=False)
                    ax.get_xaxis().get_major_formatter().set_useOffset(False)
                    
                if col == 0:
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label.set_fontsize(fontsize)
                    ax.get_yaxis().get_major_formatter().set_useOffset(False)
                    ax.set_ylabel(labels[row], fontsize=fontsize)
                    if label_offset:
                        ax.yaxis.set_label_coords(label_offset, 0.5)
                elif col <= row:
                    plt.setp(ax.get_yticklabels(), visible=False)
                    ax.get_yaxis().get_major_formatter().set_useOffset(False)
                        
                if row == 0:
                    if label_posterior:
                        ax.set_title(r'$%.3g\pm%.3g$' % (np.average(smpls_y),np.std(smpls_y)),fontsize=fontsize)
            
        for row_i, row in enumerate(rows):
            for col_i, col in enumerate(cols):
                axs[row_i][col_i].set_autoscale_on(False)
                if row_i < col_i:
                    continue
                elif row_i == col_i:
                    continue
                else:
                    axs[row_i][col_i].set_xlim(axs[col_i][col_i].get_xlim())
                    axs[row_i][col_i].set_ylim(axs[row_i][row_i].get_xlim())
                    #ax.get_xaxis().get_major_formatter().set_useOffset(False)
                    #ax.get_yaxis().get_major_formatter().set_useOffset(False)
                    
        return axs
    
    def __len__(self):
        return self._niterations
