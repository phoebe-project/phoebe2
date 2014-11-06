"""
Definition of statistical or empirical distributions.

The following definitions are defined:

.. autosummary::

    Uniform
    Normal
    Trace

Each distribution has the following attributes:

    - :envvar:`distribution`: a string containing the name of the distribution (e.g. ``normal``)
    - :envvar:`distr_pars`: parameters determining the distributions (e.g. ``mu`` and ``sigma``)

Each distribution has the following methods:

.. autosummary::

    BaseDistribution.pdf
    BaseDistribution.cdf
    Normal.draw
    Normal.get_loc
    Normal.get_scale
    Normal.get_limits
    Normal.get_grid
    Normal.get_distribution
    


"""
import numpy as np
from phoebe.units import conversions
try:
    import pymc
except ImportError:
    pass
from scipy.stats import distributions
import matplotlib.pyplot as plt

def histogram_bins(signal):
    """
    Return optimal number of bins.
    """
    select = -np.isnan(signal) & -np.isinf(signal)
    h = (3.5 * np.std(signal[select])) / (len(signal[select])**(1./3.))
    bins = int(np.ceil((max(signal[select])-min(signal[select])) / h))*2
    
    return bins

class Distribution(object):
    """
    Factory to produce new distributions
    """
    def __new__(cls, *args, **kwargs):
        if 'distribution' in kwargs:
            dist_name = kwargs.pop('distribution').lower()
        else:
            dist_name = args[0].lower()
            args = args[1:]
            
        if dist_name == 'normal':
            return Normal(*args, **kwargs)
        elif dist_name == 'uniform':
            return Uniform(*args, **kwargs)
        elif dist_name == 'trace':
            return Trace(*args, **kwargs)    
        else:
            raise ValueError("Do not recognise distribution {}".format(dist_name))


class BaseDistribution(object):
    """
    Base class for Distributions.
    """
    def pdf(self, domain=None, **kwargs):
        """
        Return the probability density function.
        
        Extra optional keyword arguments can be shrink factors etc, please see
        the documentation of the specific distributions.
        
        :param domain: domain in which to evaluate the PDF
        :type domain: array
        :return: probability density function
        :rtype: array
        """
        return self.get_distribution(distr_type='pdf', domain=domain, **kwargs)
    
    def cdf(self, domain=None, **kwargs):
        """
        Return the cumulative density function.
        
        Extra optional keyword arguments can be shrink factors etc, please see
        the documentation of the specific distributions.
        
        :param domain: domain in which to evaluate the CDF
        :type domain: array
        :return: cumulative density function
        :rtype: array
        """
        return self.get_distribution(distr_type='cdf', domain=domain, **kwargs)

    def update_distribution_parameters(self, **distribution_parameters):
        """
        Update the distribution parameters.
        
        When a sample is updated, the values get appended to the previous ones
        In all other cases, the keys are overwritten.
        """
        for key in distribution_parameters:
            if not key in self.distr_pars:
                raise ValueError('Distribution {} does not accept key {}'.format(self.distribution, key))
            elif self.distribution == 'sample' and key == 'sample':
                self.distr_pars[key] = np.hstack([self.distr_pars[key], distribution_parameters[key]])
            else:
                self.distr_pars[key] = distribution_parameters[key]
     
    def get_name(self):
        """
        Return the name of the distribution as a string
        """
        return self.distribution.title()
    
    
    def __str__(self):
        """
        String representation of class Distribution.
        """
        name = self.get_name()
        pars = ", ".join(['{}={}'.format(key,self.distr_pars[key]) for key in sorted(list(self.distr_pars.keys()))])
        return "{}({})".format(name,pars)


class DeprecatedDistribution(object):
    """
    Class representing a distribution.
    
    .. autosummary::
    
        cdf
        pdf
        get_distribution
        get_limits
        get_loc
        get_scale
        shrink
        expand
        draw
        
        update_distribution_parameters
        convert
        transform_to_unbounded
        
        
        
    
    Attempts at providing a uniform interface to different codes or routines.
    
    **Recognised distributions**:
    
    >>> d = Distribution('normal',mu=0,sigma=1)
    >>> d = Distribution('uniform',lower=0,upper=1)
    >>> d = Distribution('histogram',bins=array1,prob=array2,discrete=True)
    >>> d = Distribution('histogram',bins=array1,prob=array2,discrete=False)
    >>> d = Distribution('sample',sample=array3,discrete=False)
    >>> d = Distribution('discrete',values=array4)
    
    If C{distribution='histogram'} and C{discrete=False}, the bins are interpreted
    as the edges of the bins, and so there is one more bin than probabilities.
    
    If C{distribution='histogram'} and C{discrete=True}, the bins are
    interpreted as the set of discrete values, and so there are equally many
    bins as probabilities. It is possible to use ``valuess`` as an alias for
    bins here, then internally that parameter will be converted to ``bins``. 
    
    """
    def __init__(self, distribution, **distribution_parameters):
        """
        Initialize a distribution.
        
        @param distribution: type of the distribution
        @type distribution: str
        """
        distribution = distribution.lower()
        self.distribution = distribution
        self.distr_pars = distribution_parameters
        
        #-- discrete can be equivalent to histogram
        if distribution == 'discrete':
            distribution = 'histogram'
            distribution_parameters['discrete'] = True
            distribution_parameters['prob'] = np.ones(len(distribution_parameters['values']))
            if 'values' in distribution_parameters:
                distribution_parameters['bins'] = distribution_parameters.pop('values')
        
        #-- check contents:
        given_keys = set(list(distribution_parameters.keys()))
        
        if   distribution == 'histogram':
            required_keys = set(['bins', 'prob', 'discrete'])
        elif distribution == 'uniform':
            required_keys = set(['lower', 'upper'])
        elif distribution == 'normal':
            required_keys = set(['mu', 'sigma'])
        elif distribution == 'sample':
            required_keys = set(['sample', 'discrete'])
        elif distribution == 'trace':
            required_keys = set(['trace'])
        else:
            raise NotImplementedError("Cannot recognize distribution {}".format(distribution))
        
        if (given_keys - required_keys) or (required_keys - given_keys):
            raise ValueError('Distribution {} needs keys {} (not {})'.format(distribution, ", ".join(list(required_keys)), ', '.join(given_keys)))
        
        #-- make sure the histogram is properly normalised
        if distribution=='histogram':
            self.distr_pars['prob'] = self.distr_pars['prob']/float(np.sum(self.distr_pars['prob']))
    
    def update_distribution_parameters(self, **distribution_parameters):
        """
        Update the distribution parameters.
        
        When a sample is updated, the values get appended to the previous ones
        In all other cases, the keys are overwritten.
        """
        for key in distribution_parameters:
            if not key in self.distr_pars:
                raise ValueError('Distribution {} does not accept key {}'.format(self.distribution, key))
            elif self.distribution == 'sample' and key == 'sample':
                self.distr_pars[key] = np.hstack([self.distr_pars[key], distribution_parameters[key]])
            else:
                self.distr_pars[key] = distribution_parameters[key]
                
    
    def get_distribution(self, distr_type=None, **kwargs):
        """
        Return the distribution in a specific form.
        
        When ``distr_type='pymc'``, you need to supply the keyword C{name} in
        the kwargs.
        
        Possibilities:
        
            - ``distr_type=None``: returns the name and a dictionary with the
              distribution parameters.
            - ``distr_type='pymc'``: return the distribution as a pymc parameter.
            - ``pdf``: probability density function (x and y between limits)
            - ``cdf``: cumulative density function (x and y between limits)
        
        @param distr_type: format of the distribution
        @type distr_type: None or str
        """
        #-- plain
        if distr_type is None:
            return self.distribution,self.distr_pars
        #-- pymc
        elif distr_type=='pymc':
            if self.distribution=='normal':
                kwargs['mu'] = self.distr_pars['mu']
                kwargs['tau'] = 1./self.distr_pars['sigma']**2
            else:
                for key in self.distr_pars:
                    kwargs.setdefault(key,self.distr_pars[key])
            return getattr(pymc,self.distribution.title())(**kwargs)
        else:
            L,U = self.get_limits()
            if self.distribution=='normal':
                x = kwargs.get('x',np.linspace(L,U,1000))
                return x,getattr(distributions.norm(loc=self.distr_pars['mu'],scale=self.distr_pars['sigma']),distr_type)(x)
            elif self.distribution=='uniform':
                x = kwargs.get('x',np.linspace(L,U,1000))
                return x,getattr(distributions.uniform(loc=L,scale=U),distr_type)(x)
            elif self.distribution=='histogram':
                bins = self.distr_pars['bins']
                bins = bins[:-1] + np.diff(bins)
                return bins,self.distr_pars['prob']
            elif self.distribution=='sample' or self.distribution=='histogram':
                return self.distr_pars['bins'],self.distr_pars['prob']
            
    def convert(self,from_unit,to_unit,*args):
        """
        Convert a distribution to different units.
        
        We cannot `simply` convert a distribution, because the conversion can
        be nonlinear (e.g. from linear to log scale). Therefore, we first get
        the PDF, and convert that one. Unfortunately, this also means that
        you can only convert continuous distributions. Extension to discrete
        ones should be fairly straightforward though (just an
        ``if==distr_pars['discrete']``) thingy. But I don't feel like it
        today.
        """
        logger.info('Converting {} distr. in {} to histogram distr. in {}'.format(self.distribution,from_unit,to_unit))
        old_x,old_y = self.pdf()
        old_dx = np.diff(old_x)
        x_shifted = np.hstack([old_x[:-1] - np.diff(old_x)/2.,old_x[-2:] + np.diff(old_x)[-1]/2.])
        #-- we definitely cannot go over the limits!
        x_shifted[0] = old_x[0]
        x_shifted[-1] = old_x[-1]
        old_dx = np.diff(x_shifted)
        
        if from_=='bounded':
            #old_x[old_x<args[0]] = args[0]
            #old_x[old_x>args[1]] = args[1]
            #x_shifted[x_shifted<args[0]] = args[0]
            #x_shifted[x_shifted>args[1]] = args[1]
            new_x = transform_to_unbounded(old_x,*args)
            new_x_shifted = transform_to_unbounded(x_shifted,*args)
        else:
            new_x = conversions.convert(from_unit,to_unit,old_x)
            new_x_shifted = conversions.convert(from_unit,to_unit,x_shifted)
        new_dx = np.diff(new_x_shifted)
        new_y = old_dx/new_dx*old_y
        norm = np.trapz(new_y,x=new_x)
        new_y = new_y/norm
        self.distribution = 'histogram'
        self.distr_pars = dict(discrete=False,bins=new_x_shifted,prob=new_y)
    
    
    def pdf(self,**kwargs):
        """
        Return the probability density function.
        """
        return self.get_distribution(distr_type='pdf',**kwargs)
    
    def cdf(self,**kwargs):
        """
        Return the cumulative density function.
        """
        return self.get_distribution(distr_type='cdf',**kwargs)
    
    def get_limits(self,factor=1.0):
        """
        Return the minimum and maximum of a distribution.
        
        These are the possibilities:
        
            - **uniform**: lower and upper values
            - **normal**: mean +/- 3 x sigma
            - **sample** or **histogram**: min and max values.
            
        """
        if self.distribution=='uniform':
            lower = self.distr_pars['lower']
            upper = self.distr_pars['upper']
            window = upper - lower
            mean = (lower + upper) / 2.0
            lower = mean - factor * window/2.0
            upper = mean + factor * window/2.0  
        elif self.distribution=='normal':
            lower = self.distr_pars['mu']-3*self.distr_pars['sigma']*factor
            upper = self.distr_pars['mu']+3*self.distr_pars['sigma']*factor
        elif self.distribution=='sample' or self.distribution=='histogram':
            lower = self.distr_pars['bins'].min()
            upper = self.distr_pars['bins'].max()
            window = upper - lower
            mean = (lower + upper) / 2.0
            lower = mean - factor * window/2.0
            upper = mean + factor * window/2.0 
        else:
            raise NotImplementedError
        return lower,upper
    
    def draw(self,size=1):
        """
        Draw a (set of) random value(s) from the distribution.
        
        @param size: number of values to generate
        @type size: int
        @return: random value from the distribution
        @rtype: array[C{size}]
        """
        if self.distribution=='uniform':
            values = np.random.uniform(size=size,low=self.distr_pars['lower'],
                                 high=self.distr_pars['upper'])
        elif self.distribution=='normal':
            values = np.random.normal(size=size,loc=self.distr_pars['mu'],
                                 scale=self.distr_pars['sigma'])
        elif self.distribution=='histogram' or self.distribution=='sample':
            #-- when the Distribution is actually a sample, we need to make
            #   the histogram first ourselves.
            if self.distribution=='sample':
                myhist = np.histogram(self.distr_pars['sample'],normed=True)
                if self.distr_pars['discrete']:
                    bins = np.unique(self.distr_pars['sample'])
                else:
                    bins = myhist[1]
                prob = myhist[0]
            #-- else they are readily available
            else:
                bins,prob = self.distr_pars['bins'],self.distr_pars['prob']
            #-- draw random uniform values from the cumulative distribution
            cumul = np.cumsum(prob*np.diff(bins))#*np.diff(bins)
            cumul /= cumul.max()
            locs = np.random.uniform(size=size)
            
            #plt.figure()
            #plt.plot(bins[:-1],cumul,'ko-')
            #plt.show()
            
            #-- little different for discrete or continuous distributions
            if self.distr_pars['discrete']:
                values = bins[indices-1]
            else:
                values = np.interp(locs,cumul,bins[1:])

        else:
            raise NotImplementedError
            
        return values
    
    def shrink(self, factor=10.0):
        """
        Shrink the extent of a distribution.
        
        @param factor: factor with which to shrink the distribution
        @type factor: float
        """
        if self.distribution == 'uniform':
            lower, upper = self.distr_pars['lower'], self.distr_pars['upper']
            mean = (lower + upper) /2.0
            width = upper - lower
            self.distr_pars['lower'] = mean - width/2.0/factor
            self.distr_pars['upper'] = mean + width/2.0/factor
        elif self.distribution == 'normal':
            self.distr_pars['sigma'] = self.distr_pars['sigma'] / factor
        else:
            raise NotImplementedError
    
    def expand(self, factor=10.0):
        """
        Expand the extent of a distribution.
        
        @param factor: factor with which to expand the distribution
        @type factor: float
        """
        if self.distribution == 'uniform':
            lower, upper = self.distr_pars['lower'], self.distr_pars['upper']
            mean = (lower + upper) /2.0
            width = upper - lower
            self.distr_pars['lower'] = mean - width/2.0*factor
            self.distr_pars['upper'] = mean + width/2.0*factor
        elif self.distribution == 'normal':
            self.distr_pars['sigma'] = self.distr_pars['sigma'] * factor
        else:
            raise NotImplementedError
    
    def transform_to_unbounded(self,low,high):
        """
        Transform the distribution to be bounded.
        
        - Uniform distribution set bounds between -100 and +100
        - Normal distribution set mu=0.0, sigma=2.0
        - Sample distribution transformed to unbounded
        - Histogram distribution transformed to unbounded
        """
        #-- problems with conversions to SI
        #sample = self.draw(size=1000)
        #transf_sample = transform_to_unbounded(sample,low,high)
        #self.distribution = 'sample'
        #self.distr_pars = dict(sample=transf_sample,discrete=False)
        
        if self.distribution=='uniform':
            self.distr_pars['lower'] = -5
            self.distr_pars['upper'] = +5
        elif self.distribution=='normal':
            self.distr_pars['mu'] = 0.
            self.distr_pars['sigma'] = 2.
        elif self.distribution=='sample' or self.distribution=='histogram':
            self.convert('bounded','unbounded',low,high)
            #self.distr_pars['bins'] = transform_to_unbounded(self.distr_pars['bins'],low-1e-3*low,high+1e-3*high)
        else:
            raise NotImplementedError
            
        
        
    def __str__(self):
        """
        String representation of class Distribution.
        """
        name = self.distribution.title()
        pars = ", ".join(['{}={}'.format(key,self.distr_pars[key]) for key in sorted(list(self.distr_pars.keys()))])
        return "{}({})".format(name,pars)
    

    def get_loc(self):
        return np.nan
    
    def get_scale(self):
        if self.distribution == 'normal':
            return self.distr_pars['sigma']
        else:
            return np.nan
        
    def get_name(self):
        return self.distribution.title()


class Normal(BaseDistribution):
    """
    The Normal distribution.
    
    The normal distribution has two parameters in its :envvar:`distr_pars` attribute:
    
        - :envvar:`mu` (:math:`\mu`): the location parameter (the mean)
        - :envvar:`sigma` (:math:`\sigma`): the scale parameter (standard deviation)
    
    Example usage::
    
        >>> norm = Normal(5.0, 1.0)
        >>> print(norm)
        Normal(mu=5.0, sigma=1.0)
        
    The limits for this distribution are by default defined as the 3 sigma
    limits::
    
        >>> print(norm.get_limits())
        (2.0, 8.0)
    
    Draw 5 random values from the distribution::
    
        >>> print(norm.draw(5))
        [ 3.98397393  5.25521791  4.221863    5.14080982  5.33466166]
    
    Get a grid of values, equidistantly sampled in probability:
    
        >>> print(norm.get_grid(11))
        [ 6.38299413  5.96742157  5.67448975  5.4307273   5.21042839  5.
          4.78957161  4.5692727   4.32551025  4.03257843  3.61700587]        
        
    """
    def __init__(self, mu, sigma):
        """
        Initiate a normal distribution.
        
        @param mu: location parameter (mean)
        @type mu: float
        @param sigma: scale parameter (standard deviation)
        @type sigma: float
        """
        self.distribution = 'normal'
        self.distr_pars = dict(mu=mu, sigma=sigma)
    
    
    def get_limits(self, factor=1.0):
        """
        Return the minimum and maximum of the normal distribution.
        
        For a normal distribution, we define the upper and lower limit as the
        3*factor-sigma limits.
        """
        lower = self.distr_pars['mu'] - 3*self.distr_pars['sigma']*factor
        upper = self.distr_pars['mu'] + 3*self.distr_pars['sigma']*factor
        return lower, upper
    
    
    def draw(self, size=1):
        """
        Draw a (set of) random value(s) from the normal distribution.
        
        @param size: number of values to generate
        @type size: int
        @return: random value from the distribution
        @rtype: array[C{size}]
        """
        values = np.random.normal(size=size,loc=self.distr_pars['mu'],
                                  scale=self.distr_pars['sigma'])
        return values
    
    
    def get_grid(self, sampling=5):
        """
        Draw regularly sampled values from the distribution, spaced according to probability density.
        
        We grid uniformly in the cumulatie probability density function.
        """
        loc = self.get_loc()
        scale = self.get_scale()
        cum_sampling = np.linspace(0, 1, sampling+2)[1:-1]
        mygrid = distributions.norm(loc=loc, scale=scale).isf(cum_sampling)
        return mygrid
    
    
    def get_distribution(self, distr_type=None, domain=None, factor=1.0):
        """
        Return the distribution in some custom format.
        
        By default, the format is just the name and distribution parameters
        (attributes :envvar:`distribution` and :envvar:`distr_pars`).
        
        Other options:
            - ``distr_type='pdf'``: return the probability density function.
              If you do not give :envvar:``domain`` (the value(s) to evaluate the
              probability density function at), then the domain will be chosen
              to be an array of 1000 points equidistanly sampled between the
              :py:func:`limits <Normal.get_limits>` (which can be expanded or
              contracted with :envvar:`factor`. Returns tuple (domain, pdf) 
        """
        if distr_type is None:
            return self.distribution, self.distr_pars
        
        elif distr_type in ['pdf', 'cdf']:
            
            # If 'domain' is not given, sample equidistantly between the limits
            # of the distribution
            if domain is None:
                lower, upper = self.get_limits(factor=factor)
                domain = np.linspace(lower, upper, 1000)
            
            return domain, getattr(distributions.norm(loc=self.distr_pars['mu'],
                                                 scale=self.distr_pars['sigma']),
                                                 distr_type)(domain)
           
        # Else, we don't know what to do.
        else:
            raise ValueError("Do not understand distr_type='{}'".format(distr_type))
    


    def shrink(self, factor=10.0):
        """
        Shrink the extent of the normal distribution.
        
        @param factor: factor with which to shrink the distribution
        @type factor: float
        """
        self.distr_pars['sigma'] = self.distr_pars['sigma'] / factor
        
    
    def expand(self, factor=10.0):
        """
        Expand the extent of the normal distribution.
        
        @param factor: factor with which to expand the distribution
        @type factor: float
        """
        self.distr_pars['sigma'] = self.distr_pars['sigma'] * factor
    
    
    def transform_to_unbounded(self, low, high):
        """
        Transform the distribution to be bounded.
        
        Set mu=0.0, sigma=2.0
        """
        self.distr_pars['mu'] = 0.
        self.distr_pars['sigma'] = 2.
    
    def get_loc(self):
        """
        Get location parameter.
        """
        return self.distr_pars['mu']
    
    
    def get_scale(self):
        """
        Get scale parameter.
        """
        return self.distr_pars['sigma']
    
   
    def plot(self, on_axis='x', **kwargs):
        """
        Plot a normal prior to the current axes.
        """
        ax = kwargs.pop('ax', plt.gca())
        lower, upper = self.get_limits()
        domain = np.linspace(lower, upper, 500)
        domain, mypdf = self.pdf(domain=domain)
        if on_axis.lower()=='y':
            domain, mypdf = mypdf, domain
        getattr(ax, 'fill_between')(domain, mypdf, **kwargs)
        
        
        

class Uniform(BaseDistribution):
    """
    The Uniform distribution.
    """
    def __init__(self, lower, upper):
        self.distribution = 'uniform'
        self.distr_pars = dict(lower=lower, upper=upper)        
    
    
    def get_limits(self, factor=1.0):
        """
        Return the minimum and maximum of the uniform distribution.
        
        """
        lower = self.distr_pars['lower']
        upper = self.distr_pars['upper']
        window = upper - lower
        mean = (lower + upper) / 2.0
        lower = mean - factor * window/2.0
        upper = mean + factor * window/2.0  
        
        return lower, upper
    
        
    def draw(self, size=1):
        """
        Draw a (set of) random value(s) from the uniform distribution.
        
        @param size: number of values to generate
        @type size: int
        @return: random value from the distribution
        @rtype: array[C{size}]
        """
        values = np.random.uniform(size=size,low=self.distr_pars['lower'],
                                 high=self.distr_pars['upper'])
        return values
    
    
    def get_grid(self, sampling=5):
        """
        Draw a (set of) likely value(s) from the normal distribution.
        
        We grid uniformly in 
        """
        lower = self.distr_pars['lower']
        upper = self.distr_pars['upper']
        mygrid = np.linspace(lower, upper, sampling)
        
        return mygrid    
    
    
    def get_distribution(self, distr_type=None, domain=None, factor=1.0):
        """
        Return the distribution in some custom format.
        
        By default, the format is just the name and distribution parameters
        (attributes :envvar:`distribution` and :envvar:`distr_pars`).
        
        Other options:
            - ``distr_type='pdf'``: return the probability density function.
              If you do not give :envvar:``domain`` (the value(s) to evaluate the
              probability density function at), then the domain will be chosen
              to be an array of 1000 points equidistanly sampled between the
              :py:func:`limits <Uniform.get_limits>` (which can be expanded or
              contracted with :envvar:`factor`. Returns tuple (domain, pdf) 
        """
        if distr_type is None:
            return self.distribution, self.distr_pars
        
        elif distr_type in ['pdf', 'cdf']:
            
            # If 'domain' is not given, sample equidistantly between the limits
            # of the distribution
            if domain is None:
                lower, upper = self.get_limits(factor=factor)
                domain = np.linspace(lower, upper, 1000)
            
            return domain, getattr(distributions.uniform(loc=self.distr_pars['lower'],
                                                 scale=self.distr_pars['upper']-self.distr_pars['lower']),
                                                 distr_type)(domain)
           
        # Else, we don't know what to do.
        else:
            raise ValueError("Do not understand distr_type='{}'".format(distr_type))
    
    
    def plot(self, on_axis='x', **kwargs):
        """
        Plot a uniform prior to the current axes.
        """
        ax = kwargs.pop('ax', plt.gca())
        lower, upper = self.get_limits()
        if on_axis.lower()=='x':
            plot_func = 'axvspan'
        elif on_axis.lower()=='y':
            plot_func = 'axhspan'
        
        getattr(ax, plot_func)(lower, upper, **kwargs)
        
    
    def get_loc(self):
        """
        Get location parameter.
        """
        lower = self.distr_pars['lower']
        return lower
    
    
    def get_scale(self):
        """
        Get scale parameter.
        """
        lower = self.distr_pars['lower']
        upper = self.distr_pars['upper']
        
        return (upper - lower)
        

class Trace(BaseDistribution):
    """
    A distribution from a trace.
    
    The Trace Distribution has one parameter in its :envvar:`distr_pars` attribute:
    
        - :envvar:`trace`: the trace array
    
    Example usage::
    
        >>> trace = Trace(np.random.normal(size=10))
        >>> print(trace)
        Trace(...)
        
    """
    def __init__(self, trace):
        """
        Initiate a normal distribution.
        
        @param mu: location parameter (mean)
        @type mu: float
        @param sigma: scale parameter (standard deviation)
        @type sigma: float
        """
        self.distribution = 'trace'
        self.distr_pars = dict(trace=trace)
    
    
    def get_limits(self, factor=1.0):
        """
        Return the minimum and maximum of the normal distribution.
        
        For a normal distribution, we define the upper and lower limit as the
        3*factor-sigma limits.
        """
        median = np.median(self.distr_pars['trace'])
        minim = np.min(self.distr_pars['trace'])
        maxim = np.max(self.distr_pars['trace'])
        
        lower = median - factor * (median-minim)
        upper = median + factor * (maxim-median)
        
        return lower, upper
    
    
    def draw(self, size=1, indices=None):
        """
        Draw a (set of) random value(s) from the trace.
        
        @param size: number of values to generate
        @type size: int
        @return: random value from the distribution, used indices
        @rtype: array[C{size}]
        """
        trace = self.distr_pars['trace']
        if indices is None:
            indices = np.random.randint(len(trace), size=size)
        values = trace[indices]
        #return values, indices
        return values
    
    
    def get_grid(self, sampling=5):
        """
        Draw a (set of) likely value(s) from the normal distribution.
        
        We grid uniformly in 
        """
        raise NotImplementedError("Trace distribution needs some work")
    
    
    def get_distribution(self, distr_type=None, domain=None, factor=1.0):
        """
        Return the distribution in some custom format.
        
        By default, the format is just the name and distribution parameters
        (attributes :envvar:`distribution` and :envvar:`distr_pars`).
        
        Other options:
            - ``distr_type='pdf'``: return the probability density function.
              If you do not give :envvar:``domain`` (the value(s) to evaluate the
              probability density function at), then the domain will be chosen
              to be an array of 1000 points equidistanly sampled between the
              :py:func:`limits <Normal.get_limits>` (which can be expanded or
              contracted with :envvar:`factor`. Returns tuple (domain, pdf) 
        """
        if distr_type is None:
            return self.distribution, self.distr_pars
        
        elif distr_type == 'pdf':
            trace = self.distr_pars['trace']
            keep = -np.isnan(trace) & -np.isinf(trace)
            bins = histogram_bins(trace[keep])
            counts, domain_ = np.histogram(trace[keep],
                                          bins=bins, density=True)
            domain_ = (domain_[:-1] + domain_[1:]) / 2.0
            if domain is not None:
                counts = np.interp(domain, domain_, counts)
            else:
                domain = domain_
            return domain, counts
            
        elif distr_type == 'cdf':
            trace = self.distr_pars['trace']
            keep = -np.isnan(trace) & -np.isinf(trace)
            bins = histogram_bins(trace[keep])
            counts, domain_ = np.histogram(trace[keep],
                                          bins=bins, density=True)
            cdf = np.cumsum(np.diff(domain_) * counts)
            domain_ = (domain_[:-1] + domain_[1:]) / 2.0
            if domain is not None:
                cdf = np.interp(domain, domain_, cdf)
            else:
                domain = domain_
            return domain, cdf
           
        # Else, we don't know what to do.
        else:
            raise ValueError("Do not understand distr_type='{}'".format(distr_type))
    


    def shrink(self, factor=10.0):
        """
        Shrink the extent of the normal distribution.
        
        @param factor: factor with which to shrink the distribution
        @type factor: float
        """
        raise NotImplementedError("Needs some work")
        
    
    def expand(self, factor=10.0):
        """
        Expand the extent of the normal distribution.
        
        @param factor: factor with which to expand the distribution
        @type factor: float
        """
        raise NotImplementedError("Needs some work")
    
    
    def get_loc(self):
        """
        Get location parameter.
        """
        trace = self.distr_pars['trace']
        keep = -np.isnan(trace) & -np.isinf(trace)
        return np.median(trace[keep])
    
    
    def get_scale(self):
        """
        Get scale parameter.
        """
        trace = self.distr_pars['trace']
        keep = -np.isnan(trace) & -np.isinf(trace)
        return np.std(trace[keep])
    
    def __str__(self):
        """
        String representation of class Distribution.
        """
        old_threshold = np.get_printoptions()['threshold']
        np.set_printoptions(threshold=8)
        name = self.distribution.title()
        pars = ", ".join(['{}={}'.format(key,self.distr_pars[key]) for key in sorted(list(self.distr_pars.keys()))])
        np.set_printoptions(threshold=old_threshold) 
        return "{}({})".format(name,pars)
    
    def plot(self, on_axis='x', **kwargs):
        """
        Plot a uniform prior to the current axes.
        """
        ax = kwargs.pop('ax', plt.gca())
        kwargs.setdefault('where', 'mid')
        domain, pdf = self.pdf()
        getattr(ax, 'step')(domain, pdf, **kwargs)

        

def transform_to_unbounded(par,low,high):
    return np.tan(np.pi*((par-low)/(high-low)-0.5))

def transform_to_bounded(par,low,high):
    return (np.arctan(par)/np.pi+0.5)*(high-low)+low    
