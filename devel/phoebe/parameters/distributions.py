"""
Definition of statistical or empirical distributions.
"""
import numpy as np
from phoebe.units import conversions
try:
    import pymc
except ImportError:
    pass
from scipy.stats import distributions


class Distribution(object):
    """
    This will become the new distribution base class.
    
    Now it's a factory.
    """
    def __new__(cls, *args, **kwargs):
        if 'distribution' in kwargs:
            dist_name = kwargs.pop('distribution')
        else:
            dist_name = args[0]
            args = args[1:]
            
        if dist_name == 'normal':
            return Normal(*args, **kwargs)
        elif dist_name == 'uniform':
            return Uniform(*args, **kwargs)
        else:
            return DeprecatedDistribution(dist_name, *args, **kwargs)
            


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
        self.distribution = distribution.lower()
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
        
        if (given_keys - required_keys) or (required_keys - given_keys):
            raise ValueError('Distribution {} needs keys {}'.format(distribution, ", ".join(list(required_keys))))
        
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
            
    def convert(self,from_,to_,*args):
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
        logger.info('Converting {} distr. in {} to histogram distr. in {}'.format(self.distribution,from_,to_))
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
            new_x = conversions.convert(from_,to_,old_x)
            new_x_shifted = conversions.convert(from_,to_,x_shifted)
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


class Normal(DeprecatedDistribution):
    """
    The Normal distribution.
    """
    def __init__(self, mu, sigma):
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
        Draw a (set of_) likely value(s) from the normal distribution.
        
        We grid uniformly in 
        """
        loc = self.get_loc()
        scale = self.get_scale()
        cum_sampling = np.linspace(0, 1, sampling+2)[1:-1]
        mygrid = distributions.norm(loc=loc, scale=scale).isf(cum_sampling)
        return mygrid
        


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
        
        
        

class Uniform(DeprecatedDistribution):
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
        Draw a (set of_) likely value(s) from the normal distribution.
        
        We grid uniformly in 
        """
        lower = self.distr_pars['lower']
        upper = self.distr_pars['upper']
        mygrid = np.linspace(lower, upper, sampling)
        
        return mygrid    
    
    def get_loc(self):
        """
        Get location parameter.
        """
        lower = self.distr_pars['lower']
        upper = self.distr_pars['upper']
        return (lower + upper) / 2.0
    
    
    def get_scale(self):
        """
        Get scale parameter.
        """
        lower = self.distr_pars['lower']
        upper = self.distr_pars['upper']
        
        return (upper - lower) / 2.0
        
        

def transform_to_unbounded(par,low,high):
    return np.tan(np.pi*((par-low)/(high-low)-0.5))

def transform_to_bounded(par,low,high):
    return (np.arctan(par)/np.pi+0.5)*(high-low)+low    