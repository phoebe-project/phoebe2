"""
Fitting using lmfit
========================

Last updated: ***time***

This example shows some more technical options for fitting routines - on fake data, so we know the answer

Initialisation
--------------

"""
# First we'll import phoebe and create a logger

import phoebe
import numpy as np
import matplotlib.pyplot as plt
logger = phoebe.get_basic_logger()

# Create the fake data
# ---------------------

b = phoebe.Bundle()
b.lc_fromarrays(phase=np.linspace(0,1,200),dataref='fakedata')

b.rv_fromarrays('primary',phase=np.linspace(0,1,20),dataref='fakeRVdata')

'''
Now that we've created some fake data, let's change some parameters, add some Gaussian noise
and see if we can recover them by fitting them later!
'''

b['incl']=87.9
b['teff@primary'] = 6543
b['pot@primary'] = 5.123


b.run_compute('preview')
b.plot_syn('fakedata')
plt.show()


noise = np.random.normal(0,0.01*np.median(b['value@flux@lcsyn']),200)
rv_noise = np.random.normal(0,0.1*np.std(b['value@rv@rvsyn']),20)


np.savetxt('fakedata.lc', np.array([b['value@time@lcsyn'], b['value@flux@lcsyn']+noise]).T )
np.savetxt('fakeRVdata.lc', np.array([b['value@time@rvsyn'], b['value@rv@rvsyn']+rv_noise]).T )

'''
Loading the fake data as "observations"
-----------------------------------------

Now we'll create a new bundle and load these synethetic data as if they were observations. Notice however, that we used the "preview" 
compute label. PHOEBE 2.0 offers three different compute labels, "legacy", "preview", and "detailed". The "detailed" compute label
takes longer to run, it includes more physics such as heating, reflection, and doppler beaming. And the "legacy" compute label runs
the same physics as PHOEBE 1.0. In our case, we don't have any heating, reflection, or doppler beaming occurring, so we can simply
use the "preview" label.
'''

b = phoebe.Bundle()
b.lc_fromfile('fakedata.lc',  columns=['time','flux'], dataref='fakedata')
b.rv_fromfile('fakeRVdata.lc', 'primary',columns=['time','rv'],   dataref='fakeRVdata')

'''
By default all datasets that you have loaded will be used for fitting. 
If you wanted, you could disable any by label using b.disable_lc('fakedata') or b.disable_rv('fakervdata').

Setup Fitting
----------------

Now we need to choose our parameters to fit and setup their priors.
PHOEBE 2 currently supports uniform and normal distributions.
'''
# inclination (87.9)

b.set_prior('incl', distribution='uniform', lower=80, upper=90)
b.set_adjust('incl')

#teff@primary (6543)

b.set_prior('teff@primary', distribution='uniform', lower=5000, upper=7000)
b.set_adjust('teff@primary')

# pot@primary (5.123)

b.set_prior('pot@primary', distribution='uniform', lower=4.0, upper=8.0)
b.set_adjust('pot@primary')

b.set_prior('q@orbit',distribution='uniform',lower=0.01,upper=5.)
b.set_adjust('q@orbit')

'''
When modeling, PHOEBE uses the parameter pblum, or the passband luminosity, to scale the model to the observations. Additionally, 
you can set the parameter l3 to be adjusted to account for the presence of third light contaminating your system. 
'''

b.set_adjust('pblum@primary@lcdep')
b.set_adjust('pblum@secondary@lcdep')

'''
 Run Fitting
 ----------------

Now we want to run three different instances of fitting routines.
First, we'll add fitting labels, and then run them individually.

'''

b.add_fitting(context='fitting:lmfit', computelabel='preview', method='nelder',label='simplex')
b.add_fitting(context='fitting:lmfit', computelabel='preview', method='leastsq',label='levmarq')
b.add_fitting(context='fitting:emcee', computelabel='preview', iters=100, walkers=10, label='my_mcmc')


b.run_fitting(label='simplex')
b.run_fitting(label='levmarq')
b.run_fitting(label='my_mcmc')

'''
Having run all all of these different fitting routines, we can compare the results, and choose which one we want to accept.

'''

print b['simplex@feedback']
print b['levmarq@feedback']
print b['my_mcmc@feedback']

b.accept_feedback('my_mcmc')


print("incl (87.9): {}".format(b['value@incl']))
print("teff@primary (6543) {}".format(b['value@teff@primary']))
print("pot@primary (5.123) {}".format(b['value@pot@primary']))

'''
Plotting
------------

Let's compare the original synthetic data to the fitted model
'''

plt.subplot(211)
b.plot_obs('fakedata')
b.plot_syn('fakedata',fmt='r-')
plt.subplot(212)
b.plot_obs('fakeRVdata')
b.plot_syn('fakeRVdata',fmt='r-')
plt.show()

'''
Limits
------------

If you issue the print command for a particular parameter such as the effective temperature of the primary
'''

print b['teff@component@primary']


'''
You'll see the line llim/ulim/step: , which is the "reasonable" bounds set by PHOEBE. These act as hard limits and override 
the authority of a prior box supplied by the user. However, if you're confident in your values, you can adjust these values by 
using the set_limits() command.
'''

b['teff@component@primary'].set_limits(llim=3000)
