"""

.. contents::


Fitting with Wilson-Devinney code
===================================

This tutorial explains how to use the Phoebe 2.0 fitting interface, while using
the Wilson-Devinney code to do the actual computations.

This script uses :download:`this active file <../phoebe-testsuite/wilson_devinney/test01lcin.active>`.

(this script typically takes a minute or two to run)

First, import some modules that we'll need later on:

"""
import phoebe
from phoebe import plotting
import matplotlib.pyplot as plt
import numpy as np

logger = phoebe.get_basic_logger()

np.random.seed(50)

# Simulate data with WD
# -----------------------
# Read in a WD \*.active file and adjust some of the parameters for the light
# curve. The read function returns a :ref:`root <parlabel-wd-root>`,
# :ref:`lc <parlabel-wd-lc>` and :ref:`root <parlabel-wd-rv>` parameterSet.

pset,lcset,rvset = phoebe.wd.lcin_to_ps('test01lcin.active',version='wd2003')

lcset['jdstrt'] = pset['hjd0']
lcset['jdend'] = pset['hjd0']+pset['period']
lcset['jdinc'] = 0.01*pset['period']
lcset['indep_type'] = 'time (hjd)'

# Generate the light curve via :py:func:`lc <phoebe.wd.wd.lc>`
# and add some noise to it. Also rescale the flux and
# add an offset: this is to mimic the arbitrary flux normalisation usually
# present in the data, and the possible occurrence of third light.

curve,params = phoebe.wd.lc(pset,request='curve',light_curve=lcset,rv_curve=rvset)
sigma = 0.02*np.ones(len(curve['lc']))
noise = np.random.normal(scale=sigma)
simul = 1.3*curve['lc'] + 2.1 + noise

# Save it to a text file, so that we can load it into to :py:class:`BodyEmulator <phoebe.wd.wd.BodyEmulator>`, and
# make a figure of the data we are going to fit:

np.savetxt('myobs.lc',np.column_stack([curve['indeps'],simul,sigma]))

plt.figure()
plt.errorbar(curve['indeps'],simul,yerr=sigma,fmt='ko')
plt.xlabel("Time [d]")
plt.ylabel("Relative flux")
plt.savefig('body_emul_data.png')

"""

.. image:: images_tut/body_emul_data.png
   :scale: 50%

"""

# Make the BodyEmulator
#-------------------------
# Parse the information from the light curve file via
# :py:func:`parse_lc <phoebe.parameters.datasets.parse_lc>`. It returns a list
# of observation ParameterSets and passband dependent parameterSets. Since there
# is only one light curve, a list of length 1 is returned. The ``pbdep`` can
# be discarded when using WD, since the ``lc`` and ``rv`` parameterSets are
# already defined:

obs,pbdep = phoebe.parameters.datasets.parse_lc('myobs.lc')

# Build the BodyEmulator: we need to pass the WD parameterSets, and the
# observations.
system = phoebe.wd.BodyEmulator(pset,lcset=lcset,rvset=rvset,obs=obs)

"""

Prepare for fitting
------------------------

Let's fit the inclination angle, the eccentricity and the value of the primary
potential using the Levenberg-Marquardt fitter: as a reference, we print out
the true values::

   print("incl={}, ecc={}, pot1={}".format(pset['incl'],pset['ecc'],pset['pot1'])) 
   incl=89.057, ecc=0.16346, pot1=6.1518

We'll set the initial values close enough to the true ones (and for reference
compute a light curve with those parameters):

"""
pset['incl'] = 90,'deg'
pset['ecc'] = 0.1
pset['pot1'] = 6.2

curve_init,params_init = phoebe.wd.lc(pset,request='curve',light_curve=lcset,rv_curve=rvset)

# We need to make it clear to the code which parameters it needs to fit. Besides
# the ones previously mentioned, we also want to fit the luminosity of the system
# (arbitrary scaling) and the contribution from third light. In this case, these
# are not really "system" parameters, but they are treated as being part of the
# observations. What we want to do is to generate model light curves with varying
# inclination angle, eccentricity and potential value, but rescale the model
# such that the data are best matched. We want to do this via a normal linear
# fit instead of using the fancy fitting algorithms. In any case, we want them
# to be :py:func:`adjustable <phoebe.parameters.parameters.ParameterSet.set_adjust>`:
pset.set_adjust(('incl','ecc','pot1'),True)
obs[0].set_adjust(('pblum','l3'),True)

"""

We make the difference between the two types of parameters by
:py:func:`setting the priors <phoebe.parameters.parameters.Parameter.set_prior>`
for the parameters we want to `really` fit. Priors are really :py:class:`distributions <phoebe.parameters.parameters.Distribution>`.
If ``pblum`` and ``l3``
don't have priors attached, they will not be included in the regular fitting
algorithm, but determined a posterior with a linear fit.

The priors are set for each `real` parameter: we set the inclination and
potential values to have a normal distribution, and the eccentricity to have
a uniform distribution.
"""
pset.get_parameter('incl').set_prior(distribution='normal',mu=90.,sigma=5.)
pset.get_parameter('ecc').set_prior(distribution='uniform',lower=0.0,upper=1.0)
pset.get_parameter('pot1').set_prior(distribution='normal',mu=6.2,sigma=0.2)

# You might wonder what the purpose is for setting priors for parameters if we
# want to use a regular nonlinear fitting routine. The reason is twofold: one
# is to tell the code we want to fit it, and the other is that the nonlinear
# fitting routines accept boundaries for the values. If you don't want to use
# these boundaries (they may influence the error determination considerably), you
# need to set ``bounded=False`` in the parameterSet controlling the properties
# of the fit:
fitparams = phoebe.ParameterSet(context='fitting:lmfit',method='leastsq',bounded=True)

"""

Printing this parameterSet reveals the following parameters::
    
    print(fitparams)
        method leastsq                               --   phoebe Nonlinear fitting method
         iters 0                                     --   phoebe Number of iterations
         label c8b61573-a686-4886-aef9-d2d0a3846c93  --   phoebe Fit run name
    compute_ci False                                 --   phoebe Compute detailed confidence intervals
       bounded True                                  --   phoebe Include boundaries in fit
      feedback {:}                                   --   phoebe Results from fitting procedure

The type of nonlinear fitting algorithm is given by ``method``, and you can
give an additional label for easy reference. If you set ``iters=0``, the algorithm
will be run just once starting from the current values (i.e. the once we've
provided). Otherwise, random initial values will be drawn from each of the priors,
as often as given. If you set ``compute_ci=True``, the ``lmfit`` package will
compute the confidence intervals of the parameters by some kind of heuristic
scanning (which can take a while!).

Perform the fit
------------------

Finally, we can :py:func:`run the fitting algorithm <phoebe.backend.fitting.run>` and save the output (a parameterSet)
to a file. We set ``accept=True``, such that the best fitting parameters will
be set, and that model will also be computed.
"""
feedback = phoebe.run(system,fitparams=fitparams,accept=True)
feedback.save('feedback.phoebe')

"""

Evaluating the fit
---------------------

The logger will automatically suppress all information lower than ``WARNING``,
to avoid cluttering of the terminal. It will finally print out a summary of the
results, i.e. the best fitting values, an estimate of the error and the
correlations between the parameters. You see there is a strong anticorrelation
between the inclination angle and the value for the primary potential::

      ecc_c0fdb0fd_ce52_4c44_a363_df0fd30b6155:      0.164752 +/- 0.003399 (2.06%) initial =  0.100000
      incl_dd88f24d_3824_4d7e_8226_4747e7c5a982:     87.363219 +/- 0.825361 (0.94%) initial =  90.000000
      pot1_9e7163dd_4efe_4b58_9a69_6313d7e2add2:     5.964313 +/- 0.069673 (1.17%) initial =  6.200000
    Correlations:
      C(incl_dd88f24d_3824_4d7e_8226_4747e7c5a982, pot1_9e7163dd_4efe_4b58_9a69_6313d7e2add2)  = -0.989 
      C(ecc_c0fdb0fd_ce52_4c44_a363_df0fd30b6155, pot1_9e7163dd_4efe_4b58_9a69_6313d7e2add2)  = -0.122 
      C(ecc_c0fdb0fd_ce52_4c44_a363_df0fd30b6155, incl_dd88f24d_3824_4d7e_8226_4747e7c5a982)  =  0.041 

The values for the ``pblum`` and ``l3`` parameters are also mentioned, and in
this case are equal to ``pblum=1.44`` and ``l3 = 1.96``.

You might notice that the names of the parameters are a bit obfuscated. This
is something internally done by the fitting routines. It is not guarenteed that
all parameter names are unique, so the code makes them unique. All the information,
however, is also stored in the ``feedback`` parameter. Printing this parameterSet
gives the following information::

    print(feedback)
       method leastsq                                                                                 --   phoebe Nonlinear fitting method
        iters 0                                                                                       --   phoebe Number of iterations
        label fa97fa63-170b-4a66-8aee-4e793ef59b35                                                    --   phoebe Fit run name
   compute_ci False                                                                                   --   phoebe Compute detailed confidence intervals
      bounded True                                                                                    --   phoebe Include boundaries in fit
     feedback {redchis:,parameters:,success:,correls:,Npars:,redchi:,values:,Ndata:,traces:,sigmas:}  --   phoebe Results from fitting procedure
     
It is the fitting ParameterSet where the parameter ``feedback`` has been filled
in. It contains all information on the fitting procedure::

    print(feedback['feedback']['success'])
    print(feedback['feedback']['redchi'])
    for par,val,sig in zip(feedback['feedback']['parameters'],\
                           feedback['feedback']['values'],\
                           feedback['feedback']['sigmas']):
        print("{} = {} +/- {}".format(par.get_qualifier(),val,sig))
    True
    1.91310195892
    incl = 87.363219459 +/- 0.825361407781
    ecc = 0.164752226301 +/- 0.00339925856617
    pot1 = 5.96431290455 +/- 0.0696729889646
    
"""
# Finally, we can plot the observations and the best fitting model, as well
# as the history of the fitting procedure:

plt.figure()
plotting.plot_lcobs(system,fmt='ko')
plotting.plot_lcsyn(system,'r-',lw=2,label='Fit')
plt.plot(curve['indeps'],1.3*curve['lc']+2.1,'g--',lw=2,label='True')
plt.plot(curve_init['indeps'],1.3*curve_init['lc']+2.1,'b-',lw=2,label='Initial')
plt.legend(loc='best')
plt.xlabel("Time [d]")
plt.ylabel("Relative flux")
plt.savefig('body_emul_fit')

plt.figure()
plt.subplot(121)
plt.plot(feedback['feedback']['traces'][0],'ko-')
plt.axhline(89.057,lw=2,color='r')
plt.axhline(feedback['feedback']['values'][0]-3*feedback['feedback']['sigmas'][0],lw=2,color='b')
plt.axhline(feedback['feedback']['values'][0]+3*feedback['feedback']['sigmas'][0],lw=2,color='b')
plt.xlabel("Iteration")
plt.ylabel('Inclination angle [deg]')
plt.subplot(122)
plt.plot(feedback['feedback']['traces'][2],'ko-')
plt.axhline(6.1518,lw=2,color='r')
plt.axhline(feedback['feedback']['values'][2]-3*feedback['feedback']['sigmas'][2],lw=2,color='b')
plt.axhline(feedback['feedback']['values'][2]+3*feedback['feedback']['sigmas'][2],lw=2,color='b')
plt.xlabel("Iteration")
plt.ylabel('Potential value')
plt.savefig('body_emul_trace')
plt.show()


"""

You can have a look at what the difference is when ``bounded`` is False to get
an idea of the influence it has.

+--------------------------------------------+--------------------------------------------+
| .. image:: images_tut/body_emul_fit.png    | .. image:: images_tut/body_emul_trace.png  |
|    :width: 500px                           |    :width: 500px                           |
+--------------------------------------------+--------------------------------------------+
"""
