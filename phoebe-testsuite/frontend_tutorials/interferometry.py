"""
Advanced data types: interferometry
===================================

Last updated: ***time***

This tutorial explains Phoebe2's ability to simulate interferometric data (i.e. visibilities and closure phases). We'll start by showing how to use data that you've already loaded by some other means into numpy arrays (:py:func:`if_fromarrays() <phoebe.frontend.bundle.Bundle.if_fromarrays>`), and finish by showing how to load OIFITS files directly (:py:func:`if_fromfile() <phoebe.frontend.bundle.Bundle.if_fromfile>`). If you're not familiar already with the basic frontend interface of Phoebe2, we recommend you to first go through the tutorials :doc:`First steps with PHOEBE 2.0-alpha <tutorial.first_steps>` and :doc:`Building my first binary from scratch with PHOEBE 2.0-alpha <tutorial.first_binary_from_scratch>`.

As before, we will start by creating a bundle. So fire up python, import phoebe and initialize a new bundle:
"""

import phoebe
eb = phoebe.Bundle()

"""
This will create a new binary with default parameters from Phoebe Legacy. For light curve computations, all default parameters make sense, however for interferometric purposes we need to set the distance explicitly to some realistic value (the default distance is 1 solar radius (!) away from the observer, in order to get the same flux units as Phoebe Legacy):
"""

eb['distance'] = 10, 'pc'

"""
Next, we need to get some interferometry data. Because we haven't anything lying around, we will simulate our own observations here. For light curves, we only need to know the times of observations before we can simulate the observations (OK, and passbands, limb darkening coefficients etc... as well but we'll ignore those for a second). In contrast, to simulate interferometry measurements we additionally need to know the coordinates of the telescopes (i.e. the baselines). These can either be given in U and V coordinates, or, equivalently, baseline length (:py:func:`add_baseline() <phoebe.parameters.datasets.IFDataSet.add_baseline>`) and position angle (:py:func:`add_position_angle() <phoebe.parameters.datasets.IFDataSet.add_position_angle>`). Phoebe2 uses U and V coordinates as basic data input:
"""

ucoord = np.zeros(100)
vcoord = np.linspace(0.1, 200, 100)

"""
Of course, we still need to have an observation time for every telescope configuration. For simulation purposes, it is perfectly possible to take an array of observation times that are all the same, though in real life it probably takes some time to change the configuration of the telescopes:
"""

time = np.zeros(100) # all UV configurations have same time of observation
time = np.linspace(0, 0.5, 100) # observation times spread throughout the night.

"""
Instead of time, you can also specify phase just as for light curves:
"""

phase = np.linspace(0, 1, 100)

"""
Next, you can add these telescope configurations and observation times to the Bundle via :py:func:`if_fromarrays() <phoebe.frontend.bundle.Bundle.if_fromarrays>`:
"""

eb.if_fromarrays(phase=phase, ucoord=ucoord, vcoord=vcoord)

"""
Note that there are still other arrays that you can pass, in particular the eff_wave array. If you do not specify
the effective wavelength of the observations (again as an array with each element representing one observation), then the effective wavelength of the passband will be used as a proxy for all of the observations. If you know look at the summary:
"""

print(eb.summary())

"""
you can see that observations have been added to the system, a synthetic (empty) dataset is prepared that will hold the result of the computations, and each component has an :ref:`ifdep <parlabel-phoebe-ifdep>` added that contains the details of how each component should be treated:
"""

print(eb['if01@primary'])

"""
Note the :ref:`bandwidth_smearing <label-bandwidth_smearing-ifdep-phoebe>` parameter, which is by default switched off. To see more information on what this parameter means and what values you can give it, type:
"""

print(eb.info('bandwidth_smearing@if01@primary'))

"""
E.g. to use one approximation for **bandwidth smearing** (Wittkowski et al. 2004), set for each component:
"""

eb['bandwidth_smearing@if01@ifdep@primary'] = 'power'
eb['bandwidth_smearing@if01@ifdep@secondary'] = 'power'

"""
Finally, to compute the whole system and the predefined coordinates and timings, issue:
"""

eb.run_compute()

"""
Now, you have access to the computed quantities as before, e.g. the complex visiblities:
"""

print(eb['vis2@if01@ifobs'])

"""
To plot e.g. baseline versus squared visibility or time versus squared visibility, you could issue:
"""

b = np.sqrt(eb['ucoord@if01@ifsyn']**2 +eb['vcoord@if01@ifsyn']**2)
plt.figure()
plt.plot(b, eb['vis2@if01@ifsyn'], 'ko-')
plt.figure()
plt.plot(eb['time@if01@ifsyn'], eb['vis2@if01@ifsyn'], 'ko-')

"""
Or equivalently through the built-in plotting functionality:
"""

plt.figure()
eb.plot_syn('if01', 'ro-')
plt.figure()
eb.plot_syn('if01', 'ro-', x_quantity='time')

"""
If your data is contained in a standard OIFITS file, things are even easier. You can download a `demo OIFITS file here <../../devel/phoebe-testlib/test_tutorials/example_oifits.fits>`_. It was made with ASPRO2 and consists of two uniform disks to simulate a simple binary system. All (simulated) are made on one specific time point, which actually make it hard for us to physically reproduce the system, but we'll try anyway by choosing a very long period:
"""

eb = phoebe.Bundle()
eb['distance'] = 1, 'kpc'
eb['period'] = 1e8, 'd'
eb['t0'] = 56848.97824354 + 0.25*eb['period']
eb['sma'] = 5, 'au'
eb['incl'] = 0, 'deg'
eb['pot@primary'] = 21.
eb['pot@secondary'] = 21.
eb['teff@primary'] = 6000.0
eb['teff@secondary'] = 6000.0
eb.set_value_all('atm', 'blackbody')
eb.set_value_all('ld_func', 'uniform')

"""
This should do it. To view the configuration of the system, we can plot the mesh:
"""

eb.plot_mesh(phase=0.25, label='preview')
plt.show()

"""
Adding the interferometric data is now extremely easy using :py:func:`if_fromfile() <phoebe.frontend.bundle.Bundle.if_fromfile>`. Since the simulations do not contain passband information, we will use the open filter to generate the images to finally Fourier transform. The effective wavelengths are stored in the OIFITS file, which are used to convert baselines to spatial frequencies. We explicitly include the closure phase computations here (by default they are not). After adding the data, we can finally also generate a true image of the system, to visually check that the flux ratio and disk sizes are the same for both components.
"""

eb.if_fromfile('example_oifits.fits', include_closure_phase=True, passband='OPEN.BOL')
eb.plot_mesh(phase=0.25, select='proj', dataref='if01', label='preview')
plt.show()

"""
If you would like to check the sampling of the observations in the UV plane, you can this by directly accessing the :ref:`ucoord <label-ucoord-ifobs-phoebe>` and :ref:`vcoord <label-vcoord-ifobs-phoebe>` parameters in the :ref:`ifobs <parlabel-ifobs-phoebe>` DataSet, or using the built-in plotting functions:
"""

plt.figure()
eb.plot_obs('if01', fmt='o', x_quantity='ucoord', y_quantity='vcoord')
plt.show()

"""
Finally, running the computations and plotting a comparison between simulations and observations is as easy as before:
"""

eb.run_compute('preview')
plt.figure() # baseline vs vis2
eb.plot_obs('if01', fmt='ko')
eb.plot_syn('if01', 'rx', ms=10, mew=2)
plt.figure() # time vs vis2
eb.plot_obs('if01', fmt='ko', x_quantity='time')
eb.plot_syn('if01', 'rx', ms=10, mew=2, x_quantity='time')
plt.figure() # time vs closure phase
eb.plot_obs('if01', fmt='ko', x_quantity='time', y_quantity='closure_phase')
eb.plot_syn('if01', 'rx', ms=10, mew=2, x_quantity='time', y_quantity='closure_phase')
plt.show()

"""
Finally, you might be interested in quantifying how the simulations compare to the data. You could do this manually, e.g. by computing the chi-square of the observed versus simulated squared visibilities:
"""

observed = eb['vis2@if01@ifobs']
simulated = eb['vis2@if01@ifsyn']
error = eb['sigma_vis2@if01@ifobs']
print( np.nansum((observed-simulated)**2 / error**2))

"""
The reason for the many 'nan' entries in the squared visibility is because of the inclusion of the closure phases. Each closure phase observation is a separate entry in the interferometry dataset, for which there is no squared visibility observation. Using the built-in functionality automatically takes care of these issues:
"""

print(eb.get_logp())

"""
You could use this log(probability) value for your optimization routines.

**File attachment:**

:download:`￼Example OIFITS file <../../devel/phoebe-testlib/test_tutorials/example_oifits.fits>`
"""
