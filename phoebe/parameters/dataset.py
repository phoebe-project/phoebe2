
from phoebe.parameters import *
from phoebe.parameters import constraint
from phoebe.atmospheres import passbands  # need to load pbtable (dictionary of available passbands)
from phoebe import u

def lc(**kwargs):
    """
    Create parameters for a new light curve dataset.

    Generally, this will be used as an input to the method argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_dataset`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """

    obs_params = []

    syn_params, constraints = lc_syn(syn=False, **kwargs)
    obs_params += syn_params.to_list()
    #obs_params += lc_dep(**kwargs).to_list()

    #~ obs_params += [FloatArrayParameter(qualifier='flag', value=kwargs.get('flag', []), default_unit=None, description='Signal flag')]
    #~ obs_params += [FloatArrayParameter(qualifier='weight', value=kwargs.get('weight', []), default_unit=None, description='Signal weight')]

    #~ obs_params += [FloatParameter(qualifier='timeoffset', value=kwargs.get('timeoffset', 0.0), default_unit=u.d, description='Zeropoint date offset for observations')]
    #~ obs_params += [FloatParameter(qualifier='statweight', value=kwargs.get('statweight', 0.0), default_unit=None, description='Statistical weight in overall fitting')]

    return ParameterSet(obs_params), constraints

def lc_syn(syn=True, **kwargs):

    syn_params = []

    syn_params += [FloatArrayParameter(qualifier='time', value=kwargs.get('time', []), default_unit=u.d, description='Observed times')]
    syn_params += [FloatArrayParameter(qualifier='flux', value=kwargs.get('flux', []), default_unit=u.W/u.m**2, description='Observed flux')]
    if not syn:
        syn_params += [FloatArrayParameter(qualifier='sigma', value=kwargs.get('sigma', []), default_unit=u.W/u.m**2, description='Observed uncertainty on flux')]


    #~ syn_params += [FloatArrayParameter(qualifier='exptime', value=kwargs.get('exptime', []), default_unit=u.s, description='Signal exposure time')]

    constraints = []

    return ParameterSet(syn_params), constraints

def lc_dep(**kwargs):
    # TODO: take component_methods as argument so that LC can copy_for envelope but RV can skip
    # TODO: RV doesn't need l3

    dep_params = []

    dep_params += [ChoiceParameter(qualifier='ld_func', copy_for={'method': ['star', 'envelope'], 'component': '*'}, component='_default', value=kwargs.get('ld_func', 'logarithmic'), choices=['uniform', 'linear', 'logarithmic', 'quadratic', 'square_root', 'power', 'claret', 'hillen', 'prsa'], description='Limb darkening model')]
    dep_params += [FloatArrayParameter(qualifier='ld_coeffs', copy_for={'method': ['star', 'envelope'], 'component': '*'}, component='_default', value=kwargs.get('ld_coeffs', [0.5, 0.5]), default_unit=u.dimensionless_unscaled, description='Limb darkening coefficients')]
    passbands.init_passbands()  # TODO: possibly move to the import of the passbands module
    dep_params += [ChoiceParameter(qualifier='passband', value=kwargs.get('passband', 'Johnson:V'), choices=passbands._pbtable.keys(), description='Passband')]
    dep_params += [ChoiceParameter(qualifier='pbscale', copy_for={'method': ['star', 'envelope'], 'component': '*'}, component='_default', value=kwargs.get('pbscale', ''), choices=['pblum', '']+kwargs.get('starrefs', []), description='Whether to use this components pblum or the scaling from another component in the system')]
    dep_params += [FloatParameter(relevant_if='pbscale:pblum', qualifier='pblum', copy_for={'method': ['star', 'envelope'], 'component': '*'}, component='_default', value=kwargs.get('pblum', 4*np.pi), default_unit=u.W, description='Passband luminosity (defined at t0)')]
    dep_params += [FloatParameter(qualifier='l3', value=kwargs.get('l3', 0.), default_unit=u.W/u.m**3, description='Third light')]
    dep_params += [FloatParameter(qualifier='alb', copy_for={'method': ['star', 'envelope'], 'component': '*'}, component='_default', value=kwargs.get('alb', 0.), default_unit=u.dimensionless_unscaled, description='Passband Bond\'s albedo, alb=0 is no reflection')]

    return ParameterSet(dep_params)

def rv(**kwargs):
    """
    Create parameters for a new radial velocity dataset.

    Generally, this will be used as an input to the method argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_dataset`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """

    obs_params = []

    #obs_params += [FloatParameter(qualifier='statweight', value = kwargs.get('statweight', 1.0), default_unit=u.dimensionless_unscaled, description='Statistical weight in overall fitting')]
    syn_params, constraints = rv_syn(syn=False, **kwargs)
    obs_params += syn_params.to_list()
    #obs_params += rv_dep(**kwargs).to_list()

    return ParameterSet(obs_params), constraints

def rv_syn(syn=True, **kwargs):
    """
    """

    syn_params = []

    syn_params += [FloatArrayParameter(qualifier='time', copy_for={'method': ['star'], 'component': '*'}, component='_default', value=kwargs.get('time', []), default_unit=u.d, description='Observed times')]
    syn_params += [FloatArrayParameter(qualifier='rv', copy_for={'method': ['star'], 'component': '*'}, component='_default', value=kwargs.get('rv', []), default_unit=u.km/u.s, description='Observed radial velocity')]
    if not syn:
        syn_params += [FloatArrayParameter(qualifier='sigma', copy_for={'method': ['star'], 'component': '*'}, component='_default', value=kwargs.get('sigma', []), default_unit=u.km/u.s, description='Observed uncertainty on rv')]

    #~ syn_params += [FloatArrayParameter(qualifier='exptime', value=kwargs.get('exptime', []), default_unit=u.s, description='Signal exposure time')]

    constraints = []

    return ParameterSet(syn_params), constraints

def rv_dep(**kwargs):
    """
    """

    dep_params = []
    # TODO: only relevent-if rv_method='flux-weighted'

    dep_params += lc_dep(**kwargs).to_list()


    return ParameterSet(dep_params)


def etv(**kwargs):
    """
    Create parameters for a new eclipse timing variations dataset.

    Generally, this will be used as an input to the method argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_dataset`

    :parameter **kwargs: default for the values of any of the ParameterSet
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """

    obs_params = []

    syn_params, constraints = etv_syn(syn=False, **kwargs)
    obs_params += syn_params.to_list()
    #obs_params += etv_dep(**kwargs).to_list()

    return ParameterSet(obs_params), constraints

def etv_syn(syn=True, **kwargs):
    """
    """

    syn_params = []

    #syn_params += [IntArrayParameter(qualifier='N', value=kwargs.get('N', []), description='Epoch since t0')]
    syn_params += [FloatArrayParameter(qualifier='N', copy_for={'method': ['star', 'envelope'], 'component': '*'}, component='_default', value=kwargs.get('N', []), default_unit=u.dimensionless_unscaled, description='Epoch since t0')]
    syn_params += [FloatArrayParameter(qualifier='time_ecl', copy_for={'method': ['star', 'envelope'], 'component': '*'}, component='_default', value=kwargs.get('time_ecl', []), default_unit=u.d, description='Time of eclipse')]
    syn_params += [FloatArrayParameter(qualifier='time_ephem', copy_for={'method': ['star', 'envelope'], 'component': '*'}, component='_default', value=kwargs.get('time_ephem', []), default_unit=u.d, description='Expected time of eclipse from the current ephemeris')]
    syn_params += [FloatArrayParameter(qualifier='etv', copy_for={'method': ['star', 'envelope'], 'component': '*'}, component='_default', value=kwargs.get('etv', []), default_unit=u.d, description='Eclipse timing variation (time_obs - time_ephem)')]
    if not syn:
        syn_params += [FloatArrayParameter(qualifier='sigma', value=kwargs.get('sigma', []), default_unit=u.d, description='Observed uncertainty on time_obs')]

    constraints = []
    constraints += [(constraint.time_ephem, kwargs.get('component', '_default'), kwargs.get('dataset', None))]
    constraints += [(constraint.etv, kwargs.get('component', '_default'), kwargs.get('dataset', None))]

    return ParameterSet(syn_params), constraints

def etv_dep(**kwargs):
    """
    """

    dep_params = []

    # TODO: only relevent-if rv_method='flux-weighted'
    # TODO: add these back in if we implement an etv_method that actually needs fluxes
    #dep_params += [ChoiceParameter(qualifier='ld_func', value=kwargs.get('ld_func', 'logarithmic'), choices=['uniform', 'linear', 'logarithmic', 'quadratic', 'square_root', 'power', 'claret', 'hillen', 'prsa'], description='Limb darkening model')]
    #dep_params += [FloatArrayParameter(qualifier='ld_coeffs', value=kwargs.get('ld_coeffs', [0.5, 0.5]), default_unit=None, description='Limb darkening coefficients')]
    #passbands.init_passbands()  # TODO: possibly move to the import of the passbands module
    #dep_params += [ChoiceParameter(qualifier='passband', value=kwargs.get('passband', 'Johnson:V'), choices=passbands._pbtable.keys(), description='Passband')]


    return ParameterSet(dep_params)


def ifm(**kwargs):
    """
    Create new parameters for a new interferometry dataset.

    Generally, this will be used as an input to the method argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_dataset`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """

    obs_params = []

    obs_params += [FloatParameter(qualifier='statweight', value = kwargs.get('statweight', 1.0), default_unit=u.dimensionless_unscaled, description='Statistical weight in overall fitting')]

    syn_params, constraints = ifm_syn(syn=False, **kwargs)
    obs_params += syn_params.to_list()
    #obs_params += ifm_dep(**kwargs).to_list()

    return ParameterSet(obs_params), constraints

def ifm_syn(syn=True, **kwargs):
    """
    """

    syn_params = []

    # independent parameters
    syn_params += [FloatArrayParameter(qualifier='time', value=kwargs.get('time', []), default_unit=u.d, description='Observed times')]
    syn_params += [FloatArrayParameter(qualifier='ucoord', value=kwargs.get('ucoord', []), default_unit=u.m, description='Projection of the baseline in the North-South direction.')]
    syn_params += [FloatArrayParameter(qualifier='vcoord', value=kwargs.get('vcoord', []), default_unit=u.m, description='Projection of the baseline in the East-West direction.')]
    syn_params += [FloatArrayParameter(qualifier='eff_wave', value=kwargs.get('eff_wave', []), default_unit=u.m, description='Effective wavelength of the repsective passband.')]
    syn_params += [FloatArrayParameter(qualifier='ucoord_2', value=kwargs.get('ucoord_2', []), default_unit=u.m, description='Projection of the second baseline in the North-South direction in a telescope triplet.')]
    syn_params += [FloatArrayParameter(qualifier='vcoord_2', value=kwargs.get('vcoord_2', []), default_unit=u.m, description='Projection of the second baseline in the East-West direction in a telescope triplet.')]

    # observables - fringe squared visibilities for a closing triplet
    syn_params += [FloatArrayParameter(qualifier='vis2', value=kwargs.get('vis2', []), default_unit=u.dimensionless_unscaled, description='Fringe visibility for the first baseline (ucoord, vcoord).')]
    syn_params += [FloatArrayParameter(qualifier='vis2_2', value=kwargs.get('vis2_2', []), default_unit=u.dimensionless_unscaled, description='Fringe visibility for the second baseline in a telescope triplet (ucoord_2, vcoord_2).')]
    syn_params += [FloatArrayParameter(qualifier='vis2_3', value=kwargs.get('vis2_3', []), default_unit=u.dimensionless_unscaled, description='Fringe visibility for the third baseline in a telescope triplet (ucoord+ucoord_2, vcoord+vcoord_2).')]

    # fringe phases for a closing triplet
    syn_params += [FloatArrayParameter(qualifier='vphase', value=kwargs.get('vphase', []), default_unit=u.rad, description='Fringe phase for the first baseline (ucoord, vcoord).')]
    syn_params += [FloatArrayParameter(qualifier='vphase_2', value=kwargs.get('vphase_2', []), default_unit=u.rad, description='Fringe phase for the second baseline in a telescope triplet (ucoord_2, vcoord_2).')]
    syn_params += [FloatArrayParameter(qualifier='vphase_3', value=kwargs.get('vphase_3', []), default_unit=u.rad, description='Fringe phase for the third baseline in a telescope triplet (ucoord+ucoord_2, vcoord+vcoord_2).')]

    # closure phase and closure amplitude for a closing triplet
    syn_params += [FloatArrayParameter(qualifier='t3_ampl', value=kwargs.get('t3_ampl', []), default_unit=u.rad, description='Triple amplitude for a closing telescope triplet.')]
    syn_params += [FloatArrayParameter(qualifier='t3_phase', value=kwargs.get('t3_phase', []), default_unit=u.rad, description='Closure phase for a closing  telescope triplet.')]

    # corresponding uncertainties
    if not syn:
        # uncertainties fringe squared visibilities
        syn_params += [FloatArrayParameter(qualifier='sigma_vis2', value=kwargs.get('sigma_vis2', []), default_unit=u.dimensionless_unscaled, description='Uncertainty - fringe visibility for the first baseline (ucoord, vcoord).')]
        syn_params += [FloatArrayParameter(qualifier='sigma_vis2_2', value=kwargs.get('sigma_vis2_2', []), default_unit=u.dimensionless_unscaled, description='Uncertainty - fringe visibility for the second baseline in a telescope triplet (ucoord_2, vcoord_2).')]
        syn_params += [FloatArrayParameter(qualifier='sigma_vis2_3', value=kwargs.get('sigma_vis2_3', []), default_unit=u.dimensionless_unscaled, description='Uncertainty - fringe visibility for the third baseline in a telescope triplet (ucoord+ucoord_2, vcoord+vcoord_2).')]

        # uncertainties fringe phases
        syn_params += [FloatArrayParameter(qualifier='sigma_vphase', value=kwargs.get('sigma_vphase', []), default_unit=u.rad, description='Uncertainty - fringe phase for the first baseline (ucoord, vcoord).')]
        syn_params += [FloatArrayParameter(qualifier='sigma_vphase_2', value=kwargs.get('sigma_vphase_2', []), default_unit=u.rad, description='Uncertainty - fringe phase for the second baseline in a telescope triplet (ucoord_2, vcoord_2).')]
        syn_params += [FloatArrayParameter(qualifier='sigma_vphase_3', value=kwargs.get('sigma_vphase_3', []), default_unit=u.rad, description='Uncertainty - fringe phase for the third baseline in a telescope triplet (ucoord+ucoord_2, vcoord+vcoord_2).')]

        # closure phase and closure amplitude for a closing triplet
        syn_params += [FloatArrayParameter(qualifier='sigma_t3_ampl', value=kwargs.get('sigma_t3_ampl', []), default_unit=u.dimensionless_unscaled, description='Uncertainty - triple amplitude for a closing telescope triplet.')]
        syn_params += [FloatArrayParameter(qualifier='sigma_t3_phase', value=kwargs.get('sigma_t3_phase', []), default_unit=u.rad, description='Uncertainty - Closure phase for a closing  telescope triplet.')]

    return ParameterSet(syn_params)

def ifm_dep(**kwargs):
    """
    """

    dep_params = []

    dep_params += [ChoiceParameter(qualifier='ld_func', copy_for={'method': ['star', 'envelope'], 'component': '*'}, component='_default', value=kwargs.get('ld_func', 'logarithmic'), choices=['uniform', 'linear', 'logarithmic', 'quadratic', 'square_root', 'power', 'claret', 'hillen', 'prsa'], description='Limb darkening model')]
    dep_params += [FloatArrayParameter(qualifier='ld_coeffs', copy_for={'method': ['star', 'envelope'], 'component': '*'}, component='_default', value=kwargs.get('ld_coeffs', [0.5, 0.5]), default_unit=u.dimensionless_unscaled, description='Limb darkening coefficients')]

    return ParameterSet(dep_params)

def orb(**kwargs):
    """
    Create parameters for a new orbit dataset.

    Generally, this will be used as an input to the method argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_dataset`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """

    obs_params = []

    #~ obs_params += [FloatArrayParameter(qualifier='exptime', value=kwargs.get('exptime', []), default_unit=u.s, description='Signal exposure time')]
    #~ obs_params += [FloatArrayParameter(qualifier='flag', value=kwargs.get('flag', []), default_unit=None, description='Signal flag')]
    #~ obs_params += [FloatArrayParameter(qualifier='weight', value=kwargs.get('weight', []), default_unit=None, description='Signal weight')]

    #~ obs_params += [FloatParameter(qualifier='timeoffset', value=kwargs.get('timeoffset', 0.0), default_unit=u.d, description='Zeropoint date offset for observations')]
    #~ obs_params += [FloatParameter(qualifier='statweight', value=kwargs.get('statweight', 0.0), default_unit=None, description='Statistical weight in overall fitting')]

    syn_params, constraints = orb_syn(syn=False, **kwargs)
    obs_params += syn_params.to_list()
    #obs_params += orb_dep(**kwargs).to_list()

    return ParameterSet(obs_params), []

def orb_syn(syn=True, **kwargs):

    syn_params = []

    syn_params += [FloatArrayParameter(qualifier='time', copy_for={'method': ['star'], 'component': '*'}, component='_default', value=kwargs.get('time', []), default_unit=u.d, description='{} times'.format('Synthetic' if syn else 'Observed'))]

    if syn:
        # syns ignore copy_for anyways
        syn_params += [FloatArrayParameter(qualifier='x', value=kwargs.get('x', []), default_unit=u.solRad, description='X position')]
        syn_params += [FloatArrayParameter(qualifier='y', value=kwargs.get('y', []), default_unit=u.solRad, description='Y position')]
        syn_params += [FloatArrayParameter(qualifier='z', value=kwargs.get('z', []), default_unit=u.solRad, description='Z position')]
        syn_params += [FloatArrayParameter(qualifier='vx', value=kwargs.get('vx', []), default_unit=u.solRad/u.d, description='X velocity')]
        syn_params += [FloatArrayParameter(qualifier='vy', value=kwargs.get('vy', []), default_unit=u.solRad/u.d, description='Y velocity')]
        syn_params += [FloatArrayParameter(qualifier='vz', value=kwargs.get('vz', []), default_unit=u.solRad/u.d, description='Z velocity')]

    constraints = []

    return ParameterSet(syn_params), constraints

def orb_dep(**kwargs):

    dep_params = []

    # ltte (per dep and per compute???)???

    return ParameterSet(dep_params)

def mesh(**kwargs):
    """
    Create parameters for a new mesh dataset.

    Generally, this will be used as an input to the method argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_dataset`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """

    obs_params = []

    syn_params, constraints = mesh_syn(syn=False, **kwargs)
    obs_params += syn_params.to_list()
    #obs_params += mesh_dep(**kwargs).to_list()

    return ParameterSet(obs_params), constraints

def mesh_syn(syn=True, **kwargs):

    syn_params = []

    time = kwargs.get('time', [])
    # for protomeshes time will be [None] as it is defined to be at periastron.
    # In that case we don't want the time array parameter
    if not len(time) or time[0] is not None:
        syn_params += [FloatArrayParameter(qualifier='time', value=kwargs.get('time', []), default_unit=u.d, description='{} times'.format('Synthetic' if syn else 'Observed'))]

    # the following will all be arrays (value per triangle) per time
    # TODO: we really probably only want this in the syn, not obs - but do want time in both
    # TODO: I guess since time probably won't be passed when creating the obs these won't be there - we just have to make sure to pass times in compute
    if syn:
        # syns ignore copy_for anyways
        for t in kwargs.get('time', []):
            syn_params += [FloatArrayParameter(qualifier='x', time=t, value=kwargs.get('x', []), default_unit=u.solRad if t is not None else u.dimensionless_unscaled, description='X coordinate of center of triangles')]
            syn_params += [FloatArrayParameter(qualifier='y', time=t, value=kwargs.get('y', []), default_unit=u.solRad if t is not None else u.dimensionless_unscaled, description='Y coordinate of center of triangles')]
            syn_params += [FloatArrayParameter(qualifier='z', time=t, value=kwargs.get('z', []), default_unit=u.solRad if t is not None else u.dimensionless_unscaled, description='Z coordinate of center of triangles')]
            if t is not None:
                # skip these for protomeshes
                syn_params += [FloatParameter(qualifier='pot', time=t, value=kwargs.get('pot', 0.0), default_unit=u.dimensionless_unscaled, description='Equipotential of the stellar surface')]
                syn_params += [FloatParameter(qualifier='rpole', time=t, value=kwargs.get('rpole', 0.0), default_unit=u.solRad, description='Polar radius of the stellar surface')]
                syn_params += [FloatParameter(qualifier='volume', time=t, value=kwargs.get('volume', 0.0), default_unit=u.solRad**3, description='Volume of the stellar surface')]


                syn_params += [FloatArrayParameter(qualifier='vx', time=t, value=kwargs.get('vx', []), default_unit=u.solRad/u.d, description='X velocity of center of triangles')]
                syn_params += [FloatArrayParameter(qualifier='vy', time=t, value=kwargs.get('vy', []), default_unit=u.solRad/u.d, description='Y velocity of center of triangles')]
                syn_params += [FloatArrayParameter(qualifier='vz', time=t, value=kwargs.get('vz', []), default_unit=u.solRad/u.d, description='Z velocity of center of triangles')]
            syn_params += [FloatArrayParameter(qualifier='areas', time=t, value=kwargs.get('areas', []), default_unit=u.solRad**2, description='Area of triangles')]
            # syn_params += [FloatArrayParameter(qualifier='volumes', time=t, value=kwargs.get('volumes', []), default_unit=u.solRad**3, description='Volume of triangles')]
            syn_params += [FloatArrayParameter(qualifier='vertices', time=t, value=kwargs.get('vertices', []), default_unit=u.solRad if t is not None else u.dimensionless_unscaled, description='Vertices of triangles')]
            syn_params += [FloatArrayParameter(qualifier='visible_centroids', time=t, value=kwargs.get('visible_centroids', []), default_unit=u.solRad  if t is not None else u.dimensionless_unscaled, description='Center of the visible portion of each triangle')]
            syn_params += [FloatArrayParameter(qualifier='normals', time=t, value=kwargs.get('normals', []), default_unit=u.dimensionless_unscaled, description='Normals of triangles')]
            syn_params += [FloatArrayParameter(qualifier='nx', time=t, value=kwargs.get('nx', []), default_unit=u.dimensionless_unscaled, description='X component of normals')]
            syn_params += [FloatArrayParameter(qualifier='ny', time=t, value=kwargs.get('ny', []), default_unit=u.dimensionless_unscaled, description='Y component of normals')]
            syn_params += [FloatArrayParameter(qualifier='nz', time=t, value=kwargs.get('nz', []), default_unit=u.dimensionless_unscaled, description='Z component of normals')]


            syn_params += [FloatArrayParameter(qualifier='cosbeta', time=t, value=kwargs.get('cosbeta', []), default_unit=u.solRad if t is not None else u.dimensionless_unscaled, description='')]

            syn_params += [FloatArrayParameter(qualifier='logg', time=t, value=kwargs.get('logg', []), default_unit=u.dimensionless_unscaled, description='Local surface gravity')]
            syn_params += [FloatArrayParameter(qualifier='teff', time=t, value=kwargs.get('teff', []), default_unit=u.K, description='Local effective temperature')]

            syn_params += [FloatArrayParameter(qualifier='r', time=t, value=kwargs.get('r', []), default_unit=u.solRad if t is not None else u.dimensionless_unscaled, description='Distance of each triangle from center of mass')]


            if t is not None:
                # skip these for protomeshes
                syn_params += [FloatArrayParameter(qualifier='r_proj', time=t, value=kwargs.get('r_proj', []), default_unit=u.solRad, description='Projected distance (on plane of sky) of each triangle from center of mass')]

                syn_params += [FloatArrayParameter(qualifier='mu', time=t, value=kwargs.get('mu', []), default_unit=u.dimensionless_unscaled, description='Mu')]

                syn_params += [FloatArrayParameter(qualifier='visibility', time=t, value=kwargs.get('visibility', []), default_unit=u.dimensionless_unscaled, description='Visiblity of triangles (1=visible, 0.5=partial, 0=hidden)')]


                for dataset, method in kwargs.get('dataset_fields', {}).items():
                    # TODO: descriptions for each column
                    if method=='RV':
                        indeps = {'rv': u.solRad/u.d, 'intens_norm_abs': u.W/u.m**3, 'intens_norm_rel': u.W/u.m**3, 'intens_proj_abs': u.W/u.m**3, 'intens_proj_rel': u.W/u.m**3, 'ampl_boost': u.dimensionless_unscaled, 'ld': u.dimensionless_unscaled}
                    elif method=='LC':
                        indeps = {'intens_norm_abs': u.W/u.m**3, 'intens_norm_rel': u.W/u.m**3, 'intens_proj_abs': u.W/u.m**3, 'intens_proj_rel': u.W/u.m**3, 'ampl_boost': u.dimensionless_unscaled, 'ld': u.dimensionless_unscaled}
                    elif method=='MESH':
                        continue
                    else:
                        raise NotImplementedError

                    if method in ['LC', 'RV']:
                        syn_params += [FloatParameter(qualifier='pblum', dataset=dataset, time=t, value=kwargs.get('pblum', 0.0), default_unit=u.watt, description='Passband Luminosity of entire star')]


                    for indep, default_unit in indeps.items():
                        syn_params += [FloatArrayParameter(indep, dataset=dataset, time=t, value=[], default_unit=default_unit, description='Per-element value for {} dataset'.format(dataset))]

    constraints = []

    return ParameterSet(syn_params), constraints

def mesh_dep(**kwargs):

    dep_params = []

    return ParameterSet(dep_params)
