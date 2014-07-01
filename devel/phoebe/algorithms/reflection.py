"""

Algorithms related to (mutual) irradation.

.. _reflection-algorithms:

One star (A) can emit radiation onto the surface of another star (B).
Essentially two things can happen with that flux:

    1. The flux emitted by star A can be absorbed by star B and used to heat up
       star B. The heating can be local, but can in principle also be
       redistributed over the whole star, if there is a heat transport mechanism
       acting on the body. Here, this process is called **heating** (confusingly
       called the reflection effect in the binary star community).
       
    2. The flux emitted by star B can be reflected on the surface of star B,
       isotropically or aspect-dependently. Here, this process is called
       **reflection**. The function that determine the aspect angle dependency
       is called the **scattering phase function**.

It is important to note that heating is a **bolometric** effect, whereas
reflection is a **passband dependent** effect. As such, the albedo parameter
that is a property of a *body* denotes how much of the incident bolometric flux
is used to heat up the body (albedo=0 means all bolometric flux is used for
heating). By definition, the bolometric albedo is a value between 0 and 1.

On the other hand, the albedo parameter that is a property of an observable
(e.g. a light curve) is a wavelength dependent parameter. Equal values for all
passbands is called grey scattering, i.e. all incoming light is reflected in the
same wavelengths. For grey scattering, the albedo is a value between 0 and 1
(with 0 begin no reflection), but when there is some kind of energy
redistribution, this doesn't need to be the case. The albedo in our definition
can exceed 1, but can never be negative (it is Bond's albedo).

Note that the **heating** process we define here is actually called the
**reflection** effect in the Wilson-Devinney code. The general effect of flux
rays bouncing around in a system, we dub **irradation**. Thus before being able
to compute heating or reflection, we need to quantify the irradiation.

In the next sections, some physical and programmatical details are discussed.
For further details, please refer to the documentation of the relevant
functions.

Just remember: snow has an albedo close to 1 (not easily heated but good reflector),
coals has albedo close to 0 (easily heated but reflects almost no light).

Section 1. Irradiation
======================

For each surface element on the irradiated star, The lines-of-sight (C{los}) to
each surface element on the irradiator are calculated. Then, the angles between
the normal of this surface element and the lines-of-sight are calculated
(C{psi1}), as well as the angles between the lines-of-sight and the normals on
the irradiator (C{psi2}).

Section 2. Heating
==================

Section 3. Reflection
=====================

The aspect-angle dependency of the reflected light is governed by the scattering
phase function. The atmospheres of hot stars mainly contain free electrons as
scattering agents, which is **Thompson scattering**. Thompson scattering is grey
(i.e. the incoming photon doesn't change its wavelength) and is, in a very good
approximation, isotropic.

In cooler stars, molecules can be present, in which case **Rayleigh scattering**
can be important. Rayleigh scattering is not grey, and the scattering phase
function is forward-backwards symmetric (but not isotropic).

The Henyey-Greenstein phase function allows non-symmetric, backwards or forwards
dominated scattering through the asymmetry parameter :math:`g`. There is also
a two-term Henyey-Greenstein phase function available that can reproduce Jupiter's
observed phase function.

The Hapke model is a multi-parameter scattering model, originally built to
mimic the properties of regolith.

Important note: in order to have a somewhat feasible computation time, we
approximate the source of the scattering photons by a point source.


References
==========

    - Wilson's canonical paper, [Wilson1990]_
    - [Budaj2011]_


.. autosummary::

    radiation_budget_fast
    single_heating_reflection
    mutual_heating
    henyey_greenstein
    henyey_greenstein2
    rayleigh
    hapke
    
"""
import pylab as pl
import numpy as np
import logging
from numpy import pi,sqrt,sin,cos
from scipy.integrate import quad
from phoebe.algorithms import eclipse
from phoebe.algorithms import freflection
from phoebe.units import constants
from phoebe.backend import decorators
from phoebe.utils import coordinates
from phoebe.utils import cgeometry
from phoebe.utils import fgeometry
from phoebe.utils import plotlib
from phoebe.atmospheres import limbdark

logger = logging.getLogger("ALGO.REFL")
logger.addHandler(logging.NullHandler())


@decorators.parse_ref
def radiation_budget_slow(irradiated,irradiator,ref=None,third_bodies=None):
    r"""
    Calculate the radiation budget for heating and reflection.
    
    For all refs in C{ref}, the following is done:
        
        - if bolometric, the incoming radiation from all irradiator triangles
          visbible onto each triangle of the irradiated will be calculated, as
          well as the emergent flux coming from each triangle on irradiated. This
          can than be used to compute local or global heating.
        
        - if passband dependent, only the incoming radiation will be computed.
          Typically used for reflection effects.
          
    Some definitions used in the code and documentation:
        
        - :math:`\vec{s}` is the line-of-sight (``los``) between a triangle :math:`i`
          on the irradiator and a triangle :math:`j` on the irradiated body.
        - :math:`\psi_2^i`: the angle between the surface normal of triangle i
          of the irradiator and the line :math:`s` that connects the two surface
          elements.
        - :math:`\psi_1^j`: is the angle between the surface normal of triangle
          j of the irradiated body and the line :math:`s`.
        - :math:`\alpha` is the phase angle, or the angle between a triangle on
          the irradiator and the observer's line-of-sight.
    
    @param irradiated: star that gets irradiated. The flux or temperature of this object will be altered
    @type irradiated: Body
    @param irradiator: star that irradiates the C{irradated} body. The properties of this object will not be altered
    @type irradiator: Body
    @param ref: ref that specificies in which filter (could be bolometric) the irradiation budget needs to be calculated in
    @type ref: string or list of strings or None
    """
    def _tief(gamma,ld_law,coeffs):
        """Small helper function to compute total intrinsic emergent flux"""
        cos_gamma = cos(gamma)
        Imu = coeffs[-1]*ld_law(cos_gamma,coeffs)
        return Imu*cos_gamma*sin(gamma) # sin(gamma) is for solid angle integration
    Nl = len(ref)
    N = len(irradiated.mesh)
    #-- run over each triangle on the irradiated star, and compute how much
    #   radiation it receives from the irradiator.
    R1 = np.ones(N) # local heating
    R2 = 0.0 # global heating (redistributed)
    inco = np.ones((N,Nl)) # total flux that is incident on the triangle
    #day = np.zeros(N,bool)
    total_surface = irradiated.mesh['size'].sum()
    ps_irradiator = [irradiator.get_parset(ref=jref) for jref in ref]
    ps_irradiated = [irradiated.get_parset(ref=jref) for jref in ref]
    #-- we need to filter the references, because they can also contain
    #   references to parametersets that are in another body!
    ref = [ps[1] for ps in ps_irradiator if not ps[1] is None]
    ref_ed = [ps[1] for ps in ps_irradiated if not ps[1] is None]
    ld_models = [ps[0]['ld_func'] for ps in ps_irradiator if not ps[1] is None]
    ld_models_ed = [ps[0]['ld_func'] for ps in ps_irradiated if not ps[1] is None]
    A_irradiateds = [ps[0]['alb'] for ps in ps_irradiated if not ps[1] is None]
    P_redistrs = [(ps[0]['redist'] if ps[1]=='__bol' else 0.) for ps in ps_irradiated if not ps[1] is None]
    #if '__bol' in ref:
        #total_surface = irradiated.mesh['size'].sum()
    
    # Compute emergent bolometric intensities (we could perhaps speed it up a
    # bit by only computing it for those triangles that are facing the other
    # star (flux that is coming out of triangle)
    ld_disk = getattr(limbdark, 'disk_'+ld_models_ed[ref_ed.index('__bol')])
    emer = ld_disk(irradiated.mesh['ld___bol'][:,:-1].T) * irradiated.mesh['ld___bol'][:,-1]
    
    #from phoebe.utils import plotlib
    #albmap = plotlib.read_bitmap(irradiated, '/home/pieterd/workspace/phoebe/tests/test_reflection/moon_map.png')
    
    #import time
    #c0 = time.time()
    for i in range(N):
        #-- maybe some lines of sights are obstructed: skip those
        if third_bodies is not None and not (irradiated.label==third_bodies.label)\
             and not (irradiator.label==third_bodies.label):
            #-- for now, we assume the irradiator is a point source - I mean,
            #   really, a point source. So if you ask why I chose the tenth
            #   triangle as the reference triangle, my answer is: it is a point
            #   source, it doesn't matter. Yes yes, it's not correct.. then
            #   improve the speed and do it yourself!
            if True:
                obstructed = eclipse.get_obstructed_los(irradiated.mesh['center'][i],irradiator.mesh['center'][10:11],third_bodies)
                if obstructed[0]:
                    continue
                irradiator_mesh = irradiator.mesh
            else:
                obstructed = eclipse.get_obstructed_los(irradiated.mesh['center'][i],irradiator.mesh['center'],third_bodies)
                if np.sum(obstructed)==len(obstructed):
                    continue
                irradiator_mesh = irradiator.mesh[-obstructed]
            if i%100==0:
                logger.info('---> continuing {}/{} reflection with 3rd body obscuration'.format(i,N))
        else:
            irradiator_mesh = irradiator.mesh
            
        #-- what are the lines of sight?
        los = irradiator_mesh['center']-irradiated.mesh['center'][i]
        #-- what are the angles between the normal and the lines-of-sight on
        #   the irradiated object?
        #cos_psi1 = coordinates.cos_angle(irradiated.mesh['normal_'][i],los,axis=-1)
        cos_psi1 = fgeometry.cos_angle_3_nx3(irradiated.mesh['normal_'][i],los)
        
        # Phase angles: angles between body-body los and the real line-of-sight
        # the phase angle is aka alpha or g. It is the angle between the object
        # and the observer
        #phase_angle = np.arccos(fgeometry.cos_angle_3_nx3(np.array([0,0,-1]),los))
        
        
        #-- what are the angles between the normals and the line-of-sight on
        #   the irradiator?
        #cos_psi2 = coordinates.cos_angle(los,irradiator_mesh['normal_'],axis=-1)
        cos_psi2 = -fgeometry.cos_angle_nx3_nx3(los,irradiator_mesh['normal_'])
        keep = (0<cos_psi1) & (0<cos_psi2) #& (cos_psi1<=1) & (cos_psi2<=1)
        if not np.sum(keep):
            continue
        #day[i] = True
        for j, jref in enumerate(ref):
            #-- what is the bolometric flux this triangle on the irradiated object
            #   receives from the irradiator? The mu-angles are cos(psi2). We also
            #   need to correct for the projected size (cos_psi2) of the triangle
            #   on the irradiator.
            ld_law = getattr(limbdark,'ld_{}'.format(ld_models[j]))
            Imu0 = irradiator_mesh['ld_{}'.format(jref)][keep, -1]
            Ibolmu = Imu0*ld_law(cos_psi2[keep], irradiator_mesh['ld_{}'.format(jref)][keep].T)
            Ibolmu = Ibolmu*irradiator_mesh['size'][keep]*cos_psi2[keep]                
                
            #-- what are the distances to each triangle on the irradiator?
            distance2 = np.sum(los[keep]**2,axis=1)

            #-- every fluxray is emmited with an angle between the normal
            #   and the radial vector through the center of the star: not needed
            #   in our gridding approach because we know the size of the triangles
            #   already exactly
            #cos_gamma = coordinates.cos_angle(irradiator.mesh['center'][keep],irradiator.mesh['normal_'][keep],axis=-1)
            #Ibolmu = Ibolmu/cos_gamma
            #-- but every fluxray is also received under a specific angle on the
            #   irradiated object. The size of the receiving triangle doesn't matter
            Ibolmu = cos_psi1[keep]*Ibolmu
            
            # Scattering phase function
            g = 0.0
            if g != 0.0:
                pf = (1.0 - g) + 2*g*(np.abs(cos_psi1-cos_psi2)<0.01)
            else:
                pf = (1.0 - g)
                
            #-- the total (summed) projected intensity on this triangle is then
            #   dependent on the distance and the albedo
            proj_Ibolmu = np.sum(pf*Ibolmu/distance2)
            
            #-- what is the total intrinsic emergent flux from this triangle? We
            #   need to integrate over a solid angle of 2pi, that is let phi run
            #   from 0->2*pi and theta from 0->pi/2
            inco[i,j] = proj_Ibolmu
            
            #-- no need to calculate emergent flux if not bolometric!
            if jref != '__bol':
                continue
            
            #emer_Ibolmu = 2*pi*quad(_tief, 0, pi/2, args=(ld_law, irradiated.mesh['ld_{}'.format(jref)][i]))[0] # 2pi is for solid angle integration over phi
            
            #-- compute the excess of the total irradiated flux over the flux that
            #   would be radiated in absence of heating. We need to take care of
            #   possible (uniform) redistribution here: only a fraction is used to
            #   heat this particular triangle, the rest is used to heat up the whole
            #   object
            #A_irradiateds[j] = albmap[i]
            R1[i] = 1.0 + (1-P_redistrs[j])*A_irradiateds[j]*proj_Ibolmu/emer[i]
            R2 +=            P_redistrs[j] *A_irradiateds[j]*proj_Ibolmu/emer[i]\
                                       *irradiated.mesh['size'][i]
    
    # Global redistribution factor:
    R2 = 1.0 + R2/total_surface
    
    return R1,R2,inco,emer,ref,A_irradiateds


@decorators.parse_ref
def radiation_budget_fast(irradiated, irradiator, ref=None, third_bodies=None,
                          irradiation_alg='point_source'):
    """
    Calculate the radiation budget for heating and reflection.
    
    For all refs in C{ref}, the following is done:
        
        - if bolometric, the incoming radiation from all irradiator triangles
          visbible onto each triangle of the irradiated will be calculated, as
          well as the emergent flux coming from each triangle on irradiated. This
          can than be used to compute local or global heating.
        
        - if passband dependent, only the incoming radiation will be computed.
          Typically used for reflection effects.
    
    @param irradiated: star that gets irradiated. The flux or temperature of this object will be altered
    @type irradiated: Body
    @param irradiator: star that irradiates the C{irradated} body. The properties of this object will not be altered
    @type irradiator: Body
    @param ref: ref that specificies in which filter (could be bolometric) the irradiation budget needs to be calculated in
    @type ref: string or list of strings or None
    """
    def _tief(gamma,ld_law,coeffs):
        """Small helper function to compute total intrinsic emergent flux"""
        cos_gamma = cos(gamma)
        Imu = coeffs[-1]*ld_law(cos_gamma,coeffs)
        return Imu*cos_gamma*sin(gamma) # sin(gamma) is for solid angle integration
    
    Nl = len(ref)
    N = len(irradiated.mesh)
    
    #-- run over each triangle on the irradiated star, and compute how much
    #   radiation it receives from the irradiator.
    R1 = np.ones(N) # local heating
    R2 = 1. # global heating (redistributed)
    inco = np.ones((N,Nl)) # total flux that is incident on the triangle
    total_surface = irradiated.mesh['size'].sum()
    ps_irradiator = [irradiator.get_parset(ref=jref) for jref in ref]
    ps_irradiated = [irradiated.get_parset(ref=jref) for jref in ref]
    #-- we need to filter the references, because they can also contain
    #   references to parametersets that are in another body!
    ref = [ps[1] for ps in ps_irradiator if not ps[1] is None]
    ref_ed = [ps[1] for ps in ps_irradiated if not ps[1] is None]
    ld_models = [ps[0]['ld_func'] for ps in ps_irradiator if not ps[1] is None]
    ld_models_ed = [ps[0]['ld_func'] for ps in ps_irradiated if not ps[1] is None]
    A_irradiateds = [ps[0]['alb'] for ps in ps_irradiated if not ps[1] is None]
    P_redistrs = [(ps[0]['redist'] if ps[1]=='__bol' else 0.) for ps in ps_irradiated if not ps[1] is None]
    H_redistrs = [(ps[0]['redisth'] if ps[1]=='__bol' else 0.) for ps in ps_irradiated if not ps[1] is None]
    #if '__bol' in ref:
        #total_surface = irradiated.mesh['size'].sum()
    
    # Compute emergent bolometric intensities (we could perhaps speed it up a
    # bit by only computing it for those triangles that are facing the other
    # star (flux that is coming out of triangle)
    ld_disk = getattr(limbdark, 'disk_'+ld_models_ed[ref_ed.index('__bol')])
    emer = ld_disk(irradiated.mesh['ld___bol'][:,:-1].T) * irradiated.mesh['ld___bol'][:,-1]
    
    
    index_bol = ref.index('__bol')
    irrorld = [irradiator.mesh['ld___bol']] + [irradiator.mesh['ld_{}'.format(iref)] for iref in ref[:index_bol]+ref[index_bol+1:]]
    alb = A_irradiateds[index_bol]
    redist = P_redistrs[index_bol]
    redisth = H_redistrs[index_bol]
    ld_laws_indices = ['claret', 'linear', 'nonlinear', 'logarithmic',
                       'quadratic','square_root', 'uniform']
    ld_laws = [ld_laws_indices.index(ld_law) for ld_law in ld_models]
    
    # It is possible that the albedo's are in the mesh, in that case we treat
    # the albedo's not as a single value but interpret it as a map
    if 'alb' in irradiated.mesh.dtype.names:
        refl_algorithm = freflection.reflectionarray 
        alb = irradiated.mesh['alb']
        redist = redist*np.ones_like(alb)
    else:
        refl_algorithm = freflection.reflection
    
        
    # There are different approximations for the irradiation: treat the
    # irradiator as a point source or an extended body
    
    if irradiation_alg == 'point_source':
        # For a point source, we need to know the center-coordinates of the
        # irradiator, it's projected surface area and the direction. This
        # way we can treat the irradiator as a single surface element.
        
        # line-of-sight and distance between two centers of the Bodies
        X1 = (irradiator.mesh['center']*irradiator.mesh['size'][:,None]/irradiator.mesh['size'].sum()).sum(axis=0)
        X2 = (irradiated.mesh['center']*irradiated.mesh['size'][:,None]/irradiated.mesh['size'].sum()).sum(axis=0)
        irradiator_mesh_normal = np.array([X2-X1])
        irradiator_mesh_normal/= np.sqrt((irradiator_mesh_normal**2).sum())
        d = np.sqrt( ((X1-X2)**2).sum())
        
        # mu-angles from irradiator to irradiated (what does the irradiator see?)
        mus = coordinates.cos_angle(irradiator.mesh['normal_'], irradiator_mesh_normal, axis=1)
                    
        # let's say that this is the location of the irradiator
        irradiator_mesh_center = np.array([X1])
        irradiator_mesh_size = np.array([1.0])
        
        # Then compute the projected intensity of the irradiator towards
        # the irradiated body
        irrorld_ = []
        keep = mus > 0
        
        # Summarize the mesh of the irradiator as one surface element
        these_fields = irradiator.mesh.dtype.names
        for i, ild in enumerate(irrorld):
            this_ld_func = getattr(limbdark, 'ld_{}'.format(ld_laws_indices[ld_laws[i]]))
            Imu = this_ld_func(mus[keep], irradiator.mesh['ld_'+ref[i]][keep].T)
            proj_Imu = irradiator.mesh['ld_'+ref[i]][keep,-1] * Imu
            if 'refl_'+ref[i] in these_fields:
                proj_Imu += irradiator.mesh['refl_'+ref[i]][keep] * mus[keep]
            proj_Imu *= irradiator.mesh['size'][keep]* mus[keep] # not sure about this mus!!
            
            ld_laws[i] = 6 # i.e. uniform, because we did all the projection already
            irrorld[i] = np.array([[0.0,0.0,0.0,0.0,proj_Imu.sum()]])
        
    # Else, we'll use the full mesh of the irradiator    
    elif irradiation_alg == 'full':
        irradiator_mesh_center = irradiator.mesh['center']
        irradiator_mesh_size = irradiator.mesh['size']
        irradiator_mesh_normal = irradiator.mesh['normal_']
    else:
        raise NotImplementedError("Irradiation algorithm {} unknown".format(irradiation_alg))
    
    R1, R2, inco = refl_algorithm(irradiator_mesh_center,
                       irradiator_mesh_size,
                       irradiator_mesh_normal, irrorld,
                       irradiated.mesh['center'], irradiated.mesh['size'],
                       irradiated.mesh['normal_'], emer,
                       alb, redist, ld_laws)
    
    # Global redistribution factor:
    R2 = 1.0 + (1-redisth)*R2/total_surface
    
    # To do horizontal redistribution instead of global, perhaps do something
    # like:
    if redisth > 0:
        logger.info("Performing latitudinal heat redistribution")
        rad, longit, colat = irradiated.get_coords(type='spherical', loc='center')
        bands = np.linspace(0, np.pi, 50)
        # For easy reference, create a version of the mesh where all triangles are
        # ordered according to colatitude. Also create an array to translate back
        # to the original frame
        sort_colat = np.argsort(colat)
        inv_sort = np.argsort(sort_colat)
        indices = np.arange(len(colat))
        R2 = R2*np.ones_like(R1)
        for i in range(len(bands)-1):
            # Select this band of triangles:
            start = np.searchsorted(colat[sort_colat], bands[i])
            end = np.searchsorted(colat[sort_colat], bands[i+1])
            use = indices[sort_colat][start:end]
            band_total_surface = irradiated.mesh['size'][use].sum()
            if not np.isscalar(redist):
                factor = (redist[use]*alb[use])
            else:
                factor = redist*alb
            R2[use] += (redisth*factor * inco[use,0]/emer[use]*irradiated.mesh['size'][use]).sum()/band_total_surface
        
    return R1, R2, inco, emer, ref, A_irradiateds

        
def single_heating_reflection(irradiated, irradiator, update_temperature=True,\
            heating=True, reflection=False, third_bodies=None,\
            irradiation_alg='point_source'):
    """
    Compute heating and reflection for an irradiating star on an irradiated star.
    """
    if heating and reflection:
        ref = 'all'
    elif heating:
        ref = '__bol'        
    elif reflection:
        ref = 'all'#'alldep'
    else: # useless option, except perhaps for debugging
        ref = 'all'
    
    R1, R2, inco, emer, refs, A_irradiateds = radiation_budget_fast(irradiated,
                                                irradiator, ref=ref,
                                                third_bodies=third_bodies,
                                                irradiation_alg=irradiation_alg)
    #-- heating part:
    if heating:
        # update luminosities and temperatures: we need to copy and replace the
        # mesh to be able to handle body bags too.
        
        # This would check if radiative equilibrium is still valid:
        #before_lum = irradiated.luminosity()
        #total_surface = irradiated.mesh['size'].sum()
        #term1 = (R1*irradiated.mesh['size']).sum()/total_surface
        #term2 = R2
        #print("Predicted luminosity factor increase = {}".format(term1 + term2)) 
        
        
        irradiated_mesh = irradiated.mesh.copy()
        teff_old = irradiated_mesh['teff'].copy()
        irradiated_mesh['teff'] *= (R1+R2-1)**0.25
        irradiated.mesh = irradiated_mesh
        irradiated.intensity(ref=['__bol'])
        #after_lum = irradiated.luminosity()
        #print("Computed luminosity factor increase = {}".format(after_lum/before_lum))
        #limbdark.local_intensity(irradiated,
        #                         irradiated.get_parset(ref='__bol')[0])
        logger.info(("single heating effect: updated luminosity{} of {} "
                       "according to max "
                       "Teff increase of {:.3f}%").format((update_temperature and ' and teff' or '(no teff)'),irradiated.get_label(), (((R1+R2-1)**0.25).max()-1)*100))
        #-- still not sure if I should include the following
        #   If I do, I don't seem to converge... --- uh yes it does! -- uh no it
        #   doesn't... I don't know anymore
        #if not update_temperature:
        #  irradiated_mesh = irradiated.mesh.copy()
        #  irradiated_mesh['teff'] = teff_old
        #  irradiated.mesh = irradiated_mesh
    
    
    # Reflection part:
    if reflection:
        
        # Whatever is not used for heating can be reflected
        for j,jref in enumerate(refs):
            
            refl_ref = 'refl_{}'.format(jref)
            
            # In our definition, Bond albedo *is* A_irradiatied (it is the
            # opposite in WD)
            try:
                bond_albedo = A_irradiateds[j]
            
                if np.isscalar(A_irradiateds[j]):
                    bond_albedo = max(0, bond_albedo)
                else:
                    bond_albedo[bond_albedo<0] = 0.
                irradiated.mesh[refl_ref] += bond_albedo*inco[:,j]
            
            except ValueError:
                raise
                raise ValueError("Did not find ref {}. Did you prepare for reflection?".format(refl_ref))
            
                
    

def single_heating(irradiated,irradiator,ld_func='claret',update_temperature=True):
    """
    Compute heating effect.
    
    Needs to be done:
        1. Skip triangles facing backwards, both on the irradiator and
        irradiated object (will make it more efficient).
        2. Iterate.
    
    Extra options to implement:
        1. simplification: assume irradiator is point source
        2. careful eclipse/horizon detection
        3. Thermal relaxation?
        
    Heat redistribution is inspired upon Budaj 2011, but should hold for
    binaries too, instead of only star-planet systems.
        
    When C{update_temperature} is False, the local luminosities will be updated
    without changing the local temperature. This can be useful for iteration of
    heating effect.
    """
    A_irradiated = irradiated.params['component'].request_value('alb')
    P_redistr = irradiated.params['component'].request_value('redist')
    A_B = 1 - A_irradiated # Bond Albedo
    def _tief(gamma,ld_law,coeffs):
        """Small helper function to compute total intrinsic emergent flux"""
        Imu = coeffs[-1]*ld_law(cos(gamma),coeffs)
        return Imu*cos(gamma)*sin(gamma) # sin(gamma) is for solid angle integration
    #-- run over each triangle on the irradiated star, and compute how much
    #   radiation it receives from the irradiator.
    N = len(irradiated.mesh)
    R1 = np.ones(N) # local heating
    R2 = 1. # global heating (redistributed)
    emer = np.ones(N)
    inco = np.ones(N)
    day = np.zeros(N,bool)
    total_surface = irradiated.mesh['size'].sum()
    for i in range(N):
        
        # What are the lines of sight?
        los = irradiator.mesh['center']-irradiated.mesh['center'][i]
        
        # What are the angles between the normal and the lines-of-sight on
        # the irradated object?
        cos_psi1 = coordinates.cos_angle(irradiated.mesh['normal_'][i],los,axis=-1)
        #cos_psi1 = cgeometry.cos_theta(irradiated.mesh['normal_'][i],los,axis=-1)
        
        # What are the angles between the normals and the line-of-sight on
        # the irradiator?
        cos_psi2 = coordinates.cos_angle(los,irradiator.mesh['normal_'],axis=-1)
        #cos_psi2 = cgeometry.cos_theta(los,irradiator.mesh['normal_'],axis=-1)
        keep = (cos_psi1<1) & (cos_psi2<1) & (0<cos_psi1) & (0<cos_psi2)
        if not np.sum(keep): continue
        day[i] = True
        
        #-- what is the bolometric flux this triangle on the irradiated object
        #   receives from the irradiator? The mu-angles are cos(psi2). We also
        #   need to correct for the projected size (cos_psi2) of the triangle
        #   on the irradiator.
        ld_law = getattr(limbdark,'ld_%s'%(ld_model))
        Imu0 = irradiator.mesh['ld___bol'][keep,-1]
        Ibolmu = Imu0*ld_law(cos_psi2[keep],irradiator.mesh['ld___bol'][keep].T)
        Ibolmu = irradiator.mesh['size'][keep]*cos_psi2[keep]*Ibolmu
        #-- every fluxray is emmited with an angle between the normal
        #   and the radial vector through the center of the star: not needed
        #   in our gridding approach because we know the size of the triangles
        #   already exactly
        #cos_gamma = coordinates.cos_angle(irradiator.mesh['center'][keep],irradiator.mesh['normal_'][keep],axis=-1)
        #Ibolmu = Ibolmu/cos_gamma
        #-- but every fluxray is also received under a specific angle on the
        #   irradiated object. The size of the receiving triangle doesn't matter
        Ibolmu = cos_psi1[keep]*Ibolmu
        #-- what are the distances to each triangle on the irradiator?
        distance = sqrt(np.sum(los[keep]**2,axis=1))
        #-- the total (summed) projected intensity on this triangle is then
        #   dependent on the distance and the albedo
        proj_Ibolmu = A_irradiated*np.sum(Ibolmu/distance**2)
        #-- what is the total intrinsic emergent flux from this triangle? We
        #   need to integrate over a solid angle of 2pi, that is let phi run
        #   from 0->2*pi and theta from 0->pi/2
        emer_Ibolmu = 2*pi*quad(_tief,0,pi/2,args=(ld_law,irradiated.mesh['ld___bol'][i]))[0] # 2pi is for solid angle integration over phi
        
        #====== START CHECK FOR EMERGENT FLUX ======
        #-- in the case of a linear limb darkening law, we can easily evaluate
        #   the local emergent flux (Eq. 5.27 from Phoebe Scientific reference)
        #__xbol = 0.123
        #__Ibol = 1.543
        #__num = 2*pi*quad(tief,0,pi/2,args=(limbdark.ld_linear,[__xbol,0,0,0,__Ibol]))[0]
        #__ana = __Ibol*pi*(1-__xbol/3.) 
        #print __num,__ana
        #======   END CHECK FOR EMERGENT FLUX ======
        
        #-- compute the excess of the total irradiated flux over the flux that
        #   would be radiated in absence of heating. We need to take care of
        #   possible (uniform) redistribution here: only a fraction is used to
        #   heat this particular triangle, the rest is used to heat up the whole
        #   object
        R1[i] = 1.0 + (1-P_redistr)*proj_Ibolmu/emer_Ibolmu
        R2 += P_redistr *proj_Ibolmu/emer_Ibolmu/total_surface
        emer[i] = emer_Ibolmu
        inco[i] = proj_Ibolmu
    
    #-- heat redistribution Budaj 2011, Eq. (12):
    #print '... TOR',irradiated.mesh['teff'].min(),irradiated.mesh['teff'].max()
    #print '... TIR',(irradiated.mesh['teff']*R1**0.25).min(),(irradiated.mesh['teff']*R1**0.25).max()
    #print '... TDN',(irradiated.mesh['teff']*R2**0.25).min(),(irradiated.mesh['teff']*R2**0.25).max()
    
    #-- update luminosities and temperatures
    teff_old = irradiated.mesh['teff'].copy()        
    irradiated.mesh['teff'] *= R1**0.25
    limbdark.local_intensity(irradiated,irradiated.get_parset(ref='__bol')[0])
    logger.info("single heating effect: updated luminosity%s according to max Teff increase of %.3f%%"%((update_temperature and ' and teff' or ' (no teff)'),((R1**0.25).max()-1)*100))
    if not update_temperature:
        irradiated.mesh['teff'] = teff_old


def mutual_heating(*objects,**kwargs):
    """
    Iteratively update local quantities due to heating.
    """
    n = kwargs.pop('niter',1)
    kwargs.setdefault('heating',True)
    kwargs.setdefault('reflection',False)
    kwargs.setdefault('irradiation_alg', 'point_source')
    #kwargs.setdefault('reflection',False)
    #kwargs.setdefault('heating',True)
    #-- expand bodybags --> this should be rewritten for deeper nesting
    objects_ = []
    for iobj in objects:
        try:
            objects_ += iobj.bodies
        except AttributeError:
            objects_.append(iobj)
    objects = objects_
    
    #-- do we want to set the local effective temperature after mutual
    #   reflection?
    update_temperature = kwargs.pop('update_temperature',True)
    logger.info('mutual heating: {:d} iteration(s) on {:d} objects (alg={})'.format(n,len(objects),kwargs['irradiation_alg']))
    for i in range(n):
        for j,obj1 in enumerate(objects):
            for k,obj2 in enumerate(objects):
                if j==k: continue
                #-- skip body if it's not an irradiator
                if not obj2.params.values()[0]['irradiator']:
                    logger.info('object %d is not an irradiator --> object %d does not receive flux'%(k,j))
                    continue
                else:
                    logger.info('object %d is an irradiator --> object %d receives flux'%(k,j))
                #-- we sure don't want to update the temperature while iterating, but
                #   maybe we want to update the temperatures during the last iteration?
                #   So only update temperature when asked and in the last iteration
                #   We need to make a special case for when we're iterating
                #   over the last body in the list
                upd_temp = (update_temperature & (i==(n-1))) and True or False
                #print '--> update temp',upd_temp
                #upd_temp = (upd_temp & ((k==(len(objects)-1)) | ((k==(len(objects)-2)) & (j==len(objects)-1)) )) and True or False
                #print '--> update temp',upd_temp
                logger.info('mutual reflection: compute RE from body %d on body %d'%(k,j))
                #single_heating(obj1,obj2,update_temperature=upd_temp,**kwargs)
                single_heating_reflection(obj1,obj2,update_temperature=upd_temp,**kwargs)
    


# Scattering phase function
def henyey_greenstein(mu, g=0.0):
    r"""
    The Henyey-Greenstein scattering phase function.
    
    .. math::
    
        P(\mu) = \frac{1-g^2}{(1+g^2 - 2g\mu)^{3/2}}
        
    where :math:`\mu` is the scattering angle, or the angle between the incoming
    radiation and the scattered radiation in the line-of-sight. The parameter
    :math:`g`  is the asymmetry factor. When :math:`g=0`, the phase function
    simplifies to isotropic scattering. Positive :math:`g` implies forward-dominated
    scattering, negative :math:`g` means backwards-dominated scattering. When
    :math:`g=1`, there is only forward scattering; i.e. for any other phase
    angle, there will be no light reflected.
    
    Reference: [Henyey1941]_
    
    @param mu: scattering phase angle
    @type mu: float or array
    @param g: asymmetry factor
    @type g: float
    @return: scattering phase function
    @rtype: float or array
    """
    phase_function = (1-g**2) / (1 + g**2 - 2*g*mu)**1.5
    return phase_function


def henyey_greenstein2(mu, g1=0.8, g2=-0.38, f=0.9):
    r"""
    The two-term Henyey-Greenstein scattering phase function.
    
    .. math::
    
        P(\mu) = f P_\mathrm{HG}(g_1, \mu) + (1-f)P_\mathrm{HG}(g_2, \mu)
    
    .. math::
    
        P_\mathrm{HG}(\mu) = \frac{1-g^2}{(1+g^2 - 2g\mu)^{3/2}}
    
    where :math:`\mu` is the scattering angle, or the angle between the incoming
    radiation and the scattered radiation in the line-of-sight. The parameter
    :math:`g_i`  is the asymmetry factor. :math:`f` denotes the fraction of
    the forward versus backward scattering. :math:`g_1` controls the sharpness
    of the forward scattering lobe (and should be positive), while :math:`g_2`
    controls the sharpness of the backscattering lobe.
    
    This scattering phase function can reproduce Jupiter's phase function, if
    :math:`g_1=0.8`, :math:`g_2=-0.38` and :math:`f=0.9` (the defaults).
    
    Reference: http://www.gps.caltech.edu/~ulyana/www_papers/exosaturn/exosaturn_submitted.pdf
    where we have a different definition of :math:`\mu`.
    
    @param mu: scattering phase angle
    @type mu: float or array
    @param g: asymmetry factor
    @type g: float
    @return: scattering phase function
    @rtype: float or array
    """
    term1 = henyey_greenstein(mu, g1)
    term2 = henyey_greenstein(mu, g2)
    phase_function = f*term1 + (1-f)*term2
    return phase_function


def rayleigh(mu):
    r"""
    Rayleigh scattering phase function.
    
    .. math::
    
        P(\mu) = \frac{3}{8} (1-\mu^2)
        
    
    @param mu: scattering phase angle
    @type mu: float or array
    @return: scattering phase function
    @rtype: float or array
    """
    phase_function = 3.0/8.0 * (1-mu**2)
    return phase_function


def hapke(mu, mup, muv, alb=0.99, q=0.6, S=0.0, h=0.995):
    r"""
    Hapke scattering phase function.
    
    (http://stratus.ssec.wisc.edu/streamer/userman/surfalb.html)
    
    .. math::
    
        P(\mu, \mu_p, \mu_v) = \frac{\omega}{4}\frac{1}{\mu_v+\mu_p} \left( (1+B(g))P(g) + H(\mu)H(\mu_p) -1\right)
        
    with
    
    .. math::
    
        H(x) = \frac{1 + 2x}{1+2\mu\sqrt{1-\omega}}\\
        B(g) = \frac{B_0}{1 + \frac{1}{h}\tan(g/2)}
    
    and
    
    .. math::
    
        B_0 = \frac{S}{\omega} \frac{(1 + \Theta^2 + 2\Theta)^{3/2}}{1-\Theta^2}
    
    where :math:`\mu=\cos(g)` is the cosine of the scattering angle :math:`g`,
    :math:`\mu_v` is the cosine of the viewing angle, and :math:`\mu_p` is the
    cosine of the incidence angle. :math:`\Theta` is the asymmetry parameter,
    :math:`\omega` the single scattering albedo, :math:`S` the hot spot
    amplitude and :math:`h` the hot spot width.
    
    Vegetation (clover): w=0.101, q=-0.263, S=0.589, h=0.046 (Pinty and Verstraete 1991)
    Snow: w=0.99, q=0.6, S=0.0, h=0.995 (Domingue 1997, Verbiscer and Veverka 1990)
                
    @param mup: cosine of incidence angle
    @param muv: cosine of viewing angle
    @param mu: cosine of scattering angle
    """
    # phase (scattering) angle
    g = np.arccos(mu)
    
    # backscattering function
    B0 = S/alb / (1-q**2) * (1 + q**2 + 2*q)**1.5    # yes, that's a plus
    Bg = B0 / (1 + 1./h*np.tan(0.5*g))
                    
    # multiple scattering term
    Hmu = (1+2*mu)  / (1 +2*mu *np.sqrt(1-alb))
    Hmup= (1+2*mup) / (1 +2*mup*np.sqrt(1-alb))
                    
    phase_function = 0.25*alb * (1.0/(mus + mup)) * ((1 + Bg)*Pg + Hmu*Hmup - 1) 
                    
    return phase_function
                