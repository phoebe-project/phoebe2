"""
Algorithms related to (mutual) irradation.

One star (A) can emit radiation onto the surface of another star (B).
Essentially two things can happen with that flux:

    1. The flux emmited by star A can be absorbed by star B and used to heat up
       star B (heating)
       
    2. The flux emmitted by star B can be reflected on the surface of star B,
       isotropically or aspect-dependently (reflection).

In the next sections, some physical and programmatical details are discussed.
For further details, please refer to the documentation of the relevant
functions.

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



References
==========

    - Wilson's canonical paper, [Wilson1990]_
    - [Budaj2011]_
    
"""
import pylab as pl
import numpy as np
import logging
from numpy import pi,sqrt,sin,cos
from scipy.integrate import quad
from phoebe.algorithms import eclipse
from phoebe.units import constants
from phoebe.backend import decorators
from phoebe.utils import coordinates
from phoebe.atmospheres import limbdark

logger = logging.getLogger("ALGO.REFL")

@decorators.parse_ref
def radiation_budget(irradiated,irradiator,ref=None,third_bodies=None):
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
        Imu = coeffs[-1]*ld_law(cos(gamma),coeffs)
        return Imu*cos(gamma)*sin(gamma) # sin(gamma) is for solid angle integration
    Nl = len(ref)
    N = len(irradiated.mesh)
    #-- run over each triangle on the irradiated star, and compute how much
    #   radiation it receives from the irradiator.
    R1 = np.ones(N) # local heating
    R2 = 1. # global heating (redistributed)
    emer = np.ones((N,Nl)) # flux that is coming out of triangle
    inco = np.ones((N,Nl)) # total flux that is incident on the triangle
    day = np.zeros(N,bool)
    total_surface = irradiated.mesh['size'].sum()
    ps_irradiator = [irradiator.get_parset(ref=jref) for jref in ref]
    ps_irradiated = [irradiated.get_parset(ref=jref) for jref in ref]
    #-- we need to filter the references, because they can also contain
    #   references to parametersets that are in another body!
    ref = [ps[1] for ps in ps_irradiator if not ps[1] is None]
    ld_models = [ps[0]['ld_func'] for ps in ps_irradiator if not ps[1] is None]
    A_irradiateds = [ps[0]['alb'] for ps in ps_irradiated if not ps[1] is None]
    P_redistrs = [(ps[0]['redist'] if ps[1]=='__bol' else 0.) for ps in ps_irradiated if not ps[1] is None]
    #if '__bol' in ref:
        #total_surface = irradiated.mesh['size'].sum()
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
        #   the irradated object?
        cos_psi1 = coordinates.cos_angle(irradiated.mesh['normal_'][i],los,axis=-1)
        #-- what are the angles between the normals and the line-of-sight on
        #   the irradiator?
        cos_psi2 = coordinates.cos_angle(los,irradiator_mesh['normal_'],axis=-1)
        keep = (cos_psi1<1) & (cos_psi2<1) & (0<cos_psi1) & (0<cos_psi2)
        if not np.sum(keep): continue
        day[i] = True
        for j,jref in enumerate(ref):
            #-- what is the bolometric flux this triangle on the irradiated object
            #   receives from the irradiator? The mu-angles are cos(psi2). We also
            #   need to correct for the projected size (cos_psi2) of the triangle
            #   on the irradiator.
            ld_law = getattr(limbdark,'ld_{}'.format(ld_models[j]))
            Imu0 = irradiator_mesh['ld_{}'.format(jref)][keep,-1]
            Ibolmu = Imu0*ld_law(cos_psi2[keep],irradiator_mesh['ld_{}'.format(jref)][keep].T)
            Ibolmu = irradiator_mesh['size'][keep]*cos_psi2[keep]*Ibolmu
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
            proj_Ibolmu = np.sum(Ibolmu/distance**2)
            #-- what is the total intrinsic emergent flux from this triangle? We
            #   need to integrate over a solid angle of 2pi, that is let phi run
            #   from 0->2*pi and theta from 0->pi/2
            inco[i,j] = proj_Ibolmu
            #-- no need to calculate emergent flux if not bolometric!
            if jref!='__bol': continue
            emer_Ibolmu = 2*pi*quad(_tief,0,pi/2,args=(ld_law,irradiated.mesh['ld_{}'.format(jref)][i]))[0] # 2pi is for solid angle integration over phi
            
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
            R1[i] = 1.0 + (1-P_redistrs[j])*A_irradiateds[j]*proj_Ibolmu/emer_Ibolmu
            R2 +=            P_redistrs[j] *A_irradiateds[j]*proj_Ibolmu/emer_Ibolmu*irradiated.mesh['size'][i]/total_surface
            emer[i] = emer_Ibolmu
            
    return R1,R2,inco,emer,ref,A_irradiateds
        
def single_heating_reflection(irradiated,irradiator,update_temperature=True,\
            heating=True,reflection=False,third_bodies=None):
    if heating and reflection:
        ref = 'all'
    elif heating:
        ref = '__bol'        
    elif reflection:
        ref = 'alldep'
    
    R1,R2,inco,emer,refs,A_irradiateds = radiation_budget(irradiated,irradiator,ref=ref,third_bodies=third_bodies)
    
    #-- heating part:
    if heating:
        # update luminosities and temperatures: we need to copy and replace the
        # mesh to be able to handle body bags too.
        irradiated_mesh = irradiated.mesh.copy()
        teff_old = irradiated_mesh['teff'].copy()        
        irradiated_mesh['teff'] *= (R1+R2-1)**0.25
        irradiated.mesh = irradiated_mesh
        limbdark.local_intensity(irradiated,irradiated.get_parset(ref='__bol')[0])
        #logger.info("single heating effect: updated luminosity%s according to max Teff increase of %.3f%%"%((update_temperature and ' and teff' or '(no teff)'),(((R1+R2-1)**0.25).max()-1)*100))
        #-- still not sure if I should include the following
        #   If I do, I don't seem to converge... --- uh yes it does!
        if not update_temperature:
           irradiated_mesh = irradiated.mesh.copy()
           irradiated_mesh['teff'] = teff_old
           irradiated.mesh = irradiated_mesh
    #-- reflection part:
    if reflection:
        # whatever is not used for heating can be reflected, we assume isotropic
        # reflection
        for j,jref in enumerate(refs):
            refl_ref = 'refl_{}'.format(jref)
            #-- Bond albedo equals 1-A_irradiated
            try:
                irradiated.mesh[refl_ref] += (1-A_irradiateds[j])*inco[:,j]/np.pi
            except ValueError:
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
        #-- what are the lines of sight?
        los = irradiator.mesh['center']-irradiated.mesh['center'][i]
        #-- what are the angles between the normal and the lines-of-sight on
        #   the irradated object?
        cos_psi1 = coordinates.cos_angle(irradiated.mesh['normal_'][i],los,axis=-1)
        #-- what are the angles between the normals and the line-of-sight on
        #   the irradiator?
        cos_psi2 = coordinates.cos_angle(los,irradiator.mesh['normal_'],axis=-1)
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
    #kwargs.setdefault('reflection',False)
    #kwargs.setdefault('heating',True)
    #-- do we want to set the local effective temperature after mutual
    #   reflection?
    update_temperature = kwargs.pop('update_temperature',True)
    logger.info('mutual heating: {:d} iteration(s) on {:d} objects'.format(n,len(objects)))
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
    
