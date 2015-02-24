import phoebe
import numpy as np
from phoebe.atmospheres.roche import binary_potential
import phoebe.algorithms.marching as marching
from phoebe.utils.coordinates import cart2spher_coord, spher2cart_coord
from phoebe.units.constants import sigma
from phoebe.units.conversions import convert
from phoebe.atmospheres.roche import potential2radius
from phoebe.dynamics.keplerorbit import place_in_binary_orbit
from phoebe.algorithms.eclipse import ray_triangle_intersection
from phoebe.atmospheres.roche import exact_lagrangian_points
from scipy.optimize import fsolve, curve_fit
from scipy.ndimage.filters import gaussian_filter1d

def exact_critical_pots(q,sma=1., d=1.,F=1.):
    
    ''' Copied from phoebe1 because phoebe2 calculate_critical_potentials sets 
    L2 and L3 to 0. '''
    
    #L1
    dxL = 1.0
    L1 = 1e-3
    while abs(dxL) > 1e-6:
        dxL=-marching.dBinaryRochedx([L1,0.0,0.0],d,q,
                                F)/marching.d2BinaryRochedx2([L1,0.0,0.0],d,q,F)
        L1 = L1 + dxL
    Phi_L1 = -binary_potential(L1*sma, np.pi/2.0, 0.0, 0.0, q, d=1., F=1.)
    
    #L2
    if (q > 1.0):
        q2 = 1.0/q 
    else:
        q2 = q
    
    dxL = 1.1e-6
    
    D = d*sma
    factor = (q2/3/(q2+1))**1./3.
    xL = 1 + factor + 1./3.*factor**2 + 1./9.*factor**3
    while (abs(dxL) > 1e-6):
        xL = xL + dxL
        Force = F*F*(q2+1)*xL-1.0/xL/xL-q2*(xL-D)/abs((D-xL)**3)-q2/D/D
        dxLdF  = 1.0/(F*F*(q2+1) + 2.0/xL/xL/xL + 2*q2/abs((D-xL)**3))
        dxL = -Force * dxLdF

    if (q > 1.0):
        xL = D - xL
    xL2 = xL*sma
    Phi_L2 = 1.0/abs(xL2) + q*(1.0/abs(xL2-1)-xL2) + 1./2.*(q+1)*xL2*xL2
    return Phi_L1, Phi_L2

def make_overcontact(system,FF,delta):
    
    ''' Creates the overcontact envelope of the system at given FF '''
    
    #some parameters needed
    orbit = system.get_orbit()[0]
    ref = system.params['syn']['lcsyn'].items()[0][0]
    
    #setting the potential of the oc from ff
    L1, L2 = exact_critical_pots(orbit['q'],sma=1., d=1.,F=1.)
    pot_oc = FF*(L2-L1) + L1
    
    #construct the parametersets and the overcontact
    overc = phoebe.PS('component', morphology = 'overcontact', pot = pot_oc, 
atm='blackbody', ld_func='linear', ld_coeffs=[0.5])
    mesh_oc = phoebe.PS('mesh:marching', delta = delta, maxpoints = 
1000000000)
    lcdep1 = phoebe.PS('lcdep', 
atm='blackbody',ld_func='linear',ld_coeffs=[0.5], passband = 
system.get_bodies()[0].params['pbdep']['lcdep'][ref]['passband'])
    orbit_overc = orbit.copy()
    orbit_overc['c1label'] = overc['label']
    pos = phoebe.ParameterSet('position')
    c3 = phoebe.BinaryRocheStar(overc, mesh=mesh_oc, orbit=orbit_overc, 
pbdep=[lcdep1], position = pos)
    c3.set_time(orbit['t0'])
    return c3


def approx_stars(system, delta, zL1):
    
    ''' Fits ellipsoids to the two detached stars in the system '''
    
    system.set_time(system.get_orbit()[0]['t0'])
    c1, c2 = system.get_bodies()[0], system.get_bodies()[0]
    orbit = c1.get_orbit()
    sma1, sma2 = orbit['sma']/(1+1/orbit['q']), orbit['sma']/(1+orbit['q'])
    
    x_sph=system.mesh['center'][np.abs(system.mesh['center'][:,1])<=delta/1.5]
    y_sph=system.mesh['center'][np.abs(system.mesh['center'][:,0])<=delta/1.5]
    
    x_sph1, y_sph1 = x_sph[x_sph[:,2] < zL1], y_sph[y_sph[:,2] < zL1]
    x_sph2, y_sph2 = x_sph[x_sph[:,2] > zL1], y_sph[y_sph[:,2] > zL1]
    
    def star1(z,a,c):
        return a**2*(1 - ((z+sma1)**2/c**2))
    
    def star2(z,b,c):
        return b**2*(1 - ((z-sma2)**2/c**2))
    
    s1x, s1pcovx = curve_fit(star1, x_sph1[:,2], (x_sph1[:,0])**2)
    s1y, s1pcovy = curve_fit(star1, y_sph1[:,2], (y_sph1[:,1])**2)
    
    s2x, s2pcovx = curve_fit(star2, x_sph2[:,2], (x_sph2[:,0])**2)
    s2y, s2pcovy = curve_fit(star2, y_sph2[:,2], (y_sph2[:,1])**2)
    
    as1, bs1 = round(s1x[0],2), round(s1y[0],2)
    cs1 = round((s1x[1]+s1y[1])/2.,2)
    as2, bs2 = round(s2x[0],2), round(s2y[0],2) 
    cs2 = round((s2x[1]+s2y[1])/2.,2)
    
    
    return [as1, bs1, cs1], [as2, bs2, cs2]
   

def approx_hypbol(c3,zL1,delta):
    
    x_section=c3.mesh['center'][np.abs(c3.mesh['center'][:,1])<=delta]
    y_section=c3.mesh['center'][np.abs(c3.mesh['center'][:,0])<=delta]
    
    # divide the sections into ones with positive (p) and negative (m) values 
    # of x and y
    
    xp = x_section[:,0][x_section[:,0]>=0.]
    z_xp = x_section[:,2][x_section[:,0]>=0.]
    xm = x_section[:,0][x_section[:,0]<=0.] 
    z_xm = x_section[:,2][x_section[:,0]<=0.]
    yp = y_section[:,1][y_section[:,1]>=0.]
    z_yp = y_section[:,2][y_section[:,1]>=0.]
    ym = y_section[:,1][y_section[:,1]<=0.] 
    z_ym = y_section[:,2][y_section[:,1]<=0.]
    
    #second derivatives go here
    dxp, dxm, dyp, dym = [], [], [], []
    #dxp - concave are positive, dxm - concave are negative
    #dyp - same
    
    for i in range(1,len(xp)-1):
        dxp.append([(xp[i-1]+xp[i+1])/2,(z_xp[i-1]+z_xp[i+1])/2.,((xp
[i+1]-xp[i])/(z_xp[ i+1]-z_xp[i]) - (xp[i] - xp[i-1])/(z_xp[i] - 
z_xp[i-1]))/((z_xp[i+1]-z_xp[i-1])/2.)])
    
    for i in range(1,len(xm)-1):
        dxm.append([(xm[i-1]+xm[i+1])/2,(z_xm[i-1]+z_xm[i+1])/2.,((xm
[i+1]-xm[i])/(z_xm[ i+1]-z_xm[i]) - (xm[i] - xm[i-1])/(z_xm[i] - 
z_xm[i-1]))/((z_xm[i+1]-z_xm[i-1])/2.)])
    
    for i in range(1,len(yp)-1):
        dyp.append([(yp[i-1]+yp[i+1])/2,(z_yp[i-1]+z_yp[i+1])/2.,((yp
[i+1]-yp[i])/(z_yp[ i+1]-z_yp[i]) - (yp[i] - yp[i-1])/(z_yp[i] - 
z_yp[i-1]))/((z_yp[i+1]-z_yp[i-1])/2.)])
    
    for i in range(1,len(ym)-1):
        dym.append([(ym[i-1]+ym[i+1])/2,(z_ym[i-1]+z_ym[i+1])/2.,((ym
[i+1]-ym[i])/(z_ym[ i+1]-z_ym[i]) - (ym[i] - ym[i-1])/(z_ym[i] - 
z_ym[i-1]))/((z_ym[i+1]-z_ym[i-1])/2.)])
    
    #define the hyperboloid region
    
    dxp, dxm = np.array(dxp), np.array(dxm)
    dyp, dym = np.array(dyp), np.array(dym)
    
    zmin_a = np.array([dxp[:,1][dxp[:,2] >= 0.].min(),dxm[:,1][dxm[:,2] <= 
0.].min(),dyp[:,1][dyp[:,2] >= 0.].min(),dym[:,1][dym[:,2] <= 0.].min()])
    zmax_a = np.array([dxp[:,1][dxp[:,2] >= 0.].max(),dxm[:,1][dxm[:,2] <= 
0.].max(),dyp[:,1][dyp[:,2] >= 0.].max(),dym[:,1][dym[:,2] <= 0.].max()])
    
    zmin = round(np.abs(zmin_a.max()),2)
    zmax = round(np.abs(zmax_a.min()),2)
    
    x_hb1 = x_section[(x_section[:,2] <= zL1) & (x_section[:,2] >= -zmin)]
    x_hb2 = x_section[(x_section[:,2] >= zL1) & (x_section[:,2] <= zmax)]
    y_hb1 = y_section[(y_section[:,2] <= zL1) & (y_section[:,2] >= -zmin)]
    y_hb2 = y_section[(y_section[:,2] >= zL1) & (y_section[:,2] <= zmax)]
    
    #fit hyperbolae
    
    def hyp(z,a,c):
        return a**2*(1.+ (z-zL1)**2/c**2)
    
    hypx1, hypcovx1 = curve_fit(hyp, x_hb1[:,2], (x_hb1[:,0])**2)
    hypy1, hypcovy1 = curve_fit(hyp, y_hb1[:,2], (y_hb1[:,1])**2)
    
    hypx2, hypcovx2 = curve_fit(hyp, x_hb2[:,2], (x_hb2[:,0])**2)
    hypy2, hypcovy2 = curve_fit(hyp, y_hb2[:,2], (y_hb2[:,1])**2)
    
    ah1, bh1 = round(hypx1[0],2), round(hypy1[0],2) 
    ch1 = round((hypx1[1]+hypy1[1])/2.,2)
    ah2, bh2 = round(hypx2[0],2), round(hypy2[0],2)
    ch2 = round((hypx2[1]+hypy2[1])/2.,2)
    
    return [ah1, bh1, ch1, zmin], [ah2, bh2, ch2, zmax]
    

def hyp_intersect(pov, poi, hyp, zL1):
    
    ''' Takes the coordinates of POV (point of view on the oc mesh) and POI 
(point of interest on a component) and constructs line, then checks for 
intersections of the line with the hyperboloid.
    
   The direction vector points from the POI to POV, so that we can deal 
with values near the hyperboloid better.'''
    
    a, b, c = pov[0]-poi[0], pov[1]-poi[1], pov[2]-poi[2]
    x0, y0, z0 = poi[0], poi[1], poi[2]
    ah, bh, ch = hyp[0], hyp[1], hyp[2]
    
    ad = (a/ah)**2 + (b/bh)**2 - (c/ch)**2
    bd = 2*(x0*a/ah**2 + y0*b/bh**2 - (z0-zL1)*c/ch**2)
    cd = (x0/ah)**2 + (y0/bh)**2 - ((z0-zL1)/ch)**2 - 1
    
    D = bd**2 - 4*ad*cd
    
    threshold = 0.1
    if D < 0.:
        return True
    elif D == 0.:
        t = -bd/(2*ad)
        if (t > threshold) & (t < 1. - threshold):
            return False
        else:
            return True
    else:
        t1 = (-bd + np.sqrt(D))/(2*ad)
        t2 = (-bd - np.sqrt(D))/(2*ad)
        
        if ((t1 > threshold) & (t1 < 1. - threshold)) or ((t2 > 
threshold) & (t2 < 1.-threshold)):
            return False
        else:
            return True

def star_intersect(pov,poi,star1,star2,sma1,sma2,zL1):
    
    ''' Takes the coordinates of POV (point of view on the oc mesh) and POI 
(point of interest on a component) and constructs line, then checks for 
intersections of the line with the stars. '''
    
    a, b, c = poi[0]-pov[0], poi[1]-pov[1], poi[2]-pov[2]
    x0, y0, z0 = pov[0], pov[1], pov[2]
    a1, b1, c1 = star1[0], star1[1], star1[2]
    a2, b2, c2 = star2[0], star2[1], star2[2]
    
    ad1 = (a/a1)**2 + (b/b1)**2 + (c/c1)**2 
    ad2 = (a/a2)**2 + (b/b2)**2 + (c/c2)**2
    bd1 = 2*(x0*a/a1**2 + y0*b/b1**2 + (z0+sma1)*c/c1**2)
    bd2 = 2*(x0*a/a2**2 + y0*b/b2**2 + (z0-sma2)*c/c2**2)
    cd1 = (x0/a1)**2 + (y0/b1)**2 + ((z0+sma1)/c1)**2 - 1 
    cd2 = (x0/a2)**2 + (y0/b2)**2 + ((z0-sma2)/c2)**2 - 1
    
    D1, D2 = bd1**2 - 4*ad1*cd1, bd2**2 - 4*ad2*cd2
    
    threshold = 0.3
    
    if D1 < 0.:
        vis1 = True
    elif D1 == 0.:
        t = -bd1/(2*ad1)
        if (t > threshold) & (t < 1. - threshold):
            vis1 = False
        else:
            vis1 = True
    else:
        t1 = (-bd1 + np.sqrt(D1))/(2*ad1)
        t2 = (-bd1 - np.sqrt(D1))/(2*ad1)
        
        if ((t1 > threshold) & (t1 < 1.- threshold)) or ((t2 > 
threshold) & (t2 < 1. - threshold)):
            vis1 = False
        else:
            vis1 = True
    
    if D2 < 0.:
        vis2 = True
    elif D2 == 0.:
        t = -bd2/(2*ad2)
        if (t > threshold) & (t < 1.-threshold):
            vis2 = False
        else:
            vis2 = True
    else:
        t1 = (-bd2 + np.sqrt(D2))/(2*ad2)
        t2 = (-bd2 - np.sqrt(D2))/(2*ad2)
        
        if ((t1 > threshold) & (t1 < 1.-threshold)) or ((t2 > threshold) 
& (t2 < 1.-threshold)):
            vis2 = False
        else:
            vis2 = True
    
    return vis1 & vis2


def compute_teffs(system,c3,FF,delta,beta):
    
    ''' Computes the temperature distribution on c3 using analytical 
approximations of the surfaces to determine visibility of the stars in the 
system '''

    orbit = c3.get_orbit()
    
    # fix the coordinate system by setting the time to hj0
    system.set_time(orbit['t0'])
    c3.set_time(orbit['t0'])
    
    system.reset()
    system.clear_synthetic()
    system.compute()
    
    c1, c2 = system.get_bodies()[0], system.get_bodies()[1]
    pot1, pot2 = c1.get_parameters()['pot'], c2.get_parameters()['pot']
    sma1, sma2 = orbit['sma']/(1+1/orbit['q']), orbit['sma']/(1+orbit['q'])
    d = system.params['position']['distance']*convert('pc','Rsol',1.)
    L1, L2, L3 = exact_lagrangian_points(q = orbit['q'], sma = orbit['sma'])
    zL1 = L1 - sma1
    
    #prepare the mesh for approxmation of hyperboloid
    delta_geom = 0.02
    c3_geom = make_overcontact(system,FF,delta = delta_geom)
    
    # parameters of the hyperboloid and ellipsoid stars
    hyp1, hyp2 = approx_hypbol(c3_geom,zL1,delta_geom)
    star1, star2 = approx_stars(system, delta, zL1)
    
    #begin the detection
    
    for i in range(0,c3.mesh.size):

        pov = c3.mesh['center'][i]
        
        # compute the lines of sight to the separate components
        
        los1 = (pov - np.array([0.,0., 
-sma1]))*1./((pov[0])**2+(pov[1])**2+(pov[2]+sma1)**2)
        los2 = (pov - 
np.array([0.,0.,sma2]))*1./((pov[0])**2+(pov[1])**2+(pov[2]-sma2)**2)

        #set all triangles of c1 and c2 as hidden
        
        c1.mesh['visible'], c1.mesh['hidden'] = False, True 
        c1.mesh['partial'] = False
        c2.mesh['visible'], c2.mesh['hidden'] = False, True
        c2.mesh['partial'] = False
        n1, center1 = c1.mesh['normal_'], c1.mesh['center']
        n2, center2 = c2.mesh['normal_'], c2.mesh['center']
        
        # compute mus and mark the visible triangles as those with mu > 0. 
        c1.mesh['mu'] = np.dot(n1,los1.T)
        c2.mesh['mu'] = np.dot(n2,los2.T)
        c1.mesh['visible'][c1.mesh['mu'] > 0.] = True
        c2.mesh['visible'][c2.mesh['mu'] > 0.] = True
        c1.mesh['hidden'] = -c1.mesh['visible'] 
        c2.mesh['hidden'] = -c2.mesh['visible']
        
        # take only the visible POIs to check for intersections
        
        poi1 = c1.mesh['center'][c1.mesh['visible']]
        poi2 = c2.mesh['center'][c2.mesh['visible']]
        visible1, visible2 = [], []
                
        for j in range(0,len(poi1)):
            visibleh = hyp_intersect(pov, poi1[j],hyp2,zL1)
            visibles = star_intersect(pov,poi1[j],star1,star2,sma1,sma2,zL1)
            visible1.append(visibleh & visibles)
        
        for j in range(0,len(poi2)):
            visibleh = hyp_intersect(pov, poi2[j],hyp2,zL1)
            visibles = star_intersect(pov,poi2[j],star1,star2,sma1,sma2,zL1)
            visible2.append(visibleh & visibles)
            
        c1.mesh['visible'][c1.mesh['visible']] = visible1
        c2.mesh['visible'][c2.mesh['visible']] = visible2
        c1.mesh['hidden'] = -c1.mesh['visible']
        c2.mesh['hidden'] = -c2.mesh['visible']

        
        #compute the flux received from the visible triangles
        
        lc = system.projected_intensity()
        r1 = np.sqrt(pov[0]**2 + pov[1]**2 + (pov[2] + sma1)**2)
        r2 = np.sqrt(pov[0]**2 + pov[1]**2 + (pov[2] - sma2)**2)

        flux=lc[0]*(d/r1)**(2+beta)+lc[1]*(d/r2)**(2+beta)
        teff_c = (flux/sigma)**(1./4.)
        c3.mesh['teff'][i] = teff_c 
    
    c3.mesh['ld___bol'] = 0.
    c3.mesh['proj___bol'] = 0.
    c3.intensity()
    
def smooth_teffs(c3,sigma_g,mode = 'constant', cval = 500.):

    ''' Smooths the temperature distribution using a gaussian filter '''
    
    # sort the temperature array for "smoother" smoothing
    
    teffs = np.array(c3.mesh['teff'])
    teffs_en = np.array(list(enumerate(teffs)))
    ind=np.lexsort((teffs_en[:,0],teffs_en[:,1]))
    teffs_en_s = teffs_en[ind]

    teffs_g = gaussian_filter1d(teffs_en_s[:,1], sigma = sigma_g, order=0, 
mode='constant', cval = teffs[0])

    teffs_en_f = np.array([teffs_en_s[:,0],teffs_g]).T
    ind1 = np.lexsort((teffs_en_f[:,1],teffs_en_f[:,0]))
    teffs_filtered = teffs_en_f[ind1]
    c3.mesh['teff'] = teffs_filtered[:,1]
   
def compute_lc(c3,times):
    
    flux = []
    
    for t in times:
        c3.set_time(t)
        c3.intensity()
        c3.detect_eclipse_horizon(eclipse_detection='hierarchical')
        flux.append(c3.projected_intensity())
        
    return times, flux

def oc_lc(system,FF=0.5,sigma=1.,beta=0.,delta=0.1,smooth_mode='reflect'):
    
    ''' The function where everything comes together. The overcontact is 
created, its temperature distribution computed, synthetic light curve created 
and assigned to the system. '''
    
    # I have no idea why I'm taking these in this way, I guess I tried and it
    # worked for what I needed
    
    ref = system.params['syn']['lcsyn'].items()[0][0]
    c1 = system.get_bodies()[0]
    lcsyn = c1.params['syn']['lcsyn'][ref]
    
    # must set the inclination of the system to 90. before computing the
    # temperature distribution to make sure the coordinate system is the one
    # that is mathematically defined in the model
    
    incl_old = system.get_orbit()[0]['incl']
    system.get_orbit()[0]['incl'] = 90.
    system.get_orbit()[1]['incl'] = 90.
    
    # set the passband to bolometric to ensure actual Teff computation
    
    passband=system.get_bodies()[0].params['pbdep']['lcdep'][ref]['passband']
    passband=str(passband)
    system.get_bodies()[0].params['pbdep']['lcdep'][ref]['passband']='OPEN.BOL'
    system.get_bodies()[1].params['pbdep']['lcdep'][ref]['passband']='OPEN.BOL'
    
    # make the overcontact and compute the temperature distribution
    c3 = make_overcontact(system, FF, delta)
    compute_teffs(system,c3,FF,delta,beta)
    smooth_teffs(c3,sigma_g=sigma,mode=smooth_mode)
    
    # return to original inclination and passband
    system.get_orbit()[0]['incl'] = incl_old
    system.get_orbit()[1]['incl'] = incl_old
    c3.params['orbit']['incl'] = incl_old
   
    system.get_bodies()[0].params['pbdep']['lcdep'][ref]['passband']=passband
    system.get_bodies()[1].params['pbdep']['lcdep'][ref]['passband']=passband
    
    # compute the light curve
    times = lcsyn['time']   
    times_oc, flux_oc = compute_lc(c3,times)
    times_oc = np.array(times_oc)
    flux_oc = np.array(flux_oc)
    
    # assign the times and flux where needed in the system
    
    system.params['syn']['lcsyn'][ref]['time'] = list(times_oc)
    system.params['syn']['lcsyn'][ref]['flux'] = list(flux_oc)
    system.get_bodies()[0].params['syn']['lcsyn'][ref]['time']=list(times_oc)
    system.get_bodies()[0].params['syn']['lcsyn'][ref]['flux']=list(flux_oc/2.) 
    system.get_bodies()[1].params['syn']['lcsyn'][ref]['time']=list(times_oc)
    system.get_bodies()[1].params['syn']['lcsyn'][ref]['flux']=list(flux_oc/2.)

    return times_oc, flux_oc, c3

def oc_lc_bundle(eb,FF=0.5,sigma=1.,beta=0.,delta=0.1,smooth_mode='reflect'):
    
    name = eb['label@orbit']
    ref = eb['ref@lcsyn@'+name]
    incl_old = float(eb['orbit']['incl'])
    eb['orbit']['incl'] = 90.
    passband = str(eb['passband@lcdep@primary'])
    eb['passband@lcdep@primary'] = 'OPEN.BOL'
    eb['passband@lcdep@secondary'] = 'OPEN.BOL'
    system = eb[name]
    c3 = make_overcontact(system, FF, delta)
    compute_teffs(system,c3,FF,delta,beta)
    c3.params['orbit']['incl'] = incl_old
    eb['orbit']['incl'] = incl_old
    eb['passband@lcdep@primary'] = passband
    eb['passband@lcdep@secondary'] = passband
    times_oc = eb['time@lcobs']
    flux_oc = []
    
    for t in times_oc:
        c3.set_time(t)
        c3.intensity()
        c3.detect_eclipse_horizon(eclipse_detection='hierarchical')
        flux_oc.append(c3.projected_intensity())
   
    times_oc = np.array(times_oc) 
    flux_oc = np.array(flux_oc)
    eb['lcsyn@primary'][ref]['time'] = list(times_oc)
    eb['lcsyn@primary'][ref]['flux'] = list(flux_oc/2.)
    eb['lcsyn@secondary'][ref]['time'] = list(times_oc)
    eb['lcsyn@secondary'][ref]['flux'] = list(flux_oc/2.)
    eb['lcsyn@'+name][ref]['time'] = list(times_oc)
    eb['lcsyn@'+name][ref]['flux'] = list(flux_oc)
    
    return times_oc, flux_oc, c3