import phoebe
import numpy as np
import matplotlib.pyplot as plt

#logger = phoebe.get_basic_logger()

def do_ldlaw(atm='kurucz', ld_func='linear', ld_coeffs=None):
    
    if ld_coeffs is None:
        ld_coeffs = atm
    star = phoebe.PS('star', atm=atm, ld_coeffs=ld_coeffs, ld_func=ld_func, shape='sphere')
    mesh = phoebe.PS('mesh:marching', delta=0.07)
    lcdep1 = phoebe.PS('lcdep', passband='OPEN.BOL', atm=atm, boosting=False,
                    ld_coeffs=ld_coeffs, ld_func=ld_func, ref='bol')
    lcdep2 = phoebe.PS('lcdep', passband='JOHNSON.V', atm=atm, boosting=False,
                    ld_coeffs=ld_coeffs, ld_func=ld_func, ref='visual')
    
    globals = phoebe.ParameterSet(context='position')
    globals['distance'] = 1.,'au'
    
    sun = phoebe.Star(star, mesh=mesh, pbdep=[lcdep1, lcdep2], position=globals)
    
    sun.set_time(0.)
    
    #lum = phoebe.convert('cgs',"Lsol",phoebe.universe.luminosity(sun))
    lum = phoebe.convert('SI',"Lsol",phoebe.universe.luminosity(sun, ref='__bol'))
    proj1 = sun.projected_intensity(ref='bol')
    proj2 = phoebe.convert('W/m3','mag',sun.projected_intensity(ref='visual'),passband='JOHNSON.V')
    #proj1 = phoebe.convert('cgs','W/m2',sun.projected_intensity(ref='bol'))
    #proj2 = phoebe.convert('erg/s/cm2/AA','mag',sun.projected_intensity(ref='visual'),passband='JOHNSON.V')
    print atm, ld_func, ld_coeffs, lum, proj1, proj2
    assert(np.abs(lum-1)<0.01)
    assert(np.abs(proj1-1360)<10.)
    assert(np.abs(proj2+26.72)<0.05)
    


def test_ldlaw():
    """
    Atmospheres: LD laws, coefficients and intensities
    """
    do_ldlaw(atm='kurucz',ld_func='linear')
    do_ldlaw(atm='kurucz',ld_func='linear', ld_coeffs=[0.02])
    do_ldlaw(atm='kurucz',ld_func='linear', ld_coeffs=[0.50])
    do_ldlaw(atm='kurucz',ld_func='linear', ld_coeffs=[0.98])
    do_ldlaw(atm='kurucz',ld_func='logarithmic')
    do_ldlaw(atm='kurucz',ld_func='logarithmic', ld_coeffs=[0.5,0.5])
    do_ldlaw(atm='kurucz',ld_func='logarithmic', ld_coeffs=[0.1,0.9])
    do_ldlaw(atm='kurucz',ld_func='logarithmic', ld_coeffs=[0.9,0.1])
    do_ldlaw(atm='kurucz',ld_func='claret')
    do_ldlaw(atm='kurucz',ld_func='claret')
    do_ldlaw(atm='blackbody',ld_func='uniform')
    do_ldlaw(atm='blackbody',ld_func='linear')
    do_ldlaw(atm='blackbody',ld_func='linear', ld_coeffs=[0.02])
    do_ldlaw(atm='blackbody',ld_func='linear', ld_coeffs=[0.50])
    do_ldlaw(atm='blackbody',ld_func='linear', ld_coeffs=[0.98])

if __name__=="__main__":
    test_ldlaw()
    
