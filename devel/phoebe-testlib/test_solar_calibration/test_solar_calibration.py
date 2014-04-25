import numpy as np
import phoebe
from phoebe.atmospheres import limbdark
from phoebe.parameters import tools

def main():
    """
    Absolute fluxes: solar calibration
    """
    sun = phoebe.ParameterSet(context='star',add_constraints=True)
    sun['shape'] = 'sphere'
    sun['atm'] = 'kurucz'
    sun['ld_coeffs'] = 'kurucz'
    sun['ld_func'] = 'claret'

    globals = phoebe.ParameterSet(context='position')
    globals['distance'] = 1.,'au'

    sun_mesh = phoebe.ParameterSet(context='mesh:marching',alg='c')
    sun_mesh['delta'] = 0.05

    lcdep1 = phoebe.ParameterSet(frame='phoebe',context='lcdep')
    lcdep1['ld_func'] = 'claret'
    lcdep1['ld_coeffs'] = 'kurucz'
    lcdep1['atm'] = 'kurucz'
    lcdep1['passband'] = 'OPEN.BOL'
    lcdep1['ref'] = 'Bolometric (numerical)'

    lcdep2 = lcdep1.copy()
    lcdep2['method'] = 'analytical'
    lcdep2['ref'] = 'Bolometric (analytical)'

    the_sun = phoebe.Star(sun,sun_mesh,pbdep=[lcdep1,lcdep2], position=globals)

    the_sun.set_time(0)

    the_sun.lc()
    
    params = the_sun.get_parameters()
    nflux = the_sun.params['syn']['lcsyn']['Bolometric (numerical)']['flux'][0]
    aflux = the_sun.params['syn']['lcsyn']['Bolometric (analytical)']['flux'][0]
    
    mupos = the_sun.mesh['mu']>0
    
    num_error_area = np.abs(np.pi-((the_sun.mesh['size']*the_sun.mesh['mu'])[mupos]).sum())/np.pi*100
    num_error_flux = np.abs(nflux-aflux)/aflux*100
    print aflux, nflux, the_sun.projected_intensity()
    real_error_flux = np.abs(1368.000-aflux)/aflux*100
    
    assert(num_error_area<=0.048)
    assert(num_error_flux<=0.049)
    assert(real_error_flux<=0.25)
    
    lumi1 = limbdark.sphere_intensity(the_sun.params['star'],the_sun.params['pbdep']['lcdep'].values()[0])[0]
    lumi2 = params.get_value('luminosity','W')
    lumsn = phoebe.constants.Lsol#_cgs
    num_error_area = np.abs(4*np.pi-the_sun.area())/4*np.pi*100
    num_error_flux = np.abs(lumi1-lumi2)/lumi1*100
    real_error_flux = np.abs(lumi1-lumsn)/lumsn*100

    assert(num_error_area<=0.48)
    assert(num_error_flux<=0.040)
    assert(real_error_flux<=0.22)
    
def as_test():
    """
    Absolute fluxes: solar calibration
    """
    main()
    
if __name__ == "__main__":
    main()
