import phoebe
import matplotlib.pyplot as plt
import numpy as np
import time




atm = 'blackbody'
alb = 1-0.932

def do_mercury(redist):
    sun, position = phoebe.create.from_library('sun')
    sun['irradiator'] = True


    mercury = phoebe.PS('star', teff=100., atm=atm, radius=(0.3829,'Rearth'),
                    mass=(0.055,'Mearth'), ld_func='uniform', shape='sphere',
                    rotperiod=(58.646,'d'),alb=alb, label='mercury', redist=redist,
                    incl=90., long=0, irradiator=False)

    orbit = phoebe.PS('orbit', period=87.9691, t0=0, t0type='superior conjunction',
                    sma=(0.307499,'au'), q=mercury['mass']/sun['mass'], ecc=0.205630,
                    c1label=sun['label'], c2label=mercury['label'],incl=86.62,
                    per0=(29.124,'deg'))

    lcdep1 = phoebe.PS('lcdep', atm='kurucz', ld_coeffs='kurucz', ld_func='claret', ref='apparent')
    lcdep2 = phoebe.PS('lcdep', atm=atm, ld_func='uniform', ref='apparent', alb=alb)
    obs = phoebe.LCDataSet(time=np.array([orbit['period']*0.25]), columns=['time'], ref=lcdep1['ref'])

    mesh1 = phoebe.PS('mesh:marching', delta=0.1)
    mesh2 = phoebe.PS('mesh:marching', delta=0.04, maxpoints=22000)

    globals = phoebe.PS('position', distance=(1,'au')) 
    
    sun = phoebe.BinaryStar(sun, mesh=mesh1, orbit=orbit, pbdep=[lcdep1])
    mercury = phoebe.BinaryStar(mercury, mesh=mesh2, orbit=orbit, pbdep=[lcdep2])

    system = phoebe.BodyBag([sun, mercury], obs=[obs], position=globals)
    system.compute(heating=True, refl=True, refl_num=1, boosting_alg='none')
    return system



def test_no_redist():
    """
    Heating and reflection: Mercury with no heat redistribution
    """
    system = do_mercury(redist=0.0)
    
    proj = system[1].projected_intensity(ref='apparent')
    teffmean = system[1].mesh['teff'].mean()
    teffmin = system[1].mesh['teff'].min()
    teffmax = system[1].mesh['teff'].max()
    
    print('proj',proj)
    print('mean',teffmean)
    print('min',teffmin)
    print('max',teffmax)
    
    assert(np.abs(proj - 6.74991749527e-12)/6.74991749527e-12<1e-2)
    assert(np.abs(teffmean-305)<5)
    assert(np.abs(teffmin-100)<0.1)
    assert(np.abs(teffmax-640)<5)


def test_little_redist():   
    """
    Heating and reflection: Mercury with little heat redistribution
    """
    system = do_mercury(redist=0.2)
    
    proj = system[1].projected_intensity(ref='apparent')
    teffmean = system[1].mesh['teff'].mean()
    teffmin = system[1].mesh['teff'].min()
    teffmax = system[1].mesh['teff'].max()
    
    print(proj)
    print(teffmean)
    print(teffmin)
    print(teffmax)
    
    assert(np.abs(proj - 6.74991749527e-12)/6.74991749527e-12<1e-2)
    assert(np.abs(teffmean - 405)<5)
    assert(np.abs(teffmin - 303)<5)
    assert(np.abs(teffmax - 615)<5)

def test_total_redist():
    """
    Heating and reflection: Mercury with complete heat redistribution
    """
    system = do_mercury(redist=1.0)
    
    proj = system[1].projected_intensity(ref='apparent')
    teffmean = system[1].mesh['teff'].mean()
    teffmin = system[1].mesh['teff'].min()
    teffmax = system[1].mesh['teff'].max()
    
    print("projected intensity",proj)
    print('mean teff',teffmean)
    print('min teff', teffmin)
    print('max teff', teffmax)
    
    assert(np.abs(system[1].projected_intensity(ref='apparent') - 6.74991749527e-12)/6.74991749527e-12<1e-2)
    assert(np.abs(system[1].mesh['teff'].mean()-453)<5)
    assert(np.abs(system[1].mesh['teff'].min()-453)<5)
    assert(np.abs(system[1].mesh['teff'].max()-453)<5)
    
if __name__=="__main__":
    
    logger = phoebe.get_basic_logger(clevel='INFO')
    test_no_redist()
    
    test_little_redist()
    test_total_redist()
