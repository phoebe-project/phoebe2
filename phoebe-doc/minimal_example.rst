:download:`Download this page as a python script <../phoebe-doc/scripts/minimal_example.py>`


::

    from phoebe import create
    from matplotlib import pyplot as plt
    import phoebe
    
    logger = phoebe.get_basic_logger()
    
    Astar = create.star_from_spectral_type('A0V')
    Bstar = create.star_from_spectral_type('B9V')
    system = create.binary_from_stars(Astar,Bstar,sma=(0.045,'au'),create_body=True)
    
    system.set_time(2454545.39)
    system.plot2D()
    plt.show()
    
    system.lc()
    system.rv()
    system.sp()
    
    mylc = system.get_synthetic('lc',cumulative=True)
    myrv = system.get_synthetic('rv',cumulative=True)
    myspec = system.get_synthetic('sp',cumulative=False)
    
    times,signal = mylc['time'],mylc['flux']
        
    P = 1.09147510362
    for time in range(1,11):
        system.set_time(2454545.39 + time/50.*P)
        system.lc()
        system.sp()
    
    mylc = system.get_synthetic('lc',cumulative=True)
    myspec = system.get_synthetic('sp',cumulative=False)
    
    
    plt.figure()
    plt.subplot(121)
    plt.plot(mylc['time'],mylc['flux'],'ko-')
    plt.subplot(122)
    for i in range(len(mylc['time'])):
        plt.plot(myspec[0]['wavelength'][i],myspec[0]['flux'][i]/myspec[0]['continuum'][i]+i/30.,'k-')
        plt.plot(myspec[1]['wavelength'][i],myspec[1]['flux'][i]/myspec[1]['continuum'][i]+i/30.,'r-')
    plt.show()
    
    
    import phoebe
    
    Bstar = create.star_from_spectral_type('B9V')
    
    
    
    
    print(Bstar)
    
    
    
    Bstar['radius'] = 3.6
    Bstar['radius'] = 3.6,'Rsol'
    Bstar['radius'] = 2503828.8,'km'
    
    
    
    comp1,comp2,orbit = create.binary_from_stars(Bstar,Astar,sma=(0.045,'au'))
    
    
    mesh = phoebe.ParameterSet(context='mesh:marching')
    lcdep = phoebe.ParameterSet(context='lcdep')
    
    
    Bstar = phoebe.BinaryRocheStar(comp1,orbit=orbit,mesh=mesh,pbdep=[lcdep])
    
    
    Astar = phoebe.BinaryRocheStar(comp2,orbit,mesh,pbdep=[lcdep])
    
    
    
    
    system = phoebe.BodyBag([Bstar,Astar])
    
    
    
    
    
    system.set_time(2454545.29)
    system.plot2D()
    plt.show()
    
    
    
    
    Bstar.plot2D()
    Astar.plot2D()
    plt.show()
    
    
    angle = -3.1415/180.*30.
    system.rotate_and_translate(theta=angle,incl=angle,incremental=True)
    Bstar.plot2D()
    Astar.plot2D()
    system.plot2D()
    print "I'm here"
    
    plt.show()
    
    
    
    system.reset()
    system.set_time(2454545.29)
    
    
    
    system.detect_eclipse_horizon(eclipse_detection='hierarchical')
    
    Bstar.plot2D()
    Astar.plot2D()
    print "I'm there"
    
    plt.show()
    
    
    for i in range(3):
        system.subdivide()
        system.detect_eclipse_horizon(eclipse_detection='hierarchical')
        Bstar.plot2D()
    print "I'm everywhere"
    
    plt.show()
    
    system.reset()  # reset the time
    
    period = system[0].params['orbit']['period']
    for time in range(50):
        system.set_time(2454545.39+time/50.*period)
        system.detect_eclipse_horizon(eclipse_detection='hierarchical')
        for i in range(3):
            system.subdivide()
            system.detect_eclipse_horizon(eclipse_detection='hierarchical')
        system.lc()
        system.unsubdivide()
    
    
    
    
    mylc = system.get_synthetic('lc',cumulative=True)
    
    plt.plot(mylc['time'],mylc['flux'],'ko-')
    plt.show()
