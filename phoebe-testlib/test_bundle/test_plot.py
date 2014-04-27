import phoebe
import numpy as np
import matplotlib.pyplot as plt


def test_plot(N=3):
    """
    Testing Bundle's plotting capabilities
    """
    # System set up
    eb = phoebe.Bundle()
    eb.set_value_all('delta',0.05)
    eb['label@new_system'] = 'awesomeness'
    eb['vgamma@position'] = (25, 'km/s')
    eb['sma'] = 4.0
    eb['q'] = 0.5
    eb['period'] = 12.34

    # Adding really fake observations
    phase = np.linspace(-0.6, 0.6, N)
    sigma = 0.1*np.ones(N)
    np.random.seed(1111)
    eb.lc_fromarrays(phase=phase, flux=np.sort(2*np.random.normal(size=N)+3885))#, sigma=sigma)
    eb.rv_fromarrays('primary',  phase=phase, rv=-5*np.sin(2*np.pi*phase)+25, sigma=sigma)
    eb.rv_fromarrays('secondary',  phase=phase, rv=10*np.sin(2*np.pi*phase)+25, sigma=sigma)

    # Computing
    eb.run_compute('detailed')

    #And plotting
    plt.figure()

    plt.subplot(222)
    out = eb.plot_obs('lc01')

    plt.subplot(221)
    eb.plot_obs('rv01')
    eb.plot_obs('rv02')

    plt.subplot(223)
    eb.plot_syn('rv01')
    eb.plot_syn('rv02')

    plt.subplot(224)
    eb.plot_syn('lc01')


    plt.figure()

    obs1 = eb.plot_obs('lc01', fmt='ko', phased=True, repeat=1, y_unit='mag')
    syn1 = eb.plot_syn('lc01', 'r-', lw=2, phased=True, repeat=1, y_unit='ppm')
    plt.legend(loc='best').get_frame().set_alpha(0.5)

    plt.figure()

    plt.subplot(121)
    obs2 = eb.plot_obs('lc01', fmt='ko', phased=False, repeat=1, y_unit='mag')
    syn2 = eb.plot_syn('lc01', 'r-', lw=2, phased=False, repeat=1, y_unit='ppm')
    plt.legend(loc='best').get_frame().set_alpha(0.5)

    plt.subplot(122)
    obs3 = eb.plot_obs('rv01', fmt='ro', x_unit='rad', y_unit='nRsol/d')
    obs4 = eb.plot_obs('rv02', fmt='bo', x_unit='rad', y_unit='nRsol/d')
    syn3 = eb.plot_syn('rv01', 'r-', lw=2, x_unit='rad', y_unit='nRsol/d')
    syn4 = eb.plot_syn('rv02', 'b-', lw=2, x_unit='rad', y_unit='nRsol/d')
    plt.legend(loc='best').get_frame().set_alpha(0.5)

    #np.savetxt('testoutput1.dat', np.column_stack([syn1['time'], syn1['flux'], obs1['phase'], obs1['flux']]))
    #np.savetxt('testoutput2.dat', np.column_stack([syn2['time'], syn2['flux'], obs2['phase'], obs2['flux']]))
    #np.savetxt('testoutput3.dat', np.column_stack([syn3['time'], syn3['rv'], obs3['phase'], obs3['rv']]))
    #np.savetxt('testoutput4.dat', np.column_stack([syn4['time'], syn4['rv'], obs4['phase'], obs4['rv']]))

    test1 = np.loadtxt('testoutput1.dat').T
    test2 = np.loadtxt('testoutput2.dat').T
    test3 = np.loadtxt('testoutput3.dat').T
    test4 = np.loadtxt('testoutput4.dat').T

    if N==3:
        assert(np.all(np.abs(syn1['time']-test1[0]))<1e-6)
        assert(np.all(np.abs(syn1['flux']-test1[1]))<1e-6)
        assert(np.all(np.abs(obs1['phase']-test1[2]))<1e-6)
        print obs1['flux'], test1[3]
        assert(np.all(np.abs(obs1['flux']-test1[3]))<1e-6)

        assert(np.all(np.abs(syn2['time']-test2[0]))<1e-6)
        assert(np.all(np.abs(syn2['flux']-test2[1]))<1e-6)
        assert(np.all(np.abs(obs2['phase']-test2[2]))<1e-6)
        assert(np.all(np.abs(obs2['flux']-test2[3]))<1e-6)

        assert(np.all(np.abs(syn3['time']-test3[0]))<1e-6)
        assert(np.all(np.abs(syn3['rv']-test3[1]))<1e-6)
        assert(np.all(np.abs(obs3['phase']-test3[2]))<1e-6)
        assert(np.all(np.abs(obs3['rv']-test3[3]))<1e-6)

        assert(np.all(np.abs(syn4['time']-test4[0]))<1e-6)
        assert(np.all(np.abs(syn4['rv']-test4[1]))<1e-6)
        assert(np.all(np.abs(obs4['phase']-test4[2]))<1e-6)
        assert(np.all(np.abs(obs4['rv']-test4[3]))<1e-6)

if __name__ == "__main__":
    logger = phoebe.get_basic_logger()
    test_plot(N=201)
    plt.show()
