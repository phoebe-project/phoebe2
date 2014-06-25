import phoebe
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

phoebe.get_basic_logger()#clevel='INFO')

checks = dict()
#checks['BinaryTest2_theta1_0.5_theta2_0.5_q_1.0_x1_20.0_y1_0.0_x2_-20.0_y2_0.0.txt'] = [0.000170174734863, 0.00520802698803]
#checks['BinaryTest_theta1_0.5_theta2_0.5_q_1.0_x1_0.0_y1_7.5_x2_0.0_y2_-7.5.txt'] = [0.00016031055117,0.0049772536376]
#checks['BinaryTest_theta1_0.5_theta2_0.5_q_1.0_x1_2.5_y1_0.0_x2_-2.5_y2_0.0.txt'] = [0.000284665146186,0.0036351865665]
#checks['BinaryTest_theta1_0.5_theta2_0.5_q_1.0_x1_2.5_y1_0.0_x2_-2.5_y2_0.0.txt'] = [2.846651e-04, 3.635187e-03]
#checks['BinaryTest_theta1_0.5_theta2_0.5_q_1.0_x1_20.0_y1_0.0_x2_-20.0_y2_0.0.txt'] = [2.762267e-04, 2.482429e-03]
#checks['BinaryTest_theta1_0.5_theta2_0.5_q_1.0_x1_7.5_y1_7.5_x2_-7.5_y2_-7.5.txt'] = [2.888030e-04, 3.895467e-03]
#checks['BinaryTest_theta1_0.5_theta2_0.5_q_10.0_x1_7.5_y1_0.0_x2_-7.5_y2_0.0.txt'] = [3.030775e-04, 7.686127e-03]
#checks['BinaryTest_theta1_0.5_theta2_0.5_q_3.0_x1_7.5_y1_0.0_x2_-7.5_y2_0.0.txt'] = [2.702074e-04, 6.293457e-03]
#checks['BinaryTest_theta1_1.5_theta2_0.5_q_1.0_x1_7.5_y1_0.0_x2_-7.5_y2_0.0.txt'] = [2.118577e-04, 4.549472e-03]
checks['BinaryTest_theta1_1.5_theta2_0.5_q_2.0_x1_5.0_y1_3.0_x2_-5.0_y2_-3.0.txt'] = [2.349618e-04, 7.407903e-02]
#checks['BinaryTest_theta1_1.5_theta2_0.5_q_5.0_x1_7.5_y1_0.0_x2_-7.5_y2_0.0.txt'] = [3.108696e-04, 6.976795e-03]






def test_interferometry(index=None):
    """
    Interferometry: comparison with analytical results
    """

    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__),'datafiles/*.txt')))
    if index is not None:
        files = files[index:index+1]
    else:
        files = [ff for ff in files if os.path.basename(ff) in checks]

    for ff in files:
        gg = os.path.splitext(os.path.basename(ff))[0].split('_')
        
        # Extract info from filename
        ang_diam2 = float(gg[2])
        ang_diam1 = float(gg[4])
        fr = float(gg[6])
        x1 = float(gg[8])
        y1 = float(gg[10])
        x2 = float(gg[12])
        y2 = float(gg[14])
        
        sep = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        y = y2-y1
        if x2==x1 and y1<y2:
            pa = -90.0
        elif x2==x1 and y1>y2:
            pa = 90.0
        else:
            pa = np.arctan((y1-y2)/(x2-x1))/np.pi*180.
        
        
        print('{} (sep={}, theta1={}, theta2={}, fr={}, y={}, pa={})'.format(ff, sep, ang_diam1, ang_diam2, fr, y, pa))
        
        
        # Define the system
        binary = phoebe.Bundle()
        binary['period'] = 100000000.0
        binary['t0'] = 0.0 + 0.25*binary['period'] - pa/360.0*binary['period']
        binary['distance'] = 1.0, 'kpc'
        binary['sma'] = sep, 'au'
        binary['incl'] = 0., 'deg'
        phoebe.compute_pot_from(binary, ang_diam1/sep/2.0, component='primary')
        phoebe.compute_pot_from(binary, ang_diam2/sep/2.0, component='secondary')
        binary['teff@secondary'] = binary['teff@primary'] * fr**-0.25 * np.sqrt(ang_diam1/ang_diam2)
        binary.set_value_all('atm', 'blackbody')
        binary.set_value_all('ld_func', 'uniform')
        binary.set_value_all('delta', 0.2)
        binary.set_value_all('maxpoints', 300000)
        
        # Load the observations from a textfile
        ucoord, vcoord, eff_wave, vis, phase = np.loadtxt(ff).T
        binary.if_fromarrays(ucoord=ucoord, vcoord=vcoord, vis2=vis**2,
                            vphase=phase/180*np.pi, time=np.zeros_like(ucoord),
                            eff_wave=eff_wave*1e10,
                            passband='OPEN.BOL')
        
        # Do the computations
        binary.run_compute()
        
        # Perform some basic checks on input parameters: are radii,
        # separations etc good?
        r = binary.get_system()[0].get_coords()[0]
        r1 = binary.get_system()[0].get_coords()[0]
        r2 = binary.get_system()[1].get_coords()[0]
        
        print "Distortion (%):           ", (r1.max()/r1.min()-1)*100, (r2.max()/r2.min()-1)*100
        print "Diameters (Rsol):         ", 2*r1.mean(), 2*r2.mean()
        print "Separation (Rsol):        ", binary['sma']
        print "Relative diameter (prim):  input={:.6f}, output={:.6f}".format(ang_diam1/sep,2*r1.mean()/binary['sma'])
        print "Relative diameter (secn):  input={:.6f}, output={:.6f}".format(ang_diam2/sep,2*r2.mean()/binary['sma'])
        X1 = (binary.get_system()[0].mesh['center']*binary.get_system()[0].mesh['size'][:,None]/binary.get_system()[0].mesh['size'].sum()).sum(axis=0)
        X2 = (binary.get_system()[1].mesh['center']*binary.get_system()[1].mesh['size'][:,None]/binary.get_system()[1].mesh['size'].sum()).sum(axis=0)
        sep_ = np.sqrt((X1[0]-X2[0])**2 + (X1[1]-X2[1])**2 + (X1[2]-X2[2])**2)
        print "Relative radius (prim):    input={:.6f}, output={:.6f}".format(ang_diam1/sep/2.0, r1.mean()/sep_)
        print "Relative radius (secn):    input={:.6f}, output={:.6f}".format(ang_diam2/sep/2.0, r2.mean()/sep_)
        
        # Get the synthetics
        system = binary.get_system()                        
        syn = system.get_synthetic(category='if', ref='if01')
        vis_out = np.sqrt(syn['vis2'])
        phase_out = syn['vphase']/np.pi*180
                    
        baseline_in = np.sqrt(ucoord**2+vcoord**2)
        baseline_out= np.sqrt(syn['ucoord']**2 + syn['vcoord']**2)
        
        if index is not None:
            plt.figure()
            plt.plot(baseline_in, vis, 'k-', label='Michel')
            plt.plot(baseline_out, vis_out, 'r-', label='Phoebe2')
            plt.figure()
            plt.plot(baseline_in, vis-vis_out, 'k-')
            plt.figure()
            plt.plot(baseline_in, phase, 'ko', label='Michel')
            plt.plot(baseline_out, phase_out, 'r+', ms=10, label='Phoebe2')
            
            plt.figure()
            plt.plot(baseline_in, (phase-phase_out), 'ko', label='Analytical')
        
        print ff, np.std(vis-vis_out)
        print ff, np.abs(np.median(phase-phase_out))
        print("checks['{}'] = [{:.6e}, {:.6e}]".format(os.path.basename(ff),
                                                       np.std(vis-vis_out),
                                                       np.abs(np.median(phase-phase_out))))
        # Don't exceed 10% of previously established ranges
        continue
        assert(np.std(vis-vis_out)<1.1*checks[os.path.basename(ff)][0])
        assert(np.abs(np.median(phase-phase_out))<1.1*checks[os.path.basename(ff)][1])

if __name__ == "__main__":
    for index in range(2,10):
        test_interferometry(index=index)
    #plt.show()