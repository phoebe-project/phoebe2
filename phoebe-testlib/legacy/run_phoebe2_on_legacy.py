import phoebeBackend as phb
from phoebe.parameters import tools
import numpy as np
import matplotlib.pyplot as plt
import time
import phoebe as phb2
import sys
import subprocess
import signal
import os
import cProfile
#~ from ivs.io import plotting

# use this script as:
# $:> python run_phoebe2_on_legacy.py lc01.phoebe

# an interactive session will pop up a plot with diagnostics
interactive = True

# Initialize Phoebe1 and Phoebe2
phb.init()
phb.configure()
phb2.atmospheres.limbdark.register_atm_table('blackbody_uniform_none_teff.fits')

# Template phases
ph = np.linspace(-0.6, 0.6, 201)


def do_test(filename):

    # To make sure we are using the same settings, open the same parameter file:
    print("Running Phoebe Legacy")
    phb.open(filename)
    
    # Change time to phase and use only black body atmospheres:
    phb.setpar("phoebe_indep", "Phase")
    phb.setpar("phoebe_atm1_switch", False)
    phb.setpar("phoebe_atm2_switch", False)
    #phb.setpar("phoebe_usecla_switch", True)
    
    lc_ph1 = np.array(phb.lc(tuple(list(ph)), 0))
    print("PHOEBE 1: HLA = %f, CLA = %f" % (phb.getpar("phoebe_plum1"), phb.getpar("phoebe_plum2")))
    
    #if interactive:
        #answer = raw_input('Save phoebe1 output file? [y/N] ')
        #if answer and answer[0].lower()=='y':
            #np.savetxt('{}.lc.data'.format(os.path.splitext(filename)[0]),
                   #np.column_stack([ph, lc_ph1]))
    
    # Phoebe2 Bundle
    mybundle = phb2.Bundle(filename)
    
    # Ignore any existing obs
    mybundle.disable_obs()
    
    # use the settings of the first lc in Phoebe Legacy for Phoebe2
    existing_refs = mybundle.get_value_all('ref@lcdep')
    if existing_refs:
        dataref = existing_refs[0]
    else:
        dataref = 'mylightcurve'
    
    mybundle.create_data(category='lc', phase=ph, dataref=dataref)
        
    # Atmospheres:
    atms = sorted(list(set(mybundle.get_value_all('atm'))))
    ld_funcs = sorted(list(set(mybundle.get_value_all('ld_func'))))

    # set custom passband and atmosphere if blackbodies are requested. Else
    # we stick to Phoebe2 default atmosphere/passbands
    twigs_atm = mybundle.search('atm')
    for atm in twigs_atm:
        if atm.split('@')[0] == 'value':
            continue
        if mybundle[atm] in ['blackbody','kurucz']:
            mybundle[atm] = 'blackbody_uniform_none_teff.fits'
            passband_twig = 'passband@{}'.format("@".join(atm.split('@')[1:]))
            if passband_twig in mybundle and mybundle[passband_twig] == 'JOHNSON.V':
                mybundle[passband_twig] = 'johnson_v.ptf'
    
    mybundle['pblum@secondary'] = phb.getpar('phoebe_plum2')
    mybundle.run_compute(label='from_legacy', irradiation_alg='point_source')
    #mybundle.get_system().compute(animate=True)
    lc_ph2 = mybundle['flux@{}@lcsyn'.format(dataref)]
    
    U = phb2.units.conversions.Unit
    R1 = U(mybundle.get_system()[0].params['component'].request_value('r_pole'), 'm')
    T1 = U(mybundle['teff@primary'], 'K')
    R2 = U(mybundle.get_system()[1].params['component'].request_value('r_pole'), 'm')
    T2 = U(mybundle['teff@secondary'], 'K')
    sigma = U('sigma')
    L1 = (4*np.pi*R1**2*sigma*T1**4).convert('Lsol')
    L2 = (4*np.pi*R2**2*sigma*T2**4).convert('Lsol')
    print("Numerical bolometric luminosity (primary) = {} Lsol".format(phb2.convert('erg/s', 'Lsol',mybundle['primary'].luminosity())))
    print("Numerical bolometric luminosity (secondary) = {} Lsol".format(phb2.convert('erg/s', 'Lsol',mybundle['secondary'].luminosity())))
    print("Eq. sphere bolometric luminosity (primary) = {}".format(L1))
    print("Eq. sphere bolometric luminosity (secondary) = {}".format(L2))
    print("Numerical passband luminosity (primary) = {} Lsol".format(phb2.convert('erg/s', 'Lsol',mybundle['primary'].luminosity(ref='LC'))))
    print("Numerical passband luminosity (secondary) = {} Lsol".format(phb2.convert('erg/s', 'Lsol',mybundle['secondary'].luminosity(ref='LC'))))
    
    
    # Passband luminosities:
    plum1, plum2 = phb.getpar('phoebe_plum1'), phb.getpar('phoebe_plum2')
    pblum1 = mybundle.get_system()[0].params['pbdep']['lcdep'][dataref]['computed_pblum']
    pblum2 = mybundle.get_system()[1].params['pbdep']['lcdep'][dataref]['computed_pblum']
    connected = mybundle.get_system()[1].params['pbdep']['lcdep'][dataref]['pblum']
    
    # Atmospheres for data (by virtue of the parser, the components are the same):
    atms = sorted(list(set(mybundle.get_value_all('atm@'+dataref))))
    ld_funcs = sorted(list(set(mybundle.get_value_all('ld_func@'+dataref))))
    
    print("Making figure")
    plt.figure(figsize=(14,11))
    plt.subplots_adjust(left=0.07, bottom=0.05, right=0.98, top=0.98)
    plt.subplot(211)
    plt.plot(ph, lc_ph1, 'ko-', label='Phoebe legacy')
    plt.plot(ph, lc_ph2, 'ro-', label='Phoebe 2')
    
    plt.annotate('Legacy / Phoebe2:', (0.1,0.20), xycoords='axes fraction', ha='left', va='bottom')
    plt.annotate('pblum1 = {:.6f} / {:.6f}'.format(plum1, pblum1), (0.1,0.15), xycoords='axes fraction', ha='left', va='bottom')
    plt.annotate('pblum2 = {:.6f} / {:.6f}'.format(plum2, pblum2), (0.1,0.10), xycoords='axes fraction', ha='left', va='bottom')
    plt.annotate('({}connected)'.format('dis' if connected==-1 else ''), (0.1, 0.05), ha='left', va='bottom', xycoords='axes fraction')
    
    plt.annotate('LD functions: {}'.format(", ".join(ld_funcs)), (0.1,0.30), xycoords='axes fraction', ha='left', va='bottom')
    plt.annotate('Atmospheres: {}'.format(", ".join(atms)), (0.1,0.25), xycoords='axes fraction', ha='left', va='bottom')
    
    plt.legend(loc='lower right').get_frame().set_alpha(0.5)
    plt.subplot(212)
    plt.plot(ph, (lc_ph1-lc_ph2)*1e6, 'ro-')
    plt.ylabel("Residuals [ppm]")

    plt.show()
        

if __name__ == "__main__":
    
    logger = phb2.get_basic_logger()
    
    filenames = sys.argv[1:]
    
    
    for filename in filenames:
        do_test(filename)
        
    phb.quit()
