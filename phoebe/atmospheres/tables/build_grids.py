import glob
import pyfits
import numpy as np
import matplotlib.pyplot as plt
import phoebe
import sys
#from phoebe.atmospheres import limbdark
from phoebe.atmospheres import create_atmospherefits as limbdark

logger = phoebe.get_basic_logger(clevel='INFO')


def build_grid(filetag='kurucz', passbands=None, ld_func='claret', fitmethod='equidist_r_leastsq',
               limb_zero=True,
               redlaw='fitzpatrick2004', Rv=3.1, z='p00', vmic=2, ebvs=None,
               vgamma=None, add_boosting_factor=True):
    if passbands is None:
        passbands = ('MOST.V','COROT.SIS','COROT.EXO','KEPLER.V',
             '2MASS.J','2MASS.H','2MASS.KS','OPEN.BOL',
             'JOHNSON.V','JOHNSON.U','JOHNSON.B','JOHNSON.J','JOHNSON.H','JOHNSON.K',
             'STROMGREN.U','STROMGREN.B','STROMGREN.V','STROMGREN.Y',
             'TYCHO2.BT','TYCHO2.VT','HIPPARCOS.HP',
             'ANS.15N','ANS.15W','ANS.18',
             'ANS.25','ANS.33',
             'JOHNSON.L','JOHNSON.M','GENEVA.V','GENEVA.B','JOHNSON.I',
             'GENEVA.V1','JOHNSON.R',
             'GENEVA.B1','GENEVA.B2','GENEVA.G','VILNIUS.V','VILNIUS.Z','GENEVA.U',
             'VILNIUS.S','VILNIUS.Y','VILNIUS.X','VILNIUS.P','VILNIUS.U','ARGUE.R',
             'KRON.R','KRON.I','TD1.1565','TD1.1965','TD1.2365','TD1.2740'
             )
    
    #if ebvs is None:
    #    ebvs = np.arange(0,0.51,0.01)
    #if vgamma is None:
    #    vgamma = np.arange(-500,501,100.)

    atm_pars = ['teff', 'logg']
    if z=='*':
        atm_pars.append('abun')
    
    atm_pars = tuple(atm_pars)
    
    if redlaw is not None:
        red_pars_fixed = dict(law=redlaw,ebv=0.,Rv=Rv)
    else:
        red_pars_fixed = dict()

    #-- if we need to interpolate in abun, we need a grid of specific intensities
    #   that is uniform in grid points over all abuns, this is exactly what I put
    #   in the folder 'spec_intens_z' (Kurucz is really annoying)
    if z == '*' and filetag == 'kurucz':
        atm_files = sorted(glob.glob('spec_intens_z/{}_mu_i*k{:.0f}.fits'.format(filetag,vmic)))
        limbdark.compute_grid_ld_coeffs(atm_files,atm_pars=atm_pars,
                red_pars_fixed=red_pars_fixed,vgamma=vgamma,
                limb_zero=limb_zero,
                law=ld_func,passbands=passbands,fitmethod=fitmethod,
                filetag=filetag,
                add_boosting_factor=add_boosting_factor)
        
    else:
        pattern = 'spec_intens/{}_mu_i{}k{:.0f}.fits'.format(filetag,z,vmic)
        logger.info("Collecting files with pattern {}".format(pattern))
        atm_files = sorted(glob.glob(pattern))
        print(atm_files)
        limbdark.compute_grid_ld_coeffs(atm_files,atm_pars=atm_pars,
                red_pars_fixed=red_pars_fixed,vgamma=vgamma,
                limb_zero=limb_zero,
                law=ld_func,passbands=passbands,fitmethod=fitmethod,
                add_boosting_factor=add_boosting_factor,
                filetag='{}_{}'.format(filetag,z))





if __name__=="__main__":
    if not sys.argv[1:]:
        #build_grid(filetag='phoenix', passbands=None, ld_func='claret', fitmethod='equidist_r_leastsq',
        #           redlaw='fitzpatrick2004', Rv=3.1, z='p00', vmic=1, ebvs=None,
        #           vgamma=None)
        #build_grid(filetag='kurucz', passbands=('JOHNSON.V',), ld_func='linear', fitmethod='equidist_r_leastsq',
                #z='p00', ebvs=None, redlaw=None, 
                #vgamma=np.linspace(-500,500,21))
        #build_grid(filetag='kurucz', passbands=('JOHNSON.V','KEPLER.V'),
                #ld_func='claret', fitmethod='equidist_r_leastsq',
                #z='m01', ebvs=None, redlaw=None, limb_zero=False)
        #build_grid(filetag='kurucz', passbands=('JOHNSON.V','KEPLER.V'),
                #ld_func='claret', fitmethod='equidist_r_leastsq',
                #z='m01', ebvs=None, redlaw=None, limb_zero=False,
                #vgamma=np.linspace(-500,500,101))
        #build_grid(filetag='kurucz', passbands=('JOHNSON.V','KEPLER.V'),
                #ld_func='claret', fitmethod='equidist_r_leastsq',
                #z='*', ebvs=None, redlaw=None, limb_zero=False,
                #add_boosting_factor=True)
        
        #build_grid(filetag='kurucz', passbands=('JOHNSON.V','KEPLER.V'),
                #ld_func='claret', fitmethod='equidist_r_leastsq',
                #z='*', ebvs=None, redlaw=None, limb_zero=False,
                #add_boosting_factor=True, vgamma=np.arange(-500,501,50))
        
        #build_grid(filetag='kurucz', passbands=('JOHNSON.V','KEPLER.V'),
                #ld_func='linear', fitmethod='equidist_r_leastsq',
                #z='*', ebvs=None, redlaw=None, limb_zero=False,
                #add_boosting_factor=True)

        #build_grid(filetag='kurucz', passbands=('JOHNSON.V','KEPLER.V'),
                #ld_func='logarithmic', fitmethod='equidist_r_leastsq',
                #z='*', ebvs=None, redlaw=None, limb_zero=False,
                #add_boosting_factor=True)
        
        #build_grid(filetag='kurucz', passbands=('JOHNSON.V','KEPLER.V'),
                #ld_func='quadratic', fitmethod='equidist_r_leastsq',
                #z='*', ebvs=None, redlaw=None, limb_zero=False,
                #add_boosting_factor=True)
         
        #limbdark.compute_grid_ld_coeffs(['spec_intens/Jorissen_m1.0_t02_st_z+0.00_a+0.00_mu.fits'],
        #      passbands=('JOHNSON.*','2MASS.*', 'KEPLER.V'), filetag='jorissen_m1.0_t02_st_z+0.00_a+0.00',
        #      fitmethod='equidist_mu_leastsq',
        #      law='hillen', debug_plot=True)
        
        #limbdark.compute_grid_ld_coeffs('blackbody', passbands=('*',), filetag='blackbody', law='uniform')
        
        #limbdark.compute_grid_ld_coeffs('blackbody', passbands=('*',), filetag='blackbody', law='uniform')
        
            
        #build_grid(filetag='kurucz', passbands=('JOHNSON.V','KEPLER.V'),
                #ld_func='claret', fitmethod='equidist_r_leastsq',
                #z='*', ebvs=None, redlaw=None, limb_zero=False,
                #add_boosting_factor=True)
    
        build_grid(filetag='kurucz', passbands=('JOHNSON.V','JOHNSON.B',
                'GAIA.BP', 'GAIA.RP', 'KEPLER.V'), ld_func='quadratic',
                fitmethod='equidist_r_leastsq', z='*', ebvs=None, redlaw=None,
                limb_zero=False, add_boosting_factor=True)
        
    
    else:
        
        # add box filters:
        passbands = []
        for clam in np.arange(3000,8001,500):
            wave = np.linspace(clam-20,clam+20,1000)
            trans = np.where(np.abs(clam-wave)<10.0, 1.0, 0.0)
            passband ='BOX_10.{:.0f}'.format(clam)
            passbands.append(passband)
            limbdark.pbmod.add_response(wave, trans, passband=passband)
        
        limbdark.compute_grid_ld_coeffs(sys.argv[1], passbands=sys.argv[2:]+passbands)
    
