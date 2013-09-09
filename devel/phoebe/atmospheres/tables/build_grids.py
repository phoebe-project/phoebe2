import glob
import pyfits
import numpy as np
import matplotlib.pyplot as plt
import phoebe
from phoebe.atmospheres import limbdark

logger = phoebe.get_basic_logger()


def build_grid(filetag='kurucz', passbands=None, ld_func='claret', fitmethod='equidist_r_leastsq',
               redlaw='fitzpatrick2004', Rv=3.1, z='p00', vmic=2, ebvs=None,
               vgamma=None):
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
                red_pars_fixed=dict(law=redlaw,ebv=0.,Rv=Rv),
                law=ld_func,passbands=passbands,fitmethod=fitmethod,filetag=filetag)
        
    else:
        pattern = 'spec_intens/{}_mu_i{}k{:.0f}.fits'.format(filetag,z,vmic)
        logger.info("Collecting files with pattern {}".format(pattern))
        atm_files = sorted(glob.glob(pattern))
        print(atm_files)
        limbdark.compute_grid_ld_coeffs(atm_files,atm_pars=atm_pars,
                red_pars_fixed=red_pars_fixed,vgamma=vgamma,
                law=ld_func,passbands=passbands,fitmethod=fitmethod,filetag='{}_{}'.format(filetag,z))


if __name__=="__main__":
    #build_grid(filetag='phoenix', passbands=None, ld_func='claret', fitmethod='equidist_r_leastsq',
    #           redlaw='fitzpatrick2004', Rv=3.1, z='p00', vmic=1, ebvs=None,
    #           vgamma=None)
    #build_grid(filetag='kurucz', passbands=('JOHNSON.V',), ld_func='linear', fitmethod='equidist_r_leastsq',
               #z='p00', ebvs=None, redlaw=None, 
               #vgamma=np.linspace(-500,500,21))
    build_grid(filetag='kurucz', passbands=('JOHNSON.V','KEPLER.V'), ld_func='quadratic', fitmethod='equidist_r_leastsq',
               z='m01', ebvs=None, redlaw=None)#, 
#               vgamma=np.linspace(-50,50,21))
    