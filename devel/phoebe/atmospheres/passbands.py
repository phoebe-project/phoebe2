#from phoebe.c import h, c, k_B
#from phoebe import u

# NOTE: we'll import directly from astropy here to avoid
# circular imports BUT any changes to these units/constants
# inside phoebe will be ignored within passbands
from astropy.constants import h, c, k_B
from astropy import units as u

import numpy as np
from scipy import interpolate, integrate
import marshal
import types
from phoebe.atmospheres import atmcof
from phoebe.algorithms import interp
import os
import glob

import logging
logger = logging.getLogger("PASSBANDS")
logger.addHandler(logging.NullHandler())

# Global passband table. This dict should never be tinkered with outside
# of the functions in this module; it might be nice to make it read-only
# at some point.
_pbtable = {}

class Passband:
    def __init__(self, ptf=None, pbset='Johnson', pbname='V', effwl=5500.0, wlunits=u.AA, calibrated=False, reference='', version=1.0, comments='', oversampling=1, from_file=False):
        """
        Passband class holds data and tools for passband-related computations, such as
        blackbody intensity, model atmosphere intensity, etc.
        
        @ptf: passband transmission file: a 2-column file with wavelength in @wlunits
              and transmission in arbitrary units
        @pbset: name of the passband set (i.e. Johnson)
        @pbname: name of the passband name (i.e. V)
        @effwl: effective wavelength in @wlunits
        @wlunits: wavelength units from astropy.units used in @ptf and @effwl
        @calibrated: true if transmission is in true fractional light,
                     false if it is in relative proportions
        @reference: passband transmission data reference (i.e. ADPS)
        @version: file version
        @comments: any additional comments about the passband
        @oversampling: the multiplicative factor of PTF dispersion to attain higher
                       integration accuracy
        @from_file: a switch that instructs the class instance to skip all calculations
                    and load all data from the file passed to the Passband.load() method.


        Step #1: initialize passband object
        
        .. testcode::
        
            >>> pb = Passband(ptf='JOHNSON.V', pbset='Johnson', pbname='V', effwl=5500.0, wlunits=u.AA, calibrated=True, reference='ADPS', version=1.0, comments='')

        Step #2: compute intensities for blackbody radiation:
        
        .. testcode ::
        
            >>> pb.compute_blackbody_response()

        Step #3: compute Castelli & Kurucz (2004) intensities. To do this,
        the tables/ck2004 directory needs to be populated with non-filtered
        intensities available for download from %static%/ck2004.tar.

        .. testcode::
            
            >>> atmdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/ck2004'))
            >>> pb.compute_ck2004_response(atmdir)

        Step #4: -- optional -- import WD tables for comparison. This can only
        be done if the passband is in the list of supported passbands in WD.
        The WD index of the passband is passed to the import_wd_atmcof()
        function below as the last argument.            

        .. testcode::

            >>> from phoebe.atmospheres import atmcof
            >>> atmdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/wd'))
            >>> atmcof.init(atmdir+'/atmcofplanck.dat', atmdir+'/atmcof.dat')
            >>> pb.import_wd_atmcof(atmdir+'/atmcofplanck.dat', atmdir+'/atmcof.dat', 7)

        Step #5: save the passband file:

        .. testcode::

            >>> atmdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/passbands'))
            >>> pb.save(atmdir + '/johnson_v.ptf')

        From now on you can use @pbset:@pbname as a passband qualifier, i.e.
        Johnson:V for the example above. Further details on supported model
        atmospheres are available by issuing:

        .. testcode::
        
            >>> pb.content


        """
        self.h = h.value
        self.c = c.value
        self.k = k_B.value

        if from_file:
            return

        # Initialize content list; each method that adds any content
        # to the passband file needs to add a corresponding label to the
        # content list.
        self.content = []
                
        # Basic passband properties:
        self.pbset = pbset
        self.pbname = pbname
        self.effwl = effwl
        self.calibrated = calibrated
        
        # Passband transmission function table:
        ptf_table = np.loadtxt(ptf).T
        ptf_table[0] = ptf_table[0]*wlunits.to(u.m)
        self.ptf_table = {'wl': np.array(ptf_table[0]), 'fl': np.array(ptf_table[1])}

        # Spline fit to the passband transmission function table:
        self.ptf_func = interpolate.splrep(self.ptf_table['wl'], self.ptf_table['fl'], s=0)
        self.ptf = lambda wl: interpolate.splev(wl, self.ptf_func)

        # Working wavelength array:
        self.wl = np.linspace(self.ptf_table['wl'][0], self.ptf_table['wl'][-1], oversampling*len(self.ptf_table['wl']))

    def save(self, archive):
        struct = dict()

        struct['content']       = self.content
        struct['pbset']         = self.pbset
        struct['pbname']        = self.pbname
        struct['effwl']         = self.effwl
        struct['calibrated']    = self.calibrated
        struct['ptf_table']     = self.ptf_table
        struct['ptf_func']      = self.ptf_func
        struct['ptf_wl']        = self.wl
        if 'blackbody' in self.content:
            struct['_bb_func']      = self._bb_func
        if 'ck2004' in self.content:
            struct['_ck2004_axes']  = self._ck2004_axes
            struct['_ck2004_grid']  = self._ck2004_grid
        if 'atmcof' in self.content:
            struct['extern_wd_idx'] = self.extern_wd_idx

        f = open(archive, 'wb')
        marshal.dump(struct, f)
        f.close()

    @classmethod
    def load(cls, archive):
        logger.debug("loading passband from {}".format(archive))
        f = open(archive, 'rb')
        struct = marshal.load(f)
        f.close()

        self = cls(from_file=True)

        self.content = struct['content']

        self.pbset = struct['pbset']
        self.pbname = struct['pbname']
        self.effwl = struct['effwl']
        self.calibrated = struct['calibrated']
        self.ptf_table = struct['ptf_table']
        self.ptf_table['wl'] = np.fromstring(self.ptf_table['wl'], dtype='float64')
        self.ptf_table['fl'] = np.fromstring(self.ptf_table['fl'], dtype='float64')
        self.wl = np.fromstring(struct['ptf_wl'], dtype='float64')

        if 'blackbody' in self.content:
            self._bb_func = list(struct['_bb_func'])
            self._bb_func[0] = np.fromstring(self._bb_func[0])
            self._bb_func[1] = np.fromstring(self._bb_func[1])
            self._bb_func = tuple(self._bb_func)
            self._log10_Inorm_bb = lambda Teff: interpolate.splev(Teff, self._bb_func)

        self.ptf_func = list(struct['ptf_func'])
        self.ptf_func[0] = np.fromstring(self.ptf_func[0])
        self.ptf_func[1] = np.fromstring(self.ptf_func[1])
        self.ptf_func = tuple(self.ptf_func)
        self.ptf = lambda wl: interpolate.splev(wl, self.ptf_func)

        if 'atmcof' in self.content:
            if not atmcof.meta.initialized:
                atmdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/wd'))
                atmcof.init(atmdir+'/atmcofplanck.dat', atmdir+'/atmcof.dat')
            self.extern_wd_idx = struct['extern_wd_idx']

        if 'ck2004' in self.content:
            # CASTELLI & KURUCZ (2004):
            # Axes needs to be a tuple of np.arrays, and grid a np.array:
            self._ck2004_axes  = tuple(map(lambda x: np.fromstring(x, dtype='float64'), struct['_ck2004_axes']))
            #~ self._ck2004_axes = (np.fromstring(x, dtype='float64') for x in struct['_ck2004_axes'])
            self._ck2004_grid = np.fromstring(struct['_ck2004_grid'], dtype='float64')
            self._ck2004_grid = self._ck2004_grid.reshape(len(self._ck2004_axes[0]), len(self._ck2004_axes[1]), len(self._ck2004_axes[2]), 1)
        
        return self

    def _planck(self, lam, Teff):
        return 2*self.h*self.c*self.c/lam**5 * 1./(np.exp(self.h*self.c/lam/self.k/Teff)-1)

    def _bb_intensity(self, Teff):
        pb = lambda w: self._planck(w, Teff)*self.ptf(w)
        return integrate.quad(pb, self.wl[0], self.wl[-1])[0]

    def compute_blackbody_response(self, Teffs=None):
        if Teffs == None:
            Teffs = np.linspace(3500, 50000, 100)

        log10ints = np.array([np.log10(self._bb_intensity(Teff)) for Teff in Teffs])
        self._bb_func = interpolate.splrep(Teffs, log10ints, s=0)
        self._log10_Inorm_bb = lambda Teff: interpolate.splev(Teff, self._bb_func)
        self.content.append('blackbody')

    def compute_ck2004_response(self, path, verbose=False):
        models = glob.glob(path+'/*M1.000.spectrum')
        Teff, logg, met, Inorm = [], [], [], []
        
        if verbose:
            print('Computing Castelli-Kurucz passband intensities for %s:%s. This will take a while.' % (self.pbset, self.pbname))

        for i, model in enumerate(models):
            spc = np.loadtxt(model).T
            Teff.append(float(model[-26:-21]))
            logg.append(float(model[-20:-18]))
            sign = 1. if model[-18]=='P' else -1.
            met.append(sign*float(model[-17:-15]))
            spc[0] /= 1e10 # AA -> m
            spc[1] *= 1e7  # erg/s/cm^2/A -> W/m^3
            wl = spc[0][(spc[0] >= self.ptf_table['wl'][0]) & (spc[0] <= self.ptf_table['wl'][-1])]
            fl = spc[1][(spc[0] >= self.ptf_table['wl'][0]) & (spc[0] <= self.ptf_table['wl'][-1])]
            fl *= self.ptf(wl)
            Inorm.append(np.log10(fl.sum())-10)  # -10 because of the 1AA dispersion
            if verbose:
                if 100*i % (len(models)) == 0:
                    print('%d%% done.' % (100*i/(len(models)-1)))

        Teff = np.array(Teff)
        logg = np.array(logg)/10
        abun = np.array(met)/10
        
        # Store axes (Teff, logg, abun) and the full grid of Inorm, with
        # nans where the grid isn't complete.
        self._ck2004_axes = (np.unique(Teff), np.unique(logg), np.unique(abun))
        self._ck2004_grid = np.nan*np.ones((len(self._ck2004_axes[0]), len(self._ck2004_axes[1]), len(self._ck2004_axes[2]), 1))
        for i, Inorm in enumerate(Inorm):
            self._ck2004_grid[Teff[i] == self._ck2004_axes[0], logg[i] == self._ck2004_axes[1], abun[i] == self._ck2004_axes[2], 0] = Inorm

        # Tried radial basis functions but they were just terrible.
        #~ self._log10_Inorm_ck2004 = interpolate.Rbf(self._ck2004_Teff, self._ck2004_logg, self._ck2004_met, self._ck2004_Inorm, function='linear')
        self.content.append('ck2004')

    def compute_ck2004_ldc(self, path, verbose=False, photon_weighted=True, weight_by_mu=False):
        models = os.listdir(path)
        Teff, logg, met, mu, Imu = [], [], [], [], []
        
        if verbose:
            print('Computing Castelli-Kurucz limb darkening coefficients for %s:%s. This will take a while.' % (self.pbset, self.pbname))
        
        
        
    def import_wd_atmcof(self, plfile, atmfile, wdidx, Nmet=19, Nlogg=11, Npb=25, Nints=4):
        """
        Parses WD's atmcof and reads in all Legendre polynomials for the
        given passband.
        
        @plfile: path and filename of atmcofplanck.dat
        @atmfile: path and filename of atmcof.dat
        @wdidx: WD index of the passed passband. This can be automated
                but it's not a high priority.
        @Nmet:  number of metallicity nodes in atmcof.dat. For the 2003 version
                the number of nodes is 19.
        @Nlogg: number of logg nodes in atmcof.dat. For the 2003 version
                the number of nodes is 11.
        @Npb:   number of passbands in atmcof.dat. For the 2003 version
                the number of passbands is 25.
        @Nints: number of temperature intervals (input lines) per entry.
                For the 2003 version the number of lines is 4.
        """

        # Initialize the external atmcof module if necessary:
        if not atmcof.meta.initialized:
            atmcof.init(plfile, atmfile)
        
        # That is all that was necessary for *_extern_planckint() and
        # *_extern_atmx() functions. However, we also want to support
        # circumventing WD subroutines and use WD tables directly. For
        # that, we need to do a bit more work.
        
        # Store the passband index for use in planckint() and atmx():
        self.extern_wd_idx = wdidx
        
        # The original atmcof.dat features 'D' instead of 'E' for
        # exponential notation. We need to provide a converter for
        # numpy's loadtxt to read that in:
        D2E = lambda s: float(s.replace('D', 'E'))
        atmtab = np.loadtxt(atmfile, converters={2: D2E, 3: D2E, 4: D2E, 5: D2E, 6: D2E, 7: D2E, 8: D2E, 9: D2E, 10: D2E, 11: D2E})

        # Break up the table along axes and extract a single passband data:
        atmtab = np.reshape(atmtab, (Nmet, Npb, Nlogg, Nints, -1))
        atmtab = atmtab[:, wdidx, :, :, :]
        
        # Finally, reverse the metallicity axis because it is sorted in
        # reverse order in atmcof:
        self.extern_wd_atmx = atmtab[::-1, :, :, :]
        self.content.append('atmcof')

    #~ def _log10_Inorm_wd(self, Teff, logg, met):
        #~ met_nodes  = np.array([-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.3, -0.2, -0.1,  0.0,  0.1,  0.2,  0.3,  0.5, 1.0])
        #~ logg_nodes = np.array([ 0.0,  0.5,  1.0,  1.5,  2.0,  2.5,  3.0,  3.5,  4.0,  4.5,  5.0])
        #~ 
        #~ # Find the indices of the lower bracket elements:
        #~ met_idx = np.searchsorted(met_nodes, met)
        #~ logg_idx = np.searchsorted(logg_nodes, logg)
#~ 
        #~ print met, met_idx
        #~ print met_nodes[met_idx-1:met_idx+1]
        #~ print logg, logg_idx
        #~ print logg_nodes[logg_idx-1:logg_idx+1]
        #~ 
        #~ # Compute intensities for all combinations:
        #~ for m, l in [(m, l) for l in (logg_idx-1, logg_idx) for m in (met_idx-1, met_idx)]:
            #~ # Lower temperature interval index:
            #~ i = np.searchsorted(self.atmcof[m, l, :, 0], Teff)
            #~ coeffs = self.atmcof[m, l, i, 2:]
            #~ print self.atmcof[m, l, :, 0]
#~ 
        #~ print met_idx, logg_idx
        #~ print met_nodes[met_idx:met_idx+2]
        #~ print logg_nodes[logg_idx:logg_idx+2]
#~ 
        #~ return 10

    def _log10_Inorm_extern_planckint(self, Teff):
        """
        Internal function to compute normal passband intensities using
        the external WD machinery that employs blackbody approximation.
        
        @Teff: effective temperature in K
        
        Returns: log10(Inorm)
        """

        # atmcof.* accepts only floats, no arrays, so we need to check
        # and wrap if arrays are passed:
        if not hasattr(Teff, '__iter__'):
            log10_Inorm, _ = atmcof.planckint(Teff, self.extern_wd_idx)
        else:
            log10_Inorm = np.empty_like(Teff)
            for i, teff in enumerate(Teff):
                log10_Inorm[i], _ = atmcof.planckint(teff, self.extern_wd_idx)
                #~ print i, teff, log10_Inorm[i]

        return log10_Inorm

    def _log10_Inorm_extern_atmx(self, Teff, logg, met):
        """
        Internal function to compute normal passband intensities using
        the external WD machinery that employs model atmospheres and
        ramps.
        
        @Teff: effective temperature in K
        @logg: surface gravity in cgs
        @met:  metallicity in dex, Solar=0.0
        
        Returns: log10(Inorm)
        """

        # atmcof.* accepts only floats, no arrays, so we need to check
        # and wrap if arrays are passed:
        if not hasattr(Teff, '__iter__'):
            log10_Inorm, Inorm = atmcof.atmx(Teff, logg, met, self.extern_wd_idx)
        else:
            log10_Inorm = np.zeros(len(Teff))
            for i in range(len(Teff)):
                log10_Inorm[i], _ = atmcof.atmx(Teff[i], logg[i], met[i], self.extern_wd_idx)

        return log10_Inorm

    def _log10_Inorm_ck2004(self, Teff, logg, met):
        if not hasattr(Teff, '__iter__'):
            req = np.array(((Teff, logg, met),))
            log10_Inorm = interp.interp(req, self._ck2004_axes, self._ck2004_grid)[0][0]
        else:
            req = np.vstack((Teff, logg, met)).T
            log10_Inorm = interp.interp(req, self._ck2004_axes, self._ck2004_grid).T[0]

        return log10_Inorm


    def Inorm(self, Teff=5772., logg=4.43, met=0.0, atm='blackbody'):
        if atm == 'blackbody':
            return 10**self._log10_Inorm_bb(Teff)
        elif atm == 'extern_planckint':
            # The factor 0.1 is from erg/s/cm^3/sr -> W/m^3/sr:
            return 0.1*10**self._log10_Inorm_extern_planckint(Teff)
        elif atm == 'extern_atmx':
            # The factor 0.1 is from erg/s/cm^3/sr -> W/m^3/sr:
            return 0.1*10**self._log10_Inorm_extern_atmx(Teff, logg, met)
        elif atm == 'atmcof':
            return 10**self._log10_Inorm_wd(Teff, logg, met)
        elif atm == 'ck2004':
            return 10**self._log10_Inorm_ck2004(Teff, logg, met)
        else:
            print('received atm=%s' % atm)
            raise NotImplementedError

def init_passbands():
    """
    This function should be called only once, at import time. It
    traverses the passbands directory and builds a lookup table of
    passband names qualified as 'pbset:pbname' and corresponding files
    and atmosphere content within. Since planckint and atmx are made
    available via external calls to fortran sources rather than
    contained in the passband files themselves, we concatenate those
    options artificially if 'atmcof' is found in passband contents.
    """

    path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/passbands'))+'/'
    for f in os.listdir(path):
        pb = Passband.load(path+f)
        _pbtable[pb.pbset+':'+pb.pbname] = {'fname': path+f, 'atms': pb.content + ['extern_planckint', 'extern_atmx'] if 'atmcof' in pb.content else pb.content}


if __name__ == '__main__':
    # Constructing a passband:

    atmdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/wd'))
    atmcof.init(atmdir+'/atmcofplanck.dat', atmdir+'/atmcof.dat')

    jV = Passband('tables/ptf/JOHNSON.V', pbset='Johnson', pbname='V', effwl=5500.0, calibrated=True, wlunits=u.AA, reference='ADPS', version=1.0, comments='')
    jV.compute_blackbody_response()
    jV.compute_ck2004_response('tables/ck2004')
    jV.import_wd_atmcof(atmdir+'/atmcofplanck.dat', atmdir+'/atmcof.dat', 7)
    jV.save('tables/passbands/JOHNSON.V')
    
    pb = Passband('tables/ptf/KEPLER.PTF', pbset='Kepler', pbname='mean', effwl=5920.0, calibrated=True, wlunits=u.AA, reference='Bachtell & Peters (2008)', version=1.0, comments='')
    pb.compute_blackbody_response()
    pb.compute_ck2004_response('tables/ck2004')
    pb.save('tables/passbands/KEPLER.PTF')

    #~ jV = Passband.load('tables/passbands/JOHNSON.V')
    
    #~ teffs = np.arange(5000, 10001, 25)
    #~ req = np.vstack((teffs, 4.43*np.ones(len(teffs)), np.zeros(len(teffs)))).T

    #~ Teff_verts = axes[0][(axes[0] > 4999)&(axes[0]<10001)]
    #~ Inorm_verts1 = grid[(axes[0] >= 4999) & (axes[0] < 10001), axes[1] == 4.5, axes[2] == 0.0, 0]
    #~ Inorm_verts2 = grid[(axes[0] >= 4999) & (axes[0] < 10001), axes[1] == 4.0, axes[2] == 0.0, 0]

    #~ res = interp.interp(req, axes, grid)
    #~ print res.shape

    #~ import matplotlib.pyplot as plt
    #~ plt.plot(teffs, res, 'b-')
    #~ plt.plot(Teff_verts, Inorm_verts1, 'ro')
    #~ plt.plot(Teff_verts, Inorm_verts2, 'go')
    #~ plt.show()
    #~ exit()

    print 'blackbody:', jV.Inorm(Teff=5880., logg=4.43, met=0.0, atm='blackbody')
    print 'planckint:', jV.Inorm(Teff=5880., logg=4.43, met=0.0, atm='extern_planckint')
    print 'atmx:     ', jV.Inorm(Teff=5880., logg=4.43, met=0.0, atm='extern_atmx')
    print 'kurucz:   ', jV.Inorm(Teff=5880., logg=4.43, met=0.0, atm='ck2004')

    # Testing arrays:
    
    print 'blackbody:', jV.Inorm(Teff=np.array((5550., 5770., 5990.)), atm='blackbody')
    print 'planckint:', jV.Inorm(Teff=np.array((5550., 5770., 5990.)), atm='extern_planckint')
    print 'atmx:     ', jV.Inorm(Teff=np.array((5550., 5770., 5990.)), logg=np.array((4.40, 4.43, 4.46)), met=np.array((0.0, 0.0, 0.0)), atm='extern_atmx')
    print 'kurucz:   ', jV.Inorm(Teff=np.array((5550., 5770., 5990.)), logg=np.array((4.40, 4.43, 4.46)), met=np.array((0.0, 0.0, 0.0)), atm='kurucz')
