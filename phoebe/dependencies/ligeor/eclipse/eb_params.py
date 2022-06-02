import numpy as np
from numpy import sin, cos, pi
import warnings
from scipy.optimize import newton, minimize

class EbParams(object):

    def __init__(self, eclipse_params, fit_eclipses=True, **kwargs):

        '''
        Computes estimates of ecc, w, rsum and teffratio based on the eclipse parameters.

        Parameters
        ----------
        eclipse_params: dict
            Dictionary of the eclipse parameters determined from the two-Gaussian model or manually. 
            Expects the following keys: primary_position, secondary_position, primary_width, secondary_width,
            primary_depth, secondary_depth, and (optional) eclipse_edges
        refine_with_ellc: bool
            If true, an ellc.lc model will be fitted to the eclipses only to further refine 
            rsum, teffratio, as well as rratio and incl.

        '''
        if not isinstance(eclipse_params, dict):
            raise TypeError('eclipse_params should be a dictionary with the following keys: \
                primary_position, secondary_position, primary_width, secondary_width, primary_depth, secondary_depth, \
                and eclipse_edges. Pass the correct type of calculate it first from a model with .compute_eclipse_params()')
        
        else:
            self.pos1 = eclipse_params['primary_position']
            self.pos2 = eclipse_params['secondary_position']
            self.width1 = eclipse_params['primary_width']
            self.width2 = eclipse_params['secondary_width']
            self.depth1 = eclipse_params['primary_depth']
            self.depth2 = eclipse_params['secondary_depth']
            self.edges = eclipse_params['eclipse_edges']
            # computation fails if sep<0, so we need to adjust for it here.
            sep = self.pos2 - self.pos1
            if sep < 0:
                self.sep = 1+sep
            else:
                self.sep = sep

            self._ecc_w()
            self._teffratio()
            self._rsum()

            if fit_eclipses:
                phases = kwargs.get('phases', [])
                fluxes = kwargs.get('fluxes', [])
                sigmas = kwargs.get('sigmas', [])

                if len(phases) == 0 or len(fluxes) == 0 or len(sigmas) == 0:
                    raise ValueError('Please provide values for the phases, fluxes and sigmas of the light curve!')

                self.refine_with_ellc(phases, fluxes, sigmas)
            else:
                self.rratio = 1.
                self.incl = 90.


    @staticmethod
    def _f (psi, sep): # used in pf_ecc_psi_w
        '''Returns the function to minimize for Psi'''
        return psi - sin(psi) - 2*pi*sep

    @staticmethod
    def _df (psi, sep): # used in pf_ecc_psi_w
        '''Returns the derivative of f for minimization'''
        return 1 - cos(psi) +1e-6

    def _ecc_w(self):
        '''Computes eccentricity and argument of periastron from the separation and widths.'''

        if np.isnan(self.sep) or np.isnan(self.width1) or np.isnan(self.width2):
            warnings.warn('Cannot esimate eccentricty and argument of periastron: incomplete geometry information')
            self.ecc = 0
            self.per0 = np.pi/2
            self.esinw = 0
            self.ecosw = 0
            
        else:
            psi = newton(func=self._f, x0=(12*np.pi*self.sep)**(1./3), fprime=self._df, args=(self.sep,), maxiter=5000)
            # ecc = sqrt( (0.25*(tan(psi-pi))**2+(swidth-pwidth)**2/(swidth+pwidth)**2)/(1+0.25*(tan(psi-pi))**2) )
            ecc = (np.sin(0.5*(psi-np.pi))**2+((self.width2-self.width1)/(self.width2+self.width1))**2*np.cos(0.5*(psi-np.pi))**2)**0.5
            try:
                w1 = np.arcsin((self.width1-self.width1)/(self.width2+self.width1)/ecc)
                w2 = np.arccos((1-ecc**2)**0.5/ecc * np.tan(0.5*(psi-np.pi)))

                w = w2 if w1 >= 0 else 2*pi-w2
            except:
                w = pi/2

            self.ecc = ecc
            self.per0 = w
            self.esinw = ecc*np.sin(w)
            self.ecosw = ecc*np.cos(w)


    def _t0_from_geometry(self, times, period=1, t0_supconj = 0, t0_near_times = True):
        '''
        Computes a new value for t0 from the position of the primary eclipse.

        Parameters
        ----------
        times: array-like
            Array of observed times
        period: float
            Orbital period of the object
        t0_supconj: float
            Initial t0 value (before fitting), if available. Default is 0.
        t0_near_times: bool
            If True, the computed t0 will be shifted to fall within the range of observed times.
        '''

        delta_t0 = self.pos1*period
        t0 = t0_supconj + delta_t0

        if t0_near_times:
            if t0 >= times.min() and t0 <= times.max():
                return t0
            else:
                return t0 + int((times.min()/period)+1)*(period)
        else:
            return t0
        

    def _teffratio(self):
        '''
        Computes the temprature ratio from eclipse depths.

        Holds only under the assumption of ecc=0, but it's the best first analytical guess we can get.
        '''
        self.teffratio = (self.depth2/self.depth1)**0.25

    
    def _rsum(self):
        '''
        Computes the sum of fractional radii from the eclipse widths, eccentricity and argument of periastron.

        The full equation for the eclipse widths contains a factor that depends on cos(incl) and requivratio.
        If we assume incl=90, that factor is 0 and the remaining is what we use to derive the equations for
        requivsum based on the widths of the primary and secondary eclipse below.
        '''
        rsum1 = np.pi*self.width1*(1-self.ecc**2)/(1+self.ecc*np.sin(self.per0))
        rsum2 = np.pi*self.width2*(1-self.ecc**2)/(1-self.ecc*np.sin(self.per0))
        self.rsum = np.nanmean([rsum1, rsum2])


    def refine_with_ellc(self, phases, fluxes, sigmas):
        '''
        Refines the eclipse fits with an ellc light curve.

        Parameters
        ----------
        phases: array-like
            Orbital phases of the observed light curve
        fluxes: array-like
            Observed fluxes
        sigmas: array-like
            Flux uncertainities
        '''
        try:
            import ellc
        except:
            raise ImportError('ellc is required for parameter refinement, please install it before running this step.')
        
        def wrap_around_05(phases):
            phases[phases>0.5] = phases[phases>0.5] - 1
            phases[phases<-0.5] = phases[phases<-0.5] + 1
            return phases

        def mask_eclipses():

            edges = wrap_around_05(np.array(self.edges))
            ecl1 = edges[:2]
            ecl2 = edges[2:]

            if ecl1[1]>ecl1[0]:
                mask1 = (phases>=ecl1[0]) & (phases<=ecl1[1])
            else:
                mask1 = (phases>=ecl1[0]) | (phases<=ecl1[1])

            if ecl2[1]>ecl2[0]:
                mask2 = (phases>=ecl2[0]) & (phases<=ecl2[1])
            else:
                mask2 = (phases>=ecl2[0]) | (phases<=ecl2[1])

            phases_ecl, fluxes_ecl, sigmas_ecl = phases[mask1 | mask2], fluxes[mask1 | mask2], sigmas[mask1 | mask2]
            phases_outofecl, fluxes_outofecl, sigmas_outofecl = phases[~(mask1 | mask2)], fluxes[~(mask1 | mask2)], sigmas[~(mask1 | mask2)]

            meanf = np.mean(fluxes_outofecl) - 1
            return phases_ecl, fluxes_ecl, sigmas_ecl, meanf


        def lc_model(phases_mask, rsum, rratio, teffratio, incl, meanf):

            r1 = rsum/(1+rratio)
            r2 = rsum*rratio/(1+rratio)
            sbratio = np.sign(teffratio) * teffratio**4

            return ellc.lc(phases_mask, r1, r2, sbratio, incl, 
                light_3 = 0, 
                t_zero = self.pos1, period = 1,
                q = 1,
                f_c = self.ecc**0.5*np.cos(self.per0), f_s = self.ecc**0.5*np.sin(self.per0),
                shape_1='roche', shape_2='roche', 
                ld_1='lin', ld_2='lin', ldc_1=0.5, ldc_2=0.5,
                gdc_1=0., gdc_2=0., heat_1=0., heat_2=0.) + meanf

        def chi2(params, lc_data, meanf):
            rsum, rratio, teffratio, incl = params
            try:
                lcm = lc_model(lc_data[:,0], rsum, rratio, teffratio, incl, meanf)
                return 0.5 * np.sum((lc_data[:,1] - lcm) ** 2 / lc_data[:,2]**2)
            except Exception as e:
                return np.inf


        phases_mask, fluxes_mask, sigmas_mask, meanf = mask_eclipses()
        # phases_mask, fluxes_mask, sigmas_mask = phases, fluxes, sigmas
        lc_data = np.array([phases_mask, fluxes_mask, sigmas_mask]).T
        rsum_0 = self.rsum
        rratio_0 = 1.0
        teffratio_0 = self.teffratio 
        incl_0 = 90. 
        params_0 = [rsum_0, rratio_0, teffratio_0, incl_0]

        res = minimize(chi2, params_0, args=(lc_data, meanf), method='nelder-mead', options={'maxiter':10000})

        [self.rsum, self.rratio, self.teffratio, self.incl] = res.x