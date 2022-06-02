import numpy as np
from operator import itemgetter
from itertools import groupby
from numpy.core.fromnumeric import mean
from scipy.optimize import minimize
from phoebe.dependencies.ligeor.utils.lcutils import *
from phoebe.dependencies.ligeor.models import Model


class Polyfit(Model):

    def __init__(self, filename='', phases=[], fluxes=[], sigmas=[], n_downsample=1, usecols=[0,1,2], delimiter=',',
                        phase_folded=True, period=1, t0=0,
                        xmin=-0.5, xmax=0.5, polyorder=2, chain_length=4):
        '''
        Initializes a polyfit with data and fit parameters.

        Parameters
        ----------
        filename: str
            Filename of light curve to load (if phases, fluxes and sigmas not provided)
        phases: array-like
            Input orbital phases, must be on the range [-0.5,0.5]
        fluxes: array-like
            Input fluxes
        sigmas: array-like
            Input sigmas (corresponding flux uncertainities)
        n_downsample: int
            Number of data points to skip if downsampling
        xmin: float
            Minimum orbital phase of the polyfit
        xmax: float
            Maximum orbital phase of the polyfit
        polyorder: int
            Polynomial order of each chain
        chain_length: int
            Length of the chain
        '''

        if polyorder != 2:
            raise NotImplemented('polyorder != 2 is not yet implemented.')
        if chain_length != 4:
            raise NotImplemented('chain_length != 4 is not yet implemented.')


        super(Polyfit, self).__init__(phases, fluxes, sigmas, filename, n_downsample, usecols, delimiter, phase_folded, period, t0)


        # for now, build a data attribute to match the Polychain methods
        self.data = np.array([self.phases, self.fluxes, self.sigmas]).T

        self.xmin = xmin
        self.xmax = xmax
        self.polyorder = polyorder
        self.chain_length = chain_length
        # self.sdata = self.data.copy()#[self.data[:,0].argsort()]
        self.sdata = self.data[self.data[:,0].argsort()]

        
    def _find_knots(self, min_chain_length=8, verbose=False):
        mean = self.sdata[:,1].mean()
        lt = np.where(self.sdata[:,1]<mean)[0]
        
        chains = []
        for k, g in groupby(enumerate(lt), lambda ix: ix[0]-ix[1]):
            chain = list(map(itemgetter(1), g))
            if len(chain) > min_chain_length:
                chains.append(chain)
        if verbose:
            print(f'{len(chains)} chains found.')
        if len(chains) < 2:
            self.knots = np.array((-0.4, -0.1, 0.1, 0.4))
        elif len(chains) == 2:
            self.knots = np.sort((self.sdata[chains[0][0],0], self.sdata[chains[0][-1],0], self.sdata[chains[1][0],0], self.sdata[chains[1][-1],0]))
        else:
            lengths = np.array([len(chain) for chain in chains])
            l = np.argsort(lengths)[::-1]
            self.knots = np.sort((self.sdata[chains[l[0]][0],0], self.sdata[chains[l[0]][-1],0], self.sdata[chains[l[1]][0],0], self.sdata[chains[l[1]][-1],0]))

        return self.knots
    
    def _find_segments(self, knots):
        self.sdata = self.data.copy()
        self.sdata[:,0][self.sdata[:,0] < knots[0]] += 1
        self.sdata = self.sdata[self.sdata[:,0].argsort()]

        segs = [np.argmax(self.sdata[:,0]>knot) for knot in knots[1:]] + [len(self.sdata)]
        return segs

    def _build_A_matrix(self, knots, segs):
        self.A = np.zeros(shape=(len(self.sdata), 8))

        # segment 1:
        self.A[:segs[0],0] = self.sdata[:segs[0],0]**2
        self.A[:segs[0],1] = self.sdata[:segs[0],0]
        self.A[:segs[0],2] = 1.0

        # segment 2:
        self.A[segs[0]:segs[1],0] = knots[1]**2
        self.A[segs[0]:segs[1],1] = knots[1]
        self.A[segs[0]:segs[1],2] = 1.0
        self.A[segs[0]:segs[1],3] = self.sdata[segs[0]:segs[1],0]**2-knots[1]**2
        self.A[segs[0]:segs[1],4] = self.sdata[segs[0]:segs[1],0]-knots[1]

        # segment 3:
        self.A[segs[1]:segs[2],0] = knots[1]**2
        self.A[segs[1]:segs[2],1] = knots[1]
        self.A[segs[1]:segs[2],2] = 1.0
        self.A[segs[1]:segs[2],3] = knots[2]**2-knots[1]**2
        self.A[segs[1]:segs[2],4] = knots[2]-knots[1]
        self.A[segs[1]:segs[2],5] = self.sdata[segs[1]:segs[2],0]**2-knots[2]**2
        self.A[segs[1]:segs[2],6] = self.sdata[segs[1]:segs[2],0]-knots[2]

        # segment 4:
        x = (self.sdata[segs[2]:segs[3],0]-knots[3])/(knots[3]-knots[0]-1)
        self.A[segs[2]:segs[3],0] = knots[1]**2 + x*(knots[1]**2-knots[0]**2)
        self.A[segs[2]:segs[3],1] = knots[1] + x*(knots[1]-knots[0])
        self.A[segs[2]:segs[3],2] = 1.0
        self.A[segs[2]:segs[3],3] = knots[2]**2-knots[1]**2 + x*(knots[2]**2-knots[1]**2)
        self.A[segs[2]:segs[3],4] = knots[2]-knots[1] + x*(knots[2]-knots[1])
        self.A[segs[2]:segs[3],5] = knots[3]**2-knots[2]**2 + x*(knots[3]**2-knots[2]**2)
        self.A[segs[2]:segs[3],6] = knots[3]-knots[2] + x*(knots[3]-knots[2])
        self.A[segs[2]:segs[3],7] = self.sdata[segs[2]:segs[3],0]**2-knots[3]**2 + x*((knots[0]+1)**2-knots[3]**2)

    def _fit_chain(self, knots, min_pts_per_segment=5, return_ck=False):
        # we need to sort the knots first because the minimizer can criss-cross them:
        knots = np.sort(knots)
        
        segs = self._find_segments(knots)
        if np.ediff1d(segs).min() < min_pts_per_segment or np.any(knots < self.xmin):
            if return_ck:
                return None, 1e10
            return 1e10

        self._build_A_matrix(knots, segs)
        ck, ssr, rank, svd = np.linalg.lstsq(self.A, self.sdata[:,1], rcond=None)

        # safety switch when things go wrong:
        if len(ssr) == 0:
            ssr = 1e10

        if return_ck:
            return ck, ssr
        # print(ck, ssr, rank, svd, knots, segs)
        return ssr
    
    def _chain_coeffs(self, ck, verbose=False):
        if ck is None:
            self.coeffs = None
            return
        
        c0 = ck[2]
        c1 = (ck[0]-ck[3])*self.knots[1]**2 + (ck[1]-ck[4])*self.knots[1] + c0
        c2 = (ck[3]-ck[5])*self.knots[2]**2 + (ck[4]-ck[6])*self.knots[2] + c1
        b3 = 1./(self.knots[3]-self.knots[0]-1)*(ck[0]*(self.knots[1]**2-self.knots[0]**2)+ck[3]*(self.knots[2]**2-self.knots[1]**2)+ck[5]*(self.knots[3]**2-self.knots[2]**2)+ck[7]*((self.knots[0]+1)**2-self.knots[3]**2)+ck[1]*(self.knots[1]-self.knots[0])+ck[4]*(self.knots[2]-self.knots[1])+ck[6]*(self.knots[3]-self.knots[2]))
        c3 = (ck[5]-ck[7])*self.knots[3]**2 + (ck[6]-b3)*self.knots[3] + c2

        if verbose:
            print(f'segment 1:\n {self.knots[0]} < x <= {self.knots[1]}, a0={ck[0]} b0={ck[1]} c0={c0}')
            print(f'segment 2:\n {self.knots[1]} < x <= {self.knots[2]}, a1={ck[3]} b1={ck[4]} c1={c1}')
            print(f'segment 3:\n {self.knots[2]} < x <= {self.knots[3]}, a2={ck[5]} b2={ck[6]} c2={c2}')
            print(f'segment 4:\n {self.knots[3]} < x <= {self.knots[0]+1}, a3={ck[7]} b3={b3} c3={c3}')

        self.coeffs = ((ck[0], ck[1], c0), (ck[3], ck[4], c1), (ck[5], ck[6], c2), (ck[7], b3, c3))
    
    def _chain_extremes(self):
        if self.coeffs is None:
            self.extremes = None
            return

        exts = [-c[1]/2/c[0] for c in self.coeffs]
        knots = np.concatenate((self.knots, [self.knots[0]+1]))
        for k in range(4):
            if exts[k] < knots[k] or exts[k] > knots[k+1]:
                exts[k] = np.nan
        self.extremes = np.array(exts)
    
    def _remap_1d(self, d, sort=True):
        while len(d[d<self.xmin]) > 0:
            d[d<self.xmin] += 1
        while len(d[d>self.xmax]) > 0:
            d[d>self.xmax] -= 1

        if sort:
            d.sort()

        return d

    def _remap(self, d, sort=True):
        ncols = 1 if len(d.shape) == 1 else d.shape[1]
        x = d if ncols == 1 else d[:,0]

        while len(x[x<self.xmin]) > 0:
            x[x<self.xmin] += 1
        while len(x[x>self.xmax]) > 0:
            x[x>self.xmax] -= 1

        if sort and ncols > 1:
            d = d[d[:,0].argsort()]
        else:
            d.sort()

        return d      
    
    def fit(self, min_chain_length=8, min_pts_per_segment=10, method='Nelder-Mead', knots=None, coeffs = None, verbose=False):
        if knots is None:
            self._find_knots(min_chain_length=min_chain_length, verbose=verbose)
        else:
            self.knots = knots
        
        if coeffs is None:
            solution = minimize(self._fit_chain, self.knots, args=(min_pts_per_segment,), method=method)
            self.knots = solution.x
            ck, self.ssr = self._fit_chain(self.knots, min_pts_per_segment=min_pts_per_segment, return_ck=True)
            self._chain_coeffs(ck)
        else:
            self.coeffs = coeffs

        self._chain_extremes()
        self.model = self.fv(x=self.phases)
        self.best_fit = {}
        self.best_fit['func'] = 'polyfit'
        self.best_fit['knots'] = self.knots
        self.best_fit['coeffs'] = self.coeffs
        self.best_fit['extremes'] = self.extremes
        
        return self.knots, self.coeffs, self.extremes
    
    def fv(self, x):
        if self.coeffs is None:
            return None
        x[x<self.knots[0]] += 1
        y = np.empty_like(x)
        for k in range(len(self.knots)-1):
            s = (x>=self.knots[k]) & (x<self.knots[k+1])
            y[s] = self.coeffs[k][0]*x[s]**2 + self.coeffs[k][1]*x[s] + self.coeffs[k][2]
        s = x>=self.knots[3]
        y[s] = self.coeffs[3][0]*x[s]**2 + self.coeffs[3][1]*x[s] + self.coeffs[3][2]

        return y
    
    def compute_model(self, phases, best_fit=True):
        if not best_fit:
            raise NotImplemented('Polyfit cannot compute a custom model yet, run .fit() first and pass best_fit=True.')
        return self.fv(phases)


    def plot(self, x, savefig=None, show=True):
        import matplotlib.pyplot as plt

        if self.coeffs is None:
            return None

        if x is None:
            x = self.sdata[:,0]

        knot_fvs = self.fv(self.knots.copy())
        knots = self._remap_1d(self.knots.copy(), sort=False)

        y = self.fv(x)
        d = np.vstack((x, y)).T

        self._remap(self.sdata)
        d = self._remap(d)
        self._remap_1d(self.extremes)

        # print(knots)
        # print(knot_fvs)

        plt.plot(self.phases, self.fluxes, 'b.')
        plt.plot(knots, knot_fvs, 'ms')
        plt.plot(d[:,0], d[:,1], 'r-')

        for k in range(4):
            plt.axvline(self.extremes[k], ls='--')
        
        if savefig is not None:
            plt.savefig(savefig)

        if show:
            plt.show()
        else:
            plt.clf()


    def compute_eclipse_params(self, interactive=False):
        '''
        Compute the positions, widths and depths of the eclipses 
        based on the polyfit solution.

        The eclipse parameters are computed as following:
        - eclipses are first identified as the chains corresponding to the
          two deepest minima
        - eclipse positions are set to the minima positions
        - eclipse widths are the difference between the chain knots 
        - eclipse depths are the difference between the mean function values at the
          knots and function values at the minima.

        Returns
        -------
        results: dict
            A dictionary of the eclipse paramter values.
        '''
        ext_x = self.extremes
        ext_y = self.fv(x=ext_x)
        knots_x = self.knots
        knots_extended = np.hstack((self.knots-1, self.knots, self.knots+1))

        xs_all = np.hstack((np.array(ext_x), np.array(knots_x))).flatten()
        ys_all = self.fv(x=xs_all)
        ys_labels = np.array(['ext', 'ext', 'ext', 'ext', 'knot', 'knot', 'knot', 'knot'])
        sort = np.argsort(ys_all)
        if ys_labels[sort][0] == 'knot' and ys_labels[sort][1] == 'ext':
            # no primary eclipse, just log details of secondary
            primary_pos = xs_all[sort][0]
            eclipse_arg = np.argwhere(ext_x == xs_all[sort][1]).flatten()[0]
            knots_ecl = np.array([knots_extended[eclipse_arg+4], knots_extended[eclipse_arg+5]])
            mean_outofecl = self.fv(knots_ecl).mean()
            
            self.eclipse_params = {
            'primary_width': np.nan,
            'secondary_width': np.abs(knots_ecl[1]-knots_ecl[0]),
            'primary_position': np.nan,
            'secondary_position': ext_x[eclipse_arg],
            'primary_depth': np.nan,
            'secondary_depth': mean_outofecl - ext_y[eclipse_arg],
            'eclipse_edges': np.hstack((np.array([np.nan,np.nan]), knots_ecl)),
            'eclipse_coeffs': [np.nan*np.ones((3)), self.coeffs[eclipse_arg]]
        }
            
        if ys_labels[sort][0] == 'ext' and ys_labels[sort][1] == 'knot':
                # no primary eclipse, just log details of secondary
            secondary_pos = xs_all[sort][1]
            eclipse_arg = np.argwhere(ext_x == xs_all[sort][0]).flatten()[0]
            knots_ecl = np.array([knots_extended[eclipse_arg+4], knots_extended[eclipse_arg+5]])
            mean_outofecl = self.fv(knots_ecl).mean()
            
            self.eclipse_params = {
            'primary_width': np.abs(knots_ecl[1]-knots_ecl[0]),
            'secondary_width': np.nan,
            'primary_position': ext_x[eclipse_arg],
            'secondary_position': np.nan,
            'primary_depth': mean_outofecl - ext_y[eclipse_arg],
            'secondary_depth': np.nan,
            'eclipse_edges': np.hstack((knots_ecl, np.array([np.nan,np.nan]))),
            'eclipse_coeffs': [self.coeffs[eclipse_arg], np.nan*np.ones((3))]
        }
        
        
        if ys_labels[sort][0] == 'ext' and ys_labels[sort][1] == 'ext':
            # fv for x=np.nan doesn't return nan!
            #TODO: troubleshoot .fv for a discrete set of values and why np.nan results in a value
            ext_y[np.isnan(self.extremes)] = np.nan
            eclipse_args = np.argsort(ext_y)[:2]

            # let's extend the knots array left and right so it's continuous (for computing the width)
            knots1 = np.array([knots_extended[eclipse_args[0]+4], knots_extended[eclipse_args[0]+5]])
            knots2 = np.array([knots_extended[eclipse_args[1]+4], knots_extended[eclipse_args[1]+5]])

            mean_outofecl = self.fv(np.hstack((knots1, knots2))).mean()

            self.eclipse_params = {
                'primary_width': np.abs(knots1[1]-knots1[0]),
                'secondary_width': np.abs(knots2[1]-knots2[0]),
                'primary_position': ext_x[eclipse_args[0]],
                'secondary_position': ext_x[eclipse_args[1]],
                'primary_depth': mean_outofecl - ext_y[eclipse_args[0]],
                'secondary_depth': mean_outofecl - ext_y[eclipse_args[1]],
                'eclipse_edges': np.hstack((knots1, knots2)),
                'eclipse_coeffs': [self.coeffs[eclipse_args[0]], self.coeffs[eclipse_args[1]]]
            }
        
        if ys_labels[sort][0] == 'knot' and ys_labels[sort][1] == 'knot':
                # no primary eclipse, just log details of secondary
            primary_pos = xs_all[sort][0]
            secondary_pos = xs_all[sort][1]

            self.eclipse_params = {
            'primary_width': np.nan,
            'secondary_width': np.nan,
            'primary_position': primary_pos,
            'secondary_position': secondary_pos,
            'primary_depth': np.nan,
            'secondary_depth': np.nan,
            'eclipse_edges': np.array([np.nan, np.nan, np.nan, np.nan]),
            'eclipse_coeffs': [np.nan*np.ones((3)), np.nan*np.ones((3))]
        }


        if interactive:
           self.interactive_eclipse()


        self.compute_eclipse_area(ecl=1)
        self.compute_eclipse_area(ecl=2)

        # self.eclipse_params = self.check_eclipses_credibility()

        # check if eclipses need to be swapped:
        if ~np.isnan(self.eclipse_params['secondary_depth']) and ~np.isnan(self.eclipse_params['primary_depth']): 

            if self.eclipse_params['secondary_depth'] > self.eclipse_params['primary_depth']:
                # the secondary is deeper than the primary, so we swap them
                pos1, d1, w1, edge1 = self.eclipse_params['secondary_position'], self.eclipse_params['secondary_depth'], self.eclipse_params['secondary_width'], self.eclipse_params['eclipse_edges'][2:]
                pos2, d2, w2, edge2 = self.eclipse_params['primary_position'], self.eclipse_params['primary_depth'], self.eclipse_params['primary_width'], self.eclipse_params['eclipse_edges'][:2]

                self.eclipse_params['primary_position'] = pos1 
                self.eclipse_params['primary_width'] = w1
                self.eclipse_params['primary_depth'] = d1
                self.eclipse_params['secondary_position'] = pos2 
                self.eclipse_params['secondary_width'] = w2
                self.eclipse_params['secondary_depth'] = d2
                self.eclipse_params['eclipse_edges'] = [edge1[0],edge1[1], edge2[0], edge2[1]]

        elif ~np.isnan(self.eclipse_params['secondary_depth']) and np.isnan(self.eclipse_params['primary_depth']):
            # there is only one eclipse and it's fitted by the "secondary", so we need to move it to primary
            pos1, d1, w1, edge1 = self.eclipse_params['secondary_position'], self.eclipse_params['secondary_depth'], self.eclipse_params['secondary_width'], self.eclipse_params['eclipse_edges'][2:]

            self.eclipse_params['primary_position'] = pos1 
            self.eclipse_params['primary_width'] = w1
            self.eclipse_params['primary_depth'] = d1
            self.eclipse_params['secondary_position'] = np.nan
            self.eclipse_params['secondary_width'] = np.nan
            self.eclipse_params['secondary_depth'] = np.nan
            self.eclipse_params['eclipse_edges'] = [edge1[0],edge1[1], np.nan, np.nan]

        return self.eclipse_params
        

    def compute_eclipse_area(self, ecl=1):
        '''
        Computes the area under an eclipse.

        An eclipse is defined as being positioned between the knots around the deepest minima.

        Parameters
        ----------
        ecl: int
            The eclipse whose area is to be computed (1 or 2)
        
        Returns
        -------
        eclipse_area: float
            The computed area under the chosen eclipse.
        '''

        if hasattr(self, 'eclipse_area'):
            pass
        else:
            self.eclipse_area = {}

        coeffs = self.eclipse_params['eclipse_coeffs'][ecl-1]
        if ecl == 1:
            edges = [self.eclipse_params['eclipse_edges'][0], self.eclipse_params['eclipse_edges'][1]]
        else:
            edges = [self.eclipse_params['eclipse_edges'][2], self.eclipse_params['eclipse_edges'][3]]

        self.eclipse_area[ecl] = coeffs[0]/3*(edges[1]**3-edges[0]**3) + coeffs[1]/2*(edges[1]**2-edges[0]**2) + coeffs[2]*(edges[1]-edges[0])
