import numpy as np
from phoebe.dependencies.ligeor.utils.lcutils import *

class Model(object):

    def __init__(self, phases, fluxes, sigmas, filename, n_downsample, usecols, delimiter, phase_folded, period, t0):
        '''
        Computes a model light curve of the input data.

        Parameters
        ----------
        phases: array-like
            Input orbital phases, must be on the range [-0.5,0.5]
        fluxes: array-like
            Input fluxes
        sigmas: array-like
            Input sigmas (corresponding flux uncertainities)
        filename: str
            Filename from which to load a PHASE FOLDED light curve.
        n_downsample: int
            Number of data points to skip in loaded light curve (for downsampling)
        usecols: array-like, len 2 or 3
            Indices of the phases, fluxes and sigmas columns in file.
        '''

        if len(phases) == 0 or len(fluxes) == []:
            try:
                lc = load_lc(filename, n_downsample=n_downsample, phase_folded=phase_folded, usecols=usecols, delimiter=delimiter)
                if phase_folded:
                    self.phases = lc['phases']
                    self.fluxes = lc['fluxes']
                    self.sigmas = lc['sigmas']
                else:
                    self.phases, self.fluxes, self.sigmas = phase_fold(lc['times'], lc['fluxes'], lc['sigmas'], period=period, t0=t0)

            except Exception as e:
                raise ValueError(f'Loading light curve failed with exception {e}')
        else:
            self.filename = filename
            self.phases = phases 
            self.fluxes = fluxes 
            self.sigmas = sigmas


    def interactive_eclipse(self):
        '''
        Displays an interactive plot with draggable lines for the eclipse edges and positions.
        '''

        from ligeor.utils.interactive import DraggableLine
        phases_w, fluxes_w, sigmas_w = extend_phasefolded_lc(self.phases, self.fluxes, self.sigmas)
        phases_m, fluxes_m, sigmas_m = extend_phasefolded_lc(self.phases, self.model, np.nan*np.ones_like(self.model))
        [ecl1_l, ecl1_r, ecl2_l, ecl2_r] = self.eclipse_params['eclipse_edges']
        pos1, pos2 = self.eclipse_params['primary_position'], self.eclipse_params['secondary_position']
        
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.plot(phases_w, fluxes_w, 'k.')
        plt.plot(phases_m, fluxes_m, '-', label=self.best_fit['func'])
        lines = []
        lines.append(ax.axvline(x=pos1, c='#2B71B1', lw=2, label='primary'))
        lines.append(ax.axvline(x=pos2, c='#FF702F', lw=2, label='secondary'))
        lines.append(ax.axvline(x=ecl1_l, c='#2B71B1', lw=2, ls='--'))
        lines.append(ax.axvline(x=ecl1_r, c='#2B71B1', lw=2, ls='--'))
        lines.append(ax.axvline(x=ecl2_l, c='#FF702F', lw=2, ls='--'))
        lines.append(ax.axvline(x=ecl2_r, c='#FF702F', lw=2, ls='--'))
        drs = []
        for l,label in zip(lines,['pos1', 'pos2', 'ecl1_l', 'ecl1_r', 'ecl2_l', 'ecl2_r']):   
            dr = DraggableLine(l)
            dr.label = label
            dr.connect()   
            drs.append(dr) 
        ax.legend()
        plt.show(block=True)

        print('adjusting values')

        pos1 = drs[0].point.get_xdata()[0]
        pos2 = drs[1].point.get_xdata()[0]
        ecl1_l = drs[2].point.get_xdata()[0]
        ecl1_r = drs[3].point.get_xdata()[0]
        ecl2_l = drs[4].point.get_xdata()[0]
        ecl2_r = drs[5].point.get_xdata()[0]

        width1 = ecl1_r - ecl1_l
        width2 = ecl2_r - ecl2_l

        #TODO: replace max with mean out of eclipses (or at edges)
        depth1 = self.fluxes.max() - self.fluxes[np.argmin(np.abs(self.phases-pos1))]
        depth2 = self.fluxes.max() - self.fluxes[np.argmin(np.abs(self.phases-pos2))]
        
        eclipse_edges = [ecl1_l, ecl1_r, ecl2_l, ecl2_r]

        self.eclipse_params = {
            'primary_width': width1,
            'secondary_width': width2,
            'primary_position': pos1,
            'secondary_position': pos2,
            'primary_depth': depth1,
            'secondary_depth': depth2,
            'eclipse_edges': eclipse_edges
    }



    def check_eclipses_credibility(self):
        '''
        Checks if the detected eclipses are statistically significant.

        The checks performed are:
        - whether the eclipse is fitted to a noise feature. If True, the eclipse parameters are discarded.
        - whether the two detected eclipses overlap. If True, only the shallower eclipse is discarded.
        '''
        
        eclipse_params = self.eclipse_params.copy()
        if hasattr(self, 'eclipse_params'):

            # check secondary eclipse
            if ~np.isnan(eclipse_params['secondary_position']):
                if (
                check_eclipse_fitting_noise(self.model, self.fluxes, eclipse_params['secondary_depth']) or 
                check_eclipse_fitting_cosine(eclipse_params['secondary_width'])):
                    print('SECONDARY ECLIPSE FITTING NOISE OR COSINE')
                    eclipse_params['secondary_position'] = np.nan 
                    eclipse_params['secondary_width'] = np.nan 
                    eclipse_params['secondary_depth'] = np.nan
                    eclipse_params['eclipse_edges'][2] = np.nan
                    eclipse_params['eclipse_edges'][3] = np.nan

            # check primary eclipse
            if ~np.isnan(eclipse_params['primary_position']):
                if (
                check_eclipse_fitting_noise(self.model, self.fluxes, eclipse_params['primary_depth']) or 
                check_eclipse_fitting_cosine(eclipse_params['primary_width'])):
                    print('PRIMARY ECLIPSE FITTING NOISE OR COSINE')
                
                    if ~np.isnan(eclipse_params['secondary_position']):
                        print('replacing with secondary')
                        eclipse_params['primary_position'] = float(eclipse_params['secondary_position'])
                        eclipse_params['primary_width'] = float(eclipse_params['secondary_width'])
                        eclipse_params['primary_depth'] = float(eclipse_params['secondary_depth'])
                        eclipse_params['eclipse_edges'][0] = float(eclipse_params['eclipse_edges'][2])
                        eclipse_params['eclipse_edges'][1] = float(eclipse_params['eclipse_edges'][3])
                        eclipse_params['secondary_position'] = np.nan 
                        eclipse_params['secondary_width'] = np.nan 
                        eclipse_params['secondary_depth'] = np.nan
                        eclipse_params['eclipse_edges'][2] = np.nan
                        eclipse_params['eclipse_edges'][3] = np.nan
                    else:
                        print('replacing with nans!')
                        eclipse_params['primary_position'] = np.nan 
                        eclipse_params['primary_width'] = np.nan 
                        eclipse_params['primary_depth'] = np.nan
                        eclipse_params['eclipse_edges'][0] = np.nan
                        eclipse_params['eclipse_edges'][1] = np.nan
                    
            # check overlapping eclipses
            if ~np.isnan(eclipse_params['primary_position']) and ~np.isnan(eclipse_params['secondary_position']):
                pos_dist = np.abs(eclipse_params['primary_position'] - eclipse_params['secondary_position'])
                if  pos_dist < eclipse_params['primary_width'] or pos_dist < eclipse_params['secondary_width']:
                    # keep only the deeper eclipse
                    if eclipse_params['primary_depth'] > eclipse_params['secondary_depth']:
                        # keep only the primary eclipse
                        eclipse_params['secondary_position'] = np.nan 
                        eclipse_params['secondary_width'] = np.nan 
                        eclipse_params['secondary_depth'] = np.nan
                        eclipse_params['eclipse_edges'][2] = np.nan
                        eclipse_params['eclipse_edges'][3] = np.nan

                    else:
                        # keep the secondary as primary and discard secondary
                        eclipse_params['primary_position'] = float(eclipse_params['secondary_position'])
                        eclipse_params['primary_width'] = float(eclipse_params['secondary_width'])
                        eclipse_params['primary_depth'] = float(eclipse_params['secondary_depth'])
                        eclipse_params['eclipse_edges'][0] = float(eclipse_params['eclipse_edges'][2])
                        eclipse_params['eclipse_edges'][1] = float(eclipse_params['eclipse_edges'][3])
                        eclipse_params['secondary_position'] = np.nan 
                        eclipse_params['secondary_width'] = np.nan 
                        eclipse_params['secondary_depth'] = np.nan
                        eclipse_params['eclipse_edges'][2] = np.nan
                        eclipse_params['eclipse_edges'][3] = np.nan

        else:
            raise ValueError('Eclipse parameters not computed! Call self.compute_eclipse_params() first.')

        return eclipse_params

