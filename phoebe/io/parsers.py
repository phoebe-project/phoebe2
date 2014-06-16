"""
Parse data and parameter files.

.. autosummary::

    legacy_to_phoebe
    
"""

import re
import os
import logging
import glob
import numpy as np
import phoebe
from phoebe.parameters import parameters
from phoebe.parameters import tools
from phoebe.parameters import datasets
from phoebe.backend import universe
from phoebe.backend import observatory
from phoebe.algorithms import marching
from phoebe.wd import wd
from phoebe.parameters import datasets
from phoebe.units import conversions
import matplotlib.pyplot as plt
import os.path

logger = logging.getLogger('PARSER')
logger.addHandler(logging.NullHandler())

def try_to_locate_file(filename, rootfile=None):
    """
    Check if a file is available.
    """
    found = False
    if os.path.isfile(filename):
        found = filename
    elif os.path.isfile(os.path.basename(filename)):
        found = os.path.basename(filename)
    else:
        rootdir = os.path.dirname(os.path.abspath(rootfile))
        if os.path.isfile(os.path.join(rootdir, filename)):
            found = os.path.join(rootdir, filename)
    
    return found

def translate_filter(val, separate):
    if val[2:5] == 'Bes':
        val = val[2:-2].upper()
        val = ".".join(val.split('L IR:'))
        if val[0:8] == 'BESSEL.L':
            val = 'BESSEL.LPRIME'
    elif separate:
        val = ".".join(val.split(separate.group(0)))
        val = val[2:-1].upper()
        if val == 'COROT.SISMO':
            val = 'COROT.SIS'
        elif val == 'KEPLER.MEAN':
            val = 'KEPLER.MEAN'
        elif val == 'IRAC.CH1':
            val = 'IRAC.36'
        elif val == 'MOST.DEFAULT':
            val = 'MOST.V'
        elif val == 'STROMGREN.HBETA_NARROW':
            val = 'STROMGREN.HBN'
        elif val == 'STROMGREN.HBETA_WIDE':
            val = 'STROMGREN.HBW'
        elif val == 'BOLOMETRIC.3000A-10000A':
            val = 'OPEN.BOL'
        elif val == 'HIPPARCOS.BT':
            val = 'TYCHO.BT'
        elif val == 'HIPPARCOS.VT':
            val = 'TYCHO.VT'
        elif val[0:5] == 'SLOAN':
            val = "SDSS".join(val.split('SLOAN'))
            val = val[:-1]
    return val

def legacy_to_phoebe2(inputfile):
    
    #-- initialise the orbital and component parameter sets. 
    orbit = parameters.ParameterSet('orbit', c1label='primary', c2label='secondary',
                                    label='new_system', t0type='superior_conjunction')
    comps = [parameters.ParameterSet('component', ld_coeffs=[0.5,0.5], label='primary'),
             parameters.ParameterSet('component', ld_coeffs=[0.5,0.5], label='secondary')]
    position = parameters.ParameterSet('position', distance=(1.,'Rsol'))
    compute = parameters.ParameterSet('compute', beaming_alg='none', refl=False,
                                      heating=True, label='legacy',
                                      eclipse_alg='binary', subdiv_num=3)
    meshes = [parameters.ParameterSet('mesh:marching'),
              parameters.ParameterSet('mesh:marching')]
    all_rvobs = []
    all_lcobs = []
    all_lcdeps = [[],[]]
    all_rvdeps = [[],[]]
    
    translation = dict(dpdt='dpdt', filename='filename', grb='gravb', sma='sma',
                       ecc='ecc', period='period', incl='incl', pshift='phshift',
                       rm='q', perr0='per0', met='abun', pot='pot', teff='teff',
                       alb='alb', f='syncpar', vga='vgamma', hjd0='t0',
                       dperdt='dperdt', extinction='extinction')
    
    legacy_units = dict(per0='rad', period='d', incl='deg', dpdt='d/d',
                        dperdt='rad/d', sma='Rsun', vgamma='km/s', teff='K')
    
    computehla = False
    usecla = False
    computevga = False
    
    # Open the parameter file and read each line
    with open(inputfile,'r') as ff:
        while True:
            this_set = None
            
            l = ff.readline()
            if not l:
                break
            
            # Strip whitespace
            l = l.strip()
            
            # Empty line
            if not l:
                continue
            
            #-- ignore initial comment
            if l[0]=='#':
                continue
            
            # try splitting the parameter name and value
            try: 
                key, val = l.split('=')
            except:
                logger.error("line " + l[:-1] + " could not be parsed ('{}' probably not Phoebe legacy file)".format(inputfile))
                raise IOError("Cannot parse phoebe file '{}': line '{}'".format(inputfile, l[:-1]))
             
            key = key.strip()

			# Start with parameters that determine container sizes:
            if key == 'phoebe_lcno':
                all_lcobs = [datasets.LCDataSet(user_columns=['time','flux','sigma'], user_units=dict(time='JD',flux='W/m3',sigma='W/m3')) for i in range(int(val))]
                all_lcdeps[0] = [parameters.ParameterSet('lcdep', ld_coeffs=[0.5,0.5]) for i in range(int(val))]
                all_lcdeps[1] = [parameters.ParameterSet('lcdep', ld_coeffs=[0.5,0.5]) for i in range(int(val))]
                
                # We need to add extinction parameters to all lcdeps
                ext_par = parameters.Parameter('extinction', context='lcdep', cast_type=float)
                for i in range(int(val)):
                    all_lcdeps[0][i].add(ext_par.copy())
                    all_lcdeps[1][i].add(ext_par.copy())
                continue            

            if key == 'phoebe_rvno':
                all_rvobs = [datasets.RVDataSet(user_columns=['time','rv','sigma'], user_units=dict(time='JD',rv='km/s',sigma='km/s')) for i in range(int(val))]
                rv_component_spec = [None for i in range(int(val))]
                all_rvdeps[0] = [parameters.ParameterSet('rvdep', ld_coeffs=[0.5,0.5]) for i in range(int(val))]
                all_rvdeps[1] = [parameters.ParameterSet('rvdep', ld_coeffs=[0.5,0.5]) for i in range(int(val))]
                continue

            #-- if this is an rvdep or lcdep, check which index it has
            # and remove it from the string:
            pattern = re.search('\[(\d)\]', key)
            separate = re.search('\:', val)
                        
            # this is a light or radial velocity file?
            if pattern:
                index = int(pattern.group(1))-1
                key = "".join(key.split(pattern.group(0)))
            else:
                index = None
            
            key_split = key.split('_')
            
            # Ignore GUI-, WD-, and DC-specific settings:
            if key_split[0] in ['gui', 'wd', 'dc']:
                continue
                        
            # get the true qualifier, split in qualifier and postfix (if any)
            leg_qualifier_split = "_".join(key_split[1:]).split('.')
            leg_qualifier = leg_qualifier_split[0]
            
            if len(leg_qualifier_split) == 2:
                postfix = leg_qualifier_split[1]
            elif len(leg_qualifier_split) == 1:
                postfix = None
            else:
                raise ValueError("Exception invoked in parsers.py, please report this.")
            
            # Parse component number (if applicable):
            if leg_qualifier[-1].isdigit() and ( int(leg_qualifier[-1]) == 1 or int(leg_qualifier[-1]) == 2 ):
                compno = int(leg_qualifier[-1])-1
                leg_qualifier = leg_qualifier[:-1]
            else:
                compno = None
                        
            # Ignore obsolete or inapplicable qualifiers:
            if leg_qualifier in [
				'opsf', 'spots_no', 'spno', 'logg', 'sbr', 'mass', 'radius',
                'plum', 'mbol', 'indep', 'spots_corotate', 'spots_units',
                'synscatter_seed', 'synscatter_switch', 'synscatter_sigma',
                'synscatter_levweight',
                'nms_accuracy', 'nms_iters_max', 'el3_units', 'grid_coarsesize',
                'passband_mode', 'ie_switch', 'spectra_disptype', 'bins_switch',
                'bins']:
                continue
            if leg_qualifier[:3] == 'dc_':
                continue

#           print "qualifier:", leg_qualifier, "postfix:", postfix, "compno:", compno

            # special cases
            if leg_qualifier == 'name':
                val = val.strip()[1:-1]
                if val:
                    orbit['label'] = val
                continue
            if leg_qualifier == 'reffect_switch':
                compute['irradiation_alg'] == 'full' if int(val) else 'point_source'
                continue
            if leg_qualifier == 'reffect_reflections':
                compute['refl_num'] = 1#val
                continue
            if leg_qualifier == 'grid_finesize':
                meshes[compno]['delta'] = marching.gridsize_to_delta(int(val))
                continue
            if leg_qualifier == 'usecla_switch':
                usecla = int(val)
                continue
            if leg_qualifier == 'compute_vga_switch':
                computevga = int(val)
                continue
            if leg_qualifier == 'compute_hla_switch':
                computehla = int(val)
                continue
            if leg_qualifier == 'proximity_rv1_switch':
                proximity_rv1 = int(val)
                continue
            if leg_qualifier == 'proximity_rv2_switch':
                proximity_rv2 = int(val)
                continue
            
            # passband luminosities
            if leg_qualifier in 'cla':
                if postfix == 'VAL':
                    all_lcdeps[1][index]['pblum'] = val
                    continue
                else:
                    for ii in all_lcdeps[1]:
                        this_param = ii.get_parameter('pblum')
                        if postfix == 'ADJ':
                            this_param.set_adjust(int(val))
                        elif postfix == 'STEP':
                            this_param.set_step(float(val))
                        elif postfix == 'MIN':
                            this_param.set_limits(llim=float(val))
                        elif postfix == 'MAX':
                            this_param.set_limits(ulim=float(val))
                    continue
            if leg_qualifier in 'hla':
                if postfix == 'VAL':
                    all_lcdeps[0][index]['pblum'] = val
                    continue
                else:
                    for ii in all_lcdeps[0]:
                        this_param = ii.get_parameter('pblum')
                        if postfix == 'ADJ':
                            this_param.set_adjust(int(val))
                        elif postfix == 'STEP':
                            this_param.set_step(float(val))
                        elif postfix == 'MIN':
                            this_param.set_limits(llim=float(val))
                        elif postfix == 'MAX':
                            this_param.set_limits(ulim=float(val))
                    continue
            
            if leg_qualifier[:5] == 'ld_lc':
                if leg_qualifier[-1] == 'x':
                    ldindex = 0
                else:
                    ldindex = 1
                    
                if postfix == 'VAL' or postfix is None:
                    all_lcdeps[compno][index]['ld_coeffs'][ldindex] = float(val)
                    continue
                else:
                    for ii in all_lcdeps[compno]:
                        this_param = ii.get_parameter('pblum')
                        if postfix == 'ADJ':
                            this_param.set_adjust(int(val))
                        elif postfix == 'STEP':
                            this_param.set_step(float(val))
                        elif postfix == 'MIN':
                            this_param.set_limits(llim=float(val))
                        elif postfix == 'MAX':
                            this_param.set_limits(ulim=float(val))
                    continue
            
            if leg_qualifier[:5] == 'ld_rv':
                if leg_qualifier[-1] == 'x':
                    ldindex = 0
                else:
                    ldindex = 1
                    
                if postfix == 'VAL' or postfix is None:
                    all_rvdeps[compno][index]['ld_coeffs'][ldindex] = float(val)
                    continue
                else:
                    for ii in all_rvdeps[compno]:
                        this_param = ii.get_parameter('pblum')
                        if postfix == 'ADJ':
                            this_param.set_adjust(int(val))
                        elif postfix == 'STEP':
                            this_param.set_step(float(val))
                        elif postfix == 'MIN':
                            this_param.set_limits(llim=float(val))
                        elif postfix == 'MAX':
                            this_param.set_limits(ulim=float(val))
                    continue
            
            
            if leg_qualifier == 'el3':
                index = 0
                if postfix == 'VAL':
                    all_lcdeps[0][index]['l3'] = val
                    continue
                else:
                    for ii in all_lcdeps[0]:
                        this_param = ii.get_parameter('l3')
                        if postfix == 'ADJ':
                            this_param.set_adjust(int(val))
                        elif postfix == 'STEP':
                            this_param.set_step(float(val))
                        elif postfix == 'MIN':
                            this_param.set_limits(llim=float(val))
                        elif postfix == 'MAX':
                            this_param.set_limits(ulim=float(val))
                    continue

            if leg_qualifier == 'extinction':
                index = 0
                if postfix == 'VAL':
                    all_lcdeps[0][index]['extinction'] = val
                    continue
                else:
                    for ii in all_lcdeps[0]:
                        this_param = ii.get_parameter('extinction')
                        if postfix == 'ADJ':
                            this_param.set_adjust(int(val))
                        elif postfix == 'STEP':
                            this_param.set_step(float(val))
                        elif postfix == 'MIN':
                            this_param.set_limits(llim=float(val))
                        elif postfix == 'MAX':
                            this_param.set_limits(ulim=float(val))
                    continue
            
            
            if compno is not None:
                if leg_qualifier == 'ld_xbol':
                    comps[compno]['ld_coeffs'][0] = float(val)
                    continue
                elif leg_qualifier == 'ld_ybol':
                    comps[compno]['ld_coeffs'][1] = float(val)
                    continue
                
                # otherwise remember
                this_set = comps[compno]
            
            # For datasets and pbdeps
            if index is not None:
                if '_lc_' in leg_qualifier:
                    this_obs = all_lcobs[index]
                    this_dep = all_lcdeps[0][index]
                    ll = leg_qualifier.split('_')
                    datatype, leg_qualifier = ll[0], '_'.join(ll[1:])
                
                elif '_rv_' in leg_qualifier:
                    this_obs = all_rvobs[index]
                    this_dep = all_rvdeps[0][index]
                    ll = leg_qualifier.split('_')
                    datatype, leg_qualifier = ll[0], '_'.join(ll[1:])
                
                    

            else:
                this_obs = []
                this_dep = []
            
            
            # translation:
            if leg_qualifier in translation:
                qualifier = translation[leg_qualifier]
                
                if this_set is not None:
                    pass
                elif qualifier in orbit:
                    this_set = orbit
                elif qualifier in position:
                    this_set = position
                elif qualifier in this_obs:
                    this_set = this_obs
                elif qualifier in this_dep:
                    this_set = this_dep
                else:
                    print('error {} {} {} {} {}'.format(qualifier, leg_qualifier, index, compno, val))
                
                this_param = this_set.get_parameter(qualifier)
                
                # Convert Legacy to Phoebe2 units if possible
                if qualifier in legacy_units:
                    val = float(val)
                    val = conversions.convert(legacy_units[qualifier],
                                              this_param.get_unit(),
                                              val)
                    
                if postfix == 'VAL':
                    if qualifier == 'alb':
                        val = 1-float(val)
                    
                    this_param.set_value(val)
                    continue
                elif postfix == 'ADJ':
                    this_param.set_adjust(int(val))
                    continue
                elif postfix == 'STEP':
                    this_param.set_step(float(val))
                    continue
                elif postfix == 'MIN':
                    this_param.set_limits(llim=float(val))
                    continue
                elif postfix == 'MAX':
                    this_param.set_limits(ulim=float(val))
                    continue
            else:
                if leg_qualifier == 'lc_filter':
                    all_lcdeps[0][index]['passband'] = translate_filter(val, separate)
                    all_lcdeps[1][index]['passband'] = translate_filter(val, separate)
                    continue
                
                if leg_qualifier == 'rv_filter':
                    all_rvdeps[0][index]['passband'] = translate_filter(val, separate)
                    all_rvdeps[1][index]['passband'] = translate_filter(val, separate)
                    continue
                
                if leg_qualifier == 'ld_model':
                    ld_model = val.strip()[1:-1].split()[0].lower()
                    comps[0]['ld_func'] = ld_model
                    comps[1]['ld_func'] = ld_model
                    for dep in all_lcdeps[0]:
                        dep['ld_func'] = ld_model
                    for dep in all_lcdeps[1]:
                        dep['ld_func'] = ld_model
                    for dep in all_rvdeps[0]:
                        dep['ld_func'] = ld_model
                    for dep in all_rvdeps[1]:
                        dep['ld_func'] = ld_model
                    continue
                
                if leg_qualifier == 'rv_active':
                    all_rvobs[index].set_enabled(int(val))
                    continue
                
                if leg_qualifier == 'lc_active':
                    all_lcobs[index].set_enabled(int(val))
                    continue
                
                if leg_qualifier == 'lc_dep':
                    val = val.strip()
                    if val[1:-1] == 'Magnitude':
                        all_lcobs[index]['user_units']['flux'] = 'mag'
                        all_lcobs[index]['user_columns'][1] = 'mag'
                    elif val[1:-1] == 'Flux':
                        all_lcobs[index]['user_units']['flux'] = 'W/m3'
                        all_lcobs[index]['user_columns'][1] = 'flux'
                    else:
                        raise ValueError("Flux unit not recognised")
                    continue
                
                if leg_qualifier == 'rv_dep':
                    rv_component_spec[index] = str(val.strip()[1:-1])
                    continue
                
                if leg_qualifier == 'lc_indep':
                    val = val.strip()
                    if val[1:-1] == 'Time (HJD)':
                        all_lcobs[index]['user_units']['time'] = 'JD'
                        all_lcobs[index]['user_columns'][0] = 'time'
                    elif val[1:-1] == 'Phase':
                        all_lcobs[index]['user_units']['phase'] = 'cy'
                        all_lcobs[index]['user_columns'][0] = 'phase'
                        all_lcobs[index]['columns'][all_lcobs[index]['columns'].index('time')] = 'phase'
                    else:
                        raise ValueError("Time unit not recognised")
                    continue
                
                if leg_qualifier == 'rv_indep':
                    val = val.strip()
                    if val[1:-1] == 'Time (HJD)':
                        all_rvobs[index]['user_units']['time'] = 'JD'
                        all_rvobs[index]['user_columns'][0] = 'time'
                    elif val[1:-1] == 'Phase':
                        all_rvobs[index]['user_units']['phase'] = 'cy'
                        all_rvobs[index]['user_columns'][0] = 'phase'
                        all_rvobs[index]['columns'][all_rvobs[index]['columns'].index('time')] = 'phase'
                    else:
                        raise ValueError("Time unit not recognised")
                    continue
                
                if leg_qualifier == 'lc_id':
                    val = val.strip().replace(' ','_')
                    all_lcobs[index]['ref'] = val[1:-1]
                    all_lcdeps[0][index]['ref'] = val[1:-1]
                    all_lcdeps[1][index]['ref'] = val[1:-1]
                    continue
                
                if leg_qualifier == 'rv_id':
                    val = val.strip().replace(' ','_')
                    all_rvobs[index]['ref'] = val[1:-1]
                    all_rvdeps[0][index]['ref'] = val[1:-1]
                    all_rvdeps[1][index]['ref'] = val[1:-1]
                    continue
                
                if leg_qualifier == 'lc_indweight':
                    if val.strip()[1:-1] == 'Unavailable':
                        all_lcobs[index]['columns'] = all_lcobs[index]['columns'][:2]
                        all_lcobs[index]['user_columns'] = all_lcobs[index]['user_columns'][:2]
                    elif val.strip()[1:-1] == 'Standard weight':
                        all_lcobs[index]['columns'][2] = 'weight'
                        all_lcobs[index]['user_columns'][2] = 'weight'
                    elif val.strip()[1:-1] == 'Standard deviation':
                        all_lcobs[index]['columns'][2] = 'sigma'
                        all_lcobs[index]['user_columns'][2] = 'sigma'
                    else:
                        raise ValueError("unrecognised lc_indweight")
                    continue
                
                if leg_qualifier == 'rv_indweight':
                    if val.strip()[1:-1] == 'Unavailable':
                        all_rvobs[index]['columns'] = all_rvobs[index]['columns'][:2]
                        all_rvobs[index]['user_columns'] = all_rvobs[index]['user_columns'][:2]
                    elif val.strip()[1:-1] == 'Standard weight':
                        all_rvobs[index]['columns'][2] = 'weight'
                        all_rvobs[index]['user_columns'][2] = 'weight'
                    elif val.strip()[1:-1] == 'Standard deviation':
                        all_rvobs[index]['columns'][2] = 'sigma'
                        all_rvobs[index]['user_columns'][2] = 'sigma'
                    else:
                        raise ValueError("unrecognised rv_indweight")
                    continue
                
                
                if leg_qualifier == 'rv_filename':
                    all_rvobs[index]['filename'] = val.strip()[1:-1]
                    continue
                
                if leg_qualifier == 'lc_filename':
                    all_lcobs[index]['filename'] = val.strip()[1:-1]
                    continue
                
                if leg_qualifier == 'atm1_switch':
                    atm = 'blackbody' if int(val) == 0 else 'kurucz'
                    comps[0]['atm'] = atm
                    for dep in all_lcdeps[0]:
                        dep['atm'] = atm
                    continue
                
                if leg_qualifier == 'atm2_switch':
                    atm = 'blackbody' if int(val) == 0 else 'kurucz'
                    comps[1]['atm'] = atm
                    for dep in all_lcdeps[1]:
                        dep['atm'] = atm
                    continue
                
                #print 'error2', leg_qualifier, index, compno, val
                continue
            
            
    # now check usecla
    # now check compute_vga
        
    # Create basic stars
    star1 = universe.BinaryRocheStar(comps[0],orbit,meshes[0])
    star2 = universe.BinaryRocheStar(comps[1],orbit,meshes[1])
    
    # add RV curves
    for i in range(len(all_rvobs)):
        # set data
        found = try_to_locate_file(all_rvobs[i]['filename'], rootfile=inputfile)
        if found:
            out = np.loadtxt(found, unpack=True)
            for col_in, col_file in zip(all_rvobs[i]['columns'], out):
                all_rvobs[i][col_in] = col_file
        all_rvobs[i]['filename'] = ""#all_rvobs[i]['ref'] + '_phoebe2.lc'
            
        if not all_rvobs[i]['sigma'].shape:
            all_rvobs[i]['sigma'] = len(all_rvobs[i])*[float(all_rvobs[i]['sigma'])]
        
        if computevga:
            all_rvobs[i].set_adjust('offset', True)
        
        if rv_component_spec[i] == 'Primary RV':
            all_rvdeps[0][i]['method'] = ['flux-weighted', 'dynamical'][proximity_rv1]
            star1.add_pbdeps(all_rvdeps[0][i])
            star1.add_obs(all_rvobs[i])
        else:
            all_rvdeps[1][i]['method'] = ['flux-weighted', 'dynamical'][proximity_rv2]
            star2.add_pbdeps(all_rvdeps[1][i])
            star2.add_obs(all_rvobs[i])
            
    
    # Add LC deps
    for i in range(len(all_lcobs)):
        found = try_to_locate_file(all_lcobs[i]['filename'], rootfile=inputfile)
        if found:
            out = np.loadtxt(found, unpack=True)
            for col_in, col_file in zip(all_lcobs[i]['columns'], out):
                all_lcobs[i][col_in] = col_file
            
            if not all_lcobs[i]['sigma'].shape:
                all_lcobs[i]['sigma'] = len(all_lcobs[i])*[float(all_lcobs[i]['sigma'])]
            
            # Convert from mag to flux (perhaps with uncertainties)
            if all_lcobs[i]['user_columns'][1] == 'mag':
                passband = all_lcdeps[0][i]['passband']
                flux = all_lcobs[i]['flux']
                sigma = all_lcobs[i]['sigma']
                if len(sigma):
                    flux, sigma = conversions.convert('mag', 'W/m3', flux, sigma, passband=passband)
                else:
                    flux = conversions.convert('mag', 'W/m3', flux, passband=passband)
                all_lcobs[i]['flux'] = flux
                all_lcobs[i]['sigma'] = sigma
                
        all_lcobs[i]['filename'] = ""#all_lcobs[i]['ref'] + '_phoebe2.lc'
        
        
        if computehla:
            all_lcdeps[0][i]['pblum'] = -1
            all_lcobs[i].set_adjust('scale', True)
        
        if not usecla:
            all_lcdeps[1][i]['pblum'] = -1
        star1.add_pbdeps(all_lcdeps[0][i])
        star2.add_pbdeps(all_lcdeps[1][i])
    
    bodybag = universe.BodyBag([star1, star2], position=position)
    
    # Add data
    for i in range(len(all_lcobs)):
        bodybag.add_obs(all_lcobs[i])
    
    # Set the name of the thing
    bodybag.set_label(orbit['label'])
    
    
    
    
    return bodybag, compute
    
    
    

def legacy_to_phoebe(inputfile, create_body=False,
                     mesh='wd', root=None):
    """
    Convert a legacy PHOEBE file to the parameterSets. 
    
    Currently this file converts all values, the limits and 
    step size for each parameter in the orbit and components.
    
    Returns two components (containing a body, light curve 
    dependent parameter set and radial velocity dependent 
    parameter set.) and an orbit.
    
    If create_body=True then a body bag is returned or if create_bundle=True
    then a bundel is returned in place of the parameter sets.
    
    @param inputfile: legacy phoebe parameter file
    @type inputfile: str
    @param mesh: if set to 'marching' and C{create_body=True}, a marching mesh
     will be added to the Body. Else, a WD mesh will be added.
    @type mesh: str (one of C{wd} or C{marching})
    @param root: use this root directory to link to data files
    @type root: str or None. If root is ``__relative__``, the root directory
     will be the directory of the inputfile
    """
    if root == '__relative__':
        root = os.path.dirname(os.path.abspath(inputfile))
    #-- initialise all the variables and lists needed for the rvdep parameters
    rv_dep = []
    prim_rv = 0
    sec_rv = 0
    rv_pb = []
    rv_pbweight = []
    lc_pbweight = []
    ld_coeffs1 = []
    ld_coeffs2 = []
    rv_file = []
    lc_file = []
    rvtime = []
    lctime = []
    rvsigma = []
    lcsigma = []
    obsrv1 = []
    obsrv2 = []
    obslc = []
    rvfile = []
    lcfile = []
    rvname = []
    lcname = []
    
    #-- initialise the orbital and component parameter sets. 
    orbit = parameters.ParameterSet(frame='phoebe',context='orbit',add_constraints=True)
    comp1 = parameters.ParameterSet(frame='phoebe',context='component',label='primary',
                                    add_constraints=True)
    comp2 = parameters.ParameterSet(frame='phoebe',context='component',label='secondary',
                                    add_constraints=True)
    position = parameters.ParameterSet('position', distance=(1.,'Rsol'))
    compute = parameters.ParameterSet('compute', beaming_alg='none', refl=False,
                                      heating=True, label='legacy',
                                      eclipse_alg='binary', subdiv_num=3)
    
    mesh1wd = parameters.ParameterSet(frame='phoebe',context='mesh:wd',add_constraints=True)
    mesh2wd = parameters.ParameterSet(frame='phoebe',context='mesh:wd',add_constraints=True)

    #-- Open the parameter file and read each line
    ff = open(inputfile,'r')
    lines = ff.readlines()
    ff.close()
    for l in lines:
        #-- ignore initial comment
        if l[0]=='#': continue

        #-- try splitting the parameter name and value
        try: 
            key, val = l.split('=')
        except:
            logger.error("line " + l[:-1] + " could not be parsed ('{}' probably not Phoebe legacy file)".format(inputfile))
            raise IOError("Cannot parse phoebe file '{}': line '{}'".format(inputfile, l[:-1]))

        #-- if this is an rvdep or lcdep, check which index it has
        # and remove it from the string:
        pattern = re.search('\[(\d)\]',key)
        separate = re.search('\:',val)
        if pattern:
            index = int(pattern.group(1))-1
            key = "".join(key.split(pattern.group(0)))
        if val[2:5] == 'Bes':
            val = val[2:-2].upper()
            val = ".".join(val.split('L IR:'))
            if val[0:8] == 'BESSEL.L':
                val = 'BESSEL.LPRIME'
        elif separate:
            val = ".".join(val.split(separate.group(0)))
            val = val[2:-2].upper()
            if val == 'COROT.SISMO':
                val = 'COROT.SIS'
            elif val == 'KEPLER.MEAN':
                val = 'KEPLER.V'
            elif val == 'IRAC.CH1':
                val = 'IRAC.36'
            elif val == 'MOST.DEFAULT':
                val = 'MOST.V'
            elif val == 'STROMGREN.HBETA_NARROW':
                val = 'STROMGREN.HBN'
            elif val == 'STROMGREN.HBETA_WIDE':
                val = 'STROMGREN.HBW'
            elif val == 'BOLOMETRIC.3000A-10000A':
                val = 'OPEN.BOL'
            elif val == 'HIPPARCOS.BT':
                val = 'TYCHO.BT'
            elif val == 'HIPPARCOS.VT':
                val = 'TYCHO.VT'
            elif val[0:5] == 'SLOAN':
                val = "SDSS".join(val.split('SLOAN'))
                val = val[:-1]
        #-- Remove any trailing white space
        while key[len(key)-1] == ' ': key=key[:-1]

        #-- for each phoebe parameter incorporate an if statement
        # consider the orbit first:
        if key == 'phoebe_lcno':
            lcno = int(val)
            lcdep1 = [parameters.ParameterSet(context='lcdep') for i in range(lcno)]
            lcdep2 = [parameters.ParameterSet(context='lcdep') for i in range(lcno)]

        elif key == 'phoebe_rvno':
            rvno = int(val)
            
        elif key == 'phoebe_dpdt.VAL':
            orbit['dpdt'] = (val,'d/d')
        elif key == 'phoebe_dpdt.MAX':
            orbit.get_parameter('dpdt').set_limits(ulim=float(val)*31557600)
            lower, upper = orbit.get_parameter('dpdt').get_limits()
            orbit.get_parameter('dpdt').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)        
        elif key == 'phoebe_dpdt.MIN':
            orbit.get_parameter('dpdt').set_limits(llim=float(val)*31557600)
        elif key == 'phoebe_dpdt.STEP':
            orbit.get_parameter('dpdt').set_step(step=float(val)*31557600) 
        elif key == 'phoebe_dpdt.ADJ':
            orbit.get_parameter('dpdt').set_adjust(int(val))
               
        elif key == 'phoebe_dperdt.VAL':
            orbit['dperdt'] = (val,'rad/d')    
        elif key == 'phoebe_dperdt.MAX':
            orbit.get_parameter('dperdt').set_limits(ulim=float(val)*np.pi*365.25/180)
            lower, upper = orbit.get_parameter('dperdt').get_limits()
            orbit.get_parameter('dperdt').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_dperdt.MIN':
            orbit.get_parameter('dperdt').set_limits(llim=float(val)*np.pi*365.25/180)
        elif key == 'phoebe_dperdt.STEP':
            orbit.get_parameter('dperdt').set_step(step=float(val)*np.pi*365.25/180)                    
        elif key == 'phoebe_dperdt.ADJ':
            orbit.get_parameter('dperdt').set_adjust(int(val))
                     
        elif key == 'phoebe_ecc.VAL':
            orbit['ecc'] = val   
        elif key == 'phoebe_ecc.MAX':
            orbit.get_parameter('ecc').set_limits(ulim=float(val))
            lower, upper = orbit.get_parameter('ecc').get_limits()
            orbit.get_parameter('ecc').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_ecc.MIN':
            orbit.get_parameter('ecc').set_limits(llim=float(val))    
        elif key == 'phoebe_ecc.STEP':
            orbit.get_parameter('ecc').set_step(step=float(val))            
        elif key == 'phoebe_ecc.ADJ':
            orbit.get_parameter('ecc').set_adjust(int(val))
            
        elif key == 'phoebe_hjd0.VAL':
            orbit['t0'] = (val,'JD') 
        elif key == 'phoebe_hjd0.MAX':
            orbit.get_parameter('t0').set_limits(ulim=float(val))
            lower, upper = orbit.get_parameter('t0').get_limits()
            orbit.get_parameter('t0').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_hjd0.MIN':
            orbit.get_parameter('t0').set_limits(llim=float(val))            
        elif key == 'phoebe_hjd0.STEP':
            orbit.get_parameter('t0').set_step(step=float(val))            
        elif key == 'phoebe_hjd0.ADJ':
            orbit.get_parameter('hjd0').set_adjust(int(val))
            
        elif key == 'phoebe_incl.VAL':
            orbit['incl'] = (val,'deg') 
        elif key == 'phoebe_incl.MAX':
            orbit.get_parameter('incl').set_limits(ulim=float(val))
            lower, upper = orbit.get_parameter('incl').get_limits()
            orbit.get_parameter('incl').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_incl.MIN':
            orbit.get_parameter('incl').set_limits(llim=float(val))                        
        elif key == 'phoebe_incl.STEP':
            orbit.get_parameter('incl').set_step(step=float(val))            
        elif key == 'phoebe_incl.ADJ':
            orbit.get_parameter('incl').set_adjust(int(val))

        elif key == 'phoebe_period.VAL':
            orbit['period'] = (val,'d')
        elif key == 'phoebe_period.MAX':
            orbit.get_parameter('period').set_limits(ulim=float(val))
            lower, upper = orbit.get_parameter('period').get_limits()
            orbit.get_parameter('period').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_period.MIN':
            orbit.get_parameter('period').set_limits(llim=float(val))                        
        elif key == 'phoebe_period.STEP':
            orbit.get_parameter('period').set_step(step=float(val))          
        elif key == 'phoebe_period.ADJ':
            orbit.get_parameter('period').set_adjust(int(val))
            
        elif key == 'phoebe_perr0.VAL':
            orbit['per0'] = (val,'rad')
        elif key == 'phoebe_perr0.MAX':
            orbit.get_parameter('per0').set_limits(ulim=float(val)/np.pi*180)
            lower, upper = orbit.get_parameter('per0').get_limits()
            orbit.get_parameter('per0').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_perr0.MIN':
            orbit.get_parameter('per0').set_limits(llim=float(val)/np.pi*180)            
        elif key == 'phoebe_perr0.STEP':
            orbit.get_parameter('per0').set_step(step=float(val)/np.pi*180)           
        elif key == 'phoebe_perr0.ADJ':
            orbit.get_parameter('per0').set_adjust(int(val))
            
        elif key == 'phoebe_pshift.VAL':
            orbit['phshift'] = val   
        elif key == 'phoebe_pshift.MAX':
            orbit.get_parameter('phshift').set_limits(ulim=float(val))
            lower, upper = orbit.get_parameter('phshift').get_limits()
            orbit.get_parameter('phshift').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_pshift.MIN':
            orbit.get_parameter('phshift').set_limits(llim=float(val))            
        elif key == 'phoebe_pshift.STEP':
            orbit.get_parameter('phshift').set_step(step=float(val))                        
        elif key == 'phoebe_pshift.ADJ':
            orbit.get_parameter('phshift').set_adjust(int(val))

        elif key == 'phoebe_rm.VAL':
            orbit['q'] = val
        elif key == 'phoebe_rm.MAX':
            orbit.get_parameter('q').set_limits(ulim=float(val))
            lower, upper = orbit.get_parameter('q').get_limits()
            orbit.get_parameter('q').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_rm.MIN':
            orbit.get_parameter('q').set_limits(llim=float(val))
        elif key == 'phoebe_rm.STEP':
            orbit.get_parameter('q').set_step(step=float(val))                     
        elif key == 'phoebe_rm.ADJ':
            orbit.get_parameter('rm').set_adjust(int(val))
                       
        elif key == 'phoebe_vga.VAL':
            position['vgamma'] = (val,'km/s')  
        elif key == 'phoebe_vga.MAX':
            position.get_parameter('vgamma').set_limits(ulim=float(val))
            lower, upper = position.get_parameter('vgamma').get_limits()
            position.get_parameter('vgamma').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_vga.MIN':
            position.get_parameter('vgamma').set_limits(llim=float(val))
        elif key == 'phoebe_vga.STEP':
            position.get_parameter('vgamma').set_step(step=float(val))                     
        elif key == 'phoebe_vga.ADJ':
            position.get_parameter('vgamma').set_adjust(int(val))
                     
        elif key == 'phoebe_sma.VAL':
            orbit['sma'] = (val,'Rsol')    
        elif key == 'phoebe_sma.MAX':
            orbit.get_parameter('sma').set_limits(ulim=float(val))
            lower, upper = orbit.get_parameter('sma').get_limits()
            orbit.get_parameter('sma').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_sma.MIN':
            orbit.get_parameter('sma').set_limits(llim=float(val))
        elif key == 'phoebe_sma.STEP':
            orbit.get_parameter('sma').set_step(step=float(val))                     
        elif key == 'phoebe_sma.ADJ':
            orbit.get_parameter('sma').set_adjust(int(val))

        #-- gridsizes:
        elif key == 'phoebe_grid_finesize1':
            mesh1wd['gridsize'] = val
        elif key == 'phoebe_grid_finesize2':
            mesh2wd['gridsize'] = val

        #-- populate the components (only 2 components in legacy)
        elif key == 'phoebe_atm1_switch':
            if val[1:-1] == '1':
                comp1['atm'] = 'kurucz'
            else:    
                comp1['atm'] = 'blackbody'  
        elif key == 'phoebe_atm2_switch':
            if val[1:-1] == '1':
                comp2['atm'] = 'kurucz'
            else:    
                comp2['atm'] = 'blackbody'   
 


        elif key == 'phoebe_alb1.VAL':
            comp1['alb'] = 1-float(val)
            # Switch on heating in the other component
            if comp1['alb'] < 1:
                comp2['irradiator'] = True
            else:
                comp2['irradiator'] = False
        elif key == 'phoebe_alb1.MAX':
            comp1.get_parameter('alb').set_limits(ulim=1-float(val))
            lower, upper = comp1.get_parameter('alb').get_limits()
            comp1.get_parameter('alb').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_alb1.MIN':
            comp1.get_parameter('alb').set_limits(llim=1-float(val))
        elif key == 'phoebe_alb1.STEP':
            comp1.get_parameter('alb').set_step(step=float(val))   
        elif key == 'phoebe_alb1.ADJ':
            comp1.get_parameter('alb').set_adjust(int(val))

        elif key == 'phoebe_alb2.VAL':
            comp2['alb'] = 1-float(val)
            alb2 = comp2['alb']
            # Switch on heating in the oterh component
            if comp2['alb'] < 1:
                comp1['irradiator'] = True
            else:
                comp1['irradiator'] = False
        elif key == 'phoebe_alb2.MAX':
            comp2.get_parameter('alb').set_limits(ulim=1-float(val))
            lower, upper = comp2.get_parameter('alb').get_limits()
            comp2.get_parameter('alb').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_alb2.MIN':
            comp2.get_parameter('alb').set_limits(llim=1-float(val))
        elif key == 'phoebe_alb2.STEP':
            comp2.get_parameter('alb').set_step(step=float(val))   
        elif key == 'phoebe_alb2.ADJ':
            comp2.get_parameter('alb').set_adjust(int(val))

        elif key == 'phoebe_f1.VAL':
            comp1['syncpar'] = val  
        elif key == 'phoebe_f1.MAX':
            comp1.get_parameter('syncpar').set_limits(ulim=float(val))
            lower, upper = comp1.get_parameter('syncpar').get_limits()
            comp1.get_parameter('syncpar').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_f1.MIN':
            comp1.get_parameter('syncpar').set_limits(llim=float(val))
        elif key == 'phoebe_f1.STEP':
            comp1.get_parameter('syncpar').set_step(step=float(val))   
        elif key == 'phoebe_f1.ADJ':
            comp1.get_parameter('syncpar').set_adjust(int(val))

        elif key == 'phoebe_f2.VAL':
            comp2['syncpar'] = val  
        elif key == 'phoebe_f2.MAX':
            comp2.get_parameter('syncpar').set_limits(ulim=float(val))
            lower, upper = comp2.get_parameter('syncpar').get_limits()
            comp2.get_parameter('syncpar').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_f2.MIN':
            comp2.get_parameter('syncpar').set_limits(llim=float(val))
        elif key == 'phoebe_f2.STEP':
            comp2.get_parameter('syncpar').set_step(step=float(val))   
        elif key == 'phoebe_f2.ADJ':
            comp2.get_parameter('syncpar').set_adjust(int(val))

        elif key == 'phoebe_grb1.VAL':
            comp1['gravb'] = val    
        elif key == 'phoebe_grb1.MAX':
            comp1.get_parameter('gravb').set_limits(ulim=float(val))
            lower, upper = comp1.get_parameter('gravb').get_limits()
            comp1.get_parameter('gravb').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_grb1.MIN':
            comp1.get_parameter('gravb').set_limits(llim=float(val))
        elif key == 'phoebe_grb1.STEP':
            comp1.get_parameter('gravb').set_step(step=float(val))   
        elif key == 'phoebe_grb1.ADJ':
            comp1.get_parameter('gravb').set_adjust(int(val))

        elif key == 'phoebe_grb2.VAL':
            comp2['gravb'] = val    
        elif key == 'phoebe_grb2.MAX':
            comp2.get_parameter('gravb').set_limits(ulim=float(val))
            lower, upper = comp2.get_parameter('gravb').get_limits()
            comp2.get_parameter('gravb').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_grb2.MIN':
            comp2.get_parameter('gravb').set_limits(llim=float(val))
        elif key == 'phoebe_grb2.STEP':
            comp2.get_parameter('gravb').set_step(step=float(val))   
        elif key == 'phoebe_grb2.ADJ':
            comp2.get_parameter('gravb').set_adjust(int(val))

        elif key == 'phoebe_pot1.VAL':
            comp1['pot'] = val 
        elif key == 'phoebe_pot1.MAX':
            comp1.get_parameter('pot').set_limits(ulim=float(val))
            lower, upper = comp1.get_parameter('pot').get_limits()
            comp1.get_parameter('pot').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_pot1.MIN':
            comp1.get_parameter('pot').set_limits(llim=float(val))
        elif key == 'phoebe_pot1.STEP':
            comp1.get_parameter('pot').set_step(step=float(val))   
        elif key == 'phoebe_pot1.ADJ':
            comp1.get_parameter('pot').set_adjust(int(val))

        elif key == 'phoebe_pot2.VAL':
            comp2['pot'] = val  
        elif key == 'phoebe_pot2.MAX':
            comp2.get_parameter('pot').set_limits(ulim=float(val))
            lower, upper = comp2.get_parameter('pot').get_limits()
            comp2.get_parameter('pot').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_pot2.MIN':
            comp2.get_parameter('pot').set_limits(llim=float(val))
        elif key == 'phoebe_pot2.STEP':
            comp2.get_parameter('pot').set_step(step=float(val))   
        elif key == 'phoebe_pot2.ADJ':
            comp2.get_parameter('pot').set_adjust(int(val))

        elif key == 'phoebe_teff1.VAL':
            comp1['teff'] = (val,'K') 
        elif key == 'phoebe_teff1.MAX':
            comp1.get_parameter('teff').set_limits(ulim=float(val))
            lower, upper = comp1.get_parameter('teff').get_limits()
            comp1.get_parameter('teff').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_teff1.MIN':
            comp1.get_parameter('teff').set_limits(llim=float(val))
        elif key == 'phoebe_teff1.STEP':
            comp1.get_parameter('teff').set_step(step=float(val))   
        elif key == 'phoebe_teff1.ADJ':
            comp1.get_parameter('teff').set_adjust(int(val))

        elif key == 'phoebe_teff2.VAL':
            comp2['teff'] = (val,'K')
        elif key == 'phoebe_teff2.MAX':
            comp2.get_parameter('teff').set_limits(ulim=float(val))
            lower, upper = comp2.get_parameter('teff').get_limits()
            comp2.get_parameter('teff').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_teff2.MIN':
            comp2.get_parameter('teff').set_limits(llim=float(val))
        elif key == 'phoebe_teff2.STEP':
            comp2.get_parameter('teff').set_step(step=float(val))   
        elif key == 'phoebe_teff2.ADJ':
            comp2.get_parameter('teff').set_adjust(int(val))

        #elif key == 'phoebe_reffect_switch':
            #if val[1:-2] == "1":
                #comp1['irradiator'] = True
                #comp2['irradiator'] = True
            #else:
                #comp1['irradiator'] = True
                #comp2['irradiator'] = True
                #-- set this way because there is always a single reflection
                #-- reflection performed in phoebe legacy.
    

        elif key == 'phoebe_met1.VAL':
            comp1['abun'] = val     
        elif key == 'phoebe_met1.MAX':
            comp1.get_parameter('abun').set_limits(ulim=float(val))
            lower, upper = comp1.get_parameter('abun').get_limits()
            comp1.get_parameter('abun').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_met1.MIN':
            comp1.get_parameter('abun').set_limits(llim=float(val))
        elif key == 'phoebe_met1.STEP':
            comp1.get_parameter('abun').set_step(step=float(val))   
        elif key == 'phoebe_met1.ADJ':
            comp1.get_parameter('abun').set_adjust(int(val))

        elif key == 'phoebe_met2.VAL':
            comp2['abun'] = val   
        elif key == 'phoebe_met2.MAX':
            comp2.get_parameter('abun').set_limits(ulim=float(val))
            lower, upper = comp2.get_parameter('abun').get_limits()
            comp2.get_parameter('abun').set_prior(distribution='uniform',
                                                  lower=lower, upper=upper)
        elif key == 'phoebe_met2.MIN':
            comp2.get_parameter('abun').set_limits(llim=float(val))
        elif key == 'phoebe_met2.STEP':
            comp2.get_parameter('abun').set_step(step=float(val))   
        elif key == 'phoebe_met2.ADJ':
            comp2.get_parameter('abun').set_adjust(int(val))

        elif key == 'phoebe_ld_model':
            val = val[2:-2]
            if val == "Logarithmic law":
                comp1['ld_func'] = 'logarithmic'
                comp2['ld_func'] = 'logarithmic'
            elif val == "Linear cosine law":
                comp1['ld_func'] = 'linear'  
                comp2['ld_func'] = 'linear'  
            elif val == "Square root law":
                comp1['ld_func'] = 'square root'
                comp2['ld_func'] = 'square root'
        
        elif key == 'phoebe_reffect_switch':
            switch = int(val)
            if switch == 0:
                compute['irradiation_alg'] = 'point_source'
            elif switch == 1:
                compute['irradiation_alg'] = 'full'
        
        if key == 'phoebe_ld_xbol1':
            ld_xbol1 = float(val)
        if key == 'phoebe_ld_xbol2':
            ld_xbol2 = float(val)
        if key == 'phoebe_ld_ybol1':
            ld_ybol1 = float(val)
        if key == 'phoebe_ld_ybol2':
            ld_ybol2 = float(val)
            
        #-- Global name of the system
        if key == 'phoebe_name':
            system_label = val[2:-2]
            
            if not system_label:
                system_label = 'new system'

        #-- now populate the lcdep and rvdep parameters 
        if key == 'phoebe_lc_filename':
            if val[2:-2] != "Undefined":
                lcdep1[index]['ref'] = val[2:-2]
                lcdep2[index]['ref'] = val[2:-2]
        
        if key == 'phoebe_ld_lcy1':
            ld_lcy1 = float(val[1:-2])
        if key == 'phoebe_ld_lcy2':
            ld_lcy2 = float(val[1:-2])
        if key == 'phoebe_ld_lcx1.VAL':
            ld_lcx1 = float(val[1:-2])
            lcdep1[index]['ld_coeffs'] = [ld_lcx1, ld_lcy1]
        if key == 'phoebe_ld_lcx2.VAL':
            ld_lcx2 = float(val[1:-2]) 
            lcdep2[index]['ld_coeffs'] = [ld_lcx2, ld_lcy2]

        if key == 'phoebe_lc_filter':
            lcdep1[index]['passband'] = val
            lcdep2[index]['passband'] = val

        if key == 'phoebe_hla.VAL':
            lcdep1[index]['pblum'] = float(val)
        if key == 'phoebe_cla.VAL':
            lcdep2[index]['pblum'] = float(val)
        if key == 'phoebe_el3.VAL':
            lcdep1[index]['l3'] = float(val)
            lcdep2[index]['l3'] = float(val)

        if key == 'phoebe_ld_rvy1':
            ld_rvy1 = float(val[1:-2])
            ld_coeffs1.append([ld_rvx1, ld_rvy1])
        if key == 'phoebe_ld_rvy2':
            ld_rvy2 = float(val[1:-2])
            ld_coeffs2.append([ld_rvx2, ld_rvy2])
        if key == 'phoebe_ld_rvx1':
            ld_rvx1 = float(val[1:-2])
        if key == 'phoebe_ld_rvx2':
            ld_rvx2 = float(val[1:-2])
     
        if key == 'phoebe_rv_sigma':
            rv_pbweight.append(float(val))
            
        if key == 'phoebe_lc_sigma':
            lc_pbweight.append(float(val))
                    
        if key == 'phoebe_rv_filter':
            rv_pb.append(val)
            
        if key == 'phoebe_rv_dep':
            rv_dep.append(val[2:-2])
            if val[2:-2] == 'Primary RV':
                prim_rv+=1
            else:
                sec_rv +=1
                
        if key == 'phoebe_rv_filename':
            filename = val[2:-2]
            # If a root directory is given, set the filepath to be relative
            # from there (but store absolute path name)
            #if root is not None:
            #    filename = os.path.join(root, os.path.basename(filename))
            rv_file.append(filename)
            
        if key == 'phoebe_lc_filename':
            filename = val[2:-2]
            # If a root directory is given, set the filepath to be relative
            # from there (but store absolute path name)
            #if root is not None:
            #    filename = os.path.join(root, os.path.basename(filename))
            lc_file.append(filename)        
            
        if key == 'phoebe_rv_id':
            rvname.append(val)
            
        if key == 'phoebe_lc_id':
            lcname.append(val)      
                     
        if key == 'phoebe_rv_indep':
            if val[2:6] == 'Time':
                rvtime.append('time')
            else:
                rvtime.append('phase')
            
        if key == 'phoebe_lc_indep':
            if val[2:6] == 'Time':
                lctime.append('time')
            else:
                lctime.append('phase')
                
        if key == 'phoebe_rv_indweight':
            if val[11:-2] == 'deviation':
                rvsigma.append('sigma')  
            elif val[11:-2] == 'weight':
                rvsigma.append('weight')
            else:
                rvsigma.append('undefined')
                    
        if key == 'phoebe_lc_indweight':
            if val[11:-2] == 'deviation':
                lcsigma.append('sigma')  
            elif val[11:-2] == 'weight':
                lcsigma.append('weight')
            else:
                lcsigma.append('undefined')

        if key == 'phoebe_hla.ADJ':
            adjust_hla = int(val)
        if key == 'phoebe_cla.ADJ':
            adjust_cla = int(val)
        if key == 'phoebe_el3.ADJ':
            adjust_l3 = int(val)
        
        if key == 'phoebe_usecla_switch':
            use_cla = int(val)
                
    orbit['long_an'] = 0.
 
    comp1['ld_coeffs'] = [ld_xbol1,ld_ybol1]
    comp2['ld_coeffs'] = [ld_xbol2,ld_ybol2]   
    
    #-- for all the light curves, copy the atmosphere type, albedo values and
    #-- limb darkening function. Do this for each component.
    obslc=[]
    i=0
    j=0
    l=0
    for i in range(lcno):    
        lcdep1[i]['atm'] = comp1['atm']
        lcdep1[i]['alb'] = comp1['alb'] 
        lcdep1[i]['ld_func'] = comp1['ld_func']
        lcdep1[i]['ref'] = "lightcurve_"+str(i)
        lcdep2[i]['atm'] = comp2['atm']
        lcdep2[i]['alb'] = comp2['alb'] 
        lcdep2[i]['ld_func'] = comp2['ld_func']
        #-- make sure lables are the same
        lcdep2[i]['ref'] = lcdep1[i]['ref']
        
        if not use_cla:
            lcdep2[i]['pblum'] = -1
            
        lcdep1[i].get_parameter('pblum').set_adjust(adjust_hla)
        lcdep2[i].get_parameter('pblum').set_adjust(adjust_cla)
        lcdep1[i].get_parameter('l3').set_adjust(adjust_l3)
        lcdep2[i].get_parameter('l3').set_adjust(adjust_l3)
        
        if lc_file[i] != "Undefined":
            if lcsigma[i] == 'undefined': 
                if lctime[i]=='time':
                    found_file = try_to_locate_file(lc_file[i], inputfile)
                    if found_file:
                        col1lc,col2lc = np.loadtxt(found_file, unpack=True)
                        obslc.append(datasets.LCDataSet(time=col1lc, flux=col2lc,columns=[lctime[i],'flux'], 
                        ref="lightcurve_"+str(j), filename=str(found_file), statweight=lc_pbweight[i], user_components=lcname[i]))
                    else:
                        logger.warning("The (time) light curve file {} cannot be located.".format(lc_file[i]))                    
                else:
                    found_file = try_to_locate_file(lc_file[i], inputfile)
                    if found_file:
                        col1lc,col2lc = np.loadtxt(found_file, unpack=True)
                        obslc.append(datasets.LCDataSet(time=col1lc, flux=col2lc,columns=[lctime[i],'flux'], 
                        ref="lightcurve_"+str(j), filename=str(found_file), statweight=lc_pbweight[i], user_components=lcname[i]))
                    else:
                        logger.warning("The (phase) light curve file {} cannot be located.".format(lc_file[i]))                    
            else:
                if lctime[i]=='time':
                    if lcsigma[i]=='sigma': 
                        found_file = try_to_locate_file(lc_file[i], inputfile)
                        if found_file:
                            col1lc,col2lc,col3lc = np.loadtxt(found_file, unpack=True)
                            obslc.append(datasets.LCDataSet(time=col1lc,flux=col2lc,sigma=col3lc,columns=[lctime[i],'flux',lcsigma[i]], 
                            ref="lightcurve_"+str(j), filename=str(found_file), statweight=lc_pbweight[i], user_components=lcname[i]))
                        else:
                            logger.warning("The light curve (1) file {} cannot be located.".format(lc_file[i]))                    
                    else:
                        found_file = try_to_locate_file(lc_file[i], inputfile)
                        if found_file:
                            col1lc,col2lc,col3lc = np.loadtxt(found_file, unpack=True)
                            obslc.append(datasets.LCDataSet(time=col1lc,flux=col2lc,sigma=1./col3lc**2,columns=[lctime[i],'flux','sigma'], 
                            ref="lightcurve_"+str(j), filename=str(found_file), statweight=lc_pbweight[i], user_components=lcname[i]))
                        else:
                            logger.warning("The light curve (2) file {} cannot be located.".format(lc_file[i]))                    
                else:
                    if lcsigma[i]=='sigma': 
                        if os.path.isfile(lc_file[i]) or os.path.isfile(os.path.basename(lc_file[i])):
                            col1lc,col2lc,col3lc = np.loadtxt(lc_file[i], unpack=True)
                            obslc.append(datasets.LCDataSet(phase=col1lc,flux=col2lc,sigma=col3lc,columns=[lctime[i],'flux',lcsigma[i]], 
                            ref="lightcurve_"+str(j), filename=str(found_file), statweight=lc_pbweight[i], user_components=lcname[i]))
                        else:
                            logger.warning("The light curve file (3) {} cannot be located.".format(lc_file[i]))                    
                    else:
                        if os.path.isfile(lc_file[i]) or os.path.isfile(os.path.basename(lc_file[i])):
                            col1lc,col2lc,col3lc = np.loadtxt(lc_file[i], unpack=True)
                            obslc.append(datasets.LCDataSet(phase=col1lc,flux=col2lc,sigma=1./col3lc**2,columns=[lctime[i],'flux','sigma'], 
                            ref="lightcurve_"+str(j), filename=str(found_file), statweight=lc_pbweight[i], user_components=lcname[i]))
                        else:
                            logger.warning("The light curve file (4) {} cannot be located.".format(lc_file[i]))                    
 
                l+=1#counts the number of the observations for component 1
            j+=1#counts the number of observations and synthetic rv curves for component 1
        
               
    rvdep1 = [parameters.ParameterSet(context='rvdep') for i in range(prim_rv)]
    rvdep2 = [parameters.ParameterSet(context='rvdep') for i in range(sec_rv)]
  
        
    j=0
    k=0
    l=0
    m=0
    i=0
    obsrv1=[]
    obsrv2=[]
    for i in range(rvno):

        if rv_dep[i] == 'Primary RV':
            rvdep1[j]['ld_coeffs'] = ld_coeffs1[i]
            rvdep1[j]['passband'] = rv_pb[i]
            rvdep1[j]['ref'] = "primaryrv_"+str(j)
            rvdep1[j]['atm'] = comp1['atm']
            rvdep1[j]['alb'] = comp1['alb']
            rvdep1[j]['ld_func'] = comp1['ld_func']
            if os.path.isfile(rv_file[i]):
                if rvsigma[i] == 'undefined': 
                    if rvtime[i]=='time':
                        if os.path.isfile(rv_file[i]) or os.path.isfile(os.path.basename(rv_file[i])):
                            col1rv1,col2rv1 = np.loadtxt(rv_file[i], unpack=True)
                            obsrv1.append(datasets.RVDataSet(time=col1rv1, rv=col2rv1,columns=[rvtime[i],'rv'], 
                            ref="primaryrv_"+str(j), filename=str(rv_file[i]), statweight=rv_pbweight[i], user_components=rvname[i]))
                        else:
                            logger.warning("The radial velocity file {} cannot be located.".format(rv_file[i]))                    
                    else:
                        if os.path.isfile(rv_file[i]) or os.path.isfile(os.path.basename(rv_file[i])):
                            col1rv1,col2rv1 = np.loadtxt(rv_file[i], unpack=True)
                            obsrv1.append(datasets.RVDataSet(phase=col1rv1, rv=col2rv1,columns=[rvtime[i],'rv'], 
                            ref="primaryrv_"+str(j), filename=str(rv_file[i]), statweight=rv_pbweight[i], user_components=rvname[i]))
                        else:
                            logger.warning("The radial velocity file {} cannot be located.".format(rv_file[i]))  
                else:
                    if rvtime[i]=='time':
                        if rvsigma[i]=='sigma': 
                            if os.path.isfile(rv_file[i]) or os.path.isfile(os.path.basename(rv_file[i])):
                                col1rv1,col2rv1,col3rv1 = np.loadtxt(rv_file[i], unpack=True)
                                obsrv1.append(datasets.RVDataSet(time=col1rv1,rv=col2rv1,sigma=col3rv1,columns=[rvtime[i],'rv',rvsigma[i]], 
                                ref="primaryrv_"+str(j), filename=str(rv_file[i]), statweight=rv_pbweight[i], user_components=rvname[i]))
                            else:
                                logger.warning("The radial velocity file {} cannot be located.".format(rv_file[i]))
                        else:
                            if os.path.isfile(rv_file[i]) or os.path.isfile(os.path.basename(rv_file[i])):
                                col1rv1,col2rv1,col3rv1 = np.loadtxt(rv_file[i], unpack=True)
                                obsrv1.append(datasets.RVDataSet(time=col1rv1,rv=col2rv1,weight=col3rv1,columns=[rvtime[i],'rv',rvsigma[i]], 
                                ref="primaryrv_"+str(j), filename=str(rv_file[i]), statweight=rv_pbweight[i], user_components=rvname[i]))
                            else:
                                logger.warning("The radial velocity file {} cannot be located.".format(rv_file[i]))                            
                    else:
                        if rvsigma[i]=='weight': 
                            if os.path.isfile(rv_file[i]) or os.path.isfile(os.path.basename(rv_file[i])):
                                col1rv1,col2rv1,col3rv1 = np.loadtxt(rv_file[i], unpack=True)
                                obsrv1.append(datasets.RVDataSet(phase=col1rv1,rv=col2rv1,weight=col3rv1,columns=[rvtime[i],'rv',rvsigma[i]],
                                ref="primaryrv_"+str(j), filename=str(rv_file[i]), statweight=rv_pbweight[i], user_components=rvname[i]))
                            else:
                                logger.warning("The radial velocity file {} cannot be located.".format(rv_file[i]))            
                        else:
                            if os.path.isfile(rv_file[i]) or os.path.isfile(os.path.basename(rv_file[i])):
                                col1rv1,col2rv1,col3rv1 = np.loadtxt(rv_file[i], unpack=True)
                                obsrv1.append(datasets.RVDataSet(phase=col1rv1,rv=col2rv1,sigma=col3rv1,columns=[rvtime[i],'rv',rvsigma[i]], 
                                ref="primaryrv_"+str(j), filename=str(rv_file[i]), statweight=rv_pbweight[i], user_components=rvname[i]))
                            else:
                                logger.warning("The radial velocity file {} cannot be located.".format(rv_file[i]))         
                l+=1#counts the number of the observations for component 1
            j+=1#counts the number of observations and synthetic rv curves for component 1

        else:
            rvdep2[k]['ld_coeffs'] = ld_coeffs2[i] 
            rvdep2[k]['passband'] = rv_pb[i]
            rvdep2[k]['atm'] = comp2['atm']
            rvdep2[k]['alb'] = comp2['alb']
            rvdep2[k]['ld_func'] = comp2['ld_func']
            rvdep2[k]['ref'] = "secondaryrv_"+str(k)           
            if os.path.isfile(rv_file[i]):
                if rvsigma[i] == 'undefined': 
                    if rvtime[i]=='time':
                        if os.path.isfile(rv_file[i]) or os.path.isfile(os.path.basename(rv_file[i])):
                            col1rv2,col2rv2 = np.loadtxt(rv_file[i], unpack=True)
                            obsrv2.append(datasets.RVDataSet(time=col1rv2, rv=col2rv2,columns=[rvtime[i],'rv'], 
                            ref="secondaryrv_"+str(k), filename=str(rv_file[i]), statweight=rv_pbweight[i], user_components=rvname[i]))
                        else:
                            logger.warning("The radial velocity file {} cannot be located.".format(rv_file[i]))         
                    else:
                        if os.path.isfile(rv_file[i]) or os.path.isfile(os.path.basename(rv_file[i])):
                            col1rv2,col2rv2 = np.loadtxt(rv_file[i], unpack=True)
                            obsrv2.append(datasets.RVDataSet(phase=col1rv2, rv=col2rv2,columns=[rvtime[i],'rv'], 
                            ref="secondaryrv_"+str(k), filename=str(rv_file[i]), statweight=rv_pbweight[i], user_components=rvname[i]))
                        else:
                            logger.warning("The radial velocity file {} cannot be located.".format(rv_file[i]))         
                else:
                    if rvtime[i]=='time':
                        if rvsigma[i]=='sigma': 
                            if os.path.isfile(rv_file[i]) or os.path.isfile(os.path.basename(rv_file[i])):
                                col1rv2,col2rv2,col3rv2 = np.loadtxt(rv_file[i], unpack=True)
                                obsrv2.append(datasets.RVDataSet(time=col1rv2,rv=col2rv2,sigma=col3rv2,columns=[rvtime[i],'rv',rvsigma[i]], 
                                ref="secondaryrv_"+str(k), filename=str(rv_file[i]), statweight=rv_pbweight[i], user_components=rvname[i]))
                            else:
                                logger.warning("The radial velocity file {} cannot be located.".format(rv_file[i]))         
                        else:
                            if os.path.isfile(rv_file[i]) or os.path.isfile(os.path.basename(rv_file[i])):
                                col1rv2,col2rv2,col3rv2 = np.loadtxt(rv_file[i], unpack=True)
                                obsrv2.append(datasets.RVDataSet(time=col1rv2,rv=col2rv2,weight=col3rv2,columns=[rvtime[i],'rv',rvsigma[i]], 
                                ref="secondaryrv_"+str(k), filename=str(rv_file[i]), statweight=rv_pbweight[i], user_components=rvname[i]))
                            else:
                                logger.warning("The radial velocity file {} cannot be located.".format(rv_file[i]))         
                            
                    else:
                        if rvsigma[i]=='weight': 
                            if os.path.isfile(rv_file[i]) or os.path.isfile(os.path.basename(rv_file[i])):
                                col1rv2,col2rv2,col3rv2 = np.loadtxt(rv_file[i], unpack=True)
                                obsrv2.append(datasets.RVDataSet(phase=col1rv2,rv=col2rv2,weight=col3rv2,columns=[rvtime[i],'rv',rvsigma[i]], 
                                ref="secondaryrv_"+str(k), filename=str(rv_file[i]), statweight=rv_pbweight[i], user_components=rvname[i]))
                            else:
                                logger.warning("The radial velocity file {} cannot be located.".format(rv_file[i]))         
                        else:
                            if os.path.isfile(rv_file[i]) or os.path.isfile(os.path.basename(rv_file[i])):
                                col1rv2,col2rv2,col3rv2 = np.loadtxt(rv_file[i], unpack=True)
                                obsrv2.append(datasets.RVDataSet(phase=col1rv2,rv=col2rv2,sigma=col3rv2,columns=[rvtime[i],'rv',rvsigma[i]], 
                                ref="secondaryrv_"+str(k), filename=str(rv_file[i]), statweight=rv_pbweight[i], user_components=rvname[i]))
                            else:
                                logger.warning("The radial velocity file {} cannot be located.".format(rv_file[i]))         
 
                m+=1#counts the number of the observation for component 2
            k+=1#counts the number of observations and synthetic rv curves for component 2
#love you!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
    #-- copy the component labels to the orbits
    orbit['c1label'] = comp1['label']
    orbit['c2label'] = comp2['label']
    orbit['label'] = system_label
    
    # t0 is the time of superior conjunction in Phoebe Legacy
    orbit['t0type'] = 'superior conjunction'


    logger.info("Loaded contents from file {}".format(inputfile))
    
    # Mesh preparation
    # MARCHING
    mesh1mar = parameters.ParameterSet(context='mesh:marching')
    mesh2mar = parameters.ParameterSet(context='mesh:marching')
    #-- empirical calibration (conversion between gridsize from WD and
    #   pyphoebe marching delta)
    mesh1mar['delta'] = 10**(-0.98359345*np.log10(mesh1wd['gridsize'])+0.4713824)
    mesh2mar['delta'] = 10**(-0.98359345*np.log10(mesh2wd['gridsize'])+0.4713824)
    
    if mesh == 'marching':
        mesh1 = mesh1mar
        mesh2 = mesh2mar
    else:
        mesh1 = mesh1wd
        mesh2 = mesh2wd
    
    # make sure default sigmas are attached to all observations
    for gobs in [obsrv1, obsrv2, obslc]:
        for iobs in gobs:
            iobs.estimate_sigma(from_col=None, force=False)
        
    
    if create_body:
        
        #need an if statement here incase no obs
        if prim_rv !=0 and sec_rv !=0:
            star1 = universe.BinaryRocheStar(comp1,orbit,mesh1,pbdep=lcdep1+rvdep1,obs=obsrv1)
            star2 = universe.BinaryRocheStar(comp2,orbit,mesh2,pbdep=lcdep2+rvdep2,obs=obsrv2)
        elif prim_rv == 0 and sec_rv == 0:
            star1 = universe.BinaryRocheStar(comp1,orbit,mesh1,pbdep=lcdep1+rvdep1)
            star2 = universe.BinaryRocheStar(comp2,orbit,mesh2,pbdep=lcdep2+rvdep2)
        elif prim_rv == 0 and sec_rv !=0:
            star1 = universe.BinaryRocheStar(comp1,orbit,mesh1,pbdep=lcdep1+rvdep1)
            star2 = universe.BinaryRocheStar(comp2,orbit,mesh2,pbdep=lcdep2+rvdep2,obs=obsrv2)
        elif prim_rv !=0 and sec_rv == 0:
            star1 = universe.BinaryRocheStar(comp1,orbit,mesh1,pbdep=lcdep1+rvdep1,obs=obsrv1)
            star2 = universe.BinaryRocheStar(comp2,orbit,mesh2,pbdep=lcdep2+rvdep2) 
                   
        if lcno !=0:
            bodybag = universe.BodyBag([star1,star2],solve_problems=True, position=position,obs=obslc)
        else:
            bodybag = universe.BodyBag([star1,star2],solve_problems=True, position=position)
        
        # Set the name of the thing
        bodybag.set_label(orbit['label'])
        
    #if create_bundle:
    #
    #    bundle = Bundle(bodybag)
    #    
    #    return bundle
    
    if create_body:
        return bodybag, compute
    
    body1 = comp1, mesh1, lcdep1, rvdep1, obsrv1
    body2 = comp2, mesh2, lcdep2, rvdep2, obsrv2
    logger.info("Successfully parsed Phoebe Legacy file {}".format(inputfile))
    
    return body1, body2, orbit, position, obslc, compute 


def _to_legacy_atm(component, ref, category='lc'):
    """
    Parse ld_coeffs, ld_func, atm
    """
    warnings = []
    if not isinstance(component, parameters.ParameterSet):
        if ref != '__bol':
            pbdep = component.params['pbdep'][category+'dep'][ref]
        else:
            pbdep = component.params['component']
    else:
        pbdep = component
    
    # ld_model
    ld_model = pbdep['ld_func']
    if ld_model == 'logarithmic':
        ld_model = 'Logarithmic law'
    elif ld_model == 'linear':
        ld_model = 'Linear cosine law'
    elif ld_model == 'square_root':
        ld_model = 'Square root law'
    else:
        ld_model = 'Linear cosine law'
        pbdep['ld_coeffs'] = [0.5, 0.0]
        warnings.append('Cannot translate ld_func={} to legacy: set linear cosine law with coefficient 0.5'.format(pbdep['ld_func']))
        
    # if the ld_coeffs are taken from a grid, get the average ones from the mesh
    ld_coeffs = pbdep['ld_coeffs']
    if isinstance(ld_coeffs, str):
        ld_coeffs = component.mesh['ld_{}'.format(ref)]
        ld_coeffs = ld_coeffs.mean(axis=1)[:2]
        warnings.append("Cannot translate ld_coeffs={} to legacy: taking mean value over the surface".format(pbdep['ld_coeffs']))
    else:
        ld_coeffs = ld_coeffs[:2]
    
    if len(ld_coeffs) == 1:
        ld_coeffs = ld_coeffs + [0.0]
    elif not len(ld_coeffs) == 2:
        raise ValueError("Cannot translate ld_coefffs={} to legacy at all".format(pbdep['ld_coeffs']))
    
    # atmosphere file
    atm = pbdep['atm']
    if atm == 'blackbody':
        atm = 0
    elif atm == 'kurucz':
        atm = 1
    else:
        warnings.append("Cannot translate atm={} to legacy: falling back to blackbody".format(pbdep['atm']))
        atm = 0
        
    return atm, ld_model, ld_coeffs, warnings


def _to_legacy_lc(lcobs, lcdep1, lcdep2, phoebe_ld_model, phoebe_atm1_switch,
                  phoebe_atm2_switch):
    """
    Parse ld_coeffs, ld_func, atm
    """
    warnings = []
    parameters = dict()
    
    atm1, ld_model1, ld_coeffs1, warnings1 = _to_legacy_atm(lcdep1, lcdep1['ref'], category='lc')
    atm2, ld_model2, ld_coeffs2, warnings2 = _to_legacy_atm(lcdep2, lcdep1['ref'], category='lc')
    
    warnings += warnings1
    warnings += warnings2
    
    # ld models need to be the same and equal to the bolometric one
    if lcdep1['ld_func'] != phoebe_ld_model:
        warnings.append('Primary passband ld_model of {} forced to bolometric one, also copying coefficients'.format(lcobs['ref']))
    
    if lcdep1['passband'] != lcdep2['passband']:
        raise ValueError("Passband incompatibility!")
   
    parameters['phoebe_lc_id'] = lcobs['ref']
    parameters['phoebe_lc_filename'] = lcobs['filename']
    parameters['phoebe_lc_dep'] = "Time (HJD)" if 'time' in lcobs['columns'] else "Phase"
    parameters['phoebe_lc_indep'] = "Flux" if 'flux' in lcobs['columns'] else "Magnitude"
    parameters['phoebe_lc_filter'] = lcdep1['passband']
    parameters['phoebe_ld_lcx1'] = ld_coeffs1[0]
    parameters['phoebe_ld_lcy1'] = ld_coeffs1[1]
    parameters['phoebe_ld_lcx2'] = ld_coeffs2[0]
    parameters['phoebe_ld_lcy2'] = ld_coeffs2[1]
        
    parameters['phoebe_ld_lcx1_adj'] = 0
    parameters['phoebe_ld_lcx2_adj'] = 0
    parameters['phoebe_ld_lcx1_min'] = 0
    parameters['phoebe_ld_lcx2_min'] = 0
    parameters['phoebe_ld_lcx1_max'] = 1
    parameters['phoebe_ld_lcx2_max'] = 1
    parameters['phoebe_ld_lcx1_step'] = 1e-6
    parameters['phoebe_ld_lcx2_step'] = 1e-6
    
    if not 'extinction' in lcdep1:
        parameters['phoebe_extinction'] = 0.0
        parameters['phoebe_extinction_adj'] = 0
        parameters['phoebe_extinction_min'] = 0.0
        parameters['phoebe_extinction_max'] = 100.0
        parameters['phoebe_extinction_step'] = 1e-6
        
    
    parameters['phoebe_el3'] = lcdep1['l3']
    par_p1 = 'el3'
    par_p2 = 'l3'
    out = _to_legacy_parameter(lcdep1.get_parameter(par_p2))
    parameters['phoebe_{}_val'.format(par_p1)] = out[0]
    parameters['phoebe_{}_adj'.format(par_p1)] = out[1]
    parameters['phoebe_{}_step'.format(par_p1)] = out[4]
    parameters['phoebe_{}_min'.format(par_p1)] = out[2]
    parameters['phoebe_{}_max'.format(par_p1)] = out[3]
    
    parameters['phoebe_lc_active'] = lcobs.get_enabled()
    
    # Passband luminosities
    pblum1 = lcdep1['pblum']
    pblum2 = lcdep2['pblum']
    
    par_p1 = 'hla'
    par_p2 = 'pblum'
    out = _to_legacy_parameter(lcdep1.get_parameter('pblum'))
    parameters['phoebe_{}_val'.format(par_p1)] = out[0]
    parameters['phoebe_{}_adj'.format(par_p1)] = out[1]
    parameters['phoebe_{}_step'.format(par_p1)] = out[4]
    parameters['phoebe_{}_min'.format(par_p1)] = out[2]
    parameters['phoebe_{}_max'.format(par_p1)] = out[3]
    
    par_p1 = 'cla'
    par_p2 = 'pblum'
    out = _to_legacy_parameter(lcdep2.get_parameter('pblum'))
    parameters['phoebe_{}_val'.format(par_p1)] = out[0]
    parameters['phoebe_{}_adj'.format(par_p1)] = out[1]
    parameters['phoebe_{}_step'.format(par_p1)] = out[4]
    parameters['phoebe_{}_min'.format(par_p1)] = out[2]
    parameters['phoebe_{}_max'.format(par_p1)] = out[3]
    
    if lcobs.get_adjust('scale'):
        parameters['phoebe_compute_hla_switch'] = 1
    else:
        parameters['phoebe_compute_hla_switch'] = 0
        
    parameters['phoebe_usecla_switch'] = 1
    
    if pblum1 == -1:
        pblum1 = 4*np.pi
    if pblum2 == -1:
        pblum2 = 0.0
        parameters['phoebe_usecla_switch'] = 0        
        
    parameters['phoebe_hla'] = pblum1
    parameters['phoebe_cla'] = pblum2

    # Cadence
    parameters['phoebe_cadence_switch'] = len(lcobs['samprate']) if 'samprate' in lcobs else 0
    if parameters['phoebe_cadence_switch']:
        parameters['phoebe_cadence_rate'] =  lcobs['samprate'][0] if 'samprate' in lcobs else 10
        parameters['phoebe_cadence'] = lcobs['exptime'][0] if 'exptime' in lcobs else 0
    else:
        parameters['phoebe_cadence_rate'] = 0
        parameters['phoebe_cadence'] = 0
        
    return parameters, warnings




def _to_legacy_parameter(parameter, units=None):
    value = parameter.get_value()
    adj = 1 if parameter.get_adjust() else 0
    if parameter.has_limits():
        minim = parameter.get_limits()[0]
        maxim = parameter.get_limits()[1]
        step = (maxim-minim)*1e-6
    else:
        minim = -1000000
        maxim = +1000000
        step = (maxim-minim)*1e-6
    
    # Some special cases:
    if parameter.get_qualifier()=='alb':
        value = 1-value
        minim = 1-minim
        maxim = 1-maxim
        
    return value, adj, minim, maxim, step
    


def phoebe_to_legacy(bundle, outfile):
    """
    Parse Phoebe2 to legacy.
    """
    system = bundle.get_system()
    refs = system.get_refs(per_category=True)
    warnings = []
    
    if not (hasattr(system, 'bodies') and len(system)==2 and isinstance(system[0], universe.BinaryRocheStar) and isinstance(system[1], universe.BinaryRocheStar)):
        raise ValueError("System is not a normal binary, cannot parse to Phoebe legacy file")
    
    # number of light curves and radial velocity curves:
    if 'lc' in refs:
        phoebe_lcno = len(refs['lc'])
    else:
        phoebe_lcno = 0
    if 'rv' in refs:
        phoebe_rvno = len(refs['rv'])
    else:
        phoebe_rvno = 0
    
    phoebe_name = system.get_label()
    
    # Compute stuff
    #==================
    compute = bundle.get_compute(label='detailed')
    phoebe_reffect_switch = 0 if compute['irradiation_alg']=='point_source' else 1
    phoebe_reffect_reflections = compute['refl_num']
    
    # Orbital stuff
    #==================
    orbit = system[0].params['orbit']
    par_p2s = ['dpdt', 'sma', 't0', 'dperdt', 'phshift', 'q', 'ecc', 'per0', 'period']
    par_p1s = ['dpdt', 'sma', 'hjd0', 'dperdt','pshift', 'rm', 'ecc', 'perr0', 'period']
    for par_p2, par_p1 in zip(par_p2s, par_p1s):
        out = _to_legacy_parameter(orbit.get_parameter(par_p2))
        locals()['phoebe_{}_val'.format(par_p1)] = out[0]
        locals()['phoebe_{}_adj'.format(par_p1)] = out[1]
        locals()['phoebe_{}_step'.format(par_p1)] = out[4]
        locals()['phoebe_{}_min'.format(par_p1)] = out[2]
        locals()['phoebe_{}_max'.format(par_p1)] = out[3]
        
    # Position stuff
    #==================
    position = system.params['position']
    par_p2s = ['vgamma']
    par_p1s = ['vga']
    for par_p2, par_p1 in zip(par_p2s, par_p1s):
        out = _to_legacy_parameter(position.get_parameter(par_p2))
        locals()['phoebe_{}_val'.format(par_p1)] = out[0]
        locals()['phoebe_{}_adj'.format(par_p1)] = out[1]
        locals()['phoebe_{}_step'.format(par_p1)] = out[4]
        locals()['phoebe_{}_min'.format(par_p1)] = out[2]
        locals()['phoebe_{}_max'.format(par_p1)] = out[3]
            
    # Component stuff
    #==================
    comp1 = system[0].params['component']
    comp2 = system[1].params['component']
    
    try:
        #'X-ray binary'
        #'Overcontact binary of the W Uma type'
        #'Overcontact binary not in thermal contact'
        #'Semi-detached binary, primary star fills Roche lobe'
        #'Semi-detached binary, secondary star fills Roche lobe'
        #'Double contact binary'
        phoebe_model = dict(detached='Detached binary', unconstrained='Unconstrained binary system')[comp1['morphology']]
    except KeyError:
        raise KeyError("Model {} not implemented yet in parser".format(comp1['morphology']))
    
    par_p2s = ['syncpar', 'gravb', 'abun','pot','teff','alb']
    par_p1s = ['f','grb','met','pot','teff','alb']
    for i,comp in zip(range(1,3),[comp1,comp2]):
        for par_p2, par_p1 in zip(par_p2s, par_p1s):
            out = _to_legacy_parameter(comp.get_parameter(par_p2))
            locals()['phoebe_{}{}_val'.format(par_p1,i)] = out[0]
            locals()['phoebe_{}{}_adj'.format(par_p1,i)] = out[1]
            locals()['phoebe_{}{}_step'.format(par_p1,i)] = out[4]
            locals()['phoebe_{}{}_min'.format(par_p1,i)] = out[2]
            locals()['phoebe_{}{}_max'.format(par_p1,i)] = out[3]
    
    phoebe_grid_finesize1 = marching.delta_to_gridsize(system[0].params['mesh']['delta'])
    phoebe_grid_finesize2 = marching.delta_to_gridsize(system[1].params['mesh']['delta'])
    
    # Atmosphere stuff
    #==================
    phoebe_atm1_switch, phoebe_ld_model, ld_coeffs1_bol, warnings1 = _to_legacy_atm(system[0], '__bol')
    phoebe_atm2_switch, ld_model2_bol, ld_coeffs2_bol, warnings2 = _to_legacy_atm(system[1], '__bol')
    
    warnings += warnings1
    warnings += warnings2
    phoebe_ld_xbol1, phoebe_ld_ybol1 = ld_coeffs1_bol
    
    if ld_model2_bol != phoebe_ld_model:
        warnings.append("Two components have different limb darkening law: copying bolometric ld info from primary")
        phoebe_ld_xbol2, phoebe_ld_ybol2 = ld_coeffs1_bol
    else:
        phoebe_ld_xbol2, phoebe_ld_ybol2 = ld_coeffs2_bol
    
    # Light curve stuff
    #===================
    addendum = ''
    light_curve_template = """
phoebe_lc_id[{nr}] = "{phoebe_lc_id}"    
phoebe_lc_dep[{nr}] = "{phoebe_lc_dep}"
phoebe_lc_indep[{nr}] = "{phoebe_lc_indep}"
phoebe_lc_filter[{nr}] = "{phoebe_lc_filter}"
phoebe_lc_filename[{nr}] = "{phoebe_lc_filename}"
phoebe_lc_indweight[{nr}] = "Standard deviation"
phoebe_lc_sigma[{nr}] = 1.0
phoebe_ld_lcx1[{nr}].VAL = {phoebe_ld_lcx1}
phoebe_ld_lcy1[{nr}].VAL = {phoebe_ld_lcy1}
phoebe_ld_lcx2[{nr}].VAL = {phoebe_ld_lcx2}
phoebe_ld_lcy2[{nr}].VAL = {phoebe_ld_lcy2}
phoebe_hla[{nr}].VAL = {phoebe_hla}
phoebe_cla[{nr}].VAL = {phoebe_cla}
phoebe_extinction[{nr}].VAL = {phoebe_extinction}
phoebe_el3[{nr}].VAL = {phoebe_el3}
phoebe_lc_active[{nr}] = {phoebe_lc_active}
"""
    if 'lc' in refs:
        for i,ref in enumerate(refs['lc']):
            lcobs = system.get_obs(ref=ref, category='lc')
            lcdep1 = system.get_parset(ref=ref, category='lc')[0]
            lcdep2 = system.get_parset(ref=ref, category='lc')[0]
            lcvals, warnings = _to_legacy_lc(lcobs, lcdep1, lcdep2,
                                phoebe_ld_model, phoebe_atm1_switch,
                                phoebe_atm2_switch)
            lcvals['nr'] = i+1
            light_curve_template.format(**lcvals)
        # Common parameters for all light curves
        for key in lcvals:
            locals()[key] = lcvals[key]
    
    
    common_light_curve_template = """
phoebe_ld_lcx1.ADJ  = {phoebe_ld_lcx1_adj}
phoebe_ld_lcx1.STEP = {phoebe_ld_lcx1_step}
phoebe_ld_lcx1.MIN  = {phoebe_ld_lcx1_min}
phoebe_ld_lcx1.MAX  = {phoebe_ld_lcx1_max}
phoebe_ld_lcx2.ADJ  = {phoebe_ld_lcx2_adj}
phoebe_ld_lcx2.STEP = {phoebe_ld_lcx2_step}
phoebe_ld_lcx2.MIN  = {phoebe_ld_lcx2_min}
phoebe_ld_lcx2.MAX  = {phoebe_ld_lcx2_max}
phoebe_hla.ADJ  = {phoebe_hla_adj}
phoebe_hla.STEP = {phoebe_hla_step}
phoebe_hla.MIN  = {phoebe_hla_min}
phoebe_hla.MAX  = {phoebe_hla_max}
phoebe_cla.ADJ  = {phoebe_cla_adj}
phoebe_cla.STEP = {phoebe_cla_step}
phoebe_cla.MIN  = {phoebe_cla_min}
phoebe_cla.MAX  = {phoebe_cla_max}
phoebe_extinction.ADJ  = {phoebe_extinction_adj}
phoebe_extinction.STEP = {phoebe_extinction_step}
phoebe_extinction.MIN  = {phoebe_extinction_min}
phoebe_extinction.MAX  = {phoebe_extinction_max}
phoebe_el3.ADJ  = {phoebe_el3_adj}
phoebe_el3.STEP = {phoebe_el3_step}
phoebe_el3.MIN  = {phoebe_el3_min}
phoebe_el3.MAX  = {phoebe_el3_max}
phoebe_compute_hla_switch = {phoebe_compute_hla_switch}
phoebe_usecla_switch = {phoebe_usecla_switch}
"""
    if 'lc' in refs:
        addendum += common_light_curve_template.format(**locals())
    
    
    
    template_filename =  os.path.join(os.path.dirname(os.path.abspath(__file__)),'legacy_template.phoebe')
    with open(template_filename) as template_file:
        with open(outfile,'w') as out_file:
            out_file.write('#' + '\n#'.join(warnings))
            out_file.write(template_file.read().format(**locals()))
            out_file.write(addendum)
            


def wd_to_phoebe(filename, mesh='marching', create_body=True):
    if not create_body:
        raise NotImplementedError('create_body=False')
    if not mesh=='marching':
        raise NotImplementedError("mesh!='marching'")
    
    ps, lc, rv = wd.lcin_to_ps(filename, version='wd2003')
    comp1, comp2, orbit, position = wd.wd_to_phoebe(ps, lc, rv)
    position['distance'] = 1.,'Rsol'
    star1, lcdep1, rvdep1 = comp1
    star2, lcdep2, rvdep2 = comp2
    lcdep1['pblum'] = lc['hla']
    lcdep2['pblum'] = lc['cla']
    delta = 10**(-0.98359345*np.log10(ps['n1'])+0.4713824)
    mesh1 = phoebe.ParameterSet(frame='phoebe', context='mesh:marching',
                                delta=delta, alg='c')
    mesh2 = phoebe.ParameterSet(frame='phoebe', context='mesh:marching',
                                delta=delta, alg='c')
    curve, params = wd.lc(ps, request='curve', light_curve=lc, rv_curve=rv)
    lcobs = phoebe.LCDataSet(columns=['time','flux','sigma'], time=curve['indeps'],
                             sigma=0.001*curve['lc'],
                             flux=curve['lc'], ref=lcdep1['ref'])
    rvobs1 = phoebe.RVDataSet(columns=['time','rv','sigma'], time=curve['indeps'],
                             sigma=0.001*curve['rv1']*100,
                             rv=curve['rv1']*100, ref=rvdep1['ref'])
    rvobs2 = phoebe.RVDataSet(columns=['time','rv','sigma'], time=curve['indeps'],
                             sigma=0.001*curve['rv2']*100,
                             rv=curve['rv2']*100, ref=rvdep2['ref'])

    star1 = phoebe.BinaryRocheStar(star1, orbit, mesh1, pbdep=[lcdep1, rvdep1], obs=[rvobs1])
    star2 = phoebe.BinaryRocheStar(star2, orbit, mesh2, pbdep=[lcdep2, rvdep2], obs=[rvobs2])
    system = phoebe.BodyBag([star1, star2], position=position, obs=[lcobs], label='system')
    logger.info("Successfully parsed WD lcin file {}".format(filename))
    return system


def phoebe_to_wd(system, create_body=False):
    """
    Convert a Phoebe2.0 system to WD parameterSets.
    
    Not finished yet!
    """
    system = system.copy()
    ps = parameters.ParameterSet('root', frame='wd')
    lc = parameters.ParameterSet('lc', frame='wd')
    rv = parameters.ParameterSet('rv', frame='wd')
    
    orbit = system[0].params['orbit']
    tools.to_supconj(orbit)
    comp1 = system[0].params['component']
    body1 = system[0]
    
    
    if len(system.bodies)==1: # assume overcontact
        ps['model'] = 'overcontact'
        comp2 = comp1
        body2 = body1
    else:
        ps['model'] = 'unconstrained'
        comp2 = system[1].params['component']
        body2 = system[1]
    
    
    position = system.params['position']
    mesh1 = system[0].params['mesh']
    
    ps['name'] = orbit['label']
    ps['hjd0'] = orbit['t0']
    ps['period'] = orbit['period']
    ps['dpdt'] = orbit['dpdt']
    ps['pshift'] = orbit['phshift']
    ps['sma'] = orbit.request_value('sma','Rsol'), 'Rsol'
    ps['rm'] = orbit['q']
    ps['ecc'] = orbit['ecc']
    ps['omega'] = orbit.request_value('per0','deg'), 'deg'
    ps['incl'] = orbit.request_value('incl','deg'), 'deg'
    
    ps['teff1'] = comp1.request_value('teff','K'), 'K'
    ps['teff2'] = comp2.request_value('teff','K'), 'K'
    ps['f1'] = comp1['syncpar']
    ps['f2'] = comp2['syncpar']
    ps['alb1'] = comp1['alb']
    ps['alb2'] = comp2['alb']
    ps['pot1'] = comp1['pot']
    ps['pot2'] = comp2['pot']
    ps['grb1'] = comp1['gravb']
    ps['grb2'] = comp2['gravb']
    ps['n1'] = marching.delta_to_gridsize(mesh1['delta'])
    ps['n2'] = marching.delta_to_gridsize(mesh1['delta'])
    ps['ipb'] = 1
    
    ps['ld_model'] = 'linear'#comp1['ld_func']
    
    if False:
        if isinstance(comp1['ld_coeffs'],str):
            ld_model = comp1['ld_func']
            atm_kwargs = dict(teff=ps.request_value('teff1','K'),logg=4.4)
            basename = '{}_{}{:02.0f}_{}_equidist_r_leastsq_teff_logg.fits'.format('kurucz','p',0,ld_model)        
            atm = os.path.join(limbdark.get_paths()[1],basename)
            passband = lcset.get_parameter('filter').get_choices()[lcset['filter']-1].upper()
            ldbol1 = limbdark.interp_ld_coeffs(atm, passband, atm_kwargs=atm_kwargs)
        else:
            ldbol1 = comp1['ld_coeffs']
        
        if isinstance(comp1['ld_coeffs'],str):
            ld_model = comp1['ld_func']
            atm_kwargs = dict(teff=ps.request_value('teff2','K'),logg=4.4)
            basename = '{}_{}{:02.0f}_{}_equidist_r_leastsq_teff_logg.fits'.format('kurucz','p',0,ld_model)        
            atm = os.path.join(limbdark.get_paths()[1],basename)
            passband = lcset.get_parameter('filter').get_choices()[lcset['filter']-1].upper()
            ldbol1 = limbdark.interp_ld_coeffs(atm, passband, atm_kwargs=atm_kwargs)
        else:
            ldbol1 = comp1['ld_coeffs']
        
        
        # at least one ld coefficient
        ps['ld_xbol1'] = ldbol1[0]
        ps['ld_xbol2'] = ldbol2[0]
        
        if comp1['ld_func'] != 'linear':
            ps['ld_ybol1'] = ldbol1[1]
            ps['ld_ybol2'] = ldbol2[1]
    
    ps['vga'] = position.request_value('vgamma', 'km/s')/100., 'km/s'
    
    # Light curve
    params = parameters.ParameterSet('compute')
    try:
        observatory.extract_times_and_refs(system, params, tol=1e-8)
    except ValueError:
        pass
    lc['indep_type'] = 'time (hjd)'
    lc['indep'] = params['time']
    
    pblum1 = body1.params['pbdep']['lcdep'].values()[0]['pblum']
    pblum2 = body2.params['pbdep']['lcdep'].values()[0]['pblum']
    
    ps['mode'] = 2
    if pblum1>=0:
        lc['hla'] = pblum1   
        ps['ipb'] = 0 # think it should be the other way around, maybe legacy_to_phoebe is wrong?
    if pblum2>=0:
        lc['cla'] = pblum2
        ps['ipb'] = 0
    lc['filter'] = body1.params['pbdep']['lcdep'].values()[0]['passband']
    ld_coeffs = body1.params['pbdep']['lcdep'].values()[0]['ld_coeffs'][0]
    if not isinstance(ld_coeffs,str):
        lc['ld_lcx1'] = ld_coeffs
    ld_coeffs = body2.params['pbdep']['lcdep'].values()[0]['ld_coeffs'][0]
    if not isinstance(ld_coeffs,str):
        lc['ld_lcx2'] = ld_coeffs
    
    lc['ref'] = body1.params['pbdep']['lcdep'].values()[0]['ref']
    
    try:
        ld_coeffs = body1.params['pbdep']['lcdep'].values()[0]['ld_coeffs'][1]
        if not isinstance(ld_coeffs,str):
            lc['ld_lcy1'] = ld_coeffs
        ld_coeffs = body2.params['pbdep']['lcdep'].values()[0]['ld_coeffs'][1]
        if not isinstance(ld_coeffs,str):
            lc['ld_lcy2'] = ld_coeffs
    except:
        pass
    
    if create_body:
        obs = system.params['obs']['lcobs'].values()
        return wd.BodyEmulator(ps,lcset=lc, rvset=rv, obs=obs)
    
    else:
        return ps, lc, rv




def from_supconj_to_perpass(orbit):
    """
    Convert an orbital set where t0 is superior conjunction to periastron passage.
    
    Typically, parameterSets coming from Wilson-Devinney or Phoebe Legacy
    have T0 as superior conjunction.
    
    Inverse function is L{from_perpass_to_supconj}.
    
    See Phoebe Scientific reference Eqs. (3.30) and (3.35).
    
    @param orbit: parameterset of frame C{phoebe} and context C{orbit}
    @type orbit: parameterset of frame C{phoebe} and context C{orbit}
    """
    t_supconj = orbit['t0']
    phshift = orbit['phshift']
    P = orbit['period']
    per0 = orbit.get_value('per0','rad')
    t0 = t_supconj + (phshift - 0.25 + per0/(2*np.pi))*P1203
    
    orbit['t0'] = t0
