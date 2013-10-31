"""
Parse data and parameter files.

.. autosummary::

    legacy_to_phoebe
    
"""

import re
import os
import logging
import glob
import logging.handlers
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
from phoebe.frontend.bundle import Bundle
import matplotlib.pyplot as plt
import os.path

logger = logging.getLogger('PARSER')

def legacy_to_phoebe(inputfile, create_body=False, create_bundle=False,
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
    comp1 = parameters.ParameterSet(frame='phoebe',context='component',label='myprimary',
                                    add_constraints=True)
    comp2 = parameters.ParameterSet(frame='phoebe',context='component',label='mysecondary',
                                    add_constraints=True)
    globals = parameters.ParameterSet('globals')
    
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

        #-- for each phoebe parameter incorportae an if statement
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
        elif key == 'phoebe_dpdt.MIN':
            orbit.get_parameter('dpdt').set_limits(llim=float(val)*31557600)
        elif key == 'phoebe_dpdt.STEP':
            orbit.get_parameter('dpdt').set_step(step=float(val)*31557600) 
               
        elif key == 'phoebe_dperdt.VAL':
            orbit['dperdt'] = (val,'rad/d')    
        elif key == 'phoebe_dperdt.MAX':
            orbit.get_parameter('dperdt').set_limits(ulim=float(val)*np.pi*365.25/180)
        elif key == 'phoebe_dperdt.MIN':
            orbit.get_parameter('dperdt').set_limits(llim=float(val)*np.pi*365.25/180)
        elif key == 'phoebe_dperdt.STEP':
            orbit.get_parameter('dperdt').set_step(step=float(val)*np.pi*365.25/180)                    
                     
        elif key == 'phoebe_ecc.VAL':
            orbit['ecc'] = val   
        elif key == 'phoebe_ecc.MAX':
            orbit.get_parameter('ecc').set_limits(ulim=float(val))
        elif key == 'phoebe_ecc.MIN':
            orbit.get_parameter('ecc').set_limits(llim=float(val))    
        elif key == 'phoebe_ecc.STEP':
            orbit.get_parameter('ecc').set_step(step=float(val))            
            
        elif key == 'phoebe_hjd0.VAL':
            orbit['t0'] = (val,'JD') 
        elif key == 'phoebe_hjd0.MAX':
            orbit.get_parameter('t0').set_limits(ulim=float(val))
        elif key == 'phoebe_hjd0.MIN':
            orbit.get_parameter('t0').set_limits(llim=float(val))            
        elif key == 'phoebe_hjd0.STEP':
            orbit.get_parameter('t0').set_step(step=float(val))            
            
        elif key == 'phoebe_incl.VAL':
            orbit['incl'] = (val,'deg') 
        elif key == 'phoebe_incl.MAX':
            orbit.get_parameter('incl').set_limits(ulim=float(val))
        elif key == 'phoebe_incl.MIN':
            orbit.get_parameter('incl').set_limits(llim=float(val))                        
        elif key == 'phoebe_incl.STEP':
            orbit.get_parameter('incl').set_step(step=float(val))            
            
        elif key == 'phoebe_period.VAL':
            orbit['period'] = (val,'d')
        elif key == 'phoebe_period.MAX':
            orbit.get_parameter('period').set_limits(ulim=float(val))
        elif key == 'phoebe_period.MIN':
            orbit.get_parameter('period').set_limits(llim=float(val))                        
        elif key == 'phoebe_period.STEP':
            orbit.get_parameter('period').set_step(step=float(val))          
            
        elif key == 'phoebe_perr0.VAL':
            orbit['per0'] = (val,'rad')
        elif key == 'phoebe_perr0.MAX':
            orbit.get_parameter('per0').set_limits(ulim=float(val)/np.pi*180)
        elif key == 'phoebe_perr0.MIN':
            orbit.get_parameter('per0').set_limits(llim=float(val)/np.pi*180)            
        elif key == 'phoebe_perr0.STEP':
            orbit.get_parameter('per0').set_step(step=float(val)/np.pi*180)           
            
        elif key == 'phoebe_pshift.VAL':
            orbit['phshift'] = val   
        elif key == 'phoebe_pshift.MAX':
            orbit.get_parameter('phshift').set_limits(ulim=float(val))
        elif key == 'phoebe_pshift.MIN':
            orbit.get_parameter('phshift').set_limits(llim=float(val))            
        elif key == 'phoebe_pshift.STEP':
            orbit.get_parameter('phshift').set_step(step=float(val))                        
            
        elif key == 'phoebe_rm.VAL':
            orbit['q'] = val
        elif key == 'phoebe_rm.MAX':
            orbit.get_parameter('q').set_limits(ulim=float(val))
        elif key == 'phoebe_rm.MIN':
            orbit.get_parameter('q').set_limits(llim=float(val))
        elif key == 'phoebe_rm.STEP':
            orbit.get_parameter('q').set_step(step=float(val))                     
                       
        elif key == 'phoebe_vga.VAL':
            globals['vgamma'] = (val,'km/s')  
        elif key == 'phoebe_vga.MAX':
            globals.get_parameter('vgamma').set_limits(ulim=float(val))
        elif key == 'phoebe_vga.MIN':
            globals.get_parameter('vgamma').set_limits(llim=float(val))
        elif key == 'phoebe_vga.STEP':
            globals.get_parameter('vgamma').set_step(step=float(val))                     
                     
        elif key == 'phoebe_sma.VAL':
            orbit['sma'] = (val,'Rsol')    
        elif key == 'phoebe_sma.MAX':
            orbit.get_parameter('sma').set_limits(ulim=float(val))
        elif key == 'phoebe_sma.MIN':
            orbit.get_parameter('sma').set_limits(llim=float(val))
        elif key == 'phoebe_sma.STEP':
            orbit.get_parameter('sma').set_step(step=float(val))                     
        
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
            comp1['alb'] = val
            # Switch on heating in the other component
            if comp1['alb'] > 0:
                comp2['irradiator'] = True
        elif key == 'phoebe_alb1.MAX':
            comp1.get_parameter('alb').set_limits(ulim=float(val))
        elif key == 'phoebe_alb1.MIN':
            comp1.get_parameter('alb').set_limits(llim=float(val))
        elif key == 'phoebe_alb1.STEP':
            comp1.get_parameter('alb').set_step(step=float(val))   

        elif key == 'phoebe_alb2.VAL':
            comp2['alb'] = val   
            alb2 = comp2['alb']
            # Switch on heating in the oterh component
            if comp2['alb'] > 0:
                comp1['irradiator'] = True
        elif key == 'phoebe_alb2.MAX':
            comp2.get_parameter('alb').set_limits(ulim=float(val))
        elif key == 'phoebe_alb2.MIN':
            comp2.get_parameter('alb').set_limits(llim=float(val))
        elif key == 'phoebe_alb2.STEP':
            comp2.get_parameter('alb').set_step(step=float(val))   

        elif key == 'phoebe_f1.VAL':
            comp1['syncpar'] = val  
        elif key == 'phoebe_f1.MAX':
            comp1.get_parameter('syncpar').set_limits(ulim=float(val))
        elif key == 'phoebe_f1.MIN':
            comp1.get_parameter('syncpar').set_limits(llim=float(val))
        elif key == 'phoebe_f1.STEP':
            comp1.get_parameter('syncpar').set_step(step=float(val))   

        elif key == 'phoebe_f2.VAL':
            comp2['syncpar'] = val  
        elif key == 'phoebe_f2.MAX':
            comp2.get_parameter('syncpar').set_limits(ulim=float(val))
        elif key == 'phoebe_f2.MIN':
            comp2.get_parameter('syncpar').set_limits(llim=float(val))
        elif key == 'phoebe_f2.STEP':
            comp2.get_parameter('syncpar').set_step(step=float(val))   

        elif key == 'phoebe_grb1.VAL':
            comp1['gravb'] = val    
        elif key == 'phoebe_grb1.MAX':
            comp1.get_parameter('gravb').set_limits(ulim=float(val))
        elif key == 'phoebe_grb1.MIN':
            comp1.get_parameter('gravb').set_limits(llim=float(val))
        elif key == 'phoebe_grb1.STEP':
            comp1.get_parameter('gravb').set_step(step=float(val))   

        elif key == 'phoebe_grb2.VAL':
            comp2['gravb'] = val    
        elif key == 'phoebe_grb2.MAX':
            comp2.get_parameter('gravb').set_limits(ulim=float(val))
        elif key == 'phoebe_grb2.MIN':
            comp2.get_parameter('gravb').set_limits(llim=float(val))
        elif key == 'phoebe_grb2.STEP':
            comp2.get_parameter('gravb').set_step(step=float(val))   

        elif key == 'phoebe_pot1.VAL':
            comp1['pot'] = val 
        elif key == 'phoebe_pot1.MAX':
            comp1.get_parameter('pot').set_limits(ulim=float(val))
        elif key == 'phoebe_pot1.MIN':
            comp1.get_parameter('pot').set_limits(llim=float(val))
        elif key == 'phoebe_pot1.STEP':
            comp1.get_parameter('pot').set_step(step=float(val))   

        elif key == 'phoebe_pot2.VAL':
            comp2['pot'] = val  
        elif key == 'phoebe_pot2.MAX':
            comp2.get_parameter('pot').set_limits(ulim=float(val))
        elif key == 'phoebe_pot2.MIN':
            comp2.get_parameter('pot').set_limits(llim=float(val))
        elif key == 'phoebe_pot2.STEP':
            comp2.get_parameter('pot').set_step(step=float(val))   

        elif key == 'phoebe_teff1.VAL':
            comp1['teff'] = (val,'K') 
        elif key == 'phoebe_teff1.MAX':
            comp1.get_parameter('teff').set_limits(ulim=float(val))
        elif key == 'phoebe_teff1.MIN':
            comp1.get_parameter('teff').set_limits(llim=float(val))
        elif key == 'phoebe_teff1.STEP':
            comp1.get_parameter('teff').set_step(step=float(val))   

        elif key == 'phoebe_teff2.VAL':
            comp2['teff'] = (val,'K')
        elif key == 'phoebe_teff2.MAX':
            comp2.get_parameter('teff').set_limits(ulim=float(val))
        elif key == 'phoebe_teff2.MIN':
            comp2.get_parameter('teff').set_limits(llim=float(val))
        elif key == 'phoebe_teff2.STEP':
            comp2.get_parameter('teff').set_step(step=float(val))   

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
        elif key == 'phoebe_met1.MIN':
            comp1.get_parameter('abun').set_limits(llim=float(val))
        elif key == 'phoebe_met1.STEP':
            comp1.get_parameter('abun').set_step(step=float(val))   

        elif key == 'phoebe_met2.VAL':
            comp2['abun'] = val   
        elif key == 'phoebe_met2.MAX':
            comp2.get_parameter('abun').set_limits(ulim=float(val))
        elif key == 'phoebe_met2.MIN':
            comp2.get_parameter('abun').set_limits(llim=float(val))
        elif key == 'phoebe_met2.STEP':
            comp2.get_parameter('abun').set_step(step=float(val))   
    
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
            if root is not None:
                filename = os.path.join(root, os.path.basename(filename))
            rv_file.append(filename)
            
        if key == 'phoebe_lc_filename':
            filename = val[2:-2]
            # If a root directory is given, set the filepath to be relative
            # from there (but store absolute path name)
            if root is not None:
                filename = os.path.join(root, os.path.basename(filename))
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
        
        if lc_file[i] != "Undefined":
            if lcsigma[i] == 'undefined': 
                if lctime[i]=='time':
                    if os.path.isfile(lc_file[i]) or os.path.isfile(os.path.basename(lc_file[i])):
                        col1lc,col2lc = np.loadtxt(lc_file[i], unpack=True)
                        obslc.append(datasets.LCDataSet(time=col1lc, flux=col2lc,columns=[lctime[i],'flux'], 
                        ref="lightcurve_"+str(j), filename=str(lc_file[i]), statweight=lc_pbweight[i], user_components=lcname[i]))
                    else:
                        logger.warning("The light curve file {} cannot be located.".format(lc_file[i]))                    
                else:
                    if os.path.isfile(lc_file[i]) or os.path.isfile(os.path.basename(lc_file[i])):
                        col1lc,col2lc = np.loadtxt(lc_file[i], unpack=True)
                        obslc.append(datasets.LCDataSet(phase=col1lc, flux=col2lc,columns=[lctime[i],'flux'], 
                        ref="lightcurve_"+str(j), filename=str(lc_file[i]), statweight=lc_pbweight[i], user_components=lcname[i]))
                    else:
                        logger.warning("The light curve file {} cannot be located.".format(lc_file[i]))                    
            else:
                if lctime[i]=='time':
                    if lcsigma[i]=='sigma': 
                        if os.path.isfile(lc_file[i]) or os.path.isfile(os.path.basename(lc_file[i])):
                            col1lc,col2lc,col3lc = np.loadtxt(lc_file[i], unpack=True)
                            obslc.append(datasets.LCDataSet(time=col1lc,flux=col2lc,sigma=col3lc,columns=[lctime[i],'flux',lcsigma[i]], 
                            ref="lightcurve_"+str(j), filename=str(lc_file[i]), statweight=lc_pbweight[i], user_components=lcname[i]))
                        else:
                            logger.warning("The light curve file {} cannot be located.".format(lc_file[i]))                    
                    else:
                        if os.path.isfile(lc_file[i]) or os.path.isfile(os.path.basename(lc_file[i])):
                            col1lc,col2lc,col3lc = np.loadtxt(lc_file[i], unpack=True)
                            obslc.append(datasets.LCDataSet(time=col1lc,flux=col2lc,sigma=1./col3lc**2,columns=[lctime[i],'flux','sigma'], 
                            ref="lightcurve_"+str(j), filename=str(lc_file[i]), statweight=lc_pbweight[i], user_components=lcname[i]))
                        else:
                            logger.warning("The light curve file {} cannot be located.".format(lc_file[i]))                    
                else:
                    if lcsigma[i]=='sigma': 
                        if os.path.isfile(lc_file[i]) or os.path.isfile(os.path.basename(lc_file[i])):
                            col1lc,col2lc,col3lc = np.loadtxt(lc_file[i], unpack=True)
                            obslc.append(datasets.LCDataSet(phase=col1lc,flux=col2lc,sigma=col3lc,columns=[lctime[i],'flux',lcsigma[i]], 
                            ref="lightcurve_"+str(j), filename=str(lc_file[i]), statweight=lc_pbweight[i], user_components=lcname[i]))
                        else:
                            logger.warning("The light curve file {} cannot be located.".format(lc_file[i]))                    
                    else:
                        if os.path.isfile(lc_file[i]) or os.path.isfile(os.path.basename(lc_file[i])):
                            col1lc,col2lc,col3lc = np.loadtxt(lc_file[i], unpack=True)
                            obslc.append(datasets.LCDataSet(phase=col1lc,flux=col2lc,sigma=1./col3lc**2,columns=[lctime[i],'flux','sigma'], 
                            ref="lightcurve_"+str(j), filename=str(lc_file[i]), statweight=lc_pbweight[i], user_components=lcname[i]))
                        else:
                            logger.warning("The light curve file {} cannot be located.".format(lc_file[i]))                    
 
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
    orbit['label'] = 'myorbit'
    
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
    
    if create_bundle or create_body:
        
        ##need an if statement here incase no obs
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
            bodybag = universe.BodyBag([star1,star2],solve_problems=True, globals=globals,obs=obslc)
        else:
            bodybag = universe.BodyBag([star1,star2],solve_problems=True, globals=globals)
        
        # Set the name of the thing
        bodybag.set_label(system_label)
        
    if create_bundle:

        bundle = Bundle(bodybag)
        
        return bundle
    
    if create_body:
        return bodybag
    
    body1 = comp1, mesh1, lcdep1, rvdep1, obsrv1
    body2 = comp2, mesh2, lcdep2, rvdep2, obsrv2
    logger.info("Successfully parsed Phoebe Legacy file {}".format(inputfile))
    return body1, body2, orbit, globals, obslc 



def wd_to_phoebe(filename, mesh='marching', create_body=True):
    if not create_body:
        raise NotImplementedError('create_body=False')
    if not mesh=='marching':
        raise NotImplementedError("mesh!='marching'")
    
    ps, lc, rv = wd.lcin_to_ps(filename, version='wd2003')
    comp1, comp2, orbit, globals = wd.wd_to_phoebe(ps, lc, rv)
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
    system = phoebe.BodyBag([star1, star2], obs=[lcobs], label='system')
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
    
    
    globals = system.params['globals']
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
    
    ps['vga'] = globals.request_value('vgamma', 'km/s')/100., 'km/s'
    
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
    lc['ld_lcx1'] = body1.params['pbdep']['lcdep'].values()[0]['ld_coeffs'][0]
    lc['ld_lcx2'] = body2.params['pbdep']['lcdep'].values()[0]['ld_coeffs'][0]
    lc['ref'] = body1.params['pbdep']['lcdep'].values()[0]['ref']
    
    try:
        lc['ld_lcy1'] = body1.params['pbdep']['lcdep'].values()[0]['ld_coeffs'][1]
        lc['ld_lcy2'] = body2.params['pbdep']['lcdep'].values()[0]['ld_coeffs'][1]
    except IndexError:
        logger.warning("Second passband LD coefficients not parsed; ld_model={}".format(comp1['ld_func']))
    
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
    t0 = t_supconj + (phshift - 0.25 + per0/(2*np.pi))*P
    orbit['t0'] = t0
