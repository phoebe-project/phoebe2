import re
import logging
import numpy as np
from phoebe.parameters import parameters
from phoebe.backend import universe

logger = logging.getLogger("KELLY")

def legacy_to_phoebe(inputfile, create_body=False, mesh='wd'):
    """
    Convert a legacy PHOEBE file to the parameterSets. 
    
    Currently this file converts all values, the limits and 
    step size for each parameter in the orbit, but only the 
    value for all other parameters.
    
    Returns two components (containing a body, light curve 
    dependent parameter set and radial velocity dependent 
    parameter set.) and an orbit.
    
    @param inputfile: legacy phoebe parameter file
    @type inputfile: str
    @param mesh: if set to 'marching' and C{create_body=True}, a marching mesh
    will be added to the Body. Else, a WD mesh will be added.
    @type mesh: str (one of C{wd} or C{marching})
    """
    
    #-- initialise all the variables and lists needed for the rvdep parameters
    rv_dep = []
    prim_rv = 0
    sec_rv = 0
    rv_pb = []
    ld_coeffs1 = []
    ld_coeffs2 = []
    rv_file = []

    #-- initialise the orbital and component parameter sets. 
    orbit = parameters.ParameterSet(frame='phoebe',context='orbit',add_constraints=True)
    comp1 = parameters.ParameterSet(frame='phoebe',context='component',label='star1',
                                    add_constraints=True)
    comp2 = parameters.ParameterSet(frame='phoebe',context='component',label='star2',
                                    add_constraints=True)
    mesh1wd = parameters.ParameterSet(frame='phoebe',context='mesh:wd',add_constraints=True)
    mesh2wd = parameters.ParameterSet(frame='phoebe',context='mesh:wd',add_constraints=True)

    #-- Open the parameter file and read each line
    ff = open(inputfile,'r')
    lines = ff.readlines()
    for l in lines:
        #-- ignore initial comment
        if l[0]=='#': continue

        #-- try splitting the parameter name and value
        try: 
            key, val = l.split('=')
        except:
            logger.info("line " + l[:-1] + " could not be parsed")
            continue

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
            orbit['vgamma'] = (val,'km/s')  
        elif key == 'phoebe_vga.MAX':
            orbit.get_parameter('vgamma').set_limits(ulim=float(val))
        elif key == 'phoebe_vga.MIN':
            orbit.get_parameter('vgamma').set_limits(llim=float(val))
        elif key == 'phoebe_vga.STEP':
            orbit.get_parameter('vgamma').set_step(step=float(val))                     
                     
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
        elif key == 'phoebe_alb2.VAL':
            comp2['alb'] = val   
            alb2 = comp2['alb']

        elif key == 'phoebe_f1.VAL':
            comp1['syncpar'] = val  
        elif key == 'phoebe_f2.VAL':
            comp2['syncpar'] = val  

        elif key == 'phoebe_grb1.VAL':
            comp1['gravb'] = val    
        elif key == 'phoebe_grb2.VAL':
            comp2['gravb'] = val    

        elif key == 'phoebe_pot1.VAL':
            comp1['pot'] = val 
        elif key == 'phoebe_pot2.VAL':
            comp2['pot'] = val  

        elif key == 'phoebe_teff1.VAL':
            comp1['teff'] = (val,'K') 
        elif key == 'phoebe_teff2.VAL':
            comp2['teff'] = (val,'K')

        elif key == 'phoebe_reffect_switch':
            if val[1:-2] == "1":
                comp1['irradiator'] = True
                comp2['irradiator'] = True
            else:
                comp1['irradiator'] = True
                comp2['irradiator'] = True     
    

        elif key == 'phoebe_met1.VAL':
            comp1['abun'] = val     
        elif key == 'phoebe_met2.VAL':
            comp2['abun'] = val   
       
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
            lcdep1[index]['pblum'] = float(val)/(4*np.pi)
        if key == 'phoebe_cla.VAL': 
            lcdep2[index]['pblum'] = float(val)/(4*np.pi)
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
     

        if key == 'phoebe_rv_filter':
            rv_pb.append(val)
            
        if key == 'phoebe_rv_dep':
            rv_dep.append(val[2:-2])
            if val[2:-2] == 'Primary RV':
                prim_rv+=1
            else:
                sec_rv +=1
                
        if key == 'phoebe_rv_filename':
            rv_file.append(val[2:-2])
                    
    orbit['long_an'] = 0.
 
    comp1['ld_coeffs'] = [ld_xbol1,ld_ybol1]
    comp2['ld_coeffs'] = [ld_xbol2,ld_ybol2]   
    
    
    for i in range(lcno):    
        lcdep1[i]['atm'] = comp1['atm']
        lcdep1[i]['alb'] = comp1['alb'] 
        lcdep1[i]['ld_func'] = comp1['ld_func']
    
        lcdep2[i]['atm'] = comp1['atm']
        lcdep2[i]['alb'] = comp1['alb'] 
        lcdep2[i]['ld_func'] = comp1['ld_func']
        #-- make sure lables are the same
        lcdep2[i]['ref'] = lcdep1[i]['ref']
               
    rvdep1 = [parameters.ParameterSet(context='rvdep') for i in range(prim_rv)]
    rvdep2 = [parameters.ParameterSet(context='rvdep') for i in range(sec_rv)]
  
        
    j=0
    k=0
    for i in range(rvno):

        if rv_dep[i] == 'Primary RV':
            rvdep1[j]['ld_coeffs'] = ld_coeffs1[i]
            rvdep1[j]['passband'] = rv_pb[i]
            rvdep1[j]['ref'] = rv_file[i]
            rvdep1[j]['atm'] = comp1['atm']
            rvdep1[j]['alb'] = comp1['alb']
            if rv_file[i] != "Undefined":
                rvdep1[j]['ref'] = rv_file[i]
            j+=1
            
        else:
            rvdep2[k]['ld_coeffs'] = ld_coeffs2[i] 
            rvdep2[k]['passband'] = rv_pb[i]
            rvdep2[k]['atm'] = comp2['atm']
            rvdep2[k]['alb'] = comp2['alb']
            if rv_file[i] != "Undefined":
                rvdep2[k]['ref'] = rv_file[i]            
            k+=1
            
    #-- copy the component labels to the orbits
    orbit['c1label'] = comp1['label']
    orbit['c2label'] = comp2['label']
    
    # t0 is the time of superior conjunction in Phoebe Legacy
    orbit['t0type'] = 'superior conjunction'


    body1 = comp1, lcdep1, rvdep1
    body2 = comp2, lcdep2, rvdep2  
    
    logger.info("Loaded contents from file {}".format(inputfile))
    
    if create_body:
        if mesh=='marching':
            mesh1 = parameters.ParameterSet(context='mesh:marching')
            mesh2 = parameters.ParameterSet(context='mesh:marching')
            #-- empirical calibration (conversion between gridsize from WD and
            #   pyphoebe marching delta)
            mesh1['delta'] = 10**(-0.98359345*np.log10(mesh1wd['gridsize'])+0.4713824)
            mesh2['delta'] = 10**(-0.98359345*np.log10(mesh2wd['gridsize'])+0.4713824)
        else:
            mesh1 = mesh1wd
            mesh2 = mesh2wd
        star1 = universe.BinaryRocheStar(comp1,orbit,mesh1,pbdep=lcdep1+rvdep1)
        star2 = universe.BinaryRocheStar(comp2,orbit,mesh2,pbdep=lcdep2+rvdep2)
   
        bbag = universe.BodyBag([star1,star2],solve_problems=True)
        return bbag
     
    return body1, body2, orbit 




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