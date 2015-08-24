
import logging
import numpy as np
from phoebe.backend import universe
from phoebe.algorithms import marching
from phoebe.units.conversions import convert
from phoebe.dynamics import keplerorbit
from copy import deepcopy
import commands
import os

try:
    import phoebeBackend as phb1
except ImportError:
    pass

logger = logging.getLogger("ALT_BACKENDS")
logger.addHandler(logging.NullHandler())

def translate_filter(val):
#    if val[2:5] == 'Bes':
#        val = val[2:-2].upper()
#        val = ".".join(val.split('L IR:'))
#        if val[0:8] == 'BESSEL.L':
#            val = 'BESSEL.LPRIME'
    if val == 'BESSEL.LPRIME':
        val = 'BESSEL.L'
    elif val == 'COROT.SIS':
        val = 'COROT.SISMO'
    elif val =='KEPLER.MEAN':
        val = 'KEPLER.MEAN'
    elif val == 'IRAC.36':
        val = 'IRAC.CH1'
    elif val == 'MOST.V':
        val = 'MOST.DEFAULT'
    elif val == 'STROMGREN.HBN':
        val = 'STROMGREN.HBETA_NARROW'
    elif val == 'STROMGREN.HBW':
        val = 'STROMGREN.HBETA_WIDE'
    elif val == 'OPEN.BOL':
        val = 'BOLOMETRIC.3000A-10000A'
    elif val == 'TYCHO.BT':
        val = 'HIPPARCOS.BT'
    elif val == 'TYCHO.VT':
        val = 'HIPPARCOS.VT'
    val = (val[0]+val[1:-1].lower()+val[-1]).replace('.', ':')
    return val
#still need to find this filter and check others


def set_ld_legacy(name, value, on=None, comp=None):
    lds = {'component':'phoebe_ld_#bol%', 'lcdep':'phoebe_ld_lc#%', 'rvdep':'phoebe_ld_rv#%'}
    comps = {'prim':'1', 'sec':'2'}
    letter = {0:'x', 1:'y'}
    if on != None:
        if type(value) != list:
            value = [value]
        for y in range(len(value)):
            #quit()
#            print lds[name].replace('#', letter[y]).replace(('%', comps[comp])
#            quit()
#            print lds[name].replace('#', letter[y]).replace('%', comps[comp])+'['+str(on)+']'
            try:
#                phb1.setpar(lds[name].replace('#', letter[y]).replace('%', comps[comp])+'['+str(on)+']', value[y]) 
                phb1.setpar(lds[name].replace('#', letter[y]).replace('%', comps[comp]), value[y]) 
            except:
                logger.warning('ld coeffs must be numbers')
            if name == 'rvdep':
                quit()
#            quit()
    else:
        for y in len(value):
            try:
                phb1.setpar(lds[name].replace('#', letter[y]).replace('%', comps[comp]), value[y]) 
            except:
                logger.warning('ld coeffs must be numbers')
        


def set_param_legacy(ps, param, value, on = None, ty=None, comp=None, un=None):
    params = {'morphology':{'name':'phoebe_model'}, 'asini':{'name':'phoebe_asini_value'},'filename':{'name':'phoebe_#_filename'}, 'statweight':{'name':'phoebe_#_sigma'},'passband':{'name':'phoebe_#_filter'},'columns':{},'Rv':{'name':'phoebe_ie_factor'},'extinction':{'name':'phoebe_ie_excess'},'method':{'name':'phoebe_proximity_rv#_switch'},'incl':{'name':'phoebe_incl','unit':'deg'}, 't0':{'name':'phoebe_hjd0'}, 'period':{'name':'phoebe_period', 'unit':'d'}, 'dpdt':{'name':'phoebe_dpdt', 'unit':'d/d'}, 'phshift':{'name':'phoebe_pshift'}, 'sma':{'name': 'phoebe_sma', 'unit':'Rsun'}, 'q':{'name':'phoebe_rm'}, 'vgamma':{'name':'phoebe_vga', 'unit':'km/s'},  'teff':{'name':'phoebe_teff#', 'unit':'K'}, 'pot':{'name':'phoebe_pot#'}, 'abun':{'name':'phoebe_met#'},  'syncpar':{'name':'phoebe_f#'}, 'alb':{'name':'phoebe_alb#'}, 'gravb':{'name':'phoebe_grb#'}, 'ecc':{'name':'phoebe_ecc'}, 'per0':{'name':'phoebe_perr0','unit':'rad'}, 'dperdt':{'name':'phoebe_dperdt','unit':'rad/d'}, 'ld_func':{'name':'phoebe_ld_model'}, 'pblum':{}, 'atm':{'name':'phoebe_atm#_switch'},'refl':{'name':'phoebe_reffect_switch'},'refl_num':{'name':'phoebe_reffect_reflections'},'gridsize':{'name':'phoebe_grid_finesize#'},'delta':{'name':'phoebe_grid_finesize#'},'scale':{'name':'phoebe_compute_hla_switch'}, 'offset':{'name':'phoebe_compute_vga_switch'}}

    value = deepcopy(value) #DO NOT ERASE. if this is not done python will occasionally replace values in the system with values here.

    boolean = {True:1, False:0}
    comps = {'prim':'1', 'sec':'2'}

    if param not in params.keys():
        logger.warning('{} parameter ignored in phoebe legacy'.format(param))

    elif param=='t0':
        if ps.get_value('t0type') == 'periastron passage':
            value = keplerorbit.from_perpass_to_supconj(ps.get_value('t0', 'JD'), ps.get_value('period', 'd'), ps.get_value('per0', 'rad'))
        else:
            value = ps.get_value('t0', 'JD')
        phb1.setpar(params[param]['name'], value)

    elif param == 'filename':
        phb1.setpar(params[param]['name'].replace('#', str(ty)), value)
    elif param == 'scale':
        valn = ps.get_adjust('scale')  
        phb1.setpar(params[param]['name'], boolean[valn])        
    elif param == 'offset':
        valn = ps.get_adjust('offset')
        phb1.setpar(params[param]['name'], boolean[valn])
    elif param =='morphology':
        #add more types when supported

        morphs = {'detached':'Detached binary', 'unconstrained':'Unconstrained binary system'}
        phb1.setpar(params[param]['name'].replace('#', comps[comp]), morphs[value])

    elif param =='asini':
        print params[param]['name']
#        try:
#            phb1.setpar('phoebe_asini_switch', boolean[True])
#            phb1.setpar(params[param]['name'].replace('#', comps[comp]), value)
#        except:
#            logger.warning('{} parameter not implemented in phoebe legacy'.format('phoebe_asini_switch'))
    elif param == 'columns':
        vars = {'time':'Time (HJD)', 'phase':'Phase','flux':'Flux','weight':'Standard weight', 'sigma':'Standard deviation', 'rv1':'Primary RV', 'rv2':'Secondary RV'}  
        
        phb1.setpar('phoebe_indep', vars[value[0]])

        print "value[1] before", value[1]
        if ty == 'rv':
            if value[1] != 'rv1' and value[1] != 'rv2':
                value[1]=value[1]+comps[comp]

        print "value[1]", value[1]
        print "comps[comp]", comp
        print "ty", ty
        print 'phoebe_#_indep'.replace('#', str(ty))
        print 'phoebe_#_dep'.replace('#', str(ty))
        print vars[value[1]]
        phb1.setpar('phoebe_#_indep'.replace('#', str(ty)), vars[value[0]])
        phb1.setpar('phoebe_#_dep'.replace('#', str(ty)), vars[value[1]])
        if len(value)==3:
            phb1.setpar('phoebe_#_indweight'.replace('#', str(ty)), vars[value[2]])
   
    elif param == 'passband':
         
        if ps.get_context() != 'reddening:interstellar':

            value = translate_filter(value)
            phb1.setpar(params[param]['name'].replace('#', str(ty)), value)
    elif param == 'statweight':
         phb1.setpar(params[param]['name'].replace('#', str(ty)), value)   
# some functions parameters need to be converted
    elif param == 'alb':
#        print value, (1-value)
        phb1.setpar(params[param]['name'].replace('#', comps[comp]), (1.-value))
    elif param == 'extinction':
        print ps.get_context()
        if ps.get_context() == 'reddening:interstellar':

           rv =  ps['Rv']
           valn = value/rv
           phb1.setpar(params[param]['name'], valn)
        else:
            print "phoebe 1 doesn't use multiple lightcurve dependent extintions"    
    elif param =='method':
        if value == 'flux-weighted':
            phb1.setpar(params[param]['name'].replace('#', comps[comp]), 0)
        else:
            phb1.setpar(params[param]['name'].replace('#', comps[comp]), 1)

    elif param == 'atm':

        if (value == 'kurucz') or ('atmcof.dat' in value):
            print comp
            phb1.setpar(params[param]['name'].replace('#', comps[comp]), 1)

        elif value == 'blackbody' or ('atmcofplanck.dat' in value):
            phb1.setpar(params[param]['name'].replace('#', comps[comp]), 0)

        else:
            raise TypeError('Atmosphere not compatible with Phoebe 1. Parameter could not be set')
    elif param =='pblum':
        lum = {'prim':'phoebe_hla', 'sec':'phoebe_cla'}
        if value != -1:
            phb1.setpar(lum[comp], value)
        else:
            phb1.setpar('phoebe_usecla_switch', boolean[False])
            
    elif param == 'refl':
            if comp != None:
                phb1.setpar(params[param]['name'].replace('#', comps[comp]), boolean[value])
            else:
                phb1.setpar(params[param]['name'], boolean[value])
    elif param == 'delta':
        #marching.delta_to_gridsize is wrong 
        valn =  int(np.round(marching.delta_to_gridsize(value)))
        phb1.setpar(params[param]['name'].replace('#', comps[comp]), valn)
#        phb1.setpar('phoebe_grid_finesize1', valn)
    
    elif param == 'ld_func':    
        if value == 'logarithmic':
            phb1.setpar(params[param]['name'].replace('#', comps[comp]), 'Logarithmic law')
        elif value == 'linear':
            phb1.setpar(params[param]['name'].replace('#', comps[comp]), 'Linear cosine law')
        elif value =='square_root':
            phb1.setpar(params[param]['name'].replace('#', comps[comp]), 'Square Root Law')
            
            
        
    else: 
        qualifier = params[param]
        print qualifier
        print un
        if 'unit' in qualifier.keys() and qualifier['unit']!= un:
#            print "I WENT HERE FIRST?"
            nval = convert(un, qualifier['unit'], value)
            if comp is None:
                phb1.setpar(params[param]['name'], nval)
            else:
                phb1.setpar(params[param]['name'].replace('#', comps[comp]), nval)
   
        else:
            if comp is None:
                phb1.setpar(params[param]['name'], value)
            else:
                phb1.setpar(params[param]['name'].replace('#', comps[comp]), value)
    
def compute_legacy(system, *args, **kwargs):
    
    compute = kwargs['params']
  
    # import phoebe legacy    
    
    # check to make sure only binary
    if not hasattr(system, 'bodies') or len(system.bodies) != 2:
        raise TypeError("object must be a binary to run phoebe legacy")
        return
        
    # check to make sure BRS
    if not all([isinstance(body, universe.BinaryRocheStar) for body in system.bodies]):
        raise TypeError("both stars must be BinaryRocheStars to run phoebe legacy")
        return

    #initialize phoebe 1
    phb1.init()
    phb1.configure()
    #boolean definitions for use later with various parameters
    boolean = {True:1, False:0}
    parsets = {'orbit':'system', 'position':'system', 'reddening':'system', 'component':'comp', 'mesh':'comp', 'compute':'compute'}
    compsi = {1:'prim', 2:'sec'}
    #determine and initialize data curves

#    lcno = len(system1[0].params['pbdep']['lcdep'])
#    rvno = len(system1[0].params['pbdep']['rvdep'])+len(system1[1].params['pbdep']['rvdep'])
    
#    phb.setpar("phoebe_lcno", lcno)
#    phb.setpar("phoebe_rvno", rvno)
    refs = system.get_refs(per_category=True)
    lcnames = refs.get('lc', [])
    rvnames = refs.get('rv', [])

    for x in range(len(rvnames)):                           
        try:                                                                                         
            system[0].params['obs']['rvobs'][rvnames[x]]    
            system[1].params['obs']['rvobs'][rvnames[x]]    
            rvnames.append(rvnames[x])                      
        except:
            pass

    print len(rvnames), rvnames
#    quit()
    phb1.setpar("phoebe_lcno", len(lcnames))
    phb1.setpar("phoebe_rvno", len(rvnames))


    # add lcdep and lcobs

    for i in range(len(lcnames)):
        on = i+1

        psd = system[0].params['pbdep']['lcdep'][lcnames[i]]
        psd2 = system[1].params['pbdep']['lcdep'][lcnames[i]]
        for param in psd:
            if param =='ld_coeffs':
                print 'ld_coeffs', param
                set_ld_legacy('lcdep', psd.get_value(param), on=on, comp='prim')
                set_ld_legacy('lcdep', psd2.get_value(param), on=on, comp='sec')
            else:

                if psd.get_value(param) != psd2.get_value(param):
                    set_param_legacy(psd, param, psd.get_value(param), on=on, ty='lc', comp='prim')
                    set_param_legacy(psd, param, psd.get_value(param), on=on, ty='lc',comp='sec')
                else:
                    set_param_legacy(psd, param, psd.get_value(param), on=on, ty='lc', comp='prim')
        #check for corresponding obs file and load its parameters as well

#        if lcnames[i] in system.params['obs']['lcobs']:
        if 'lcobs' in system.params['obs'] and lcnames[i] in system.params['obs']['lcobs']:
            pso = system.params['obs']['lcobs'][lcnames[i]]
            #determine if lc is active
#            phb1.setpar('phoebe_lc_active',boolean[pso.get_enabled()]) 

            for param in pso:
                set_param_legacy(pso, param, pso.get_value(param), on=on, ty='lc')
#    quit()
    # add rvdeps and rvobs

    for i in range(len(rvnames)):
        on = i+1
        print len(rvnames), rvnames
        print "this is i", i
        psd = system[i].params['pbdep']['rvdep'][rvnames[i]]
        for param in psd:
            if param =='ld_coeffs':
                print 'ld_coeffs', param
                 
                set_ld_legacy('lcdep', psd.get_value(param), on=on, comp=compsi[on])
#                set_ld_legacy('lcdep', psd2.get_value(param), on=on, comp='sec')
            else:


#                if psd.get_value(param) != psd2.get_value(param):
                set_param_legacy(psd, param, psd.get_value(param), on=on, ty='rv', comp =compsi[on])
#                    set_param_legacy(psd, param, psd.get_value(param), on=on, ty='rv',comp='sec')
#                else:
#                    set_param_legacy(psd, param, psd.get_value(param), on=on, ty='rv', comp='prim')
            
        if 'rvobs' in system[i].params['obs'] and rvnames[i] in system[i].params['obs']['rvobs']:
            pso = system[i].params['obs']['rvobs'][rvnames[i]]
            #determine if rv is active
#            phb1.setpar('phoebe_rv_active['+str(i+1)+']',boolean[pso.get_enabled()])
            for param in pso:
                print compsi[on]
                set_param_legacy(pso,param, pso.get_value(param), on=on, ty='rv',comp=compsi[on])
    
    # add all other parameter sets
    
    for x in parsets:
        if parsets[x] == 'system':
            if x == 'orbit':
                print 1, x
                ps = system[0].params[x]
            else:
                ps = system.params[x]
            for param in ps:
                print param, x
                
                try:
                    unit = ps.get_unit(param)
                    set_param_legacy(ps, param, ps.get_value(param), un=unit)
                except:
                    set_param_legacy(ps, param, ps.get_value(param))

        elif parsets[x] == 'comp':
                print x, parsets[x]
                ps = system[0].params[x]
                ps2 = system[1].params[x] 
                if ps.get_context() == ps2.get_context():   
                    for param in ps:
                         print param
                         if param == 'ld_coeffs':
                            set_ld_legacy(x, ps.get_value(param), on=on, comp='prim')
                            set_ld_legacy(x, ps2.get_value(param), on=on, comp='sec')
                         else:
                            try:
                                unit = ps.get_unit(param)
                                set_param_legacy(ps, param, ps.get_value(param), un=unit, comp='prim')
                                set_param_legacy(ps2, param, ps2.get_value(param), un=unit, comp='sec')
                            except:
                                set_param_legacy(ps, param, ps.get_value(param), comp='prim')
                                set_param_legacy(ps2, param, ps2.get_value(param), comp='sec')
                else:
                    for param in ps:
                        if param == 'ld_coeffs':
                            set_ld_legacy('lcdep', ps.get_value(param), on=on, comp='prim') 
                        else:
                            try:
                                unit = ps.get_unit(param)
                                set_param_legacy(ps, param, ps.get_value(param), un=unit, comp='prim')
                            except:
                                set_param_legacy(ps, param, ps.get_value(param), comp='prim')   
                    for param in ps2:

                        if param == 'ld_coeffs':
                            set_ld_legacy('lcdep', ps2.get_value(param), on=on, comp='sec')
                        else:
                             try:
                                unit = ps2.get_unit(param)
                                set_param_legacy(ps2, param, ps2.get_value(param), un=unit, comp='sec')
                             except:            
                                set_param_legacy(ps2, param, ps2.get_value(param), comp='sec')
        elif parsets[x] == 'compute':
    
            for param in compute:
                try:
                   unit = ps.get_unit(param)
                   set_param_legacy(compute, param, compute.get_value(param), un=unit)
                except:
                   set_param_legacy(compute, param, compute.get_value(param))
             
    # Now compute the loaded system
    phb1.save('backtest.phoebe')
    for i in range(len(lcnames)):
        print "lcname", lcnames[i]
        # Is there an observation to compute?
        
        psd = system.params['obs']['lcobs'][lcnames[i]]
        
        if psd['time'].size:
            qual = 'time'
            indep = psd['time']
            print "it is time"
        elif psd['phase'].size:
            qual = 'phase'
            indep = psd['phase']
            print "phase you later"

        else:

            raise ValueError('{} has no independent variable from which to compute a light curve'.format(lcnames[i]))
        
        print lcnames[i]
#        print indep
#        print type(indep)
#        ph = np.linspace(-0.5, 0.5, 201)
#        indep= ph 
        flux = phb1.lc(tuple(indep.tolist()), i)
#        this is where it should really go, but until get_synthetic is changed we will put it in the primary lcsyn
#        psd = system.params['syn']['lcsyn'][lcnames[i]]
        psd = system[0].params['syn']['lcsyn'][lcnames[i]]
        psd['flux'] = flux
        psd[qual] = indep
        psd = system[1].params['syn']['lcsyn'][lcnames[i]]
        psd[qual] = indep
        psd['flux'] = np.zeros(len(flux))
        print psd
#        print phb1.getpar('phoebe_period')
#        quit()
#        print psd 
#        print "i'm trying to get something"
#        system.get_synthetic(category='lc')
#        psd = system.params['syn']['lcsyn'][lcnames[i]] 
#        print "DID I get it?"
#        print psd
#    print "seriously wth?"

    for i in range(len(rvnames)):
        # Is there an observation to compute?
        
        psd = system[i].params['obs']['rvobs'][rvnames[i]]
        
        if psd['time'].size:
            qual = 'time'
            indep = psd['time']
        elif psd['phase'].size:
            qual = 'phase'
            indep = psd['phase']

        else:
            raise ValueError('{} has no independent variable from which to compute a light curve'.format(lcnames[i]))
        
#        print i
#        print indep
#        print type(indep)
#        ph = np.linspace(-0.5, 0.5, 201)
#        indep= ph
        if  i == 0:
            rv = phb1.rv1(tuple(indep.tolist()), i)
        if  i == 1:
            rv = phb1.rv2(tuple(indep.tolist()), i)
        psd = system[i].params['syn']['rvsyn'][rvnames[i]]
        psd['rv'] = rv 
        psd[qual] = indep
#        print psd
    
#    print system.params['syn']['lcsyn']['Undefined']
#    print system.params['syn']['lcsyn']['lc02']
    #set parameters   
    # check for mesh:wd PS
    # if mesh:wd:
    #     use
    # elif mesh:marching:
    #     convert using options in **kwargs
    
    # check for any non LC/RV and disable with warning
    
    # create phoebe legacy system
  
#    for i,obj in enumerate(system):
#        print obj
#        ps = obj.params['component']
#        for param in ps:
#            set_param_legacy(phb1, param, ps.get_value(param))
            
#    ps = obj.params['orbit']
#    for params in ps:
#        set_param_legacy(phb1, param, ps.get_value(param))
#    eb = phoebe.Bundle(system) 
    phb1.save('backtest.phoebe')
    print phb1.getpar('phoebe_period')
    return system    
    
    
def compute_pd(system, *args, **kwargs):
    """
    use Josh Carter's photodynamical code (photodynam) to compute velocities (dynamical only), 
    orbital positions and velocities (center of mass only), and light curves (assumes spherical stars).  The
    code is available here:
    
    https://github.com/dfm/photodynam

    photodynam must be installed and available on the system in order to use this plugin.
    
    Please cite both

        Science 4 February 2011: Vol. 331 no. 6017 pp. 562-565 DOI:10.1126/science.1201274
        MNRAS (2012) 420 (2): 1630-1635. doi: 10.1111/j.1365-2966.2011.20151.x

    when using this code.
    
    Parameters that are used by this backend:
        Compute PS:
        - stepsize
        - orbiterror
        
        Orbit PS:
        - sma
        - ecc
        - incl
        - per0
        - long_an
        - t0 (at periastron passage, will convert if provided at superior conjunction)
        
        Component PS:
        - mass (from q and P if BinaryRocheStars)
        - radius (equivalent radius from volume computed by pot if BinaryRocheStars)
        
        lcdep:
        - pblum
        - ld_coeffs (if ld_func=='linear')
        
    Values that are filled by this backend:
        lcsyn:
        - time
        - flux (only per-system, don't retrieve fluxes for individual components)
        
        rvsyn (dynamical only):
        - time
        - rv
    """
    
    def write_fi(bodies, orbits, time0, step_size, orbit_error, pblums, u1s, u2s):
        """
        write input file for pd
        """
        fi = open('_tmp_pd_inp', 'w')  
        fi.write('{} {}\n'.format(len(bodies), time0))
        fi.write('{} {}\n'.format(step_size, orbit_error))
        fi.write('\n')
        fi.write(' '.join([str(b['m']) for b in bodies])+'\n')
        fi.write(' '.join([str(b['r']) for b in bodies])+'\n')
        
        
        if -1 in pblums:
            logger.error('pblums must be set in order to run photodynam')
            return system
        
        fi.write(' '.join([str(pbl / (4*np.pi)) for pbl in pblums])+'\n')
    
        fi.write(' '.join(u1s)+'\n')
        fi.write(' '.join(u2s)+'\n')
        fi.write('\n')

        for o in orbits:
            
            if o['ps'].get_value('t0type') == 'superior conjunction':
                t0 = keplerorbit.from_supconj_to_perpass(o['ps'].get_value('t0', 'JD'), o['ps'].get_value('period', 'd'), o['ps'].get_value('per0', 'rad'))
            else:
                t0 = o['ps'].get_value('t0', 'JD')
            
            om = 2 * np.pi * (time0 - t0) / o['ps'].get_value('period', 'd')
            fi.write('{} {} {} {} {} {}\n'.format(o['a'], o['e'], o['i'], o['o'], o['l'], om))
        fi.close()
        
        return
        
    def write_fr(cols, obs):
        """
        write report file for pd
        """
        fr = open('_tmp_pd_rep', 'w')
        fr.write('{}\n'.format(cols))
        for t in obs.get_value('time', 'JD'):
            fr.write('{}\n'.format(t))
        fr.close()

        return
        
    def run_pd():
        """
        run pd and return the columns
        """
        out = commands.getoutput('photodynam _tmp_pd_inp _tmp_pd_rep > _tmp_pd_out')
        return np.loadtxt('_tmp_pd_out', unpack=True)
    
    params = kwargs['params']
    step_size = params.get_value('stepsize')
    orbit_error = params.get_value('orbiterror')

    # TODO: should be able to compute RVs (throw warning if not dynamical)

    out = commands.getoutput('photodynam')
    if 'not found' in out:
        logger.error('photodynam not found: please install first')
        return system
    
    refs = system.get_refs(per_category=True)
    lcnames = refs.get('lc', [])
    rvnames = refs.get('rv', [])
    orbnames = refs.get('orb', [])
    
    # we need to walk through the system and do several things:
    # for each component, determine:
    # - flux (for each lcname, from pblum/4pi since this will assume spherical)
    # - mass
    # - radius
    # - u1, u2 (for each lcname, linear limb-darkening coeffs)
    # for each orbit, determine:
    # - sma (a)
    # - ecc (e)
    # - incl (i)
    # - per0 (o) 
    # - nodal longitude (l)
    # - mean anomaly at the starting time for a given lcname (m)

    bodies = []
    orbits = []
    
    for item in reversed(list(system.walk_bodies())):
        if hasattr(item, 'bodies'):
            d = {'item': item}

            ps = item.get_children()[0].params['orbit']
            
            d['ps'] = ps
            
            d['a'] = ps.get_value('sma', 'AU')
            d['e'] = ps.get_value('ecc')
            d['i'] = ps.get_value('incl', 'rad')
            d['o'] = ps.get_value('per0', 'rad')
            d['l'] = ps.get_value('long_an', 'rad')
            # mean anomaly (m) is per lcdep (in for loop below)
            
            orbits.append(d)
            
        else:
            d = {'item': item}
            
            
            d['m'] = item.get_mass() / 3378.44105  # G is 1
            d['r'] = convert('Rsol', 'AU', item.get_radius(method='equivalent'))
            # flux is per lcdep (in for loop below)
            # u1, u2 are per lcdep (in for loop below)
            
            bodies.append(d)
            
    # loop through all the rv-datasets and make a call to photodynam for each
    for rvname in rvnames:
        # we don't really know which bodies we need yet... looping through all
        # may be a little redundant and add some time overhead, but oh well
        for i,b in enumerate(bodies):
            rvobs = b['item'].get_obs(ref=rvname)
            if rvobs:                
                #~ print "***", b['item'].get_label(), rvname, 3*(i+1)
                rvsyn = b['item'].get_synthetic(ref=rvname)
                
                time0 = rvobs.get_value('time', 'JD')[0] # TODO: should this necessarily be in JD?
                
                pblums = [1 for bi in bodies]
                u1s = ['0' for bi in bodies]
                u2s = ['0' for bi in bodies]
            
                write_fi(bodies, orbits, time0, step_size, orbit_error, pblums, u1s, u2s)
                
                write_fr('t v', rvobs)

                out = run_pd()
                # out is a list of arrays in the format:
                # time, vx_0, vx_0, vy_0, vx_1, ...
                # we want time and -vz_i
                time = out[0]
                rv = convert('AU/d', 'km/s', -1*out[3+(i*3)])   # TODO: is this always km/s or can the user request something different?

                rvsyn.set_value('time', time)
                rvsyn.set_value('rv', rv)

    for orbname in orbnames:
        for i,b in enumerate(bodies):
            orbobs = b['item'].get_obs(ref=orbname)
            if orbobs:
                orbsyn = b['item'].get_synthetic(ref=orbname)
                
                time0 = orbobs.get_value('time', 'JD')[0] # TODO: should this necessarily be in JD?
                
                pblums = [1 for bi in bodies]
                u1s = ['0' for bi in bodies]
                u2s = ['0' for bi in bodies]

                write_fi(bodies, orbits, time0, step_size, orbit_error, pblums, u1s, u2s)
                
                write_fr('t x v', orbobs)
                
                out = run_pd()
                # out is a list of arrays in the format:
                # time, x_0, y_0, z_0, x_1, ..., vx_0, vy_0, vz_0, vx_1, ...
                time = out[0]
                
                nbodies = len(bodies)
                ##    t x0 y0 z0 x1 y1 z1 ... vx0 vy0 vz0 vx1 vy1 vz1
                ##
                ##i=0 0  1  2  3               7   8   9
                ##i=1 0           4  5  6                  10  11  12
                
                x = convert('AU', 'Rsol', out[1+(i*3)])
                y = convert('AU', 'Rsol', out[2+(i*3)])
                z = convert('AU', 'Rsol', out[3+(i*3)])
                vx = convert('AU/d', 'Rsol/d', out[3*nbodies+1+(i*3)])
                vy = convert('AU/d', 'Rsol/d', out[3*nbodies+2+(i*3)])
                vz = convert('AU/d', 'Rsol/d', out[3*nbodies+3+(i*3)])
                
                orbsyn.set_value('time', time)
                orbsyn.set_value('x', x)
                orbsyn.set_value('y', y)
                orbsyn.set_value('z', z)
                orbsyn.set_value('vx', vx)
                orbsyn.set_value('vy', vy)
                orbsyn.set_value('vz', vz)
                

    # loop through all the lc-datasets and make a call to photodynam for each
    for lcname in lcnames:
        lcobs = system.params['obs']['lcobs'][lcname]
        lcsyn = system.params['syn']['lcsyn'][lcname]
        time0 = lcobs.get_value('time', 'JD')[0]   # TODO: should this necessarily be in JD?        
        pblums = [b['item'].get_parset(lcname)[0].get_value('pblum') for b in bodies]
        
        u1s = []
        u2s = []
        for b in bodies:
            lcdep = b['item'].get_parset(lcname)[0]
            if lcdep.get_value('ld_func') == 'linear':
                ld_coeffs = lcdep.get_value('ld_coeffs')
            else:
                ld_coeffs = (0,0)
                logger.warning('ld_func for {} {} must be linear, but is not: defaulting to linear with coeffs of {}'.format(lcname, b['item'].get_label(), ld_coeffs))
                
            u1s.append(str(ld_coeffs[0]))
            u2s.append(str(ld_coeffs[1]))
        
        write_fi(bodies, orbits, time0, step_size, orbit_error, pblums, u1s, u2s)

        write_fr('t F', lcobs)
        
        time, flux = run_pd()

        # fill synthetics - since PHOEBE computes the fluxes by adding from all components,
        # we'll set the flux to the first body and the rest to 0
        for i,b in enumerate(bodies):
            lcsyn = b['item'].params['syn']['lcsyn'][lcname]
            
            lcsyn.set_value('time', time)
            if i==0:
                lcsyn.set_value('flux', flux - 0.92)  # TODO: fix this (shouldn't need to subtract 0.92 to match phoebe2)
            else:
                lcsyn.set_value('flux', np.zeros(len(flux)))      
                
        # TODO: account for l3, distance

    # cleanup (delete tmp files)
    #~ for fname in ['_tmp_pd_inp', '_tmp_pd_rep', '_tmp_pd_out']:
        #~ os.remove(fname)
    
    return system
