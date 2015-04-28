
import logging
import numpy as np
from phoebe.backend import universe
from phoebe.algorithms import marching
from phoebe.units.conversions import convert
try:
    import phoebeBackend as phb1
except ImportError:
    pass

logger = logging.getLogger("FRONTEND.BACKENDS")
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
    comps = {'prim':1, 'sec':2}
    letter = {0:'x', 1:'y'}
    if on != None:
        if type(value) != list:
            value = [value]
        for y in range(len(value)):
            try:
                phb1.setpar(lds[name].replace('#', letter[y]).replace('%', comps[comp])+'['+str(on)+']', value[y]) 
            except:
                logger.warning('ld coeffs must be numbers')
    else:
        for y in len(value):
            try:
                phb1.setpar(lds[name].replace('#', letter[y]).replace('%', comps[comp]), value[y]) 
            except:
                logger.warning('ld coeffs must be numbers')
        


def set_param_legacy(ps, param, value, on = None, ty=None, comp=None, un=None):
    params = {'morphology':{'name':'phoebe_model'}, 'asini':{'name':'phoebe_asini_value'},'filename':{'name':'phoebe_#_filename'}, 'statweight':{'name':'phoebe_#_sigma'},'passband':{'name':'phoebe_#_filter'},'columns':{},'Rv':{'name':'phoebe_ie_factor'},'extinction':{'name':'phoebe_ie_excess'},'method':{'name':'phoebe_proximity_rv#_switch'},'incl':{'name':'phoebe_incl','unit':'deg'}, 't0':{'name':'phoebe_hjd0'}, 'period':{'name':'phoebe_period', 'unit':'d'}, 'dpdt':{'name':'phoebe_dpdt', 'unit':'d/d'}, 'phshift':{'name':'phoebe_pshift'}, 'sma':{'name': 'phoebe_sma', 'unit':'Rsun'}, 'q':{'name':'phoebe_rm'}, 'vgamma':{'name':'phoebe_vga', 'unit':'km/s'},  'teff':{'name':'phoebe_teff#', 'unit':'K'}, 'pot':{'name':'phoebe_pot#'}, 'abun':{'name':'phoebe_met#'},  'syncpar':{'name':'phoebe_f#'}, 'alb':{'name':'phoebe_alb#'}, 'gravb':{'name':'phoebe_grb#'}, 'ecc':{'name':'phoebe_ecc'}, 'per0':{'name':'phoebe_perr0','unit':'rad'}, 'dperdt':{'name':'phoebe_dperdt','unit':'rad/d'}, 'ld_func':{'name':'phoebe_ld_model'}, 'pblum':{}, 'atm':{'name':'phoebe_atm#_switch'},'refl':{'name':'phoebe_reffect_switch'},'gridsize':{'name':'phoebe_grid_finesize#'},'delta':{'name':'phoebe_grid_finesize#'},'scale':{'name':'phoebe_compute_hla_switch'}, 'offset':{'name':'phoebe_compute_vga_switch'}}

    boolean = {True:1, False:0}
    comps = {'prim':'1', 'sec':'2'}

    if param not in params.keys():
        logger.warning('{} parameter ignored in phoebe legacy'.format(param))


    elif param == 'filename':
        phb1.setpar(params[param]['name'].replace('#', str(ty)), value)
    elif param == 'scale':
        valn = ps.get_adjust('scale')  
        phb1.setpar(params[param]['name'], boolean[valn])        
    elif param == 'offset':
        valn = ps.get_adjust('offset')
        print valn
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
        if ty == 'rv':
            value[1]=value[1]+comps[comp]
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
            print comp
            print params[param]['name'].replace('#', comps[comp])
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

            phb1.setpar(params[param]['name'].replace('#', comps[comp]), boolean[value])

    elif param == 'delta':

        valn =  int(np.round(marching.delta_to_gridsize(value)))
        print valn
        print params[param]['name'].replace('#', comps[comp])
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
            print un, qualifier['unit'], value
            nval = convert(un, qualifier['unit'], value)
            print nval
            print params[param]['name']
            if comp is None:
                phb1.setpar(params[param]['name'], nval)
            else:
                phb1.setpar(params[param]['name'].replace('#', comps[comp]), nval)
   
        else:
#            print "I WENT HERE", params[param]['name']
#            print params[param]['name'].replace('#', comps[comp])
#            print params[param]['name'], comp
            if comp is None:
#                print "so I went here and still fucking failed"
                phb1.setpar(params[param]['name'], value)
            else:
                print params[param]['name'].replace('#', comps[comp])
                phb1.setpar(params[param]['name'].replace('#', comps[comp]), value)
    
def compute_legacy(system, *args, **kwargs):
    

  
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
    parsets = {'orbit':'system', 'position':'system', 'reddening':'system', 'component':'comp', 'mesh':'comp'}
    compsi = {1:'prim', 2:'sec'}
    #determine and initialize data curves

#    lcno = len(system1[0].params['pbdep']['lcdep'])
#    rvno = len(system1[0].params['pbdep']['rvdep'])+len(system1[1].params['pbdep']['rvdep'])
    
#    phb.setpar("phoebe_lcno", lcno)
#    phb.setpar("phoebe_rvno", rvno)
    refs = system.get_refs(per_category=True)
    lcnames = refs.get('lc', [])
    rvnames = refs.get('rv', [])
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
                print param
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
                    print 'tried'
                    unit = ps.get_unit(param)
                    print unit
                    set_param_legacy(ps, param, ps.get_value(param), un=unit)
                except:
                    print 'and failed'
                    set_param_legacy(ps, param, ps.get_value(param))

        else:
                print 2, x
                ps = system[0].params[x]
                ps2 = system[1].params[x] 
                if ps.get_context() == ps2.get_context():   
                    for param in ps:
                         print param
                         if param == 'ld_coeffs':
                            set_ld_legacy('lcdep', ps.get_value(param), on=on, comp='prim')
                            set_ld_legacy('lcdep', ps2.get_value(param), on=on, comp='sec')
                         else:
                            try:
                                print 'tried'
                                unit = ps.get_unit(param)
                                set_param_legacy(ps, param, ps.get_value(param), un=unit, comp='prim')
                                set_param_legacy(ps2, param, ps2.get_value(param), un=unit, comp='sec')
                            except:
                                print 'and failed'
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
    
    # Now compute the loaded system
    phb1.save('backtest.phoebe')
    for i in range(len(lcnames)):
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
            print "it is time"
        elif psd['phase'].size:
            qual = 'phase'
            indep = psd['phase']
            print "phase you later"

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
