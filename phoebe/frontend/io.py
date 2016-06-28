import numpy as np
import phoebe as phb

import logging
logger = logging.getLogger("IO")
logger.addHandler(logging.NullHandler())

"""
Dictionaries of parameters for conversion between phoebe1 and phoebe 2

"""

_1to2par = { 'ld_model':'ld_func', 'bol':'ld_coeffs_bol','rv': 'ld_coeffs', 'lc':'ld_coeffs','active': 'enabled', 'model': 'morphology', 'filename': 'filename', 'sigma': 'statweight', 'filter': 'passband', 'excess': 'extinction', 'hjd0': 't0_supconj', 'period':'period', 'dpdt': 'dpdt', 'pshift':'phshift', 'sma':'sma', 'rm': 'q', 'incl': 'incl', 'pot':'pot', 'met':'abun', 'f': 'syncpar', 'alb': 'alb_bol', 'grb':'gravb_bol', 'ecc': 'ecc', 'perr0':'per0', 'dperdt': 'dperdt', 'hla': 'pblum', 'cla': 'pblum', 'el3': 'l3', 'reffect': 'mult_refl', 'reflections':'refl_num', 'finesize': 'gridsize', 'vga': 'vgamma', 'teff':'teff', 'msc1':'msc1', 'msc2':'msc2', 'ie':'ie', 'proximity_rv':'proximity_rv','atm': 'atm'}

_2to1par = {v:k for k,v in _1to2par.items()}

_units1 = {'incl': 'deg', 'period': 'd', 'dpdt': 'd/d', 'sma': 'Rsun', 'vga':'km/s', 'teff': 'K', 'perr0': 'rad', 'dperdt': 'rad/d'}

_parsect = {'t0':'component',  'period':'component', 'dpdt':'component', 'pshift':'component', 'sma':'component', 'rm': 'component', 'incl':'component', 'perr0':'component', 'dperdt':'component', 'hla': 'component', 'cla':'component', 'el3':'component', 'reffect':'compute', 'reflections':'compute', 'finegrid':'mesh', 'vga':'system', 'msc1_switch': 'compute', 'msc2_switch': 'compute', 'ie_switch':'compute', 'proximity_rv1':'compute', 'proximity_rv2': 'compute'}


#_bool1to2 = {1:True, 0:False}

_bool2to1 = {True:1, False:0}

"""
ld_legacy -

"""

def ld_to_phoebe(pn, d, rvdep=None, dataid=None):

    if 'bol' in pn:
        d['context'] = 'component'
        pnew = 'bol'
    elif 'rv' in pn:
        d['context'] = 'dataset'
        d['dataset'] = rvdep
        pnew = 'lc'
    elif 'lc' in pn:
        d['context'] = 'dataset'
        d['dataset'] = dataid
        pnew = 'rv'
    else:
        pnew = 'ld_model'
    if 'x' in pn:
        d['index'] = 0
    elif 'y' in pn:
        d['index'] = 1

    return [pnew, d]
"""
ret_dict - finds the phoebe 2 twig associated with a given parameter

pname:  name of the parameter

dataid: name of the dataset

rvdep = determines whether rv belongs to primary or secondary

"""


def ret_dict(pname, val, dataid=None, rvdep=None, comid=None):
#    pname = pname.split('[')[0]
    pieces = pname.split('_')
    pnew = pieces[-1]
    d = {}
    if pnew == 'switch':
       pnew = pieces[1]
       if pnew  == 'proximity':
           pnew = pnew+'_'+pieces[2]
           d['context'] = 'compute'
# component specific parameters end with 1 or 2 in phoebe1
    if pnew[-1] == '1':
        d['component'] = 'primary'
#        d.setdefault('context', 'component')
        d['context'] = 'component'
        pnew = pnew[:-1]

    elif pnew[-1] == '2':
        d['component'] = 'secondary'
#        d.setdefault('context', 'component')
        d['context'] = 'component'
        pnew = pnew[:-1]

# even though gridsize belongs to each component it is located in the compute parameter set


    if pieces[1] == 'lc':
        d['dataset'] = dataid
        d['context'] = 'dataset'

    elif pieces[1] == 'rv':
        d['component'] = rvdep
        d['dataset'] = dataid
        d['context'] = 'dataset'

    elif pieces[1] == 'ld':

        pnew, d  = ld_to_phoebe(pnew, d, rvdep, dataid)

    if pnew == 'hla':
        d['component'] = 'primary'
    elif pnew == 'cla':
        d['component'] = 'secondary'


    if pnew in _1to2par:
        try:
            d.setdefault('context', _parsect[pnew])
        except:
            try:
                d['context']
            except:
                d['context'] = None

        if 'finesize' in pnew:
            d['context'] = 'compute'

        if 'proximity' in pnew:
            d['context'] = 'compute'

        if d['context'] == 'component':

            d.setdefault('component', 'binary')

        if d['context'] == 'compute':
            d.setdefault('compute', comid)

        d['value'] = val
        d['qualifier'] = _1to2par[pnew]

        if pnew in _units1:
            d['unit'] = _units1[pnew]
    else:
        d ={}
        logger.info("Parameter "+str(pname)+" has no Phoebe 2 counterpart")

#    d['qualifier'] = pnew
#    else:

    # check if it belongs in the component parameter set
#        if len(comp) > 0:
#            twig = comp

#        else:
#            twig = ''
    return pnew, d

#def par_swap(pname, val, dataid=None, rvdep=None, comid=None):
#    pieces = pname.split('_')
#    pnew = pieces[-1]
#    if pnew == 'switch':
#       pnew = pieces[1]

# build your original dictionary
#    pnew, d = ret_dict(pname, dataid, rvdep, comid)

# now for exceptions

#    if pnew == 'filter':

       # write method for this

#    elif pnew == 'excess':

#    elif pnew == 'alb':
#        val = 1-val

#    elif pnew == 'atm':

#    elif pnew == 'finegrid':

#    else:
#        d['qualifier'] = _1to2par[pnew]
#        if pnew in _units1:
#            d['unit'] = _units1[pnew]
#    try:
#        phb.set_value(**d, check_relevant=False)

#    except:

#        logger.warning('Cannot find relevant phoebe 2 parameter for '+str(pname))

"""
Load in datasets from files listed in .phoebe file

"""

"""
Load in datasets from files listed in .phoebe file

"""

def load_dataset(eb, filename, dataid, indep, dep, indweight, typ, mzero=None):#, comp='primary'):

    #unpack data if it exists
    rvs = eb.get_dataset(method='RV').datasets
    #it can't be an empty array or checking against it will fail
    if len(rvs) == 0:
        rvs.append(' ')
    if filename != 'Undefined':

        if indweight == 'Undefined':
            times, var = np.loadtxt(filename, unpack = True)
            vals = [times, var]
        else:
            times, var, sigmas = np.loadtxt(filename, unpack = True)
            vals = [times, var, sigmas]
        if dep == 'Magnitude':
            m = vals[1]
            flux = 10**(-0.4*(m-mzero))
            vals[1] = flux

            #vals = vals[0:1]
    #determine if we need 1 rv dataset or 2

#        rvs = eb.get_dataset(method='RV').datasets
        if typ == 'rv':
            if dep == 'primary':
                odep = 'secondary'
            else:
                odep='primary'
            try:
                timeso = eb.get_value(section = 'dataset', dataset=rvs[0], qualifier='time', component=odep)
                if timeso.all() == times.all():
                    dataid = rvs[0]
            except:
                dataid = dataid
## add dataset

    if dataid == 'Undefined':
        dataid = None

    if dataid != rvs[0]:
        try:
            eb._check_label(dataid)
            if not dataid:
                eb.add_dataset(typ)
            else:
                eb.add_dataset(typ, dataset=dataid)
        except:
            logger.warning("The name picked for the "+typ+" dataset is forbidden. Applying default name instead")
#           alcn = "%02d" % (x,)
#            dataid = 'lc'+str(alcn)
            eb.add_dataset(typ)


    if filename != 'Undefined':
        d = {}
        d['section'] = 'dataset'
        d['method'] = typ
        d['dataset'] = dataid

        if typ == 'lc':

            params = ['time', 'flux', 'sigma']

        if typ == 'rv':
            d['component'] = dep
            params = ['time', 'rv', 'sigma']

        for x in range(len(vals)):

            d['qualifier'] = params[x]
            d['value'] = vals[x]
            eb.set_value(check_relevant=False, **d)

    return eb

"""
Load a phoebe legacy file complete with all the bells and whistles

filename - a .phoebe file (from phoebe 1)

"""

def load_legacy(filename, add_compute_legacy=True, add_compute_phoebe=True):

# load an empty legacy bundle and initialize obvious parameter sets
    eb = phb.Bundle.default_binary()
    eb.disable_history()
    comid = []
    if add_compute_legacy == True:
        comid.append('detailed')
        eb.add_compute('phoebe', compute=comid[0])
    if add_compute_phoebe == True:
        comid.append('backend')
        eb.add_compute('legacy', compute=comid[-1])
#    eb = phb2.Bundle()

# phoebe 1 is all about binaries so initialize both components

#    eb.add_component('star', component='primary')
#    eb.add_component('star', component='secondary')
#    eb.add_orbit('binary')
#    eb.set_hierarchy(phb2.hierarchy.binaryorbit, eb['binary'], eb['primary'], eb['secondary'])

# load the phoebe file

    params = np.loadtxt(filename, dtype='str', delimiter = ' = ')

#basic filter on parameters that make no sense in phoebe 2
    ind = [list(params[:,0]).index(s) for s in params[:,0] if not ".ADJ" in s and not ".MIN" in s and not ".MAX" in s and not ".STEP" in s and not "gui_" in s]
    params = params[ind]

# determine number of lcs and rvs
    rvno = np.int(params[:,1][list(params[:,0]).index('phoebe_rvno')])
    lcno = np.int(params[:,1][list(params[:,0]).index('phoebe_lcno')])
# delete parameters that have already been accounted for and find lc and rv parameters
    mzero = np.float(params[:,1][list(params[:,0]).index('phoebe_mnorm')])
    params = np.delete(params, [list(params[:,0]).index('phoebe_lcno'), list(params[:,0]).index('phoebe_rvno')], axis=0)
# but first FORCE hla and cla to follow conventions so the parser doesn't freak out.
    for x in range(1,lcno+1):
        hlain = list(params[:,0]).index('phoebe_hla['+str(x)+'].VAL')
        clain = list(params[:,0]).index('phoebe_cla['+str(x)+'].VAL')

        params[:,0][hlain] = 'phoebe_lc_hla1['+str(x)+'].VAL'
        params[:,0][clain] = 'phoebe_lc_cla2['+str(x)+'].VAL'

    lcin = [list(params[:,0]).index(s) for s in params[:,0] if "lc" in s]
    rvin = [list(params[:,0]).index(s) for s in params[:,0] if "rv" in s and not "proximity" in s]

    lcpars = params[lcin]
    rvpars = params[rvin]
    lcin.extend(rvin)
    params = np.delete(params, lcin, axis=0)

# create datasets and fill with the correct parameters

# but first FORCE hla and cla to follow conventions so the parser doesn't freak out.



# First LCs
    for x in range(1,lcno+1):
        lcs = eb.get_dataset(method='LC').datasets
    #list of parameters related to current dataset

        lcint = [list(lcpars[:,0]).index(s) for s in lcpars[:,0] if "["+str(x)+"]" in s]
        lcpt = lcpars[lcint]
    # get name of dataset and add
        datain = list(lcpt[:,0]).index('phoebe_lc_id['+str(x)+']')
        dataid = lcpt[:,1][datain].strip('"')
        lcpt = np.delete(lcpt, datain, axis=0)
        if dataid == 'Undefined':
            dataid = None
        try:
            eb._check_label(dataid)
            if not dataid:
                eb.add_dataset('lc')
            else:
                eb.add_dataset('lc', dataset=dataid)
        except ValueError:
            # TODO: add custom exception for forbiddenlabelerror?
            logger.warning("The name picked for the lightcurve is forbidden. Applying default name instead")
#            alcn = "%02d" % (x,)
#            dataid = 'lc'+str(alcn)
            eb.add_dataset('lc')
        lcsnew = eb.get_dataset(method='LC').datasets
        dataid = [i for i in lcsnew if not i in lcs]
        dataid = dataid[0]

    # now go through parameters and input the results into phoebe2
        for y in range(len(lcpt)):
            for j in comid:

                pname = lcpt[:,0][y]
                val = lcpt[:,1][y].strip('"')
                pname = pname.split('[')[0]
                pnew, d = ret_dict(pname, val, dataid=dataid, comid=j)
                if len(d) > 0:
                    #TODO: change filter in several places when this is implemented
                        if pnew not in ['filter']:
                            eb.set_value_all(check_relevant=False, **d)
                        else:
                            logger.warning("this parameter should be, but is not currently supported") #to do
#            pnew, twig = ret_twig(pname, dataid)
            # PUT parameter change here

#Now RVs
    for x in range(1,rvno+1):
        rvs = eb.get_dataset(method='RV').datasets
    #list of parameters related to current dataset
        rvint = [list(rvpars[:,0]).index(s) for s in rvpars[:,0] if "["+str(x)+"]" in s]
        rvpt = rvpars[rvint]
    # get name of dataset and add
        datapars = ['filename','id', 'indep', 'dep', 'indweight']
        datavals = []
        for y in datapars:
            datain = list(rvpt[:,0]).index('phoebe_rv_'+y+'['+str(x)+']')
            if y == 'dep':
                rvdep = rvpt[:,1][datain].split(' ')[0].lower().strip('"')
                dataid = rvdep
            else:
                dataid = rvpt[:,1][datain].strip('"')

            datavals.append(dataid)
        datavals.append('rv')
        eb = load_dataset(eb, *datavals)

        #in place to take make sure the dataset name is correct. Issues can occur if the dataset name is forbidden
        rvsnew = eb.get_dataset(method='RV').datasets
        dataid = [i for i in rvsnew if not i in rvs]
        if len(rvsnew) == 1:
            dataid = rvsnew[0]
        else:
            dataid = dataid[0]

    # now go through parameters and input the results into phoebe2

        for y in range(len(rvpt)):
            for j in comid:
                pname = rvpt[:,0][y]
                val = rvpt[:,1][y].strip('"')
                pname = pname.split('[')[0]
                pnew, d = ret_dict(pname, val, dataid = dataid, rvdep=rvdep, comid=j)
                if len(d) > 0:

                    if pnew not in ['filter']:
                        eb.set_value_all(check_relevant=False, **d) #Theoretically set value all is FINE, but I'm not sure if i like this. It comes about because certain parameters may exist in more than one place where it isn't easy to catch.

    for x in range(len(params)):
        for j in comid:
            pname = params[:,0][x]
            pname = pname.split('.')[0]
            val = params[:,1][x].strip('"')
            pnew, d = ret_dict(pname, val, comid=j)
            if pnew == 'ld_model':
                val = val.split(' ')[0]
                d['value'] = val[0].lower()+val[1::]
            if pnew == 'pot':
                #print "dict", d
                d['method'] = 'star'
                d.pop('qualifier') #remove qualifier from dictionary to avoid conflicts in the future
                d.pop('value') #remove qualifier from dictionary to avoid conflicts in the future
                eb.flip_constraint(solve_for='rpole', constraint_func='potential', **d) #this WILL CHANGE & CHANGE back at the very end
                #print "val", val
                d['value'] = val
                d['qualifier'] = 'pot'
                d['method'] = None
                d['context'] = 'component'
                #print "d end", d
    #        elif pnew == 'filter':

             # write method for this

    #        elif pnew == 'excess':
                     # requires two parameters that phoebe 1 doesn't have access to Rv and extinction
            elif pnew == 'alb':
                val = 1.-float(val)
                d['value'] = val
            elif pnew == 'atm':
                val = int(val)

                if val == 0:
                    d['value'] = 'blackbody'
                if val == 1:
                    d['value'] = 'kurucz'
                logger.warning('If you would like to use phoebe 1 atmospheres, you must add this manually')
            elif pnew == 'finesize':
                    # set gridsize
                d['value'] = val
                eb.set_value(check_relevant=False, **d)
                    # change parameter and value to delta
                val = 10**(-0.98359345*np.log10(np.float(val))+0.4713824)
                d['qualifier'] = 'delta'
                d['value'] = val
            if len(d) > 0:

                if pnew not in ['filter']:

                    eb.set_value_all(check_relevant=False, **d)
    #print "before", eb['pot@secondary']
    #print "rpole before", eb['rpole@secondary']

    eb.flip_constraint(solve_for='pot', constraint_func='potential', component='primary')
    eb.flip_constraint(solve_for='pot', constraint_func='potential', component='secondary')
    #print eb['pot@secondary']
    #print "rpole after", eb['rpole@secondary']
    # turn on relevant switches like heating. If
    return eb

"""

Return phoebe1 parameter value in the right units (or without units)
eb -phoebe 2 bundle
d - dictionary with parameter, context, dataset etc.

"""


def par_value(param, index=None):

# build a dictionary

    d={}
    d['qualifier'] = param.qualifier
    d['component'] = param.component
    d['dataset'] = param.dataset
    d['compute'] = param.compute
    d['method'] = param.method
# Determine what type of parameter you have and find it's value
    if isinstance(param, phb.parameters.FloatParameter) and not isinstance(param, phb.parameters.FloatArrayParameter):
        ptype = 'float'
    # since it's a float it must have units. Therefore return value in correct units
        pnew = _2to1par[d['qualifier']]
        try:
            unit = _units1[pnew]
        except:
            unit = None

        val = param.get_quantity(unit=unit).value
        if d['qualifier'] == 'alb':
            val = [1.0-val]
        else:
            val = [val]
    elif isinstance(param, phb.parameters.ChoiceParameter):
        ptype = 'choice'
        val = [param.get_value()]
        if d['qualifier'] == 'atm':
        # in phoebe one this is a boolean parameter because you have the choice of either kurucz or blackbody

            ptype='boolean'
        if d['qualifier'] == 'ld_func':
            val = val[0]
            val = ['"'+val[0].upper() + val[1::]+' Law"']
    elif isinstance(param, phb.parameters.BoolParameter):

        ptype = 'boolean'
        val = [_bool2to1[param.get_value()]]

    elif isinstance(param, phb.parameters.IntParameter):
        ptype = 'int'
        val = [param.get_value()]
    elif isinstance(param, phb.parameters.FloatArrayParameter):
        val1 = param.get_value()[0]
        val2 = param.get_value()[1]
        val = [val1, val2]
        ptype='array'
    else:
        ptype = 'unknown'
        val = [param.get_value()]

    return [val, ptype]

"""

Return phoebe1 parameter name from phoebe 2 info


"""

def ret_ldparname(param, component=None, dtype=None, dnum=None, ptype=None, index=None):
    if 'bol' in param:
        if ptype=='array':
            pnew1 = 'xbol'
            pnew2 = 'ybol'
            pnew = [pnew1, pnew2]
        else:
            pnew = ['model']
    else:
        if ptype == 'array':
            pnew1 = str(dtype)+'x'
            pnew2 = str(dtype)+'y'
            pnew = [pnew1, pnew2]

    if component == 'primary':
        pnew = [x + '1' for x in pnew]

    elif component == 'secondary':
        pnew = [x + '2' for x in pnew]

    if dnum != None:
        dset = '['+str(dnum)+'].VAL'
    else:
        dset = ''

    return ['phoebe_ld_'+x+dset for x in pnew]

def ret_parname(param, component=None, dtype=None, dnum=None, ptype=None, index=None):

# separate lds from everything because they suck
    if 'ld' in param:

        pname = ret_ldparname(param, component=component, dtype=dtype, dnum=dnum, ptype=ptype, index=index)
    else:
    # first determine name of parameters and whether it is associated with a com
        if component == 'primary':

            if param == 'pblum':
                pnew = 'hla'

            elif param in ['enabled','statweight','l3','passband']:
                pnew = _2to1par[param]

            else:
                pnew = _2to1par[param]+'1'

        elif component == 'secondary':

            if param == 'pblum':
                pnew = 'cla'

            elif param in ['enabled','statweight','l3','passband']:
                pnew = _2to1par[param]


            else:
                pnew = _2to1par[param]+'2'
        else:

            pnew = _2to1par[param]

    # one parameter doesn't follow the rules, so i'm correcting for it
            if pnew == 'reflections':
                pnew = 'reffect_reflections'
    #determine the context of the parameter and the number of the dataset

        if dtype != None:
            dtype = dtype+'_'

        else:
            dtype = ''

        if dnum != None:

            dset = '['+str(dnum)+']'
        else:
            dset = ''
    # determine the determinant of the parameter based on parameter type

        if ptype == 'float':
            det = '.VAL'

        elif ptype == 'boolean' and dtype=='':
            det = '_switch'
        else:
            det = ''

        pname = ['phoebe_'+str(dtype)+str(pnew)+dset+det]

    return pname

"""

Create a .phoebe file from phoebe 1 from a phoebe 2 bundle.

"""

def pass_to_legacy(eb, filename='2to1.phoebe'):


    # check to make sure you have exactly two stars and exactly one orbit
    stars = eb['hierarchy'].get_stars()
    orbits = eb['hierarchy'].get_orbits()

    if len(stars) != 2 or len(orbits) != 1:
        raise ValueError("Phoebe 1 only supports binaries. Either provide a different system or edit the hierarchy.")
#  catch all the datasets
# Find if there is more than one limb darkening law
    ldlaws = set([p.get_value() for p in eb.filter(qualifier='ld_func').to_list()])
    if len(set(ldlaws)) > 1:
        raise ValueError("Phoebe 1 takes only one limb darkening law.")

    lcs = eb.get_dataset(method='LC').datasets
    rvs = eb.get_dataset(method='RV').datasets
    if len(ldlaws) == 0:
        pass
    elif list(ldlaws)[0] not in ['linear', 'logarithmic', 'square root']:
        raise ValueError(list(ldlaws)[0]+" is not an acceptable value for phoebe 1. Accepted options are 'linear', 'logarithmic' or 'square root'")

    #make lists to put results with important things already added

    parnames = ['phoebe_rvno', 'phoebe_lcno']
    parvals = [len(rvs), len(lcs)]
    types = ['int', 'int']
    #Force the independent variable to be time

    parnames.append('phoebe_indep')
    parvals.append('"Time (HJD)"')
    types.append('choice')
    primary, secondary = stars

    prpars = eb.filter(component=primary, context='component')
    secpars = eb.filter(component=secondary, context='component')

    # get primary parameters and convert

    for param in prpars.to_list():


#        if isinstance(eb.get_parameter(prpars[x], component='primary'), phoebe.parameters.FloatParameter):

#        param = eb.get_parameter(prpars[x], component='primary')
        try:
            pnew = _2to1par[param.qualifier]
            if param.qualifier in ['sma', 'period', 'incl','enabled','statweight','l3', 'ld_func']:
                param = None
            elif 'ld_' in param.qualifier:
                param = None

        except:
            logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')
            param=None
        if param != None:

            val, ptype = par_value(param)
            if param.qualifier == 'alb_bol':
                val = [1-float(val[0])]
            pname = ret_parname(param.qualifier, component = param.component, ptype=ptype)
            if pname[0] not in parnames:
                parnames.extend(pname)
                parvals.extend(val)
                if ptype == 'array':
                    types.append(ptype)
                    types.append(ptype)
                else:
                    types.append(ptype)

    for param in secpars.to_list():
        # make sure this parameter exists in phoebe 1
#        param = eb.get_parameter(secpars[x], component= 'secondary')
        try:
            pnew = _2to1par[param.qualifier]
            if param.qualifier in ['sma', 'period', 'incl', 'ld_func', 'ld_func_bol']:
                param = None

        except:
            logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')
            param = None

# get rid of confusing parameters like sma and period which only exist for orbits in phoebe 1

        if param != None:

            val, ptype = par_value(param)
            if param.qualifier == 'alb_bol':
                val = [1-float(val[0])]
            pname = ret_parname(param.qualifier, component = param.component, ptype=ptype)
            if pname[0] not in parnames:
                parnames.extend(pname)
                parvals.extend(val)
                if ptype == 'array':
                    types.append(ptype)
                    types.append(ptype)
                else:
                    types.append(ptype)

#  catch all the datasets

    lcs = eb.get_dataset(method='LC').datasets
    rvs = eb.get_dataset(method='RV').datasets

# add all important parameters that must go at the top of the file


# loop through lcs

    for x in range(len(lcs)):
        quals = eb.filter(dataset=lcs[x], context='dataset')
        #phoebe 2 is ALWAYS times so pass time as the ind variable
        parnames.append('phoebe_lc_indep['+str(x+1)+']')
        parvals.append('Time (HJD)')
        types.append('choice')
        parnames.append('phoebe_lc_dep['+str(x+1)+']')
        parvals.append('Flux')
        types.append('choice')
        parnames.append('phoebe_lc_id['+str(x+1)+']')
        parvals.append(lcs[x])
        types.append('choice')
        for param in quals.to_list():
#            if len(eb.filter(qualifier=quals[y], dataset=lcs[x])) == 1:
#                if isinstance(eb.get_parameter(prpars[0], component='primary'), phoebe.parameters.Float                elif 'ld_' in param.qualifier:
#                    param = None
#Parameter):
#                param = eb.get_parameter(quals[y])
#                ptype = str(type(eb.get_parameter(prpars[x], component='primary'))).split("'")[1].split('.')[-1]

            try:
                pnew = _2to1par[param.qualifier]
                if param.qualifier in [ 'alb', 'l3', 'ld_func']:
                    param = None
            except:

                logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')
                param = None

            if param != None:

                val, ptype = par_value(param)
                if param.qualifier == 'pblum':
                    pname = ret_parname(param.qualifier, component= param.component, dnum = x+1, ptype=ptype)

                else:
                    pname = ret_parname(param.qualifier, component=param.component, dtype='lc', dnum = x+1, ptype=ptype)
                if pname[0] not in parnames:
                    parnames.extend(pname)
                    parvals.extend(val)
                    if ptype == 'array':
                        types.append(ptype)
                        types.append(ptype)
                    else:
                        types.append(ptype)
#            else:
#                param1 = eb.get_parameter(quals[y], component='primary')
#
#                try:
#                    pnew = _1to2par(param1.qualifier)
#                except:
#                    param1 = None
#                    logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')

#                if param1 != None:


#                    param2 = eb.get_parameter(quals[y], component='secondary')
#                    val, ptype = par_value(param1)
#                    val2, ptype2 = par_value(param2)
#                    pname1 = ret_parname(quals[y], component='primary', dtype='lc', dnum = x+1, ptype=ptype1)
#                    pname2 = ret_parname(quals[y], component='secondary', dtype='lc', dnum = x+1, ptype=ptype2)

#                    parnames.append(pname1)
#                    parnames.append(pname2)
#                    parvals.append(val1)
#                    parvals.append(val2)


#loop through rvs
#if there is more than one rv...try this

    for y in range(len(rvs)):
        quals = eb.filter(dataset=rvs[y], context='dataset')

        #if there is more than 1 rv try this
        try:
            comp = eb.get_parameter(qualifier='time', dataset=rvs[y]).component
            parnames.append('phoebe_rv_indep['+str(y+1)+']')
            parvals.append('Time (HJD)')
            types.append('choice')
        # dependent variable is just Primary or secondary
            parnames.append('phoebe_rv_dep['+str(y+1)+']')
            parvals.append('"'+comp[0].upper()+comp[1::]+' RV"')
            types.append('choice')
            parnames.append('phoebe_rv_id['+str(y+1)+']')
            parvals.append(rvs[y])
            types.append('choice')
            for param in quals.to_list():

#            if len(eb.filter(qualifier=quals[y], dataset=rvs[x])) == 1:
                try:
                    pnew = _2to1par[param.qualifier]
                    if param.qualifier in ['ld_func']:
                        param = None

                except:
                    logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')
                    param = None

                if param != None:
                    val, ptype = par_value(param)
                    pname = ret_parname(param.qualifier, component = param.component, dtype='rv', dnum = y+1, ptype=ptype)
# if is tries to append a value that already exists...stop that from happening
                    if pname[0] not in parnames:
                        parnames.extend(pname)
                        parvals.extend(val)
                        if ptype == 'array':
                            types.append(ptype)
                            types.append(ptype)
                        else:
                            types.append(ptype)

# hacky but it works. If you have more than one component in an array
        except:
            comp = ['primary', 'secondary']
            parvals[0] = 2
            for i in range(len(comp)):
                parnames.append('phoebe_rv_indep['+str(i+1)+']')
                parvals.append('"Time (HJD)"')
                types.append('choice')
            # dependent variable is just Primary or secondary
                parnames.append('phoebe_rv_dep['+str(i+1)+']')
                parvals.append('"'+comp[i][0].upper()+comp[i][1::]+' RV"')
                types.append('choice')
                parnames.append('phoebe_rv_id['+str(y+1)+']')
                parvals.append(rvs[y])
                types.append('choice')

                for param in quals.to_list():

    #            if len(eb.filter(qualifier=quals[y], dataset=rvs[x])) == 1:
                    try:
                        pnew = _2to1par[param.qualifier]
                        if param.qualifier in ['ld_func']:
                            param = None

                    except:
                        logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')
                        param = None

                    if param != None:
                        val, ptype = par_value(param)
                        pname = ret_parname(param.qualifier, component = param.component, dtype='rv', dnum = i+1, ptype=ptype)
    # if is tries to append a value that already exists...stop that from happening
                        if pname[0] not in parnames:
                            parnames.extend(pname)
                            parvals.extend(val)
                            if ptype == 'array':
                                types.append(ptype)
                                types.append(ptype)
                            else:
                                types.append(ptype)


#            else:
#
#                try:
#                    pnew = _1to2par(param.qualifier)
#                except:
#                    param = None
#                    logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')

#                if param != None:

#                    val, ptype = par_value(param)
#                    pname = ret_parname(quals[y], component=comp, dtype='rv', dnum = x+1,ptype=ptype)

#                    parnames.append(pname)
#                    parvals.append(val)

#loop through the orbit

    oquals = eb.get_orbit(orbits[0])

    for param in oquals.to_list():
        try:
            pnew = _2to1par[param.qualifier]
            if param.qualifier in ['ld_func']:
                    param = None

        except:

            logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')
            param = None
        if param != None:
            val, ptype = par_value(param)
            pname = ret_parname(param.qualifier,ptype=ptype)
            parnames.extend(pname)
            parvals.extend(val)

            if ptype == 'array':
                types.append(ptype)
                types.append(ptype)
            else:
                types.append(ptype)

#loop through LEGACY compute parameter set

    comquals = eb.get_compute(method='legacy')-eb.get_compute(method='legacy', component='_default')

    for param in comquals.to_list():

        if param.qualifier == 'heating':
            if param.get_value() == False:
               in1 =  parnames.index('phoebe_alb1.VAL')
               in2 =  parnames.index('phoebe_alb2.VAL')
               parvals[in1] = 0.0
               parvals[in2] = 0.0

        try:
            pnew = _2to1par[param.qualifier]
            if param.qualifier in ['ld_func']:
                param = None
        except:

            logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')
            param = None

        if param != None:
            val, ptype = par_value(param)
            if param.qualifier == 'gridsize':
                pname = ret_parname(param.qualifier, component = param.component, dtype='grid', ptype=ptype)
            elif param.qualifier =='atm':
                atmval = {'kurucz':1, 'blackbody':0}
                pname = ret_parname(param.qualifier, component = param.component, ptype=ptype)

                val = str(atmval[val[0]])
            else:
                pname = ret_parname(param.qualifier, component = param.component, ptype=ptype)

            if pname[0] not in parnames:
                parnames.extend(pname)
                parvals.extend(val)
                types.append(ptype)
                if ptype == 'array':
                    types.append(ptype)
                    types.append(ptype)
                else:
                    types.append(ptype)

    sysquals = eb.filter('system')

    for param in sysquals.to_list():
        try:
            pnew = _2to1par[param.qualifier]
        except:
            logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')
            param = None

        if param != None:

            val, ptype = par_value(param)
            pname = ret_parname(param.qualifier, component = param.component, ptype=ptype)
            if pname[0] not in parnames:
                parnames.extend(pname)
                parvals.extend(val)
                types.append(ptype)

# Now loop through all the ld_coeffs because they are such a pain that they need their own context

#    ldquals = eb.filter(qualifier='ld_*')

#    for param in ldquals.to_list():

#        try:
#            pnew = _2to1par[param.qualifier]
#            if param.qualifier == 'ld_func':
#                param = None
#        except:

#            logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')
#            param = None

#        if param != None:

#            if isinstance(param, phb.parameters.FloatArrayParameter):
#                val1, ptype = par_value(param, index=0)
#                val2, ptype = par_value(param, index=1)
#                pname1 = ret_ldparname(param.qualifier,component = param.component,ptype=ptype,index=0)
#                pname2 = ret_ldparname(param.qualifier,component = param.component, ptype=ptype, index=1)
#                parnames.append(pname1)
#                parnames.append(pname2)
#                parvals.append(val1)
#                parvals.append(val2)
#            else:
#                val, ptype = par_value(param)
#                pname = ret_ldparname(param.qualifier,ptype=ptype)
#                parnames.append(pname)
#                parvals.append(val)
    # separate into primary, secondary, and component none
#        prpars = eb.filter(dataset=lcs[x], component='primary').to_list()
#        secpars = eb.filter(dataset=lcs[x], component='secondary').to_list()
#        lcpars = eb.filter(dataset=lcs[x])-eb.filter(dataset=lcs[x], component=primary, secondary

    # write to file
    f = open(filename, 'w')
    f.write('# Phoebe 1 file created from phoebe 2 bundle\n')
    for x in range(len(parnames)):
#        if types[x] == 'float':
#            value = round(parvals[x],6)
#        elif types[x] == 'choice':
#            value = '"'+str(parvals[x])+'"'
#        else:
        value = parvals[x]
        # TODO: set precision on floats?
        f.write(str(parnames[x])+' = '+str(value)+'\n')

    f.close()
#    raise NotImplementedError

    return
