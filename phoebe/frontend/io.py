import numpy as np
import phoebe as phb
import os.path
import logging
from phoebe import conf
from libphoebe import roche_critical_potential
logger = logging.getLogger("IO")
logger.addHandler(logging.NullHandler())

"""
Dictionaries of parameters for conversion between phoebe1 and phoebe 2

"""

_1to2par = {'ld_model':'ld_func',
            'bol':'ld_coeffs_bol',
            'rvcoff': 'ld_coeffs',
            'lccoff':'ld_coeffs',
            'active': 'enabled',
#            'model': 'morphology',
            'filter': 'passband',
            'hjd0':'t0_ref',
#            'hjd0': 't0_supconj',
            'period': 'period',
            'dpdt': 'dpdt',
#            'pshift':'phshift',
            'sma':'sma',
            'rm': 'q',
            'incl': 'incl',
            'pot':'pot',
            'met':'abun',
            'f': 'syncpar',
            'alb': 'irrad_frac_refl_bol',
            'grb':'gravb_bol',
            'ecc': 'ecc',
            'perr0':'per0',
            'dperdt': 'dperdt',
            'hla': 'pblum',
            'cla': 'pblum',
            'el3': 'l3',
            'reflections':'refl_num',
            'finesize': 'gridsize',
            'vga': 'vgamma',
            'teff':'teff',
            'msc1':'msc1',
            'msc2':'msc2',
            'ie':'ie',
            'atm': 'atm',
            'flux':'fluxes',
            'vel':'rvs',
            'sigmarv':'sigmas',
            'sigmalc':'sigmas',
            'time':'times',
            'longitude':'colon',
            'radius': 'radius',
            'tempfactor':'relteff',
            'colatitude':'colat',
            'cadence': 'exptime',
            }
#            'rate':'rate'}

#TODO: add back proximity_rv maybe?
#TODO: add back 'excess': 'extinction',

_2to1par = {v:k for k,v in _1to2par.items()}

_units1 = {'incl': 'deg',
           'period': 'd',
           'dpdt': 'd/d',
           'sma': 'Rsun',
           'vga':'km/s',
           'teff': 'K',
           'perr0': 'rad',
           'dperdt': 'rad/d',
           'flux':'W/m2',
           'sigmalc': 'W/m2',
           'sigmarv': 'km/s',
           'vel':'km/s',
           'time':'d',
           'exptime':'s'}

_parsect = {'t0':'component',
            'period':'component',
            'dpdt':'component',
            'pshift':'component',
            'sma':'component',
            'rm': 'component',
            'incl':'component',
            'perr0':'component',
            'dperdt':'component',
            'hla': 'component',
            'cla':'component',
       #     'el3':'component',
            'reffect':'compute',
            'reflections':'compute',
            'finegrid':'mesh',
            'vga':'system',
            'msc1_switch': 'compute',
            'msc2_switch': 'compute',
            'ie_switch':'compute',
            'proximity_rv1':'compute',
            'proximity_rv2': 'compute'}


#_bool1to2 = {1:True, 0:False}

_bool2to1 = {True:1, False:0}

_bool1to2 = {1:True, 0:False}

"""
ld_legacy -

"""

def ld_to_phoebe(pn, d, rvdep=None, dataid=None, law=None):
    if 'bol' in pn:
        d['context'] = 'component'
        pnew = 'bol'
    elif 'rv' in pn:
        d['context'] = 'dataset'
        d['dataset'] = dataid
        pnew = 'rvcoff'
    elif 'lc' in pn:
        d['context'] = 'dataset'
        d['dataset'] = dataid
        pnew = 'lccoff'
    else:
        pnew = 'ld_model'
    if 'x' in pn:
        d['index'] = 0
    elif 'y' in pn and law != 'Linear cosine law':
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

    elif pieces[1] == 'spots':
        d['component'] = rvdep
        d['context'] = 'feature'
        d['feature'] = dataid

    if pnew == 'hla':
        d['component'] = 'primary'
    elif pnew == 'cla':
        d['component'] = 'secondary'

# two different radius parameters, only include spots
#    print "this is pnew", pnew
    if pnew == 'radius' and pieces[1] != 'spots':
        pnew = None
        d = {}

    if pnew in _1to2par:
        try:
            d.setdefault('context', _parsect[pnew])
        except:
            try:
                d['context']
            except:
                d['context'] = None

        if pnew in ['atm', 'model', 'cindex', 'finesize', 'reffect', 'reflections']:
            d['context']  ='compute'

        if 'proximity' in pnew:
            d['context'] = 'compute'

        if d['context'] == 'component':

            d.setdefault('component', 'binary')

#        if d['context'] == 'compute':
#            d.setdefault('compute', comid)

        if pnew == 'reflections':
            d['value'] = int(val)#+1

        else:
            d['value'] = val

        d['qualifier'] = _1to2par[pnew]

        if pnew in _units1:
            d['unit'] = _units1[pnew]

    else:
        d ={}
        logger.info("Parameter "+str(pname)+" has no Phoebe 2 counterpart")


    return pnew, d

def load_lc_data(filename, indep, dep, indweight=None, mzero=None, dir='./'):

    """
    load dictionary with lc data
    """
    if '/' in filename:
        path, filename = os.path.split(filename)
    else:
        # TODO: this needs to change to be directory of the .phoebe file
        path = dir

    load_file = os.path.join(path, filename)
    lcdata = np.loadtxt(load_file)
    ncol = len(lcdata[0])
    if dep == 'Magnitude':
        mag = lcdata[:,1]
        flux = 10**(-0.4*(mag-mzero))
        lcdata[:,1] = flux

    d = {}
    d['phoebe_lc_time'] = lcdata[:,0]
    d['phoebe_lc_flux'] = lcdata[:,1]
    if indweight=="Standard deviation":
        if ncol >= 3:
            d['phoebe_lc_sigmalc'] = lcdata[:,2]
        else:
            logger.warning('A sigma column was mentioned in the .phoebe file but is not present in the lc data file')
    elif indweight =="Standard weight":
                if ncol >= 3:
                    sigma = np.sqrt(1/lcdata[:,2])
                    d['phoebe_lc_sigmalc'] = sigma
                    logger.warning('Standard weight has been converted to Standard deviation.')

                else:
                    logger.warning('A sigma column was mentioned in the .phoebe file but is not present in the lc data file')
    else:
        logger.warning('Phoebe 2 currently only supports standard deviaton')

#    dataset.set_value(check_visible=False, **d)

    return d

def load_rv_data(filename, indep, dep, indweight=None, dir='./'):

    """
    load dictionary with rv data.
    """

    if '/' in filename:
        path, filename = os.path.split(filename)
    else:
        path = dir

    load_file = os.path.join(path, filename)
    rvdata = np.loadtxt(load_file)

    d ={}
    d['phoebe_rv_time'] = rvdata[:,0]
    d['phoebe_rv_vel'] = rvdata[:,1]
    ncol = len(rvdata[0])

    if indweight=="Standard deviation":

        if ncol >= 3:
            d['phoebe_rv_sigmarv'] = rvdata[:,2]
        else:
            logger.warning('A sigma column is mentioned in the .phoebe file but is not present in the rv data file')
    elif indweight =="Standard weight":
                if ncol >= 3:
                    sigma = np.sqrt(1/rvdata[:,2])
                    d['phoebe_rv_sigmarv'] = sigma
                    logger.warning('Standard weight has been converted to Standard deviation.')
    else:
        logger.warning('Phoebe 2 currently only supports standard deviaton')

    return d

def det_dataset(eb, passband, dataid, comp, time):

    """
    Since RV datasets can have values related to each component in phoebe2, but are component specific in phoebe1, it is important to determine which dataset to add parameters to. This function will do that.
    eb - bundle
    rvpt - relevant phoebe 1 parameters

    """
    rvs = eb.get_dataset(kind='rv').datasets
    #first check to see if there are currently in RV datasets
    if dataid == 'Undefined':
        dataid = None
    if len(rvs) == 0:
    #if there isn't we add one the easy part

        try:
            eb._check_label(dataid)

            rv_dataset = eb.add_dataset('rv', dataset=dataid, times=[])

        except ValueError:

            logger.warning("The name picked for the lightcurve is forbidden. Applying default name instead")
            rv_dataset = eb.add_dataset('rv', times=[])

    else:
    #now we have to determine if we add to an existing dataset or make a new one
        rvs = eb.get_dataset(kind='rv').datasets
        found = False
        #set the component of the companion
        if comp == 'primary':
            comp_o = 'secondary'
        else:
            comp_o = 'primary'
        for x in rvs:
            test_dataset = eb.get_dataset(x)
            if len(test_dataset.get_value(qualifier='rvs', component=comp)) == 0:                #so at least it has an empty spot now check against filter and length
                time_o = test_dataset.get_value('times', component=comp_o)
                passband_o = test_dataset.get_value('passband')
                if np.all(time_o == time) and (passband == passband_o):
                    rv_dataset = test_dataset
                    found = True

        if not found:
            try:
                eb._check_label(dataid)

                rv_dataset = eb.add_dataset('rv', dataset=dataid, times=[])

            except ValueError:

                logger.warning("The name picked for the lightcurve is forbidden. Applying default name instead")
                rv_dataset = eb.add_dataset('rv', times=[])

    return rv_dataset


"""
Load a phoebe legacy file complete with all the bells and whistles

filename - a .phoebe file (from phoebe 1)

"""

def load_legacy(filename, add_compute_legacy=True, add_compute_phoebe=True):
    conf_state = conf.interactive
    conf.interactive_off()
    legacy_file_dir = os.path.dirname(filename)

# load the phoebe file

    params = np.loadtxt(filename, dtype='str', delimiter = ' = ')

    morphology = params[:,1][list(params[:,0]).index('phoebe_model')]


# load an empty legacy bundle and initialize obvious parameter sets
    if 'Overcontact' in morphology:
        contact_binary= True
        eb = phb.Bundle.default_binary(contact_binary=True)
    elif 'Semi-detached' in morphology:
        semi_detached = True
        contact_binary = False
        eb = phb.Bundle.default_binary()
    else:
        semi_detached = False
        contact_binary = False
        eb = phb.Bundle.default_binary()
    eb.disable_history()
    comid = []
#    if add_compute_phoebe == True:
    #    comid.append('phoebe01')
#        eb.add_compute('phoebe')#, compute=comid[0])
    if add_compute_legacy == True:
    #    comid.append('lega1')
        eb.add_compute('legacy')#, compute=comid[-1])


#basic filter on parameters that make no sense in phoebe 2
    ind = [list(params[:,0]).index(s) for s in params[:,0] if not ".ADJ" in s and not ".MIN" in s and not ".MAX" in s and not ".STEP" in s and not "gui_" in s]
    params = params[ind]

# determine number of lcs and rvs
    rvno = np.int(params[:,1][list(params[:,0]).index('phoebe_rvno')])
    lcno = np.int(params[:,1][list(params[:,0]).index('phoebe_lcno')])
# and spots
    spotno = np.int(params[:,1][list(params[:,0]).index('phoebe_spots_no')])

# delete parameters that have already been accounted for and find lc and rv parameters

    params = np.delete(params, [list(params[:,0]).index('phoebe_lcno'), list(params[:,0]).index('phoebe_rvno')], axis=0)


# check to see if reflection is on

    ref_effect = np.int(params[:,1][list(params[:,0]).index('phoebe_reffect_switch')])

    if ref_effect == 0:

        params[:,1][list(params[:,0]).index('phoebe_reffect_reflections')] = 0
        logger.warning('Phoebe Legacy reflection effect switch is set to false so refl_num is being set to 0.')

    if not add_compute_legacy:
        params = np.delete(params, [list(params[:,0]).index('phoebe_reffect_reflections'), list(params[:,0]).index('phoebe_ie_switch')], axis=0)

    if 'Overcontact' in morphology:
        params = np.delete(params, [list(params[:,0]).index('phoebe_pot2.VAL')], axis=0)
        if 'UMa'in morphology:
            params[:,1][list(params[:,0]).index('phoebe_teff2.VAL')] = params[:,1][list(params[:,0]).index('phoebe_teff1.VAL')]
            params[:,1][list(params[:,0]).index('phoebe_grb2.VAL')] = params[:,1][list(params[:,0]).index('phoebe_grb1.VAL')]
            params[:,1][list(params[:,0]).index('phoebe_alb2.VAL')] = params[:,1][list(params[:,0]).index('phoebe_alb1.VAL')]
    elif 'Semi-detached' in morphology:

        if "primary" in morphology:
            params = np.delete(params, [list(params[:,0]).index('phoebe_pot1.VAL')], axis=0)
        elif "secondary" in morphology:
            params = np.delete(params, [list(params[:,0]).index('phoebe_pot2.VAL')], axis=0)

    else:
        #none of the other constraints are useful
        pass

#pull out global values for fti
    try:
        fti = _bool1to2[int(params[:,1][list(params[:,0]).index('phoebe_cadence_switch')])]
        fti_exp = params[:,1][list(params[:,0]).index('phoebe_cadence')]
        fti_ovs = params[:,1][list(params[:,0]).index('phoebe_cadence_rate')]
        fti_ts = params[:,1][list(params[:,0]).index('phoebe_cadence_timestamp')].strip('"')
        params = np.delete(params, [list(params[:,0]).index('phoebe_cadence'), list(params[:,0]).index('phoebe_cadence_switch'), list(params[:,0]).index('phoebe_cadence_rate'), list(params[:,0]).index('phoebe_cadence_timestamp')], axis=0)

    except:

        fti = False
        fti_exp = '1766'
        fti_ovs = '0'
        fti_ts = 'Mid-exposure'
#    params =  np.delete(params, [list(params[:,0]).index('phoebe_cadence'), list(params[:,0]).index('phoebe_cadence_switch')], axis=0)#, list(params[:,0]).index('phoebe_cadence_rate'), list(params[:,0]).index('phoebe_cadence_timestamp')], axis=0)

#    fti_type = params[:,1][list(params[:,0]).index('phoebe_cadenc_rate')]
# create mzero and grab it if it exists
    mzero = None
    if 'phoebe_mnorm' in params:
        mzero = np.float(params[:,1][list(params[:,0]).index('phoebe_mnorm')])
# determine if luminosities are decoupled and set pblum_ref accordingly
    try:
        decoupled_luminosity = np.int(params[:,1][list(params[:,0]).index('phoebe_usecla_switch')])
    except:
        pass
#    if decoupled_luminosity == 0:
#        eb.set_value(qualifier='pblum_ref', component='secondary', value='primary')
#    else:
#        eb.set_value(qualifier='pblum_ref', component='secondary', value='self')

#Determin LD law

    ldlaw = params[:,1][list(params[:,0]).index('phoebe_ld_model')]
# FORCE hla and cla to follow conventions so the parser doesn't freak out.
    for x in range(1,lcno+1):
        hlain = list(params[:,0]).index('phoebe_hla['+str(x)+'].VAL')
#        clain = list(params[:,0]).index('phoebe_cla['+str(x)+'].VAL')
        params[:,0][hlain] = 'phoebe_lc_hla1['+str(x)+'].VAL'
#        params[:,0][clain] = 'phoebe_lc_cla2['+str(x)+'].VAL'
        hla = np.float(params[:,1][hlain]) #pull for possible conversion of l3
#        cla = np.float(params[:,1][clain]) #pull for possible conversion of l3

        if contact_binary:
            params = np.delete(params, [list(params[:,0]).index('phoebe_lc_cla2['+str(x)+'].VAL')], axis=0)

#and split into lc and rv and spot parameters

    lcin = [list(params[:,0]).index(s) for s in params[:,0] if "lc" in s]
    rvin = [list(params[:,0]).index(s) for s in params[:,0] if "rv" in s and not "proximity" in s]
    spotin = [list(params[:,0]).index(s) for s in params[:,0] if "spots" in s]
    lcpars = params[lcin]
    rvpars = params[rvin]
    spotpars = params[spotin]
    lcin.extend(rvin)
    lcin.extend(spotin)
    params = np.delete(params, lcin, axis=0)

# create datasets and fill with the correct parameters


# First LC
    for x in range(1,lcno+1):

        #list of parameters related to current dataset

        lcint = [list(lcpars[:,0]).index(s) for s in lcpars[:,0] if "["+str(x)+"]" in s]
        lcpt = lcpars[lcint]



        #determine whether individual fti parameter exists and add them if not
 #       print 'phoebe_lc_cadence_switch['+str(x)+']', int(lcpt[:,1][list(lcpt[:,0]).index('phoebe_lc_cadence_switch['+str(x)+']')])
 #       fti_ts_ind = lcpt[:,1][list(lcpt[:,0]).index('phoebe_lc_cadence_timestamp['+str(x)+']')].strip('"')

 #       print fti_ts_ind

#        if fti_ts_ind != 'Mid-exposure':
#            print "i shouldn't go here"

#            logger.warning('Phoebe 2 only uses Mid-Exposure for calculating finite exposure times.')


        try:
            fti_ind = _bool1to2[int(lcpt[:,1][list(lcpt[:,0]).index('phoebe_lc_cadence_switch['+str(x)+']')])]
            fti_ts_ind = lcpt[:,1][list(lcpt[:,0]).index('phoebe_lc_cadence_timestamp['+str(x)+']')].strip('"')

            if fti_ts_ind != 'Mid-exposure':
                logger.warning('Phoebe 2 only uses Mid-Exposure for calculating finite exposure times.')

        #    print 'phoebe_lc_cadence_switch['+str(x)+']', fti_ind

        except:

            logger.warning('Your .phoebe file was created using a version of phoebe which does not support dataset dependent finite integration time parameters')
            fti_val = _bool2to1[fti]
            ftia = np.array(['phoebe_lc_cadence_switch['+str(x)+']', fti_val])
            fti_expa = np.array(['phoebe_lc_cadence['+str(x)+']', fti_exp])
            fti_ovsa = np.array(['phoebe_lc_cadence_rate['+str(x)+']', fti_ovs])
            fti_tsa = np.array(['phoebe_lc_cadence_timestamp['+str(x)+']', fti_ts])
            lcpt = np.vstack((lcpt,ftia,fti_expa,fti_ovsa, fti_tsa))

            fti_ind = False


        if not fti_ind:

            if fti_ts != 'Mid-exposure':
                logger.warning('Phoebe 2 only uses Mid-Exposure times for calculating finite exposure times.')
            if fti:
                lcpt[:,1][list(lcpt[:,0]).index('phoebe_lc_cadence_rate['+str(x)+']')] = fti_ovs

            else:
                lcpt[:,1][list(lcpt[:,0]).index('phoebe_lc_cadence_rate['+str(x)+']')] = 'None'

            lcpt[:,1][list(lcpt[:,0]).index('phoebe_lc_cadence['+str(x)+']')] = fti_exp

#            lcpt[:,1][list(lcpt[:,0]).index('phoebe_lc_cadence_rate['+str(x)+']')] = fti_ovs






#STARTS HERE
        lc_dict = {}

        for y in range(len(lcpt)):
            parameter = lcpt[y][0].split('[')[0]
            lc_dict[parameter] = lcpt[:,1][y].strip('"')

        #add third light
        l3 = np.float(params[:,1][list(params[:,0]).index('phoebe_el3['+str(x)+'].VAL')])


        if params[:,1][list(params[:,0]).index('phoebe_el3_units')].strip('"') == 'Total light':
            logger.warning('l3 as a percentage of total light is currently not supported in phoebe 2')
            l3=0
#            l3 = l3/(4.0*np.pi)*(hla+cla)/(1.0-l3)

        lc_dict['phoebe_lc_el3'] = l3

    # Determine the correct dataset to open
    # create rv data dictionary
#        indweight = lc_dict['phoebe_lc_indweight']

#        if indweight == 'Unavailable':
#            indweight = None

        #make sure filename parameter exists and if not add it

        try:
            lc_dict['phoebe_lc_filename']
        except:
            lc_dict['phoebe_lc_filename'] = 'Undefined'



        if mzero != None and lc_dict['phoebe_lc_filename'] != 'Undefined':

            indweight = lc_dict['phoebe_lc_indweight']

            if indweight == 'Unavailable':
                indweight = None

            data_dict = load_lc_data(filename=lc_dict['phoebe_lc_filename'],  indep=lc_dict['phoebe_lc_indep'], dep=lc_dict['phoebe_lc_dep'], indweight=indweight, mzero=mzero, dir=legacy_file_dir)

            lc_dict.update(data_dict)

    # get dataid and load
#datain = list(lcpt[:,0]).index('phoebe_lc_id['+str(x)+']')
        dataid = lc_dict['phoebe_lc_id']
        del lc_dict['phoebe_lc_id']

        if dataid == 'Undefined':

            lc_dataset = eb.add_dataset('lc')

        else:

            try:
                eb._check_label(dataid)

                lc_dataset = eb.add_dataset('lc', dataset=dataid)

            except ValueError:

                logger.warning("The name picked for the lightcurve is forbidden. Applying default name instead")
                lc_dataset = eb.add_dataset('lc')

    # get name of new dataset

        dataid = lc_dataset.dataset

    #enable dataset
        enabled = lc_dict['phoebe_lc_active']
        del lc_dict['phoebe_lc_active']

        d ={'qualifier':'enabled', 'dataset':dataid, 'value':enabled}
        eb.set_value_all(check_visible= False, **d)

    #set pblum reference

        if decoupled_luminosity == 0:
            eb.set_value(qualifier='pblum_ref', component='secondary', value='primary', dataset=dataid)
        else:
            eb.set_value(qualifier='pblum_ref', component='secondary', value='self', dataset=dataid)

    #get available passbands

        choices = lc_dataset.get_parameter('passband').choices

    #create parameter dictionary

    # cycle through all parameters. separate out data parameters and model parameters. add model parameters

        for k in lc_dict:
            pnew, d = ret_dict(k, lc_dict[k], dataid=dataid)
#            print d
        # as long as the parameter exists add it
            if len(d) > 0:

                if d['qualifier'] == 'passband' and d['value'] not in choices:
                    d['value'] = 'Johnson:V'

#                if d['qualifier'] == 'pblum' and contact_binary:

#                    d['component'] = 'contact_envelope'

                try:
                    eb.set_value_all(check_visible=False, **d)
    #                del d['value']
     #               print "Value", eb.get_value(**d)
                except ValueError, msg:
                    raise ValueError(msg.message + " ({})".format(d))

#Now rvs
    for x in range(1,rvno+1):
        rvs = eb.get_dataset(kind='rv').datasets

    #list of parameters related to current dataset
        rvint = [list(rvpars[:,0]).index(s) for s in rvpars[:,0] if "["+str(x)+"]" in s]
        rvpt = rvpars[rvint]

    #determine whether to use global or individual fti parameters
        try:
            fti_ind = _bool1to2[int(rvpt[:,1][list(rvpt[:,0]).index('phoebe_rv_cadence_switch['+str(x)+']')])]
            rvpt = np.delete(rvpt, list(rvpt[:,0]).index('phoebe_rv_cadence['+str(x)+']'), axis=0)
            rvpt = np.delete(rvpt, list(rvpt[:,0]).index('phoebe_rv_cadence_rate['+str(x)+']'), axis=0)
        except:
            logger.warning('finite integration time is not currently supported for RV datasets in Phoebe 2')

    #    if fti_ind:
    #        if fti:
    #            rvpt = np.delete(rvpt, list(rvpt[:,0]).index('phoebe_rv_cadence['+str(x)+']'))
            #    rvpt[:,1][list(rvpt[:,0]).index('phoebe_rv_cadence['+str(x)+']')] = fti_exp

    #        else:
    #            rvpt = np.delete(rvpt, list(rvpt[:,0]).index('phoebe_rv_cadence['+str(x)+']'))
            #    rvpt[:,1][list(rvpt[:,0]).index('phoebe_rv_cadence['+str(x)+']')] = 0.0

    #        rvpt = np.delete(rvpt, list(rvpt[:,0]).index('phoebe_rv_cadence['+str(x)+']'))
    #        rvpt[:,1][list(rvpt[:,0]).index('phoebe_rv_cadence_rate['+str(x)+']')] = fti_ovs


#create rv dictionary
        rv_dict = {}
        for x in range(len(rvpt)):
#    parameter = rvpt[x][0].split('_')[-1].split('[')[0]
            parameter = rvpt[x][0].split('[')[0]

            rv_dict[parameter] = rvpt[:,1][x].strip('"')
    # grab some parameters we'll need

        passband = rv_dict['phoebe_rv_filter']
        time = []
        dataid = rv_dict['phoebe_rv_id']
        del rv_dict['phoebe_rv_id']

        comp = rv_dict['phoebe_rv_dep'].split(' ')[0].lower()

    # create rv data dictionary
#        indweight = rv_dict['phoebe_rv_indweight']

#        if indweight == 'Unavailable':
#            indweight = None

        try:
            rv_dict['phoebe_rv_filename']
        except:
            rv_dict['phoebe_rv_filename'] = 'Undefined'


        if rv_dict['phoebe_rv_filename'] != 'Undefined':
            indweight = rv_dict['phoebe_rv_indweight']

            if indweight == 'Unavailable':
                indweight = None

            data_dict = load_rv_data(filename=rv_dict['phoebe_rv_filename'], indep=rv_dict['phoebe_rv_indep'], dep=rv_dict['phoebe_rv_dep'], indweight=indweight, dir=legacy_file_dir)

            rv_dict.update(data_dict)
            time = rv_dict['phoebe_rv_time']
        #

        rv_dataset = det_dataset(eb, passband, dataid, comp, time)
        dataid = rv_dataset.dataset
        #enable dataset
        enabled = rv_dict['phoebe_rv_active']
        del rv_dict['phoebe_rv_active']

        d ={'qualifier':'enabled', 'dataset':dataid, 'value':enabled}
        eb.set_value_all(check_visible= False, **d)

    #get available passbands and set

        choices = rv_dataset.get_parameter('passband').choices
        pnew, d = ret_dict('phoebe_rv_filter', rv_dict['phoebe_rv_filter'], dataid=dataid)
        if d['qualifier'] == 'passband' and d['value'] not in choices:
                d['value'] = 'Johnson:V'
        eb.set_value_all(check_visible= False, **d)
        del rv_dict['phoebe_rv_filter']
# now go through parameters and input the results into phoebe2
        for k  in rv_dict:

            pnew, d = ret_dict(k, rv_dict[k], rvdep = comp, dataid=dataid)

            if len(d) > 0:
                eb.set_value_all(check_visible= False, **d)

# And finally spots

    spot_unit =  spotpars[:,1][list(spotpars[:,0]).index('phoebe_spots_units')].strip('"').lower()[:-1]

    for x in range(1,spotno+1):

        spotin = [list(spotpars[:,0]).index(s) for s in spotpars[:,0] if "["+str(x)+"]" in s]
        spotpt = spotpars[spotin]
        source =  np.int(spotpt[:,1][list(spotpt[:,0]).index('phoebe_spots_source['+str(x)+']')])
        spotpt = np.delete(spotpt, list(spotpt[:,0]).index('phoebe_spots_source['+str(x)+']'), axis=0)


        if source == 1:
            component = 'primary'
        elif source == 2:
            component = 'secondary'
        else:
            raise ValueError("spot component not specified and cannot be added")

#   create spot

        spot = eb.add_feature('spot', component=component)
        #TODO check this tomorrow
        dataid = spot.features[0]
#   add spot parameters

        for k in range(len(spotpt)):
            param = spotpt[:,0][k].split('[')[0]
            value = spotpt[:,1][k]
            # print "param", param
            pnew, d = ret_dict(param, value, rvdep = component, dataid=dataid)
            if len(d) > 0:
                if d['qualifier'] != 'relteff':
                    d['unit'] = spot_unit
                # print "dictionary", d
                eb.set_value_all(check_visible= False, **d)



    for x in range(len(params)):

        pname = params[:,0][x]
        pname = pname.split('.')[0]
        val = params[:,1][x].strip('"')
        pnew, d = ret_dict(pname, val)
        if pnew == 'ld_model':
            ldlaws_1to2= {'Linear cosine law': 'linear', 'Logarithmic law': 'logarithmic', 'Square root law': 'square_root'}
            if val == 'Linear cosine law':
                logger.warning('Linear cosine law is not currently supported. Converting to linear instead')
            d['value'] = ldlaws_1to2[val]#val[0].lower()+val[1::]

            # since ld_coeffs is dataset specific make sure there is at least one dataset
            if lcno != 0 or rvno != 0:
                eb.set_value_all(check_visible=False, **d)
            #now change to take care of bolometric values
            d['qualifier'] = d['qualifier']+'_bol'
        if pnew == 'pot':
            #print "dict", d
            d['kind'] = 'star'
            d.pop('qualifier') #remove qualifier from dictionary to avoid conflicts in the future
            d.pop('value') #remove qualifier from dictionary to avoid conflicts in the future

            if not contact_binary:
                eb.flip_constraint(solve_for='rpole', qualifier='pot', **d)
#                eb.flip_constraint(solve_for='rpole', constraint_func='potential', **d) #this WILL CHANGE & CHANGE back at the very end
            #print "val", val
            else:
                d['component'] = 'contact_envelope'
            d['value'] = val
            d['qualifier'] = 'pot'
            d['kind'] = None
            d['context'] = 'component'
    # change t0_ref and set hjd0
        if pnew == 'hjd0':

            d.pop('qualifier') #avoiding possible conflicts
            d.pop('value') #avoiding possible conflicts
            #
            #
            eb.flip_constraint(solve_for='t0_supconj', constraint_func='t0_ref_supconj', **d)
    #        elif pnew == 'filter':
    #       make sure t0 accounts for any phase shift present in phoebe 1
            try:
                #if being reimported after phoebe2 save this parameter won't exist
                pshift_in = list(params[:,0]).index('phoebe_pshift.VAL')
                pshift = np.float(params[:,1][pshift_in])
            except:
                pshift = 0.0

            period_in = list(params[:,0]).index('phoebe_period.VAL')
            period = np.float(params[:,1][period_in])

            t0 = float(val)+pshift*period
    #       new
            d['value'] = t0
            d['qualifier'] = 't0_ref'
             # write method for this

    #        elif pnew == 'excess':
                     # requires two parameters that phoebe 1 doesn't have access to Rv and extinction
        # elif pnew == 'alb':
            # val = 1.-float(val)
            # d['value'] = val
        elif pnew == 'atm':
            val = int(val)

            if val == 0:
                d['value'] = 'extern_planckint'
            if val == 1:
                d['value'] = 'extern_atmx'
            logger.warning('If you would like to use phoebe 1 atmospheres, you must add this manually')
            if add_compute_legacy:
                d['kind'] = 'legacy'
                eb.set_value(check_relevant=False, **d)

            d['kind'] = 'phoebe'
            d['value'] = 'ck2004'
#            atm_choices = eb.get_compute('detailed').get_parameter('atm', component='primary').choices
#            if d['value'] not in atm_choices:
                #TODO FIND appropriate default
#                d['value'] = 'atmcof'

#       change vgamma so that the definitions match

        elif pnew == 'vga':

            d['value'] = -1*float(val)

        elif pnew == 'finesize':
                    # set gridsize
            d['value'] = val
            if conf.devel:
                eb.set_value_all(check_visible=False, **d)
            # change parameter and value to ntriangles
            val = N_to_Ntriangles(int(np.float(val)))
            d['qualifier'] = 'ntriangles'
            d['value'] = val
#        elif pnew == 'refl_num':
        if len(d) > 0:
#            print d
            eb.set_value_all(check_visible=False, **d)
    #print "before", eb['pot@secondary']
    #print "rpole before", eb['rpole@secondary']
    if semi_detached:
        q = eb.get_value(qualifier ='q')
        d = 1. - eb.get_value(qualifier='ecc')
        if 'primary' in morphology:

            eb.add_constraint('critical_rpole', component='primary')
#            eb.flip_constraint(solve_for='rpole', constraint_func='potential', component='primary')
#            f = eb.get_value(qualifier='syncpar', component='primary')
#            crit_pots = roche_critical_potential(q,f,d)
#            eb.set_value(qualifier='pot', component='primary', context='component', value=crit_pots['L1'])
        elif 'secondary' in morphology:
            eb.add_constraint('critical_rpole', component='secondary')
#            eb.flip_constraint(solve_for='rpole', constraint_func='potential', component='secondary')
#            f = eb.get_value(qualifier='syncpar', component='secondary')
#            crit_pots = roche_critical_potential(1/q,f,d)
#            eb.set_value(qualifier='pot', component='secondary', context='component', value=crit_pots['L1'])

#flip back all constraints
    if not contact_binary:
        #avoid semi_detached where constraint hasn't been flipped
        if semi_detached and 'primary' not in morphology:
            eb.flip_constraint(solve_for='pot', qualifier='rpole', component='primary')
        elif semi_detached and 'secondary' not in morphology:
            eb.flip_constraint(solve_for='pot', qualifier='rpole', component='secondary')
        else:
            eb.flip_constraint(solve_for='pot', qualifier='rpole', component='primary')
            eb.flip_constraint(solve_for='pot', qualifier='rpole', component='secondary')

#        eb.flip_constraint(solve_for='pot', constraint_func='potential', component='primary')
#        eb.flip_constraint(solve_for='pot', constraint_func='potential', component='secondary')
    # get rid of seconddary coefficient if ldlaw  is linear
    eb.flip_constraint(solve_for='t0_ref', constraint_func='t0_ref_supconj')

    if 'Linear' in ldlaw:

        ldcos = eb.filter('ld_coeffs')
        ldcosbol = eb.filter('ld_coeffs_bol')
        for x in range(len(ldcos)):
            val = ldcos[x].value[0]
            ldcos[x].set_value(np.array([val]))

        for x in range(len(ldcosbol)):

            val = ldcosbol[x].value[0]
            ldcosbol[x].set_value(np.array([val]))
    if conf_state:
        eb.run_delayed_constraints()
        conf.interactive_on()
    #print eb['pot@secondary']
    #print "rpole after", eb['rpole@secondary']
    # turn on relevant switches like heating. If
    return eb

"""

Return phoebe1 parameter value in the right units (or without units)
eb -phoebe 2 bundle
d - dictionary with parameter, context, dataset etc.

"""


def par_value(param, index=None, **kwargs):
# build a dictionary

    d={}
    d['qualifier'] = param.qualifier
    d['component'] = param.component
    d['dataset'] = param.dataset
    d['compute'] = param.compute
    d['kind'] = param.kind

# Determine what type of parameter you have and find it's value
    if isinstance(param, phb.parameters.FloatParameter) and not isinstance(param, phb.parameters.FloatArrayParameter):
        ptype = 'float'
    # since it's a float it must have units. Therefore return value in correct units
        pnew = _2to1par[d['qualifier']]
        try:
            unit = _units1[pnew]
        except:
            unit = None

        val = param.get_quantity(unit=unit, **kwargs).value
        # if d['qualifier'] == 'alb':
            # val = [1.0-val]

        # else:
            # val = [val]
        val = [val]
    elif isinstance(param, phb.parameters.ChoiceParameter):
        ptype = 'choice'
        val = [param.get_value(**kwargs)]
        if d['qualifier'] == 'atm':
        # in phoebe one this is a boolean parameter because you have the choice of either kurucz or blackbody

            ptype='boolean'

        if d['qualifier'] == 'ld_func' or d['qualifier'] == 'ld_func_bol':

            ldlaws_2to1= {'linear':'Linear cosine law', 'logarithmic':'Logarithmic law', 'square_root':'Square root law'}
            val = ldlaws_2to1[val[0]]
            val = ['"'+str(val)+'"']

    elif isinstance(param, phb.parameters.BoolParameter):

        ptype = 'boolean'
        val = [_bool2to1[param.get_value(**kwargs)]]

    elif isinstance(param, phb.parameters.IntParameter):
        ptype = 'int'
        val = [param.get_value(**kwargs)]

        # if d['qualifier'] == 'refl_num':

        #    val = [param.get_value()-1]
            # val = [param.get_value()]

        # else:
            # val = [param.get_value()]

    elif isinstance(param, phb.parameters.FloatArrayParameter):
        # val1 = param.get_value()[0]
        # val2 = param.get_value()[1]
        # val = [val1, val2]

        val = param.get_value(**kwargs).tolist()

        ptype='array'
        if len(val) == 1:
            val.append(0.0)

    else:
        ptype = 'unknown'
        val = [param.get_value(**kwargs)]

    return [val, ptype]

"""

Return phoebe1 parameter name from phoebe 2 info


"""

def ret_ldparname(param, comp_int=None, dtype=None, dnum=None, ptype=None, index=None):
    if 'bol' in param:

        if ptype=='array':
            pnew1 = 'xbol'
            pnew2 = 'ybol'
            pnew = [pnew1, pnew2]
        else:
            return ['phoebe_ld_model']
    else:
        if ptype == 'array':
            pnew1 = str(dtype)+'x'
            pnew2 = str(dtype)+'y'
            pnew = [pnew1, pnew2]
        else:
            return ['phoebe_ld_model']

    if comp_int == 1:
        pnew = [x + '1' for x in pnew]

    elif comp_int == 2:
        pnew = [x + '2' for x in pnew]

    if dnum != None:
        dset = '['+str(dnum)+'].VAL'
    else:
        dset = ''
    return ['phoebe_ld_'+x+dset for x in pnew]

def ret_parname(param, comp_int=None, dtype=None, dnum=None, ptype=None, index=None):

# separate lds from everything because they suck

    if 'ld' in param:

        pname = ret_ldparname(param, comp_int=comp_int, dtype=dtype, dnum=dnum, ptype=ptype, index=index)
    else:
    # first determine name of parameters and whether it is associated with a com

        if comp_int == 1:

            if param == 'pblum':
                pnew = 'hla'

            elif param in ['enabled','statweight','l3','passband']:
                pnew = _2to1par[param]

            else:
                pnew = _2to1par[param]+'1'

        elif comp_int == 2:

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
            if param != 'l3':
                dtype = dtype+'_'
            else:
                dtype = ''
        else:
            dtype = ''

        if dnum != None:

            dset = '['+str(dnum)+']'
        else:
            dset = ''
    # determine the determinant of the parameter based on parameter type
        # print "check", param, ptype, dtype
        if ptype == 'float':
            det = '.VAL'

        elif ptype == 'boolean' and dtype=='':
            # print "inside", param, ptype, dtype
            det = '_switch'
        else:
            det = ''

        pname = ['phoebe_'+str(dtype)+str(pnew)+dset+det]

    return pname

"""

Create a .phoebe file from phoebe 1 from a phoebe 2 bundle.

"""

def pass_to_legacy(eb, filename='2to1.phoebe', compute=None, **kwargs):

    eb.run_delayed_constraints()


    # check to make sure you have exactly two stars and exactly one orbit
    stars = eb['hierarchy'].get_stars()
    orbits = eb['hierarchy'].get_orbits()
    primary, secondary = stars

    if len(stars) != 2 or len(orbits) != 1:
        raise ValueError("Phoebe 1 only supports binaries. Either provide a different system or edit the hierarchy.")
# check for contact_binary
    contact_binary = eb.hierarchy.is_contact_binary(primary)
# check for semi_detached

# grab all possible constraints that could affect semi_detached status
    sd_constraints = eb.filter(context='constraint', qualifier='pot')+eb.filter(context='constraint', qualifier='rpole')
    no_sd_constraints = 0 #keep track of number of constraints as phoebe 1 can't
    #handle two semi_detached stars

    for i in range(len(sd_constraints)):
        if sd_constraints[i].constraint_func =='critical_rpole' or sd_constraints[i].constraint_func =='critical_potential':
            no_sd_constraints = no_sd_constraints+1
            semi_detached = True
            if primary == sd_constraints[i].component:
                semid_comp = 'primary' #semidatched is primary
            if secondary == sd_constraints[i].component:
                    semid_comp = 'secondary'
            if no_sd_constraints > 1:
                semid_comp = 'primary'
                logger.warning('Phoebe 1 does not support double Roche lobe overflow system. Defaulting to Primary star only.')
        else:
            semi_detached = False

#    if 'rpole' in eb['constraint'].qualifiers:
#        semi_detached = eb.get_parameter('rpole', context='constraint').constraint_func == 'critical_rpole'
#  catch all the datasets
    # define datasets


    lcs = eb.get_dataset(kind='lc').datasets
    rvs = eb.get_dataset(kind='rv').datasets
    spots = eb.features

    #make lists to put results with important things already added

    parnames = ['phoebe_rvno', 'phoebe_spots_no', 'phoebe_lcno']
    parvals = [len(rvs), len(spots), len(lcs)]
    types = ['int', 'int']
    #Force the independent variable to be time

    parnames.append('phoebe_indep')
    parvals.append('"Time (HJD)"')
    types.append('choice')
    # Force el3 unit to be flux
    parnames.append('phoebe_el3_units')
    parvals.append('"Flux"')
    types.append('choice')

# add limb darkening law first because it exists many places in phoebe2



    ldlaws = set([p.get_value() for p in eb.filter(qualifier='ld_func').to_list()])

    ldlaws_bol = set([p.get_value() for p in eb.filter(qualifier='ld_func_bol').to_list()])


    #no else
    if len(ldlaws) == 1:

        #check values
        if list(ldlaws)[0] not in ['linear', 'logarithmic', 'square_root']:
            raise ValueError(list(ldlaws)[0]+" is not an acceptable value for phoebe 1. Accepted options are 'linear', 'logarithmic' or 'square_root'")
        #define choices
        if ldlaws != ldlaws_bol:
            logger.warning('ld_func_bol does not match ld_func. ld_func will be chosen')

        param = eb.filter('ld_func', component=primary)[0]
        val, ptype = par_value(param)
        pname = ret_parname(param.qualifier)
        #load to array
        parnames.extend(pname)
        parvals.extend(val)
        types.append(ptype)

    elif len(set(ldlaws)) > 1:
        raise ValueError("Phoebe 1 takes only one limb darkening law.")

    else:
        if list(ldlaws_bol)[0] not in ['linear', 'logarithmic', 'square_root']:
            raise ValueError(list(ldlaws)[0]+" is not an acceptable value for phoebe 1. Accepted options are 'linear', 'logarithmic' or 'square_root'")

        param = eb.filter('ld_func_bol', component=primary)[0]
        val, ptype = par_value(param)
        pname = ret_parname(param.qualifier)
        #load to array
        parnames.extend(pname)
        parvals.extend(val)
        types.append(ptype)
#        raise ValueError("You have not defined a valid limb darkening law.")


#    if len(ldlaws) == 0:
#        pass



    if len(lcs) != 0:

        pblum_ref = eb.get_value(dataset = lcs[0], qualifier = 'pblum_ref', component=secondary)
        # print "pblum_ref", pblum_ref
        if pblum_ref == 'self':

            decouple_luminosity = '1'

        else:

            decouple_luminosity = '0'

        parnames.append('phoebe_usecla_switch')
        parvals.append(decouple_luminosity)
        types.append('boolean')

    prpars = eb.filter(component=primary, context='component')
    secpars = eb.filter(component=secondary, context='component')
    if contact_binary:
        #note system
        parnames.append('phoebe_model')
        parvals.append('"Overcontact binary not in thermal contact"')
        types.append('choice')

        comp_int = 1
        envelope = eb.hierarchy.get_siblings_of(primary)[-1]
#        cepars = eb.filter(component='contact_envelope', context='component')
#   potential
        val = [eb.get_value(qualifier='pot', component=envelope, context='component')]
        ptype = 'float'
        # note here that phoebe1 assigns this to the primary, not envelope
        pname = ret_parname('pot', comp_int=comp_int, ptype=ptype)
        parnames.extend(pname)
        parvals.extend(val)
    elif semi_detached:
        parnames.append('phoebe_model')
        parvals.append('"Semi-detached binary, '+semid_comp+' star fills Roche lobe')
        types.append('choice')

#   pblum
        # TODO BERT: need to deal with multiple datasets
 #       for x in range(len(lcs)):
 #           val = [eb.get_value(qualifier='pblum', component=primary, context='dataset', dataset=lcs[x])]
 #           ptype = 'float'
 #           pname = ret_parname('pblum', comp_int=comp_int, ptype=ptype)
 #           parnames.extend(pname)
 #           parvals.extend(val)
    # get primary parameters and convert
    else:
        parnames.append('phoebe_model')
        parvals.append('"Detached binary"')
        types.append('choice')

    for param in prpars.to_list():

        comp_int = 1
#        if isinstance(eb.get_parameter(prpars[x], component='primary'), phoebe.parameters.FloatParameter):

#        param = eb.get_parameter(prpars[x], component='primary')
        try:
            pnew = _2to1par[param.qualifier]
            if param.qualifier in ['sma', 'period', 'incl','enabled','statweight','l3'] or param.component == '_default':
                param = None
#            elif 'ld_' in param.qualifier:
#                param = None

        except:
            logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')
            param=None
        if param != None:
            val, ptype = par_value(param)

            # if param.qualifier == 'irrad_frac_refl_bol':
                # val = [1-float(val[0])]
            pname = ret_parname(param.qualifier, comp_int = comp_int, ptype=ptype)
            # print val, ptype, pname
            if pname[0] not in parnames:

                parnames.extend(pname)
                parvals.extend(val)
                if ptype == 'array':
                    types.append(ptype)
                    types.append(ptype)
                else:
                    types.append(ptype)

    for param in secpars.to_list():
        comp_int = 2
        # make sure this parameter exists in phoebe 1
#        param = eb.get_parameter(secpars[x], component= 'secondary')
        try:
            pnew = _2to1par[param.qualifier]
            if param.qualifier in ['sma', 'period', 'incl', 'ld_func', 'ld_func_bol'] or param.component == '_default':
                param = None

        except:
            logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')
            param = None

# get rid of confusing parameters like sma and period which only exist for orbits in phoebe 1

        if param != None:

            val, ptype = par_value(param)
            # if param.qualifier == 'irrad_frac_refl_bol':
                # val = [1-float(val[0])]
            pname = ret_parname(param.qualifier, comp_int = comp_int, ptype=ptype)
            if pname[0] not in parnames:
                parnames.extend(pname)
                parvals.extend(val)
                if ptype == 'array':
                    types.append(ptype)
                    types.append(ptype)
                else:
                    types.append(ptype)

#  catch all the datasets

#    lcs = eb.get_dataset(kind='LC').datasets
#    rvs = eb.get_dataset(kind='RV').datasets
#    spots = eb.features

# add all important parameters that must go at the top of the file


# loop through lcs

    for x in range(len(lcs)):
        quals = eb.filter(dataset=lcs[x], context='dataset')+eb.filter(dataset=lcs[x], context='compute')
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
            if param.component == primary:
                comp_int = 1
            elif param.component == secondary:
                comp_int = 2
            else:
                comp_int = None

            try:
                pnew = _2to1par[param.qualifier]
                if param.qualifier in [ 'alb', 'fluxes', 'sigmas', 'times'] or param.component == '_default':

                    param = None
            except:

                logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')
                param = None

            if param != None:

                val, ptype = par_value(param)

                if param.qualifier == 'pblum':
                    if contact_binary:
                        pname = ret_parname(param.qualifier, comp_int= 1, dnum = x+1, ptype=ptype)
                    else:
                        pname = ret_parname(param.qualifier, comp_int= comp_int, dnum = x+1, ptype=ptype)

                elif param.qualifier == 'exptime':

                    logger.warning("Finite integration Time is not fully supported and will be turned off by legacy wrapper before computation")
                    pname = ['phoebe_cadence_switch']
                    val = ['0']
                    ptype='boolean'
#                    if pname[0] not in parnames:
#                        parnames.extend(pname)
#                        parvals.extend(val)
#                        types.append('boolean')
#                elif param.qualifier == 'l3':
#                    pname = ['phoebe_el3']
#                    val = val*4*np.pi
#                    ptype = 'float'
                else:

                    pname = ret_parname(param.qualifier, comp_int=comp_int, dtype='lc', dnum = x+1, ptype=ptype)
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
        quals = eb.filter(dataset=rvs[y], context='dataset')+eb.filter(dataset=rvs[y], context='compute')

        #if there is more than 1 rv try this
        try:
            comp = eb.get_parameter(qualifier='times', dataset=rvs[y]).component
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
                if param.component == primary:
                    comp_int = 1
                elif param.component == secondary:
                    comp_int = 2
                else:
                    comp_int = None

#            if len(eb.filter(qualifier=quals[y], dataset=rvs[x])) == 1:
                try:
                    pnew = _2to1par[param.qualifier]
                    if param.qualifier in ['ld_func', 'rvs', 'times', 'sigmas'] or param.component == '_default':
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
                parnames.append('phoebe_rv_id['+str(i+1)+']')
                parvals.append(rvs[y])
                types.append('choice')

                for param in quals.to_list():
                    if param.component == primary:
                        comp_int = 1
                    elif param.component == secondary:
                        comp_int = 2
                    else:
                        comp_int = None

    #            if len(eb.filter(qualifier=quals[y], dataset=rvs[x])) == 1:
                    try:
                        pnew = _2to1par[param.qualifier]
                        if param.qualifier in ['ld_func', 'times', 'rvs', 'sigmas'] or param.component == '_default':

                            param = None

                    except:
                        logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')
                        param = None

                    if param != None:
                        val, ptype = par_value(param)
                        pname = ret_parname(param.qualifier, comp_int = comp_int, dtype='rv', dnum = i+1, ptype=ptype)
    # if is tries to append a value that already exists...stop that from happening
                        if pname[0] not in parnames:
                            parnames.extend(pname)
                            parvals.extend(val)
                            if ptype == 'array':
                                types.append(ptype)
                                types.append(ptype)
                            else:
                                types.append(ptype)

#spots

    parnames.append('phoebe_spots_units')
    parvals.append('"Degrees"')
    types.append('choice')
    for y in range(len(spots)):

        quals = eb.filter(feature=spots[y], context='feature')


        for param in quals.to_list():
            if param.component == primary:
                comp_int = 1
            elif param.component == secondary:
                comp_int = 2
            else:
                comp_int = None

            try:
                pnew = _2to1par[param.qualifier]

            except:
                param = None

            if param != None:

                val, ptype = par_value(param)

                pname = ret_parname(param.qualifier, comp_int=None, dtype='spots', dnum = y+1, ptype=ptype)

                parnames.extend(pname)
                parvals.extend(val)


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

    # comquals = eb.get_compute(kind='legacy', check_visible=False)-eb.get_compute(kind='legacy', component='_default')
    computeps = eb.get_compute(compute=compute, kind='legacy', check_visible=False)

    for param in computeps.to_list():
        if param.component == primary:
            comp_int = 1
        elif param.component == secondary:
            comp_int = 2
        else:
            comp_int = None

        if param.component == '_default':
            continue

#        if param.qualifier == 'heating':
#            if param.get_value() == False:
#               in1 =  parnames.index('phoebe_alb1.VAL')
#               in2 =  parnames.index('phoebe_alb2.VAL')
#               parvals[in1] = 0.0
#               parvals[in2] = 0.0
        #TODO add reflection switch
        if param.qualifier == 'refl_num':
            if param.get_value(**kwargs) == 0:
                in1 =  parnames.index('phoebe_alb1.VAL')
                in2 =  parnames.index('phoebe_alb2.VAL')
                parvals[in1] = 0.0
                parvals[in2] = 0.0
            elif  param.get_value(**kwargs) == 1:
                pname = 'phoebe_reffect_switch'
                val = '0'
                ptype='boolean'
                parnames.append(pname)
                parvals.append(val)
                types.append(ptype)

            else:
                pname = 'phoebe_reffect_switch'
                val = '1'
                ptype = 'boolean'
                parnames.append(pname)
                parvals.append(val)
                types.append(ptype)
        try:
            pnew = _2to1par[param.qualifier]
            if param.qualifier in ['ld_func'] or param.dataset:
                param = None
        except:

            logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')
            param = None

        if param != None:
            val, ptype = par_value(param, **kwargs)
            if param.qualifier == 'gridsize':
                pname = ret_parname(param.qualifier, comp_int = comp_int, dtype='grid', ptype=ptype)
            elif param.qualifier =='atm':
                atmval = {'extern_atmx':1, 'extern_planckint':0}
                pname = ret_parname(param.qualifier, comp_int = comp_int, ptype=ptype)

                val = str(atmval[val[0]])
            else:
                pname = ret_parname(param.qualifier, comp_int = comp_int, ptype=ptype)

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
        if param.component == primary:
            comp_int = 1
        elif param.component == secondary:
            comp_int = 2
        else:
            comp_int = None

        try:
            pnew = _2to1par[param.qualifier]
        except:
            logger.warning(str(param.qualifier)+' has no phoebe 1 corollary')
            param = None

        if param != None:

            val, ptype = par_value(param)
        #exceptions that must be caught like vgamma

            if param.qualifier == 'vgamma':
                # print val, type(val)
                val = [-1*float(val[0])]
#                pname = ret_parname(param.qualifier, comp_int = comp_int, ptype=ptype)

#            else:
#                pname = ret_parname(param.qualifier, comp_int = comp_int, ptype)
            pname = ret_parname(param.qualifier, comp_int = comp_int, ptype=ptype)
            if pname[0] not in parnames:

                parnames.extend(pname)
                parvals.extend(val)
                types.append(ptype)

# Phoebe1 has certain parameters that do not have phoebe 2 corollaries. If you did
#not load a phoebe1 compute parameter set these must be set to defaults
    kinds = []

    for x in eb.computes:
        kinds.append(eb[x].meta['kind'])
    if 'legacy' not in kinds:
        #phoebe 1 defaults
        namep = ['phoebe_reffect_reflections'] ; val = ['1'] ; ptype = 'int'
        parnames.extend(namep); parvals.extend(val) ; types.append(ptype)
        namep = ['phoebe_reffect_switch'] ; val = ['0'] ; ptype = 'boolean'
        parnames.extend(namep); parvals.extend(val) ; types.append(ptype)
        namep = ['phoebe_ie_switch'] ; val = ['0'] ; ptype = 'boolean'
        parnames.extend(namep); parvals.extend(val) ; types.append(ptype)

    #add default parameters that phoebe1
#        print(parnames)
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
    f.write('# Phoebe 1 file created from phoebe 2 bundle. Some functionality may be lost\n')
    # print "***", len(parnames), len(parvals)
    for x in range(len(parnames)):
#        if types[x] == 'float':
#            value = round(parvals[x],6)
#        elif types[x] == 'choice':
#            value = '"'+str(parvals[x])+'"'
#        else:
        # print parnames[x]
        # print parvals[x]
        value = parvals[x]
        # TODO: set precision on floats?
        f.write(str(parnames[x])+' = '+str(value)+'\n')

    f.close()
#    raise NotImplementedError

    return

def N_to_Ntriangles(N):
    """
    @N: WD style gridsize

    Converts WD style grid size @N to the number of triangles on the
    surface.

    Returns: number of triangles.
    """

    theta = np.array([np.pi/2*(k-0.5)/N for k in range(1, N+1)])
    phi = np.array([[np.pi*(l-0.5)/Mk for l in range(1, Mk+1)] for Mk in np.array(1 + 1.3*N*np.sin(theta), dtype=int)])
    Ntri = 2*np.array([len(p) for p in phi]).sum()

    return Ntri
